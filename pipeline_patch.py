"""
pipeline.py  —  Complete hybrid FFT-LogReg + CNN cascade pipeline
=================================================================
Stages (run in order):
  1. train_fft   — train FFT/PSD/HFE + LogReg, save fft_logreg.pkl
  2. train_cnn   — train CNN on .npy data, save TF SavedModel
                   (--cnn_arch simple | mil)
  3. baseline    — random 50/50 router benchmark
  4. cascade     — confidence-based cascade inference
  5. tune        — auto-tune thresholds on val, report test metrics
  6. heatmap     — MIL only: export patch-level fake-evidence heatmaps

CNN architectures
-----------------
  simple : 原 SimpleCNN, 整图 128×128 输入, 输出整图概率
  mil    : Attention-MIL CNN. 把 128×128 切成 16 个 32×32 patch (4×4 grid),
           每个 patch 过共享 CNN backbone 得 feature, 再用 attention 加权聚合:
              bag_logit = Σ α_i × patch_softmax_i
           推理时同时输出 attention 权重 α 和 patch 级 softmax,
           可生成"伪造证据热力图" heatmap = α × P(fake | patch).

Usage
-----
  # Full pipeline (all stages, MIL CNN, 默认导出热力图)
  python pipeline.py --fake /path/fake.npy --real /path/real.npy \
      --out ./runs/exp1 --cnn_arch mil

  # 用旧 SimpleCNN 跑 (无热力图导出)
  python pipeline.py --fake ... --real ... --out ./runs/exp1 \
      --cnn_arch simple --skip_heatmap

  # Skip stages already done
  python pipeline.py --fake ... --real ... --out ./runs/exp1 \
      --skip_train_fft --skip_train_cnn

  # Cascade only (both models already trained)
  python pipeline.py --fake ... --real ... --out ./runs/exp1 \
      --skip_train_fft --skip_train_cnn --skip_baseline

Label convention: 0 = fake (GAN-generated), 1 = real.
Config defaults match mid-scale split: 5999/1799/1799 per class.
CNN_BATCH=128 (适合 8GB 以下 GPU).
"""

import argparse
import os
import time
import warnings
import numpy as np
from scipy import sparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    HAS_PYFFTW = True
except ImportError:
    HAS_PYFFTW = False
    warnings.warn("pyfftw not installed — falling back to np.fft. "
                   "Install with: pip install pyfftw")

# ─────────────────────────────────────────────────────────────────────────────
# 0. Shared constants / helpers
# ─────────────────────────────────────────────────────────────────────────────

N_TRAIN = 5999
N_VAL   = 1799
N_TEST  = 1799
SEED    = 42
IMG_H   = 128
IMG_W   = 128

FFT_BINS      = 64
FFT_HFE_RATIO = 0.25
FFT_BATCH     = 128

CNN_EPOCHS      = 50
CNN_BATCH       = 128
CNN_LR          = 1e-3
CNN_EARLY_STOP  = 5        # patience

# ── Patch / MIL 配置 ─────────────────────────────────────────────────────────
PATCH_SIZE   = 32          # 32×32 patch
PATCH_STRIDE = 32          # 无重叠 → 4×4=16 patches per 128×128 image
GRID_H       = IMG_H // PATCH_STRIDE   # 4
GRID_W       = IMG_W // PATCH_STRIDE   # 4
N_PATCHES    = GRID_H * GRID_W         # 16
ATTN_HIDDEN  = 64          # attention 分支隐层维度


# ── Metrics ──────────────────────────────────────────────────────────────────

def roc_auc(y_true, y_score):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        pass
    yt = np.asarray(y_true, np.int32)
    ys = np.asarray(y_score, np.float64)
    order = np.argsort(ys); yt = yt[order]
    n_pos = yt.sum(); n_neg = len(yt) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.arange(1, len(yt) + 1)
    return float((ranks[yt == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def eval_metrics(y_true, y_score, thr=0.5):
    y_pred = (np.asarray(y_score) >= thr).astype(int)
    acc = float((y_pred == np.asarray(y_true)).mean())
    return acc, roc_auc(y_true, y_score)


def log(out_dir, text):
    print(text)
    path = os.path.join(out_dir, "results.txt")
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading & splitting
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split(fake_path, real_path,
                   n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, seed=SEED):
    """
    Returns (tr_imgs, y_tr), (va_imgs, y_va), (te_imgs, y_te)
    Labels: 0 = fake (GAN-generated), 1 = real
    Images: float32 (N, 128, 128), raw pixel values 0-255
    """
    fake = np.load(fake_path).astype(np.float32)
    real = np.load(real_path).astype(np.float32)

    def _split(arr):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(arr))
        a, b = n_train, n_train + n_val
        c = b + n_test
        assert len(arr) >= c, (
            f"Need {c} samples but only have {len(arr)}")
        return idx[:a], idx[a:b], idx[b:c]

    f_tr, f_va, f_te = _split(fake)
    r_tr, r_va, r_te = _split(real)

    def _cat(f_idx, r_idx):
        imgs = np.concatenate([fake[f_idx], real[r_idx]], 0)
        lbls = np.array([0] * len(f_idx) + [1] * len(r_idx), np.int32)
        return imgs, lbls

    return _cat(f_tr, r_tr), _cat(f_va, r_va), _cat(f_te, r_te)


# ─────────────────────────────────────────────────────────────────────────────
# 2. FFT / PSD / HFE feature extraction  (optimized)
# ─────────────────────────────────────────────────────────────────────────────
# 优化点:
#   a) 稀疏矩阵向量化 — for+bincount → CSR sparse matmul, ~5-10×
#   b) pyfftw 替换     — FFTW SIMD + 多线程 FFT, ~2-5×
#   c) ThreadPool 并行 — batch 间多线程, ~2-4× (FFT/sparse 均释放 GIL)
# ─────────────────────────────────────────────────────────────────────────────

def _build_radial_cache(h=128, w=128, bins=64):
    """
    构建 radial binning 的 CSR 稀疏矩阵 M (bins, H*W).
    M[bin, pixel] = 1 / count_of_bin, 使得:
      prof = (M @ psd_flat.T).T   →  直接得到 radial profile 均值
    一次稀疏矩阵乘法替代整个 Python for 循环.
    """
    cy, cx = h // 2, w // 2
    y, x   = np.indices((h, w))
    r      = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rb     = (r / (r.max() + 1e-8) * (bins - 1)).astype(np.int32)
    rb_flat = rb.ravel()

    n_pixels = h * w
    cnt = np.bincount(rb_flat, minlength=bins).astype(np.float64)
    cnt[cnt == 0] = 1.0

    # COO → CSR: 矩阵乘法最快的稀疏格式
    row = rb_flat                             # bin index per pixel
    col = np.arange(n_pixels, dtype=np.int32) # pixel index
    val = (1.0 / cnt[rb_flat]).astype(np.float32)

    M = sparse.csr_matrix(
        (val, (row, col)),
        shape=(bins, n_pixels),
        dtype=np.float32,
    )

    return {"M": M, "bins": bins, "rb_flat": rb_flat,
            "cnt": cnt.astype(np.float32)}


def _fft2_fast(imgs):
    """2D FFT + fftshift, 优先 pyfftw (SIMD+多线程), fallback numpy."""
    if HAS_PYFFTW:
        F = pyfftw.interfaces.numpy_fft.fft2(
            imgs, axes=(-2, -1),
            threads=-1,
            planner_effort='FFTW_MEASURE',
        )
    else:
        F = np.fft.fft2(imgs, axes=(-2, -1))
    return np.fft.fftshift(F, axes=(-2, -1))


def _fft_batch(imgs, cache, hfe_ratio=0.25):
    """
    向量化 FFT batch 处理:
      1) pyfftw FFT → PSD
      2) 稀疏矩阵 M @ psd.T → radial profile (无 for 循环)
      3) log1p + HFE + z-score 归一化
    输出与原版完全兼容: (B, bins+1) float32
    """
    bins = cache["bins"]
    M    = cache["M"]

    # 1) FFT → PSD
    F   = _fft2_fast(imgs)
    psd = (F.real ** 2 + F.imag ** 2).astype(np.float32)

    # 2) Radial binning — 一步稀疏矩阵乘法
    B = imgs.shape[0]
    psd_flat = psd.reshape(B, -1)           # (B, H*W)
    prof = (M @ psd_flat.T).T               # (B, bins)

    # 3) 后处理 (与原版一致)
    prof = np.log1p(prof)
    k    = int(bins * (1 - hfe_ratio))
    hfe  = prof[:, k:].sum(1) / (prof.sum(1) + 1e-8)
    mu   = prof.mean(1, keepdims=True)
    std  = prof.std(1,  keepdims=True) + 1e-8
    prof = (prof - mu) / std

    return np.concatenate([prof, hfe[:, None]], axis=1).astype(np.float32)


def extract_fft_features(images, bins=FFT_BINS, hfe_ratio=FFT_HFE_RATIO,
                          batch_size=512, n_workers=None):
    """
    优化版 FFT 特征提取 (接口与原版兼容).
    batch_size 默认提升到 512 (向量化后大 batch 更高效).
    n_workers: 并行线程数, None=auto, 1=串行.
    """
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 4)

    cache  = _build_radial_cache(h=images.shape[1], w=images.shape[2], bins=bins)
    n      = len(images)
    starts = list(range(0, n, batch_size))

    def _process(start):
        end = min(start + batch_size, n)
        return start, _fft_batch(images[start:end], cache, hfe_ratio)

    if n_workers <= 1 or len(starts) <= 1:
        chunks = [_process(s)[1] for s in starts]
    else:
        # FFT 和稀疏矩阵乘法均释放 GIL, 多线程有效
        chunks = [None] * len(starts)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_process, s): idx
                       for idx, s in enumerate(starts)}
            for fut in as_completed(futures):
                idx = futures[fut]
                _, result = fut.result()
                chunks[idx] = result

    return np.concatenate(chunks, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. FFT stage: train LogReg, save pkl
# ─────────────────────────────────────────────────────────────────────────────

def stage_train_fft(out_dir, tr_imgs, y_tr, va_imgs, y_va, te_imgs, y_te):
    import joblib
    from sklearn.linear_model import LogisticRegression

    print("\n" + "=" * 60)
    print("[Stage 1] Train FFT/PSD/HFE + LogReg")
    print("=" * 60)

    t0   = time.time()
    X_tr = extract_fft_features(tr_imgs)
    X_va = extract_fft_features(va_imgs)
    X_te = extract_fft_features(te_imgs)
    feat_time = time.time() - t0
    total_n   = len(tr_imgs) + len(va_imgs) + len(te_imgs)
    print(f"  Feature extraction: {feat_time:.1f}s  ({total_n/feat_time:.0f} img/s)")

    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(X_tr, y_tr)

    va_score = clf.predict_proba(X_va)[:, 1]
    te_score = clf.predict_proba(X_te)[:, 1]
    va_acc, va_auc = eval_metrics(y_va, va_score)
    te_acc, te_auc = eval_metrics(y_te, te_score)

    model_path = os.path.join(out_dir, "fft_logreg.pkl")
    feat_path  = os.path.join(out_dir, "fft_features.npz")
    joblib.dump(clf, model_path)
    np.savez(feat_path,
             X_tr=X_tr, y_tr=y_tr,
             X_va=X_va, y_va=y_va,
             X_te=X_te, y_te=y_te)

    print(f"  FFT VAL  acc={va_acc:.4f}  auc={va_auc:.4f}")
    print(f"  FFT TEST acc={te_acc:.4f}  auc={te_auc:.4f}")
    print(f"  Saved model → {model_path}")

    # ── Sanity check 3: FFT score 分布 ────────────────────────────────────
    print("\n[Sanity Check 3] FFT score 分布 (测试集)")
    print(f"  score range : {te_score.min():.4f} ~ {te_score.max():.4f}")
    print(f"  score mean  : {te_score.mean():.4f}")
    print(f"  score std   : {te_score.std():.4f}  (接近0说明全堆一端，任务过于简单)")
    # 按类别分别看
    fake_scores = te_score[y_te == 0]
    real_scores = te_score[y_te == 1]
    print(f"  fake(0) score mean: {fake_scores.mean():.4f}  std: {fake_scores.std():.4f}")
    print(f"  real(1) score mean: {real_scores.mean():.4f}  std: {real_scores.std():.4f}")
    sep = real_scores.mean() - fake_scores.mean()
    print(f"  类间均值差 (越大越容易分): {sep:.4f}")
    if abs(sep) > 0.8:
        print("  ℹ️  INFO: 两类 FFT 特征差异极大，"
              "AUC=1.0 可能是任务本身过于简单（域级别差异），不一定是泄露")

    return clf, {"X_tr": X_tr, "X_va": X_va, "X_te": X_te,
                 "va_score": va_score, "te_score": te_score,
                 "va_acc": va_acc, "va_auc": va_auc,
                 "te_acc": te_acc, "te_auc": te_auc}


# ─────────────────────────────────────────────────────────────────────────────
# 4. CNN stage: build SimpleCNN, train on .npy, save SavedModel
# ─────────────────────────────────────────────────────────────────────────────

def _build_simple_cnn(input_shape=(128, 128, 1), n_classes=2):
    """
    Lightweight CNN matching the 'cnn' branch of classifier.py.
    Conv→Pool×3 + Dense×2.  Uses mixed_float16 if called inside
    a mixed-precision context.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inp = layers.Input(shape=input_shape)
    x   = layers.Conv2D(32,  3, activation="relu", padding="same")(inp)
    x   = layers.MaxPool2D()(x)
    x   = layers.Conv2D(64,  3, activation="relu", padding="same")(x)
    x   = layers.MaxPool2D()(x)
    x   = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(256, activation="relu")(x)
    x   = layers.Dropout(0.5)(x)
    # output cast to float32 to be safe with mixed precision
    out = layers.Dense(n_classes, activation="softmax", dtype="float32")(x)
    return models.Model(inp, out)


# ─────────────────────────────────────────────────────────────────────────────
# 4b. Patch 切割工具 + MIL CNN
# ─────────────────────────────────────────────────────────────────────────────

def image_to_patches_np(imgs, patch=PATCH_SIZE, stride=PATCH_STRIDE):
    """
    (N, H, W) float32 → (N, n_patches, patch, patch) float32, 无重叠切片.
    用 stride_tricks 做 view, 不复制内存.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    N, H, W = imgs.shape
    assert (H - patch) % stride == 0 and (W - patch) % stride == 0, \
        f"H/W 必须能被 stride 整除: H={H} W={W} patch={patch} stride={stride}"

    # sliding_window_view: (N, H-patch+1, W-patch+1, patch, patch)
    win = sliding_window_view(imgs, (patch, patch), axis=(1, 2))
    # 按 stride 取样: (N, gh, gw, patch, patch)
    win = win[:, ::stride, ::stride, :, :]
    gh, gw = win.shape[1], win.shape[2]
    # 拍平 grid 维: (N, gh*gw, patch, patch)
    return win.reshape(N, gh * gw, patch, patch)


# ─────────────────────────────────────────────────────────────────────────────
# 4c. MIL 专用自定义 Layer (替代 Lambda, 避免 SavedModel 序列化失败)
# ─────────────────────────────────────────────────────────────────────────────
# Lambda 层会把 tf 模块 closure 进函数体, 导致 model.save 时 deepcopy 配置
# 失败 ("cannot pickle 'module' object"). 用 Layer 子类则只存基础 dtype/name
# 等可序列化参数, 完全规避此问题.
#
# 这些类在模块级"懒构造": 第一次访问 _get_mil_layers() 时再 import tf 并
# 定义类, 这样 import pipeline 不会强行依赖 tf.
# ─────────────────────────────────────────────────────────────────────────────

_MIL_LAYER_CLASSES = None

def _get_mil_layers():
    """
    返回 (MergeBagIntoBatch, SplitBagFromBatch,
          SoftmaxOverPatches, SumOverPatches) 4 个 Layer 子类.
    """
    global _MIL_LAYER_CLASSES
    if _MIL_LAYER_CLASSES is not None:
        return _MIL_LAYER_CLASSES

    import tensorflow as tf
    from tensorflow.keras import layers as _layers

    class MergeBagIntoBatch(_layers.Layer):
        """(B, n_patches, P, P, C) → (B*n_patches, P, P, C)"""
        def call(self, x):
            shape = tf.shape(x)
            return tf.reshape(
                x, [shape[0] * shape[1], shape[2], shape[3], shape[4]])

        def compute_output_shape(self, input_shape):
            B, N, P1, P2, C = input_shape
            B_new = None if (B is None or N is None) else B * N
            return (B_new, P1, P2, C)

    class SplitBagFromBatch(_layers.Layer):
        """(B*n_patches, feat_dim) → (B, n_patches, feat_dim)"""
        def __init__(self, n_patches, feat_dim, **kw):
            super().__init__(**kw)
            self.n_patches = int(n_patches)
            self.feat_dim  = int(feat_dim)

        def call(self, x):
            return tf.reshape(x, [-1, self.n_patches, self.feat_dim])

        def compute_output_shape(self, input_shape):
            return (None, self.n_patches, self.feat_dim)

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"n_patches": self.n_patches,
                        "feat_dim":  self.feat_dim})
            return cfg

    class SoftmaxOverPatches(_layers.Layer):
        """softmax over axis=1 (the patch dimension)."""
        def call(self, x):
            return tf.nn.softmax(x, axis=1)

        def compute_output_shape(self, input_shape):
            return input_shape

    class SumOverPatches(_layers.Layer):
        """reduce_sum over axis=1, removes the patch dimension."""
        def call(self, x):
            return tf.reduce_sum(x, axis=1)

        def compute_output_shape(self, input_shape):
            # input: (B, n_patches, feat) → output: (B, feat)
            return input_shape[:1] + input_shape[2:]

    _MIL_LAYER_CLASSES = (MergeBagIntoBatch, SplitBagFromBatch,
                          SoftmaxOverPatches, SumOverPatches)
    return _MIL_LAYER_CLASSES


def _build_mil_cnn(patch_shape=(PATCH_SIZE, PATCH_SIZE, 1),
                   n_patches=N_PATCHES, n_classes=2,
                   attn_hidden=ATTN_HIDDEN):
    """
    Attention-MIL CNN:
      Input: (B, n_patches, patch, patch, 1)
        ↓ reshape → (B*n_patches, patch, patch, 1)
        ↓ Conv backbone → (B*n_patches, feat_dim)
        ↓ reshape → (B, n_patches, feat_dim)
        ↓ ┬─ attention head : (B, n_patches, 1)  softmax over patches
          └─ patch logit head: (B, n_patches, n_classes)  softmax
        ↓ bag_logit = Σ α_i × patch_logit_i
      Output: (B, n_classes)  整图概率

    模型有两个额外可访问的输出 (用于推理时取热力图):
      - attention weights α: shape (B, n_patches)
      - patch logits         : shape (B, n_patches, n_classes)
    通过 model.get_layer("attn_softmax") / "patch_softmax" 拿到.

    实现注意:
      所有 reshape / softmax / reduce_sum 都用自定义 Layer 子类, 而不是
      Lambda. 这是因为 model.save(save_format='tf') 会对每个 layer config
      做 deepcopy, 而 Lambda 把 tf 模块 closure 进函数体导致
      'cannot pickle module object' 错误.
    """
    from tensorflow.keras import layers, models
    (MergeBagIntoBatch, SplitBagFromBatch,
     SoftmaxOverPatches, SumOverPatches) = _get_mil_layers()

    bag_input = layers.Input(shape=(n_patches,) + patch_shape,
                             name="bag_input")

    # → (B*n_patches, P, P, C)
    x = MergeBagIntoBatch(name="merge_batch_patch")(bag_input)

    # ── Patch backbone (32×32 输入, 两次 pool 降到 8×8) ──
    x = layers.Conv2D(32,  3, activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)                         # 32 → 16
    x = layers.Conv2D(64,  3, activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)                         # 16 → 8
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)            # → (B*n_patches, 128)
    feat_dim = 128

    # 恢复 bag 维度: (B, n_patches, feat_dim)
    feat = SplitBagFromBatch(n_patches=n_patches, feat_dim=feat_dim,
                             name="restore_bag")(x)

    # ── Attention 分支 ──
    a = layers.Dense(attn_hidden, activation="tanh",
                     name="attn_hidden")(feat)            # (B, n_patches, hid)
    a = layers.Dense(1, name="attn_score")(a)             # (B, n_patches, 1)
    # softmax over patch dimension (axis=1)
    a = SoftmaxOverPatches(name="attn_softmax")(a)        # (B, n_patches, 1)

    # ── Patch logit 分支 ──
    p_logits = layers.Dropout(0.3)(feat)
    p_logits = layers.Dense(n_classes, activation="softmax",
                            dtype="float32",
                            name="patch_softmax")(p_logits)  # (B, n_patches, C)

    # ── Bag logit = Σ α_i × patch_logit_i ──
    weighted = layers.Multiply(name="weighted_patch")([a, p_logits])  # broadcasting
    bag_out  = SumOverPatches(name="bag_sum",
                              dtype="float32")(weighted)  # (B, n_classes)

    return models.Model(bag_input, bag_out, name="MIL_CNN")


def _make_tf_dataset_mil(images, labels, batch_size, shuffle=False):
    """
    images: (N, H, W) float32 0-255
    → (N, n_patches, P, P, 1) float32 0-1 tf.data pipeline
    """
    import tensorflow as tf
    patches = image_to_patches_np(images)               # (N, n_patches, P, P)
    patches = (patches[..., np.newaxis] / 255.0).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((patches, labels))
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images))
    return ds.batch(batch_size, drop_remainder=False).prefetch(
        tf.data.experimental.AUTOTUNE)


def _make_tf_dataset(images, labels, batch_size, shuffle=False):
    """
    images: (N, H, W) float32 0-255
    → normalised (N, H, W, 1) float32 0-1 tf.data pipeline
    """
    import tensorflow as tf
    imgs = (images[:, :, :, np.newaxis] / 255.0).astype(np.float32)
    ds   = tf.data.Dataset.from_tensor_slices((imgs, labels))
    ds   = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images))
    return ds.batch(batch_size, drop_remainder=False).prefetch(
        tf.data.experimental.AUTOTUNE)


def stage_train_cnn(out_dir, tr_imgs, y_tr, va_imgs, y_va, te_imgs, y_te,
                    arch="mil"):
    """
    arch: 'simple' = 原 SimpleCNN (整图输入)
          'mil'    = Attention-MIL CNN (patch bag 输入)
    """
    import tensorflow as tf
    from tensorflow.keras import mixed_precision

    print("\n" + "=" * 60)
    print(f"[Stage 2] Train CNN  (arch={arch})")
    print("=" * 60)

    mixed_precision.set_global_policy("mixed_float16")
    tf.config.optimizer.set_jit(True)

    if arch == "simple":
        train_ds = _make_tf_dataset(tr_imgs, y_tr, CNN_BATCH, shuffle=True)
        val_ds   = _make_tf_dataset(va_imgs, y_va, CNN_BATCH)
        test_ds  = _make_tf_dataset(te_imgs, y_te, CNN_BATCH)
    elif arch == "mil":
        train_ds = _make_tf_dataset_mil(tr_imgs, y_tr, CNN_BATCH, shuffle=True)
        val_ds   = _make_tf_dataset_mil(va_imgs, y_va, CNN_BATCH)
        test_ds  = _make_tf_dataset_mil(te_imgs, y_te, CNN_BATCH)
    else:
        raise ValueError(f"unknown arch={arch}")

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if arch == "simple":
            model = _build_simple_cnn(input_shape=(IMG_H, IMG_W, 1), n_classes=2)
        else:
            model = _build_mil_cnn(
                patch_shape=(PATCH_SIZE, PATCH_SIZE, 1),
                n_patches=N_PATCHES, n_classes=2,
                attn_hidden=ATTN_HIDDEN,
            )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(CNN_LR),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=["acc"],
            jit_compile=True,
        )

    model_dir = os.path.join(out_dir, "cnn_model")
    os.makedirs(model_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=CNN_EARLY_STOP,
            restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(out_dir, "cnn_logs"), update_freq=50),
    ]

    model.summary()
    model.fit(train_ds, epochs=CNN_EPOCHS,
              validation_data=val_ds, callbacks=callbacks)

    # ── Evaluate ──
    def _predict(ds, images):
        probs = []
        for batch_x, _ in ds:
            out = model(batch_x, training=False).numpy()
            probs.append(out[:, 1])
        return np.concatenate(probs)[:len(images)]

    va_score = _predict(val_ds,  va_imgs)
    te_score = _predict(test_ds, te_imgs)
    va_acc, va_auc = eval_metrics(y_va, va_score)
    te_acc, te_auc = eval_metrics(y_te, te_score)

    # ── 模型保存: 多重 fallback ─────────────────────────────────────────────
    # Keras 2 在某些 mixed-precision + 自定义 Layer 组合下, model.save 会
    # 在 deepcopy layer config 时撞到 "cannot pickle 'module' object".
    # 策略: SavedModel → .h5 → weights-only. 任何一种成功即可.
    import shutil
    saved_to = None
    save_errors = []
    for attempt in ("savedmodel", "h5", "weights"):
        try:
            if attempt == "savedmodel":
                model.save(model_dir, save_format="tf")
                saved_to = model_dir
            elif attempt == "h5":
                saved_to = model_dir + ".h5"
                model.save(saved_to, save_format="h5")
            else:  # weights only
                w_dir = model_dir + "_weights"
                os.makedirs(w_dir, exist_ok=True)
                w_path = os.path.join(w_dir, "model_weights")
                model.save_weights(w_path)
                # 同时存一个 arch 标记, 加载时按 arch 重建模型再 load_weights
                with open(os.path.join(w_dir, "arch.txt"), "w") as f:
                    f.write(arch + "\n")
                saved_to = w_dir
            break
        except Exception as e:
            save_errors.append(
                f"  [{attempt}] failed: {type(e).__name__}: "
                f"{str(e)[:200]}")
            # 清理失败 attempt 留下的脏文件
            if attempt == "savedmodel" and os.path.isdir(model_dir):
                shutil.rmtree(model_dir, ignore_errors=True)
            elif attempt == "h5" and os.path.exists(model_dir + ".h5"):
                os.remove(model_dir + ".h5")
            saved_to = None

    if saved_to is None:
        raise RuntimeError(
            "Failed to save model with all strategies:\n" +
            "\n".join(save_errors))

    print(f"  CNN VAL  acc={va_acc:.4f}  auc={va_auc:.4f}")
    print(f"  CNN TEST acc={te_acc:.4f}  auc={te_auc:.4f}")
    if save_errors:
        print(f"  Save fallback chain (tried {len(save_errors)+1} strategies):")
        for line in save_errors:
            print(line)
    print(f"  Saved model → {saved_to}")

    return model, {"va_score": va_score, "te_score": te_score,
                   "va_acc": va_acc, "va_auc": va_auc,
                   "te_acc": te_acc, "te_auc": te_auc}


# ─────────────────────────────────────────────────────────────────────────────
# 5. CNN predict helper (used in cascade & tune)
# ─────────────────────────────────────────────────────────────────────────────

def cnn_predict_proba(model, images, batch_size=CNN_BATCH):
    """
    images: (N, H, W) float32 0-255 → (N,) prob class-1
    自动识别模型输入是整图 (SimpleCNN) 还是 patch bag (MIL_CNN).
    """
    import tensorflow as tf

    # 通过 model.input_shape 判断架构: 4D = SimpleCNN, 5D = MIL
    in_shape = model.input_shape
    is_mil = (len(in_shape) == 5)

    if is_mil:
        patches = image_to_patches_np(images)               # (N, n_patches, P, P)
        x = (patches[..., np.newaxis] / 255.0).astype(np.float32)
    else:
        x = (images[:, :, :, np.newaxis] / 255.0).astype(np.float32)

    ds  = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
    out = []
    for batch in ds:
        out.append(model(batch, training=False).numpy()[:, 1])
    return np.concatenate(out)[:len(images)]


# ─────────────────────────────────────────────────────────────────────────────
# 5b. MIL 热力图推理: 同时返回 score 和 4×4 patch-level heatmap
# ─────────────────────────────────────────────────────────────────────────────

def cnn_predict_with_heatmap(model, images, batch_size=CNN_BATCH):
    """
    仅适用于 MIL_CNN.
    Returns:
      scores  : (N,)              整图 fake 概率 (= class-1 概率)
      heatmaps: (N, GRID_H, GRID_W)
                heatmap[i,j] = attention[i,j] × patch_fake_prob[i,j]
                "伪造证据图": attention 高且倾向 fake 的 patch 值越高
    """
    import tensorflow as tf
    from tensorflow.keras import models as kmodels

    in_shape = model.input_shape
    if len(in_shape) != 5:
        raise ValueError("cnn_predict_with_heatmap requires MIL_CNN "
                         f"(5D input), got input_shape={in_shape}")

    # 拼一个 multi-output 模型: [bag_score, attention, patch_softmax]
    attn_layer  = model.get_layer("attn_softmax")
    patch_layer = model.get_layer("patch_softmax")
    multi = kmodels.Model(
        inputs=model.input,
        outputs=[model.output, attn_layer.output, patch_layer.output],
    )

    patches = image_to_patches_np(images)               # (N, n_patches, P, P)
    x = (patches[..., np.newaxis] / 255.0).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)

    all_scores, all_heat = [], []
    for batch in ds:
        bag_out, attn, patch_p = multi(batch, training=False)
        bag_out  = bag_out.numpy()                      # (b, 2)
        attn     = attn.numpy().squeeze(-1)             # (b, n_patches)
        patch_p  = patch_p.numpy()[..., 1]              # (b, n_patches)  fake prob
        heat     = (attn * patch_p).reshape(
            -1, GRID_H, GRID_W).astype(np.float32)      # (b, gh, gw)
        all_scores.append(bag_out[:, 1])
        all_heat.append(heat)

    scores   = np.concatenate(all_scores)[:len(images)]
    heatmaps = np.concatenate(all_heat,  0)[:len(images)]
    return scores, heatmaps


# ─────────────────────────────────────────────────────────────────────────────
# 5c. 热力图导出: 选 top-k fake 预测, overlay 到原图保存 png
# ─────────────────────────────────────────────────────────────────────────────

def save_heatmaps(images, heatmaps, scores, labels,
                  out_dir, top_k=20, prefix="heatmap",
                  pick="most_fake"):
    """
    images   : (N, H, W) float32 0-255
    heatmaps : (N, gh, gw) float32, 任意范围 (会被逐图归一化到 [0,1])
    scores   : (N,) 整图 P(real) (= class-1 概率, 因为 label 1=real)
    labels   : (N,) 0=fake / 1=real ground truth
    pick     : 'most_fake'   = 模型最确信是 fake 的样本 (score 最低)
               'most_real'   = 模型最确信是 real 的样本 (score 最高)
               'most_uncertain' = score 最接近 0.5 的样本
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    n = len(images)
    k = min(top_k, n)

    # label 约定: 0=fake (GAN-generated), 1=real, score = P(class=1) = P(real)
    if pick == "most_fake":
        order = np.argsort(scores)[:k]
    elif pick == "most_real":
        order = np.argsort(-scores)[:k]
    elif pick == "most_uncertain":
        order = np.argsort(np.abs(scores - 0.5))[:k]
    else:
        raise ValueError(f"unknown pick={pick}")

    H, W = images.shape[1], images.shape[2]
    for rank, idx in enumerate(order):
        img    = images[idx]
        heat   = heatmaps[idx]
        score  = float(scores[idx])
        label  = int(labels[idx])
        label_str = "fake" if label == 0 else "real"
        fake_prob = 1.0 - score                         # P(fake)

        heat_t = _bilinear_upsample(heat, H, W)

        # 逐图 min-max 归一化, 突出图内对比
        if heat_t.max() > heat_t.min():
            heat_n = (heat_t - heat_t.min()) / (heat_t.max() - heat_t.min())
        else:
            heat_n = np.zeros_like(heat_t)

        # overlay: 灰度底图 + jet 热力 (alpha=0.45)
        fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.imshow(heat_n, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax.set_title(
            f"#{rank:02d}  GT={label_str}  P(fake)={fake_prob:.3f}",
            fontsize=9,
        )
        ax.axis("off")
        fname = (f"{prefix}_{rank:02d}_idx{idx:05d}_"
                 f"gt{label_str}_pfake{fake_prob:.3f}.png")
        plt.savefig(os.path.join(out_dir, fname),
                    bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    print(f"  Saved {k} heatmaps ({pick}) → {out_dir}")


def _bilinear_upsample(arr2d, H, W):
    """简单的双线性上采样: float → uint8 → PIL bilinear → float, 保留原值域."""
    a = arr2d.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros((H, W), dtype=np.float32) + lo
    a01 = (a - lo) / (hi - lo)
    u8  = (a01 * 255).astype(np.uint8)
    big = Image.fromarray(u8).resize((W, H), Image.BILINEAR)
    return np.asarray(big, dtype=np.float32) / 255.0 * (hi - lo) + lo


# ─────────────────────────────────────────────────────────────────────────────
# 6. Baseline: random 50/50 router
# ─────────────────────────────────────────────────────────────────────────────

def stage_baseline(out_dir, fft_model, cnn_model,
                   fft_feats, te_imgs, y_te):
    print("\n" + "=" * 60)
    print("[Stage 3] Random 50/50 router — baseline")
    print("=" * 60)

    X_te = fft_feats["X_te"]
    n    = len(y_te)
    rng  = np.random.RandomState(SEED)
    to_fft = rng.rand(n) < 0.5           # True → FFT branch
    to_cnn = ~to_fft

    # ── FFT branch ──
    t0 = time.time()
    fft_scores_sub = fft_model.predict_proba(X_te[to_fft])[:, 1]
    t_fft = time.time() - t0

    # ── CNN branch ──
    t0 = time.time()
    cnn_scores_sub = cnn_predict_proba(cnn_model, te_imgs[to_cnn])
    t_cnn = time.time() - t0

    combined = np.empty(n)
    combined[to_fft] = fft_scores_sub
    combined[to_cnn] = cnn_scores_sub

    acc, auc = eval_metrics(y_te, combined)
    n_fft = int(to_fft.sum());  n_cnn = int(to_cnn.sum())

    txt = (
        f"\n  Routing: FFT={n_fft} ({100*n_fft/n:.1f}%)  "
        f"CNN={n_cnn} ({100*n_cnn/n:.1f}%)\n"
        f"  FFT branch latency : {t_fft*1000:.1f} ms "
        f"({n_fft/(t_fft+1e-9):.0f} img/s)\n"
        f"  CNN branch latency : {t_cnn*1000:.1f} ms "
        f"({n_cnn/(t_cnn+1e-9):.0f} img/s)\n"
        f"  Random-router TEST acc={acc:.4f}  auc={auc:.4f}"
    )
    log(out_dir, txt)
    return acc, auc


# ─────────────────────────────────────────────────────────────────────────────
# 7. Cascade predict core
# ─────────────────────────────────────────────────────────────────────────────

def cascade_predict(fft_model, cnn_model,
                    X_fft, images,
                    low_thr=0.2, high_thr=0.8,
                    verbose=True):
    """
    FFT scores outside [low_thr, high_thr] → decided immediately.
    Ambiguous middle band → CNN precision path.

    Returns scores (N,), ambiguous_mask, timing_dict
    """
    n = len(images)

    t0 = time.time()
    fft_scores = fft_model.predict_proba(X_fft)[:, 1]
    t_fft = time.time() - t0

    ambiguous = (fft_scores >= low_thr) & (fft_scores <= high_thr)
    n_cnn = int(ambiguous.sum())
    n_fft = n - n_cnn

    scores = fft_scores.copy()
    t_cnn  = 0.0

    if n_cnn > 0:
        t0 = time.time()
        cnn_scores       = cnn_predict_proba(cnn_model, images[ambiguous])
        t_cnn            = time.time() - t0
        scores[ambiguous] = cnn_scores

    if verbose:
        total = t_fft + t_cnn
        print(f"    FFT-only : {n_fft:>6} ({100*n_fft/n:.1f}%)  "
              f"{t_fft*1000:.1f} ms")
        print(f"    CNN      : {n_cnn:>6} ({100*n_cnn/n:.1f}%)  "
              f"{t_cnn*1000:.1f} ms")
        print(f"    Total    : {total*1000:.1f} ms  "
              f"eff. throughput {n/(total+1e-9):.0f} img/s")

    return scores, ambiguous, {
        "t_fft": t_fft, "t_cnn": t_cnn,
        "n_fft": n_fft, "n_cnn": n_cnn,
    }

#新计时函数
def real_e2e_cascade_predict(fft_model, cnn_model, images,
                             low_thr=0.2, high_thr=0.8,
                             fft_batch=FFT_BATCH,
                             verbose=True):
    """
    真正端到端吞吐量：
    从原始 images 开始
      -> 现场提 FFT 特征
      -> FFT 打分
      -> 不确定样本送 CNN
    """
    n = len(images)

    # 1) FFT 特征提取（这一步以前没算进 cascade throughput）
    t0 = time.time()
    X_fft = extract_fft_features(images, batch_size=fft_batch)
    t_feat = time.time() - t0

    # 2) FFT 分类
    t0 = time.time()
    fft_scores = fft_model.predict_proba(X_fft)[:, 1]
    t_fft_cls = time.time() - t0

    # 3) 找出需要送 CNN 的不确定样本
    ambiguous = (fft_scores >= low_thr) & (fft_scores <= high_thr)
    n_cnn = int(ambiguous.sum())
    n_fft = n - n_cnn

    scores = fft_scores.copy()

    # 4) CNN 精判
    t_cnn = 0.0
    if n_cnn > 0:
        t0 = time.time()
        scores[ambiguous] = cnn_predict_proba(cnn_model, images[ambiguous])
        t_cnn = time.time() - t0

    total = t_feat + t_fft_cls + t_cnn

    if verbose:
        print("\n[Real E2E Cascade]")
        print(f"  FFT feature extraction : {t_feat*1000:.1f} ms")
        print(f"  FFT classification     : {t_fft_cls*1000:.1f} ms")
        print(f"  CNN inference          : {t_cnn*1000:.1f} ms")
        print(f"  Total                  : {total*1000:.1f} ms")
        print(f"  Throughput             : {n/(total+1e-9):.2f} img/s")
        print(f"  FFT={n_fft} ({100*n_fft/n:.1f}%)  CNN={n_cnn} ({100*n_cnn/n:.1f}%)")

    return scores, ambiguous, {
        "t_feat": t_feat,
        "t_fft_cls": t_fft_cls,
        "t_cnn": t_cnn,
        "total": total,
        "n_fft": n_fft,
        "n_cnn": n_cnn,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 8. Stage: cascade with given thresholds
# ─────────────────────────────────────────────────────────────────────────────

def stage_cascade(out_dir, fft_model, cnn_model,
                  fft_feats, va_imgs, y_va, te_imgs, y_te,
                  low_thr=0.2, high_thr=0.8):
    print("\n" + "=" * 60)
    print(f"[Stage 4] Cascade  low={low_thr:.2f}  high={high_thr:.2f}")
    print("=" * 60)

    print("  Val set:")
    va_scores, _, _ = cascade_predict(
        fft_model, cnn_model, fft_feats["X_va"], va_imgs,
        low_thr, high_thr)
    va_acc, va_auc = eval_metrics(y_va, va_scores)
    print(f"    acc={va_acc:.4f}  auc={va_auc:.4f}")

    print("  Test set:")
    te_scores, _, te_t = cascade_predict(
        fft_model, cnn_model, fft_feats["X_te"], te_imgs,
        low_thr, high_thr)
    te_acc, te_auc = eval_metrics(y_te, te_scores)
    print(f"    acc={te_acc:.4f}  auc={te_auc:.4f}")

    log(out_dir,
        f"\n[Cascade thr=({low_thr:.2f},{high_thr:.2f})] "
        f"FFT={te_t['n_fft']} CNN={te_t['n_cnn']}  "
        f"TEST acc={te_acc:.4f} auc={te_auc:.4f}")

    return te_acc, te_auc


# ─────────────────────────────────────────────────────────────────────────────
# 9. Stage: auto-tune thresholds
# ─────────────────────────────────────────────────────────────────────────────

def stage_tune(out_dir, fft_model, cnn_model,
               fft_feats, va_imgs, y_va, te_imgs, y_te,
               alpha=0.05):
    """
    Grid-search (low, high) on val set.
    Objective: AUC − α × cnn_fraction
    α = 0.05 means each 1% more CNN usage costs 0.0005 AUC.
    """
    print("\n" + "=" * 60)
    print(f"[Stage 5] Auto-tune thresholds  (speed_penalty α={alpha})")
    print("=" * 60)

    candidates = np.round(np.arange(0.05, 0.50, 0.05), 2)
    results = []

    for low in candidates:
        for high in [round(1.0 - l, 2) for l in candidates]:
            if high <= low:
                continue
            va_s, va_mask, _ = cascade_predict(
                fft_model, cnn_model, fft_feats["X_va"], va_imgs,
                float(low), float(high), verbose=False)
            auc       = roc_auc(y_va, va_s)
            cnn_frac  = float(va_mask.mean())
            composite = auc - alpha * cnn_frac
            results.append((float(low), float(high), auc, cnn_frac, composite))

    results.sort(key=lambda r: r[4], reverse=True)
    print(f"\n  Top-5 configs (val):")
    print(f"  {'low':>5} {'high':>5} {'auc':>7} {'cnn%':>7} {'score':>8}")
    for r in results[:5]:
        print(f"  {r[0]:>5.2f} {r[1]:>5.2f} "
              f"{r[2]:>7.4f} {100*r[3]:>6.1f}%  {r[4]:>8.4f}")

    best_low, best_high = results[0][0], results[0][1]
    print(f"\n  Best thresholds: low={best_low}  high={best_high}")

    print("  Test set (best thresholds):")
    te_s, te_mask, te_t = cascade_predict(
        fft_model, cnn_model, fft_feats["X_te"], te_imgs,
        best_low, best_high)
    te_acc, te_auc = eval_metrics(y_te, te_s)
    n = len(y_te)

    summary = (
        f"\n[Tuned Cascade best=({best_low:.2f},{best_high:.2f})] "
        f"FFT={te_t['n_fft']} ({100*te_t['n_fft']/n:.1f}%)  "
        f"CNN={te_t['n_cnn']} ({100*te_t['n_cnn']/n:.1f}%)  "
        f"TEST acc={te_acc:.4f} auc={te_auc:.4f}"
    )
    log(out_dir, summary)

    return best_low, best_high, te_acc, te_auc


# ─────────────────────────────────────────────────────────────────────────────
# 9b. Stage: heatmap export  (MIL only)
# ─────────────────────────────────────────────────────────────────────────────

def stage_heatmap(out_dir, cnn_model, te_imgs, y_te,
                  n_heatmaps=20):
    """
    用 MIL_CNN 导出 patch-level 伪造证据热力图.
    导出三类:
      - most_fake/      : 模型最确信是 fake 的 top-K
      - most_real/      : 模型最确信是 real 的 top-K
      - most_uncertain/ : 模型最不确定的 top-K
    """
    print("\n" + "=" * 60)
    print(f"[Stage 6] Export patch-level heatmaps  (n={n_heatmaps})")
    print("=" * 60)

    # 防御: 仅 MIL_CNN 支持
    if len(cnn_model.input_shape) != 5:
        msg = "  Skipped — current CNN is SimpleCNN (no patch heads)."
        print(msg)
        log(out_dir, msg)
        return

    scores, heatmaps = cnn_predict_with_heatmap(cnn_model, te_imgs)
    print(f"  scores  : range=[{scores.min():.3f}, {scores.max():.3f}]  "
          f"mean={scores.mean():.3f}")
    print(f"  heatmap : shape={heatmaps.shape}  "
          f"per-image mean={heatmaps.mean():.4f}")

    base = os.path.join(out_dir, "heatmaps")
    save_heatmaps(te_imgs, heatmaps, scores, y_te,
                  out_dir=os.path.join(base, "most_fake"),
                  top_k=n_heatmaps, prefix="fake",
                  pick="most_fake")
    save_heatmaps(te_imgs, heatmaps, scores, y_te,
                  out_dir=os.path.join(base, "most_real"),
                  top_k=n_heatmaps, prefix="real",
                  pick="most_real")
    save_heatmaps(te_imgs, heatmaps, scores, y_te,
                  out_dir=os.path.join(base, "most_uncertain"),
                  top_k=n_heatmaps, prefix="unc",
                  pick="most_uncertain")

    # 也存一份原始 numpy, 方便后续自定义可视化
    np.savez(os.path.join(base, "raw.npz"),
             scores=scores, heatmaps=heatmaps, y_te=y_te)
    log(out_dir,
        f"[Heatmap] exported {3*n_heatmaps} png + raw.npz → {base}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Hybrid FFT+CNN pipeline")
    p.add_argument("--fake", required=True,
                   help="Fake (GAN-generated) images .npy path → label 0")
    p.add_argument("--real", required=True,
                   help="Real images .npy path → label 1")
    p.add_argument("--out",   required=True, help="Output directory")

    p.add_argument("--skip_train_fft", action="store_true")
    p.add_argument("--skip_train_cnn", action="store_true")
    p.add_argument("--skip_baseline",  action="store_true")
    p.add_argument("--skip_heatmap",   action="store_true")

    # Cascade thresholds (used if not running tune first)
    p.add_argument("--low_thr",  type=float, default=0.2)
    p.add_argument("--high_thr", type=float, default=0.8)
    p.add_argument("--alpha",    type=float, default=0.05,
                   help="Speed-penalty weight for threshold tuning")

    p.add_argument("--seed",       type=int, default=SEED)
    p.add_argument("--n_train",    type=int, default=N_TRAIN)
    p.add_argument("--n_val",      type=int, default=N_VAL)
    p.add_argument("--n_test",     type=int, default=N_TEST)
    p.add_argument("--cnn_epochs", type=int, default=CNN_EPOCHS)

    # ── 架构 / 热力图 ──
    p.add_argument("--cnn_arch", choices=["simple", "mil"], default="mil",
                   help="simple = 原 SimpleCNN (整图); "
                        "mil = Attention-MIL (patch bag, 支持热力图导出)")
    p.add_argument("--n_heatmaps", type=int, default=20,
                   help="每个类别 (most_fake/most_real/most_uncertain) "
                        "导出多少张热力图 (仅 mil 架构有效)")
    return p.parse_args()


def main():
    import joblib

    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"\nLoading data…  fake={args.fake}  real={args.real}")
    (tr_imgs, y_tr), (va_imgs, y_va), (te_imgs, y_te) = load_and_split(
        args.fake, args.real,
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        seed=args.seed)
    print(f"  train {len(y_tr)}  val {len(y_va)}  test {len(y_te)}")

    # ── Sanity check 1: 标签平衡 ───────────────────────────────────────────
    print("\n[Sanity Check 1] 标签平衡")
    print(f"  y_tr balance (期望~0.5): {y_tr.mean():.4f}")
    print(f"  y_va balance (期望~0.5): {y_va.mean():.4f}")
    print(f"  y_te balance (期望~0.5): {y_te.mean():.4f}")

    # ── Sanity check 2: train/test 索引无重叠 ──────────────────────────────
    # load_and_split 内部用 permutation，这里直接验证图像内容不重叠
    print("\n[Sanity Check 2] Train/Test 图像重叠检测")
    # 用每张图的均值作为快速指纹（精确检测用 hash，但均值已足够定位问题）
    tr_fp = tr_imgs.mean(axis=(1, 2))
    te_fp = te_imgs.mean(axis=(1, 2))
    # 找 test 里和 train 均值完全相同（差 < 0.01）的样本数
    overlap_count = 0
    for fp in te_fp:
        if np.any(np.abs(tr_fp - fp) < 0.01):
            overlap_count += 1
    print(f"  疑似重叠样本数 (期望0): {overlap_count} / {len(te_fp)}")
    if overlap_count > 0:
        print("  ⚠️  WARNING: 检测到潜在数据泄露！")

    # ── Stage 1: FFT ───────────────────────────────────────────────────────
    fft_pkl  = os.path.join(args.out, "fft_logreg.pkl")
    feat_npz = os.path.join(args.out, "fft_features.npz")

    if args.skip_train_fft:
        print(f"\n[Stage 1] Skipped — loading {fft_pkl}")
        fft_model = joblib.load(fft_pkl)
        npz = np.load(feat_npz)
        fft_feats = {k: npz[k] for k in npz.files}
    else:
        fft_model, fft_feats = stage_train_fft(
            args.out, tr_imgs, y_tr, va_imgs, y_va, te_imgs, y_te)
        log(args.out,
            f"[FFT] val acc={fft_feats['va_acc']:.4f} "
            f"auc={fft_feats['va_auc']:.4f}  "
            f"test acc={fft_feats['te_acc']:.4f} "
            f"auc={fft_feats['te_auc']:.4f}")

    # ── Stage 2: CNN ───────────────────────────────────────────────────────
    import tensorflow as tf
    cnn_dir = os.path.join(args.out, "cnn_model")

    if args.skip_train_cnn:
        # 自动探测保存格式: SavedModel dir / .h5 / weights-only dir
        h5_path  = cnn_dir + ".h5"
        w_dir    = cnn_dir + "_weights"
        sm_index = os.path.join(cnn_dir, "saved_model.pb")

        # MIL 模型有自定义 Layer 子类, 加载时需注册 custom_objects.
        (MergeBagIntoBatch, SplitBagFromBatch,
         SoftmaxOverPatches, SumOverPatches) = _get_mil_layers()
        custom_objs = {
            "MergeBagIntoBatch":  MergeBagIntoBatch,
            "SplitBagFromBatch":  SplitBagFromBatch,
            "SoftmaxOverPatches": SoftmaxOverPatches,
            "SumOverPatches":     SumOverPatches,
        }

        if os.path.isfile(sm_index):
            print(f"\n[Stage 2] Skipped — loading SavedModel {cnn_dir}")
            cnn_model = tf.keras.models.load_model(
                cnn_dir, custom_objects=custom_objs)
        elif os.path.exists(h5_path):
            print(f"\n[Stage 2] Skipped — loading h5 {h5_path}")
            cnn_model = tf.keras.models.load_model(
                h5_path, custom_objects=custom_objs)
        elif os.path.isdir(w_dir):
            # weights-only: 用当前 --cnn_arch 重建空模型再 load_weights
            arch_file = os.path.join(w_dir, "arch.txt")
            if os.path.exists(arch_file):
                with open(arch_file) as f:
                    saved_arch = f.read().strip()
                if saved_arch != args.cnn_arch:
                    print(f"  WARNING: saved arch={saved_arch} but "
                          f"--cnn_arch={args.cnn_arch}; using saved.")
                    args.cnn_arch = saved_arch
            print(f"\n[Stage 2] Skipped — rebuilding {args.cnn_arch} CNN "
                  f"and loading weights from {w_dir}")
            if args.cnn_arch == "mil":
                cnn_model = _build_mil_cnn(
                    patch_shape=(PATCH_SIZE, PATCH_SIZE, 1),
                    n_patches=N_PATCHES, n_classes=2,
                    attn_hidden=ATTN_HIDDEN)
            else:
                cnn_model = _build_simple_cnn(
                    input_shape=(IMG_H, IMG_W, 1), n_classes=2)
            cnn_model.load_weights(os.path.join(w_dir, "model_weights"))
        else:
            raise FileNotFoundError(
                f"No model found. Tried:\n"
                f"  SavedModel : {cnn_dir}\n"
                f"  h5         : {h5_path}\n"
                f"  weights    : {w_dir}")
    else:
        global CNN_EPOCHS
        CNN_EPOCHS = args.cnn_epochs
        cnn_model, cnn_res = stage_train_cnn(
            args.out, tr_imgs, y_tr, va_imgs, y_va, te_imgs, y_te,
            arch=args.cnn_arch)
        log(args.out,
            f"[CNN arch={args.cnn_arch}] "
            f"val acc={cnn_res['va_acc']:.4f} "
            f"auc={cnn_res['va_auc']:.4f}  "
            f"test acc={cnn_res['te_acc']:.4f} "
            f"auc={cnn_res['te_auc']:.4f}")

    # ── Stage 3: Random baseline ───────────────────────────────────────────
    if not args.skip_baseline:
        stage_baseline(args.out, fft_model, cnn_model,
                       fft_feats, te_imgs, y_te)

    # ── Stage 4: Cascade (fixed thresholds) ───────────────────────────────
    stage_cascade(args.out, fft_model, cnn_model,
                  fft_feats, va_imgs, y_va, te_imgs, y_te,
                  low_thr=args.low_thr, high_thr=args.high_thr)

    # ── Stage 5: Auto-tune ────────────────────────────────────────────────
    best_low, best_high, te_acc, te_auc = stage_tune(
        args.out, fft_model, cnn_model,
        fft_feats, va_imgs, y_va, te_imgs, y_te,
        alpha=args.alpha)
    
    #计时补丁
        # ── Extra: real end-to-end throughput ───────────────────────────────
    e2e_scores, e2e_mask, e2e_t = real_e2e_cascade_predict(
        fft_model, cnn_model, te_imgs,
        low_thr=best_low, high_thr=best_high
    )
    e2e_acc, e2e_auc = eval_metrics(y_te, e2e_scores)

    log(args.out,
        f"[Real E2E Cascade] TEST acc={e2e_acc:.4f} "
        f"auc={e2e_auc:.4f}  "
        f"throughput={len(y_te)/(e2e_t['total']+1e-9):.2f} img/s  "
        f"(FFTfeat={e2e_t['t_feat']:.4f}s, "
        f"FFTcls={e2e_t['t_fft_cls']:.4f}s, "
        f"CNN={e2e_t['t_cnn']:.4f}s)")

    # ── Stage 6: Patch-level heatmap export (MIL only) ────────────────────
    if not args.skip_heatmap:
        stage_heatmap(args.out, cnn_model, te_imgs, y_te,
                      n_heatmaps=args.n_heatmaps)

    # ── Final comparison table ────────────────────────────────────────────
    header = (
        "\n" + "=" * 60 +
        "\n  FINAL COMPARISON\n" + "=" * 60 +
        f"\n  Pure FFT      TEST auc={fft_feats['te_auc']:.4f}  "
        f"acc={fft_feats['te_acc']:.4f}" +
        f"\n  Tuned Cascade TEST auc={te_auc:.4f}  acc={te_acc:.4f}"
        f"  (thr={best_low:.2f}/{best_high:.2f})" +
        "\n" + "=" * 60
    )
    log(args.out, header)


if __name__ == "__main__":
    main()