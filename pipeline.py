"""
pipeline.py  —  Complete hybrid FFT-LogReg + CNN cascade pipeline
=================================================================
Stages (run in order):
  1. train_fft   — train FFT/PSD/HFE + LogReg, save fft_logreg.pkl
  2. train_cnn   — train SimpleCNN on .npy data, save TF SavedModel
  3. baseline    — random 50/50 router benchmark
  4. cascade     — confidence-based cascade inference
  5. tune        — auto-tune thresholds on val, report test metrics

Usage
-----
  # Full pipeline (all stages)
  python pipeline.py --lsun /path/lsun.npy --celeb /path/celeb.npy --out ./runs/exp1

  # Skip stages already done
  python pipeline.py --lsun ... --celeb ... --out ./runs/exp1 \
      --skip_train_fft --skip_train_cnn

  # Cascade only (both models already trained)
  python pipeline.py --lsun ... --celeb ... --out ./runs/exp1 \
      --skip_train_fft --skip_train_cnn --skip_baseline

Config defaults match mid-scale split: 5999/1799/1799 per class.
"""

import argparse
import os
import time
import numpy as np

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
CNN_BATCH       = 256
CNN_LR          = 1e-3
CNN_EARLY_STOP  = 5        # patience


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

def load_and_split(lsun_path, celeb_path,
                   n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, seed=SEED):
    """
    Returns (tr_imgs, y_tr), (va_imgs, y_va), (te_imgs, y_te)
    Labels: 0 = LSUN (fake/generated), 1 = CelebA (real)
    Images: float32 (N, 128, 128), raw pixel values 0-255
    """
    lsun  = np.load(lsun_path).astype(np.float32)
    celeb = np.load(celeb_path).astype(np.float32)

    def _split(arr):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(arr))
        a, b = n_train, n_train + n_val
        c = b + n_test
        assert len(arr) >= c, (
            f"Need {c} samples but only have {len(arr)}")
        return idx[:a], idx[a:b], idx[b:c]

    l_tr, l_va, l_te = _split(lsun)
    c_tr, c_va, c_te = _split(celeb)

    def _cat(l_idx, c_idx):
        imgs = np.concatenate([lsun[l_idx], celeb[c_idx]], 0)
        lbls = np.array([0] * len(l_idx) + [1] * len(c_idx), np.int32)
        return imgs, lbls

    return _cat(l_tr, c_tr), _cat(l_va, c_va), _cat(l_te, c_te)


# ─────────────────────────────────────────────────────────────────────────────
# 2. FFT / PSD / HFE feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _build_radial_cache(h=128, w=128, bins=64):
    cy, cx = h // 2, w // 2
    y, x   = np.indices((h, w))
    r      = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rb     = (r / (r.max() + 1e-8) * (bins - 1)).astype(np.int32)
    rb_flat = rb.ravel()
    cnt     = np.bincount(rb_flat, minlength=bins).astype(np.float32)
    cnt[cnt == 0] = 1.0
    return {"rb_flat": rb_flat, "cnt": cnt, "bins": bins}


def _fft_batch(imgs, cache, hfe_ratio=0.25):
    bins    = cache["bins"]
    rb_flat = cache["rb_flat"]
    cnt     = cache["cnt"]
    F   = np.fft.fftshift(np.fft.fft2(imgs, axes=(-2, -1)), axes=(-2, -1))
    psd = (F.real ** 2 + F.imag ** 2).astype(np.float32)
    B   = imgs.shape[0]
    prof = np.empty((B, bins), np.float32)
    for i in range(B):
        s = np.bincount(rb_flat, weights=psd[i].ravel(), minlength=bins)
        prof[i] = s / cnt
    prof = np.log1p(prof)
    k    = int(bins * (1 - hfe_ratio))
    hfe  = prof[:, k:].sum(1) / (prof.sum(1) + 1e-8)
    mu   = prof.mean(1, keepdims=True)
    std  = prof.std(1,  keepdims=True) + 1e-8
    prof = (prof - mu) / std
    return np.concatenate([prof, hfe[:, None]], axis=1).astype(np.float32)


def extract_fft_features(images, bins=FFT_BINS, hfe_ratio=FFT_HFE_RATIO,
                          batch_size=FFT_BATCH):
    cache  = _build_radial_cache(h=images.shape[1], w=images.shape[2], bins=bins)
    chunks = []
    for i in range(0, len(images), batch_size):
        chunks.append(_fft_batch(images[i:i+batch_size], cache, hfe_ratio))
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


def stage_train_cnn(out_dir, tr_imgs, y_tr, va_imgs, y_va, te_imgs, y_te):
    import tensorflow as tf
    from tensorflow.keras import mixed_precision

    print("\n" + "=" * 60)
    print("[Stage 2] Train SimpleCNN")
    print("=" * 60)

    mixed_precision.set_global_policy("mixed_float16")
    tf.config.optimizer.set_jit(True)

    train_ds = _make_tf_dataset(tr_imgs, y_tr, CNN_BATCH, shuffle=True)
    val_ds   = _make_tf_dataset(va_imgs, y_va, CNN_BATCH)
    test_ds  = _make_tf_dataset(te_imgs, y_te, CNN_BATCH)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = _build_simple_cnn(input_shape=(IMG_H, IMG_W, 1), n_classes=2)
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

    model.save(model_dir, save_format="tf")
    print(f"  CNN VAL  acc={va_acc:.4f}  auc={va_auc:.4f}")
    print(f"  CNN TEST acc={te_acc:.4f}  auc={te_auc:.4f}")
    print(f"  Saved model → {model_dir}")

    return model, {"va_score": va_score, "te_score": te_score,
                   "va_acc": va_acc, "va_auc": va_auc,
                   "te_acc": te_acc, "te_auc": te_auc}


# ─────────────────────────────────────────────────────────────────────────────
# 5. CNN predict helper (used in cascade & tune)
# ─────────────────────────────────────────────────────────────────────────────

def cnn_predict_proba(model, images, batch_size=CNN_BATCH):
    """images: (N, H, W) float32 0-255 → (N,) prob class-1"""
    import tensorflow as tf
    imgs = (images[:, :, :, np.newaxis] / 255.0).astype(np.float32)
    ds   = tf.data.Dataset.from_tensor_slices(imgs).batch(batch_size)
    out  = []
    for batch in ds:
        out.append(model(batch, training=False).numpy()[:, 1])
    return np.concatenate(out)[:len(images)]


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
# 10. Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Hybrid FFT+CNN pipeline")
    p.add_argument("--lsun",  required=True, help="LSUN .npy path")
    p.add_argument("--celeb", required=True, help="CelebA .npy path")
    p.add_argument("--out",   required=True, help="Output directory")

    p.add_argument("--skip_train_fft", action="store_true")
    p.add_argument("--skip_train_cnn", action="store_true")
    p.add_argument("--skip_baseline",  action="store_true")

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
    return p.parse_args()


def main():
    import joblib

    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"\nLoading data…  lsun={args.lsun}  celeb={args.celeb}")
    (tr_imgs, y_tr), (va_imgs, y_va), (te_imgs, y_te) = load_and_split(
        args.lsun, args.celeb,
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
        print(f"\n[Stage 2] Skipped — loading {cnn_dir}")
        cnn_model = tf.keras.models.load_model(cnn_dir)
    else:
        global CNN_EPOCHS
        CNN_EPOCHS = args.cnn_epochs
        cnn_model, cnn_res = stage_train_cnn(
            args.out, tr_imgs, y_tr, va_imgs, y_va, te_imgs, y_te)
        log(args.out,
            f"[CNN] val acc={cnn_res['va_acc']:.4f} "
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