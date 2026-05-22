"""
inference.py — GAN 检测推理引擎
================================
对外提供 `GANDetector` 类, 给 Streamlit web 后端调用.
不依赖训练代码, 只依赖预处理函数和模型构建函数.

加载逻辑:
  - FFT-LogReg     : 从 ./runs/stylegan_mil_full/fft_logreg.pkl 加载
  - MIL CNN        : 从 ./runs/stylegan_mil_full/cnn_model 加载 (SavedModel)
  - SimpleCNN      : 从 ./runs/stylegan_simple_full/cnn_model 加载

推理流程 (cascade):
  1. 把 PIL.Image 经过和训练时完全一致的预处理 → (128, 128) uint8
  2. 提 FFT 径向特征 → LogReg → score
  3. 如果 score 落在 [low_thr, high_thr] 区间 → 走 CNN 精判 → 覆盖 score
  4. MIL 路径同时返回 4×4 热力图; SimpleCNN 路径热力图为 None

使用示例:
  >>> det = GANDetector(arch="mil")
  >>> result = det.predict_single("/path/to/image.png")
  >>> print(result["label"], result["score"])
  >>> heat = result["heatmap"]   # 4×4 ndarray 或 None
"""

import os
import io
import time
import numpy as np
from PIL import Image
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# 复用 prepare_gray_npy_cache 和 pipeline_patch 里的函数 / 常量
# ─────────────────────────────────────────────────────────────────────────────
# 设计选择: 不直接 import, 而是把需要的代码"内联"复制过来.
# 原因:
#   - prepare_gray_npy_cache.py 顶层有 main() 在 if __name__ == "__main__"
#     之外, import 时不会执行, 所以 import 是安全的.
#   - 但 pipeline_patch.py import 会把 tensorflow 提前 import, 拖慢启动;
#     而且我们只需要里面的 4 个函数和 1 个 dict 配置, 不必引入整个文件.
#   - inline 复制能让 inference.py 自包含, 部署更简单.
# ─────────────────────────────────────────────────────────────────────────────

# ── prepare_gray_npy_cache 的预处理 (一字不差复制, 保证训练/推理一致) ──
def _load_gray_128_uint8(pil_or_path, size=(128, 128)):
    """
    输入: PIL.Image 或 路径字符串 或 bytes
    输出: (128, 128) uint8 ndarray
    复刻 prepare_gray_npy_cache.load_gray_128_uint8 的全部逻辑:
      convert("L") → 中心 crop 成正方形 → BILINEAR resize 到 128×128
    """
    if isinstance(pil_or_path, (bytes, bytearray)):
        img = Image.open(io.BytesIO(pil_or_path))
    elif isinstance(pil_or_path, str):
        img = Image.open(pil_or_path)
    elif isinstance(pil_or_path, Image.Image):
        img = pil_or_path
    else:
        raise TypeError(f"unsupported input type: {type(pil_or_path)}")

    img = img.convert("L")
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top  = (h - s) // 2
    img  = img.crop((left, top, left + s, top + s))
    img  = img.resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


# ── pipeline_patch 的 patch 切割 (用于 MIL 输入) ──
PATCH_SIZE   = 32
PATCH_STRIDE = 32
GRID_H       = 128 // PATCH_STRIDE  # 4
GRID_W       = 128 // PATCH_STRIDE  # 4
N_PATCHES    = GRID_H * GRID_W      # 16


def _image_to_patches(img128):
    """
    img128: (128, 128) float32
    返回:    (16, 32, 32) float32
    """
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(img128, (PATCH_SIZE, PATCH_SIZE))
    win = win[::PATCH_STRIDE, ::PATCH_STRIDE]   # (4, 4, 32, 32)
    return win.reshape(N_PATCHES, PATCH_SIZE, PATCH_SIZE)


# ── pipeline_patch 的 FFT 径向特征提取 (单张图版本) ──
FFT_BINS      = 64
FFT_HFE_RATIO = 0.25


def _build_radial_cache(h=128, w=128, bins=FFT_BINS):
    """构建径向 binning 索引和归一化系数, 仅在第一次调用时构造一次."""
    from scipy import sparse
    cy, cx = h // 2, w // 2
    y, x   = np.indices((h, w))
    r      = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rb     = (r / (r.max() + 1e-8) * (bins - 1)).astype(np.int32)
    rb_flat = rb.ravel()

    cnt = np.bincount(rb_flat, minlength=bins).astype(np.float64)
    cnt[cnt == 0] = 1.0

    row = rb_flat
    col = np.arange(h * w, dtype=np.int32)
    val = (1.0 / cnt[rb_flat]).astype(np.float32)

    M = sparse.csr_matrix(
        (val, (row, col)),
        shape=(bins, h * w),
        dtype=np.float32,
    )
    return M


_RADIAL_CACHE = None

def _get_radial_cache():
    global _RADIAL_CACHE
    if _RADIAL_CACHE is None:
        _RADIAL_CACHE = _build_radial_cache()
    return _RADIAL_CACHE


def _extract_fft_feature(img128_float32, hfe_ratio=FFT_HFE_RATIO):
    """
    img128: (128, 128) float32, 像素 0-255
    输出:    (FFT_BINS + 1,) float32  与训练时格式完全一致
    """
    M = _get_radial_cache()

    # FFT → PSD
    F   = np.fft.fft2(img128_float32)
    F   = np.fft.fftshift(F)
    psd = (F.real ** 2 + F.imag ** 2).astype(np.float32)

    # Radial binning
    prof = (M @ psd.ravel()).astype(np.float32)         # (bins,)

    # log1p + HFE + z-score (与训练时完全一致)
    prof = np.log1p(prof)
    k    = int(FFT_BINS * (1 - hfe_ratio))
    hfe  = prof[k:].sum() / (prof.sum() + 1e-8)
    mu   = prof.mean()
    std  = prof.std() + 1e-8
    prof = (prof - mu) / std

    return np.concatenate([prof, [hfe]]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MIL 模型自定义 Layer (跟 pipeline_patch 完全一致, 加载 SavedModel 时需要)
# ─────────────────────────────────────────────────────────────────────────────

_MIL_LAYER_CLASSES = None

def _get_mil_layers():
    """延迟导入 tf, 第一次调用时构造 4 个自定义 Layer 类."""
    global _MIL_LAYER_CLASSES
    if _MIL_LAYER_CLASSES is not None:
        return _MIL_LAYER_CLASSES

    import tensorflow as tf
    from tensorflow.keras import layers as _layers

    class MergeBagIntoBatch(_layers.Layer):
        def call(self, x):
            shape = tf.shape(x)
            return tf.reshape(
                x, [shape[0] * shape[1], shape[2], shape[3], shape[4]])

        def compute_output_shape(self, input_shape):
            B, N, P1, P2, C = input_shape
            B_new = None if (B is None or N is None) else B * N
            return (B_new, P1, P2, C)

    class SplitBagFromBatch(_layers.Layer):
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
        def call(self, x):
            return tf.nn.softmax(x, axis=1)

        def compute_output_shape(self, input_shape):
            return input_shape

    class SumOverPatches(_layers.Layer):
        def call(self, x):
            return tf.reduce_sum(x, axis=1)

        def compute_output_shape(self, input_shape):
            return input_shape[:1] + input_shape[2:]

    _MIL_LAYER_CLASSES = (MergeBagIntoBatch, SplitBagFromBatch,
                          SoftmaxOverPatches, SumOverPatches)
    return _MIL_LAYER_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# 模型路径配置
# ─────────────────────────────────────────────────────────────────────────────
# 默认路径 (Windows / Linux 都用正斜杠, os.path.join 会处理)

DEFAULT_RUNS_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "runs")

DEFAULT_PATHS = {
    "fft_pkl":        os.path.join(DEFAULT_RUNS_ROOT,
                                   "stylegan_mil_full", "fft_logreg.pkl"),
    "mil_cnn_dir":    os.path.join(DEFAULT_RUNS_ROOT,
                                   "stylegan_mil_full", "cnn_model"),
    "simple_cnn_dir": os.path.join(DEFAULT_RUNS_ROOT,
                                   "stylegan_simple_full", "cnn_model"),
}

# Cascade 阈值 (来自你 tune 的最佳配置)
DEFAULT_LOW_THR  = 0.45
DEFAULT_HIGH_THR = 0.65

# 二分类判定阈值 (score < 此值 → fake)
FAKE_THRESHOLD = 0.5

# Label 约定
LABEL_FAKE = 0
LABEL_REAL = 1


# ─────────────────────────────────────────────────────────────────────────────
# 主类
# ─────────────────────────────────────────────────────────────────────────────

class GANDetector:
    """
    用法:
      det = GANDetector(arch="mil")        # 加载 MIL + FFT
      out = det.predict_single("img.png")  # 推理一张
      out = det.predict_batch(["a.png", "b.png", ...])
    """

    def __init__(self,
                 arch="mil",
                 fft_pkl_path=None,
                 cnn_model_dir=None,
                 low_thr=DEFAULT_LOW_THR,
                 high_thr=DEFAULT_HIGH_THR):
        """
        arch          : "mil" 或 "simple"
        fft_pkl_path  : FFT-LogReg pkl 路径; None 用默认
        cnn_model_dir : CNN SavedModel 目录; None 按 arch 用默认
        low/high_thr  : Cascade 路由阈值
        """
        if arch not in ("mil", "simple"):
            raise ValueError(f"arch must be 'mil' or 'simple', got {arch!r}")

        self.arch     = arch
        self.low_thr  = low_thr
        self.high_thr = high_thr

        if fft_pkl_path is None:
            fft_pkl_path = DEFAULT_PATHS["fft_pkl"]
        if cnn_model_dir is None:
            cnn_model_dir = (DEFAULT_PATHS["mil_cnn_dir"] if arch == "mil"
                             else DEFAULT_PATHS["simple_cnn_dir"])

        self.fft_pkl_path  = fft_pkl_path
        self.cnn_model_dir = cnn_model_dir

        # 加载 FFT 模型
        if not os.path.exists(fft_pkl_path):
            raise FileNotFoundError(f"FFT model not found: {fft_pkl_path}")
        self.fft_model = joblib.load(fft_pkl_path)

        # 加载 CNN 模型 (lazy: 第一次 predict 时才真正 load, 加快启动)
        self._cnn_model = None
        if not os.path.exists(cnn_model_dir):
            raise FileNotFoundError(
                f"CNN model dir not found: {cnn_model_dir}")

    # ── lazy load CNN ──
    @property
    def cnn_model(self):
        if self._cnn_model is None:
            import tensorflow as tf
            if self.arch == "mil":
                (MergeBagIntoBatch, SplitBagFromBatch,
                 SoftmaxOverPatches, SumOverPatches) = _get_mil_layers()
                custom_objs = {
                    "MergeBagIntoBatch":  MergeBagIntoBatch,
                    "SplitBagFromBatch":  SplitBagFromBatch,
                    "SoftmaxOverPatches": SoftmaxOverPatches,
                    "SumOverPatches":     SumOverPatches,
                }
                self._cnn_model = tf.keras.models.load_model(
                    self.cnn_model_dir, custom_objects=custom_objs,
                    compile=False)
            else:
                self._cnn_model = tf.keras.models.load_model(
                    self.cnn_model_dir, compile=False)
        return self._cnn_model

    # ── 内部: MIL 模型同时拿到 score 和 heatmap ──
    def _mil_forward(self, img128_uint8_array):
        """
        img128_uint8_array: (B, 128, 128) uint8
        返回 (scores, heatmaps):
          scores   : (B,) float32 P(class=1) = P(real)
          heatmaps : (B, 4, 4) float32

        关于 heatmap 的语义:
          heatmap[i,j] = attention[i,j] × P(real | patch[i,j])
          这与训练时 cnn_predict_with_heatmap / save_heatmaps 的实现完全
          一致 (取 patch_softmax[..., 1]). 数值越高 = 该 patch 既受 attention
          关注, 又倾向 real. 数值越低 = 倾向 fake 或被忽略.
          web 端可显示 (1 - heatmap) 得到 "fake 证据" 视图.
        """
        import tensorflow as tf
        from tensorflow.keras import models as kmodels

        # 切 patch
        B = img128_uint8_array.shape[0]
        patches = np.stack(
            [_image_to_patches(img128_uint8_array[i].astype(np.float32))
             for i in range(B)], axis=0)                # (B, 16, 32, 32)
        x = (patches[..., np.newaxis] / 255.0).astype(np.float32)

        # 拼 multi-output 模型
        attn_layer  = self.cnn_model.get_layer("attn_softmax")
        patch_layer = self.cnn_model.get_layer("patch_softmax")
        multi = kmodels.Model(
            inputs  = self.cnn_model.input,
            outputs = [self.cnn_model.output, attn_layer.output,
                       patch_layer.output],
        )

        bag_out, attn, patch_p = multi(x, training=False)
        bag_out  = bag_out.numpy()
        attn     = attn.numpy().squeeze(-1)             # (B, 16)
        patch_p  = patch_p.numpy()[..., 1]              # (B, 16) P(real|patch)
        heat     = (attn * patch_p).reshape(
            B, GRID_H, GRID_W).astype(np.float32)

        scores = bag_out[:, 1]                          # P(real)
        return scores.astype(np.float32), heat

    # ── 内部: SimpleCNN forward (无 heatmap) ──
    def _simple_forward(self, img128_uint8_array):
        """
        img128_uint8_array: (B, 128, 128) uint8
        返回 scores: (B,) float32
        """
        import tensorflow as tf
        x = (img128_uint8_array[..., np.newaxis] / 255.0).astype(np.float32)
        out = self.cnn_model(x, training=False).numpy()
        return out[:, 1].astype(np.float32)

    # ── 公开接口: 单张推理 ──
    def predict_single(self, image_input):
        """
        image_input: 路径字符串 / PIL.Image / bytes
        返回 dict:
          {
            "label"     : "fake" 或 "real",
            "score"     : float P(real),
            "fake_prob" : float P(fake),
            "route"     : "fft" 或 "cnn",   走哪条 cascade 路径
            "heatmap"   : np.ndarray (4,4) float32 或 None,
            "preprocessed": np.ndarray (128,128) uint8,
            "latency_ms": float,
            "arch"      : "mil" 或 "simple",
          }
        """
        return self.predict_batch([image_input])[0]

    # ── 公开接口: 批量推理 (cascade 高效路径) ──
    def predict_batch(self, image_inputs, on_progress=None):
        """
        image_inputs: list of (路径 / PIL.Image / bytes)
        on_progress : 可选回调 fn(done, total) 用于 streamlit 进度条
        返回 list[dict] (与 predict_single 同结构)
        """
        n = len(image_inputs)
        if n == 0:
            return []

        # ── 1) 全部预处理到 (n, 128, 128) uint8 ──
        t0 = time.time()
        imgs = np.zeros((n, 128, 128), dtype=np.uint8)
        for i, x in enumerate(image_inputs):
            imgs[i] = _load_gray_128_uint8(x)

        # ── 2) FFT 特征 ──
        fft_feats = np.zeros((n, FFT_BINS + 1), dtype=np.float32)
        for i in range(n):
            fft_feats[i] = _extract_fft_feature(imgs[i].astype(np.float32))

        fft_scores = self.fft_model.predict_proba(fft_feats)[:, 1]

        # ── 3) Cascade 路由 ──
        ambiguous = (fft_scores >= self.low_thr) & (fft_scores <= self.high_thr)
        routes    = np.where(ambiguous, "cnn", "fft")
        scores    = fft_scores.copy()
        heatmaps  = [None] * n

        # 不确定样本走 CNN
        if ambiguous.any():
            amb_idx = np.where(ambiguous)[0]
            amb_imgs = imgs[ambiguous]
            if self.arch == "mil":
                cnn_scores, cnn_heats = self._mil_forward(amb_imgs)
                for j, idx in enumerate(amb_idx):
                    scores[idx]    = cnn_scores[j]
                    heatmaps[idx]  = cnn_heats[j]
            else:
                cnn_scores = self._simple_forward(amb_imgs)
                for j, idx in enumerate(amb_idx):
                    scores[idx] = cnn_scores[j]

        elapsed_ms = (time.time() - t0) * 1000.0
        per_image_ms = elapsed_ms / n

        # ── 4) 整理结果 ──
        results = []
        for i in range(n):
            score = float(scores[i])
            fake_prob = 1.0 - score
            label = "fake" if score < FAKE_THRESHOLD else "real"
            results.append({
                "label":      label,
                "score":      score,
                "fake_prob":  fake_prob,
                "route":      str(routes[i]),
                "heatmap":    heatmaps[i],   # ndarray 或 None
                "preprocessed": imgs[i],
                "latency_ms": per_image_ms,
                "arch":       self.arch,
            })
            if on_progress is not None:
                on_progress(i + 1, n)

        return results


# ─────────────────────────────────────────────────────────────────────────────
# 热力图可视化 (供 web 端调用, 三种视图模式)
# ─────────────────────────────────────────────────────────────────────────────

def render_heatmap_overlay(img128_uint8, heat4x4,
                           mode="overlay", alpha=0.45, dpi=120):
    """
    img128_uint8 : (128,128) uint8 灰度图
    heat4x4      : (4,4) float32 热力图 或 None
    mode         : "image"   只显示原图
                   "heatmap" 只显示热力图 (上采样 + jet)
                   "overlay" 灰度底图 + 半透明 jet 热力图 (默认)
    返回: (H, W, 3) uint8 RGB ndarray, 可直接 st.image() 显示
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(4, 4), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    if mode == "image" or heat4x4 is None:
        ax.imshow(img128_uint8, cmap="gray", vmin=0, vmax=255)
    elif mode == "heatmap":
        heat_up = _bilinear_upsample(heat4x4, 128, 128)
        heat_n  = _normalize01(heat_up)
        ax.imshow(heat_n, cmap="jet", vmin=0, vmax=1)
    elif mode == "overlay":
        heat_up = _bilinear_upsample(heat4x4, 128, 128)
        heat_n  = _normalize01(heat_up)
        ax.imshow(img128_uint8, cmap="gray", vmin=0, vmax=255)
        ax.imshow(heat_n, cmap="jet", alpha=alpha, vmin=0, vmax=1)
    else:
        raise ValueError(f"unknown mode: {mode}")

    fig.tight_layout(pad=0)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    plt.close(fig)
    return rgba[..., :3]   # 只要 RGB


def _bilinear_upsample(arr2d, H, W):
    """4×4 → H×W 双线性插值, 保留原值域."""
    a = arr2d.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros((H, W), dtype=np.float32) + lo
    a01 = (a - lo) / (hi - lo)
    u8  = (a01 * 255).astype(np.uint8)
    big = Image.fromarray(u8).resize((W, H), Image.BILINEAR)
    return np.asarray(big, dtype=np.float32) / 255.0 * (hi - lo) + lo


def _normalize01(arr):
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def make_thumbnail_png_bytes(img128_uint8):
    """把 (128,128) uint8 编码成 PNG bytes, 用于存数据库."""
    buf = io.BytesIO()
    Image.fromarray(img128_uint8, mode="L").save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# 模型可用性检查 (供 web 启动时调用)
# ─────────────────────────────────────────────────────────────────────────────

def check_models_available():
    """
    返回 dict: {
       "fft":        bool,
       "mil_cnn":    bool,
       "simple_cnn": bool,
       "any_cnn":    bool,    # 至少一个 CNN 可用
       "details":    {...}    # 每个路径的详情
    }
    """
    paths = DEFAULT_PATHS
    fft_ok    = os.path.exists(paths["fft_pkl"])
    mil_ok    = (os.path.isfile(os.path.join(paths["mil_cnn_dir"],
                                              "saved_model.pb")) or
                 os.path.isdir(os.path.join(paths["mil_cnn_dir"],
                                            "variables")))
    simple_ok = (os.path.isfile(os.path.join(paths["simple_cnn_dir"],
                                              "saved_model.pb")) or
                 os.path.isdir(os.path.join(paths["simple_cnn_dir"],
                                            "variables")))
    return {
        "fft":        fft_ok,
        "mil_cnn":    mil_ok,
        "simple_cnn": simple_ok,
        "any_cnn":    mil_ok or simple_ok,
        "details": {
            "fft_pkl":        paths["fft_pkl"],
            "mil_cnn_dir":    paths["mil_cnn_dir"],
            "simple_cnn_dir": paths["simple_cnn_dir"],
        },
    }


if __name__ == "__main__":
    # 简单 smoke test
    print("Available models:", check_models_available())