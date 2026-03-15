import os, time
import numpy as np

# ========= Utility: robust AUC (fallback if sklearn missing) =========
def roc_auc_score_manual(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)
    order = np.argsort(y_score)
    y_true = y_true[order]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.arange(1, len(y_true) + 1)
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)

# ========= Precompute radial bins =========
def build_radial_cache(h=128, w=128, bins=64):
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rmax = r.max() + 1e-8
    rb = (r / rmax * (bins - 1)).astype(np.int32)

    rb_flat = rb.ravel()
    cnt_per_bin = np.bincount(rb_flat, minlength=bins).astype(np.float32)
    cnt_per_bin[cnt_per_bin == 0] = 1.0

    return {
        "h": h,
        "w": w,
        "bins": bins,
        "rb_flat": rb_flat,
        "cnt_per_bin": cnt_per_bin,
    }

# ========= Fast feature extraction =========
def radial_profile_batch(psd_batch, cache):
    B = psd_batch.shape[0]
    bins = cache["bins"]
    rb_flat = cache["rb_flat"]
    cnt_per_bin = cache["cnt_per_bin"]

    out = np.empty((B, bins), dtype=np.float32)
    for i in range(B):
        flat = psd_batch[i].ravel()
        sum_per_bin = np.bincount(rb_flat, weights=flat, minlength=bins).astype(np.float32)
        out[i] = sum_per_bin / cnt_per_bin
    return out

def fft_psd_hfe_feature_batch(imgs, cache, hfe_ratio=0.25):
    F = np.fft.fft2(imgs, axes=(-2, -1))
    F = np.fft.fftshift(F, axes=(-2, -1))
    psd = (F.real * F.real + F.imag * F.imag).astype(np.float32)

    prof = radial_profile_batch(psd, cache)
    prof = np.log1p(prof).astype(np.float32)

    bins = prof.shape[1]
    k = int(bins * (1 - hfe_ratio))
    hfe = prof[:, k:].sum(axis=1) / (prof.sum(axis=1) + 1e-8)

    mu = prof.mean(axis=1, keepdims=True)
    std = prof.std(axis=1, keepdims=True) + 1e-8
    prof = (prof - mu) / std

    feats = np.concatenate([prof, hfe[:, None].astype(np.float32)], axis=1)
    return feats.astype(np.float32)

def extract_features_from_array(images, bins=64, hfe_ratio=0.25, batch_size=128, image_size=(128, 128)):
    if images.ndim != 3 or images.shape[1:] != image_size:
        raise ValueError(f"Unexpected images shape: {images.shape}, expected (N, {image_size[0]}, {image_size[1]})")

    cache = build_radial_cache(h=image_size[1], w=image_size[0], bins=bins)
    chunks = []

    for i in range(0, len(images), batch_size):
        imgs = images[i:i + batch_size].astype(np.float32, copy=False)
        feats = fft_psd_hfe_feature_batch(imgs, cache, hfe_ratio=hfe_ratio)
        chunks.append(feats)

    return np.concatenate(chunks, axis=0).astype(np.float32)

# ========= Split =========
def build_split_indices(n_total, n_train, n_val, n_test, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_total)
    need = n_train + n_val + n_test
    if n_total < need:
        raise ValueError(f"Not enough samples: have {n_total}, need {need}")
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:n_train + n_val + n_test]
    return train_idx, val_idx, test_idx

# ========= Classifier =========
def train_logreg_sklearn(X, y):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(X, y)
    return clf

def predict_proba_sklearn(clf, X):
    return clf.predict_proba(X)[:, 1]

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def train_logreg_numpy(X, y, lr=0.1, steps=2000, l2=1e-4):
    N, D = X.shape
    w = np.zeros(D, dtype=np.float64)
    b = 0.0
    y = y.astype(np.float64)

    for _ in range(steps):
        z = X @ w + b
        p = sigmoid(z)
        gw = (X.T @ (p - y)) / N + l2 * w
        gb = float(np.mean(p - y))
        w -= lr * gw
        b -= lr * gb
    return w, b

def predict_proba_numpy(wb, X):
    w, b = wb
    return sigmoid(X @ w + b)

# ========= Metrics =========
def eval_metrics(y_true, y_score, thr=0.5):
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)
    y_pred = (y_score >= thr).astype(np.int32)
    acc = float((y_pred == y_true).mean())

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = roc_auc_score_manual(y_true, y_score)
    return acc, auc

def append_to_output(txt_path, content):
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(content + "\n")

# ========= Main =========
def main():
    LSUN_IMG_NPY = "/dct/database/gandct/cached_gray_128_pack/lsun_bedroom_train_5000_128_images.npy"
    CELEBA_IMG_NPY = "/dct/database/gandct/cached_gray_128_pack/celea_test_128_images.npy"

    TRAIN_N = 5999
    VAL_N = 1799
    TEST_N = 1799
    SEED = 42

    BINS = 64
    HFE_RATIO = 0.25
    BATCH_SIZE = 128
    OUT_TXT = "/dct/final_models/ouput.txt"

    lsun_images = np.load(LSUN_IMG_NPY).astype(np.float32)
    celeb_images = np.load(CELEBA_IMG_NPY).astype(np.float32)

    l_tr_idx, l_va_idx, l_te_idx = build_split_indices(len(lsun_images), TRAIN_N, VAL_N, TEST_N, seed=SEED)
    c_tr_idx, c_va_idx, c_te_idx = build_split_indices(len(celeb_images), TRAIN_N, VAL_N, TEST_N, seed=SEED)

    tr_images = np.concatenate([lsun_images[l_tr_idx], celeb_images[c_tr_idx]], axis=0)
    va_images = np.concatenate([lsun_images[l_va_idx], celeb_images[c_va_idx]], axis=0)
    te_images = np.concatenate([lsun_images[l_te_idx], celeb_images[c_te_idx]], axis=0)

    y_tr = np.array([0] * len(l_tr_idx) + [1] * len(c_tr_idx), dtype=np.int32)
    y_va = np.array([0] * len(l_va_idx) + [1] * len(c_va_idx), dtype=np.int32)
    y_te = np.array([0] * len(l_te_idx) + [1] * len(c_te_idx), dtype=np.int32)

    t0 = time.time()
    X_tr = extract_features_from_array(tr_images, bins=BINS, hfe_ratio=HFE_RATIO, batch_size=BATCH_SIZE)
    X_va = extract_features_from_array(va_images, bins=BINS, hfe_ratio=HFE_RATIO, batch_size=BATCH_SIZE)
    X_te = extract_features_from_array(te_images, bins=BINS, hfe_ratio=HFE_RATIO, batch_size=BATCH_SIZE)
    t1 = time.time()

    total_imgs = len(tr_images) + len(va_images) + len(te_images)
    feat_ips = total_imgs / (t1 - t0)

    use_sklearn = True
    try:
        clf = train_logreg_sklearn(X_tr, y_tr)
        va_score = predict_proba_sklearn(clf, X_va)
        te_score = predict_proba_sklearn(clf, X_te)
    except Exception:
        use_sklearn = False
        wb = train_logreg_numpy(X_tr.astype(np.float64), y_tr, lr=0.2, steps=2000, l2=1e-4)
        va_score = predict_proba_numpy(wb, X_va.astype(np.float64))
        te_score = predict_proba_numpy(wb, X_te.astype(np.float64))

    va_acc, va_auc = eval_metrics(y_va, va_score)
    te_acc, te_auc = eval_metrics(y_te, te_score)

    t2 = time.time()
    _ = extract_features_from_array(te_images, bins=BINS, hfe_ratio=HFE_RATIO, batch_size=BATCH_SIZE)
    t3 = time.time()
    test_extract_ips = len(te_images) / (t3 - t2)

    summary = (
        "============================================================\n"
        f"[FFT/PSD/HFE baseline OPT + PACK NPY] seed={SEED} bins={BINS} hfe_ratio={HFE_RATIO} batch={BATCH_SIZE}\n"
        f"LSUN_IMG_NPY={LSUN_IMG_NPY}\n"
        f"CELEBA_IMG_NPY={CELEBA_IMG_NPY}\n"
        f"Split(per-class) train={TRAIN_N} val={VAL_N} test={TEST_N}\n"
        f"Feature throughput (train+val+test extract): {feat_ips:.2f} images/s\n"
        f"Test extract-only throughput: {test_extract_ips:.2f} images/s\n"
        f"Classifier: {'sklearn_logreg' if use_sklearn else 'numpy_logreg'}\n"
        f"VAL  acc={va_acc:.4f} auc={va_auc:.4f}\n"
        f"TEST acc={te_acc:.4f} auc={te_auc:.4f}\n"
        "============================================================"
    )
    print(summary)
    append_to_output(OUT_TXT, summary)

if __name__ == "__main__":
    main()