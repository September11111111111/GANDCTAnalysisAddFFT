import os
import argparse
import numpy as np
from PIL import Image
import pywt
from skimage.transform import resize

def to_gray_float(img_rgb: np.ndarray) -> np.ndarray:
    # img_rgb: HxWx3 uint8
    img = img_rgb.astype(np.float32) / 255.0
    # standard luminance
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

def subband_to_uint8(x: np.ndarray) -> np.ndarray:
    # abs + log1p for dynamic range compression
    x = np.log1p(np.abs(x))
    # robust min-max
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi <= lo:
        hi = lo + 1e-6
    x = (x - lo) / (hi - lo)
    x = np.clip(x, 0, 1)
    return (x * 255.0).astype(np.uint8)

def process_one(in_path: str, out_path: str, wavelet: str):
    img = Image.open(in_path).convert("RGB")
    arr = np.asarray(img)  # HxWx3
    if arr.shape[0] != 128 or arr.shape[1] != 128:
        # 保险：如果不是 128，就resize到128
        arr = np.asarray(img.resize((128, 128), Image.BILINEAR))

    gray = to_gray_float(arr)  # 128x128 float
    # 1-level DWT => subbands are 64x64
    cA, (cH, cV, cD) = pywt.dwt2(gray, wavelet=wavelet)

    # upsample back to 128x128
    H = resize(cH, (128, 128), anti_aliasing=True, preserve_range=True)
    V = resize(cV, (128, 128), anti_aliasing=True, preserve_range=True)
    D = resize(cD, (128, 128), anti_aliasing=True, preserve_range=True)

    ch1 = subband_to_uint8(H)  # LH-like (horizontal details)
    ch2 = subband_to_uint8(V)  # HL-like (vertical details)
    ch3 = subband_to_uint8(D)  # HH (diagonal details)

    out = np.stack([ch1, ch2, ch3], axis=-1)  # 128x128x3 uint8
    Image.fromarray(out).save(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("IN_DIR")
    ap.add_argument("OUT_DIR")
    ap.add_argument("--wavelet", default="haar", help="e.g. haar, db2, sym2")
    args = ap.parse_args()

    os.makedirs(args.OUT_DIR, exist_ok=True)
    files = sorted(os.listdir(args.IN_DIR))

    ok = 0
    skipped = 0
    for fn in files:
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        in_path = os.path.join(args.IN_DIR, fn)
        out_path = os.path.join(args.OUT_DIR, fn.rsplit(".", 1)[0] + ".png")
        try:
            process_one(in_path, out_path, args.wavelet)
            ok += 1
            if ok % 500 == 0:
                print(f"Processed {ok} images...")
        except Exception as e:
            skipped += 1
            print(f"[skip] {fn}: {e}")

    print(f"Done. OK={ok}, skipped={skipped}, out={args.OUT_DIR}")

if __name__ == "__main__":
    main()