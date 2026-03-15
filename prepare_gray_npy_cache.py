import os
import time
import numpy as np
from PIL import Image

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def list_imgs(d):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(exts)]
    files.sort()
    return files

def load_gray_128(path, size=(128, 128)):
    img = Image.open(path).convert("L").resize(size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

def pack_dataset(src_dir, out_prefix, size=(128, 128)):
    files = list_imgs(src_dir)
    n = len(files)
    if n == 0:
        raise ValueError(f"No images found in {src_dir}")

    h, w = size
    images = np.empty((n, h, w), dtype=np.float32)
    names = []

    t0 = time.time()
    fail = 0

    for i, path in enumerate(files):
        try:
            images[i] = load_gray_128(path, size=size)
            names.append(os.path.basename(path))
        except Exception as e:
            fail += 1
            print(f"[FAIL] {path} -> {e}")
            images[i] = 0.0
            names.append(os.path.basename(path))

        if (i + 1) % 500 == 0 or (i + 1) == n:
            print(f"[{i+1}/{n}] fail={fail}")

    img_out = out_prefix + "_images.npy"
    name_out = out_prefix + "_names.npy"

    np.save(img_out, images)
    np.save(name_out, np.array(names, dtype=object))

    dt = max(time.time() - t0, 1e-8)
    print("=" * 60)
    print(f"Source: {src_dir}")
    print(f"Images saved to: {img_out}")
    print(f"Names  saved to: {name_out}")
    print(f"Shape: {images.shape}, dtype: {images.dtype}")
    print(f"Failures: {fail}")
    print(f"Elapsed: {dt:.2f}s")
    print(f"Throughput: {n / dt:.2f} images/s")
    print("=" * 60)

def main():
    root = "/dct/database/gandct"
    out_dir = os.path.join(root, "cached_gray_128_pack")
    ensure_dir(out_dir)

    src_lsun = os.path.join(root, "processed", "lsun_bedroom_train_5000_128")
    src_celeb = os.path.join(root, "processed", "celea_test_128")

    out_lsun = os.path.join(out_dir, "lsun_bedroom_train_5000_128")
    out_celeb = os.path.join(out_dir, "celea_test_128")

    pack_dataset(src_lsun, out_lsun, size=(128, 128))
    pack_dataset(src_celeb, out_celeb, size=(128, 128))

if __name__ == "__main__":
    main()