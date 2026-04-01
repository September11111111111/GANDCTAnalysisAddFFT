import os
import time
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from numpy.lib.format import open_memmap


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_imgs(d):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(exts)]
    files.sort()
    return files


def load_gray_128_uint8(path, size=(128, 128)):
    with Image.open(path) as img:
        img = img.convert("L")

        w, h = img.size
        s = min(w, h)  # 取短边，裁成正方形

        left = (w - s) // 2
        top = (h - s) // 2
        right = left + s
        bottom = top + s

        img = img.crop((left, top, right, bottom))   # 先 crop
        img = img.resize(size, Image.BILINEAR)       # 再 resize 到 128x128

        arr = np.asarray(img, dtype=np.uint8)
    return arr


def _load_one(args):
    idx, path, size = args
    try:
        arr = load_gray_128_uint8(path, size=size)
        return idx, arr, os.path.basename(path), None
    except Exception as e:
        return idx, None, os.path.basename(path), str(e)


def pack_dataset(src_dir, out_prefix, size=(128, 128), max_workers=None, limit=None, chunksize=64):
    files = list_imgs(src_dir)
    if limit is not None:
        files = files[:limit]

    n = len(files)
    if n == 0:
        raise ValueError(f"No images found in {src_dir}")

    h, w = size
    img_out = out_prefix + "_images.npy"
    name_out = out_prefix + "_names.npy"

    # 直接创建 .npy 内存映射文件，边处理边写
    images = open_memmap(img_out, mode="w+", dtype=np.uint8, shape=(n, h, w))
    names = [""] * n

    t0 = time.time()
    fail = 0
    done = 0

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 4) - 1)

    tasks = [(i, path, size) for i, path in enumerate(files)]

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for idx, arr, basename, err in pool.map(_load_one, tasks, chunksize=chunksize):
            if err is not None:
                fail += 1
                print(f"[FAIL] {files[idx]} -> {err}")
                images[idx] = 0
            else:
                images[idx] = arr

            names[idx] = basename
            done += 1

            if done % 1000 == 0 or done == n:
                elapsed = time.time() - t0
                speed = done / max(elapsed, 1e-8)
                print(f"[{done}/{n}] fail={fail}  {speed:.1f} img/s")

    # 刷盘
    images.flush()

    max_len = max(len(x) for x in names) if names else 1
    np.save(name_out, np.array(names, dtype=f"<U{max_len}"))

    dt = max(time.time() - t0, 1e-8)
    print("=" * 60)
    print(f"Source: {src_dir}")
    print(f"Images saved to: {img_out}")
    print(f"Names  saved to: {name_out}")
    print(f"Shape: {(n, h, w)}, dtype: uint8")
    print(f"Failures: {fail}")
    print(f"Elapsed: {dt:.2f}s  ({dt/60:.2f}min)")
    print(f"Throughput: {n / dt:.2f} images/s")
    print("=" * 60)

    return dt


def main():
    total_t0 = time.time()

    root = "D:/design1/GANDCTAnalysis/database/gandct"
    out_dir = os.path.join(root, "cached_gray_128_pack")
    ensure_dir(out_dir)

    src_fake = "D:\design1\GANDCTAnalysis\database\gandct\processed\\thumbnails128x128"
    out_fake = os.path.join(out_dir, "thumbnails_20000")

    print("脚本已启动")
    print("src_fake =", src_fake)
    print("out_fake =", out_fake)

    if not os.path.exists(src_fake):
        raise FileNotFoundError(f"Source directory not found: {src_fake}")

    files = list_imgs(src_fake)
    print(f"检测到图片数量: {len(files)}")
    print("本次将取前 20000 张图片")

    pack_dataset(src_fake, out_fake, size=(128, 128), limit=20000, chunksize=64)

    total_dt = time.time() - total_t0
    print()
    print("=" * 60)
    print(f"全部任务完成！总耗时: {total_dt:.2f}s  ({total_dt/60:.2f}min)")
    print("=" * 60)


if __name__ == "__main__":
    main()