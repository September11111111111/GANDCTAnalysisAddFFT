"""Script for cropping LSUN adopted from: https://github.com/ningyu1991/GANFingerprints/"""
import argparse
import os

from PIL import Image
import numpy as np
from skimage.transform import resize

from concurrent.futures import ProcessPoolExecutor


def transform_image(stupid):
    file_path, directory, output = stupid
    try:
        if not (file_path.endswith("png") or file_path.endswith("jpeg") or file_path.endswith("jpg")):
            return 0

        image = np.asarray(Image.open(f"{directory}/{file_path}"))

        # 兼容灰度图 -> 3通道
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        # 兼容 RGBA -> RGB
        if image.ndim == 3 and image.shape[2] >= 3:
            image = image[:, :, :3]

        # 非法维度直接跳过
        if image.ndim != 3:
            print(f"[skip] unexpected ndim={image.ndim}: {file_path}")
            return 0

        x, y, c = image.shape
        if x <= 0 or y <= 0:
            print(f"[skip] invalid shape {image.shape}: {file_path}")
            return 0

        if x != 128 or y != 128:
            # 更稳的中心裁剪：按短边裁成正方形
            side = min(x, y)
            x_start = (x - side) // 2
            y_start = (y - side) // 2

            image = np.copy(image)
            image = image[x_start:x_start + side, y_start:y_start + side]

            # 防止出现空裁剪
            if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
                print(f"[skip] empty crop -> {image.shape}: {file_path}")
                return 0

            image = resize(image.astype(np.float64), (128, 128), anti_aliasing=True, preserve_range=True)
            image = np.clip(image, 0, 255.).astype(np.uint8)

        Image.fromarray(image).save(f"{output}/{file_path}")
        return 1

    except Exception as e:
        print(f"[skip] {file_path}: {e}")
        return 0


def main(args):
    os.makedirs(args.OUTPUT, exist_ok=True)

    paths = os.listdir(args.DIRECTORY)
    packed = map(lambda p: (p, args.DIRECTORY, args.OUTPUT), paths)
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(transform_image, packed))

    ok = sum(results)
    total = len(paths)
    print(f"Done. Saved {ok}/{total} images to {args.OUTPUT}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("DIRECTORY", help="Source directory.", type=str)
    parser.add_argument("OUTPUT", help="Output directory.", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())