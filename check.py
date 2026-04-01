import numpy as np

path = r"D:\design1\GANDCTAnalysis\database\gandct\cached_gray_128_pack\celeba_test_20000_images.npy"
arr = np.load(path, mmap_mode="r")
print("shape:", arr.shape)
print("dtype:", arr.dtype)
print("min:", arr.min())
print("max:", arr.max())

