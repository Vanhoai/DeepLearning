import os
import numpy as np
import cv2 as cv
from numpy.typing import NDArray


def pad_image_2d(image, padding: int):
    if padding <= 0:
        return image

    padded = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode="constant")
    return padded


def conv2d(image: NDArray, kernels: NDArray, stride: int, padding: int):
    H, W, C = image.shape
    K, F, _, D = kernels.shape
    assert C == D, "Image channels must match kernel depth"

    padded = pad_image_2d(image, padding)
    HO = (H - F + 2 * padding) // stride + 1
    WO = (W - F + 2 * padding) // stride + 1

    O = np.zeros((HO, WO, K), dtype=np.float32)
    for k in range(K):
        for i in range(HO):
            for j in range(WO):
                region = padded[i * stride:i * stride + F, j * stride:j * stride + F, :]
                s = np.sum(region * kernels[k])
                O[i, j, k] = s

    return O


def main():
    root = os.getcwd()

    path = os.path.join(root, "assets", "man.jpg")
    img = cv.imread(path, cv.COLOR_BGR2RGB)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")

    # 3 Kernels, with shape (3, 3, 3)
    kernels = np.random.rand(3, 3, 3, 3)
    stride = 1
    padding = 1
    output = conv2d(img, kernels, stride, padding)
    print(f"Output shape: {output.shape}")
