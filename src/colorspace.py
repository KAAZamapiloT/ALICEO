import cv2 as cv
import numpy as np


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype != np.uint8:
        raise ValueError("Input image must have dtype uint8")
    return cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    if not isinstance(rgb, np.ndarray):
        raise TypeError("Input must be a NumPy array")

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Input must be an RGB image with 3 channels")

    if rgb.dtype != np.uint8:
        raise ValueError("Input image must have dtype uint8")

    # OpenCV uses YCrCb internally
    ycrcb = cv.cvtColor(rgb, cv.COLOR_RGB2YCrCb)

    # Reorder channels: Y, Cr, Cb → Y, Cb, Cr
    ycbcr = ycrcb[:, :, [0, 2, 1]]

    return ycbcr


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    if not isinstance(ycbcr, np.ndarray):
        raise TypeError("Input must be a NumPy array")

    if ycbcr.ndim != 3 or ycbcr.shape[2] != 3:
        raise ValueError("Input must be a YCbCr image with 3 channels")

    if ycbcr.dtype != np.uint8:
        raise ValueError("Input image must have dtype uint8")

    # Reorder channels: Y, Cb, Cr → Y, Cr, Cb
    ycrcb = ycbcr[:, :, [0, 2, 1]]

    rgb = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2RGB)

    return rgb
