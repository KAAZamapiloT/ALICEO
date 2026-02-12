
import cv2 as cv
import numpy as np


def enhance_luminance_clahe(
    ycbcr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:


    if ycbcr.dtype != np.uint8:
        raise ValueError("Input image must have dtype uint8")

    if ycbcr.ndim != 3 or ycbcr.shape[2] != 3:
        raise ValueError("Input must be a YCbCr image with 3 channels")

    # Extract luminance channel
    Y = ycbcr[:, :, 0]

    # Create CLAHE object
    clahe = cv.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    # Apply CLAHE to luminance
    Y_enhanced = clahe.apply(Y)

    # Replace luminance channel
    ycbcr_enhanced = ycbcr.copy()
    ycbcr_enhanced[:, :, 0] = Y_enhanced

    return ycbcr_enhanced


def enhance_luminance_gamma(
    ycbcr: np.ndarray,
    gamma: float = 1.2
) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("Gamma must be positive")

    if ycbcr.dtype != np.uint8:
        raise ValueError("Input image must have dtype uint8")

    # Extract luminance
    Y = ycbcr[:, :, 0].astype(np.float32) / 255.0

    # Gamma correction
    Y_gamma = np.power(Y, 1.0 / gamma)

    # Scale back
    Y_gamma = np.clip(Y_gamma * 255.0, 0, 255).astype(np.uint8)

    # Replace channel
    ycbcr_gamma = ycbcr.copy()
    ycbcr_gamma[:, :, 0] = Y_gamma

    return ycbcr_gamma
