# evaluation/metrics.py

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(
    reference: np.ndarray,
    test: np.ndarray
) -> float:


    if reference.shape != test.shape:
        raise ValueError("Images must have the same shape")

    if reference.dtype != np.uint8 or test.dtype != np.uint8:
        raise ValueError("Images must have dtype uint8")

    return peak_signal_noise_ratio(
        reference,
        test,
        data_range=255
    )


def compute_ssim(
    reference: np.ndarray,
    test: np.ndarray
) -> float:


    if reference.shape != test.shape:
        raise ValueError("Images must have the same shape")

    if reference.dtype != np.uint8 or test.dtype != np.uint8:
        raise ValueError("Images must have dtype uint8")

    return structural_similarity(
        reference,
        test,
        channel_axis=2,
        data_range=255
    )
