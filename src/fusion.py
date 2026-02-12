import numpy as np


def fuse_luminance(ycbcr:np.ndarray,mono:np.ndarray,alpha:float)->np.ndarray:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")

    if ycbcr.dtype != np.uint8 or mono.dtype != np.uint8:
        raise ValueError("Inputs must have dtype uint8")

    if ycbcr.shape[:2] != mono.shape:
        raise ValueError("YCbCr and mono image must have same spatial dimensions")

        # Extract luminance channel
    Y_rgb = ycbcr[:, :, 0].astype(np.float32)
    Y_mono = mono.astype(np.float32)

    # Weighted fusion
    Y_fused = alpha * Y_rgb + (1.0 - alpha) * Y_mono

    # Clip and convert back
    Y_fused = np.clip(Y_fused, 0, 255).astype(np.uint8)

    # Replace luminance channel
    ycbcr_fused = ycbcr.copy()
    ycbcr_fused[:, :, 0] = Y_fused

    return ycbcr_fused