# src/main.py

import cv2 as cv
import numpy as np
import os

from src.colorspace import rgb_to_ycbcr, ycbcr_to_rgb
from src.fusion import fuse_luminance
from src.enhance import enhance_luminance_clahe
from src.align import align_mono_to_rgb
from evaluation.metrics import compute_psnr, compute_ssim



def load_rgb_image(path: str) -> np.ndarray:
    img = cv.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def load_mono_image(path: str) -> np.ndarray:
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def save_rgb_image(path: str, img: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv.imwrite(path, cv.cvtColor(img, cv.COLOR_RGB2BGR))


def main():
    # -----------------------------
    # Paths (edit if needed)
    # -----------------------------
    rgb_path = "data/rgb_lowlight.png"
    mono_path = "data/mono_lowlight.png"
    reference_path = "data/reference.png"  # optional (for metrics)

    output_dir = "results"
    output_image_path = os.path.join(output_dir, "fused_enhanced.png")

    # -----------------------------
    # Load input images
    # -----------------------------
    print("[INFO] Loading images...")
    rgb = load_rgb_image(rgb_path)
    mono = load_mono_image(mono_path)

    # -----------------------------
    # Align mono to RGB
    # -----------------------------
    print("[INFO] Aligning monochrome image...")
    mono_aligned = align_mono_to_rgb(rgb, mono)

    # -----------------------------
    # Convert RGB → YCbCr
    # -----------------------------
    print("[INFO] Converting RGB to YCbCr...")
    ycbcr = rgb_to_ycbcr(rgb)

    # -----------------------------
    # Luminance fusion
    # -----------------------------
    print("[INFO] Fusing luminance channels...")
    ycbcr_fused = fuse_luminance(
        ycbcr,
        mono_aligned,
        alpha=0.3
    )

    # -----------------------------
    # Contrast enhancement
    # -----------------------------
    print("[INFO] Enhancing luminance (CLAHE)...")
    ycbcr_enhanced = enhance_luminance_clahe(ycbcr_fused)

    # -----------------------------
    # Convert back to RGB
    # -----------------------------
    print("[INFO] Converting YCbCr to RGB...")
    final_rgb = ycbcr_to_rgb(ycbcr_enhanced)

    # -----------------------------
    # Save result
    # -----------------------------
    save_rgb_image(output_image_path, final_rgb)
    print(f"[INFO] Output saved to: {output_image_path}")

    # -----------------------------
    # Evaluation (optional)
    # -----------------------------
    if os.path.exists(reference_path):
        print("[INFO] Computing evaluation metrics...")
        reference = load_rgb_image(reference_path)

        psnr = compute_psnr(reference, final_rgb)
        ssim = compute_ssim(reference, final_rgb)

        print(f"PSNR: {psnr:.2f} dB")
        print(f"SSIM: {ssim:.4f}")
    else:
        print("[INFO] No reference image provided. Skipping metrics.")


if __name__ == "__main__":
    main()
