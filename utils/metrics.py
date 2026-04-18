from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity


def tensor_to_uint8_image(image: torch.Tensor) -> np.ndarray:
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected a CHW tensor, got shape {tuple(image.shape)}")

    image = image.detach().cpu().float().clamp(0.0, 1.0)
    if image.size(0) == 1:
        image = image.repeat(3, 1, 1)
    array = image.permute(1, 2, 0).numpy()
    return (array * 255.0).round().astype(np.uint8)


def compute_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred_uint8 = tensor_to_uint8_image(prediction)
    target_uint8 = tensor_to_uint8_image(target)
    mse = np.mean((pred_uint8.astype(np.float32) - target_uint8.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def compute_ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred_uint8 = tensor_to_uint8_image(prediction)
    target_uint8 = tensor_to_uint8_image(target)
    return float(structural_similarity(target_uint8, pred_uint8, channel_axis=2, data_range=255))


def save_comparison_image(
    input_rgb: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    save_path: str | Path,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    input_img = tensor_to_uint8_image(input_rgb)
    prediction_img = tensor_to_uint8_image(prediction)
    target_img = tensor_to_uint8_image(target)

    def annotate(image: np.ndarray, label: str) -> np.ndarray:
        canvas_bgr = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        cv2.putText(canvas_bgr, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)

    comparison = np.concatenate(
        [
            annotate(input_img, "Input"),
            annotate(prediction_img, "Output"),
            annotate(target_img, "GT"),
        ],
        axis=1,
    )
    cv2.imwrite(str(save_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
