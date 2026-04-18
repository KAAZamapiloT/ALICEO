from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class SyntheticDatasetConfig:
    input_dir: Path
    output_dir: Path
    image_size: int = 256
    max_samples: int = 10
    seed: int = 42


def _list_image_files(input_dir: Path) -> list[Path]:
    files = [path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_SUFFIXES]
    return sorted(files)


def _load_rgb_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _resize_rgb(image: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)


def _add_sensor_noise(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shot_strength = float(rng.uniform(0.008, 0.03))
    read_strength = float(rng.uniform(0.002, 0.012))
    banding_strength = float(rng.uniform(0.0005, 0.004))

    sigma = np.sqrt(np.clip(image, 0.0, 1.0) * shot_strength + read_strength ** 2)
    sensor_noise = rng.normal(0.0, sigma, size=image.shape).astype(np.float32)

    if image.ndim == 3:
        row_shape = (image.shape[0], 1, image.shape[2])
    else:
        row_shape = (image.shape[0], 1)
    row_noise = rng.normal(0.0, banding_strength, size=row_shape).astype(np.float32)
    return image + sensor_noise + row_noise


def _degrade_rgb(clean_rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    image = clean_rgb.astype(np.float32) / 255.0

    brightness_scale = float(rng.uniform(0.18, 0.42))
    gamma = float(rng.uniform(1.4, 2.3))
    color_gain = rng.uniform(0.82, 1.18, size=(1, 1, 3)).astype(np.float32)
    color_bias = rng.normal(0.0, 0.02, size=(1, 1, 3)).astype(np.float32)

    degraded = np.power(np.clip(image, 0.0, 1.0), gamma) * brightness_scale
    degraded = degraded * color_gain + color_bias

    gaussian_sigma = float(rng.uniform(0.01, 0.035))
    gaussian_noise = rng.normal(0.0, gaussian_sigma, size=degraded.shape).astype(np.float32)
    degraded = degraded + gaussian_noise
    degraded = _add_sensor_noise(degraded, rng)
    degraded = np.clip(degraded, 0.0, 1.0)
    return (degraded * 255.0).round().astype(np.uint8)


def _degrade_mono(clean_rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gray = cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    brightness_scale = float(rng.uniform(0.15, 0.4))
    gamma = float(rng.uniform(1.5, 2.4))
    degraded = np.power(np.clip(gray, 0.0, 1.0), gamma) * brightness_scale

    gaussian_sigma = float(rng.uniform(0.008, 0.03))
    gaussian_noise = rng.normal(0.0, gaussian_sigma, size=degraded.shape).astype(np.float32)
    degraded = degraded + gaussian_noise
    degraded = _add_sensor_noise(degraded, rng)
    degraded = np.clip(degraded, 0.0, 1.0)
    return (degraded * 255.0).round().astype(np.uint8)


def _prepare_output_dirs(output_dir: Path) -> tuple[Path, Path, Path]:
    rgb_low_dir = output_dir / "rgb_low"
    mono_low_dir = output_dir / "mono_low"
    gt_dir = output_dir / "gt"
    rgb_low_dir.mkdir(parents=True, exist_ok=True)
    mono_low_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    for folder in (rgb_low_dir, mono_low_dir, gt_dir):
        for file_path in folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in VALID_SUFFIXES:
                file_path.unlink()

    return rgb_low_dir, mono_low_dir, gt_dir


def generate_synthetic_dataset(config: SyntheticDatasetConfig) -> list[Path]:
    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    image_files = _list_image_files(input_dir)
    if not image_files:
        raise RuntimeError(f"No supported images found in {input_dir}")

    rgb_low_dir, mono_low_dir, gt_dir = _prepare_output_dirs(output_dir)
    selected_files = image_files[: config.max_samples]
    generated_files: list[Path] = []

    for index, image_path in enumerate(selected_files):
        rng = np.random.default_rng(config.seed + index)
        clean_rgb = _resize_rgb(_load_rgb_image(image_path), config.image_size)
        rgb_low = _degrade_rgb(clean_rgb, rng)
        mono_low = _degrade_mono(clean_rgb, rng)

        sample_name = f"sample_{index:03d}.png"
        gt_path = gt_dir / sample_name
        rgb_path = rgb_low_dir / sample_name
        mono_path = mono_low_dir / sample_name

        cv2.imwrite(str(gt_path), cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_low, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mono_path), mono_low)
        generated_files.append(gt_path)

    return generated_files


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a small deterministic synthetic low-light dataset.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing clean RGB images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Dataset folder to create.")
    parser.add_argument("--image-size", type=int, default=256, help="Square resize dimension.")
    parser.add_argument("--max-samples", type=int, default=10, help="Maximum number of samples to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for deterministic degradations.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = SyntheticDatasetConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    generated = generate_synthetic_dataset(config)
    print(f"Generated {len(generated)} samples in {config.output_dir}")


if __name__ == "__main__":
    main()
