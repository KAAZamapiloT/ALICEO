from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, random_split


VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _index_files(folder: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    }


def _describe_missing(reference: Iterable[str], current: Iterable[str]) -> list[str]:
    return sorted(set(reference) - set(current))


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_mono(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read grayscale image: {path}")
    return image


class PairedLowLightDataset(Dataset):
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.rgb_low_dir = self.root_dir / "rgb_low"
        self.mono_low_dir = self.root_dir / "mono_low"
        self.gt_dir = self.root_dir / "gt"

        required_dirs = [self.rgb_low_dir, self.mono_low_dir, self.gt_dir]
        missing_dirs = [str(path) for path in required_dirs if not path.exists()]
        if missing_dirs:
            raise FileNotFoundError(f"Missing dataset folders: {', '.join(missing_dirs)}")

        rgb_index = _index_files(self.rgb_low_dir)
        mono_index = _index_files(self.mono_low_dir)
        gt_index = _index_files(self.gt_dir)

        if not rgb_index:
            raise RuntimeError(f"No paired samples found in {self.rgb_low_dir}")

        rgb_names = set(rgb_index)
        mono_names = set(mono_index)
        gt_names = set(gt_index)
        if rgb_names != mono_names or rgb_names != gt_names:
            errors = []
            if rgb_names != mono_names:
                missing = _describe_missing(rgb_names, mono_names)
                if missing:
                    errors.append(f"mono_low missing: {', '.join(missing[:5])}")
            if rgb_names != gt_names:
                missing = _describe_missing(rgb_names, gt_names)
                if missing:
                    errors.append(f"gt missing: {', '.join(missing[:5])}")
            raise ValueError("Filename mismatch across paired folders. " + "; ".join(errors))

        self.samples = [
            {
                "name": name,
                "rgb_low_path": rgb_index[name],
                "mono_low_path": mono_index[name],
                "gt_path": gt_index[name],
            }
            for name in sorted(rgb_names)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]

        rgb_low = _load_rgb(sample["rgb_low_path"]).astype(np.float32) / 255.0
        mono_low = _load_mono(sample["mono_low_path"]).astype(np.float32) / 255.0
        gt = _load_rgb(sample["gt_path"]).astype(np.float32) / 255.0

        rgb_low_tensor = torch.from_numpy(rgb_low.transpose(2, 0, 1)).contiguous()
        mono_low_tensor = torch.from_numpy(mono_low[None, ...]).contiguous()
        gt_tensor = torch.from_numpy(gt.transpose(2, 0, 1)).contiguous()

        return {
            "name": sample["name"],
            "rgb_low": rgb_low_tensor,
            "mono_low": mono_low_tensor,
            "gt": gt_tensor,
        }


def create_data_splits(dataset: Dataset, val_ratio: float, seed: int) -> tuple[Dataset, Dataset]:
    if len(dataset) < 2:
        raise RuntimeError("At least two samples are required to create train/validation splits.")

    val_count = max(1, int(round(len(dataset) * val_ratio)))
    train_count = len(dataset) - val_count
    if train_count == 0:
        train_count = len(dataset) - 1
        val_count = 1

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_count, val_count], generator=generator)
