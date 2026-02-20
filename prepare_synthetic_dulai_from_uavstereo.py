#!/usr/bin/env python3
"""Create a synthetic DuLAI-like dataset from UAVStereo stereo pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
import requests


# DuLAI Syn-R1440-like default split counts (train/val/test).
DEFAULT_SPLIT_COUNTS = (1152, 144, 144)

DEFAULT_GAMMA_RANGE = (1.6, 2.6)
DEFAULT_BRIGHTNESS_RANGE = (0.35, 0.7)
DEFAULT_POISSON_SCALE_RANGE = (60.0, 150.0)
DEFAULT_GAUSSIAN_STD_RANGE = (0.003, 0.01)

DOWNLOAD_CHUNK_SIZE = 1024 * 1024
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class StereoPair:
    key: str
    left_path: Path
    right_path: Path


def log(message: str) -> None:
    print(f"[INFO] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def sanitize_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return clean or "sample"


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def normalize_side_stem(stem: str) -> str:
    normalized = stem.lower()
    normalized = re.sub(r"(?:[_-](?:l|left|r|right))$", "", normalized)
    return normalized


def get_files_in_dir(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    files = [path for path in directory.iterdir() if path.is_file() and is_image_file(path)]
    files.sort()
    return files


def find_case_insensitive_sibling(parent: Path, name: str) -> Optional[Path]:
    lowered = name.lower()
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == lowered:
            return child
    return None


def discover_stereo_pairs(dataset_root: Path) -> List[StereoPair]:
    pairs: List[StereoPair] = []
    seen_keys: set[str] = set()

    left_dirs = [path for path in dataset_root.rglob("*") if path.is_dir() and path.name.lower() == "imageleft"]
    left_dirs.sort()
    if not left_dirs:
        warn(f"No ImageLeft directories found under: {dataset_root}")
        return pairs

    for left_dir in left_dirs:
        right_dir = find_case_insensitive_sibling(left_dir.parent, "ImageRight")
        if right_dir is None:
            continue

        left_files = get_files_in_dir(left_dir)
        right_files = get_files_in_dir(right_dir)
        if not left_files or not right_files:
            continue

        right_map: Dict[str, Path] = {}
        for right_path in right_files:
            right_key = normalize_side_stem(right_path.stem)
            right_map.setdefault(right_key, right_path)

        for left_path in left_files:
            left_key = normalize_side_stem(left_path.stem)
            right_path = right_map.get(left_key)
            if right_path is None:
                continue

            relative_parent = str(left_dir.parent.relative_to(dataset_root)).replace("\\", "/")
            raw_key = f"{relative_parent}_{left_key}"
            short_parent = sanitize_name(relative_parent.replace("/", "_"))[:48]
            short_stem = sanitize_name(left_key)[:48]
            suffix = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:10]
            pair_key = sanitize_name(f"{short_parent}_{short_stem}_{suffix}")
            if pair_key in seen_keys:
                continue
            pairs.append(StereoPair(key=pair_key, left_path=left_path, right_path=right_path))
            seen_keys.add(pair_key)

    return pairs


def format_bytes(size_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{size_bytes}B"


def parse_google_drive_file_id(url: str) -> Optional[str]:
    parsed = urlparse(url)
    if "drive.google.com" not in parsed.netloc.lower():
        return None

    match = re.search(r"/file/d/([^/]+)", parsed.path)
    if match:
        return match.group(1)

    query = parse_qs(parsed.query)
    if "id" in query and query["id"]:
        return query["id"][0]
    return None


def guess_download_filename(url: str, default_name: str) -> str:
    parsed = urlparse(url)
    file_id = parse_google_drive_file_id(url)
    if file_id:
        return f"uavstereo_{file_id}.zip"
    filename = unquote(Path(parsed.path).name)
    if filename:
        return filename
    return default_name


def try_google_drive_confirm_download(
    session: requests.Session, file_id: str, response: requests.Response
) -> requests.Response:
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        confirm_url = "https://drive.google.com/uc"
        return session.get(confirm_url, params={"export": "download", "id": file_id, "confirm": token}, stream=True, timeout=60)

    # Fallback: parse confirmation form token from HTML.
    text = response.text
    match = re.search(r"confirm=([0-9A-Za-z_]+)", text)
    if match:
        token = match.group(1)
        confirm_url = "https://drive.google.com/uc"
        return session.get(confirm_url, params={"export": "download", "id": file_id, "confirm": token}, stream=True, timeout=60)

    return response


def download_file(url: str, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_name = guess_download_filename(url, "uavstereo_download.zip")
    destination = destination_dir / destination_name
    if destination.exists() and destination.stat().st_size > 0:
        log(f"Reusing downloaded archive: {destination}")
        return destination

    temp_path = destination.with_suffix(destination.suffix + ".part")
    session = requests.Session()
    file_id = parse_google_drive_file_id(url)
    request_url = url
    params = None
    if file_id:
        request_url = "https://drive.google.com/uc"
        params = {"export": "download", "id": file_id}

    log(f"Downloading dataset archive from: {url}")
    response = session.get(request_url, params=params, stream=True, timeout=60)
    response.raise_for_status()
    if file_id:
        response = try_google_drive_confirm_download(session, file_id, response)
        response.raise_for_status()

    total = int(response.headers.get("content-length", "0"))
    downloaded = 0
    next_percent_log = 10
    with temp_path.open("wb") as file_obj:
        for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
            if not chunk:
                continue
            file_obj.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                percent = int((downloaded * 100) / total)
                while percent >= next_percent_log and next_percent_log <= 100:
                    log(
                        f"Download progress: {next_percent_log}% "
                        f"({format_bytes(downloaded)} / {format_bytes(total)})"
                    )
                    next_percent_log += 10

    temp_path.replace(destination)
    log(f"Download completed: {destination} ({format_bytes(destination.stat().st_size)})")
    return destination


def ensure_extracted(archive_path: Path, extraction_root: Path) -> Path:
    extraction_root.mkdir(parents=True, exist_ok=True)
    target_dir = extraction_root / archive_path.stem

    marker = target_dir / ".extract_complete"
    if marker.exists() and target_dir.exists():
        log(f"Reusing existing extraction: {target_dir}")
        return target_dir

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    log(f"Extracting archive: {archive_path}")
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tar_ref:
            tar_ref.extractall(target_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    marker.write_text("ok\n", encoding="utf-8")
    log(f"Extraction completed: {target_dir}")
    return target_dir


def locate_dataset_root(root: Path) -> Path:
    # UAVStereo structure has scene folders: Forest/Mining/Residential
    scene_names = {"forest", "mining", "residential"}
    root_children = {child.name.lower() for child in root.iterdir() if child.is_dir()}
    if scene_names.issubset(root_children):
        return root

    for candidate in root.rglob("*"):
        if not candidate.is_dir():
            continue
        child_names = {child.name.lower() for child in candidate.iterdir() if child.is_dir()}
        if scene_names.issubset(child_names):
            return candidate

    return root


def build_output_dirs(output_root: Path) -> Dict[str, Dict[str, Path]]:
    structure: Dict[str, Dict[str, Path]] = {}
    for split in SPLITS:
        split_root = output_root / split
        split_dirs = {
            "input_lowlight_rgb": split_root / "input_lowlight_rgb",
            "input_lowlight_mono": split_root / "input_lowlight_mono",
            "ground_truth_rgb": split_root / "ground_truth_rgb",
        }
        for path in split_dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        structure[split] = split_dirs

    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    structure["metadata"] = {"dir": metadata_dir}
    return structure


def clear_output_split_dirs(output_root: Path) -> None:
    for split in SPLITS:
        split_root = output_root / split
        if split_root.exists():
            shutil.rmtree(split_root)
    metadata_root = output_root / "metadata"
    if metadata_root.exists():
        shutil.rmtree(metadata_root)


def validate_split_counts(counts: Sequence[int]) -> Tuple[int, int, int]:
    if len(counts) != 3:
        raise ValueError("split counts must have exactly 3 values: train val test")
    train_count, val_count, test_count = (int(x) for x in counts)
    if min(train_count, val_count, test_count) < 0:
        raise ValueError("split counts cannot be negative")
    if train_count + val_count + test_count <= 0:
        raise ValueError("sum of split counts must be > 0")
    return train_count, val_count, test_count


def adjust_split_counts_to_available(
    requested: Tuple[int, int, int], available_total: int
) -> Tuple[int, int, int]:
    requested_total = sum(requested)
    if available_total >= requested_total:
        return requested

    if available_total <= 0:
        return (0, 0, 0)

    ratios = [count / requested_total for count in requested]
    base = [int(np.floor(ratio * available_total)) for ratio in ratios]
    remainder = available_total - sum(base)

    ranked = sorted(
        range(3),
        key=lambda idx: (ratios[idx] * available_total - base[idx], requested[idx]),
        reverse=True,
    )
    for idx in ranked:
        if remainder <= 0:
            break
        base[idx] += 1
        remainder -= 1
    return tuple(base)  # type: ignore[return-value]


def apply_lowlight_exposure(image: np.ndarray, gamma: float, brightness_scale: float) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    degraded = np.power(image, gamma) * brightness_scale
    return np.clip(degraded, 0.0, 1.0)


def add_poisson_gaussian_noise(
    image: np.ndarray, rng: np.random.Generator, poisson_scale: float, gaussian_std: float
) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    poisson_part = rng.poisson(image * poisson_scale).astype(np.float32) / float(poisson_scale)
    gaussian_part = rng.normal(0.0, gaussian_std, size=image.shape).astype(np.float32)
    noisy = poisson_part + gaussian_part
    return np.clip(noisy, 0.0, 1.0)


def degrade_rgb_and_mono(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    rng: np.random.Generator,
    gamma_range: Tuple[float, float],
    brightness_range: Tuple[float, float],
    poisson_scale_range: Tuple[float, float],
    gaussian_std_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    right_y = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32) / 255.0

    gamma = float(rng.uniform(*gamma_range))
    brightness = float(rng.uniform(*brightness_range))
    poisson_scale = float(rng.uniform(*poisson_scale_range))
    gaussian_std = float(rng.uniform(*gaussian_std_range))

    gt_rgb = np.clip(left_rgb, 0.0, 1.0)

    low_rgb = apply_lowlight_exposure(left_rgb, gamma, brightness)
    low_rgb = add_poisson_gaussian_noise(low_rgb, rng, poisson_scale, gaussian_std)

    low_mono = apply_lowlight_exposure(right_y, gamma, brightness)
    low_mono = add_poisson_gaussian_noise(low_mono, rng, poisson_scale, gaussian_std)

    metadata = {
        "gamma": gamma,
        "brightness_scale": brightness,
        "poisson_scale": poisson_scale,
        "gaussian_std": gaussian_std,
    }
    return gt_rgb, low_rgb, low_mono, metadata


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255.0, 0.0, 255.0).round().astype(np.uint8)


def write_split_lists(output_root: Path, split_to_names: Dict[str, List[str]]) -> None:
    for split, names in split_to_names.items():
        list_path = output_root / "metadata" / f"{split}.txt"
        list_path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")


def parse_optional_range(raw_values: Optional[Sequence[float]], default: Tuple[float, float]) -> Tuple[float, float]:
    if raw_values is None:
        return default
    if len(raw_values) != 2:
        raise ValueError("range arguments must have exactly 2 numbers: min max")
    low, high = float(raw_values[0]), float(raw_values[1])
    if low > high:
        raise ValueError("range min cannot be greater than range max")
    return low, high


def resolve_input_source(
    root_arg: Optional[str],
    archive_arg: Optional[str],
    url_arg: Optional[str],
    workspace_dir: Path,
) -> Path:
    if root_arg:
        root_path = Path(root_arg).expanduser().resolve()
        if not root_path.exists() or not root_path.is_dir():
            raise FileNotFoundError(f"--uavstereo-root not found or not a directory: {root_path}")
        log(f"Using provided UAVStereo root: {root_path}")
        return locate_dataset_root(root_path)

    if archive_arg:
        archive_path = Path(archive_arg).expanduser().resolve()
        if not archive_path.exists() or not archive_path.is_file():
            raise FileNotFoundError(f"--uavstereo-archive not found: {archive_path}")
        extracted = ensure_extracted(archive_path, workspace_dir / "extracted")
        return locate_dataset_root(extracted)

    if url_arg:
        downloads_dir = workspace_dir / "downloads"
        archive_path = download_file(url_arg, downloads_dir)
        extracted = ensure_extracted(archive_path, workspace_dir / "extracted")
        return locate_dataset_root(extracted)

    # Auto-detect common local locations.
    candidates = [
        Path("data/UAVStereo"),
        Path("UAVStereo"),
        workspace_dir / "extracted",
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists() and candidate.is_dir():
            detected = locate_dataset_root(candidate)
            log(f"Auto-detected UAVStereo root: {detected}")
            return detected

    raise FileNotFoundError(
        "No UAVStereo source found. Provide --uavstereo-root, --uavstereo-archive, or --uavstereo-url."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create synthetic DuLAI-like RGB/mono low-light pairs from UAVStereo."
    )
    parser.add_argument("--uavstereo-root", default=None, help="Path to extracted UAVStereo root")
    parser.add_argument("--uavstereo-archive", default=None, help="Path to UAVStereo archive (.zip/.tar.*)")
    parser.add_argument(
        "--uavstereo-url",
        default=None,
        help="URL to UAVStereo archive (direct link or Google Drive file URL)",
    )
    parser.add_argument(
        "--workspace-dir",
        default="data/_uavstereo_cache",
        help="Cache directory for downloads/extractions (default: data/_uavstereo_cache)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/DuLAI_synthetic",
        help="Output dataset directory (default: data/DuLAI_synthetic)",
    )
    parser.add_argument(
        "--split-counts",
        nargs=3,
        type=int,
        default=DEFAULT_SPLIT_COUNTS,
        metavar=("TRAIN", "VAL", "TEST"),
        help="DuLAI-like split counts (default: 1152 144 144)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on processed pairs before splitting",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--force", action="store_true", help="Remove existing output split folders before writing")

    parser.add_argument(
        "--gamma-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Gamma range for low-light simulation (default: 2.0 4.0)",
    )
    parser.add_argument(
        "--brightness-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Brightness scaling range (default: 0.1 0.4)",
    )
    parser.add_argument(
        "--poisson-scale-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Poisson noise scale range (default: 30.0 120.0)",
    )
    parser.add_argument(
        "--gaussian-std-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Gaussian noise std range (default: 0.003 0.02)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    split_counts = validate_split_counts(args.split_counts)
    gamma_range = parse_optional_range(args.gamma_range, DEFAULT_GAMMA_RANGE)
    brightness_range = parse_optional_range(args.brightness_range, DEFAULT_BRIGHTNESS_RANGE)
    poisson_scale_range = parse_optional_range(args.poisson_scale_range, DEFAULT_POISSON_SCALE_RANGE)
    gaussian_std_range = parse_optional_range(args.gaussian_std_range, DEFAULT_GAUSSIAN_STD_RANGE)

    if args.max_pairs is not None and args.max_pairs <= 0:
        parser.error("--max-pairs must be > 0")

    workspace_dir = Path(args.workspace_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        dataset_root = resolve_input_source(
            root_arg=args.uavstereo_root,
            archive_arg=args.uavstereo_archive,
            url_arg=args.uavstereo_url,
            workspace_dir=workspace_dir,
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))
    log(f"UAVStereo root resolved to: {dataset_root}")

    all_pairs = discover_stereo_pairs(dataset_root)
    if not all_pairs:
        raise RuntimeError("No valid stereo pairs found (ImageLeft/ImageRight).")
    log(f"Discovered stereo pairs: {len(all_pairs)}")

    requested_total = sum(split_counts)
    if args.max_pairs is not None:
        requested_total = min(requested_total, int(args.max_pairs))

    effective_counts = adjust_split_counts_to_available(split_counts, min(len(all_pairs), requested_total))
    if sum(effective_counts) < sum(split_counts):
        warn(
            f"Requested split total {sum(split_counts)} is larger than available/allowed pairs. "
            f"Using adjusted split counts: train={effective_counts[0]}, val={effective_counts[1]}, test={effective_counts[2]}"
        )

    total_to_use = sum(effective_counts)
    if total_to_use <= 0:
        raise RuntimeError("No pairs available after applying split constraints.")

    random.seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    shuffled = list(all_pairs)
    random.shuffle(shuffled)
    selected_pairs = shuffled[:total_to_use]

    split_boundaries = {
        "train": (0, effective_counts[0]),
        "val": (effective_counts[0], effective_counts[0] + effective_counts[1]),
        "test": (effective_counts[0] + effective_counts[1], total_to_use),
    }

    if args.force:
        log("Force mode enabled: removing existing split outputs.")
        clear_output_split_dirs(output_root)

    out_dirs = build_output_dirs(output_root)
    manifest_path = out_dirs["metadata"]["dir"] / "manifest.csv"

    split_to_names: Dict[str, List[str]] = {split: [] for split in SPLITS}
    split_counts_actual: Dict[str, int] = {split: 0 for split in SPLITS}

    log("Generating synthetic low-light samples...")
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "split",
                "filename",
                "left_source",
                "right_source",
                "gamma",
                "brightness_scale",
                "poisson_scale",
                "gaussian_std",
            ],
        )
        writer.writeheader()

        for index, pair in enumerate(selected_pairs):
            if index < split_boundaries["train"][1]:
                split = "train"
            elif index < split_boundaries["val"][1]:
                split = "val"
            else:
                split = "test"

            left_bgr = cv2.imread(str(pair.left_path), cv2.IMREAD_COLOR)
            right_bgr = cv2.imread(str(pair.right_path), cv2.IMREAD_COLOR)
            if left_bgr is None or right_bgr is None:
                warn(f"Skipping unreadable pair: {pair.left_path} | {pair.right_path}")
                continue

            if left_bgr.shape[:2] != right_bgr.shape[:2]:
                right_bgr = cv2.resize(right_bgr, (left_bgr.shape[1], left_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)

            gt_rgb, low_rgb, low_mono, degradation = degrade_rgb_and_mono(
                left_bgr=left_bgr,
                right_bgr=right_bgr,
                rng=np_rng,
                gamma_range=gamma_range,
                brightness_range=brightness_range,
                poisson_scale_range=poisson_scale_range,
                gaussian_std_range=gaussian_std_range,
            )

            filename = f"{index:06d}_{pair.key}.png"
            gt_path = out_dirs[split]["ground_truth_rgb"] / filename
            low_rgb_path = out_dirs[split]["input_lowlight_rgb"] / filename
            low_mono_path = out_dirs[split]["input_lowlight_mono"] / filename

            cv2.imwrite(str(gt_path), cv2.cvtColor(to_uint8(gt_rgb), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(low_rgb_path), cv2.cvtColor(to_uint8(low_rgb), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(low_mono_path), to_uint8(low_mono))

            split_counts_actual[split] += 1
            split_to_names[split].append(filename)

            writer.writerow(
                {
                    "split": split,
                    "filename": filename,
                    "left_source": str(pair.left_path),
                    "right_source": str(pair.right_path),
                    "gamma": f"{degradation['gamma']:.6f}",
                    "brightness_scale": f"{degradation['brightness_scale']:.6f}",
                    "poisson_scale": f"{degradation['poisson_scale']:.6f}",
                    "gaussian_std": f"{degradation['gaussian_std']:.6f}",
                }
            )

            if (index + 1) % 100 == 0 or (index + 1) == total_to_use:
                log(f"Processed {index + 1}/{total_to_use} pairs")

    write_split_lists(output_root, split_to_names)

    print()
    print("=" * 58)
    print("Synthetic DuLAI-Like Dataset Summary")
    print("=" * 58)
    print(f"Dataset root: {dataset_root}")
    print(f"Output root:  {output_root}")
    print(f"Pairs discovered: {len(all_pairs)}")
    print(
        "Requested split counts: "
        f"train={split_counts[0]}, val={split_counts[1]}, test={split_counts[2]}"
    )
    print(
        "Saved split counts: "
        f"train={split_counts_actual['train']}, "
        f"val={split_counts_actual['val']}, "
        f"test={split_counts_actual['test']}"
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
