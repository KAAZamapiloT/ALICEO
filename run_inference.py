import csv
import sys
import types
from pathlib import Path

# -----------------------------------------------------------------------------
# Configurable paths
# -----------------------------------------------------------------------------
MODEL_DIR = Path("experiments/pre_trained_model")
MODEL_FILE = MODEL_DIR / "ALICC.py"
WEIGHTS_PATH = MODEL_DIR / "model_best.pth"

TESTSET_ROOT = Path(r"H:\image_processing\ALICEO\data\DuLAI_synthetic\test")

COLOR_DIR = TESTSET_ROOT / "input_lowlight_rgb"
MONO_DIR = TESTSET_ROOT / "input_lowlight_mono"
GT_DIR = TESTSET_ROOT / "ground_truth_rgb"

PRETRAINED_DIR = Path("results/pretrained/Real")
CLASSICAL_DIR = Path("results/classical/Real")
BLENDED_DIR = Path("results/blended/Real")
METRICS_CSV_PATH = Path("results/metrics_learned.csv")

# -----------------------------------------------------------------------------
# Blending config
# -----------------------------------------------------------------------------
FUSION_ALPHA = 0.3  # classical luminance fusion weight
BLEND_ALPHA = 0.3   # hybrid = blend_alpha * classical + (1 - blend_alpha) * learned

# -----------------------------------------------------------------------------
# Behavior flags
# -----------------------------------------------------------------------------
USE_ALIGNMENT = False
PAD_MODE = "reflect"  # np.pad mode
NORMALIZATION = "0_1"  # "0_1" only


# -----------------------------------------------------------------------------
# Optional dependency shims (ptflops, timm)
# -----------------------------------------------------------------------------
if "ptflops" not in sys.modules:
    ptflops_stub = types.ModuleType("ptflops")

    def _missing_ptflops(*_args, **_kwargs):
        raise ImportError("ptflops is not installed; get_model_complexity_info is unavailable in inference")

    ptflops_stub.get_model_complexity_info = _missing_ptflops
    sys.modules["ptflops"] = ptflops_stub

if "timm" not in sys.modules:
    timm_stub = types.ModuleType("timm")
    timm_models_stub = types.ModuleType("timm.models")
    timm_layers_stub = types.ModuleType("timm.models.layers")

    class DropPath:  # minimal stub
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, x):
            return x

    timm_layers_stub.DropPath = DropPath
    timm_models_stub.layers = timm_layers_stub
    timm_stub.models = timm_models_stub
    sys.modules["timm"] = timm_stub
    sys.modules["timm.models"] = timm_models_stub
    sys.modules["timm.models.layers"] = timm_layers_stub


import cv2 as cv
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Ensure we can import ALICC
sys.path.insert(0, str(MODEL_DIR.resolve()))
from ALICC import ALICC  # noqa: E402

from src.colorspace import rgb_to_ycbcr, ycbcr_to_rgb  # noqa: E402
from src.fusion import fuse_luminance  # noqa: E402
from src.enhance import enhance_luminance_clahe  # noqa: E402
from src.align import align_mono_to_rgb  # noqa: E402


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_rgb_image(path: Path) -> np.ndarray:
    img = cv.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load RGB image: {path}")
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def load_mono_image(path: Path) -> np.ndarray:
    img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load mono image: {path}")
    return img


def pad_to_multiple(img: np.ndarray, multiple: int, mode: str):
    h, w = img.shape[:2]
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0, 0, 0)

    if img.ndim == 2:
        pad_width = ((0, pad_h), (0, pad_w))
    else:
        pad_width = ((0, pad_h), (0, pad_w), (0, 0))

    padded = np.pad(img, pad_width, mode=mode)
    return padded, (0, pad_h, 0, pad_w)


def crop_to_original(img: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    return img[:orig_h, :orig_w]


def normalize_image(img: np.ndarray) -> np.ndarray:
    if NORMALIZATION == "0_1":
        return img.astype(np.float32) / 255.0
    raise ValueError(f"Unsupported normalization: {NORMALIZATION}")


def tensor_from_rgb(img_rgb: np.ndarray) -> torch.Tensor:
    # HWC -> CHW
    return torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0)


def tensor_from_mono(img_mono: np.ndarray) -> torch.Tensor:
    # HW -> 1HW
    return torch.from_numpy(img_mono).unsqueeze(0).unsqueeze(0)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")

    color_files = sorted(COLOR_DIR.glob("*.png"))
    if not color_files:
        raise RuntimeError(f"No images found in {COLOR_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading model...")
    model = ALICC(Ch_img=3, Channels=32, state="Test", REF=True, tests=False)
    model.to(device).eval()

    print("[INFO] Loading weights...")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Strip DataParallel prefix if present
    stripped_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            stripped_state[k[len("module."):]] = v
        else:
            stripped_state[k] = v

    model.load_state_dict(stripped_state, strict=True)

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    CLASSICAL_DIR.mkdir(parents=True, exist_ok=True)
    BLENDED_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Found {len(color_files)} test images.")

    psnr_values = []
    ssim_values = []
    metrics_rows = []

    for rgb_path in color_files:
        mono_path = MONO_DIR / rgb_path.name

        if not mono_path.exists():
            print(f"[WARN] Missing mono for {rgb_path.name}, skipping.")
            continue

        print(f"[INFO] Processing {rgb_path.name}...")

        rgb = load_rgb_image(rgb_path)
        mono = load_mono_image(mono_path)

        if USE_ALIGNMENT:
            print("[INFO] Aligning mono to RGB (ORB + fallback)...")
            mono_aligned = align_mono_to_rgb(rgb, mono)
        else:
            if mono.shape[:2] != rgb.shape[:2]:
                mono_aligned = cv.resize(mono, (rgb.shape[1], rgb.shape[0]))
            else:
                mono_aligned = mono

        # -----------------------------
        # Classical pipeline (uint8)
        # -----------------------------
        ycbcr = rgb_to_ycbcr(rgb)
        ycbcr_fused = fuse_luminance(ycbcr, mono_aligned, FUSION_ALPHA)
        ycbcr_enhanced = enhance_luminance_clahe(ycbcr_fused)
        classical_rgb = ycbcr_to_rgb(ycbcr_enhanced)

        # -----------------------------
        # Pretrained model pipeline
        # -----------------------------
        orig_h, orig_w = rgb.shape[:2]
        rgb_pad, _ = pad_to_multiple(rgb, 8, PAD_MODE)
        mono_pad, _ = pad_to_multiple(mono_aligned, 8, PAD_MODE)

        print(f"[INFO] Input size (padded): RGB={rgb_pad.shape}, Mono={mono_pad.shape}")

        rgb_norm = normalize_image(rgb_pad)
        mono_norm = normalize_image(mono_pad)

        rgb_tensor = tensor_from_rgb(rgb_norm).to(device)
        mono_tensor = tensor_from_mono(mono_norm).to(device)

        print("[INFO] Running inference...")
        with torch.no_grad():
            outputs = model(rgb_tensor, mono_tensor)

        if isinstance(outputs, (list, tuple)):
            restored = outputs[0]
        else:
            restored = outputs

        restored = restored.clamp(0.0, 1.0)
        restored = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
        restored = crop_to_original(restored, orig_h, orig_w)

        learned_uint8 = (restored * 255.0).round().astype(np.uint8)

        gt_path = GT_DIR / rgb_path.name
        if not gt_path.exists():
            print(f"[WARN] Missing GT for {rgb_path.name}, skipping metrics.")
        else:
            gt_rgb = load_rgb_image(gt_path)
            if gt_rgb.shape != learned_uint8.shape:
                print(
                    f"[WARN] Shape mismatch for {rgb_path.name}: "
                    f"GT={gt_rgb.shape}, pred={learned_uint8.shape}. Skipping metrics."
                )
            else:
                psnr_value = peak_signal_noise_ratio(gt_rgb, learned_uint8, data_range=255)
                ssim_value = structural_similarity(
                    gt_rgb,
                    learned_uint8,
                    channel_axis=2,
                    data_range=255,
                )
                psnr_values.append(psnr_value)
                ssim_values.append(ssim_value)
                metrics_rows.append((rgb_path.name, psnr_value, ssim_value))
                print(f"[METRIC] {rgb_path.name} | PSNR: {psnr_value:.4f} | SSIM: {ssim_value:.4f}")

        # -----------------------------
        # Hybrid blending
        # -----------------------------
        classical_f = classical_rgb.astype(np.float32) / 255.0
        learned_f = restored.astype(np.float32)
        hybrid = BLEND_ALPHA * classical_f + (1.0 - BLEND_ALPHA) * learned_f
        hybrid = np.clip(hybrid, 0.0, 1.0)
        hybrid_uint8 = (hybrid * 255.0).round().astype(np.uint8)

        # -----------------------------
        # Save outputs
        # -----------------------------
        pretrained_path = PRETRAINED_DIR / rgb_path.name
        classical_path = CLASSICAL_DIR / rgb_path.name
        blended_path = BLENDED_DIR / rgb_path.name

        cv.imwrite(str(pretrained_path), cv.cvtColor(learned_uint8, cv.COLOR_RGB2BGR))
        cv.imwrite(str(classical_path), cv.cvtColor(classical_rgb, cv.COLOR_RGB2BGR))
        cv.imwrite(str(blended_path), cv.cvtColor(hybrid_uint8, cv.COLOR_RGB2BGR))

        print(f"[INFO] Output saved to: {pretrained_path}")
        print(f"[INFO] Output saved to: {classical_path}")
        print(f"[INFO] Output saved to: {blended_path}")

    if psnr_values:
        average_psnr = float(np.mean(psnr_values))
        average_ssim = float(np.mean(ssim_values))
    else:
        average_psnr = float("nan")
        average_ssim = float("nan")
        print("[WARN] No valid GT-matched samples found. Average metrics are NaN.")

    with METRICS_CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_name", "psnr", "ssim"])
        for image_name, psnr_value, ssim_value in metrics_rows:
            writer.writerow([image_name, f"{psnr_value:.6f}", f"{ssim_value:.6f}"])
        writer.writerow(["AVERAGE", f"{average_psnr:.6f}", f"{average_ssim:.6f}"])

    print("[INFO] Learned-model evaluation complete.")
    print(f"[INFO] Average PSNR (learned): {average_psnr:.4f}")
    print(f"[INFO] Average SSIM (learned): {average_ssim:.4f}")
    print(f"[INFO] Metrics CSV saved to: {METRICS_CSV_PATH}")


if __name__ == "__main__":
    main()
