import sys
import types
from pathlib import Path

# -----------------------------------------------------------------------------
# Configurable paths
# -----------------------------------------------------------------------------
MODEL_DIR = Path("experiments/pre_trained_model")
MODEL_FILE = MODEL_DIR / "ALICC.py"
WEIGHTS_PATH = MODEL_DIR / "model_best.pth"

TESTSET_ROOT = Path("testset/Real")
COLOR_DIR = TESTSET_ROOT / "Color"
MONO_DIR = TESTSET_ROOT / "Mono"

OUTPUT_DIR = Path("results/pretrained/Real")

# -----------------------------------------------------------------------------
# Behavior flags
# -----------------------------------------------------------------------------
USE_ALIGNMENT = True
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

# Ensure we can import ALICC
sys.path.insert(0, str(MODEL_DIR.resolve()))
from ALICC import ALICC  # noqa: E402

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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Found {len(color_files)} test images.")

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
            mono = align_mono_to_rgb(rgb, mono)
        else:
            if mono.shape[:2] != rgb.shape[:2]:
                mono = cv.resize(mono, (rgb.shape[1], rgb.shape[0]))

        orig_h, orig_w = rgb.shape[:2]
        rgb, _ = pad_to_multiple(rgb, 8, PAD_MODE)
        mono, _ = pad_to_multiple(mono, 8, PAD_MODE)

        print(f"[INFO] Input size (padded): RGB={rgb.shape}, Mono={mono.shape}")

        rgb = normalize_image(rgb)
        mono = normalize_image(mono)

        rgb_tensor = tensor_from_rgb(rgb).to(device)
        mono_tensor = tensor_from_mono(mono).to(device)

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

        out_uint8 = (restored * 255.0).round().astype(np.uint8)
        out_bgr = cv.cvtColor(out_uint8, cv.COLOR_RGB2BGR)

        output_path = OUTPUT_DIR / rgb_path.name
        cv.imwrite(str(output_path), out_bgr)
        print(f"[INFO] Output saved to: {output_path}")


if __name__ == "__main__":
    main()
