from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_FILE = PROJECT_ROOT / "experiments" / "pre_trained_model" / "ALICC.py"


def _ensure_optional_dependency_stubs() -> None:
    if "ptflops" not in sys.modules:
        ptflops_stub = types.ModuleType("ptflops")

        def _missing_ptflops(*_args: Any, **_kwargs: Any) -> None:
            raise ImportError("ptflops is not installed; complexity reporting is unavailable.")

        ptflops_stub.get_model_complexity_info = _missing_ptflops
        sys.modules["ptflops"] = ptflops_stub

    if "timm" not in sys.modules:
        timm_stub = types.ModuleType("timm")
        timm_models_stub = types.ModuleType("timm.models")
        timm_layers_stub = types.ModuleType("timm.models.layers")

        class DropPath(nn.Identity):
            pass

        timm_layers_stub.DropPath = DropPath
        timm_models_stub.layers = timm_layers_stub
        timm_stub.models = timm_models_stub
        sys.modules["timm"] = timm_stub
        sys.modules["timm.models"] = timm_models_stub
        sys.modules["timm.models.layers"] = timm_layers_stub


def _load_base_model_class() -> type[nn.Module]:
    _ensure_optional_dependency_stubs()
    module_name = "_aliceo_base_model"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        if not BASE_MODEL_FILE.exists():
            raise FileNotFoundError(f"Base model definition not found: {BASE_MODEL_FILE}")
        spec = importlib.util.spec_from_file_location(module_name, BASE_MODEL_FILE)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load model module from {BASE_MODEL_FILE}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.ALICC


BaseALICC = _load_base_model_class()


class TELU(nn.Module):
    def __init__(self, clamp_value: float = 20.0):
        super().__init__()
        self.clamp_value = clamp_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exp_term = torch.exp(torch.clamp(x, min=-self.clamp_value, max=self.clamp_value))
        return x * torch.tanh(exp_term)


def replace_relu_family(module: nn.Module) -> int:
    replaced = 0
    for child_name, child in module.named_children():
        if isinstance(child, (nn.ReLU, nn.LeakyReLU)):
            setattr(module, child_name, TELU())
            replaced += 1
        else:
            replaced += replace_relu_family(child)
    return replaced


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized[key[7:] if key.startswith("module.") else key] = value
    return normalized


def load_pretrained_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    map_location: str | torch.device | None = None,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError("Unsupported checkpoint format. Expected a state-dict-like object.")

    normalized_state_dict = _normalize_state_dict_keys(state_dict)
    incompatible = model.load_state_dict(normalized_state_dict, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def extract_restored_output(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


class ModifiedALICC(BaseALICC):
    def __init__(
        self,
        Ch_img: int = 3,
        Channels: int = 32,
        state: str = "Train",
        REF: bool = True,
        tests: bool = False,
        use_telu: bool = False,
    ) -> None:
        super().__init__(Ch_img=Ch_img, Channels=Channels, state=state, REF=REF, tests=tests)
        self.use_telu = use_telu
        self.num_telu_replacements = replace_relu_family(self) if use_telu else 0

    def load_pretrained(
        self,
        checkpoint_path: str | Path,
        map_location: str | torch.device | None = None,
        strict: bool = True,
    ) -> tuple[list[str], list[str]]:
        return load_pretrained_weights(self, checkpoint_path, map_location=map_location, strict=strict)


def build_model(
    use_telu: bool,
    channels: int = 32,
    state: str = "Train",
    use_reference_mono: bool = True,
    tests: bool = False,
    checkpoint_path: str | Path | None = None,
    map_location: str | torch.device | None = None,
    strict: bool = True,
) -> ModifiedALICC:
    model = ModifiedALICC(
        Ch_img=3,
        Channels=channels,
        state=state,
        REF=use_reference_mono,
        tests=tests,
        use_telu=use_telu,
    )
    if checkpoint_path is not None:
        load_pretrained_weights(model, checkpoint_path, map_location=map_location, strict=strict)
    return model
