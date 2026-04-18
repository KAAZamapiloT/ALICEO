from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class DataConfig:
    source_images_dir: Path = PROJECT_ROOT / "data" / "source_images"
    dataset_root: Path = PROJECT_ROOT / "data" / "synthetic_low_light"
    image_size: int = 256
    max_samples: int = 10
    seed: int = 42
    val_ratio: float = 0.2
    auto_generate_dataset: bool = True


@dataclass
class ModelConfig:
    channels: int = 32
    use_reference_mono: bool = True
    use_telu: bool = True
    pretrained_weights: Path = PROJECT_ROOT / "experiments" / "pre_trained_model" / "model_best.pth"
    strict_checkpoint_loading: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 5
    num_workers: int = 0
    use_amp: bool = True
    l1_weight: float = 1.0
    edge_weight: float = 0.2
    use_perceptual_loss: bool = False
    perceptual_weight: float = 0.05
    checkpoint_dir: Path = PROJECT_ROOT / "outputs" / "checkpoints"
    log_csv_path: Path = PROJECT_ROOT / "outputs" / "logs" / "train_log.csv"


@dataclass
class EvalConfig:
    comparison_dir: Path = PROJECT_ROOT / "outputs" / "comparisons"
    max_visualizations_per_epoch: int = 4


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: convert(inner) for key, inner in value.items()}
            if isinstance(value, list):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))


def get_config() -> AppConfig:
    return AppConfig()
