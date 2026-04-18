from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import AppConfig, get_config
from data.dataset_loader import PairedLowLightDataset, create_data_splits
from data.generate_dataset import SyntheticDatasetConfig, generate_synthetic_dataset
from model.modified_model import build_model, extract_restored_output
from utils.losses import EnhancementLoss
from utils.metrics import compute_psnr, compute_ssim, save_comparison_image


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dataset(config: AppConfig) -> None:
    dataset_root = Path(config.data.dataset_root)
    expected_dirs = [dataset_root / "rgb_low", dataset_root / "mono_low", dataset_root / "gt"]
    dataset_ready = all(path.exists() and any(path.iterdir()) for path in expected_dirs)
    if dataset_ready:
        return
    if not config.data.auto_generate_dataset:
        raise RuntimeError(
            "Dataset is missing and automatic generation is disabled. "
            "Enable `config.data.auto_generate_dataset` or run data/generate_dataset.py first."
        )

    generation_config = SyntheticDatasetConfig(
        input_dir=config.data.source_images_dir,
        output_dir=config.data.dataset_root,
        image_size=config.data.image_size,
        max_samples=config.data.max_samples,
        seed=config.data.seed,
    )
    generated = generate_synthetic_dataset(generation_config)
    print(f"[DATA] Generated {len(generated)} synthetic pairs at {config.data.dataset_root}")


def build_loaders(config: AppConfig) -> tuple[DataLoader, DataLoader]:
    dataset = PairedLowLightDataset(config.data.dataset_root)
    train_set, val_set = create_data_splits(dataset, config.data.val_ratio, config.data.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def to_device(batch: dict[str, torch.Tensor | list[str]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb_low = batch["rgb_low"].to(device, non_blocking=True)
    mono_low = batch["mono_low"].to(device, non_blocking=True)
    gt = batch["gt"].to(device, non_blocking=True)
    return rgb_low, mono_low, gt


def mean_meter() -> dict[str, float]:
    return {"total": 0.0, "l1": 0.0, "edge": 0.0, "perceptual": 0.0}


def update_mean_meter(meter: dict[str, float], losses: dict[str, torch.Tensor]) -> None:
    for key in meter:
        meter[key] += float(losses[key].detach().cpu().item())


def finalize_mean_meter(meter: dict[str, float], num_steps: int) -> dict[str, float]:
    return {key: value / max(num_steps, 1) for key, value in meter.items()}


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: EnhancementLoss,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    comparison_root: Path,
    max_visualizations: int,
) -> dict[str, float]:
    model.eval()
    loss_meter = mean_meter()
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    saved_visualizations = 0

    with torch.no_grad():
        for batch in val_loader:
            rgb_low, mono_low, gt = to_device(batch, device)
            with autocast(enabled=use_amp):
                raw_prediction = extract_restored_output(model(rgb_low, mono_low))
                _, losses = criterion(raw_prediction, gt)

            clamped_prediction = raw_prediction.clamp(0.0, 1.0)
            update_mean_meter(loss_meter, losses)

            names = batch["name"]
            for sample_idx in range(clamped_prediction.size(0)):
                psnr_values.append(compute_psnr(clamped_prediction[sample_idx], gt[sample_idx]))
                ssim_values.append(compute_ssim(clamped_prediction[sample_idx], gt[sample_idx]))

                if saved_visualizations < max_visualizations:
                    sample_name = names[sample_idx] if isinstance(names, list) else f"sample_{saved_visualizations:03d}.png"
                    save_path = comparison_root / f"epoch_{epoch:03d}_{sample_name}"
                    save_comparison_image(
                        input_rgb=rgb_low[sample_idx],
                        prediction=clamped_prediction[sample_idx],
                        target=gt[sample_idx],
                        save_path=save_path,
                    )
                    saved_visualizations += 1

    averaged_losses = finalize_mean_meter(loss_meter, len(val_loader))
    averaged_losses["psnr"] = float(np.mean(psnr_values)) if psnr_values else float("nan")
    averaged_losses["ssim"] = float(np.mean(ssim_values)) if ssim_values else float("nan")
    return averaged_losses


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: AppConfig,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config.to_dict(),
            "metrics": metrics,
        },
        path,
    )


def ensure_log_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_total",
                "train_l1",
                "train_edge",
                "train_perceptual",
                "val_total",
                "val_l1",
                "val_edge",
                "val_perceptual",
                "val_psnr",
                "val_ssim",
            ]
        )


def append_log_row(path: Path, epoch: int, train_losses: dict[str, float], val_metrics: dict[str, float]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                epoch,
                f"{train_losses['total']:.6f}",
                f"{train_losses['l1']:.6f}",
                f"{train_losses['edge']:.6f}",
                f"{train_losses['perceptual']:.6f}",
                f"{val_metrics['total']:.6f}",
                f"{val_metrics['l1']:.6f}",
                f"{val_metrics['edge']:.6f}",
                f"{val_metrics['perceptual']:.6f}",
                f"{val_metrics['psnr']:.6f}",
                f"{val_metrics['ssim']:.6f}",
            ]
        )


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: EnhancementLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    loss_meter = mean_meter()

    for batch in train_loader:
        rgb_low, mono_low, gt = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            raw_prediction = extract_restored_output(model(rgb_low, mono_low))
            total_loss, losses = criterion(raw_prediction, gt)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        update_mean_meter(loss_meter, losses)

    return finalize_mean_meter(loss_meter, len(train_loader))


def main() -> None:
    config = get_config()
    seed_everything(config.data.seed)
    ensure_dataset(config)
    train_loader, val_loader = build_loaders(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.train.use_amp and device.type == "cuda"

    checkpoint_path = config.model.pretrained_weights if config.model.pretrained_weights.exists() else None
    model = build_model(
        use_telu=config.model.use_telu,
        channels=config.model.channels,
        state="Train",
        use_reference_mono=config.model.use_reference_mono,
        checkpoint_path=checkpoint_path,
        map_location=device,
        strict=config.model.strict_checkpoint_loading,
    ).to(device)

    criterion = EnhancementLoss(
        l1_weight=config.train.l1_weight,
        edge_weight=config.train.edge_weight,
        perceptual_weight=config.train.perceptual_weight,
        use_perceptual=config.train.use_perceptual_loss,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    scaler = GradScaler(enabled=use_amp)

    ensure_log_file(config.train.log_csv_path)
    best_psnr = -float("inf")

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    print(f"[INFO] TELU enabled: {config.model.use_telu}")
    if checkpoint_path is not None:
        print(f"[INFO] Loaded pretrained weights from {checkpoint_path}")

    for epoch in range(1, config.train.epochs + 1):
        train_losses = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )
        val_metrics = evaluate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            epoch=epoch,
            comparison_root=config.eval.comparison_dir,
            max_visualizations=config.eval.max_visualizations_per_epoch,
        )

        append_log_row(config.train.log_csv_path, epoch, train_losses, val_metrics)

        latest_checkpoint = config.train.checkpoint_dir / "latest.pt"
        save_checkpoint(latest_checkpoint, epoch, model, optimizer, scaler, config, val_metrics)

        if val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            best_checkpoint = config.train.checkpoint_dir / "best.pt"
            save_checkpoint(best_checkpoint, epoch, model, optimizer, scaler, config, val_metrics)

        print(
            f"[EPOCH {epoch:03d}] "
            f"train_total={train_losses['total']:.4f} "
            f"val_total={val_metrics['total']:.4f} "
            f"PSNR={val_metrics['psnr']:.4f} "
            f"SSIM={val_metrics['ssim']:.4f}"
        )

    print(f"[DONE] Logs: {config.train.log_csv_path}")
    print(f"[DONE] Checkpoints: {config.train.checkpoint_dir}")
    print(f"[DONE] Comparisons: {config.eval.comparison_dir}")


if __name__ == "__main__":
    main()
