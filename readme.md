# 🌙 ALICEO — Low-Light Image Enhancement with RGB-Monochrome Guidance

> A full training-ready research pipeline for low-light image enhancement using paired RGB and monochrome inputs, built on top of the ALICE/ALICC architecture.

---

## 🎥 Video Presentations

| Phase | Link |
|-------|------|
| 📽️ Phase 1 | [Watch on Google Drive](https://drive.google.com/file/d/1OsCJGZPyOlR695lz0FUVZpRLbUo9lNcm/view?usp=sharing) |
| 📽️ Phase 2 | [Watch on Google Drive](https://drive.google.com/file/d/15ZHldM5FCYAXwyd_iG1fXGevF4dzjAfC/view?usp=drive_link) |

## 📄 Manuscript

[Read the Manuscript](https://drive.google.com/file/d/1qB4kUIwKBdD3l0uyiyDg5oa4t-Kf4GO2/view?usp=sharing)

---

## 📖 Abstract

This project implements a complete experimental pipeline for low-light image enhancement using paired RGB and monochrome inputs. The repository started as an inference-only setup with pretrained weights and has been extended into a full training-ready system.

The new pipeline includes:
- Deterministic synthetic dataset generation
- A paired PyTorch data loader
- A TELU-enabled model wrapper compatible with pretrained weights
- Supervised training with multiple loss functions
- Evaluation using PSNR and SSIM

The work is inspired by ALICE-style low-light enhancement, where a noisy low-light RGB image is enhanced with the guidance of a structurally informative monochrome image. In the absence of a real paired training dataset, synthetic degradation is used to generate a small, reproducible benchmark for rapid experimentation.

---

## 🧩 Problem Statement

Low-light RGB images typically suffer from:
- 🌑 Low brightness
- 📡 Sensor noise
- 🎨 Color distortion
- 🔲 Weak edge detail

Monochrome images, by contrast, preserve structural information more reliably under low illumination — but carry no color. This project uses the RGB image for color content and the monochrome image for luminance guidance, then trains a deep model to reconstruct a cleaner, enhanced RGB result.

---

## 🎯 Objectives

- Support inference using existing pretrained weights
- Generate a small synthetic paired dataset from clean RGB images
- Build a trainable PyTorch pipeline around the existing model
- Introduce **TELU** as an optional activation replacement for ReLU and LeakyReLU
- Evaluate results with PSNR, SSIM, and visual comparisons

---

## 🗂️ Repository Structure

```
ALICEO/
├── config.py                          # Centralized configuration (DataConfig, ModelConfig, TrainConfig, EvalConfig)
├── train.py                           # Full supervised training loop
├── run_inference.py                   # Pretrained model inference script
├── data/
│   ├── generate_dataset.py            # Lightweight synthetic dataset generator
│   └── dataset_loader.py             # Paired PyTorch Dataset + train/val split
├── model/
│   └── modified_model.py             # TELU wrapper around ALICC
├── utils/
│   ├── losses.py                      # L1 + Edge + Perceptual loss
│   └── metrics.py                     # PSNR and SSIM
├── experiments/
│   └── pre_trained_model/
│       ├── ALICC.py                   # Original ALICC architecture
│       └── model_best.pth            # Pretrained weights
├── outputs/
│   ├── checkpoints/                   # Saved model checkpoints
│   ├── comparisons/                   # Validation visual comparisons
│   └── logs/                          # Training CSV logs
├── src/                               # Classical image processing utilities
├── results/                           # Inference outputs
├── genrate_data.ipynb                 # Jupyter notebook: dataset generation
└── train_eval.ipynb                   # Jupyter notebook: training + evaluation
```

---

## 🧪 Dataset Generation

Two pipelines are provided for creating synthetic low-light training data.

---

### 🔹 Pipeline 1 — Lightweight Synthetic Generator

**Files:** `data/generate_dataset.py` · `genrate_data.ipynb`

Generates a small paired low-light dataset from clean RGB images. Designed for rapid prototyping and debugging.

#### Input / Output

```
Input:  Folder of clean RGB images

Output: dataset/
        ├── rgb_low/     ← degraded RGB images
        ├── mono_low/    ← degraded grayscale images
        └── gt/          ← clean ground truth
```

#### Degradation Pipeline

For each sample:
1. Resize to `256 × 256`
2. Degrade RGB: brightness reduction → gamma correction → Gaussian noise → channel-wise color distortion → sensor-style shot + read noise
3. Degrade mono: grayscale conversion → low-light degradation
4. Save clean image as ground truth

#### Design Notes

- Capped at **10 samples** by default for fast iteration
- Fully deterministic (fixed seed per sample: `seed + index`)
- Output folders are cleared before each regeneration

#### Usage

```bash
python data/generate_dataset.py \
  --input-dir ./data/source_images \
  --output-dir ./data/synthetic_low_light
```

Or run the notebook:

```
genrate_data.ipynb
```

---

### 🔹 Pipeline 2 — UAVStereo-Based Synthetic Pipeline

**File:** `prepare_synthetic_dulai_from_uavstereo.py`

Generates a large-scale structured dataset using stereo image pairs for scalable training and research.

#### Key Features

- Uses **left image → RGB**, **right image → monochrome**
- Automatic train / validation / test splits
- Configurable degradation parameters
- Metadata logging via `manifest.csv`
- Automatic dataset download and extraction support

#### Output Structure

```
data/DuLAI_synthetic/
├── train/
│   ├── input_lowlight_rgb/
│   ├── input_lowlight_mono/
│   └── ground_truth_rgb/
├── val/
├── test/
└── metadata/
    └── manifest.csv
```

#### Usage

```bash
python prepare_synthetic_dulai_from_uavstereo.py \
  --uavstereo-root ./UAVStereo \
  --output-dir ./data/DuLAI_synthetic
```

---

### ⚡ Which Pipeline Should I Use?

| Use Case | Pipeline |
|----------|----------|
| Quick testing / debugging | Lightweight generator |
| Model training / research | UAVStereo pipeline |

> ⚠️ Both pipelines are deterministic when using the same seed. Existing outputs may be overwritten depending on flags used.

---

## 📦 Paired Dataset Loader

**File:** `data/dataset_loader.py`

Defines a PyTorch `Dataset` (`PairedLowLightDataset`) that:
- Loads matching files from `rgb_low/`, `mono_low/`, and `gt/`
- Validates filename consistency across all three folders at startup
- Returns float32 tensors normalized to `[0, 1]`
- Provides `create_data_splits()` for reproducible train/validation splitting

---

## ⚡ TELU Activation

**File:** `model/modified_model.py`

TELU (Tanh Exponential Linear Unit) is an optional smooth nonlinear activation:

```
TELU(x) = x · tanh(exp(x))
```

Controlled by a single flag:

```python
use_telu = True  # in config.py → ModelConfig
```

When enabled:
- All `ReLU` and `LeakyReLU` modules are replaced recursively throughout the model
- Parameter names remain unchanged — pretrained weights load safely because activation layers carry no learned parameters

When disabled, the original ALICC architecture behavior is fully preserved.

#### Why TELU May Help

Low-light enhancement requires sensitivity to very weak signals. TELU's smooth gradient flow may help the model respond more gently to low-intensity dark regions, avoiding premature suppression of subtle textures and edges.

---

## 🏗️ Model Wrapper

**File:** `model/modified_model.py`

`ModifiedALICC` wraps the original `ALICC` implementation rather than modifying the pretrained model file directly. This keeps the architecture modular and safe.

The wrapper provides:
- Dynamic import of the base architecture from `experiments/pre_trained_model/ALICC.py`
- Optional dependency stubs for `timm` and `ptflops` (so they are not required at runtime)
- Pretrained checkpoint loading with `module.` prefix stripping for DataParallel compatibility
- TELU replacement
- `extract_restored_output()` helper to safely unpack model outputs (list, tuple, or tensor)

---

## 🏋️ Training Pipeline

**File:** `train.py` · `train_eval.ipynb`

Provides a complete supervised training loop with:

| Feature | Detail |
|---------|--------|
| Loss functions | L1 (pixel) + Edge (Sobel gradients) + optional Perceptual (VGG16) |
| Optimizer | AdamW |
| Mixed precision | AMP via `torch.cuda.amp` |
| Checkpointing | `latest.pt` every epoch, `best.pt` on PSNR improvement |
| Logging | CSV log written to `outputs/logs/train_log.csv` |
| Validation | PSNR, SSIM, and comparison images after every epoch |

Loss weights are all configurable in `config.py`.

---

## 📊 Evaluation & Metrics

**File:** `utils/metrics.py`

Validation produces:
- **PSNR** — Peak Signal-to-Noise Ratio
- **SSIM** — Structural Similarity Index
- Side-by-side comparison images saved to `outputs/comparisons/` showing: input RGB · model output · ground truth

Inference evaluation (`run_inference.py`) additionally saves a `results/metrics_learned.csv` with per-image and average PSNR/SSIM.

---

## ⚙️ Configuration

All settings are centralized in `config.py` using dataclasses.

### `DataConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `source_images_dir` | `data/source_images` | Clean RGB images for generation |
| `dataset_root` | `data/synthetic_low_light` | Generated dataset output path |
| `image_size` | `256` | Square resize dimension |
| `max_samples` | `10` | Maximum samples to generate |
| `seed` | `42` | Base random seed |
| `val_ratio` | `0.2` | Fraction of data used for validation |
| `auto_generate_dataset` | `True` | Auto-generate if dataset is missing |

### `ModelConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `channels` | `32` | ALICC base channel width |
| `use_reference_mono` | `True` | Enable monochrome guidance input |
| `use_telu` | `True` | Replace ReLU/LeakyReLU with TELU |
| `pretrained_weights` | `experiments/.../model_best.pth` | Pretrained checkpoint path |
| `strict_checkpoint_loading` | `True` | Strict state dict loading |

### `TrainConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | `2` | Training batch size |
| `learning_rate` | `1e-4` | AdamW learning rate |
| `weight_decay` | `1e-4` | AdamW weight decay |
| `epochs` | `5` | Number of training epochs |
| `use_amp` | `True` | Enable mixed precision (CUDA only) |
| `l1_weight` | `1.0` | L1 loss weight |
| `edge_weight` | `0.2` | Sobel edge loss weight |
| `use_perceptual_loss` | `False` | Enable VGG perceptual loss |
| `perceptual_weight` | `0.05` | Perceptual loss weight |

---

## 🚀 Workflow

### Step 1 — Place Source Images

Put normal-light RGB images in the directory configured by `DataConfig.source_images_dir`:

```
data/source_images/
├── image_001.jpg
├── image_002.png
└── ...
```

### Step 2 — Generate Synthetic Dataset

**Option A — Command line:**
```bash
python data/generate_dataset.py \
  --input-dir data/source_images \
  --output-dir data/synthetic_low_light
```

**Option B — Jupyter Notebook:**
```
genrate_data.ipynb
```

**Option C — Automatic:** Set `auto_generate_dataset = True` in `config.py` and let `train.py` handle it.

### Step 3 — Train

```bash
python train.py
```

Or open `train_eval.ipynb` for an interactive training session.

This will:
- Build or reuse the synthetic dataset
- Split into train and validation subsets
- Load pretrained weights (if available)
- Train with TELU or original activations
- Save checkpoints, logs, and visual comparisons to `outputs/`

### Step 4 — Run Inference

```bash
python run_inference.py
```

Uses the pretrained model on the test set (`data/DuLAI_synthetic/test/`) and saves:
- `results/pretrained/` — learned model output
- `results/classical/` — classical CLAHE-based baseline
- `results/blended/` — hybrid (weighted blend of classical + learned)
- `results/metrics_learned.csv` — per-image and average PSNR/SSIM

---

## 🔬 Why Synthetic Data?

Since no real paired low-light training dataset is included, synthetic data provides:
- Input-target supervision from normal RGB images
- A fast testbed for verifying loss functions, data loading, checkpointing, and metrics
- A controlled environment for experimenting with architectural changes like TELU

### ⚠️ Limitations of Synthetic Data

Synthetic degradation cannot fully reproduce:
- Real sensor nonlinearities
- Motion blur and demosaicing artifacts
- Misalignment between RGB and monochrome cameras
- Lens noise and exposure differences
- Scene-dependent low-light failure modes

Good performance on synthetic data does not guarantee equal performance on real low-light imagery.

---

## ✅ Verification Status

The following have been verified to work correctly:

- [x] Python compilation of all new modules
- [x] Synthetic dataset generation
- [x] Data loading from `rgb_low/`, `mono_low/`, `gt/`
- [x] Pretrained checkpoint loading
- [x] Forward-pass smoke tests with `use_telu=False`
- [x] Forward-pass smoke tests with `use_telu=True`

---

## 📚 Reference

P. Yuan, L. Lin, J. Lin, Y. Liao, T. Zhao,
*Low-Light Aerial Imaging With Color and Monochrome Cameras.*