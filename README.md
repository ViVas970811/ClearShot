# ClearShot

**AI-Powered Product Photo Enhancement for E-Commerce**


ClearShot transforms amateur product photos into professional-quality images using generative diffusion models. Built for small businesses that need catalog-quality product photography without expensive studio setups.

> DATA612 Course Project | University of Maryland | Group 3

---

## The Problem

Small e-commerce businesses rely on smartphone photos taken in uncontrolled environments — cluttered backgrounds, poor lighting, and low resolution. ClearShot fixes this automatically.


## How It Works

ClearShot uses a multi-stage enhancement pipeline:

```
Input Image
    |
    v
[1. Background Removal] ---- rembg (U2-Net)
    |
    v
[2. Structural Extraction] -- Canny edges for ControlNet
    |
    v
[3. Diffusion Enhancement] -- Stable Diffusion 1.5 + LoRA + ControlNet
    |
    v
[4. Background Refinement] -- Clean studio-style background
    |
    v
[5. Super-Resolution] ------- Real-ESRGAN 2x upscale
    |
    v
Enhanced Output
```

---

## Key Features

- **Background Removal & Replacement** - Automatically isolates products and places them on clean backgrounds
- **Lighting & Color Correction** - Diffusion-based enhancement improves lighting, contrast, and color balance
- **Structure Preservation** - ControlNet conditioning ensures product shape and identity are maintained
- **Super-Resolution Upscaling** - 2x upscale via Real-ESRGAN for crisp, high-resolution output
- **Production-Ready UI** - Gradio web interface for single and batch image processing

---

## Architecture

```
+-------------------------------------------------------------+
|                      ClearShot Pipeline                      |
|                                                              |
|  +----------+   +----------+   +---------------------+      |
|  | rembg    |-->| Canny    |-->| SD 1.5 + LoRA       |      |
|  | (U2-Net) |   | Edges    |   | + ControlNet        |      |
|  +----------+   +----------+   +----------+----------+      |
|       |                                   |                  |
|       v                                   v                  |
|  +----------+                    +--------------+            |
|  | Product  |                    | Enhanced     |            |
|  | Mask     |------------------->| + Background |            |
|  +----------+                    +------+-------+            |
|                                         |                    |
|                                         v                    |
|                                  +--------------+            |
|                                  | Real-ESRGAN  |            |
|                                  | 2x Upscale   |            |
|                                  +--------------+            |
+-------------------------------------------------------------+
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/ViVas970811/ClearShot.git
cd ClearShot
pip install -r requirements.txt
```

### Produce the Dataset


```bash
# 1. Download ABO metadata (~5 min)
python data/scripts/download_abo.py --output_dir data/metadata

# 2. Curate balanced 24K subset + download images (~5 min, 24 workers)
python data/scripts/create_subset.py --catalog data/metadata/catalog.parquet --n_per_category 4700 --clean_existing

# 3. Generate synthetic degradation pairs (~15 min on 6 workers)
python data/scripts/synthesize_degradations.py --input_dir data/raw --n_variants 1 --max_workers 6

# 4. Run comprehensive EDA (~5 min)
python notebooks/run_eda.py
```

All scripts use `seed=42` for reproducibility.

### Enhance a Single Image

```python
from src.pipeline.enhancement_pipeline import ClearShotPipeline

pipeline = ClearShotPipeline()
result = pipeline.enhance("path/to/product_photo.jpg")
result.final.save("enhanced_output.jpg")
```

### Run the Web App

```bash
python app/gradio_app.py
# Open http://localhost:7860
```

### Docker

```bash
docker build -t clearshot .
docker run -p 7860:7860 --gpus all clearshot
```

---

## Dataset

Uses a curated, balanced subset of the [Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) dataset with synthetically generated degradations.

### Scale

| | Count |
|---|:---:|
| Clean ground-truth images | **24,131** |
| Degraded variants | **24,131** |
| Total training pairs | **24,131** |
| Train / Val / Test split | 19,304 / 2,413 / 2,414 |

Split is performed **by `image_id`** (not by pair) using `random_state=42` to guarantee no leakage between sets.

### Category Balance

| Category | Count | % |
|----------|:---:|:---:|
| apparel | 4,700 | 19.5% |
| electronics | 4,700 | 19.5% |
| furniture | 4,700 | 19.5% |
| home_decor | 4,700 | 19.5% |
| jewelry | 2,804 | 11.6% |
| outdoor | 1,901 | 7.9% |
| kitchen | 626 | 2.6% |

The initial iteration suffered from 69% phone-case dominance (keyword-matching mapped every `product_type` containing "CASE" to electronics). Fixed via an explicit `product_type` → category lookup table with per-category caps.

### Synthetic Degradations

Nine composable, randomized transforms simulate amateur smartphone photography:

- Gaussian noise (σ=10-50) and Gaussian blur (kernel 3-9)
- JPEG compression artifacts (quality 15-55)
- Color jitter (brightness/contrast/saturation/hue shifts)
- Uneven exposure and vignette
- Random elliptical shadows
- Background clutter (colored noise patches)
- Downscale + upscale (low-resolution simulation)

Each transform is applied independently with configurable probability, producing a healthy difficulty gradient across the training set.

### Degradation Validation

| Metric | Clean (median) | Degraded (median) | Change |
|---|:---:|:---:|:---:|
| Sharpness (Laplacian variance) | 66.83 | 36.78 | **-45.0%** |
| Brightness | 207.89 | 146.31 | -29.6% |
| Contrast | 62.44 | 44.50 | -28.7% |
| Colorfulness | 16.16 | 14.52 | -10.2% |

Paired PSNR: **11.80 dB** (median), SSIM: **0.737** (median) — substantial information loss in the recoverable regime that makes this a meaningful restoration problem for diffusion models.

---

## Evaluation

| Method | FID | SSIM | PSNR | LPIPS |
|--------|:---:|:----:|:----:|:-----:|
| OpenCV Baseline | TBD | TBD | TBD | TBD |
| SD + ControlNet (no LoRA) | TBD | TBD | TBD | TBD |
| **ClearShot (full pipeline)** | **TBD** | **TBD** | **TBD** | **TBD** |

Metrics: FID (distribution quality), SSIM (structural similarity), PSNR (signal fidelity), LPIPS (perceptual similarity).

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Diffusion Model | Stable Diffusion 1.5 (`diffusers`) |
| Fine-Tuning | LoRA via `peft` (rank=16, ~2-4M params) |
| Structural Conditioning | ControlNet (Canny edge) |
| Background Removal | `rembg` (U2-Net) |
| Super-Resolution | Real-ESRGAN |
| Web UI | Gradio |
| Training Infra | Google Colab (A100/T4) |

---

## Project Structure

```
ClearShot/
├── app/                    # Gradio web application
│   └── gradio_app.py
├── configs/                # Training and inference configs
│   ├── train_config.yaml
│   └── inference_config.yaml
├── data/
│   └── scripts/            # Data download and processing
│       ├── download_abo.py
│       ├── create_subset.py
│       └── synthesize_degradations.py
├── docs/                   # Architecture documentation
├── eda_results/            # EDA artifacts (visualizations + stats)
├── notebooks/              # EDA, training, evaluation notebooks
│   ├── run_eda.py
│   ├── 01_eda.ipynb
│   ├── 02_degradation_demo.ipynb
│   ├── 03_training.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── evaluation/         # FID, SSIM, PSNR, LPIPS metrics
│   │   └── metrics.py
│   ├── models/             # Diffusion enhancer, super-resolution
│   │   ├── diffusion_enhancer.py
│   │   └── super_resolution.py
│   ├── pipeline/           # End-to-end orchestrator
│   │   └── enhancement_pipeline.py
│   ├── preprocessing/      # Background removal, edge extraction
│   │   ├── background_removal.py
│   │   └── edge_extraction.py
│   └── training/           # Dataset, degradation, LoRA training
│       ├── dataset.py
│       ├── degradation.py
│       └── train_lora.py
├── tests/                  # Unit tests
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Team

**Group 3** - University of Maryland, DATA612

- Dhanush Garikapati
- Gowtham Tadikamalla
- Rasagna Tirumani
- Roshan Syed
- Vivek Vasisht Ediga

---

## References

1. Zhang et al. *Adding Conditional Control to Text-to-Image Diffusion Models* (ControlNet), 2023
2. Xia et al. *DiffIR: Efficient Diffusion Model for Image Restoration*, ICCV 2023
3. Jin et al. *Neural Gaffer: Relighting Any Object via Diffusion*, 2024
4. Collins et al. *ABO: Dataset and Benchmarks for Real-World 3D Object Understanding*, CVPR 2022

---

## License

MIT
