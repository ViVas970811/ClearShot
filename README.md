# ClearShot

**AI-Powered Product Photo Enhancement for E-Commerce**

**Authors:** Dhanush Garikapati, Gowtham Tadikamalla, Rasagna Tirumani, Roshan Syed, Vivek Vasisht Ediga  

ClearShot transforms amateur product photos into professional-quality images using generative diffusion models. Built for small businesses that need catalog-quality product photography without expensive studio setups.

> DATA612 Course Project | University of Maryland | Group 3

---

## The Problem

Small e-commerce businesses rely on smartphone photos taken in uncontrolled environments — cluttered backgrounds, poor lighting, and low resolution. ClearShot fixes this automatically.

**Before & After:**

| Input (Amateur) | ClearShot Output |
|:---:|:---:|
| ![before](docs/assets/before_1.jpg) | ![after](docs/assets/after_1.jpg) |

---

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

Uses a curated subset (~5,000 images) from the [Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) dataset with synthetically generated degradations:

- Gaussian noise and blur
- JPEG compression artifacts
- Color and lighting inconsistencies
- Background clutter

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
├── notebooks/              # EDA, training, evaluation notebooks
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

---

## References

1. Zhang et al. *Adding Conditional Control to Text-to-Image Diffusion Models* (ControlNet), 2023
2. Xia et al. *DiffIR: Efficient Diffusion Model for Image Restoration*, ICCV 2023
3. Jin et al. *Neural Gaffer: Relighting Any Object via Diffusion*, 2024
4. Collins et al. *ABO: Dataset and Benchmarks for Real-World 3D Object Understanding*, CVPR 2022

---

## License

MIT
