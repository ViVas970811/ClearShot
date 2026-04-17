# %%
"""
ClearShot Phase 4 - Inference Demo Script
==========================================

Demonstrates the full end-to-end enhancement pipeline:
    1. Data / weight restoration from Google Drive
    2. LoRA weight loading via peft
    3. Single-image enhancement through all 5 stages
    4. Batch enhancement on test set samples
    5. Side-by-side visual comparisons

Usage:
    - In Google Colab: Upload this file, then Run All
    - In VSCode: Open as interactive Python (# %% cell markers)
    - CLI: python notebooks/04_inference_demo.py

Prerequisites:
    - GPU runtime (T4 or better)
    - ClearShot_checkpoints and ClearShot_data folders in Google Drive
      (add shortcuts via 'Shared with me' -> right-click -> 'Add shortcut to Drive')
"""

# %% [markdown]
# ## 0. Setup & Installation

# %%
# Install dependencies (run once per Colab session)
# !pip install diffusers transformers accelerate peft controlnet-aux rembg realesrgan gradio
# !pip install Pillow==12.1.0  # Pin for rembg compatibility

# %%
import os
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project root: {PROJECT_ROOT}")
print(f"Working dir:  {os.getcwd()}")

# %% [markdown]
# ## 1. Restore Data from Google Drive (Colab Only)
#
# Skip this section if running locally with data already in place.

# %%
# --- Uncomment the block below when running on Google Colab ---

# from google.colab import drive
# drive.mount('/content/drive')
#
# # Clone the repo if not already present
# if not Path('/content/ClearShot').exists():
#     !git clone https://github.com/ViVas970811/ClearShot.git /content/ClearShot
#
# os.chdir('/content/ClearShot')
#
# # Restore data manifests (~2 minutes instead of 45-minute pipeline)
# !cp /content/drive/MyDrive/ClearShot_data/*.csv data/
# !mkdir -p data/metadata
# !cp -r /content/drive/MyDrive/ClearShot_data/metadata/* data/metadata/
#
# # Restore image data (if the tar.gz exists)
# if Path('/content/drive/MyDrive/ClearShot_data/clean_degraded.tar.gz').exists():
#     !tar xzf /content/drive/MyDrive/ClearShot_data/clean_degraded.tar.gz
#     print("Image data restored from tar.gz")
#
# # Set the LoRA weights path
# LORA_WEIGHTS_PATH = "/content/drive/MyDrive/ClearShot_checkpoints/final"
# print(f"LoRA weights: {LORA_WEIGHTS_PATH}")

# %%
# --- Local configuration (adjust paths as needed) ---
LORA_WEIGHTS_PATH = os.environ.get(
    "LORA_WEIGHTS_PATH", 
    str(PROJECT_ROOT / "ClearShot_checkpoints" / "final")
)
CONFIG_PATH = str(PROJECT_ROOT / "configs" / "inference_config.yaml")
OUTPUT_DIR = str(PROJECT_ROOT / "inference_output")

print(f"LoRA weights:  {LORA_WEIGHTS_PATH}")
print(f"Config:        {CONFIG_PATH}")
print(f"Output dir:    {OUTPUT_DIR}")

# Verify LoRA weights exist
lora_path = Path(LORA_WEIGHTS_PATH)
if lora_path.exists():
    files = list(lora_path.iterdir())
    print(f"LoRA files:    {[f.name for f in files]}")
else:
    print("WARNING: LoRA weights directory not found. Pipeline will run without LoRA.")
    LORA_WEIGHTS_PATH = None

# %% [markdown]
# ## 2. Initialize the Pipeline

# %%
from src.pipeline.enhancement_pipeline import ClearShotPipeline

pipeline = ClearShotPipeline(
    config_path=CONFIG_PATH,
    lora_weights_path=LORA_WEIGHTS_PATH,
    device=None,  # auto-detect (CUDA if available)
)

print("\nPipeline config:")
config = pipeline.get_config()
print(f"  Device: {config['device']}")
print(f"  LoRA:   {config['lora_weights_path']}")

# %% [markdown]
# ## 3. Single Image Enhancement

# %%
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for scripts
import matplotlib.pyplot as plt

# Find a test image
test_images = []
data_dirs = [
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "data" / "clean",
    PROJECT_ROOT / "data" / "degraded",
]

for d in data_dirs:
    if d.exists():
        test_images = sorted(d.rglob("*.jpg"))[:5]
        if test_images:
            print(f"Found {len(test_images)} test images in {d}")
            break

if not test_images:
    print("No test images found in data/. Creating a synthetic test image.")
    test_img = Image.new("RGB", (512, 512), (200, 180, 160))
    test_path = Path(OUTPUT_DIR) / "synthetic_test.jpg"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_img.save(test_path)
    test_images = [test_path]

# %%
# Enhance the first test image
test_image_path = test_images[0]
print(f"\nEnhancing: {test_image_path.name}")
print("-" * 50)

result = pipeline.enhance(
    str(test_image_path),
    bg_type="white",
)

# Save result with intermediates
pipeline.save_result(
    result,
    output_dir=OUTPUT_DIR,
    filename=test_image_path.stem,
    save_intermediates=True,
)

print(f"\nMetadata:")
for key, val in result.metadata.items():
    if key != "timings":
        print(f"  {key}: {val}")

# %% [markdown]
# ## 4. Visual Comparison

# %%
def plot_pipeline_stages(result, title="ClearShot Enhancement Pipeline", save_path=None):
    """Plot all pipeline stages side-by-side."""
    stages = [
        ("Original", result.original),
        ("Mask", result.mask),
        ("Edges", result.edge_map),
        ("Diffusion", result.diffusion_output),
        ("+ Background", result.with_background),
        ("Final (SR)", result.final),
    ]

    # Filter out None stages
    stages = [(name, img) for name, img in stages if img is not None]

    fig, axes = plt.subplots(1, len(stages), figsize=(4 * len(stages), 4))
    if len(stages) == 1:
        axes = [axes]

    for ax, (name, img) in zip(axes, stages):
        if img.mode == "L":
            ax.imshow(img, cmap="gray")
        elif img.mode == "RGBA":
            ax.imshow(img)
        else:
            ax.imshow(img)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison saved to: {save_path}")

    plt.close(fig)
    return fig


# Plot the single-image result
comparison_path = Path(OUTPUT_DIR) / "pipeline_comparison.png"
plot_pipeline_stages(
    result,
    title=f"ClearShot Pipeline: {test_image_path.name}",
    save_path=str(comparison_path),
)

# %% [markdown]
# ## 5. Batch Enhancement (Test Set Sample)

# %%
# Process a small batch of test images
BATCH_SIZE = 5  # Adjust based on available time/GPU

if len(test_images) > 1:
    batch_output = Path(OUTPUT_DIR) / "batch"
    batch_results = pipeline.batch_enhance(
        input_dir=str(test_images[0].parent),
        output_dir=str(batch_output),
        save_intermediates=True,
        max_images=BATCH_SIZE,
    )

    # Create a grid of before/after comparisons
    n = min(len(batch_results), BATCH_SIZE)
    if n > 0:
        fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        for i, res in enumerate(batch_results[:n]):
            axes[i, 0].imshow(res.original)
            axes[i, 0].set_title("Input", fontsize=11)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(res.final)
            axes[i, 1].set_title("ClearShot Output", fontsize=11)
            axes[i, 1].axis("off")

        fig.suptitle("ClearShot Batch Results", fontsize=14, fontweight="bold")
        plt.tight_layout()
        grid_path = Path(OUTPUT_DIR) / "batch_comparison.png"
        fig.savefig(str(grid_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nBatch comparison grid saved to: {grid_path}")
else:
    print("Only one test image available, skipping batch demo.")

# %% [markdown]
# ## 6. Summary

# %%
print("\n" + "=" * 60)
print("ClearShot Phase 4 - Inference Demo Complete")
print("=" * 60)
print(f"\nOutput directory: {OUTPUT_DIR}")

output_path = Path(OUTPUT_DIR)
if output_path.exists():
    all_files = list(output_path.rglob("*"))
    files_only = [f for f in all_files if f.is_file()]
    print(f"Files generated: {len(files_only)}")
    for f in sorted(files_only)[:20]:
        rel = f.relative_to(output_path)
        size_kb = f.stat().st_size / 1024
        print(f"  {rel} ({size_kb:.1f} KB)")

print(f"\nPipeline config: {pipeline.get_config()}")
print("\nDone!")
