# Phase 5 - Evaluation

This document explains how to run the Phase 5 evaluation end to end and what
each artifact in `evaluation_results/` means.

## 1. What Phase 5 measures

Five methods are compared on the same test subset:

| Method            | Description                                                |
|-------------------|------------------------------------------------------------|
| `opencv`          | CLAHE (luminance) + bilateral filter + unsharp mask         |
| `pil_auto`        | PIL autocontrast + brightness/contrast/color/sharpness      |
| `background_only` | rembg + white studio background, no diffusion               |
| `sd_no_lora`      | ClearShot pipeline with LoRA disabled (ablation)            |
| `clearshot`       | Full ClearShot pipeline (LoRA + ControlNet; SR optional)    |

Per-image metrics: PSNR, SSIM (higher is better), LPIPS (lower is better).
Aggregate metric: FID between each method's output directory and a clean
reference directory.

## 2. Prerequisites

The evaluation reuses Phases 1 through 4. Before running Phase 5 you need:

1. `data/test_manifest.csv` (written by `data/scripts/synthesize_degradations.py`
   or restored from `ClearShot_data/` on Drive).
2. `data/val_manifest.csv` (for a larger FID reference, optional but
   recommended - see project plan R4).
3. `data/clean/` and `data/degraded/` populated with the 512x512 pair images
   referenced by the manifests (restore via `clean_degraded.tar.gz` from
   Drive).
4. `ClearShot_checkpoints/final/adapter_config.json` +
   `adapter_model.safetensors` - the PEFT LoRA adapter from Phase 3.
5. A GPU runtime for the diffusion baselines (`sd_no_lora`, `clearshot`).
   Classical baselines and metrics are CPU-only.

If any asset is missing, only the affected baseline(s) are skipped. The rest
of the pipeline still runs and produces valid partial reports.

## 3. Configuration

Edit `configs/evaluation_config.yaml` to control:

- `data.subset_size` - how many test images to evaluate (default 500, matches
  the plan). Stratified by category so every category is represented.
- `data.reference_manifest` - which manifest to use as the FID reference.
  `val_manifest.csv` gives 2,413 clean images, which is a stable FID estimate.
- `runner.eval_resolution` - common resolution used for PSNR/SSIM/LPIPS.
  Defaults to 512 so any 1024 SR output is compared fairly against the
  512 ground truth.
- `metrics.fid_batch_size`, `metrics.fid_num_workers`, `metrics.fid_dims` -
  FID runtime controls forwarded to `pytorch-fid`.
- `baselines.<name>.enabled` - toggle individual baselines on or off.
- `reporting.comparison_grid.methods_in_order` - column order in the grid PNG.

## 4. Running the evaluation

### Option A - notebook driver (recommended)

```bash
python notebooks/05_evaluation.py
```

The script is cell-based (VSCode/Cursor interactive or Colab compatible). It
loads `evaluation_config.yaml`, instantiates the runner, runs each baseline
resumably, computes FID using the config values, writes all report artifacts,
and prints a summary table at the end.

Override the LoRA path per-run with:

```bash
LORA_WEIGHTS_PATH="/content/drive/MyDrive/ClearShot_checkpoints/final" \
    python notebooks/05_evaluation.py
```

### Option B - programmatic

```python
from src.evaluation import (
    EvaluationConfig,
    EvaluationRunner,
    ImageQualityMetrics,
    build_baseline,
)

runner = EvaluationRunner(
    EvaluationConfig(
        manifest_path="data/test_manifest.csv",
        output_dir="evaluation_results",
        subset_size=500,
    ),
    metrics=ImageQualityMetrics(),
)

baselines = [
    build_baseline("opencv"),
    build_baseline("pil_auto"),
    build_baseline("background_only"),
    build_baseline("sd_no_lora"),
    build_baseline("clearshot", lora_weights_path="ClearShot_checkpoints/final"),
]

results = runner.run_all(baselines)
fid = runner.compute_fid_all(results)
```

## 5. Resume support

The runner is resumable at the image level. If the job is killed mid-run:

- Predictions already written under `<output>/<method>/<image_id>.png` and
  already present as a row in `<output>/<method>/per_image.csv` are skipped.
- Deleting either the prediction file or the CSV row forces a re-compute for
  that single image.
- The subset manifest is frozen on first run as
  `<output>/_subset_manifest.csv` so the same images are used across re-runs.

## 6. Output layout

```
evaluation_results/
  _subset_manifest.csv              # frozen subset used across baselines
  _reference_clean/                 # clean images at eval_resolution (FID ref)
  fid.json                          # {method: fid_value}
  opencv/
    per_image.csv                   # image_id, psnr, ssim, lpips, status, ...
    <image_id>.png                  # per-image enhanced output
  pil_auto/
    ...
  background_only/
    ...
  sd_no_lora/
    ...
  clearshot/
    ...
  report/
    summary_overall.csv             # method x {psnr_mean, ssim_mean, lpips_mean, fid, n}
    summary_per_category__psnr.csv
    summary_per_category__ssim.csv
    summary_per_category__lpips.csv
    paired_ttests_vs_clearshot.csv  # paired t-tests vs reference method
    comparison_grid.png             # one row per example, columns per method
    failure_cases.png               # worst-k LPIPS rows for the reference method
```

## 7. Tests

Lightweight tests that do not require a GPU or the LoRA adapter:

```bash
python -m pytest tests/test_metrics.py tests/test_runner_smoke.py tests/test_baselines_classical.py -v
```

These cover:

- Per-image PSNR/SSIM/LPIPS on synthetic images.
- Row aggregation (overall and per-category).
- Paired t-tests.
- Classical baselines (OpenCV, PIL) return RGB, same-size outputs.
- `EvaluationRunner` writes outputs, writes a correct per_image.csv,
  and is idempotent on re-run (resume works).
