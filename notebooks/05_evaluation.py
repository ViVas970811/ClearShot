# %%
"""
ClearShot Phase 5 - Evaluation Driver
=====================================

End-to-end driver for the Phase 5 evaluation described in the project plan:
    1. Load the test manifest and stratify-sample a subset by category.
    2. Build a clean-image reference directory for FID.
    3. Run five baselines:
         - OpenCV enhancement
         - PIL auto-enhance
         - Background-only
         - Stable Diffusion + ControlNet WITHOUT LoRA (ablation)
         - ClearShot (full pipeline, with LoRA)
    4. Compute PSNR / SSIM / LPIPS per image + FID per method.
    5. Produce report tables (overall, per-category, paired t-tests),
       visual comparison grids, and a failure-case grid.

Usage:
    - Colab: upload to a GPU runtime then Run All.
    - VSCode/Cursor: open as interactive Python (# %% cell markers).
    - CLI: ``python notebooks/05_evaluation.py``

Prerequisites:
    - ``data/test_manifest.csv`` + corresponding ``clean_path`` /
      ``degraded_path`` image files present (see Phase 2-3 handoff for the
      Google Drive ``clean_degraded.tar.gz`` restore step).
    - ``ClearShot_checkpoints/final/`` with the PEFT LoRA adapter
      (``adapter_config.json`` + ``adapter_model.safetensors``).
    - GPU recommended; diffusion baselines are not practical on CPU.
"""

# %%
import os
import sys
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project root: {PROJECT_ROOT}")

# %%
CONFIG_PATH = PROJECT_ROOT / "configs" / "evaluation_config.yaml"
with open(CONFIG_PATH, "r") as f:
    eval_cfg = yaml.safe_load(f)

LORA_WEIGHTS_PATH = os.environ.get(
    "LORA_WEIGHTS_PATH",
    eval_cfg["baselines"]["clearshot"]["lora_weights_path"],
)
if LORA_WEIGHTS_PATH and not Path(LORA_WEIGHTS_PATH).is_absolute():
    LORA_WEIGHTS_PATH = str(PROJECT_ROOT / LORA_WEIGHTS_PATH)

TEST_MANIFEST = PROJECT_ROOT / eval_cfg["data"]["test_manifest"]
REFERENCE_MANIFEST = eval_cfg["data"].get("reference_manifest")
if REFERENCE_MANIFEST:
    REFERENCE_MANIFEST = PROJECT_ROOT / REFERENCE_MANIFEST
OUTPUT_DIR = PROJECT_ROOT / eval_cfg["runner"]["output_dir"]

# Allow env-var overrides so the same driver can be used for quick CPU smoke
# runs and for full GPU runs without editing the YAML.
_subset_env = os.environ.get("EVAL_SUBSET_SIZE")
SUBSET_SIZE = int(_subset_env) if _subset_env else eval_cfg["data"]["subset_size"]
_output_env = os.environ.get("EVAL_OUTPUT_DIR")
if _output_env:
    OUTPUT_DIR = Path(_output_env)
    if not OUTPUT_DIR.is_absolute():
        OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR

print(f"Test manifest:       {TEST_MANIFEST}")
print(f"Reference manifest:  {REFERENCE_MANIFEST}")
print(f"LoRA weights:        {LORA_WEIGHTS_PATH}")
print(f"Output dir:          {OUTPUT_DIR}")
print(f"Subset size:         {SUBSET_SIZE}")

# %% [markdown]
# ## 1. Build the runner

# %%
from src.evaluation import (
    EvaluationConfig,
    EvaluationRunner,
    ImageQualityMetrics,
    build_baseline,
)

run_cfg = EvaluationConfig(
    manifest_path=str(TEST_MANIFEST),
    output_dir=str(OUTPUT_DIR),
    subset_size=SUBSET_SIZE,
    stratify_by_category=eval_cfg["data"]["stratify_by_category"],
    seed=eval_cfg["data"]["seed"],
    eval_resolution=eval_cfg["runner"]["eval_resolution"],
    image_format=eval_cfg["runner"]["image_format"],
    reference_manifest_path=str(REFERENCE_MANIFEST) if REFERENCE_MANIFEST else None,
    reference_dir_name=eval_cfg["runner"]["reference_dir_name"],
    input_column=eval_cfg["runner"]["input_column"],
    target_column=eval_cfg["runner"]["target_column"],
    image_id_column=eval_cfg["runner"]["image_id_column"],
    category_column=eval_cfg["runner"]["category_column"],
)

metrics_cfg = eval_cfg["metrics"]
metrics = ImageQualityMetrics(
    device=metrics_cfg.get("device"),
    lpips_net=metrics_cfg.get("lpips_net", "alex"),
    eval_resolution=run_cfg.eval_resolution,
)

runner = EvaluationRunner(config=run_cfg, metrics=metrics)

# %% [markdown]
# ## 2. Instantiate baselines (skip any whose assets are missing)

# %%
def _baseline_configs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the YAML baselines section into an ordered list of instantiation kwargs."""
    order = ["opencv", "pil_auto", "background_only", "sd_no_lora", "clearshot"]
    out = []
    for name in order:
        if name not in cfg:
            continue
        params = dict(cfg[name])
        if not params.pop("enabled", False):
            continue
        out.append({"name": name, "params": params})
    return out


baselines = []
for entry in _baseline_configs(eval_cfg["baselines"]):
    name = entry["name"]
    params = entry["params"]
    if name == "clearshot":
        if not LORA_WEIGHTS_PATH or not Path(LORA_WEIGHTS_PATH).exists():
            print(f"[skip] '{name}': LoRA weights not found at {LORA_WEIGHTS_PATH}")
            continue
        params["lora_weights_path"] = LORA_WEIGHTS_PATH
    try:
        b = build_baseline(name, **params)
        baselines.append(b)
        print(f"  + {name} ready")
    except Exception as e:
        print(f"[skip] '{name}': {e}")

print(f"\n{len(baselines)} baselines ready to run.")

# %% [markdown]
# ## 3. Enhance + score
#
# Outputs are resumable: re-running skips images that already have both a
# prediction file on disk and a row in ``per_image.csv``.

# %%
results = runner.run_all(baselines, build_reference=True)

# %% [markdown]
# ## 4. FID (distribution-level)

# %%
fid_by_method = runner.compute_fid_all(
    results,
    batch_size=metrics_cfg.get("fid_batch_size", 32),
    num_workers=metrics_cfg.get("fid_num_workers", 2),
    dims=metrics_cfg.get("fid_dims", 2048),
)

with open(OUTPUT_DIR / "fid.json", "w") as f:
    json.dump(fid_by_method, f, indent=2)

print("FID (lower is better):")
for m, v in fid_by_method.items():
    print(f"  {m:20s} {v:.3f}")

# %% [markdown]
# ## 5. Report tables + visual grids

# %%
import pandas as pd

from src.evaluation.analysis import (
    load_all_per_image,
    make_comparison_grid,
    save_report_tables,
    select_failure_cases,
)

long_df = load_all_per_image(OUTPUT_DIR, methods=[b.name for b in baselines])
print(f"Loaded {len(long_df)} per-image rows across {long_df['method'].nunique()} methods.")

written = save_report_tables(
    OUTPUT_DIR,
    long_df,
    fid_by_method=fid_by_method,
    reference_method=eval_cfg["reporting"]["reference_method"],
)
for k, p in written.items():
    print(f"  report: {k} -> {p}")

# %%
subset_manifest = pd.read_csv(OUTPUT_DIR / "_subset_manifest.csv")

grid_cfg = eval_cfg["reporting"]["comparison_grid"]
grid_methods = [m for m in grid_cfg["methods_in_order"] if m in {b.name for b in baselines}]

n_examples = grid_cfg["n_examples"]
example_ids = (
    subset_manifest.groupby("category")
    .head(max(1, n_examples // max(1, subset_manifest["category"].nunique())))
    .head(n_examples)["image_id"]
    .tolist()
)

grid_path = OUTPUT_DIR / "report" / "comparison_grid.png"
make_comparison_grid(
    image_ids=example_ids,
    evaluation_dir=OUTPUT_DIR,
    methods_in_order=grid_methods,
    subset_manifest=subset_manifest,
    output_path=grid_path,
    tile_size=grid_cfg["tile_size"],
    title="ClearShot vs Baselines (per-category sample)",
)
print(f"Comparison grid: {grid_path}")

# %% [markdown]
# ## 6. Failure case analysis
#
# Pick the worst-k LPIPS cases for the reference method and plot them against
# the other methods so we can reason about where ClearShot struggles.

# %%
ref_method = eval_cfg["reporting"]["reference_method"]
top_k = eval_cfg["reporting"]["failure_case_top_k"]

if ref_method in long_df["method"].unique():
    worst = select_failure_cases(long_df, method=ref_method, k=top_k, by="lpips", ascending=False)
    worst_ids = worst["image_id"].astype(str).tolist()
    print(f"Worst {top_k} LPIPS cases for '{ref_method}':")
    print(worst[["image_id", "category", "psnr", "ssim", "lpips"]].to_string(index=False))

    if worst_ids:
        failure_path = OUTPUT_DIR / "report" / "failure_cases.png"
        make_comparison_grid(
            image_ids=worst_ids,
            evaluation_dir=OUTPUT_DIR,
            methods_in_order=grid_methods,
            subset_manifest=subset_manifest,
            output_path=failure_path,
            tile_size=grid_cfg["tile_size"],
            title=f"Worst-{top_k} LPIPS cases for {ref_method}",
        )
        print(f"Failure cases grid: {failure_path}")
    else:
        print(f"[skip] Failure analysis: no valid '{ref_method}' rows to visualize.")
else:
    print(
        f"[skip] Failure analysis: reference method '{ref_method}' not present in results "
        f"(available: {sorted(long_df['method'].unique())})."
    )

# %% [markdown]
# ## 7. Summary

# %%
overall = pd.read_csv(OUTPUT_DIR / "report" / "summary_overall.csv", index_col=0)
print("\n=== Overall metric summary (mean) ===")
cols = [c for c in overall.columns if c.endswith("_mean") or c in {"n", "fid"}]
print(overall[cols].round(4).to_string())
print(f"\nAll artifacts under: {OUTPUT_DIR}")
