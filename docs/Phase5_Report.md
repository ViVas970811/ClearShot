---
title: "ClearShot"
subtitle: "Phase 5 Report: Evaluation"
author: "DATA612 Course Project | University of Maryland | Group 3"
date: "April 21, 2026"
geometry: margin=1in
fontsize: 11pt
colorlinks: true
---

\thispagestyle{empty}

\begin{center}
\vspace*{2cm}

{\LARGE \textbf{ClearShot}}\\[0.5cm]
{\Large AI-Powered Product Photo Enhancement for E-Commerce}\\[1.5cm]
{\Large \textbf{Phase 5 Report: Evaluation}}\\[2cm]

DATA612 Course Project\\
University of Maryland\\
Group 3\\[2cm]

April 21, 2026
\end{center}

\newpage

# Executive Summary

Phase 5 of the ClearShot project was defined as the evaluation phase for the end-to-end product photo enhancement pipeline. The planned scope was to compare the full ClearShot system against multiple baselines, compute paired image-quality metrics and a distributional metric, and generate report-ready artifacts that support analysis of model behavior and failure cases.

This phase has been implemented locally. The evaluation package, notebook driver, configuration, tests, and documentation are in place and were validated on the current repository state. Local verification confirmed that the Phase 5 code integrates with the existing ClearShot inference pipeline, resolves the expected manifests and checkpoint paths, and is ready to be executed in the intended GPU environment.

The full production-scale evaluation was not executed on this local machine because the intended workload depends on a GPU-capable runtime already maintained by a teammate who handled the Phase 4 execution workflow. Therefore, this report documents the implementation, local validation, and handoff readiness of Phase 5. It does not claim that the full benchmark run has already been completed on the final evaluation subset.

# Phase 5 Objectives

Based on the project plan and the existing ClearShot architecture, the objectives of Phase 5 were:

1. Build an evaluation module that computes image-quality metrics for restored outputs against clean ground truth.
2. Compare ClearShot against representative classical and ablation baselines.
3. Reuse the existing Phase 4 inference pipeline rather than duplicating diffusion logic.
4. Support manifest-driven, resumable evaluation on a stratified subset of the test split.
5. Produce report-ready artifacts including per-image CSVs, aggregate summaries, comparison grids, and failure-case analysis outputs.
6. Prepare the project for a full GPU execution that can be carried out consistently in the team’s established runtime environment.

# What Was Built

The following Phase 5 deliverables were implemented locally in the repository.

| Deliverable | Purpose |
|---|---|
| `src/evaluation/metrics.py` | Paired metric computation for PSNR, SSIM, and LPIPS; FID wrapper; aggregation helpers; paired t-test support |
| `src/evaluation/baselines.py` | Baseline definitions for OpenCV, PIL auto-enhance, background-only, SD without LoRA, and full ClearShot |
| `src/evaluation/runner.py` | Resumable manifest-driven evaluation runner that writes predictions and per-image CSVs |
| `src/evaluation/analysis.py` | Summary-table generation, per-category aggregation, comparison grids, and failure-case selection |
| `src/evaluation/__init__.py` | Public Phase 5 package surface |
| `configs/evaluation_config.yaml` | Central evaluation configuration for data paths, metrics, baselines, and reporting |
| `notebooks/05_evaluation.py` | End-to-end Phase 5 driver for instantiation, execution, FID, reporting, and summary printing |
| `tests/test_metrics.py` | Unit tests for metric computation, aggregation, and paired t-tests |
| `tests/test_baselines_classical.py` | Smoke tests for classical baselines and baseline factory behavior |
| `tests/test_runner_smoke.py` | Smoke tests for runner output writing and resume behavior |
| `docs/PHASE5_EVALUATION.md` | Practical runbook for prerequisites, commands, outputs, and tests |

In addition, the project documentation was updated to reference the evaluation workflow and expected artifacts under `evaluation_results/`.

# Design and Integration Details

Phase 5 was designed to fit the existing ClearShot project structure without changing the completed Phase 1 to 4 code paths.

## Reuse of Existing Pipeline

The diffusion-based Phase 5 baselines reuse the established `ClearShotPipeline` from the inference stack rather than re-implementing diffusion, ControlNet, or LoRA loading separately. This keeps the evaluation path consistent with the same pipeline logic already used for enhancement and Phase 4 inference.

## Baseline Structure

Five evaluation methods are supported:

| Method | Role in Evaluation |
|---|---|
| `opencv` | Classical enhancement baseline using CLAHE, bilateral filtering, and unsharp masking |
| `pil_auto` | Lightweight PIL-based enhancement baseline |
| `background_only` | Segmentation and clean-background composition without diffusion |
| `sd_no_lora` | Ablation baseline using the ClearShot diffusion path without the trained LoRA adapter |
| `clearshot` | Full ClearShot evaluation baseline with the trained PEFT LoRA adapter |

## Asset and Path Handling

Phase 5 expects the following existing project assets:

- `data/test_manifest.csv`
- `data/val_manifest.csv`
- `data/clean/`
- `data/degraded/`
- `ClearShot_checkpoints/final/adapter_config.json`
- `ClearShot_checkpoints/final/adapter_model.safetensors`
- `configs/inference_config.yaml`

The evaluation notebook reads `configs/evaluation_config.yaml`, resolves paths relative to the project root, supports `LORA_WEIGHTS_PATH` as an environment override, and writes outputs under `evaluation_results/` by default.

## Evaluation Flow

The intended execution flow is:

1. Load the test manifest.
2. Select a stratified evaluation subset.
3. Build a clean reference directory for FID.
4. Run each enabled baseline and save per-image predictions.
5. Compute per-image metrics against the clean target.
6. Compute FID for each baseline.
7. Generate summary tables, comparison grids, and failure-case artifacts.

This design supports resumable evaluation by skipping any image that already has both a prediction file and a recorded CSV row.

# Local Validation and Readiness Check

Local validation was intentionally limited to lightweight checks appropriate for a non-GPU development machine.

## Confirmed Local Checks

The following facts were verified locally:

- The Phase 5 implementation files exist and import correctly.
- The evaluation configuration resolves the expected manifests and checkpoint locations.
- The mounted assets required for Phase 5 are present in the repository state.
- The five baselines instantiate from the current configuration.
- The targeted Phase 5 test suite passed locally: `13` tests passed.
- The evaluation package is logically complete and consistent with the ClearShot project structure.
- Phase 5 is ready to be handed off for full GPU execution.

## Minimal Phase 5-Side Corrections Applied

During readiness verification, small Phase 5-only improvements were made:

- the notebook driver was hardened so failure-case visualization does not crash when the reference method has no valid rows;
- the configured FID runtime settings are now forwarded from the YAML configuration into the runner;
- a local compatibility workaround was placed inside the Phase 5 evaluation path for a Python 3.12 `rembg`/`pymatting` cache issue observed on this Mac.

These changes remain scoped to Phase 5 files only.

## Partial Local Output Artifacts

The repository currently contains partial local artifacts under `evaluation_results/` from an incomplete CPU-side smoke attempt. These files are useful as implementation evidence, but they must not be treated as final project results.

Confirmed local artifacts include:

- `_subset_manifest.csv`
- `opencv/per_image.csv`
- `pil_auto/per_image.csv`
- `sd_no_lora/per_image.csv`
- `clearshot/per_image.csv`
- `report/summary_overall.csv`
- `report/summary_per_category__psnr.csv`
- `report/summary_per_category__ssim.csv`
- `report/summary_per_category__lpips.csv`
- `report/comparison_grid.png`
- `fid.json`

The partial run reflects a local CPU environment and includes failed diffusion rows and `NaN` FID values. Accordingly, these artifacts should be described only as partial validation evidence and not as the final Phase 5 benchmark.

# Environment and Runtime Notes

## Required Assets

The expected runtime structure is:

```text
ClearShot_checkpoints/final/adapter_config.json
ClearShot_checkpoints/final/adapter_model.safetensors
data/test_manifest.csv
data/val_manifest.csv
data/clean/...
data/degraded/...
```

## Expected Command

The full evaluation should be launched from the repository root with:

```bash
LORA_WEIGHTS_PATH=ClearShot_checkpoints/final python notebooks/05_evaluation.py
```

## Practical Notes

- A GPU runtime is strongly recommended for `sd_no_lora` and `clearshot`.
- The first FID run may need to download the Inception weights used by `pytorch-fid`.
- On this local Mac, Python 3.12 exposed a `rembg`/`pymatting` caching issue during smoke execution; the current Phase 5 implementation contains a local evaluation-side workaround for that case.
- The evaluation is resumable, so interrupted runs can continue from existing prediction files and CSV rows.

# Current Limitations and Pending Full Execution

This section is intentionally explicit.

The full Phase 5 evaluation has **not** been completed on the intended GPU environment from this local machine. The current machine was used for implementation, local testing, asset verification, import checks, configuration checks, baseline instantiation, and readiness review. It was not used to perform the final production-scale benchmark run across the intended evaluation workload.

Therefore:

- final large-scale quantitative metrics are still pending the teammate-run GPU execution;
- any current local `evaluation_results/` outputs should be treated as partial smoke artifacts only;
- this report documents implementation status and readiness for execution, not final claimed benchmark performance.

This distinction is important for correctness and for maintaining consistency with the actual team workflow.

# Handoff Instructions

The teammate responsible for the GPU runtime should execute Phase 5 in the established environment already used for Phase 4 inference.

## Required Files

- `data/test_manifest.csv`
- `data/val_manifest.csv`
- `data/clean/`
- `data/degraded/`
- `ClearShot_checkpoints/final/adapter_config.json`
- `ClearShot_checkpoints/final/adapter_model.safetensors`

## Command

```bash
LORA_WEIGHTS_PATH=ClearShot_checkpoints/final python notebooks/05_evaluation.py
```

## Expected Output Directory

```text
evaluation_results/
```

Expected contents include per-baseline prediction directories, per-image CSVs, `fid.json`, summary tables, comparison grids, and failure-case outputs.

## Operational Note

This execution should be performed in the teammate’s GPU-enabled working setup, not on the current local Mac development environment.

# Conclusion

Phase 5 of ClearShot has been implemented locally, kept cleanly scoped to the evaluation layer, validated through lightweight checks, and prepared for full execution in the correct GPU environment. The code, configuration, tests, and documentation are in place, and the evaluation flow integrates with the existing ClearShot pipeline without altering Phases 1 through 4.

At the current stage, the correct project statement is that Phase 5 is **implemented and ready for GPU execution**, while the final large-scale evaluation results remain **pending teammate execution in the proper runtime environment**. This is the appropriate handoff point before the project proceeds to Phase 6.
