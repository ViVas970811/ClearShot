# Phase 1 Summary - Data Pipeline Complete

## What Was Built

### 1. ABO Dataset Download (`data/scripts/download_abo.py`)
- Downloads ABO image metadata CSV (398,212 images)
- Downloads 10 listing metadata JSON files (92,320 listings)
- Merges into unified catalog: **91,950 images** with product type info
- Includes retry logic with exponential backoff
- Resumable downloads via skip-if-exists

**Artifacts:** `data/metadata/catalog.parquet` (91,950 rows)

### 2. Balanced Subset Curation (`data/scripts/create_subset.py`)
- **Key lesson learned:** The raw ABO dataset is 44% cellular phone cases. Initial keyword-matching mapping pulled "CASE" into "electronics" causing 69% electronics dominance.
- **Fix applied:** Explicit `product_type -> macro_category` lookup table, with per-category cap (default 800).
- Downloaded 5,178 images across 7 balanced categories.

**Final distribution:**
| Category    | Count | % |
|-------------|-------|-----|
| electronics | 800   | 15.4% |
| furniture   | 772   | 14.9% |
| jewelry     | 757   | 14.6% |
| outdoor     | 745   | 14.4% |
| apparel     | 741   | 14.3% |
| home_decor  | 732   | 14.1% |
| kitchen     | 631   | 12.2% |

**Artifacts:** `data/raw/{category}/*.jpg`, `data/raw/subset_manifest.csv`

### 3. Synthetic Degradation Pipeline (`src/training/degradation.py`)
Nine composable, randomized degradations simulating amateur photography:
- Color jitter (brightness/contrast/saturation/hue)
- Uneven exposure (directional lighting gradient)
- Vignette (radial darkening)
- Random shadow (elliptical overlay)
- Background clutter (colored noise patches)
- Gaussian noise
- Gaussian blur
- Downscale + upscale (low-res simulation)
- JPEG compression artifacts

Each applied with configurable probability and intensity range. Reproducible via seed.

**Artifacts:** 10,356 degraded images (2 variants × 5,178 clean), `data/pairs_manifest.csv`

### 4. Train/Val/Test Split
Split **by `image_id`** (not by pair) to prevent leakage:
- Train: 8,284 pairs (4,142 unique images, 80%)
- Val: 1,036 pairs (518 unique images, 10%)
- Test: 1,036 pairs (518 unique images, 10%)

**Artifacts:** `data/{train,val,test}_manifest.csv`

### 5. Comprehensive EDA (`notebooks/run_eda.py`)
Computed quality metrics and comparisons between clean and degraded sets.

## Key EDA Findings

### Degradation Impact
| Metric | Clean (median) | Degraded (median) | Change |
|---|---|---|---|
| brightness  | 212.63 | 148.15 | -30.3% |
| contrast    | 58.46  | 43.18  | -26.1% |
| sharpness   | 66.00  | 38.94  | **-41.0%** |
| colorfulness| 14.41  | 14.26  | -1.0% |

### Pair-wise PSNR / SSIM (500 sample)
- **Mean PSNR:** 13.84 dB (std: 6.74)
- **Median PSNR:** 12.02 dB
- **Mean SSIM:** 0.666 (std: 0.227)
- **PSNR 10-90th percentile:** 7.0 - 24.2 dB

### Findings That Inform Model Design

1. **Degradation pipeline is working well.** Sharpness drops 41% — the model has a real problem to solve.
2. **Good difficulty gradient.** PSNR spans 7-24 dB → training set has both easy and hard cases.
3. **Categories are now balanced.** No stratified batching needed; simple random batching is fine.
4. **Some pairs are very hard** (SSIM 10th percentile 0.279) → consider curriculum learning if training stalls.
5. **Brightness shifts are significant** (-64 points) → matches real amateur-photo use case.

## Visualizations Generated
All in `eda_results/`:
- `category_distribution.png` — balanced 7-way category split
- `resolution_distribution.png` — original image sizes before 512×512 resize
- `brightness_contrast_hist.png` — 4-metric clean vs degraded overlay
- `sharpness_distribution.png` — per-category boxplot (log scale)
- `sample_grid_clean.png` — 3×5 grid of clean samples per category
- `sample_grid_degraded.png` — 3×5 grid of degraded samples
- `pair_comparisons.png` — 6 side-by-side clean/degraded pairs with active degradations labelled
- `psnr_ssim_baseline.png` — PSNR/SSIM histograms + per-category PSNR boxplot

## Data Ready for Phase 2

The pipeline hands off to Phase 2 with:
- **5,178 clean reference images** (512×512, RGB, JPEG q=95)
- **10,356 degraded inputs** (512×512, RGB, JPEG q=85)
- **Pair manifests** with degradation parameter metadata
- **Reproducible split** (80/10/10 by image_id, seed=42)

## Reproduction

```bash
# Full pipeline from scratch (~5-10 minutes over reasonable bandwidth)
python data/scripts/download_abo.py --output_dir data/metadata
python data/scripts/create_subset.py --catalog data/metadata/catalog.parquet \
    --n_per_category 800 --clean_existing
python data/scripts/synthesize_degradations.py --input_dir data/raw \
    --n_variants 2 --max_workers 4
python notebooks/run_eda.py
```

## V1 (Imbalanced) vs V2 (Balanced) Comparison

The V1 iteration is preserved at `eda_results_v1_imbalanced/` for reference. Key changes:

| Aspect | V1 | V2 |
|---|---|---|
| Total images | 4,895 | 5,178 |
| Categories | 5 | 7 |
| Max category % | 69.0% (electronics/phone cases) | 15.4% (electronics) |
| Category mapping | Keyword matching | Explicit lookup |
| Per-category cap | None | 800 |

V2 is the production-ready dataset.
