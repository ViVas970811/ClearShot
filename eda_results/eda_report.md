# ClearShot - Exploratory Data Analysis Report

Generated from 24,131 clean images and 2,000 degraded images.


## 1. Dataset Overview

- **Total clean images:** 24,131
- **Total degraded variants:** 2,000
- **Categories:** 7
- **Train/Val/Test split:** 80/10/10 (by image_id to prevent leakage)

## 2. Category Distribution

- **apparel**: 4,700 images (19.5%)
- **electronics**: 4,700 images (19.5%)
- **furniture**: 4,700 images (19.5%)
- **home_decor**: 4,700 images (19.5%)
- **jewelry**: 2,804 images (11.6%)
- **outdoor**: 1,901 images (7.9%)
- **kitchen**: 626 images (2.6%)

## 3. Original Image Properties (pre-resize)

- **Width range:** 258 - 2,560 px (median: 1879)
- **Height range:** 259 - 2,560 px (median: 2097)
- **All images resized to 512x512** for training consistency.

## 4. Image Quality Comparison (Clean vs Degraded)

| Metric | Clean (median) | Degraded (median) | Change |
|---|---|---|---|
| brightness | 207.89 | 146.31 | -29.6% |
| contrast | 62.44 | 44.50 | -28.7% |
| sharpness | 66.83 | 36.78 | -45.0% |
| colorfulness | 16.16 | 14.52 | -10.2% |

## 5. Degradation Impact (PSNR/SSIM on sample of 500 pairs)

- **Mean PSNR:** 13.20 dB (std: 5.74)
- **Median PSNR:** 11.80 dB
- **Mean SSIM:** 0.678 (std: 0.217)
- **Median SSIM:** 0.737

Per-category PSNR:
- **apparel**: 14.28 ± 6.00 dB
- **electronics**: 12.50 ± 5.01 dB
- **furniture**: 13.56 ± 6.31 dB
- **home_decor**: 13.85 ± 5.48 dB
- **jewelry**: 12.04 ± 5.53 dB
- **kitchen**: 11.32 ± 4.11 dB
- **outdoor**: 12.32 ± 6.17 dB

## 6. Key Findings for Model Design


**[OK] Degradation pipeline validated.** Sharpness dropped -45.0% (median) from clean to degraded - our blur+downscale+noise augmentations create meaningfully harder inputs.

**[OK] Degradation difficulty is well-distributed.** PSNR spans 7.1 - 21.3 dB (10th-90th percentile), which gives the model a gradient of difficulties to learn from.

**[!] Some pairs are very heavily degraded** (SSIM 10th percentile: 0.328). Recommendation: during training, consider curriculum learning (easy pairs first) or cap degradation strength if model struggles to converge.

**[INFO] Lighting degradations are significant** (brightness shift: 61.6 points). The model must handle substantial lighting variation - this aligns with the amateur-photo use case.

## 7. Recommendations for Training

- **Batch sampling:** Dataset is well-balanced (max category 19.5%). Standard random batching is OK.
- **Evaluation:** Report per-category metrics, not just overall.
- **Degradation curriculum:** Start with weaker degradations, progressively increase difficulty.
- **Augmentation:** Test-time flip/rotation OK; colour-preserving only.
- **Resolution:** 512x512 is a good balance for SD 1.5; upscale to 1024 via Real-ESRGAN.
- **Reference images:** The ~2413 val-set clean images are ideal FID reference set.

## 8. Files Generated

- `eda_results/summary_stats.csv` - Per-image quality metrics
- `eda_results/category_distribution.png`
- `eda_results/resolution_distribution.png`
- `eda_results/brightness_contrast_hist.png`
- `eda_results/sharpness_distribution.png`
- `eda_results/sample_grid_clean.png`
- `eda_results/sample_grid_degraded.png`
- `eda_results/pair_comparisons.png`
- `eda_results/psnr_ssim_baseline.png`
- `eda_results/degradation_impact_metrics.csv`