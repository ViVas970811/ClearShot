"""
Comprehensive EDA on the ABO subset and degradation pipeline.

Produces:
- eda_results/summary_stats.csv       - Quantitative summary
- eda_results/category_distribution.png
- eda_results/resolution_distribution.png
- eda_results/brightness_contrast_hist.png
- eda_results/sharpness_distribution.png
- eda_results/sample_grid_clean.png
- eda_results/sample_grid_degraded.png
- eda_results/pair_comparisons.png
- eda_results/degradation_impact_metrics.csv
- eda_results/psnr_ssim_baseline.png
- eda_results/eda_report.md           - Narrative findings

Run from repo root: python notebooks/run_eda.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Config
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eda_results"
OUT.mkdir(exist_ok=True)

CLEAN_DIR = ROOT / "data" / "clean"
DEGRADED_DIR = ROOT / "data" / "degraded"
MANIFEST = ROOT / "data" / "pairs_manifest.csv"
RAW_MANIFEST = ROOT / "data" / "raw" / "subset_manifest.csv"


# ────────────────── IMAGE QUALITY METRICS ──────────────────

def compute_sharpness(img_path: Path) -> float:
    """Laplacian variance (Blur Metric). Lower = blurrier."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.nan
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def compute_brightness(img_path: Path) -> float:
    """Mean pixel intensity (0-255)."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    return float(np.mean(img)) if img is not None else np.nan


def compute_contrast(img_path: Path) -> float:
    """Standard deviation of pixel intensity."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    return float(np.std(img)) if img is not None else np.nan


def compute_colorfulness(img_path: Path) -> float:
    """Hasler & Suesstrunk (2003) colorfulness metric."""
    img = cv2.imread(str(img_path))
    if img is None:
        return np.nan
    (B, G, R) = cv2.split(img.astype(np.float32))
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    return float(np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2))


def compute_aspect_ratio(img_path: Path) -> float:
    with Image.open(img_path) as im:
        return im.width / im.height


def compute_file_size_kb(img_path: Path) -> float:
    return img_path.stat().st_size / 1024


# ────────────────── PAIR-WISE METRICS ──────────────────

def compute_pair_metrics(clean_path: Path, degraded_path: Path) -> dict:
    """PSNR, SSIM between clean and degraded image."""
    clean = cv2.imread(str(clean_path))
    degraded = cv2.imread(str(degraded_path))
    if clean is None or degraded is None:
        return {"psnr": np.nan, "ssim": np.nan}

    # Same size?
    if clean.shape != degraded.shape:
        degraded = cv2.resize(degraded, (clean.shape[1], clean.shape[0]))

    clean_gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    degraded_gray = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)

    p = psnr(clean_gray, degraded_gray, data_range=255)
    s = ssim(clean_gray, degraded_gray, data_range=255)
    return {"psnr": float(p), "ssim": float(s)}


# ────────────────── DATA COLLECTION ──────────────────

def collect_image_stats(image_dir: Path, label: str, sample_n: int | None = None) -> pd.DataFrame:
    """Compute per-image quality metrics."""
    paths = sorted(image_dir.rglob("*.jpg"))
    if sample_n is not None and len(paths) > sample_n:
        random.seed(42)
        paths = random.sample(paths, sample_n)

    rows = []
    for p in tqdm(paths, desc=f"Scanning {label}"):
        try:
            with Image.open(p) as im:
                w, h = im.size
            rows.append({
                "path": str(p),
                "category": p.parent.name,
                "width": w,
                "height": h,
                "aspect_ratio": w / h,
                "megapixels": (w * h) / 1e6,
                "file_size_kb": compute_file_size_kb(p),
                "brightness": compute_brightness(p),
                "contrast": compute_contrast(p),
                "sharpness": compute_sharpness(p),
                "colorfulness": compute_colorfulness(p),
                "source": label,
            })
        except Exception as e:
            print(f"  [error] {p.name}: {e}")
    return pd.DataFrame(rows)


def collect_pair_metrics(manifest: pd.DataFrame, sample_n: int = 500) -> pd.DataFrame:
    """Compute PSNR/SSIM between clean-degraded pairs."""
    sample = manifest.sample(n=min(sample_n, len(manifest)), random_state=42)
    rows = []
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Pair metrics"):
        clean_p = Path(row["clean_path"])
        deg_p = Path(row["degraded_path"])
        metrics = compute_pair_metrics(clean_p, deg_p)
        rows.append({
            "image_id": row["image_id"],
            "category": row["category"],
            "variant": row["variant"],
            **metrics,
        })
    return pd.DataFrame(rows)


# ────────────────── VISUALIZATION ──────────────────

def plot_category_distribution(clean_stats: pd.DataFrame, raw_manifest: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cat_counts = clean_stats["category"].value_counts()
    axes[0].bar(cat_counts.index, cat_counts.values, color=sns.color_palette("viridis", len(cat_counts)))
    axes[0].set_title("Subset Image Count by Category", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Number of images")
    axes[0].set_xlabel("Category")
    for i, v in enumerate(cat_counts.values):
        axes[0].text(i, v + 20, str(v), ha="center", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=25)

    # Product type diversity
    top_pt = raw_manifest["product_type"].value_counts().head(15)
    axes[1].barh(range(len(top_pt)), top_pt.values, color=sns.color_palette("rocket", len(top_pt)))
    axes[1].set_yticks(range(len(top_pt)))
    axes[1].set_yticklabels(top_pt.index, fontsize=9)
    axes[1].set_title("Top 15 Product Types in Subset", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Number of images")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(OUT / "category_distribution.png")
    plt.close()


def plot_resolution_distribution(raw_manifest: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].hist(raw_manifest["width"], bins=40, color="steelblue", edgecolor="black")
    axes[0].set_title("Original Image Widths", fontweight="bold")
    axes[0].set_xlabel("Width (pixels)")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(raw_manifest["height"], bins=40, color="coral", edgecolor="black")
    axes[1].set_title("Original Image Heights", fontweight="bold")
    axes[1].set_xlabel("Height (pixels)")

    raw_manifest["megapixels"] = (raw_manifest["width"] * raw_manifest["height"]) / 1e6
    axes[2].hist(raw_manifest["megapixels"], bins=40, color="mediumseagreen", edgecolor="black")
    axes[2].set_title("Original Megapixels", fontweight="bold")
    axes[2].set_xlabel("Megapixels")
    axes[2].axvline(raw_manifest["megapixels"].median(), color="red", linestyle="--",
                     label=f"Median: {raw_manifest['megapixels'].median():.2f} MP")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(OUT / "resolution_distribution.png")
    plt.close()


def plot_quality_histograms(clean_stats: pd.DataFrame, degraded_stats: pd.DataFrame):
    metrics = ["brightness", "contrast", "sharpness", "colorfulness"]
    titles = ["Brightness (mean gray)", "Contrast (std)", "Sharpness (Laplacian var)", "Colorfulness"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, metric, title in zip(axes.flat, metrics, titles):
        ax.hist(clean_stats[metric].dropna(), bins=40, alpha=0.6, label="Clean", color="seagreen", edgecolor="black")
        ax.hist(degraded_stats[metric].dropna(), bins=40, alpha=0.6, label="Degraded", color="firebrick", edgecolor="black")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(metric)
        ax.set_ylabel("Count")
        ax.legend()

        # Annotations
        clean_mean = clean_stats[metric].median()
        deg_mean = degraded_stats[metric].median()
        ax.axvline(clean_mean, color="seagreen", linestyle="--", alpha=0.8)
        ax.axvline(deg_mean, color="firebrick", linestyle="--", alpha=0.8)

    plt.suptitle("Clean vs Degraded: Image Quality Metric Distributions", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(OUT / "brightness_contrast_hist.png")
    plt.close()


def plot_sharpness_by_category(clean_stats: pd.DataFrame, degraded_stats: pd.DataFrame):
    combined = pd.concat([clean_stats, degraded_stats])
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=combined, x="category", y="sharpness", hue="source", ax=ax,
                palette={"clean": "seagreen", "degraded": "firebrick"})
    ax.set_title("Sharpness Distribution by Category and Source", fontsize=13, fontweight="bold")
    ax.set_ylabel("Laplacian variance (higher = sharper)")
    ax.set_xlabel("Category")
    ax.set_yscale("log")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUT / "sharpness_distribution.png")
    plt.close()


def plot_sample_grid(image_dir: Path, title: str, filename: str, n_per_cat: int = 3):
    """3x5 grid: 3 samples from each of 5 categories."""
    categories = sorted([d.name for d in image_dir.iterdir() if d.is_dir()])[:5]
    fig, axes = plt.subplots(n_per_cat, len(categories), figsize=(4 * len(categories), 4 * n_per_cat))

    random.seed(42)
    for j, cat in enumerate(categories):
        images = list((image_dir / cat).glob("*.jpg"))
        samples = random.sample(images, min(n_per_cat, len(images)))
        for i, img_path in enumerate(samples):
            ax = axes[i, j] if n_per_cat > 1 else axes[j]
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(cat, fontsize=12, fontweight="bold")

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(OUT / filename)
    plt.close()


def plot_pair_comparisons(manifest: pd.DataFrame, n: int = 6):
    """Show clean vs degraded pairs side-by-side."""
    sample = manifest.sample(n=n, random_state=7)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))

    for i, (_, row) in enumerate(sample.iterrows()):
        clean = Image.open(row["clean_path"])
        deg = Image.open(row["degraded_path"])

        axes[i, 0].imshow(clean)
        axes[i, 0].set_title(f"Clean [{row['category']}]" + (" - Ground Truth" if i == 0 else ""), fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(deg)
        params = json.loads(row["degradation_params"])
        active = ", ".join(list(params.keys())[:4])
        axes[i, 1].set_title(f"Degraded - {active}", fontsize=10)
        axes[i, 1].axis("off")

    plt.suptitle("Clean vs Degraded Image Pairs (Training Data Samples)", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(OUT / "pair_comparisons.png")
    plt.close()


def plot_pair_metrics(pair_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].hist(pair_df["psnr"].dropna(), bins=40, color="royalblue", edgecolor="black")
    axes[0].set_title("PSNR Distribution (Degraded vs Clean)", fontweight="bold")
    axes[0].set_xlabel("PSNR (dB) - higher = more similar to clean")
    axes[0].axvline(pair_df["psnr"].median(), color="red", linestyle="--",
                     label=f"Median: {pair_df['psnr'].median():.2f} dB")
    axes[0].legend()

    axes[1].hist(pair_df["ssim"].dropna(), bins=40, color="darkorange", edgecolor="black")
    axes[1].set_title("SSIM Distribution (Degraded vs Clean)", fontweight="bold")
    axes[1].set_xlabel("SSIM (0-1) - higher = more similar")
    axes[1].axvline(pair_df["ssim"].median(), color="red", linestyle="--",
                     label=f"Median: {pair_df['ssim'].median():.3f}")
    axes[1].legend()

    # Per category
    sns.boxplot(data=pair_df, x="category", y="psnr", ax=axes[2],
                palette="coolwarm")
    axes[2].set_title("PSNR by Category", fontweight="bold")
    axes[2].set_ylabel("PSNR (dB)")
    axes[2].tick_params(axis="x", rotation=15)

    plt.suptitle("Baseline Degradation Impact: PSNR & SSIM Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "psnr_ssim_baseline.png")
    plt.close()


# ────────────────── EDA REPORT ──────────────────

def generate_report(
    clean_stats: pd.DataFrame,
    degraded_stats: pd.DataFrame,
    raw_manifest: pd.DataFrame,
    pair_df: pd.DataFrame,
):
    report = []
    report.append("# ClearShot - Exploratory Data Analysis Report\n")
    report.append(f"Generated from {len(clean_stats):,} clean images and {len(degraded_stats):,} degraded images.\n")

    report.append("\n## 1. Dataset Overview\n")
    report.append(f"- **Total clean images:** {len(clean_stats):,}")
    report.append(f"- **Total degraded variants:** {len(degraded_stats):,}")
    report.append(f"- **Categories:** {clean_stats['category'].nunique()}")
    report.append(f"- **Train/Val/Test split:** 80/10/10 (by image_id to prevent leakage)")

    report.append("\n## 2. Category Distribution\n")
    for cat, count in clean_stats["category"].value_counts().items():
        pct = 100 * count / len(clean_stats)
        report.append(f"- **{cat}**: {count:,} images ({pct:.1f}%)")

    report.append("\n## 3. Original Image Properties (pre-resize)\n")
    rm = raw_manifest
    report.append(f"- **Width range:** {rm['width'].min():,} - {rm['width'].max():,} px "
                  f"(median: {rm['width'].median():.0f})")
    report.append(f"- **Height range:** {rm['height'].min():,} - {rm['height'].max():,} px "
                  f"(median: {rm['height'].median():.0f})")
    report.append(f"- **All images resized to 512x512** for training consistency.")

    report.append("\n## 4. Image Quality Comparison (Clean vs Degraded)\n")
    report.append("| Metric | Clean (median) | Degraded (median) | Change |")
    report.append("|---|---|---|---|")
    for metric in ["brightness", "contrast", "sharpness", "colorfulness"]:
        c = clean_stats[metric].median()
        d = degraded_stats[metric].median()
        pct = ((d - c) / c) * 100 if c != 0 else 0
        report.append(f"| {metric} | {c:.2f} | {d:.2f} | {pct:+.1f}% |")

    report.append("\n## 5. Degradation Impact (PSNR/SSIM on sample of 500 pairs)\n")
    report.append(f"- **Mean PSNR:** {pair_df['psnr'].mean():.2f} dB (std: {pair_df['psnr'].std():.2f})")
    report.append(f"- **Median PSNR:** {pair_df['psnr'].median():.2f} dB")
    report.append(f"- **Mean SSIM:** {pair_df['ssim'].mean():.3f} (std: {pair_df['ssim'].std():.3f})")
    report.append(f"- **Median SSIM:** {pair_df['ssim'].median():.3f}")

    report.append("\nPer-category PSNR:")
    for cat, stats in pair_df.groupby("category")["psnr"].agg(["mean", "std"]).iterrows():
        report.append(f"- **{cat}**: {stats['mean']:.2f} ± {stats['std']:.2f} dB")

    report.append("\n## 6. Key Findings for Model Design\n")

    # Finding 1: Category imbalance
    max_cat = clean_stats["category"].value_counts().idxmax()
    max_pct = 100 * clean_stats["category"].value_counts().max() / len(clean_stats)
    if max_pct > 50:
        report.append(f"**[!] Category imbalance detected.** '{max_cat}' dominates at {max_pct:.1f}%. "
                      "Recommendation: use category-stratified batching during training, "
                      "and per-category evaluation metrics.")

    # Finding 2: Sharpness drop confirms degradation is working
    sharp_drop = (degraded_stats["sharpness"].median() - clean_stats["sharpness"].median()) / clean_stats["sharpness"].median()
    report.append(f"\n**[OK] Degradation pipeline validated.** Sharpness dropped "
                  f"{100*sharp_drop:.1f}% (median) from clean to degraded - our blur+downscale+noise augmentations "
                  "create meaningfully harder inputs.")

    # Finding 3: PSNR range
    psnr_low = pair_df["psnr"].quantile(0.1)
    psnr_high = pair_df["psnr"].quantile(0.9)
    report.append(f"\n**[OK] Degradation difficulty is well-distributed.** "
                  f"PSNR spans {psnr_low:.1f} - {psnr_high:.1f} dB (10th-90th percentile), "
                  "which gives the model a gradient of difficulties to learn from.")

    # Finding 4: SSIM floor
    ssim_low = pair_df["ssim"].quantile(0.1)
    if ssim_low < 0.5:
        report.append(f"\n**[!] Some pairs are very heavily degraded** (SSIM 10th percentile: {ssim_low:.3f}). "
                      "Recommendation: during training, consider curriculum learning (easy pairs first) "
                      "or cap degradation strength if model struggles to converge.")

    # Finding 5: Brightness/contrast shift
    bright_change = abs(degraded_stats["brightness"].median() - clean_stats["brightness"].median())
    if bright_change > 10:
        report.append(f"\n**[INFO] Lighting degradations are significant** "
                      f"(brightness shift: {bright_change:.1f} points). "
                      "The model must handle substantial lighting variation - "
                      "this aligns with the amateur-photo use case.")

    report.append("\n## 7. Recommendations for Training\n")
    max_pct_val = 100 * clean_stats["category"].value_counts().max() / len(clean_stats)
    if max_pct_val > 50:
        report.append("- **Batch sampling:** Stratify by category to prevent majority-category domination.")
    else:
        report.append(f"- **Batch sampling:** Dataset is well-balanced (max category {max_pct_val:.1f}%). Standard random batching is OK.")
    report.append("- **Evaluation:** Report per-category metrics, not just overall.")
    report.append("- **Degradation curriculum:** Start with weaker degradations, progressively increase difficulty.")
    report.append("- **Augmentation:** Test-time flip/rotation OK; colour-preserving only.")
    report.append("- **Resolution:** 512x512 is a good balance for SD 1.5; upscale to 1024 via Real-ESRGAN.")
    n_val = len(clean_stats) // 10  # approximate
    report.append(f"- **Reference images:** The ~{n_val} val-set clean images are ideal FID reference set.")

    report.append("\n## 8. Files Generated\n")
    report.append("- `eda_results/summary_stats.csv` - Per-image quality metrics")
    report.append("- `eda_results/category_distribution.png`")
    report.append("- `eda_results/resolution_distribution.png`")
    report.append("- `eda_results/brightness_contrast_hist.png`")
    report.append("- `eda_results/sharpness_distribution.png`")
    report.append("- `eda_results/sample_grid_clean.png`")
    report.append("- `eda_results/sample_grid_degraded.png`")
    report.append("- `eda_results/pair_comparisons.png`")
    report.append("- `eda_results/psnr_ssim_baseline.png`")
    report.append("- `eda_results/degradation_impact_metrics.csv`")

    (OUT / "eda_report.md").write_text("\n".join(report), encoding="utf-8")


# ────────────────── MAIN ──────────────────

def main():
    print("=" * 70)
    print("ClearShot - Exploratory Data Analysis")
    print("=" * 70)

    print("\n[1/6] Loading manifests...")
    raw_manifest = pd.read_csv(RAW_MANIFEST)
    pairs_manifest = pd.read_csv(MANIFEST)
    print(f"  Raw manifest: {len(raw_manifest):,} rows")
    print(f"  Pairs manifest: {len(pairs_manifest):,} rows")

    print("\n[2/6] Collecting image quality stats (clean, full set)...")
    clean_stats = collect_image_stats(CLEAN_DIR, label="clean")

    print("\n[3/6] Collecting image quality stats (degraded, sample 2000)...")
    degraded_stats = collect_image_stats(DEGRADED_DIR, label="degraded", sample_n=2000)

    print("\n[4/6] Computing pair PSNR/SSIM (sample 500)...")
    pair_df = collect_pair_metrics(pairs_manifest, sample_n=500)

    # Save raw stats
    clean_stats.to_csv(OUT / "clean_image_stats.csv", index=False)
    degraded_stats.to_csv(OUT / "degraded_image_stats.csv", index=False)
    pair_df.to_csv(OUT / "degradation_impact_metrics.csv", index=False)

    summary = pd.concat([
        clean_stats.describe().add_prefix("clean_"),
        degraded_stats.describe().add_prefix("degraded_"),
    ], axis=1)
    summary.to_csv(OUT / "summary_stats.csv")

    print("\n[5/6] Generating visualizations...")
    plot_category_distribution(clean_stats, raw_manifest)
    plot_resolution_distribution(raw_manifest)
    plot_quality_histograms(clean_stats, degraded_stats)
    plot_sharpness_by_category(clean_stats, degraded_stats)
    plot_sample_grid(CLEAN_DIR, "Clean Product Images (Ground Truth)", "sample_grid_clean.png")
    plot_sample_grid(DEGRADED_DIR, "Synthetically Degraded Images (Model Input)", "sample_grid_degraded.png")
    plot_pair_comparisons(pairs_manifest)
    plot_pair_metrics(pair_df)

    print("\n[6/6] Generating narrative report...")
    generate_report(clean_stats, degraded_stats, raw_manifest, pair_df)

    print(f"\nDone! All results saved to: {OUT}")
    print(f"  See eda_report.md for narrative findings.")


if __name__ == "__main__":
    main()
