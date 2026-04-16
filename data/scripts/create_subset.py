"""
Curate a BALANCED subset of ~5,000 images from the ABO catalog.

Strategy:
1. Use explicit product_type -> macro_category mapping (no keyword matching)
2. Cap each category at a fixed MAX (default 800) to prevent domination
3. Filter for minimum resolution (256x256)
4. Download images in parallel
5. Save manifest CSV

Categories are chosen to reflect realistic e-commerce diversity:
  - apparel: shoes, boots, sandals, handbags, hats
  - furniture: chair, sofa, table, office
  - home_decor: rugs, wall_art, lighting, home_bed_and_bath
  - kitchen: kitchen items, drinking cups
  - electronics: phone cases (capped), office products
  - jewelry: rings, earrings, necklaces
  - outdoor: sporting_goods, outdoor_living, pet_supplies

Usage:
    python data/scripts/create_subset.py --catalog data/metadata/catalog.parquet \\
        --n_per_category 800
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# Explicit mapping of product_type -> macro_category
# Based on actual ABO product_type distribution (top types only)
PRODUCT_TYPE_MAP = {
    # Apparel
    "SHOES": "apparel",
    "BOOT": "apparel",
    "SANDAL": "apparel",
    "HANDBAG": "apparel",
    "HAT": "apparel",
    "ACCESSORY": "apparel",
    "WATCH": "apparel",
    "GLOVES": "apparel",
    "BELT": "apparel",
    "SUNGLASSES": "apparel",

    # Furniture
    "CHAIR": "furniture",
    "SOFA": "furniture",
    "TABLE": "furniture",
    "OFFICE_PRODUCTS": "furniture",
    "HOME_FURNITURE_AND_DECOR": "furniture",
    "BED": "furniture",
    "SHELF": "furniture",
    "OTTOMAN": "furniture",
    "STOOL_SEATING": "furniture",
    "DESK": "furniture",

    # Home Decor
    "RUG": "home_decor",
    "WALL_ART": "home_decor",
    "LIGHT_BULB": "home_decor",
    "LIGHT_FIXTURE": "home_decor",
    "HOME_BED_AND_BATH": "home_decor",
    "HOME": "home_decor",
    "LAMP": "home_decor",
    "MIRROR": "home_decor",
    "VASE": "home_decor",
    "CLOCK": "home_decor",
    "CURTAIN": "home_decor",
    "PILLOW": "home_decor",
    "DECORATION": "home_decor",

    # Kitchen
    "KITCHEN": "kitchen",
    "DRINKING_CUP": "kitchen",
    "DRINKING_CONTAINER": "kitchen",
    "COOKWARE": "kitchen",
    "DINNERWARE": "kitchen",
    "BAKEWARE": "kitchen",
    "FOOD_STORAGE_CONTAINER": "kitchen",

    # Electronics (capped heavily)
    "CELLULAR_PHONE_CASE": "electronics",
    "COMPUTER_COMPONENT": "electronics",
    "HEADPHONE": "electronics",
    "SPEAKERS": "electronics",
    "ELECTRONIC_CABLE": "electronics",
    "AUDIO_OR_VIDEO": "electronics",
    "COMPUTER_SPEAKER": "electronics",

    # Jewelry
    "FINERING": "jewelry",
    "FINEEARRING": "jewelry",
    "FINENECKLACEBRACELETANKLET": "jewelry",
    "EARRING": "jewelry",
    "NECKLACE": "jewelry",
    "BRACELET": "jewelry",

    # Outdoor & Other
    "SPORTING_GOODS": "outdoor",
    "OUTDOOR_LIVING": "outdoor",
    "PET_SUPPLIES": "outdoor",
    "TOYS_AND_GAMES": "outdoor",
    "HARDWARE_HANDLE": "outdoor",
}

# Categories to include (ordered by priority if we need to drop)
TARGET_CATEGORIES = [
    "apparel", "furniture", "home_decor", "kitchen",
    "electronics", "jewelry", "outdoor"
]


def assign_category(product_type: str) -> str:
    """Map product_type to macro_category via explicit lookup."""
    return PRODUCT_TYPE_MAP.get(product_type, "other")


def download_image(url: str, dest: Path, timeout: int = 15) -> bool:
    if dest.exists():
        return True
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(dest, "JPEG", quality=95)
        return True
    except Exception:
        return False


def curate_subset(catalog: pd.DataFrame, n_per_category: int = 800) -> pd.DataFrame:
    """Select a balanced subset with strict caps per category.

    Deduplicates image_ids upfront so sampling draws from unique images.
    """
    catalog = catalog.copy()
    catalog["macro_category"] = catalog["product_type"].apply(assign_category)

    # Resolution filter
    catalog = catalog[(catalog["height"] >= 256) & (catalog["width"] >= 256)]
    print(f"After resolution filter: {len(catalog):,} rows")

    # Dedupe on image_id BEFORE sampling (same image can appear in multiple listings).
    # Keep the first occurrence; pandas is stable.
    n_before = len(catalog)
    catalog = catalog.drop_duplicates(subset=["image_id"]).reset_index(drop=True)
    print(f"After image_id dedup: {len(catalog):,} unique images "
          f"(removed {n_before - len(catalog):,} dupes)")

    # Show raw category sizes on unique images
    raw_counts = catalog["macro_category"].value_counts()
    print("\nAvailable unique images by category:")
    for cat, count in raw_counts.items():
        print(f"  {cat:15s}: {count:6,}")

    # Sample up to n_per_category from each target category
    samples = []
    print(f"\nSampling up to {n_per_category} per category...")
    for cat in TARGET_CATEGORIES:
        cat_df = catalog[catalog["macro_category"] == cat]
        if len(cat_df) == 0:
            continue
        n_sample = min(n_per_category, len(cat_df))
        sampled = cat_df.sample(n=n_sample, random_state=42)
        samples.append(sampled)
        print(f"  {cat:15s}: sampled {n_sample}")

    subset = pd.concat(samples).reset_index(drop=True)

    print(f"\nFinal balanced subset: {len(subset):,} images")
    print(f"Final distribution:")
    for cat, count in subset["macro_category"].value_counts().items():
        print(f"  {cat:15s}: {count:5d} ({100*count/len(subset):.1f}%)")

    return subset


def download_subset_images(
    subset: pd.DataFrame, output_dir: Path, use_small: bool = True, max_workers: int = 20
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    url_col = "image_url_small" if use_small else "image_url"

    def _download_one(row):
        cat_dir = output_dir / row["macro_category"]
        cat_dir.mkdir(parents=True, exist_ok=True)
        dest = cat_dir / f"{row['image_id']}.jpg"
        success = download_image(row[url_col], dest)
        return row["image_id"], success, str(dest) if success else None

    print(f"\nDownloading {len(subset):,} images ({max_workers} threads)...")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_one, row): idx for idx, row in subset.iterrows()}
        success_count = fail_count = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            img_id, success, path = future.result()
            results.append({"image_id": img_id, "downloaded": success, "local_path": path})
            if success:
                success_count += 1
            else:
                fail_count += 1

    results_df = pd.DataFrame(results)
    subset = subset.merge(results_df, on="image_id")
    subset = subset[subset["downloaded"] == True].drop(columns=["downloaded"])
    print(f"\nDownloaded: {success_count:,} | Failed: {fail_count:,}")
    return subset


def main():
    parser = argparse.ArgumentParser(description="Curate balanced ABO subset")
    parser.add_argument("--catalog", type=str, default="data/metadata/catalog.parquet")
    parser.add_argument("--n_per_category", type=int, default=800,
                        help="Max images per category (keeps balance)")
    parser.add_argument("--output_dir", type=str, default="data/raw")
    parser.add_argument("--use_small", action="store_true", default=True)
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--clean_existing", action="store_true", default=False,
                        help="Remove existing data/raw before downloading")
    args = parser.parse_args()

    print("=" * 60)
    print("ABO BALANCED Subset Curation")
    print("=" * 60)

    if args.clean_existing:
        import shutil
        out = Path(args.output_dir)
        if out.exists():
            print(f"Removing existing {out}...")
            shutil.rmtree(out)

    catalog = pd.read_parquet(args.catalog)
    print(f"Loaded catalog: {len(catalog):,} images")

    subset = curate_subset(catalog, args.n_per_category)
    subset = download_subset_images(
        subset, Path(args.output_dir),
        use_small=args.use_small, max_workers=args.max_workers
    )

    manifest_path = Path(args.output_dir) / "subset_manifest.csv"
    subset.to_csv(manifest_path, index=False)
    print(f"\nManifest saved to {manifest_path}")
    print(f"Final dataset: {len(subset):,} images across {subset['macro_category'].nunique()} categories")


if __name__ == "__main__":
    main()
