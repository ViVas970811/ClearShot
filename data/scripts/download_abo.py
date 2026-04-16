"""
Download Amazon Berkeley Objects (ABO) dataset metadata and images.

The ABO dataset is publicly hosted on AWS S3. This script:
1. Downloads the image metadata CSV (image paths, dimensions, listing IDs)
2. Downloads listing metadata JSONs (product categories, attributes)
3. Merges into a single catalog for subset curation

Usage:
    python data/scripts/download_abo.py --output_dir data/metadata
"""
import argparse
import gzip
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ABO_BASE = "https://amazon-berkeley-objects.s3.amazonaws.com"
IMAGES_META_URL = f"{ABO_BASE}/images/metadata/images.csv.gz"
LISTINGS_META_URLS = [
    f"{ABO_BASE}/listings/metadata/listings_{i}.json.gz" for i in range(10)
]


def download_file(url: str, dest: Path, desc: str = "", retries: int = 5) -> Path:
    """Download a file with progress bar, resume support, and retries."""
    import time
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return dest

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))

            with open(dest, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=desc or dest.name
            ) as bar:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
            return dest
        except (requests.ConnectionError, requests.Timeout) as e:
            wait = 2 ** attempt
            print(f"  [retry {attempt}/{retries}] {type(e).__name__}, waiting {wait}s...")
            if dest.exists():
                dest.unlink()  # remove partial download
            time.sleep(wait)

    raise RuntimeError(f"Failed to download {url} after {retries} retries")


def download_images_metadata(output_dir: Path) -> pd.DataFrame:
    """Download and parse image metadata CSV."""
    gz_path = output_dir / "images.csv.gz"
    csv_path = output_dir / "images.csv"

    if csv_path.exists():
        print("[skip] images.csv already exists")
        return pd.read_csv(csv_path)

    print("Downloading image metadata...")
    download_file(IMAGES_META_URL, gz_path, "images.csv.gz")

    print("Decompressing...")
    with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} image records")
    return df


def download_listings_metadata(output_dir: Path) -> pd.DataFrame:
    """Download and parse all listing metadata JSON files."""
    listings_path = output_dir / "listings_combined.parquet"

    if listings_path.exists():
        print("[skip] listings_combined.parquet already exists")
        return pd.read_parquet(listings_path)

    all_listings = []
    for url in LISTINGS_META_URLS:
        fname = url.split("/")[-1]
        gz_path = output_dir / fname
        download_file(url, gz_path, fname)

        print(f"  Parsing {fname}...")
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    record = {
                        "item_id": item.get("item_id", ""),
                        "marketplace": item.get("marketplace", ""),
                        "main_image_id": item.get("main_image_id", ""),
                        "product_type": _extract_product_type(item),
                        "item_name": _extract_item_name(item),
                        "brand": _extract_brand(item),
                        "color": _extract_color(item),
                        "bullet_point_count": len(item.get("bullet_point", [])),
                    }
                    all_listings.append(record)
                except json.JSONDecodeError:
                    continue

    df = pd.DataFrame(all_listings)
    df.to_parquet(listings_path, index=False)
    print(f"  Parsed {len(df):,} listings")
    return df


def _extract_product_type(item: dict) -> str:
    pt = item.get("product_type", [])
    if isinstance(pt, list) and pt:
        val = pt[0].get("value", "") if isinstance(pt[0], dict) else str(pt[0])
        return val
    return ""


def _extract_item_name(item: dict) -> str:
    names = item.get("item_name", [])
    if isinstance(names, list) and names:
        val = names[0].get("value", "") if isinstance(names[0], dict) else str(names[0])
        return val
    return ""


def _extract_brand(item: dict) -> str:
    brands = item.get("brand", [])
    if isinstance(brands, list) and brands:
        val = brands[0].get("value", "") if isinstance(brands[0], dict) else str(brands[0])
        return val
    return ""


def _extract_color(item: dict) -> str:
    colors = item.get("color", [])
    if isinstance(colors, list) and colors:
        val = colors[0].get("value", "") if isinstance(colors[0], dict) else str(colors[0])
        return val
    return ""


def merge_catalog(images_df: pd.DataFrame, listings_df: pd.DataFrame) -> pd.DataFrame:
    """Merge image metadata with listing metadata to create a unified catalog."""
    # The images CSV has: image_id, height, width, path
    # The listings have: item_id, main_image_id, product_type, etc.
    # We need to join on image_id = main_image_id for main images,
    # but also keep all images linked to listings

    # Create image URL from path
    images_df = images_df.copy()
    images_df["image_url"] = images_df["path"].apply(
        lambda p: f"{ABO_BASE}/images/original/{p}"
    )
    images_df["image_url_small"] = images_df["path"].apply(
        lambda p: f"{ABO_BASE}/images/small/{p}"
    )

    # Merge: images have image_id, listings have main_image_id
    catalog = images_df.merge(
        listings_df, left_on="image_id", right_on="main_image_id", how="inner"
    )

    print(f"  Merged catalog: {len(catalog):,} images with listing info")
    return catalog


def main():
    parser = argparse.ArgumentParser(description="Download ABO dataset metadata")
    parser.add_argument(
        "--output_dir", type=str, default="data/metadata",
        help="Directory to store downloaded metadata"
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ABO Dataset Metadata Download")
    print("=" * 60)

    images_df = download_images_metadata(output_dir)
    listings_df = download_listings_metadata(output_dir)
    catalog = merge_catalog(images_df, listings_df)

    catalog_path = output_dir / "catalog.parquet"
    catalog.to_parquet(catalog_path, index=False)
    print(f"\nCatalog saved to {catalog_path}")
    print(f"  Total images with listing info: {len(catalog):,}")
    print(f"  Unique product types: {catalog['product_type'].nunique():,}")
    print(f"  Unique listings: {catalog['item_id'].nunique():,}")

    return catalog


if __name__ == "__main__":
    main()
