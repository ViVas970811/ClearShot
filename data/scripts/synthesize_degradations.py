"""
Generate synthetically degraded versions of clean product images.

For each clean image, generates 1-3 degraded variants with randomized
degradation parameters. Saves pairs and a manifest CSV for training.

Usage:
    python data/scripts/synthesize_degradations.py \
        --input_dir data/raw \
        --output_dir data/degraded \
        --clean_dir data/clean \
        --n_variants 2
"""
import argparse
import json
import shutil
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.training.degradation import DegradationPipeline, DegradationConfig


def process_single_image(args_tuple):
    """Process a single image: copy clean + generate degraded variants."""
    img_path, clean_dir, degraded_dir, n_variants, base_seed = args_tuple
    img_path = Path(img_path)
    results = []

    try:
        img = Image.open(img_path).convert("RGB")
        # Resize to 512x512 for training consistency
        img = img.resize((512, 512), Image.LANCZOS)

        stem = img_path.stem
        category = img_path.parent.name

        # Save clean version
        clean_cat_dir = Path(clean_dir) / category
        clean_cat_dir.mkdir(parents=True, exist_ok=True)
        clean_path = clean_cat_dir / f"{stem}.jpg"
        img.save(clean_path, "JPEG", quality=95)

        # Generate degraded variants
        for v in range(n_variants):
            seed = base_seed + hash(f"{stem}_{v}") % (2**31)
            config = DegradationConfig(seed=seed)
            pipeline = DegradationPipeline(config)
            degraded, params = pipeline.apply(img)

            deg_cat_dir = Path(degraded_dir) / category
            deg_cat_dir.mkdir(parents=True, exist_ok=True)
            deg_path = deg_cat_dir / f"{stem}_v{v}.jpg"
            degraded.save(deg_path, "JPEG", quality=85)

            results.append({
                "clean_path": str(clean_path),
                "degraded_path": str(deg_path),
                "category": category,
                "variant": v,
                "image_id": stem,
                "degradation_params": json.dumps(params, default=str),
            })

    except Exception as e:
        print(f"  [error] {img_path.name}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate degraded image pairs")
    parser.add_argument("--input_dir", type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="data/degraded")
    parser.add_argument("--clean_dir", type=str, default="data/clean")
    parser.add_argument("--n_variants", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    clean_dir = Path(args.clean_dir)

    # Collect all images
    image_paths = sorted(input_dir.rglob("*.jpg"))
    print(f"Found {len(image_paths):,} images in {input_dir}")
    print(f"Generating {args.n_variants} degraded variant(s) per image")
    print(f"Total pairs to generate: {len(image_paths) * args.n_variants:,}")

    # Prepare arguments
    task_args = [
        (str(p), str(clean_dir), str(output_dir), args.n_variants, args.seed)
        for p in image_paths
    ]

    all_results = []
    print(f"\nProcessing with {args.max_workers} workers...")
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_single_image, a): a for a in task_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Degrading"):
            results = future.result()
            all_results.extend(results)

    # Save manifest
    manifest = pd.DataFrame(all_results)
    manifest_path = Path("data") / "pairs_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print(f"\nDone!")
    print(f"  Clean images: {clean_dir}")
    print(f"  Degraded images: {output_dir}")
    print(f"  Total pairs: {len(manifest):,}")
    print(f"  Manifest: {manifest_path}")

    # Split into train/val/test
    from sklearn.model_selection import train_test_split

    unique_ids = manifest["image_id"].unique()
    train_ids, temp_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    train_manifest = manifest[manifest["image_id"].isin(train_ids)]
    val_manifest = manifest[manifest["image_id"].isin(val_ids)]
    test_manifest = manifest[manifest["image_id"].isin(test_ids)]

    train_manifest.to_csv("data/train_manifest.csv", index=False)
    val_manifest.to_csv("data/val_manifest.csv", index=False)
    test_manifest.to_csv("data/test_manifest.csv", index=False)

    print(f"\n  Train: {len(train_manifest):,} pairs ({len(train_ids)} images)")
    print(f"  Val:   {len(val_manifest):,} pairs ({len(val_ids)} images)")
    print(f"  Test:  {len(test_manifest):,} pairs ({len(test_ids)} images)")


if __name__ == "__main__":
    main()
