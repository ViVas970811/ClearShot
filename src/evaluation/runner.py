"""
Resumable evaluation runner for Phase 5.

Given a test manifest (``clean_path``, ``degraded_path``, ``category``, ``image_id``)
and a list of :class:`BaselineEnhancer` instances, the runner will:

1. (Optionally) stratify-sample a subset by ``category`` so every category is
   represented even when ``subset_size`` is small.
2. For each baseline, write enhanced outputs under
   ``<output_dir>/<baseline>/<image_id>.png``.
3. Score each (pred, target) pair with :class:`ImageQualityMetrics` and append
   one row per image to ``<output_dir>/<baseline>/per_image.csv``.
4. Resume support: images whose output file + metric row already exist are
   skipped. Deleting either one forces a re-run for that image.
5. (Optionally) also build a reference directory with ``clean_path`` copies so
   FID can be computed across the same subset afterwards.

Per-image CSV schema:
    ``image_id, category, clean_path, degraded_path, pred_path, method,
     psnr, ssim, lpips, status, error``

``status`` is one of ``"ok"``, ``"enhance_failed"``, ``"metrics_failed"``,
``"skipped"``.
"""

from __future__ import annotations

import csv
import shutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import pandas as pd
from PIL import Image
from tqdm import tqdm

from .baselines import BaselineEnhancer
from .metrics import ImageQualityMetrics


PER_IMAGE_COLUMNS: List[str] = [
    "image_id",
    "category",
    "clean_path",
    "degraded_path",
    "pred_path",
    "method",
    "psnr",
    "ssim",
    "lpips",
    "status",
    "error",
]


@dataclass
class EvaluationConfig:
    """Paths and sampling knobs for an evaluation run."""

    manifest_path: Union[str, Path]
    output_dir: Union[str, Path]
    subset_size: Optional[int] = None
    stratify_by_category: bool = True
    seed: int = 42
    eval_resolution: int = 512
    reference_manifest_path: Optional[Union[str, Path]] = None
    reference_dir_name: str = "_reference_clean"
    input_column: str = "degraded_path"
    target_column: str = "clean_path"
    image_id_column: str = "image_id"
    category_column: str = "category"
    image_format: str = "png"


@dataclass
class EvaluationResult:
    """Summary of a single baseline run."""

    method: str
    per_image_csv: Path
    pred_dir: Path
    n_rows: int
    n_new: int
    n_skipped: int
    duration_seconds: float
    errors: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_manifest(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"clean_path", "degraded_path"} - set(df.columns)
    if missing:
        raise ValueError(f"Manifest {path} is missing required columns: {missing}")
    if "image_id" not in df.columns:
        df["image_id"] = df["clean_path"].apply(lambda p: Path(p).stem)
    if "category" not in df.columns:
        df["category"] = df["clean_path"].apply(lambda p: Path(p).parent.name)
    return df


def _sample_subset(
    manifest: pd.DataFrame,
    n: Optional[int],
    stratify: bool,
    seed: int,
    category_col: str,
) -> pd.DataFrame:
    if n is None or n >= len(manifest):
        return manifest.reset_index(drop=True)
    if stratify and category_col in manifest.columns:
        cats = manifest[category_col].unique()
        per_cat = max(1, n // len(cats))
        chunks = []
        for cat in cats:
            cat_df = manifest[manifest[category_col] == cat]
            take = min(per_cat, len(cat_df))
            chunks.append(cat_df.sample(n=take, random_state=seed))
        sampled = pd.concat(chunks)
        if len(sampled) < n:
            remaining = manifest.drop(sampled.index)
            extra = remaining.sample(
                n=min(n - len(sampled), len(remaining)), random_state=seed
            )
            sampled = pd.concat([sampled, extra])
        return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return manifest.sample(n=n, random_state=seed).reset_index(drop=True)


def _read_existing_rows(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    if "image_id" not in df.columns:
        return {}
    return {str(r["image_id"]): r.to_dict() for _, r in df.iterrows()}


def _write_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_IMAGE_COLUMNS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in PER_IMAGE_COLUMNS})


def _build_reference_dir(
    manifest: pd.DataFrame,
    ref_dir: Path,
    target_column: str,
    image_id_column: str,
    image_format: str,
    eval_resolution: Optional[int],
) -> int:
    """Materialize clean reference images into ``ref_dir`` for FID computation.

    Existing files are kept, so this is resumable and cheap on re-runs.
    """
    ref_dir.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for _, row in manifest.iterrows():
        src = Path(row[target_column])
        if not src.exists():
            continue
        dst = ref_dir / f"{row[image_id_column]}.{image_format}"
        if dst.exists():
            continue
        try:
            img = Image.open(src).convert("RGB")
            if eval_resolution and img.size != (eval_resolution, eval_resolution):
                img = img.resize((eval_resolution, eval_resolution), Image.BILINEAR)
            img.save(dst)
            n_new += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [ref] skipped {src.name}: {e}")
    return n_new


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class EvaluationRunner:
    """Orchestrates baseline enhancement + metric computation with resume support."""

    def __init__(
        self,
        config: EvaluationConfig,
        metrics: Optional[ImageQualityMetrics] = None,
        progress: bool = True,
    ) -> None:
        self.config = config
        self.metrics = metrics or ImageQualityMetrics(eval_resolution=config.eval_resolution)
        self.progress = progress

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.manifest = _sample_subset(
            _load_manifest(config.manifest_path),
            n=config.subset_size,
            stratify=config.stratify_by_category,
            seed=config.seed,
            category_col=config.category_column,
        )

        self.subset_manifest_path = self.output_dir / "_subset_manifest.csv"
        self.manifest.to_csv(self.subset_manifest_path, index=False)
        print(
            f"[EvaluationRunner] Using {len(self.manifest)} images from "
            f"{config.manifest_path}. Subset manifest saved to {self.subset_manifest_path}."
        )

    # ------------------------------------------------------------------
    # Run a single baseline
    # ------------------------------------------------------------------

    def run_baseline(self, baseline: BaselineEnhancer) -> EvaluationResult:
        """Enhance + score every image for one baseline. Returns a summary."""
        method = baseline.name
        pred_dir = self.output_dir / method
        pred_dir.mkdir(parents=True, exist_ok=True)

        per_image_csv = pred_dir / "per_image.csv"
        done_rows = _read_existing_rows(per_image_csv)

        t_start = time.time()
        n_new = 0
        n_skipped = 0
        errors: List[Dict[str, Any]] = []

        iterable: Iterable = self.manifest.itertuples(index=False)
        if self.progress:
            iterable = tqdm(
                list(iterable),
                desc=f"[{method}]",
                total=len(self.manifest),
            )

        for row in iterable:
            row_dict = row._asdict() if hasattr(row, "_asdict") else dict(row)
            image_id = str(row_dict[self.config.image_id_column])
            input_path = Path(row_dict[self.config.input_column])
            target_path = Path(row_dict[self.config.target_column])
            category = row_dict.get(self.config.category_column, "")
            pred_path = pred_dir / f"{image_id}.{self.config.image_format}"

            already_scored = image_id in done_rows and pred_path.exists()
            if already_scored:
                n_skipped += 1
                continue

            base_row: Dict[str, Any] = {
                "image_id": image_id,
                "category": category,
                "clean_path": str(target_path),
                "degraded_path": str(input_path),
                "pred_path": str(pred_path),
                "method": method,
                "psnr": None,
                "ssim": None,
                "lpips": None,
                "status": "ok",
                "error": "",
            }

            try:
                image = Image.open(input_path).convert("RGB")
            except Exception as e:  # noqa: BLE001
                base_row.update(status="enhance_failed", error=f"read_input: {e}")
                _write_row(per_image_csv, base_row)
                errors.append({"image_id": image_id, "stage": "read_input", "error": str(e)})
                continue

            try:
                pred = baseline.enhance(image)
                if pred.mode != "RGB":
                    pred = pred.convert("RGB")
                pred.save(pred_path)
            except Exception as e:  # noqa: BLE001
                base_row.update(
                    status="enhance_failed",
                    error=f"{type(e).__name__}: {e}",
                )
                _write_row(per_image_csv, base_row)
                errors.append(
                    {
                        "image_id": image_id,
                        "stage": "enhance",
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(),
                    }
                )
                continue

            try:
                target = Image.open(target_path).convert("RGB")
                pair_metrics = self.metrics.compute_pairwise(pred, target)
                base_row.update(pair_metrics)
            except Exception as e:  # noqa: BLE001
                base_row.update(
                    status="metrics_failed",
                    error=f"{type(e).__name__}: {e}",
                )
                _write_row(per_image_csv, base_row)
                errors.append(
                    {
                        "image_id": image_id,
                        "stage": "metrics",
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                continue

            _write_row(per_image_csv, base_row)
            n_new += 1

        duration = time.time() - t_start
        total_rows = len(_read_existing_rows(per_image_csv))

        print(
            f"[EvaluationRunner] '{method}' done: {n_new} new, {n_skipped} skipped, "
            f"{len(errors)} errors, total rows on disk: {total_rows}, "
            f"took {duration:.1f}s"
        )

        return EvaluationResult(
            method=method,
            per_image_csv=per_image_csv,
            pred_dir=pred_dir,
            n_rows=total_rows,
            n_new=n_new,
            n_skipped=n_skipped,
            duration_seconds=duration,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Run many baselines
    # ------------------------------------------------------------------

    def run_all(
        self,
        baselines: Sequence[BaselineEnhancer],
        build_reference: bool = True,
    ) -> Dict[str, EvaluationResult]:
        """Run every baseline in order. Optionally build the FID reference dir first."""
        if build_reference:
            self.build_reference_dir()
        results: Dict[str, EvaluationResult] = {}
        for b in baselines:
            print(f"\n{'='*60}\n[EvaluationRunner] Running baseline: {b.name}\n{'='*60}")
            results[b.name] = self.run_baseline(b)
        return results

    # ------------------------------------------------------------------
    # Reference directory (for FID)
    # ------------------------------------------------------------------

    def build_reference_dir(self) -> Path:
        """Materialize clean targets of the subset into a directory for FID.

        Uses the subset the runner is evaluating on rather than the full val
        set. Callers that want a larger / separate FID reference can set
        ``reference_manifest_path`` in :class:`EvaluationConfig`.
        """
        ref_manifest = self.manifest
        if self.config.reference_manifest_path is not None:
            ref_manifest = _load_manifest(self.config.reference_manifest_path)

        ref_dir = self.output_dir / self.config.reference_dir_name
        n_new = _build_reference_dir(
            ref_manifest,
            ref_dir,
            target_column=self.config.target_column,
            image_id_column=self.config.image_id_column,
            image_format=self.config.image_format,
            eval_resolution=self.config.eval_resolution,
        )
        print(
            f"[EvaluationRunner] Reference dir: {ref_dir} "
            f"(+{n_new} new, total {sum(1 for _ in ref_dir.glob('*'))})"
        )
        return ref_dir

    # ------------------------------------------------------------------
    # FID for all baselines
    # ------------------------------------------------------------------

    def compute_fid_all(
        self,
        results: Dict[str, EvaluationResult],
        ref_dir: Optional[Path] = None,
        batch_size: int = 32,
        num_workers: int = 2,
        dims: int = 2048,
    ) -> Dict[str, float]:
        """Compute FID for each baseline's prediction dir against the reference.

        Returns a ``{method: fid_value}`` mapping. NaN is stored on failure so
        one broken baseline does not abort the whole report.
        """
        if ref_dir is None:
            ref_dir = self.output_dir / self.config.reference_dir_name
        fid_by_method: Dict[str, float] = {}
        for method, res in results.items():
            try:
                fid = self.metrics.compute_fid(
                    res.pred_dir,
                    ref_dir,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    dims=dims,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[EvaluationRunner] FID failed for '{method}': {e}")
                fid = float("nan")
            fid_by_method[method] = fid
            print(f"[FID] {method}: {fid:.3f}")
        return fid_by_method
