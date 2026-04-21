"""
Analysis and report utilities for Phase 5.

Consumes per-image CSVs produced by :class:`EvaluationRunner` and produces the
report-ready artifacts called for in the project plan:

- Overall metric comparison table across all methods (CSV + DataFrame).
- Per-category breakdown (long-format DataFrame and per-metric pivot CSV).
- Paired t-tests vs a reference method (typically ``clearshot``).
- Visual comparison grids (input | baselines... | reference | ground-truth).
- Failure-case picker (worst-k rows by LPIPS for a chosen method).

All functions are intentionally small and side-effect-free except for the
``save_*`` helpers and ``make_comparison_grid``, which write files on disk.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .metrics import DEFAULT_METRIC_KEYS, aggregate_rows, paired_ttests


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_all_per_image(
    evaluation_dir: Union[str, Path],
    methods: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Concatenate every ``per_image.csv`` under ``evaluation_dir``.

    Args:
        evaluation_dir: Top-level output dir used by :class:`EvaluationRunner`.
        methods: Optional explicit list of subdirectory names to include.

    Returns:
        Long-format DataFrame with one row per (image_id, method).
    """
    evaluation_dir = Path(evaluation_dir)
    if methods is None:
        methods = [p.name for p in evaluation_dir.iterdir() if p.is_dir() and not p.name.startswith("_")]

    frames: List[pd.DataFrame] = []
    for m in methods:
        csv_path = evaluation_dir / m / "per_image.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "method" not in df.columns:
            df["method"] = m
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["image_id", "method", "psnr", "ssim", "lpips"])

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------


def overall_summary(
    long_df: pd.DataFrame,
    metric_keys: Sequence[str] = DEFAULT_METRIC_KEYS,
    fid_by_method: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """Mean / median / std per method, optionally with FID column.

    Rows with non-ok status are ignored.
    """
    if "status" in long_df.columns:
        long_df = long_df[long_df["status"] == "ok"]

    grouped = long_df.groupby("method")
    rows: List[Dict[str, Any]] = []
    for method, sub in grouped:
        d: Dict[str, Any] = {"method": method, "n": len(sub)}
        for k in metric_keys:
            if k not in sub.columns:
                continue
            vals = pd.to_numeric(sub[k], errors="coerce").dropna()
            d[f"{k}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            d[f"{k}_median"] = float(vals.median()) if len(vals) else float("nan")
            d[f"{k}_std"] = float(vals.std(ddof=0)) if len(vals) else float("nan")
        if fid_by_method is not None and method in fid_by_method:
            d["fid"] = float(fid_by_method[method])
        rows.append(d)

    return pd.DataFrame(rows).set_index("method").sort_index()


def per_category_summary(
    long_df: pd.DataFrame,
    metric_keys: Sequence[str] = DEFAULT_METRIC_KEYS,
    aggregator: str = "mean",
) -> Dict[str, pd.DataFrame]:
    """Per-metric method x category table (pivoted DataFrame).

    Returns:
        ``{metric: DataFrame}`` where each DataFrame has methods as rows and
        categories as columns.
    """
    if "status" in long_df.columns:
        long_df = long_df[long_df["status"] == "ok"].copy()

    if aggregator not in {"mean", "median"}:
        raise ValueError("aggregator must be 'mean' or 'median'")
    agg_fn = "mean" if aggregator == "mean" else "median"

    out: Dict[str, pd.DataFrame] = {}
    for k in metric_keys:
        if k not in long_df.columns:
            continue
        long_df[k] = pd.to_numeric(long_df[k], errors="coerce")
        pivot = long_df.pivot_table(
            index="method", columns="category", values=k, aggfunc=agg_fn
        )
        pivot["__all__"] = long_df.groupby("method")[k].agg(agg_fn)
        out[k] = pivot.sort_index()
    return out


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------


def ttests_vs_reference(
    long_df: pd.DataFrame,
    reference_method: str = "clearshot",
    metric_keys: Sequence[str] = DEFAULT_METRIC_KEYS,
) -> pd.DataFrame:
    """Paired t-tests: reference method vs every other method, per metric.

    Args:
        long_df: Long-format per-image dataframe (see :func:`load_all_per_image`).
        reference_method: The method considered as ``A`` in the ``A vs B`` paired
            test. Typically ``"clearshot"``.

    Returns:
        Tidy DataFrame with columns
        ``[reference, other, metric, n, mean_diff, t_statistic, p_value]``.
    """
    if "status" in long_df.columns:
        long_df = long_df[long_df["status"] == "ok"]

    methods = sorted(long_df["method"].unique())
    if reference_method not in methods:
        raise ValueError(
            f"Reference method '{reference_method}' not present. Available: {methods}"
        )

    ref_rows = long_df[long_df["method"] == reference_method].to_dict("records")

    out: List[Dict[str, Any]] = []
    for m in methods:
        if m == reference_method:
            continue
        other_rows = long_df[long_df["method"] == m].to_dict("records")
        stats = paired_ttests(ref_rows, other_rows, metric_keys=metric_keys)
        for metric_name, s in stats.items():
            out.append(
                {
                    "reference": reference_method,
                    "other": m,
                    "metric": metric_name,
                    "n": s["n"],
                    "mean_diff": s["mean_diff"],
                    "t_statistic": s["t_statistic"],
                    "p_value": s["p_value"],
                }
            )
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Visual comparison grids
# ---------------------------------------------------------------------------


def _open_rgb(path: Union[str, Path], size: Optional[Tuple[int, int]] = None):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BILINEAR)
    return img


def make_comparison_grid(
    image_ids: Sequence[str],
    evaluation_dir: Union[str, Path],
    methods_in_order: Sequence[str],
    subset_manifest: pd.DataFrame,
    output_path: Union[str, Path],
    tile_size: int = 256,
    show_target: bool = True,
    show_input: bool = True,
    title: Optional[str] = None,
) -> Path:
    """Build an N x (cols) comparison grid PNG with matplotlib.

    Columns: ``Input (degraded)`` -> each method in order -> ``Ground truth``.

    Args:
        image_ids: Which images to include (one row per id).
        evaluation_dir: Root output dir used by :class:`EvaluationRunner`.
        methods_in_order: Method names (directories under ``evaluation_dir``) to plot.
        subset_manifest: DataFrame with ``image_id``, ``degraded_path``, ``clean_path``.
        output_path: Where to save the grid PNG.
        tile_size: Side length for each tile (also used to resize images for uniformity).
        show_target: Include ground-truth column on the right.
        show_input: Include input (degraded) column on the left.
        title: Optional figure title.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evaluation_dir = Path(evaluation_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mani = subset_manifest.set_index("image_id")
    rows = len(image_ids)
    cols = len(methods_in_order) + int(show_target) + int(show_input)
    size = (tile_size, tile_size)

    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)
    if cols == 1:
        axes = np.expand_dims(axes, 1)

    headers: List[str] = []
    if show_input:
        headers.append("Input")
    headers.extend(list(methods_in_order))
    if show_target:
        headers.append("Ground truth")

    for r_idx, img_id in enumerate(image_ids):
        if img_id not in mani.index:
            continue
        row = mani.loc[img_id]
        col = 0
        if show_input:
            try:
                tile = _open_rgb(row["degraded_path"], size=size)
                axes[r_idx, col].imshow(tile)
            except Exception as e:  # noqa: BLE001
                axes[r_idx, col].text(0.5, 0.5, f"missing\n{e}", ha="center", va="center")
            axes[r_idx, col].axis("off")
            col += 1
        for m in methods_in_order:
            pred_path = evaluation_dir / m / f"{img_id}.png"
            try:
                tile = _open_rgb(pred_path, size=size)
                axes[r_idx, col].imshow(tile)
            except Exception as e:  # noqa: BLE001
                axes[r_idx, col].text(0.5, 0.5, f"missing\n{e}", ha="center", va="center")
            axes[r_idx, col].axis("off")
            col += 1
        if show_target:
            try:
                tile = _open_rgb(row["clean_path"], size=size)
                axes[r_idx, col].imshow(tile)
            except Exception as e:  # noqa: BLE001
                axes[r_idx, col].text(0.5, 0.5, f"missing\n{e}", ha="center", va="center")
            axes[r_idx, col].axis("off")

    for c_idx, h in enumerate(headers):
        axes[0, c_idx].set_title(h, fontsize=11, fontweight="bold")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Failure case analysis
# ---------------------------------------------------------------------------


def select_failure_cases(
    long_df: pd.DataFrame,
    method: str,
    k: int = 8,
    by: str = "lpips",
    ascending: bool = False,
) -> pd.DataFrame:
    """Pick the worst-k rows for a chosen method and metric.

    For ``lpips`` (lower = better), the "worst" rows are the ones with the
    *highest* values, so ``ascending=False`` picks the worst cases by default.
    For ``psnr``/``ssim`` (higher = better), pass ``ascending=True``.
    """
    sub = long_df[long_df["method"] == method].copy()
    if "status" in sub.columns:
        sub = sub[sub["status"] == "ok"]
    sub[by] = pd.to_numeric(sub[by], errors="coerce")
    sub = sub.dropna(subset=[by])
    sub = sub.sort_values(by=by, ascending=ascending).head(k)
    return sub.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------


def save_report_tables(
    evaluation_dir: Union[str, Path],
    long_df: pd.DataFrame,
    fid_by_method: Optional[Mapping[str, float]] = None,
    reference_method: str = "clearshot",
    metric_keys: Sequence[str] = DEFAULT_METRIC_KEYS,
) -> Dict[str, Path]:
    """Compute and save all summary tables as CSV. Returns the written paths.

    Writes:
        - ``summary_overall.csv``
        - ``summary_per_category__{metric}.csv`` for each metric
        - ``paired_ttests_vs_{reference_method}.csv`` when the reference exists
    """
    evaluation_dir = Path(evaluation_dir)
    report_dir = evaluation_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}

    overall = overall_summary(long_df, metric_keys=metric_keys, fid_by_method=fid_by_method)
    overall_path = report_dir / "summary_overall.csv"
    overall.to_csv(overall_path)
    written["summary_overall"] = overall_path

    per_cat = per_category_summary(long_df, metric_keys=metric_keys, aggregator="mean")
    for metric_name, pivot in per_cat.items():
        p = report_dir / f"summary_per_category__{metric_name}.csv"
        pivot.to_csv(p)
        written[f"summary_per_category_{metric_name}"] = p

    try:
        tt = ttests_vs_reference(long_df, reference_method=reference_method, metric_keys=metric_keys)
        tt_path = report_dir / f"paired_ttests_vs_{reference_method}.csv"
        tt.to_csv(tt_path, index=False)
        written["ttests"] = tt_path
    except ValueError as e:
        print(f"[analysis] Skipping t-tests: {e}")

    return written
