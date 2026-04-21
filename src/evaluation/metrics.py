"""
Image quality metrics for Phase 5 evaluation.

Implements PSNR, SSIM, LPIPS (per-image, paired) and FID (aggregate, distribution-level)
plus lightweight helpers for overall and per-category aggregation of per-image rows.

Design notes:
    - Single-image metrics accept PIL.Image.Image inputs (RGB) and resize both
      tensors to a common resolution before scoring so pipelines that produce
      different output sizes (eg. Real-ESRGAN 2x -> 1024x1024 vs. 512x512 ground
      truth) can be compared fairly.
    - The LPIPS model is loaded lazily and cached on the instance.
    - FID uses the ``pytorch-fid`` package (``calculate_fid_given_paths``). Both
      directories must contain matched-content images at the same resolution.
    - Aggregation helpers operate on lists of flat dicts so callers can stream
      per-image results to/from CSV without any tight coupling.

Typical usage:

    >>> metrics = ImageQualityMetrics(device="cuda")
    >>> row = metrics.compute_pairwise(pred_pil, target_pil, extra={"method": "clearshot"})
    >>> # row == {"psnr": 24.3, "ssim": 0.88, "lpips": 0.12, "method": "clearshot"}
    >>> fid = metrics.compute_fid(pred_dir="out/clearshot", ref_dir="data/clean_val")
    >>> summary = aggregate_rows(rows, group_by=None)            # overall
    >>> by_cat  = aggregate_rows(rows, group_by="category")      # per-category
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PairwiseMetrics:
    """Per-image paired metrics between a predicted and a target image."""

    psnr: float
    ssim: float
    lpips: float

    def to_dict(self) -> Dict[str, float]:
        return {"psnr": self.psnr, "ssim": self.ssim, "lpips": self.lpips}


@dataclass
class AggregateMetrics:
    """Aggregated statistics (mean/median/std/n) across a list of per-image rows."""

    n: int
    mean: Dict[str, float]
    median: Dict[str, float]
    std: Dict[str, float]
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            **self.extra,
        }


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class ImageQualityMetrics:
    """Paired + distributional image-quality metrics used for Phase 5 evaluation.

    Args:
        device: ``"cuda"``, ``"cpu"``, or ``None`` to auto-detect.
        lpips_net: LPIPS backbone - ``"alex"`` (default, standard), ``"vgg"``, or ``"squeeze"``.
        eval_resolution: Common spatial resolution that predicted and target
            images are resized to before scoring. Matches the training resolution
            used by the LoRA adapter (512) by default. Pass ``None`` to disable
            resizing, which requires all inputs to already match.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        lpips_net: str = "alex",
        eval_resolution: Optional[int] = 512,
    ) -> None:
        import torch

        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.lpips_net = lpips_net
        self.eval_resolution = eval_resolution

        self._lpips_model = None
        self._ssim_fn = None
        self._psnr_fn = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _get_lpips(self):
        if self._lpips_model is None:
            import lpips

            self._lpips_model = lpips.LPIPS(net=self.lpips_net, verbose=False).to(
                self.device
            )
            self._lpips_model.eval()
        return self._lpips_model

    def _get_torchmetrics(self) -> Tuple[Any, Any]:
        """Return (ssim_fn, psnr_fn) from torchmetrics.functional."""
        if self._ssim_fn is None or self._psnr_fn is None:
            from torchmetrics.functional.image import (
                peak_signal_noise_ratio,
                structural_similarity_index_measure,
            )

            self._ssim_fn = structural_similarity_index_measure
            self._psnr_fn = peak_signal_noise_ratio
        return self._ssim_fn, self._psnr_fn

    # ------------------------------------------------------------------
    # Image preparation
    # ------------------------------------------------------------------

    def _prepare_pair(
        self, pred: Image.Image, target: Image.Image
    ) -> Tuple["np.ndarray", "np.ndarray"]:
        """Return matched-size float32 RGB arrays in [0, 1]."""
        if pred.mode != "RGB":
            pred = pred.convert("RGB")
        if target.mode != "RGB":
            target = target.convert("RGB")

        if self.eval_resolution is not None:
            size = (self.eval_resolution, self.eval_resolution)
            if pred.size != size:
                pred = pred.resize(size, Image.BILINEAR)
            if target.size != size:
                target = target.resize(size, Image.BILINEAR)
        elif pred.size != target.size:
            raise ValueError(
                f"pred size {pred.size} != target size {target.size} and "
                "eval_resolution is disabled. Set eval_resolution or resize inputs."
            )

        pred_arr = np.asarray(pred, dtype=np.float32) / 255.0
        target_arr = np.asarray(target, dtype=np.float32) / 255.0
        return pred_arr, target_arr

    def _to_torch(self, arr: "np.ndarray") -> "Any":
        """HWC float32 [0,1] -> 1xCxHxW torch tensor on self.device."""
        torch = self._torch
        t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return t

    # ------------------------------------------------------------------
    # Paired metrics
    # ------------------------------------------------------------------

    def psnr(self, pred: Image.Image, target: Image.Image) -> float:
        pred_arr, target_arr = self._prepare_pair(pred, target)
        _, psnr_fn = self._get_torchmetrics()
        t_pred = self._to_torch(pred_arr)
        t_target = self._to_torch(target_arr)
        value = psnr_fn(t_pred, t_target, data_range=1.0).item()
        return float(value)

    def ssim(self, pred: Image.Image, target: Image.Image) -> float:
        pred_arr, target_arr = self._prepare_pair(pred, target)
        ssim_fn, _ = self._get_torchmetrics()
        t_pred = self._to_torch(pred_arr)
        t_target = self._to_torch(target_arr)
        value = ssim_fn(t_pred, t_target, data_range=1.0).item()
        return float(value)

    def lpips(self, pred: Image.Image, target: Image.Image) -> float:
        torch = self._torch
        pred_arr, target_arr = self._prepare_pair(pred, target)
        t_pred = self._to_torch(pred_arr) * 2.0 - 1.0
        t_target = self._to_torch(target_arr) * 2.0 - 1.0
        with torch.no_grad():
            value = self._get_lpips()(t_pred, t_target).item()
        return float(value)

    def compute_pairwise(
        self,
        pred: Image.Image,
        target: Image.Image,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute PSNR, SSIM, LPIPS for a single (pred, target) pair.

        Args:
            pred: Enhanced image produced by a method under evaluation.
            target: Ground-truth clean image.
            extra: Optional extra fields to merge into the returned row (e.g.
                ``{"method": "clearshot", "image_id": "abc", "category": "apparel"}``).

        Returns:
            Flat ``dict`` with keys ``psnr``, ``ssim``, ``lpips`` plus any extras.
        """
        torch = self._torch
        pred_arr, target_arr = self._prepare_pair(pred, target)

        ssim_fn, psnr_fn = self._get_torchmetrics()
        t_pred_01 = self._to_torch(pred_arr)
        t_target_01 = self._to_torch(target_arr)

        with torch.no_grad():
            psnr_val = psnr_fn(t_pred_01, t_target_01, data_range=1.0).item()
            ssim_val = ssim_fn(t_pred_01, t_target_01, data_range=1.0).item()
            t_pred_norm = t_pred_01 * 2.0 - 1.0
            t_target_norm = t_target_01 * 2.0 - 1.0
            lpips_val = self._get_lpips()(t_pred_norm, t_target_norm).item()

        row: Dict[str, Any] = {
            "psnr": float(psnr_val),
            "ssim": float(ssim_val),
            "lpips": float(lpips_val),
        }
        if extra:
            row.update(extra)
        return row

    # ------------------------------------------------------------------
    # Distributional metric
    # ------------------------------------------------------------------

    def compute_fid(
        self,
        pred_dir: Union[str, Path],
        ref_dir: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 2,
        dims: int = 2048,
    ) -> float:
        """Frechet Inception Distance between two directories of images.

        Thin wrapper over ``pytorch_fid.fid_score.calculate_fid_given_paths``.
        Both directories must contain at least 2 images and should ideally have
        at least 2,000 samples for a stable FID.

        Args:
            pred_dir: Directory with enhanced predictions.
            ref_dir: Directory with ground-truth / reference images.
            batch_size: InceptionV3 batch size.
            num_workers: DataLoader workers.
            dims: Feature dimensionality (2048 = standard pool3 features).

        Returns:
            FID value (lower is better).
        """
        from pytorch_fid.fid_score import calculate_fid_given_paths

        pred_dir = str(pred_dir)
        ref_dir = str(ref_dir)

        for d in (pred_dir, ref_dir):
            p = Path(d)
            if not p.exists():
                raise FileNotFoundError(f"FID directory does not exist: {d}")
            n_images = sum(1 for _ in p.glob("*"))
            if n_images < 2:
                raise ValueError(f"FID directory needs at least 2 images: {d} has {n_images}")

        value = calculate_fid_given_paths(
            paths=[pred_dir, ref_dir],
            batch_size=batch_size,
            device=self.device,
            dims=dims,
            num_workers=num_workers,
        )
        return float(value)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _mean_std_median(values: Sequence[float]) -> Tuple[float, float, float]:
    cleaned = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not cleaned:
        return float("nan"), float("nan"), float("nan")
    mean = float(sum(cleaned) / len(cleaned))
    median = float(statistics.median(cleaned))
    std = float(statistics.pstdev(cleaned)) if len(cleaned) > 1 else 0.0
    return mean, median, std


DEFAULT_METRIC_KEYS: Tuple[str, ...] = ("psnr", "ssim", "lpips")


def aggregate_rows(
    rows: Iterable[Dict[str, Any]],
    metric_keys: Sequence[str] = DEFAULT_METRIC_KEYS,
    group_by: Optional[str] = None,
) -> Dict[str, AggregateMetrics]:
    """Aggregate per-image metric rows into overall or grouped summaries.

    Args:
        rows: Iterable of flat dicts; each must contain every key in ``metric_keys``.
        metric_keys: Which numeric fields to aggregate.
        group_by: Optional row field to bucket by (eg. ``"category"``). If ``None``
            the result has a single key, ``"__overall__"``.

    Returns:
        Mapping from group-name to :class:`AggregateMetrics`.
    """
    rows = list(rows)
    if not rows:
        return {}

    if group_by is None:
        buckets: Dict[str, List[Dict[str, Any]]] = {"__overall__": rows}
    else:
        buckets = {}
        for r in rows:
            key = r.get(group_by, "__missing__")
            buckets.setdefault(str(key), []).append(r)

    summaries: Dict[str, AggregateMetrics] = {}
    for group_name, group_rows in buckets.items():
        mean_d, median_d, std_d = {}, {}, {}
        for k in metric_keys:
            vals = [r.get(k) for r in group_rows if r.get(k) is not None]
            m, med, s = _mean_std_median(vals)
            mean_d[k] = m
            median_d[k] = med
            std_d[k] = s
        summaries[group_name] = AggregateMetrics(
            n=len(group_rows),
            mean=mean_d,
            median=median_d,
            std=std_d,
        )
    return summaries


def paired_ttests(
    rows_a: Sequence[Dict[str, Any]],
    rows_b: Sequence[Dict[str, Any]],
    metric_keys: Sequence[str] = DEFAULT_METRIC_KEYS,
    match_on: str = "image_id",
) -> Dict[str, Dict[str, float]]:
    """Paired two-sided t-tests (method A vs method B) per metric.

    Rows are matched by ``match_on`` (defaults to ``image_id``). Only images
    present in both row sets are used.

    Returns:
        ``{metric: {"t_statistic": ..., "p_value": ..., "mean_diff": ..., "n": ...}}``.
        ``mean_diff`` is defined as ``mean(A) - mean(B)`` on the paired subset.
    """
    from scipy import stats

    a_by_key = {r[match_on]: r for r in rows_a if match_on in r}
    b_by_key = {r[match_on]: r for r in rows_b if match_on in r}
    shared_keys = sorted(set(a_by_key) & set(b_by_key))

    results: Dict[str, Dict[str, float]] = {}
    for k in metric_keys:
        a_vals = [a_by_key[i][k] for i in shared_keys if k in a_by_key[i] and k in b_by_key[i]]
        b_vals = [b_by_key[i][k] for i in shared_keys if k in a_by_key[i] and k in b_by_key[i]]
        if len(a_vals) < 2:
            results[k] = {
                "t_statistic": float("nan"),
                "p_value": float("nan"),
                "mean_diff": float("nan"),
                "n": len(a_vals),
            }
            continue
        t_stat, p_val = stats.ttest_rel(a_vals, b_vals)
        results[k] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "mean_diff": float(np.mean(a_vals) - np.mean(b_vals)),
            "n": len(a_vals),
        }
    return results
