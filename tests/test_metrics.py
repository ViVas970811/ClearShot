"""Unit tests for :mod:`src.evaluation.metrics`.

These tests use tiny synthetic PIL images and mocked rows so they run quickly
without requiring the dataset, LoRA checkpoints, or a GPU.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    DEFAULT_METRIC_KEYS,
    ImageQualityMetrics,
    aggregate_rows,
    paired_ttests,
)


def _solid(color, size=64):
    return Image.new("RGB", (size, size), color)


@pytest.fixture(scope="module")
def metrics() -> ImageQualityMetrics:
    return ImageQualityMetrics(device="cpu", eval_resolution=64)


def test_psnr_identical_is_high(metrics: ImageQualityMetrics) -> None:
    img = _solid((120, 130, 140))
    value = metrics.psnr(img, img)
    assert value > 60  # torchmetrics clips at ~80 for identical inputs


def test_ssim_identical_is_one(metrics: ImageQualityMetrics) -> None:
    img = _solid((200, 100, 50))
    value = metrics.ssim(img, img)
    assert value == pytest.approx(1.0, abs=1e-3)


def test_compute_pairwise_returns_all_keys(metrics: ImageQualityMetrics) -> None:
    pred = _solid((120, 130, 140))
    target = _solid((120, 130, 145))
    row = metrics.compute_pairwise(pred, target, extra={"image_id": "x", "method": "m"})
    for k in ("psnr", "ssim", "lpips", "image_id", "method"):
        assert k in row
    assert math.isfinite(row["psnr"])
    assert 0.0 <= row["ssim"] <= 1.0
    assert row["lpips"] >= 0.0


def test_pairwise_worse_pair_has_worse_metrics(metrics: ImageQualityMetrics) -> None:
    target = _solid((128, 128, 128))
    close = Image.fromarray(
        (np.asarray(target).astype(np.int32) + 5).clip(0, 255).astype(np.uint8)
    )
    far = _solid((0, 0, 0))

    close_row = metrics.compute_pairwise(close, target)
    far_row = metrics.compute_pairwise(far, target)

    assert close_row["psnr"] > far_row["psnr"]
    assert close_row["ssim"] > far_row["ssim"]
    assert close_row["lpips"] < far_row["lpips"] + 1e-6


def test_aggregate_rows_overall_and_per_category() -> None:
    rows = [
        {"image_id": "a", "category": "cat1", "psnr": 20.0, "ssim": 0.8, "lpips": 0.2},
        {"image_id": "b", "category": "cat1", "psnr": 30.0, "ssim": 0.9, "lpips": 0.1},
        {"image_id": "c", "category": "cat2", "psnr": 10.0, "ssim": 0.5, "lpips": 0.4},
    ]

    overall = aggregate_rows(rows)
    assert set(overall.keys()) == {"__overall__"}
    assert overall["__overall__"].n == 3
    assert overall["__overall__"].mean["psnr"] == pytest.approx(20.0)

    by_cat = aggregate_rows(rows, group_by="category")
    assert set(by_cat.keys()) == {"cat1", "cat2"}
    assert by_cat["cat1"].n == 2
    assert by_cat["cat1"].mean["psnr"] == pytest.approx(25.0)
    assert by_cat["cat2"].n == 1
    assert by_cat["cat2"].median["lpips"] == pytest.approx(0.4)


def test_paired_ttests_reports_significance() -> None:
    rng = np.random.default_rng(0)
    ids = [f"img_{i}" for i in range(30)]
    a_rows = []
    b_rows = []
    for i, img_id in enumerate(ids):
        base = float(rng.normal(25.0, 1.0))
        a_rows.append({"image_id": img_id, "psnr": base + 3.0, "ssim": 0.9, "lpips": 0.2})
        b_rows.append({"image_id": img_id, "psnr": base, "ssim": 0.85, "lpips": 0.25})

    stats = paired_ttests(a_rows, b_rows, metric_keys=("psnr",))
    assert stats["psnr"]["n"] == 30
    assert stats["psnr"]["mean_diff"] > 2.5
    assert stats["psnr"]["p_value"] < 1e-6


def test_paired_ttests_handles_empty_overlap() -> None:
    a_rows = [{"image_id": "a", "psnr": 20.0}]
    b_rows = [{"image_id": "b", "psnr": 21.0}]
    stats = paired_ttests(a_rows, b_rows, metric_keys=("psnr",))
    assert stats["psnr"]["n"] == 0
    assert math.isnan(stats["psnr"]["t_statistic"])
    assert math.isnan(stats["psnr"]["p_value"])
