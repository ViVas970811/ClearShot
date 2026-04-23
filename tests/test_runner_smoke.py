"""Smoke test for :class:`EvaluationRunner` using a mock baseline.

Builds a tiny synthetic manifest, writes a handful of clean + degraded JPEGs,
registers a no-op ``identity`` baseline, then confirms:
    - outputs are written
    - per_image.csv is created and contains the expected columns
    - a second run is fully skipped (resume works)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.baselines import BaselineEnhancer  # noqa: E402
from src.evaluation.metrics import ImageQualityMetrics  # noqa: E402
from src.evaluation.runner import EvaluationConfig, EvaluationRunner  # noqa: E402


class _IdentityBaseline(BaselineEnhancer):
    name = "identity"

    def enhance(self, image: Image.Image) -> Image.Image:
        return image if image.mode == "RGB" else image.convert("RGB")


@pytest.fixture
def tiny_manifest(tmp_path: Path) -> Path:
    clean_dir = tmp_path / "clean"
    deg_dir = tmp_path / "deg"
    clean_dir.mkdir()
    deg_dir.mkdir()

    rows = []
    for i in range(3):
        cat = "cat_a" if i < 2 else "cat_b"
        image_id = f"img{i}"
        clean_p = clean_dir / f"{image_id}.jpg"
        deg_p = deg_dir / f"{image_id}.jpg"
        Image.new("RGB", (64, 64), (40 * i + 20, 80, 120)).save(clean_p)
        Image.new("RGB", (64, 64), (40 * i + 10, 70, 110)).save(deg_p)
        rows.append(
            {
                "image_id": image_id,
                "category": cat,
                "clean_path": str(clean_p),
                "degraded_path": str(deg_p),
            }
        )

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def test_runner_writes_outputs_and_is_resumable(
    tmp_path: Path, tiny_manifest: Path
) -> None:
    output_dir = tmp_path / "out"
    cfg = EvaluationConfig(
        manifest_path=str(tiny_manifest),
        output_dir=str(output_dir),
        subset_size=None,
        eval_resolution=64,
    )
    metrics = ImageQualityMetrics(device="cpu", eval_resolution=64)
    runner = EvaluationRunner(cfg, metrics=metrics, progress=False)

    result = runner.run_baseline(_IdentityBaseline())
    assert result.per_image_csv.exists()
    df = pd.read_csv(result.per_image_csv)
    assert len(df) == 3
    for col in ("image_id", "method", "psnr", "ssim", "lpips", "status"):
        assert col in df.columns
    assert (df["status"] == "ok").all()
    for image_id in df["image_id"]:
        assert (output_dir / "identity" / f"{image_id}.png").exists()

    result2 = runner.run_baseline(_IdentityBaseline())
    assert result2.n_new == 0
    assert result2.n_skipped == 3


def test_runner_builds_reference_directory(
    tmp_path: Path, tiny_manifest: Path
) -> None:
    output_dir = tmp_path / "out2"
    cfg = EvaluationConfig(
        manifest_path=str(tiny_manifest),
        output_dir=str(output_dir),
        subset_size=None,
        eval_resolution=64,
    )
    metrics = ImageQualityMetrics(device="cpu", eval_resolution=64)
    runner = EvaluationRunner(cfg, metrics=metrics, progress=False)
    ref_dir = runner.build_reference_dir()
    assert ref_dir.exists()
    assert sum(1 for _ in ref_dir.glob("*.png")) == 3
