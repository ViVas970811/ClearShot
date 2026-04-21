"""Smoke tests for the three classical baselines.

The diffusion baselines (``sd_no_lora``, ``clearshot``) are intentionally not
tested here because they require the ClearShot pipeline, a GPU, and the
trained LoRA adapter. Their I/O contract is covered indirectly by the runner
tests where a mock baseline is used.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.baselines import (  # noqa: E402
    BaselineEnhancer,
    OpenCVBaseline,
    PILBaseline,
    build_baseline,
)


@pytest.mark.parametrize("baseline_cls", [OpenCVBaseline, PILBaseline])
def test_classical_baseline_returns_same_size_rgb(baseline_cls) -> None:
    img = Image.new("RGB", (128, 128), (120, 120, 120))
    baseline: BaselineEnhancer = baseline_cls()
    out = baseline.enhance(img)
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"
    assert out.size == img.size


def test_build_baseline_factory_known_names() -> None:
    assert build_baseline("opencv").name == "opencv"
    assert build_baseline("pil_auto").name == "pil_auto"


def test_build_baseline_factory_unknown_name_raises() -> None:
    with pytest.raises(ValueError):
        build_baseline("does_not_exist")
