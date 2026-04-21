"""ClearShot Phase 5 evaluation package.

Public API:
    ImageQualityMetrics        - paired + distributional quality metrics
    PairwiseMetrics            - dataclass for a single (pred, target) result
    AggregateMetrics           - dataclass for aggregated summaries
    aggregate_rows             - overall / per-category aggregation helper
    paired_ttests              - paired two-sided t-tests per metric
    BaselineEnhancer           - abstract interface for a baseline
    OpenCVBaseline             - CLAHE + bilateral + unsharp mask
    PILBaseline                - autocontrast + tone tweaks
    BackgroundOnlyBaseline     - rembg + studio background, no diffusion
    SDNoLoRABaseline           - ClearShot pipeline without LoRA
    ClearShotBaseline          - full ClearShot pipeline with LoRA
    build_baseline             - baseline factory
    EvaluationConfig           - runner configuration dataclass
    EvaluationRunner           - resumable orchestrator
    EvaluationResult           - per-baseline run summary
"""

from .metrics import (
    AggregateMetrics,
    DEFAULT_METRIC_KEYS,
    ImageQualityMetrics,
    PairwiseMetrics,
    aggregate_rows,
    paired_ttests,
)
from .baselines import (
    BackgroundOnlyBaseline,
    BaselineEnhancer,
    ClearShotBaseline,
    OpenCVBaseline,
    PILBaseline,
    SDNoLoRABaseline,
    build_baseline,
)
from .runner import EvaluationConfig, EvaluationResult, EvaluationRunner

__all__ = [
    "AggregateMetrics",
    "DEFAULT_METRIC_KEYS",
    "ImageQualityMetrics",
    "PairwiseMetrics",
    "aggregate_rows",
    "paired_ttests",
    "BackgroundOnlyBaseline",
    "BaselineEnhancer",
    "ClearShotBaseline",
    "OpenCVBaseline",
    "PILBaseline",
    "SDNoLoRABaseline",
    "build_baseline",
    "EvaluationConfig",
    "EvaluationResult",
    "EvaluationRunner",
]
