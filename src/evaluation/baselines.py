"""
Baseline enhancement methods for Phase 5 comparisons.

All baselines expose the same minimal interface:

    baseline.name  -> str (used for output directory naming)
    baseline.enhance(pil_image: Image.Image) -> Image.Image

so the :mod:`src.evaluation.runner` treats them uniformly. Classical baselines
(OpenCV, PIL, background-only) are lightweight and CPU-only. The diffusion
baselines wrap the existing :class:`ClearShotPipeline` and therefore require
a GPU + the LoRA adapter (for the full baseline).

Baselines defined here (mapping to the Phase 5 plan):
    1. :class:`OpenCVBaseline`         - classical enhancement (CLAHE + bilateral + unsharp)
    2. :class:`PILBaseline`            - PIL auto-enhance (autocontrast + tone tweaks)
    3. :class:`BackgroundOnlyBaseline` - rembg + white background, no diffusion
    4. :class:`SDNoLoRABaseline`       - ClearShot pipeline w/o LoRA (ablation)
    5. :class:`ClearShotBaseline`      - full ClearShot pipeline with LoRA

The diffusion wrappers deliberately *reuse* ``ClearShotPipeline`` rather than
reconstructing SD/ControlNet calls, which keeps Phase 4 intermediates (mask,
edges, etc.) consistent with the Phase 5 evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


def _patch_numba_cache_for_rembg() -> None:
    """Work around rembg/pymatting import failures in this local Python 3.12 env."""
    import numba

    if getattr(numba, "_clearshot_eval_rembg_cache_patch", False):
        return

    original_njit = numba.njit

    def patched_njit(*args, **kwargs):
        kwargs["cache"] = False
        return original_njit(*args, **kwargs)

    numba.njit = patched_njit
    numba._clearshot_eval_rembg_cache_patch = True


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaselineEnhancer(ABC):
    """Abstract base class for a single enhancement baseline."""

    name: str = "baseline"

    @abstractmethod
    def enhance(self, image: Image.Image) -> Image.Image:
        """Enhance a single PIL RGB image and return an RGB PIL image."""

    def get_config(self) -> Dict[str, Any]:
        """Return a small dict describing this baseline (logged by the runner)."""
        return {"name": self.name, "class": self.__class__.__name__}


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------


class OpenCVBaseline(BaselineEnhancer):
    """Classical CV enhancement: CLAHE on luminance + bilateral filter + unsharp mask.

    Implements the ``OpenCV Enhancement`` baseline from the Phase 5 plan. No
    learned model and no background replacement, so PSNR/SSIM against the clean
    target reflect purely local contrast/denoise gains.
    """

    name = "opencv"

    def __init__(
        self,
        clahe_clip: float = 2.0,
        clahe_tile: int = 8,
        bilateral_d: int = 7,
        bilateral_sigma_color: int = 50,
        bilateral_sigma_space: int = 50,
        unsharp_amount: float = 0.6,
    ) -> None:
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.unsharp_amount = unsharp_amount

    def enhance(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        rgb = np.array(image)

        # CLAHE on the Y (luminance) channel to lift contrast without shifting hue.
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=(self.clahe_tile, self.clahe_tile),
        )
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

        # Edge-preserving denoise.
        rgb = cv2.bilateralFilter(
            rgb,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space,
        )

        # Unsharp mask (add back a scaled high-frequency residual).
        blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=1.5)
        sharpened = cv2.addWeighted(
            rgb, 1.0 + self.unsharp_amount, blurred, -self.unsharp_amount, 0
        )
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        return Image.fromarray(sharpened)


class PILBaseline(BaselineEnhancer):
    """PIL-only auto-enhance: autocontrast + color/brightness/contrast/sharpness tweaks.

    Uses fixed multiplicative factors for the four ``ImageEnhance`` axes rather
    than ``ImageOps.equalize`` because equalization tends to blow out product
    highlights on plain backgrounds.
    """

    name = "pil_auto"

    def __init__(
        self,
        autocontrast_cutoff: float = 1.0,
        brightness: float = 1.05,
        contrast: float = 1.15,
        color: float = 1.10,
        sharpness: float = 1.40,
    ) -> None:
        self.autocontrast_cutoff = autocontrast_cutoff
        self.brightness = brightness
        self.contrast = contrast
        self.color = color
        self.sharpness = sharpness

    def enhance(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        out = ImageOps.autocontrast(image, cutoff=self.autocontrast_cutoff)
        out = ImageEnhance.Brightness(out).enhance(self.brightness)
        out = ImageEnhance.Contrast(out).enhance(self.contrast)
        out = ImageEnhance.Color(out).enhance(self.color)
        out = ImageEnhance.Sharpness(out).enhance(self.sharpness)
        return out


class BackgroundOnlyBaseline(BaselineEnhancer):
    """Background removal + white studio background, no diffusion.

    Ablates the generative step entirely: shows how much of the final quality
    comes from segmentation + clean-background composition alone.
    """

    name = "background_only"

    def __init__(self, bg_type: str = "white", shadow: bool = True, model_name: str = "u2net") -> None:
        _patch_numba_cache_for_rembg()
        from ..preprocessing.background_removal import BackgroundRemover

        self._remover = BackgroundRemover(model_name=model_name)
        self.bg_type = bg_type
        self.shadow = shadow

    def enhance(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        product_rgba, mask = self._remover.remove_background(image)
        return self._remover.apply_studio_background(
            product_rgba, mask, bg_type=self.bg_type, shadow=self.shadow
        )


# ---------------------------------------------------------------------------
# Diffusion baselines (wrap ClearShotPipeline)
# ---------------------------------------------------------------------------


class _PipelineBackedBaseline(BaselineEnhancer):
    """Shared machinery for baselines that wrap ``ClearShotPipeline``.

    Subclasses only differ in whether a LoRA path is passed through.
    """

    def __init__(
        self,
        config_path: str = "configs/inference_config.yaml",
        lora_weights_path: Optional[str] = None,
        device: Optional[str] = None,
        enable_sr: Optional[bool] = None,
        bg_type: Optional[str] = None,
        seed: Optional[int] = 42,
    ) -> None:
        _patch_numba_cache_for_rembg()
        from ..pipeline.enhancement_pipeline import ClearShotPipeline

        self._pipeline = ClearShotPipeline(
            config_path=config_path,
            lora_weights_path=lora_weights_path,
            device=device,
        )
        self.enable_sr = enable_sr
        self.bg_type = bg_type
        self.seed = seed

    def enhance(self, image: Image.Image) -> Image.Image:
        result = self._pipeline.enhance(
            image,
            bg_type=self.bg_type,
            enable_sr=self.enable_sr,
            seed=self.seed,
        )
        return result.final

    def enhance_full(self, image: Image.Image) -> Any:
        """Run the pipeline and return the full EnhancementResult (intermediates).

        Useful for the visual comparison grid which wants to show the mask/edges
        that produced the final image.
        """
        return self._pipeline.enhance(
            image,
            bg_type=self.bg_type,
            enable_sr=self.enable_sr,
            seed=self.seed,
        )

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update(self._pipeline.get_config())
        return cfg


class SDNoLoRABaseline(_PipelineBackedBaseline):
    """Full ClearShot pipeline *without* the trained LoRA adapter.

    This is the ablation described in the Phase 2-3 handoff ("with vs without
    LoRA") and listed as Baseline 4 in the Phase 5 plan.
    """

    name = "sd_no_lora"

    def __init__(
        self,
        config_path: str = "configs/inference_config.yaml",
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            config_path=config_path,
            lora_weights_path=None,
            device=device,
            **kwargs,
        )


class ClearShotBaseline(_PipelineBackedBaseline):
    """Full ClearShot pipeline with trained LoRA adapter (reference method)."""

    name = "clearshot"

    def __init__(
        self,
        lora_weights_path: str,
        config_path: str = "configs/inference_config.yaml",
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        lp = Path(lora_weights_path)
        if not lp.exists():
            raise FileNotFoundError(
                f"LoRA weights directory does not exist: {lora_weights_path}. "
                "Download ClearShot_checkpoints/final/ from the project Google Drive first."
            )
        super().__init__(
            config_path=config_path,
            lora_weights_path=str(lp),
            device=device,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_baseline(name: str, **kwargs: Any) -> BaselineEnhancer:
    """Instantiate a baseline by its short name.

    Known names (case-insensitive):
        ``opencv``, ``pil_auto``, ``background_only``, ``sd_no_lora``, ``clearshot``.
    """
    key = name.lower().strip()
    if key == "opencv":
        return OpenCVBaseline(**kwargs)
    if key == "pil_auto":
        return PILBaseline(**kwargs)
    if key == "background_only":
        return BackgroundOnlyBaseline(**kwargs)
    if key == "sd_no_lora":
        return SDNoLoRABaseline(**kwargs)
    if key == "clearshot":
        return ClearShotBaseline(**kwargs)
    raise ValueError(
        f"Unknown baseline '{name}'. Valid options: opencv, pil_auto, "
        f"background_only, sd_no_lora, clearshot."
    )
