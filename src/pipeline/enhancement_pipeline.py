"""
End-to-end ClearShot enhancement pipeline.

Orchestrates all five stages of the product photo enhancement process:
    1. Background removal (rembg / U2-Net)
    2. Structural edge extraction (Canny / HED)
    3. Diffusion-based enhancement (SD 1.5 + ControlNet + LoRA)
    4. Background refinement (clean studio background)
    5. Super-resolution upscaling (Real-ESRGAN 2x)

Usage:
    pipeline = ClearShotPipeline(lora_weights_path="checkpoints/final")
    result = pipeline.enhance("path/to/product_photo.jpg")
    result.final.save("enhanced_output.png")
"""

import time
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from PIL import Image
from tqdm import tqdm


@dataclass
class EnhancementResult:
    """Holds all intermediate and final outputs from the enhancement pipeline.

    Attributes:
        original: Input image as received
        product_rgba: Product isolated on transparent background (RGBA)
        mask: Binary segmentation mask (grayscale, 255=product)
        edge_map: Structural edge map used for ControlNet conditioning (RGB)
        diffusion_output: Raw output from SD + ControlNet + LoRA (RGB, 512x512)
        with_background: Diffusion output with clean studio background applied (RGB)
        final: Super-resolution upscaled final output (RGB, 1024x1024)
        metadata: Dict with timing info, config values, and pipeline state
    """
    original: Image.Image
    product_rgba: Optional[Image.Image] = None
    mask: Optional[Image.Image] = None
    edge_map: Optional[Image.Image] = None
    diffusion_output: Optional[Image.Image] = None
    with_background: Optional[Image.Image] = None
    final: Optional[Image.Image] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ClearShotPipeline:
    """End-to-end product photo enhancement pipeline.

    Chains five stages: bg removal -> edge extraction -> diffusion enhancement
    -> background refinement -> super-resolution.

    All parameters are driven by configs/inference_config.yaml unless overridden.
    Sub-modules are lazily initialized to avoid loading GPU models until needed.

    Usage:
        # Single image
        pipeline = ClearShotPipeline(lora_weights_path="checkpoints/final")
        result = pipeline.enhance("path/to/photo.jpg")
        pipeline.save_result(result, "output/")

        # Batch processing
        results = pipeline.batch_enhance("input_dir/", "output_dir/")
    """

    def __init__(
        self,
        config_path: str = "configs/inference_config.yaml",
        lora_weights_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            config_path: Path to inference configuration YAML file
            lora_weights_path: Path to directory containing trained LoRA weights
                               (adapter_config.json + adapter_model.safetensors).
                               If None, runs without LoRA (baseline SD + ControlNet).
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        import torch

        self.config = self._load_config(config_path)
        self.lora_weights_path = lora_weights_path

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Lazy-initialized sub-modules (loaded on first use)
        self._bg_remover = None
        self._edge_extractor = None
        self._diffusion_enhancer = None
        self._super_resolver = None

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Load inference config from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            print(f"Warning: Config file {config_path} not found, using defaults.")
            return {}
        with open(config_file, "r") as f:
            return yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Lazy sub-module initialization
    # ------------------------------------------------------------------

    @property
    def bg_remover(self):
        """Background removal module (lazy-loaded)."""
        if self._bg_remover is None:
            from ..preprocessing.background_removal import BackgroundRemover

            bg_cfg = self.config.get("preprocessing", {}).get("background_removal", {})
            model_name = bg_cfg.get("model", "u2net")
            self._bg_remover = BackgroundRemover(model_name=model_name)
            print(f"[ClearShot] BackgroundRemover loaded (model={model_name})")

        return self._bg_remover

    @property
    def edge_extractor(self):
        """Structural edge extractor (lazy-loaded)."""
        if self._edge_extractor is None:
            from ..preprocessing.edge_extraction import StructuralExtractor

            edge_cfg = self.config.get("preprocessing", {}).get("edge_extraction", {})
            method = edge_cfg.get("method", "canny")
            low = edge_cfg.get("low_threshold", 100)
            high = edge_cfg.get("high_threshold", 200)
            self._edge_extractor = StructuralExtractor(
                method=method, low_threshold=low, high_threshold=high
            )
            print(f"[ClearShot] StructuralExtractor loaded (method={method})")

        return self._edge_extractor

    @property
    def diffusion_enhancer(self):
        """Diffusion enhancement model (lazy-loaded, heaviest component)."""
        if self._diffusion_enhancer is None:
            from ..models.diffusion_enhancer import DiffusionEnhancer

            model_cfg = self.config.get("pipeline", {})
            import torch
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            self._diffusion_enhancer = DiffusionEnhancer(
                device=self.device,
                dtype=dtype,
            )

            # Load LoRA weights via peft (correct format for Phase 3 weights)
            if self.lora_weights_path is not None:
                lora_path = Path(self.lora_weights_path)
                if lora_path.exists():
                    self._diffusion_enhancer.load_lora_peft(str(lora_path))
                    print(f"[ClearShot] LoRA weights loaded from {lora_path}")
                else:
                    print(f"[ClearShot] WARNING: LoRA path not found: {lora_path}")

            print(f"[ClearShot] DiffusionEnhancer loaded (device={self.device})")

        return self._diffusion_enhancer

    @property
    def super_resolver(self):
        """Super-resolution upscaler (lazy-loaded)."""
        if self._super_resolver is None:
            from ..models.super_resolution import SuperResolver

            sr_cfg = self.config.get("super_resolution", {})
            model_name = sr_cfg.get("model", "RealESRGAN_x2plus")
            scale = sr_cfg.get("scale", 2)
            self._super_resolver = SuperResolver(
                model_name=model_name, scale=scale, device=self.device
            )
            print(f"[ClearShot] SuperResolver loaded (model={model_name}, scale={scale}x)")

        return self._super_resolver

    # ------------------------------------------------------------------
    # Core enhancement
    # ------------------------------------------------------------------

    def enhance(
        self,
        image,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        bg_type: Optional[str] = None,
        enable_sr: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> EnhancementResult:
        """
        Enhance a single product image through the full pipeline.

        Args:
            image: PIL Image (RGB) or path to image file.
            prompt: Text prompt for diffusion model. Uses config default if None.
            negative_prompt: Negative prompt. Uses config default if None.
            bg_type: Background type ('white', 'gradient', 'studio'). Uses config default if None.
            enable_sr: Whether to run super-resolution. Uses config default if None.
            seed: Random seed for diffusion model. Uses config default if None.

        Returns:
            EnhancementResult with all intermediate and final outputs.
        """
        t_start = time.time()
        timings = {}

        # --- Load image if path ---
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preserve the caller's original image in the result before any resizing.
        result = EnhancementResult(original=image.copy())

        # Normalize to 512x512 for the diffusion pipeline. SD 1.5 was trained at
        # 512x512 and the LoRA adapter from Phase 3 was fine-tuned at 512x512, so
        # running inference at the training resolution yields the best quality AND
        # guarantees a deterministic final output size (1024x1024 after 2x SR).
        # Without this step, a 256x256 input would produce a 512x512 final, which
        # violates the pipeline's advertised output size.
        PIPELINE_RES = 512
        if image.size != (PIPELINE_RES, PIPELINE_RES):
            image = image.resize((PIPELINE_RES, PIPELINE_RES), Image.BILINEAR)

        # --- Read config defaults ---
        pipe_cfg = self.config.get("pipeline", {})
        prompt_cfg = self.config.get("prompt", {})
        bg_cfg = self.config.get("background", {})
        sr_cfg = self.config.get("super_resolution", {})

        prompt = prompt or prompt_cfg.get(
            "template",
            "professional product photography, studio lighting, clean white background, high quality, 4k, detailed",
        )
        negative_prompt = negative_prompt or prompt_cfg.get(
            "negative",
            "blurry, noisy, low quality, distorted, deformed, watermark, text",
        )
        bg_type = bg_type or bg_cfg.get("mode", "white")
        enable_sr = enable_sr if enable_sr is not None else sr_cfg.get("enabled", True)
        seed = seed if seed is not None else pipe_cfg.get("seed", None)

        # --- Stage 1: Background Removal ---
        t0 = time.time()
        product_rgba, mask = self.bg_remover.remove_background(image)
        result.product_rgba = product_rgba
        result.mask = mask
        timings["bg_removal"] = time.time() - t0
        print(f"  [1/5] Background removal: {timings['bg_removal']:.2f}s")

        # --- Stage 2: Edge Extraction ---
        t0 = time.time()
        edge_map = self.edge_extractor.extract_for_controlnet(image)
        result.edge_map = edge_map
        timings["edge_extraction"] = time.time() - t0
        print(f"  [2/5] Edge extraction: {timings['edge_extraction']:.2f}s")

        # --- Stage 3: Diffusion Enhancement ---
        t0 = time.time()
        diffusion_output = self.diffusion_enhancer.enhance(
            image=image,
            control_image=edge_map,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=pipe_cfg.get("num_inference_steps", 30),
            guidance_scale=pipe_cfg.get("guidance_scale", 7.5),
            controlnet_conditioning_scale=pipe_cfg.get("controlnet_conditioning_scale", 0.8),
            strength=pipe_cfg.get("strength", 0.45),
            seed=seed,
        )
        result.diffusion_output = diffusion_output
        timings["diffusion"] = time.time() - t0
        print(f"  [3/5] Diffusion enhancement: {timings['diffusion']:.2f}s")

        # --- Stage 4: Background Refinement ---
        t0 = time.time()
        # Use the mask from stage 1 to isolate product from diffusion output,
        # then apply a clean studio background
        diff_rgba = self._apply_mask_to_image(diffusion_output, mask)
        with_background = self.bg_remover.apply_studio_background(
            diff_rgba, mask, bg_type=bg_type, shadow=True
        )
        result.with_background = with_background
        timings["bg_refinement"] = time.time() - t0
        print(f"  [4/5] Background refinement: {timings['bg_refinement']:.2f}s")

        # --- Stage 5: Super-Resolution ---
        if enable_sr:
            t0 = time.time()
            final = self.super_resolver.upscale(with_background)
            result.final = final
            timings["super_resolution"] = time.time() - t0
            print(f"  [5/5] Super-resolution: {timings['super_resolution']:.2f}s")
        else:
            result.final = with_background
            print(f"  [5/5] Super-resolution: skipped")

        # --- Metadata ---
        total_time = time.time() - t_start
        result.metadata = {
            "timings": timings,
            "total_time_seconds": total_time,
            "input_size": image.size,
            "output_size": result.final.size,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "bg_type": bg_type,
            "sr_enabled": enable_sr,
            "seed": seed,
            "lora_loaded": self.diffusion_enhancer.lora_loaded,
            "device": self.device,
        }
        print(f"  Total: {total_time:.2f}s | {image.size} -> {result.final.size}")

        return result

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def batch_enhance(
        self,
        input_dir: str,
        output_dir: str,
        save_intermediates: bool = False,
        max_images: Optional[int] = None,
        **kwargs,
    ) -> List[EnhancementResult]:
        """
        Enhance all images in a directory.

        Args:
            input_dir: Path to directory containing input images
            output_dir: Path to save enhanced outputs
            save_intermediates: Whether to save intermediate stage outputs
            max_images: Maximum number of images to process (for quick testing)
            **kwargs: Additional arguments passed to enhance()

        Returns:
            List of EnhancementResult for each processed image
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_files = sorted([
            f for f in input_path.rglob("*")
            if f.suffix.lower() in extensions
        ])

        if max_images is not None:
            image_files = image_files[:max_images]

        print(f"\n{'='*60}")
        print(f"ClearShot Batch Enhancement")
        print(f"  Input:  {input_dir} ({len(image_files)} images)")
        print(f"  Output: {output_dir}")
        print(f"{'='*60}\n")

        results = []
        for i, img_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {img_file.name}")
            try:
                result = self.enhance(img_file, **kwargs)
                self.save_result(
                    result, output_path,
                    filename=img_file.stem,
                    save_intermediates=save_intermediates,
                )
                results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")

        print(f"\n{'='*60}")
        print(f"Batch complete: {len(results)}/{len(image_files)} images enhanced")
        print(f"{'='*60}")

        return results

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    def save_result(
        self,
        result: EnhancementResult,
        output_dir: str,
        filename: str = "enhanced",
        save_intermediates: bool = True,
    ):
        """
        Save enhancement result to disk.

        Args:
            result: EnhancementResult from enhance()
            output_dir: Directory to save files
            filename: Base filename (without extension)
            save_intermediates: Whether to save intermediate stage outputs
        """
        out_cfg = self.config.get("output", {})
        fmt = out_cfg.get("format", "png")
        quality = out_cfg.get("quality", 95)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Always save the final output
        final_path = output_path / f"{filename}_final.{fmt}"
        if fmt.lower() == "jpg" or fmt.lower() == "jpeg":
            result.final.save(final_path, "JPEG", quality=quality)
        else:
            result.final.save(final_path, "PNG")
        print(f"  Saved: {final_path}")

        # Save intermediates if requested
        save_inter = save_intermediates or out_cfg.get("save_intermediates", False)
        if save_inter:
            inter_dir = output_path / f"{filename}_stages"
            inter_dir.mkdir(parents=True, exist_ok=True)

            if result.original is not None:
                result.original.save(inter_dir / "0_original.png")

            if result.mask is not None:
                result.mask.save(inter_dir / "1_mask.png")

            if result.edge_map is not None:
                result.edge_map.save(inter_dir / "2_edges.png")

            if result.diffusion_output is not None:
                result.diffusion_output.save(inter_dir / "3_diffusion.png")

            if result.with_background is not None:
                result.with_background.save(inter_dir / "4_with_bg.png")

            if result.final is not None:
                result.final.save(inter_dir / "5_final.png")

            print(f"  Intermediates saved to: {inter_dir}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_mask_to_image(image: Image.Image, mask: Image.Image) -> Image.Image:
        """Apply segmentation mask to create RGBA image with transparent background.

        Args:
            image: RGB image (e.g., diffusion output)
            mask: Grayscale mask (255=product, 0=background)

        Returns:
            RGBA image with background made transparent
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Ensure mask matches image size
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.BILINEAR)

        if mask.mode != "L":
            mask = mask.convert("L")

        # Create RGBA by combining RGB + mask as alpha
        rgba = image.copy().convert("RGBA")
        rgba.putalpha(mask)
        return rgba

    def get_config(self) -> Dict[str, Any]:
        """Return the full pipeline configuration."""
        return {
            "inference_config": self.config,
            "lora_weights_path": str(self.lora_weights_path) if self.lora_weights_path else None,
            "device": self.device,
            "modules_loaded": {
                "bg_remover": self._bg_remover is not None,
                "edge_extractor": self._edge_extractor is not None,
                "diffusion_enhancer": self._diffusion_enhancer is not None,
                "super_resolver": self._super_resolver is not None,
            },
        }
