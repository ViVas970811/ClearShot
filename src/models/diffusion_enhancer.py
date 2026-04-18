"""
Diffusion-based image enhancement using Stable Diffusion 1.5 + ControlNet + LoRA.
Uses img2img pipeline with Canny edge conditioning to enhance product photos
while preserving structural identity.
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any
from pathlib import Path


class DiffusionEnhancer:
    """SD 1.5 + ControlNet img2img pipeline with optional LoRA weights."""

    def __init__(
        self,
        base_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet_model: str = "lllyasviel/control_v11p_sd15_canny",
        lora_weights_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        from diffusers import (
            StableDiffusionControlNetImg2ImgPipeline,
            ControlNetModel,
            UniPCMultistepScheduler,
        )

        self.device = device
        self.dtype = dtype

        controlnet = ControlNetModel.from_pretrained(
            controlnet_model, torch_dtype=dtype
        )

        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        self.pipe.to(device)

        # Memory optimizations: try xformers, fall back to attention slicing
        if device == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except (ModuleNotFoundError, ImportError):
                self.pipe.enable_attention_slicing()

        if lora_weights_path is not None:
            self.load_lora(lora_weights_path)

        self.lora_loaded = lora_weights_path is not None

    def load_lora(self, weights_path: str):
        weights_path = Path(weights_path)
        if weights_path.is_dir():
            self.pipe.load_lora_weights(str(weights_path))
        else:
            self.pipe.load_lora_weights(
                str(weights_path.parent), weight_name=weights_path.name
            )
        self.lora_loaded = True

    def load_lora_peft(self, weights_path: str):
        """Load LoRA weights saved in peft format (adapter_config.json + adapter_model.safetensors).

        Use this method for weights produced by Phase 3 training (src/training/train_lora.py),
        which saves via peft's save_pretrained(). The standard load_lora() method uses
        pipe.load_lora_weights() which expects diffusers-format keys and will silently
        fail to load peft-format weights.

        The wrapped model is cast to both the target device AND dtype. Without the dtype
        cast, fp16 base UNet + fp32 PEFT adapters produce a runtime mismatch on GPU
        (works on CPU because both default to fp32).

        Args:
            weights_path: Path to directory containing adapter_config.json
                          and adapter_model.safetensors
        """
        from peft import PeftModel

        self.pipe.unet = PeftModel.from_pretrained(
            self.pipe.unet, weights_path
        )
        # Cast both device AND dtype; PEFT adapters default to fp32 even when the
        # base UNet is fp16, which would cause a dtype mismatch at inference on GPU.
        self.pipe.unet.to(self.device, dtype=self.dtype)
        self.lora_loaded = True

    def enhance(
        self,
        image: Image.Image,
        control_image: Image.Image,
        prompt: str = "professional product photography, studio lighting, clean white background, high quality, 4k, detailed",
        negative_prompt: str = "blurry, noisy, low quality, distorted, deformed, watermark, text",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.8,
        strength: float = 0.45,
        seed: Optional[int] = None,
    ) -> Image.Image:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        if image.mode != "RGB":
            image = image.convert("RGB")
        if control_image.mode != "RGB":
            control_image = control_image.convert("RGB")

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            control_image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            strength=strength,
            generator=generator,
        )

        return result.images[0]

    def enhance_batch(
        self,
        images: list,
        control_images: list,
        prompt: str = "professional product photography, studio lighting, clean white background, high quality, 4k, detailed",
        negative_prompt: str = "blurry, noisy, low quality, distorted, deformed, watermark, text",
        **kwargs,
    ) -> list:
        results = []
        for img, ctrl in zip(images, control_images):
            enhanced = self.enhance(
                img, ctrl,
                prompt=prompt,
                negative_prompt=negative_prompt,
                **kwargs,
            )
            results.append(enhanced)
        return results

    def get_pipeline_config(self) -> Dict[str, Any]:
        return {
            "base_model": self.pipe.config._name_or_path if hasattr(self.pipe.config, "_name_or_path") else "unknown",
            "device": str(self.device),
            "dtype": str(self.dtype),
            "lora_loaded": self.lora_loaded,
            "scheduler": self.pipe.scheduler.__class__.__name__,
        }
