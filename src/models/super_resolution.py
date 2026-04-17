"""
Super-resolution upscaling using Real-ESRGAN.

Wraps the realesrgan package with pretrained RealESRGAN_x2plus weights
for 2x upscaling (512 -> 1024). No training required.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, List


class SuperResolver:
    """Real-ESRGAN based super-resolution upscaler for product images.

    Usage:
        resolver = SuperResolver()
        high_res = resolver.upscale(low_res_image)  # 512x512 -> 1024x1024
    """

    def __init__(
        self,
        model_name: str = "RealESRGAN_x2plus",
        scale: int = 2,
        device: Optional[str] = None,
        half: bool = True,
    ):
        """
        Initialize the super-resolution model.

        Args:
            model_name: Model weights to use. Options:
                - 'RealESRGAN_x2plus' (default, good quality/speed tradeoff)
                - 'RealESRGAN_x4plus' (4x upscale, slower)
            scale: Upscale factor (must match model_name)
            device: 'cuda' or 'cpu'. Auto-detected if None.
            half: Use FP16 for faster inference (CUDA only)
        """
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.scale = scale
        self.model_name = model_name
        self._upsampler = None
        self._half = half and (device == "cuda")

    def _get_upsampler(self):
        """Lazy-load the Real-ESRGAN model on first use."""
        if self._upsampler is not None:
            return self._upsampler

        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        # Configure architecture based on model name
        if self.model_name == "RealESRGAN_x2plus":
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=2,
            )
            netscale = 2
        elif self.model_name == "RealESRGAN_x4plus":
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4,
            )
            netscale = 4
        else:
            raise ValueError(
                f"Unknown model: {self.model_name}. "
                f"Use 'RealESRGAN_x2plus' or 'RealESRGAN_x4plus'."
            )

        # RealESRGANer handles model download automatically
        self._upsampler = RealESRGANer(
            scale=netscale,
            model_path=None,  # auto-download from GitHub releases
            model=model,
            tile=0,  # no tiling (full image at once)
            tile_pad=10,
            pre_pad=0,
            half=self._half,
            device=self.device,
        )

        return self._upsampler

    def upscale(self, image: Image.Image) -> Image.Image:
        """
        Upscale a single image.

        Args:
            image: PIL Image (RGB), typically 512x512

        Returns:
            PIL Image (RGB) at upscaled resolution (e.g., 1024x1024 for 2x)
        """
        import cv2

        upsampler = self._get_upsampler()

        # PIL (RGB) -> numpy (BGR) for OpenCV/Real-ESRGAN
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Upscale
        output_bgr, _ = upsampler.enhance(img_bgr, outscale=self.scale)

        # BGR -> RGB -> PIL
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(output_rgb)

    def upscale_batch(
        self, images: List[Image.Image], show_progress: bool = True
    ) -> List[Image.Image]:
        """
        Upscale a list of images.

        Args:
            images: List of PIL Images
            show_progress: Whether to show a tqdm progress bar

        Returns:
            List of upscaled PIL Images
        """
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(images, desc="Super-resolution")
        else:
            iterator = images

        return [self.upscale(img) for img in iterator]

    def get_config(self) -> dict:
        """Return current configuration."""
        return {
            "model_name": self.model_name,
            "scale": self.scale,
            "device": self.device,
            "half_precision": self._half,
        }
