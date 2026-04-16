"""
Synthetic degradation pipeline for simulating amateur product photography.

Applies randomized, composable degradations to clean product images to create
realistic training pairs (degraded -> clean) for the diffusion enhancement model.

Each degradation is applied independently with a configurable probability,
producing varied difficulty levels across the training set.
"""
import io
import random
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass
class DegradationConfig:
    """Configuration for the degradation pipeline."""
    # Probability of applying each degradation
    p_noise: float = 0.7
    p_blur: float = 0.5
    p_jpeg: float = 0.6
    p_color_jitter: float = 0.7
    p_vignette: float = 0.4
    p_uneven_exposure: float = 0.4
    p_background_clutter: float = 0.5
    p_shadow: float = 0.3
    p_downscale: float = 0.4

    # Degradation intensity ranges
    noise_sigma_range: tuple = (10, 50)
    blur_kernel_range: tuple = (3, 9)  # must be odd
    jpeg_quality_range: tuple = (15, 55)
    brightness_range: tuple = (0.6, 1.4)
    contrast_range: tuple = (0.6, 1.4)
    saturation_range: tuple = (0.5, 1.5)
    hue_shift_range: tuple = (-15, 15)
    downscale_range: tuple = (0.3, 0.7)

    # Reproducibility
    seed: Optional[int] = None


class DegradationPipeline:
    """
    Applies randomized degradations to simulate amateur product photos.

    Usage:
        pipeline = DegradationPipeline(DegradationConfig(seed=42))
        degraded, params = pipeline.apply(clean_image)
    """

    def __init__(self, config: Optional[DegradationConfig] = None):
        self.config = config or DegradationConfig()
        self.rng = random.Random(self.config.seed)
        self.np_rng = np.random.RandomState(self.config.seed)

    def apply(self, image: Image.Image) -> tuple[Image.Image, dict]:
        """
        Apply random degradations to a clean image.

        Returns:
            (degraded_image, degradation_params_dict)
        """
        img = image.copy().convert("RGB")
        params = {}

        # 1. Color jitter (brightness, contrast, saturation, hue)
        if self.rng.random() < self.config.p_color_jitter:
            img, p = self._color_jitter(img)
            params["color_jitter"] = p

        # 2. Uneven exposure
        if self.rng.random() < self.config.p_uneven_exposure:
            img, p = self._uneven_exposure(img)
            params["uneven_exposure"] = p

        # 3. Vignette
        if self.rng.random() < self.config.p_vignette:
            img, p = self._vignette(img)
            params["vignette"] = p

        # 4. Shadow overlay
        if self.rng.random() < self.config.p_shadow:
            img, p = self._random_shadow(img)
            params["shadow"] = p

        # 5. Background clutter (colored noise patches)
        if self.rng.random() < self.config.p_background_clutter:
            img, p = self._background_clutter(img)
            params["background_clutter"] = p

        # 6. Gaussian noise
        if self.rng.random() < self.config.p_noise:
            img, p = self._gaussian_noise(img)
            params["noise"] = p

        # 7. Gaussian blur
        if self.rng.random() < self.config.p_blur:
            img, p = self._gaussian_blur(img)
            params["blur"] = p

        # 8. Downscale and re-upscale (simulates low-res capture)
        if self.rng.random() < self.config.p_downscale:
            img, p = self._downscale_upscale(img)
            params["downscale"] = p

        # 9. JPEG compression (always last to simulate save artifacts)
        if self.rng.random() < self.config.p_jpeg:
            img, p = self._jpeg_compress(img)
            params["jpeg"] = p

        return img, params

    def _color_jitter(self, img: Image.Image) -> tuple[Image.Image, dict]:
        cfg = self.config
        brightness = self.rng.uniform(*cfg.brightness_range)
        contrast = self.rng.uniform(*cfg.contrast_range)
        saturation = self.rng.uniform(*cfg.saturation_range)

        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Color(img).enhance(saturation)

        # Hue shift via HSV
        hue_shift = self.rng.randint(*cfg.hue_shift_range)
        if hue_shift != 0:
            arr = np.array(img)
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.int16)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            hsv = hsv.astype(np.uint8)
            arr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            img = Image.fromarray(arr)

        return img, {"brightness": brightness, "contrast": contrast,
                      "saturation": saturation, "hue_shift": hue_shift}

    def _gaussian_noise(self, img: Image.Image) -> tuple[Image.Image, dict]:
        sigma = self.rng.uniform(*self.config.noise_sigma_range)
        arr = np.array(img, dtype=np.float32)
        noise = self.np_rng.normal(0, sigma, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr), {"sigma": sigma}

    def _gaussian_blur(self, img: Image.Image) -> tuple[Image.Image, dict]:
        lo, hi = self.config.blur_kernel_range
        kernel = self.rng.randrange(lo, hi + 1, 2)  # ensure odd
        arr = np.array(img)
        arr = cv2.GaussianBlur(arr, (kernel, kernel), 0)
        return Image.fromarray(arr), {"kernel": kernel}

    def _jpeg_compress(self, img: Image.Image) -> tuple[Image.Image, dict]:
        quality = self.rng.randint(*self.config.jpeg_quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB"), {"quality": quality}

    def _vignette(self, img: Image.Image) -> tuple[Image.Image, dict]:
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]
        strength = self.rng.uniform(0.3, 0.8)

        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r_max = np.sqrt(cx ** 2 + cy ** 2)
        mask = 1 - strength * (r / r_max) ** 2
        mask = np.clip(mask, 0, 1)

        arr = arr * mask[:, :, np.newaxis]
        return Image.fromarray(arr.astype(np.uint8)), {"strength": strength}

    def _uneven_exposure(self, img: Image.Image) -> tuple[Image.Image, dict]:
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Create a gradient in a random direction
        angle = self.rng.uniform(0, 2 * np.pi)
        intensity = self.rng.uniform(0.3, 0.7)

        y, x = np.mgrid[:h, :w].astype(np.float32)
        gradient = (np.cos(angle) * (x / w) + np.sin(angle) * (y / h))
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
        mask = 1 - intensity * gradient

        arr = arr * mask[:, :, np.newaxis]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), {
            "angle": angle, "intensity": intensity
        }

    def _random_shadow(self, img: Image.Image) -> tuple[Image.Image, dict]:
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Random elliptical shadow
        cx = self.rng.randint(0, w)
        cy = self.rng.randint(0, h)
        a = self.rng.randint(w // 4, w)
        b = self.rng.randint(h // 4, h)
        shadow_intensity = self.rng.uniform(0.3, 0.6)

        y, x = np.ogrid[:h, :w]
        mask = ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2
        shadow = np.where(mask <= 1, 1 - shadow_intensity * (1 - mask), 1.0)

        arr = arr * shadow[:, :, np.newaxis]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), {
            "cx": cx, "cy": cy, "intensity": shadow_intensity
        }

    def _background_clutter(self, img: Image.Image) -> tuple[Image.Image, dict]:
        """Add colored noise patches to simulate cluttered backgrounds."""
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]
        n_patches = self.rng.randint(2, 6)

        for _ in range(n_patches):
            px = self.rng.randint(0, max(1, w - w // 4))
            py = self.rng.randint(0, max(1, h - h // 4))
            pw = self.rng.randint(w // 8, w // 3)
            ph = self.rng.randint(h // 8, h // 3)
            color = [self.rng.randint(50, 200) for _ in range(3)]
            alpha = self.rng.uniform(0.1, 0.35)

            x1, y1 = px, py
            x2, y2 = min(px + pw, w), min(py + ph, h)
            patch = np.full((y2 - y1, x2 - x1, 3), color, dtype=np.float32)
            arr[y1:y2, x1:x2] = (1 - alpha) * arr[y1:y2, x1:x2] + alpha * patch

        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), {
            "n_patches": n_patches
        }

    def _downscale_upscale(self, img: Image.Image) -> tuple[Image.Image, dict]:
        factor = self.rng.uniform(*self.config.downscale_range)
        w, h = img.size
        small_w, small_h = max(32, int(w * factor)), max(32, int(h * factor))
        img = img.resize((small_w, small_h), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)
        return img, {"factor": factor}
