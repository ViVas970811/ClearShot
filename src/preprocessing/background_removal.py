"""
Background removal and studio background application for product images.
Uses rembg (pretrained U2-Net) for segmentation.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import cv2


class BackgroundRemover:
    """Removes backgrounds from product images and applies clean studio backgrounds."""

    def __init__(self, model_name: str = "u2net"):
        from rembg import new_session
        self.session = new_session(model_name)

    def remove_background(self, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Remove background from a product image.

        Args:
            image: PIL Image (RGB)

        Returns:
            Tuple of (product_rgba, mask)
            - product_rgba: RGBA image with transparent background
            - mask: Grayscale PIL Image (255=product, 0=background)
        """
        from rembg import remove
        product_rgba = remove(image, session=self.session)

        if product_rgba.mode == "RGBA":
            mask = product_rgba.split()[3]
        else:
            mask = Image.new("L", image.size, 255)

        return product_rgba, mask

    def apply_white_background(self, product_rgba: Image.Image, mask: Optional[Image.Image] = None) -> Image.Image:
        """Composite product onto a clean white background. Returns RGB."""
        if product_rgba.mode != "RGBA":
            product_rgba = product_rgba.convert("RGBA")

        white_bg = Image.new("RGBA", product_rgba.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(white_bg, product_rgba)
        return composited.convert("RGB")

    def apply_studio_background(
        self,
        product_rgba: Image.Image,
        mask: Optional[Image.Image] = None,
        bg_type: str = "white",
        shadow: bool = True,
    ) -> Image.Image:
        """
        Apply a studio background to the product.

        Args:
            product_rgba: RGBA image with transparent background
            mask: Optional mask (unused if product_rgba has alpha)
            bg_type: One of 'white', 'gradient', 'studio'
            shadow: Whether to add a soft drop shadow

        Returns:
            RGB image with studio background
        """
        if product_rgba.mode != "RGBA":
            product_rgba = product_rgba.convert("RGBA")

        w, h = product_rgba.size

        if bg_type == "gradient":
            bg_array = np.zeros((h, w, 4), dtype=np.uint8)
            for y in range(h):
                gray = int(255 - (y / h) * 30)
                bg_array[y, :] = [gray, gray, gray, 255]
            background = Image.fromarray(bg_array, "RGBA")

        elif bg_type == "studio":
            bg_array = np.zeros((h, w, 4), dtype=np.uint8)
            cy, cx = h // 2, w // 2
            max_dist = np.sqrt(cx ** 2 + cy ** 2)
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / max_dist
            gray = (250 - (dist * 40)).clip(210, 250).astype(np.uint8)
            bg_array[:, :, 0] = gray
            bg_array[:, :, 1] = gray
            bg_array[:, :, 2] = gray
            bg_array[:, :, 3] = 255
            background = Image.fromarray(bg_array, "RGBA")

        else:
            background = Image.new("RGBA", (w, h), (255, 255, 255, 255))

        # Soft drop shadow
        if shadow and product_rgba.mode == "RGBA":
            alpha = np.array(product_rgba.split()[3]).astype(np.float32)
            shadow_offset = max(3, int(min(w, h) * 0.01))
            shadow_array = np.zeros((h, w, 4), dtype=np.uint8)
            shifted = np.roll(np.roll(alpha, shadow_offset, axis=0), shadow_offset, axis=1)
            blurred = cv2.GaussianBlur(shifted, (21, 21), 7)
            shadow_array[:, :, 3] = (blurred * 0.15).clip(0, 255).astype(np.uint8)
            shadow_img = Image.fromarray(shadow_array, "RGBA")
            background = Image.alpha_composite(background, shadow_img)

        result = Image.alpha_composite(background, product_rgba)
        return result.convert("RGB")

    def batch_process(
        self,
        input_dir: str,
        output_dir: str,
        bg_type: str = "white",
        save_masks: bool = False,
    ) -> int:
        """
        Process all images in a directory.

        Returns:
            Number of images processed
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if save_masks:
            mask_path = output_path / "masks"
            mask_path.mkdir(exist_ok=True)

        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_files = [f for f in input_path.rglob("*") if f.suffix.lower() in extensions]

        count = 0
        for img_file in image_files:
            try:
                image = Image.open(img_file).convert("RGB")
                product_rgba, mask = self.remove_background(image)
                result = self.apply_studio_background(product_rgba, mask, bg_type=bg_type)

                rel_path = img_file.relative_to(input_path)
                out_file = output_path / rel_path.with_suffix(".png")
                out_file.parent.mkdir(parents=True, exist_ok=True)
                result.save(out_file)

                if save_masks:
                    mask.save(mask_path / rel_path.with_suffix(".png"))

                count += 1
            except Exception as e:
                print(f"Failed to process {img_file.name}: {e}")

        print(f"Processed {count}/{len(image_files)} images -> {output_dir}")
        return count
