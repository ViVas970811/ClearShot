"""
Structural map extraction for ControlNet conditioning.
Supports Canny edges (primary) and HED soft edges (secondary).
"""

import numpy as np
from PIL import Image
from typing import Optional
import cv2


class StructuralExtractor:
    """Extracts structural maps from product images for ControlNet conditioning."""

    def __init__(self, method: str = "canny", low_threshold: int = 100, high_threshold: int = 200):
        """
        Args:
            method: 'canny' (fast, deterministic) or 'hed' (soft edges via controlnet_aux)
            low_threshold: Canny lower threshold
            high_threshold: Canny upper threshold
        """
        self.method = method
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self._hed_detector = None

    def _get_hed_detector(self):
        """Lazy-load HED detector only when needed."""
        if self._hed_detector is None:
            from controlnet_aux import HEDdetector
            self._hed_detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
        return self._hed_detector

    def extract_canny(self, image: Image.Image) -> Image.Image:
        """
        Extract Canny edge map.

        Args:
            image: PIL Image (RGB)

        Returns:
            PIL Image (grayscale) with Canny edges
        """
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        return Image.fromarray(edges, "L")

    def extract_hed(self, image: Image.Image) -> Image.Image:
        """
        Extract HED soft edge map using controlnet_aux.

        Args:
            image: PIL Image (RGB)

        Returns:
            PIL Image with HED soft edges
        """
        detector = self._get_hed_detector()
        hed_map = detector(image)
        if hed_map.mode != "L":
            hed_map = hed_map.convert("L")
        return hed_map

    def extract(self, image: Image.Image, method: Optional[str] = None) -> Image.Image:
        """
        Extract structural map using the specified or default method.

        Args:
            image: PIL Image (RGB)
            method: Override default method ('canny' or 'hed')

        Returns:
            PIL Image with structural map
        """
        method = method or self.method

        if method == "canny":
            return self.extract_canny(image)
        elif method == "hed":
            return self.extract_hed(image)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'canny' or 'hed'.")

    def extract_for_controlnet(self, image: Image.Image, method: Optional[str] = None) -> Image.Image:
        """
        Extract and format structural map for ControlNet input.
        ControlNet expects a 3-channel RGB image where all channels are the edge map.

        Args:
            image: PIL Image (RGB)
            method: Override default method

        Returns:
            PIL Image (RGB) ready for ControlNet conditioning
        """
        edge_map = self.extract(image, method=method)
        if edge_map.mode != "L":
            edge_map = edge_map.convert("L")

        # ControlNet expects 3-channel input
        edge_array = np.array(edge_map)
        edge_rgb = np.stack([edge_array] * 3, axis=-1)
        return Image.fromarray(edge_rgb, "RGB")
