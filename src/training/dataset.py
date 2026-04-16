"""
PyTorch Dataset for loading clean-degraded image pairs with edge maps
for Stable Diffusion + ControlNet LoRA training.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from typing import Optional, Dict
from torchvision import transforms


class ProductEnhancementDataset(Dataset):
    """
    Loads degraded-clean pairs from a manifest CSV.
    Returns degraded image, clean image, Canny edge map, and text prompt.
    """

    def __init__(
        self,
        manifest_path: str,
        resolution: int = 512,
        prompt: str = "professional product photography, studio lighting, clean white background, high quality, 4k, detailed",
        canny_low: int = 100,
        canny_high: int = 200,
        tokenizer=None,
    ):
        """
        Args:
            manifest_path: Path to CSV with columns: clean_path, degraded_path
            resolution: Target image resolution (images should already be this size)
            prompt: Text prompt used during training
            canny_low: Canny edge detection lower threshold
            canny_high: Canny edge detection upper threshold
            tokenizer: Optional CLIPTokenizer for pre-tokenizing prompts
        """
        self.manifest = pd.read_csv(manifest_path)
        self.resolution = resolution
        self.prompt = prompt
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.tokenizer = tokenizer

        # Verify required columns exist
        required = {"clean_path", "degraded_path"}
        missing = required - set(self.manifest.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}")

        # Standard normalization for SD latent space: [0,1] -> [-1,1]
        self.image_transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # Conditioning images (edge maps) are normalized to [0,1] only
        self.conditioning_transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.manifest)

    def _extract_canny(self, image: Image.Image) -> Image.Image:
        """Extract Canny edges from a PIL image, return as 3-channel RGB."""
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        edges_rgb = np.stack([edges] * 3, axis=-1)
        return Image.fromarray(edges_rgb, "RGB")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.manifest.iloc[idx]

        clean_path = row["clean_path"]
        degraded_path = row["degraded_path"]

        clean_image = Image.open(clean_path).convert("RGB")
        degraded_image = Image.open(degraded_path).convert("RGB")

        # Extract Canny edges from the clean image for ControlNet conditioning
        edge_map = self._extract_canny(clean_image)

        # Apply transforms
        clean_tensor = self.image_transform(clean_image)
        degraded_tensor = self.image_transform(degraded_image)
        edge_tensor = self.conditioning_transform(edge_map)

        sample = {
            "clean": clean_tensor,
            "degraded": degraded_tensor,
            "edge_map": edge_tensor,
            "prompt": self.prompt,
        }

        # Pre-tokenize if tokenizer is available
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                self.prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            sample["input_ids"] = tokens.input_ids.squeeze(0)

        return sample


def get_dataloader(
    manifest_path: str,
    batch_size: int = 1,
    resolution: int = 512,
    num_workers: int = 4,
    shuffle: bool = True,
    tokenizer=None,
) -> torch.utils.data.DataLoader:
    """
    Convenience function to create a DataLoader from a manifest CSV.

    Args:
        manifest_path: Path to train/val/test manifest
        batch_size: Batch size
        resolution: Image resolution
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle
        tokenizer: Optional CLIPTokenizer

    Returns:
        DataLoader
    """
    dataset = ProductEnhancementDataset(
        manifest_path=manifest_path,
        resolution=resolution,
        tokenizer=tokenizer,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return loader
