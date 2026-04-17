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
        self.manifest = pd.read_csv(manifest_path)
        self.resolution = resolution
        self.prompt = prompt
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.tokenizer = tokenizer

        required = {"clean_path", "degraded_path"}
        missing = required - set(self.manifest.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_and_resize(self, path: str) -> Image.Image:
        """Load an image and resize to target resolution."""
        img = Image.open(path).convert("RGB")
        if img.size != (self.resolution, self.resolution):
            img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        return img

    def _image_to_tensor(self, img: Image.Image, normalize: bool = True) -> torch.Tensor:
        """Convert PIL image to tensor. If normalize=True, maps [0,1] -> [-1,1] for SD latent space."""
        arr = np.array(img).astype(np.float32) / 255.0  # [0, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
        if normalize:
            tensor = tensor * 2.0 - 1.0  # [-1, 1]
        return tensor

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

        clean_image = self._load_and_resize(row["clean_path"])
        degraded_image = self._load_and_resize(row["degraded_path"])

        # Canny edges from clean image for ControlNet conditioning
        edge_map = self._extract_canny(clean_image)

        # Convert to tensors
        clean_tensor = self._image_to_tensor(clean_image, normalize=True)       # [-1, 1]
        degraded_tensor = self._image_to_tensor(degraded_image, normalize=True)  # [-1, 1]
        edge_tensor = self._image_to_tensor(edge_map, normalize=False)           # [0, 1]

        sample = {
            "clean": clean_tensor,
            "degraded": degraded_tensor,
            "edge_map": edge_tensor,
            "prompt": self.prompt,
        }

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
    dataset = ProductEnhancementDataset(
        manifest_path=manifest_path,
        resolution=resolution,
        tokenizer=tokenizer,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
