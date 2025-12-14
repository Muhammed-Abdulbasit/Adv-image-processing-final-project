import os
import random
from glob import glob

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset


def load_image(path, target_size=None):
    img = Image.open(path).convert("RGB")
    if target_size is not None:
        img = img.resize(target_size, Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0  # [0,1]
    arr = np.transpose(arr, (2, 0, 1))  # C,H,W
    return arr


class SyntheticReflectionDataset(Dataset):
    """
    Synthetic dataset:
    - pick two images: clean and reflection
    - create mixture: m = clean + alpha * blurred(reflection)
    - return (m_lr, clean_lr) for base training
    - also (m_hr, clean_hr) for upsampler training
    """

    def __init__(
        self,
        clean_dir,
        refl_dir=None,
        hr_size=(512, 512),
        lr_size=(256, 256),
        mode="base",  # "base" or "upsampler"
    ):
        self.clean_paths = sorted(glob(os.path.join(clean_dir, "*")))
        if len(self.clean_paths) == 0:
            raise ValueError(f"No images found in {clean_dir}")

        self.refl_paths = (
            sorted(glob(os.path.join(refl_dir, "*")))
            if refl_dir is not None
            else self.clean_paths
        )
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.mode = mode

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        # Pick clean image
        clean_path = self.clean_paths[idx]
        clean_hr = load_image(clean_path, target_size=self.hr_size)

        # Pick random reflection image
        refl_path = random.choice(self.refl_paths)
        refl_hr = load_image(refl_path, target_size=self.hr_size)

        # Convert back to PIL to blur reflection a bit
        refl_img = Image.fromarray(
            (np.transpose(refl_hr, (1, 2, 0)) * 255).astype(np.uint8)
        ).filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 3.0)))
        refl_hr = np.transpose(
            np.array(refl_img).astype(np.float32) / 255.0, (2, 0, 1)
        )

        # Random reflection strength
        alpha = random.uniform(0.3, 0.8)
        mix_hr = clean_hr + alpha * refl_hr
        mix_hr = np.clip(mix_hr, 0.0, 1.0)

        # Low-res versions
        clean_lr_img = Image.fromarray(
            (np.transpose(clean_hr, (1, 2, 0)) * 255).astype(np.uint8)
        ).resize(self.lr_size, Image.BICUBIC)
        mix_lr_img = Image.fromarray(
            (np.transpose(mix_hr, (1, 2, 0)) * 255).astype(np.uint8)
        ).resize(self.lr_size, Image.BICUBIC)

        clean_lr = np.transpose(
            np.array(clean_lr_img).astype(np.float32) / 255.0, (2, 0, 1)
        )
        mix_lr = np.transpose(
            np.array(mix_lr_img).astype(np.float32) / 255.0, (2, 0, 1)
        )

        if self.mode == "base":
            # For base training, just return LR inputs
            return (
                torch.from_numpy(mix_lr),
                torch.from_numpy(clean_lr),
            )
        elif self.mode == "upsampler":
            # For upsampler training, return both LR and HR
            return (
                torch.from_numpy(mix_lr),
                torch.from_numpy(clean_lr),
                torch.from_numpy(mix_hr),
                torch.from_numpy(clean_hr),
            )
        else:
            raise ValueError(f"Unknown mode {self.mode}")
