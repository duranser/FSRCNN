from __future__ import annotations

import random

from PIL import Image
from torch.utils.data import Dataset

from .utils import (
    bicubic,
    crop_to_scale,
    flip_left_right,
    flip_top_bottom,
    list_images,
    make_aligned_pair,
    open_rgb,
    pil_to_tensor,
)


class SRPatchDataset(Dataset):
    """Random aligned LR/HR patches generated from complete degraded images.

    Important ordering:
      1. augment the complete HR RGB image,
      2. crop it to a multiple of scale,
      3. convert to MATLAB-compatible limited-range Y (for channels=1),
      4. bicubic-downsample the complete image,
      5. crop the LR patch and its exactly corresponding HR patch.

    SRImageDataset uses the same color conversion and degradation path for
    validation and testing.
    """

    def __init__(
        self,
        hr_dir,
        scale: int,
        lr_patch_size: int = 48,
        channels: int = 1,
        repeat: int = 100,
        augment: bool = True,
        paper_aug: bool = True,
    ) -> None:
        self.paths = list_images(hr_dir)
        if not self.paths:
            raise FileNotFoundError(f"No images found in {hr_dir}")
        self.scale = int(scale)
        self.lr_patch_size = int(lr_patch_size)
        self.hr_patch_size = self.lr_patch_size * self.scale
        self.channels = int(channels)
        self.repeat = max(1, int(repeat))
        self.augment = bool(augment)
        self.paper_aug = bool(paper_aug)

    def __len__(self) -> int:
        return len(self.paths) * self.repeat

    def _ensure_min_size(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width >= self.hr_patch_size and height >= self.hr_patch_size:
            return image
        ratio = max(
            self.hr_patch_size / max(width, 1),
            self.hr_patch_size / max(height, 1),
        )
        new_width = max(self.hr_patch_size, int(round(width * ratio)))
        new_height = max(self.hr_patch_size, int(round(height * ratio)))
        return image.resize((new_width, new_height), bicubic())

    def __getitem__(self, index):
        hr_rgb = open_rgb(self.paths[index % len(self.paths)])

        if self.augment and self.paper_aug:
            scale_aug = random.choice([1.0, 0.9, 0.8, 0.7, 0.6])
            if scale_aug != 1.0:
                width, height = hr_rgb.size
                hr_rgb = hr_rgb.resize(
                    (
                        max(self.hr_patch_size, int(round(width * scale_aug))),
                        max(self.hr_patch_size, int(round(height * scale_aug))),
                    ),
                    bicubic(),
                )
            rotation = random.choice([0, 90, 180, 270])
            if rotation:
                hr_rgb = hr_rgb.rotate(rotation, expand=True)

        if self.augment:
            if random.random() < 0.5:
                hr_rgb = hr_rgb.transpose(flip_left_right())
            if random.random() < 0.5:
                hr_rgb = hr_rgb.transpose(flip_top_bottom())

        hr_rgb = crop_to_scale(self._ensure_min_size(hr_rgb), self.scale)
        lr_full, hr_full = make_aligned_pair(hr_rgb, self.scale, self.channels)

        max_x = lr_full.width - self.lr_patch_size
        max_y = lr_full.height - self.lr_patch_size
        if max_x < 0 or max_y < 0:
            raise RuntimeError(
                f"LR image {lr_full.size} is smaller than LR patch "
                f"{self.lr_patch_size}x{self.lr_patch_size}"
            )

        lr_x = random.randint(0, max_x)
        lr_y = random.randint(0, max_y)
        hr_x = lr_x * self.scale
        hr_y = lr_y * self.scale

        lr_patch = lr_full.crop(
            (
                lr_x,
                lr_y,
                lr_x + self.lr_patch_size,
                lr_y + self.lr_patch_size,
            )
        )
        hr_patch = hr_full.crop(
            (
                hr_x,
                hr_y,
                hr_x + self.hr_patch_size,
                hr_y + self.hr_patch_size,
            )
        )

        return pil_to_tensor(lr_patch), pil_to_tensor(hr_patch)


class SRImageDataset(Dataset):
    def __init__(self, hr_dir, scale: int, channels: int = 1) -> None:
        self.paths = list_images(hr_dir)
        if not self.paths:
            raise FileNotFoundError(f"No images found in {hr_dir}")
        self.scale = int(scale)
        self.channels = int(channels)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        lr, hr = make_aligned_pair(open_rgb(path), self.scale, self.channels)
        return pil_to_tensor(lr), pil_to_tensor(hr), str(path)
