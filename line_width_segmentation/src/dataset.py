from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(image_dir: str | Path) -> list[Path]:
    image_dir = Path(image_dir)
    return sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_train_transform(patch_size: int) -> A.Compose:
    return A.Compose([
        A.PadIfNeeded(
            min_height=patch_size,
            min_width=patch_size,
            border_mode=cv2.BORDER_REFLECT_101,
            p=1.0,
        ),
        A.RandomCrop(height=patch_size, width=patch_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.08,
            rotate_limit=8,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.4,
        ),
        A.RandomBrightnessContrast(p=0.35),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
    ])


def build_val_transform(patch_size: int) -> A.Compose:
    return A.Compose([
        A.PadIfNeeded(
            min_height=patch_size,
            min_width=patch_size,
            border_mode=cv2.BORDER_REFLECT_101,
            p=1.0,
        ),
        A.CenterCrop(height=patch_size, width=patch_size, p=1.0),
    ])


class TrackDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        transform: A.Compose | None = None,
    ) -> None:
        self.image_paths = list_images(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        mask_path = self.mask_dir / image_path.name

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read mask: {mask_path}")
        mask = (mask > 127).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        mask = mask.astype(np.float32)[None, :, :]

        return torch.from_numpy(image), torch.from_numpy(mask)
