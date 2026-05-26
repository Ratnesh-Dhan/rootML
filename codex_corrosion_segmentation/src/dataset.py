import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class CorrosionSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, color_to_class, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.color_to_class = color_to_class
        self.transform = transform
        self.image_paths = sorted(
            [
                p for p in self.image_dir.iterdir()
                if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
            ]
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _find_mask_path(self, image_path):
        candidates = [
            self.mask_dir / f"{image_path.stem}.png",
            self.mask_dir / f"{image_path.stem}.jpg",
            self.mask_dir / f"{image_path.stem}.jpeg",
            self.mask_dir / f"{image_path.stem}.bmp",
            self.mask_dir / f"{image_path.stem}.tif",
            self.mask_dir / f"{image_path.stem}.tiff",
        ]

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(f"No matching mask found for image: {image_path.name}")

    def _bgr_mask_to_class_mask(self, mask_bgr):
        class_mask = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)

        for bgr_color, class_id in self.color_to_class.items():
            matches = np.all(mask_bgr == np.array(bgr_color, dtype=np.uint8), axis=-1)
            class_mask[matches] = class_id

        return class_mask

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self._find_mask_path(image_path)

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if mask_bgr is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask = self._bgr_mask_to_class_mask(mask_bgr)

        if image_rgb.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image and mask size mismatch: {image_path.name}, "
                f"image={image_rgb.shape[:2]}, mask={mask.shape[:2]}"
            )

        if self.transform:
            augmented = self.transform(image=image_rgb, mask=mask)
            image_rgb = augmented["image"]
            mask = augmented["mask"]

        image_rgb = image_rgb.astype(np.float32) / 255.0

        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).long()

        # image: [3,H,W]
        # mask: [H,W], NOT one-hot encoded
        return image_tensor, mask_tensor, image_path.name