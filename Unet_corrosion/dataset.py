from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
from pathlib import Path


class CorrosionSegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        color_to_class,
        transform=None,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.color_to_class = color_to_class
        self.transform = transform

        self.image_paths = sorted(
            [
                p
                for p in self.image_dir.iterdir()
                if p.suffix.lower()
                in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def _find_mask_path(self, image_path):
        return self.mask_dir / f"{image_path.stem}.png"

    def _bgr_mask_to_class_mask(self, mask_bgr):
        class_mask = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)

        for color, class_id in self.color_to_class.items():
            matches = np.all(mask_bgr == np.array(color), axis=-1)
            class_mask[matches] = class_id

        return class_mask

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self._find_mask_path(image_path)

        image = cv2.cvtColor(
            cv2.imread(str(image_path)),
            cv2.COLOR_BGR2RGB,
        )

        mask_bgr = cv2.imread(str(mask_path))
        mask = self._bgr_mask_to_class_mask(mask_bgr)

        if self.transform:
            transformed = self.transform(
                image=image,
                mask=mask,
            )

            image = transformed["image"]
            mask = transformed["mask"]

        image = image.astype(np.float32) / 255.0

        image = torch.tensor(
            image,
            dtype=torch.float32,
        ).permute(2, 0, 1)

        mask = torch.tensor(
            mask,
            dtype=torch.long,
        )

        return image, mask