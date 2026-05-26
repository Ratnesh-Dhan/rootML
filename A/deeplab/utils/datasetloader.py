import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

COLOR_MAP = {
    (0, 0, 0): 0,
    (0, 0, 128): 1,
    (0, 128, 0): 2,
    (0, 128, 128): 3
}

class CorrosionDataset(Dataset):

    def __init__(self, image_paths, mask_paths, transforms=None):

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def rgb_to_mask(self, mask):

        class_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

        for color, class_id in COLOR_MAP.items():

            matches = np.all(mask == color, axis=-1)

            class_mask[matches] = class_id

        return class_mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx])

        mask = self.rgb_to_mask(mask)

        image = image.astype(np.float32) / 255.0

        if self.transforms:

            augmented = self.transforms(
                image=image,
                mask=mask
            )

            image = augmented["image"]
            mask = augmented["mask"]

        image = torch.tensor(image).permute(2,0,1).float()
        mask = torch.tensor(mask).long()

        return image, mask