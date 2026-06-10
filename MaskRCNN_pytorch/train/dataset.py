from classMap import CLASS_MAP
import base64
import zlib
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import json
import io
from PIL import Image
import torchvision.transforms.functional as F
import torch

class VOCDataset(Dataset):

    def __init__(self, root):

        self.root = Path(root)

        self.img_dir = self.root / "img"
        self.ann_dir = self.root / "ann"

        self.images = sorted(
            self.img_dir.glob("*.jpg")
        )

    def __len__(self):
        return len(self.images)

    def decode_bitmap(self, encoded):
        compressed = base64.b64decode(encoded)
        decoded = zlib.decompress(compressed)

        mask = np.array(
            Image.open(io.BytesIO(decoded))
        )

        return (mask > 0).astype(np.uint8)
    
    def __getitem__(self, idx):

        image_path = self.images[idx]

        ann_path = (
            self.ann_dir /
            f"{image_path.name}.json"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = F.to_tensor(image)

        with open(ann_path) as f:
            ann = json.load(f)

        height = ann["size"]["height"]
        width = ann["size"]["width"]

        boxes = []
        labels = []
        masks = []

        for obj in ann["objects"]:

            if obj["classTitle"] == "neutral":
                continue

            cropped_mask = self.decode_bitmap(
                obj["bitmap"]["data"]
            )

            x, y = obj["bitmap"]["origin"]

            h, w = cropped_mask.shape

            full_mask = np.zeros(
                (height, width),
                dtype=np.uint8
            )

            full_mask[
                y:y+h,
                x:x+w
            ] = cropped_mask

            ys, xs = np.where(
                full_mask > 0
            )

            if len(xs) == 0:
                continue

            xmin = xs.min()
            xmax = xs.max()

            ymin = ys.min()
            ymax = ys.max()

            boxes.append(
                [xmin, ymin, xmax, ymax]
            )

            labels.append(
                CLASS_MAP[
                    obj["classTitle"]
                ]
            )

            masks.append(
                full_mask
            )

        target = {
            "boxes": torch.as_tensor(
                boxes,
                dtype=torch.float32
            ),

            "labels": torch.as_tensor(
                labels,
                dtype=torch.int64
            ),

            "masks": torch.as_tensor(
                np.stack(masks),
                dtype=torch.uint8
            )
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))