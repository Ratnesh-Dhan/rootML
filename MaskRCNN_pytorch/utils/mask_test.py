
import json
import base64
import zlib
import io

import cv2
import numpy as np
from PIL import Image


IMG_PATH = "/mnt/z/DATASETS/pascal-voc-2012-DatasetNinja/train/img/2007_000032.jpg"
ANN_PATH = "/mnt/z/DATASETS/pascal-voc-2012-DatasetNinja/train/ann/2007_000032.jpg.json"

with open(ANN_PATH) as f:
    ann = json.load(f)

img = cv2.imread(IMG_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

height = ann["size"]["height"]
width = ann["size"]["width"]

full_mask = np.zeros(
    (height, width),
    dtype=np.uint8
)

total_length = len(ann['objects'])

for idx in range(total_length):
    
    obj = ann["objects"][idx]

    compressed = base64.b64decode(
        obj["bitmap"]["data"]
    )

    decoded = zlib.decompress(compressed)

    cropped_mask = np.array(
        Image.open(io.BytesIO(decoded))
    )

    cropped_mask = (
        cropped_mask > 0
    ).astype(np.uint8)
    print("cropped mask :", np.unique(cropped_mask))
    # full_mask = np.zeros(
    #     (height, width),
    #     dtype=np.uint8
    # )

    x, y = obj["bitmap"]["origin"]

    h, w = cropped_mask.shape

    full_mask[y:y+h, x:x+w] = cropped_mask

overlay = img.copy()
overlay[full_mask == 1] = (
    0.5 * overlay[full_mask == 1]
    + 0.5 * np.array([255, 0, 0])
).astype(np.uint8)

Image.fromarray(overlay).save("overlay.png")

print(full_mask.shape)
print(np.unique(full_mask))