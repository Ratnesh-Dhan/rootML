import json
import base64
import zlib
import io

import cv2
import numpy as np
from PIL import Image

IMG_PATH = "/mnt/z/DATASETS/pascal-voc-2012-DatasetNinja/train/img/2007_000032.jpg"
ANN_PATH = "/mnt/z/DATASETS/pascal-voc-2012-DatasetNinja/train/ann/2007_000032.jpg.json"

# ----------------------------------------------------
# Load image
# ----------------------------------------------------
with open(ANN_PATH) as f:
    ann = json.load(f)

img = cv2.imread(IMG_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

height = ann["size"]["height"]
width = ann["size"]["width"]

print(f"Image Size: {width}x{height}")
print(f"Objects: {len(ann['objects'])}")

# ----------------------------------------------------
# Combined mask
# ----------------------------------------------------
full_mask = np.zeros(
    (height, width),
    dtype=np.uint8
)

# ----------------------------------------------------
# Process every object
# ----------------------------------------------------
for idx, obj in enumerate(ann["objects"]):

    print("\n-----------------------")
    print(f"Object {idx}")
    print("Class:", obj["classTitle"])

    compressed = base64.b64decode(
        obj["bitmap"]["data"]
    )

    decoded = zlib.decompress(compressed)

    raw_mask = np.array(
        Image.open(io.BytesIO(decoded))
    )

    print("Raw shape:", raw_mask.shape)
    print("Raw dtype:", raw_mask.dtype)
    print("Raw unique:", np.unique(raw_mask))

    cropped_mask = (
        raw_mask > 0
    ).astype(np.uint8)

    x, y = obj["bitmap"]["origin"]

    h, w = cropped_mask.shape

    instance_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    instance_mask[
        y:y+h,
        x:x+w
    ] = cropped_mask

    # Save individual mask
    Image.fromarray(
        instance_mask * 255
    ).save(
        f"instance_{idx}_{obj['classTitle']}.png"
    )

    full_mask[
        instance_mask > 0
    ] = 1

# ----------------------------------------------------
# Save combined mask
# ----------------------------------------------------
Image.fromarray(
    full_mask * 255
).save(
    "combined_mask.png"
)

# ----------------------------------------------------
# Overlay
# ----------------------------------------------------
overlay = img.copy()

overlay[full_mask == 1] = (
    0.5 * overlay[full_mask == 1]
    + 0.5 * np.array([255, 0, 0])
).astype(np.uint8)

Image.fromarray(
    overlay
).save(
    "overlay.png"
)

print("\n================================")
print("Combined mask unique:", np.unique(full_mask))
print("Mask pixels:", np.sum(full_mask))
print("Saved:")
print("  combined_mask.png")
print("  overlay.png")
print("  instance_*.png")