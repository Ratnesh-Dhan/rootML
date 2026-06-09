import json
import base64
import zlib
import io
import numpy as np

from PIL import Image

with open("/mnt/d/DATASETS/pascal-voc-2012-DatasetNinja/train/ann/2007_000032.jpg.json") as f:
    ann = json.load(f)

obj = ann["objects"][0]

encoded = obj["bitmap"]["data"]

decoded = zlib.decompress(
    base64.b64decode(encoded)
)

mask = np.array(
    Image.open(io.BytesIO(decoded))
)

print(mask.shape)
print(mask.dtype)

if mask.ndim == 3:
    print(mask[0,0])