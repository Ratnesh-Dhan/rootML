import cv2
import numpy as np
from matplotlib import pyplot

COLOR_MAP = {
    (0, 0, 0): 0,         # background
    (0, 0, 128): 1,       # red
    (0, 128, 0): 2,       # green
    (0, 128, 128): 3      # olive
}

def rgb_to_mask(mask_path):
    mask = cv2.imread(mask_path)
    print(np.unique(mask.reshape(-1, 3), axis=0))

    class_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

    for color, class_id in COLOR_MAP.items():
        matches = np.all(mask == color, axis=-1)
        class_mask[matches] = class_id

    return class_mask

mask = rgb_to_mask("/mnt/d/DATASETS/CORROSION/Corrosion_Condition_State_Classification/Corrosion_Condition_State_Classification/512x512/Train/mask_512/29.png")
# print(np.unique(mask.reshape(-1,3), axis=0))
print(np.unique(mask))