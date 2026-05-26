from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[1]
print("ROOT_DIR:", ROOT_DIR)

# Base Path
base_path = "/mnt/z/DATASETS/Corrosion_Condition_State_Classification/512x512"

TRAIN_IMAGE_DIR = os.path.join(base_path, "Train", "images_512")
TRAIN_MASK_DIR = os.path.join(base_path, "Train", "mask_512")

VAL_IMAGE_DIR = os.path.join(base_path, "Test", "images_512")
VAL_MASK_DIR = os.path.join(base_path, "Test", "mask_512")

OUTPUT_DIR = ROOT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
SANITY_DIR = OUTPUT_DIR / "sanity_checks"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SANITY_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)

IMAGE_SIZE = 512
NUM_CLASSES = 4

ENCODER_NAME = "efficientnet-b0"
ENCODER_WEIGHTS = "imagenet"

BATCH_SIZE = 4
NUM_WORKERS = 4

EPOCHS = 80
LR = 1e-4
WEIGHT_DECAY = 1e-4

DEVICE = "cuda"

CLASS_NAMES = [
    "background",
    "class1",
    "class2",
    "class3",
]

# Your masks are described as BGR colors because OpenCV reads PNGs as BGR.
COLOR_TO_CLASS = {
    (0, 0, 0): 0,        # background
    (0, 0, 128): 1,      # class1
    (0, 128, 0): 2,      # class2
    (0, 128, 128): 3,    # class3
}

CLASS_TO_COLOR = {
    0: (0, 0, 0),
    1: (0, 0, 128),
    2: (0, 128, 0),
    3: (0, 128, 128),
}