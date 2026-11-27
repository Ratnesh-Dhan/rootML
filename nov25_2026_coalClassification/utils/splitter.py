
import random
import shutil
from pathlib import Path
import os

# Paths
SOURCE_DIR = Path("/mnt/z/DATASETS/Coal_Patch/Original")
OUTPUT_DIR = Path("/mnt/z/DATASETS/Coal_Patch/working_directory")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_DIR = OUTPUT_DIR / "development"
TEST_DIR = OUTPUT_DIR / "test"

TEST_RATIO = 0.10
SEED = 42

random.seed(SEED)

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

for class_dir in SOURCE_DIR.iterdir():

    if not class_dir.is_dir():
        continue

    images = [
        file for file in class_dir.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.shuffle(images)

    test_count = round(len(images) * TEST_RATIO)

    test_images = images[:test_count]
    development_images = images[test_count:]

    # Create class folders
    train_class_dir = TRAIN_DIR / class_dir.name
    test_class_dir = TEST_DIR / class_dir.name

    train_class_dir.mkdir(parents=True, exist_ok=True)
    test_class_dir.mkdir(parents=True, exist_ok=True)

    # Copy development patches
    for image in development_images:
        shutil.copy2(image, train_class_dir / image.name)

    # Copy test patches
    for image in test_images:
        shutil.copy2(image, test_class_dir / image.name)

    print(
        f"{class_dir.name}: "
        f"Development = {len(development_images)}, "
        f"Test = {len(test_images)}"
    )

print("\nDataset splitting completed.")
