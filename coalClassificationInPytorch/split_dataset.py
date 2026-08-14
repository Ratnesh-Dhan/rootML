import os
import random
import shutil
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

SOURCE_DIR = Path(
    r"/media/zumbie/6CA45A53A45A203E/2026-coal_samples/Himanshu Coal Samples 2026/16 size"
)

OUTPUT_DIR = Path(
    r"/media/zumbie/6CA45A53A45A203E/2026-coal_samples/Himanshu Coal Samples 2026/16_split"
)

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.30

SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# Split Dataset
# ============================================================

def split_dataset():

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Source directory does not exist:\n{SOURCE_DIR}"
        )

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory already exists:\n{OUTPUT_DIR}\n\n"
            f"Delete it manually if you want to create a fresh split."
        )

    train_dir = OUTPUT_DIR / "TRAIN"
    validation_dir = OUTPUT_DIR / "VALIDATION"

    train_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)

    random.seed(SEED)

    # --------------------------------------------------------
    # Find classes
    # --------------------------------------------------------

    class_dirs = sorted(
        [
            directory
            for directory in SOURCE_DIR.iterdir()
            if directory.is_dir()
        ]
    )

    if not class_dirs:
        raise RuntimeError(
            f"No class folders found in:\n{SOURCE_DIR}"
        )

    print(f"Found {len(class_dirs)} classes:\n")

    for class_dir in class_dirs:
        print(f"  {class_dir.name}")

    print("\n" + "=" * 60)

    total_train = 0
    total_validation = 0

    # --------------------------------------------------------
    # Process each class separately
    # --------------------------------------------------------

    for class_dir in class_dirs:

        images = [
            file
            for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not images:
            print(f"\nWARNING: No images found in {class_dir.name}")
            continue

        # Randomize images within this class
        random.shuffle(images)

        total_images = len(images)

        train_count = int(total_images * TRAIN_RATIO)

        train_images = images[:train_count]
        validation_images = images[train_count:]

        # Create corresponding class directories
        train_class_dir = train_dir / class_dir.name
        validation_class_dir = validation_dir / class_dir.name

        train_class_dir.mkdir(parents=True)
        validation_class_dir.mkdir(parents=True)

        # ----------------------------------------------------
        # Copy training images
        # ----------------------------------------------------

        for image in train_images:
            shutil.copy2(
                image,
                train_class_dir / image.name
            )

        # ----------------------------------------------------
        # Copy validation images
        # ----------------------------------------------------

        for image in validation_images:
            shutil.copy2(
                image,
                validation_class_dir / image.name
            )

        total_train += len(train_images)
        total_validation += len(validation_images)

        print(
            f"\n{class_dir.name}"
        )

        print(
            f"  Total      : {total_images}"
        )

        print(
            f"  Training   : {len(train_images)} "
            f"({len(train_images) / total_images * 100:.1f}%)"
        )

        print(
            f"  Validation : {len(validation_images)} "
            f"({len(validation_images) / total_images * 100:.1f}%)"
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    total = total_train + total_validation

    print("\n" + "=" * 60)
    print("DATASET SPLIT COMPLETE")
    print("=" * 60)

    print(f"\nTotal images      : {total}")
    print(
        f"Training images   : {total_train} "
        f"({total_train / total * 100:.2f}%)"
    )
    print(
        f"Validation images : {total_validation} "
        f"({total_validation / total * 100:.2f}%)"
    )

    print(f"\nTraining directory:")
    print(train_dir)

    print(f"\nValidation directory:")
    print(validation_dir)

    print(f"\nRandom seed: {SEED}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    split_dataset()