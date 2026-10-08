import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import torch
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from configs.config import (
    IMAGE_SIZE,
    NUM_CLASSES,
    ENCODER_NAME,
    CLASS_TO_COLOR,
    COLOR_TO_CLASS_Dataset,
)


# ============================================================
# PATHS
# ============================================================

IMAGE_FOLDER = r"/mnt/z/DATASETS/Corrosion_Condition_State_Classification/512x512/Test/images_512"

MASK_FOLDER = r"/mnt/z/DATASETS/Corrosion_Condition_State_Classification/512x512/Test/mask_512"

OUTPUT_FOLDER = r"./four_partition_output_new_2"

CHECKPOINT_PATH = Path(
    r"/mnt/z/codes/rootML/codex_corrosion_segmentation/outputs/checkpoints/adamw/best.pth"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "Background",
    1: "Lite",
    2: "Moderate",
    3: "Severe",
}


# ============================================================
# MODEL
# ============================================================

def build_model():

    return smp.DeepLabV3Plus(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )


# ============================================================
# CLASS MASK -> BGR COLOR MASK
# ============================================================

def class_mask_to_bgr(mask, class_to_color):
    """
    Convert class-index mask to BGR color mask.

    mask:
        [H, W]

    class_to_color:
        {
            class_id: (B, G, R)
        }

    Returns:
        BGR uint8 image.
    """

    h, w = mask.shape

    color_mask = np.zeros(
        (h, w, 3),
        dtype=np.uint8,
    )

    for class_id, color in class_to_color.items():

        class_id = int(class_id)

        color = np.array(
            color,
            dtype=np.uint8,
        )

        color_mask[mask == class_id] = color

    return color_mask


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_gt_mask(mask_path):
    """
    Convert the ORIGINAL DATASET mask into class IDs.

    IMPORTANT:

    The dataset mask colors are BGR because cv2.imread()
    loads the PNG as BGR.

    COLOR_TO_CLASS_Dataset:

        dataset BGR color -> class ID

    Example:

        (0, 0, 128) -> class 1
        (0, 128, 0) -> class 2
        (0, 128, 128) -> class 3
    """

    mask_bgr = cv2.imread(
        str(mask_path),
        cv2.IMREAD_COLOR,
    )

    if mask_bgr is None:

        raise RuntimeError(
            f"Cannot read mask: {mask_path}"
        )

    # --------------------------------------------------------
    # Create class-index mask
    # --------------------------------------------------------

    class_mask = np.zeros(
        mask_bgr.shape[:2],
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Dataset color -> class ID
    # --------------------------------------------------------

    for color, class_id in COLOR_TO_CLASS_Dataset.items():

        color = np.array(
            color,
            dtype=np.uint8,
        )

        class_id = int(class_id)

        matches = np.all(
            mask_bgr == color,
            axis=-1,
        )

        class_mask[matches] = class_id

    return class_mask


# ============================================================
# OPTIONAL DEBUG
# ============================================================

def debug_mask(mask_path):
    """
    Print the colors actually present in a dataset mask.
    """

    mask_bgr = cv2.imread(
        str(mask_path),
        cv2.IMREAD_COLOR,
    )

    if mask_bgr is None:
        raise RuntimeError(
            f"Cannot read mask: {mask_path}"
        )

    colors, counts = np.unique(
        mask_bgr.reshape(-1, 3),
        axis=0,
        return_counts=True,
    )

    print("\n" + "=" * 70)
    print("DATASET MASK DEBUG")
    print("=" * 70)

    print("\nColors actually found in dataset:")

    for color, count in zip(colors, counts):

        color_tuple = tuple(
            int(x)
            for x in color
        )

        class_id = COLOR_TO_CLASS_Dataset.get(
            color_tuple,
            None,
        )

        if class_id is not None:

            print(
                f"BGR {color_tuple} "
                f"-> Class {class_id} "
                f"({CLASS_NAMES.get(class_id, 'Unknown')}) "
                f"-> {count:,} pixels"
            )

        else:

            print(
                f"BGR {color_tuple} "
                f"-> UNKNOWN "
                f"-> {count:,} pixels"
            )

    print("=" * 70)


# ============================================================
# CALCULATE PERCENTAGES
# ============================================================

def calculate_percentages(mask):
    """
    Calculate percentage of each corrosion class.

    Background is ignored.
    """

    total = np.sum(
        mask > 0
    )

    if total == 0:

        return {
            1: 0.0,
            2: 0.0,
            3: 0.0,
        }

    percentages = {}

    for cls in [1, 2, 3]:

        percentages[cls] = (
            100.0
            * np.sum(mask == cls)
            / total
        )

    return percentages


# ============================================================
# PREDICTION + VISUALIZATION
# ============================================================

@torch.no_grad()
def predict_single_image(
    image_path,
    mask_path,
    model,
    output_path,
    device,
):

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    image_bgr = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:

        print(
            "Couldn't read:",
            image_path,
        )

        return

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    original_h, original_w = image_bgr.shape[:2]

    # ========================================================
    # RESIZE IMAGE FOR MODEL
    # ========================================================

    transform = A.Compose([
        A.Resize(
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
    ])

    img = transform(
        image=image_rgb
    )["image"]

    img = img.astype(
        np.float32
    ) / 255.0

    # ========================================================
    # NUMPY -> TORCH
    # ========================================================

    tensor = (
        torch.from_numpy(img)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    # ========================================================
    # MODEL
    # ========================================================

    logits = model(tensor)

    # ========================================================
    # ARGMAX -> CLASS IDs
    # ========================================================

    pred = torch.argmax(
        logits,
        dim=1,
    ).squeeze().cpu().numpy().astype(
        np.uint8
    )

    # ========================================================
    # RESIZE PREDICTION BACK
    # ========================================================

    pred = cv2.resize(
        pred,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST,
    )

    # ========================================================
    # LOAD GT
    # ========================================================
    #
    # IMPORTANT:
    #
    # Original dataset color
    #       ↓
    # COLOR_TO_CLASS_Dataset
    #       ↓
    # class IDs
    #
    # ========================================================

    gt = load_gt_mask(
        mask_path
    )

    # ========================================================
    # CLASS IDs -> YOUR DISPLAY COLORS
    # ========================================================
    #
    # Both GT and prediction now use the SAME visualization
    # colors.
    #
    # GT:
    #
    # dataset color
    #      ↓
    # class ID
    #      ↓
    # CLASS_TO_COLOR
    #
    # Prediction:
    #
    # model class ID
    #      ↓
    # CLASS_TO_COLOR
    #
    # ========================================================

    gt_color = class_mask_to_bgr(
        gt,
        CLASS_TO_COLOR,
    )

    pred_color = class_mask_to_bgr(
        pred,
        CLASS_TO_COLOR,
    )

    # ========================================================
    # CREATE OVERLAYS
    # ========================================================

    gt_overlay = cv2.addWeighted(
        image_bgr,
        0.65,
        gt_color,
        0.35,
        0,
    )

    pred_overlay = cv2.addWeighted(
        image_bgr,
        0.65,
        pred_color,
        0.35,
        0,
    )

    # ========================================================
    # BGR -> RGB FOR MATPLOTLIB
    # ========================================================

    gt_overlay = cv2.cvtColor(
        gt_overlay,
        cv2.COLOR_BGR2RGB,
    )

    pred_overlay = cv2.cvtColor(
        pred_overlay,
        cv2.COLOR_BGR2RGB,
    )

    # ========================================================
    # PERCENTAGES
    # ========================================================

    gt_pct = calculate_percentages(
        gt
    )

    pred_pct = calculate_percentages(
        pred
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        f"\nImage: {image_path.name}"
    )

    print("Ground Truth:")

    print(
        f"  Lite     : {gt_pct[1]:6.2f}%"
    )

    print(
        f"  Moderate : {gt_pct[2]:6.2f}%"
    )

    print(
        f"  Severe   : {gt_pct[3]:6.2f}%"
    )

    print("Prediction:")

    print(
        f"  Lite     : {pred_pct[1]:6.2f}%"
    )

    print(
        f"  Moderate : {pred_pct[2]:6.2f}%"
    )

    print(
        f"  Severe   : {pred_pct[3]:6.2f}%"
    )

    # ========================================================
    # FIGURE
    # ========================================================

    fig, ax = plt.subplots(
        1,
        4,
        figsize=(28, 7),
    )

    # ========================================================
    # ORIGINAL
    # ========================================================

    ax[0].imshow(
        image_rgb
    )

    ax[0].set_title(
        "Original Image",
        fontsize=20,
    )

    ax[0].axis("off")

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    ax[1].imshow(
        gt_overlay
    )

    ax[1].set_title(
        "Ground Truth Overlay",
        fontsize=20,
    )

    ax[1].axis("off")

    # ========================================================
    # PREDICTION
    # ========================================================

    ax[2].imshow(
        pred_overlay
    )

    ax[2].set_title(
        "Prediction Overlay",
        fontsize=20,
    )

    ax[2].axis("off")

    # ========================================================
    # TEXT
    # ========================================================

    ax[3].axis("off")

    # --------------------------------------------------------
    # Ground Truth
    # --------------------------------------------------------

    ax[3].text(
        0,
        0.95,
        "Ground Truth",
        fontsize=22,
        weight="bold",
        va="top",
        family="monospace",
    )

    ax[3].text(
        0,
        0.85,
        f"Lite     : {gt_pct[1]:6.2f}%",
        fontsize=20,
        va="top",
        family="monospace",
        color="green",
    )

    ax[3].text(
        0,
        0.75,
        f"Moderate : {gt_pct[2]:6.2f}%",
        fontsize=20,
        va="top",
        family="monospace",
        color="orange",
    )

    ax[3].text(
        0,
        0.65,
        f"Severe   : {gt_pct[3]:6.2f}%",
        fontsize=20,
        va="top",
        family="monospace",
        color="red",
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    ax[3].text(
        0,
        0.50,
        "Prediction",
        fontsize=22,
        weight="bold",
        va="top",
        family="monospace",
    )

    ax[3].text(
        0,
        0.40,
        f"Lite     : {pred_pct[1]:6.2f}%",
        fontsize=20,
        va="top",
        family="monospace",
        color="green",
    )

    ax[3].text(
        0,
        0.30,
        f"Moderate : {pred_pct[2]:6.2f}%",
        fontsize=20,
        va="top",
        family="monospace",
        color="orange",
    )

    ax[3].text(
        0,
        0.20,
        f"Severe   : {pred_pct[3]:6.2f}%",
        fontsize=20,
        va="top",
        family="monospace",
        color="red",
    )

    # ========================================================
    # SAVE
    # ========================================================

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path.name}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # OUTPUT
    # ========================================================

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True,
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # ========================================================
    # PRINT MAPPINGS
    # ========================================================

    print("\n" + "=" * 70)
    print("CLASS MAPPINGS")
    print("=" * 70)

    print("\nDataset BGR -> Class ID:")

    for color, class_id in COLOR_TO_CLASS_Dataset.items():

        print(
            f"  {color} -> "
            f"{class_id} "
            f"({CLASS_NAMES.get(class_id, 'Unknown')})"
        )

    print("\nClass ID -> Display BGR:")

    for class_id, color in CLASS_TO_COLOR.items():

        print(
            f"  {class_id} "
            f"({CLASS_NAMES.get(class_id, 'Unknown')}) "
            f"-> {color}"
        )

    print("=" * 70)

    # ========================================================
    # BUILD MODEL
    # ========================================================

    model = build_model().to(
        device
    )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    print(
        f"\nLoading checkpoint:\n"
        f"{CHECKPOINT_PATH}"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Checkpoint loaded successfully."
    )

    # ========================================================
    # EXTENSIONS
    # ========================================================

    valid_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    ]

    # ========================================================
    # FIND IMAGES
    # ========================================================

    images = sorted(
        [
            x
            for x in os.listdir(
                IMAGE_FOLDER
            )
            if Path(x).suffix.lower()
            in valid_extensions
        ]
    )

    print(
        f"Found {len(images)} images"
    )

    # ========================================================
    # PROCESS
    # ========================================================

    for image_name in images:

        image_path = (
            Path(IMAGE_FOLDER)
            / image_name
        )

        mask_path = (
            Path(MASK_FOLDER)
            / (
                Path(image_name).stem
                + ".png"
            )
        )

        # ----------------------------------------------------
        # Check mask
        # ----------------------------------------------------

        if not mask_path.exists():

            print(
                f"Mask not found: "
                f"{mask_path.name}"
            )

            continue

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        output_path = (
            Path(OUTPUT_FOLDER)
            / f"{Path(image_name).stem}.png"
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        predict_single_image(
            image_path=image_path,
            mask_path=mask_path,
            model=model,
            output_path=output_path,
            device=device,
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()