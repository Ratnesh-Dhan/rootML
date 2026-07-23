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
)
ENCODER_NAME = "efficientnet-b0"
# ============================================================
# CHANGE THESE PATHS
# ============================================================

IMAGE_FOLDER = r"/mnt/d/DATASETS/CORROSION/Corrosion_Condition_State_Classification/Corrosion_Condition_State_Classification/512x512/Test/images_512"
MASK_FOLDER = r"/mnt/d/DATASETS/CORROSION/Corrosion_Condition_State_Classification/Corrosion_Condition_State_Classification/512x512/Test/mask_512"
OUTPUT_FOLDER = r"./outputs"

CHECKPOINT_PATH = Path(
    r"/mnt/d/Models/Corrosion_Condition_State_Classification_Models&Outputs/checkpoints/last.pth"
)

# ============================================================


def build_model():
    return smp.DeepLabV3Plus(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )


def class_mask_to_bgr(mask, class_to_color):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in class_to_color.items():
        color_mask[mask == class_id] = color

    return color_mask


def load_gt_mask(mask_path):
    """
    Convert color PNG mask into class-index mask.
    """

    mask_bgr = cv2.imread(str(mask_path))

    if mask_bgr is None:
        raise RuntimeError(f"Cannot read mask {mask_path}")

    class_mask = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)

    for class_id, color in CLASS_TO_COLOR.items():
        color = np.array(color, dtype=np.uint8)
        matches = np.all(mask_bgr == color, axis=-1)
        class_mask[matches] = class_id

    return class_mask


def calculate_percentages(mask):
    """
    Percentage of each corrosion class.
    Background ignored.
    """

    total = np.sum(mask > 0)

    if total == 0:
        return {1: 0, 2: 0, 3: 0}

    percentages = {}

    for cls in [1, 2, 3]:
        percentages[cls] = 100 * np.sum(mask == cls) / total

    return percentages


@torch.no_grad()
def predict_single_image(
    image_path,
    mask_path,
    model,
    output_path,
    device,
):

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        print("Couldn't read:", image_path)
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    original_h, original_w = image_bgr.shape[:2]

    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    ])

    img = transform(image=image_rgb)["image"]
    img = img.astype(np.float32) / 255.0

    tensor = (
        torch.from_numpy(img)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    logits = model(tensor)

    pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    pred = cv2.resize(
        pred,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST,
    )

    gt = load_gt_mask(mask_path)

    pred_color = class_mask_to_bgr(pred, CLASS_TO_COLOR)
    gt_color = class_mask_to_bgr(gt, CLASS_TO_COLOR)

    gt_overlay = cv2.addWeighted(image_bgr, 0.65, gt_color, 0.35, 0)
    pred_overlay = cv2.addWeighted(image_bgr, 0.65, pred_color, 0.35, 0)

    gt_overlay = cv2.cvtColor(gt_overlay, cv2.COLOR_BGR2RGB)
    pred_overlay = cv2.cvtColor(pred_overlay, cv2.COLOR_BGR2RGB)

    gt_pct = calculate_percentages(gt)
    pred_pct = calculate_percentages(pred)

    fig, ax = plt.subplots(1, 4, figsize=(28, 7))

    ax[0].imshow(image_rgb)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(gt_overlay)
    ax[1].set_title("Ground Truth Overlay")
    ax[1].axis("off")

    ax[2].imshow(pred_overlay)
    ax[2].set_title("Prediction Overlay")
    ax[2].axis("off")

    ax[3].axis("off")

    txt = (
        "Ground Truth\n\n"
        f"Class 1 : {gt_pct[1]:6.2f}%\n"
        f"Class 2 : {gt_pct[2]:6.2f}%\n"
        f"Class 3 : {gt_pct[3]:6.2f}%\n\n"
        "Prediction\n\n"
        f"Class 1 : {pred_pct[1]:6.2f}%\n"
        f"Class 2 : {pred_pct[2]:6.2f}%\n"
        f"Class 3 : {pred_pct[3]:6.2f}%"
    )

    ax[3].text(
        0,
        1,
        txt,
        fontsize=16,
        va="top",
        family="monospace",
    )

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path.name}")


def main():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    model = build_model().to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    images = sorted(
        [
            x
            for x in os.listdir(IMAGE_FOLDER)
            if Path(x).suffix.lower() in valid_extensions
        ]
    )

    print(f"Found {len(images)} images")

    for image_name in images:

        image_path = Path(IMAGE_FOLDER) / image_name
        mask_path = Path(MASK_FOLDER) / (Path(image_name).stem + ".png")

        if not mask_path.exists():
            print(f"Mask not found: {mask_path.name}")
            continue

        output_path = Path(OUTPUT_FOLDER) / f"{Path(image_name).stem}.png"

        predict_single_image(
            image_path=image_path,
            mask_path=mask_path,
            model=model,
            output_path=output_path,
            device=device,
        )


if __name__ == "__main__":
    main()