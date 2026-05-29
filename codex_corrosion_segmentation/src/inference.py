import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import cv2
import torch
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp
from matplotlib import pyplot as plt

from configs.config import (
    IMAGE_SIZE,
    NUM_CLASSES,
    ENCODER_NAME,
    CLASS_TO_COLOR,
    PREDICTION_DIR,
)


def class_mask_to_bgr(mask, class_to_color):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, bgr_color in class_to_color.items():
        color_mask[mask == class_id] = bgr_color

    return color_mask


def build_model():
    return smp.DeepLabV3Plus(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )


@torch.no_grad()
def predict_single_image(image_path, checkpoint_path, output_path, device):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    original_h, original_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    transform = A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        ]
    )

    augmented = transform(image=image_rgb)
    image = augmented["image"].astype(np.float32) / 255.0

    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    image_tensor = image_tensor.to(device)

    model = build_model().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logits = model(image_tensor)

    # logits: [1,4,H,W]
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    pred_bgr = class_mask_to_bgr(pred, CLASS_TO_COLOR)
 
    pred_bgr = cv2.resize(
        pred_bgr,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST,
    )

    overlay = cv2.addWeighted(image_bgr, 0.65, pred_bgr, 0.35, 0)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # cv2.imwrite(str(output_path), pred_bgr)
    # cv2.imwrite(str(output_path.with_name(output_path.stem + "_overlay.png")), overlay)

    # print(f"Saved mask: {output_path}")
    # print(f"Saved overlay: {output_path.with_name(output_path.stem + '_overlay.png')}")

    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20,20))
    plt.subplot(1,2,1)
    plt.imshow(image_rgb)
    plt.title("Input Image")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(overlay)
    plt.title("Corrosion Detected")
    plt.axis("off")
    plt.savefig(output_path.with_name(output_path.stem + "_overlay.png"), bbox_inches="tight")
    print(f"Saved prediction: {output_path.with_name(output_path.stem + '_overlay.png')}")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--image", type=str, required=True)
#     # parser.add_argument("--checkpoint", type=str, required=True)
#     parser.add_argument("--output", type=str, default=None)

#     args = parser.parse_args()

#     image_path = Path(args.image)
#     # checkpoint_path = Path(args.checkpoint)
#     checkpoint_path = Path("../outputs/checkpoints/best.pth")

#     if args.output:
#         output_path = Path(args.output)
#     else:
#         PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
#         output_path = PREDICTION_DIR / f"{image_path.stem}_pred_mask.png"

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     predict_single_image(
#         image_path=image_path,
#         checkpoint_path=checkpoint_path,
#         output_path=output_path,
#         device=device,
#     )

def main():
    image_folder = "/mnt/z/DATASETS/corrosion_detect/images"
    output_folder = "/mnt/z/DATASETS/output_corrosion_29_5_26"
    os.makedirs(output_folder, exist_ok=True)
    # checkpoint_path = Path(args.checkpoint)
    checkpoint_path = Path("./outputs/checkpoints/best.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for image in os.listdir(image_folder):
        image_path = Path(image_folder) / image
        output_path = Path(output_folder) / f"{image_path.stem}_pred_mask.png"
        predict_single_image(
            image_path=image_path,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            device=device,
        )


if __name__ == "__main__":
    main()