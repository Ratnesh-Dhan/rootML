from ast import Interpolation
from Unet_corrosion.inference import corrosion_pixels
import segmentation_models_pytorch as smp
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
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
)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
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

def build_unet_model(num_classes):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )
    return model

def build_deeplabv3_model():
    return smp.DeepLabV3Plus(encoder_name=ENCODER_NAME, encoder_weights=None, in_channels=3, classes=NUM_CLASSES, activation=None)

IMAGE_DIR = "/mnt/z/DATASETS/Corrosion_Condition_State_Classification/512x512/Test/images_512"
OUTPUT_DIR = "/mnt/z/DATASETS/CORROSION_SEGMENTATION_with_both_models"

Path(OUTPUT_DIR).mkdir(
    parents=True,
    exist_ok=True
)

# Load unet binary model
unetModel = build_unet_model(num_classes=2)
checkpoint = torch.load(
    "/mnt/z/codes/rootML/Unet_corrosion/model/best_model.pth",
    map_location=DEVICE
)
unetModel.load_state_dict(checkpoint)
unetModel.to(DEVICE)
unetModel.eval()

# Load DeeplabV3 3 class segmentation model (Only to determine class of the corrosion not overlay)
deepLabModel = build_deeplabv3_model().to(DEVICE)
deepLabmodelCheckpoint = torch.load("/mnt/z/codes/rootML/codex_corrosion_segmentation/outputs/checkpoints/best/pth", map_location=DEVICE)
deepLabModel.load_state_dict(checkpoint["model_state_dict"])
deepLabModel.eval()

# Inference with both models
image_paths = []

for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
    image_paths.extend(
        Path(IMAGE_DIR).glob(ext)
    )

with torch.no_grad():
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None: 
            continue
        original_h, original_w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (512,512))
        image_tensor = (torch.from_numpy(image_resized.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(DEVICE))
        # Inference with U-NET model..
        outputs = unetModel(image_tensor)
        pred_mask = torch.argmax(outputs, dim=1)[0]
        pred_mask = (pred_mask.cpu().numpy().astype(np.uint8))
        pred_mask =cv2.resize(pred_mask, (original_w, original_h), Interpolation=cv2.INTER_NEAREST)

        # Creating overlay
        overlay = image_bgr.copy()
        corrosion_pixels = pred_mask == 1
        overlay[corrosion_pixels] = (0.4*overlay[corrosion_pixels]+0.6*np.array([0,0,255]))
        overlay = overlay.astype(np.uint8)

        # Inference with DeepLab model..
        transform = A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE)])
        augmented = transform(image=image_rgb)
        image = augmented['image'].astype(np.float32) / 255.0
        deepLab_image_tensor = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float()
        deepLab_image_tensor = deepLab_image_tensor.to(DEVICE)
        logits = deepLabModel(deepLab_image_tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        pred = cv2.resize(pred, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        pred_pct = calculate_percentages(pred)
        txt = (
            "Prediction\n\n"
            f"Lite : {pred_pct[1]:6.2f}%\n"
            f"Mild : {pred_pct[2]:6.2f}%\n"
            f"Severe : {pred_pct[3]:6.2f}%"
        )

        # Saving images
        _, ax = plt.subplots(1,3, figsize=(21, 7))
        ax[0].imshow(image_rgb)
        ax[0].set_title("Input Image")
        ax[0].axis("off")

        ax[1].imshow(overlay)
        ax[1].set_title("Predicted Corrosion")
        ax[1].axis("off")

        ax[2].axis("off")
        ax[2].text(0,1,txt,fontsize=16,va="top",family="monospace")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, os.path.basename(image_path)), dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {os.path.join(OUTPUT_DIR, os.path.basename(image_path))}")

