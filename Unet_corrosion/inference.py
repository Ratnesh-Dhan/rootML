import cv2
import torch
import numpy as np
from pathlib import Path

from model import build_model


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_DIR = "test_images"
OUTPUT_DIR = "predictions"

Path(OUTPUT_DIR).mkdir(
    parents=True,
    exist_ok=True
)

# -------------------------
# Load Model
# -------------------------
model = build_model(num_classes=2)

checkpoint = torch.load(
    "best_model.pth",
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

# -------------------------
# Inference
# -------------------------
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

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        image_resized = cv2.resize(
            image_rgb,
            (512, 512)
        )

        image_tensor = (
            torch.from_numpy(
                image_resized.astype(np.float32) / 255.0
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(DEVICE)
        )

        outputs = model(image_tensor)

        pred_mask = torch.argmax(
            outputs,
            dim=1
        )[0]

        pred_mask = (
            pred_mask.cpu()
            .numpy()
            .astype(np.uint8)
        )

        pred_mask = cv2.resize(
            pred_mask,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST
        )

        # -------------------------
        # Create Overlay
        # -------------------------
        overlay = image_bgr.copy()

        corrosion_pixels = pred_mask == 1

        overlay[corrosion_pixels] = (
            0.4 * overlay[corrosion_pixels]
            + 0.6 * np.array([0, 0, 255])
        )

        overlay = overlay.astype(np.uint8)

        # -------------------------
        # Save mask
        # -------------------------
        mask_vis = pred_mask * 255

        cv2.imwrite(
            str(
                Path(OUTPUT_DIR)
                / f"{image_path.stem}_mask.png"
            ),
            mask_vis
        )

        # -------------------------
        # Save overlay
        # -------------------------
        cv2.imwrite(
            str(
                Path(OUTPUT_DIR)
                / f"{image_path.stem}_overlay.png"
            ),
            overlay
        )

        print(
            f"Processed {image_path.name}"
        )

print("Done.")