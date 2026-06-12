import torch
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from dataset import CorrosionSegmentationDataset
from transforms import (
    get_train_transform,
    get_val_transform,
)
from model import build_model
from utils import compute_iou, compute_class_iou

import random
import numpy as np

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    
COLOR_TO_CLASS = {
    (0, 0, 0): 0,        # background
    (0, 0, 128): 1,      # class1
    (0, 128, 0): 2,      # class2
    (0, 128, 128): 3,    # class3
}

CLASS_NAMES = [
    "background",
    "mild",
    "moderate",
    "severe"
]

train_dataset = CorrosionSegmentationDataset(
    image_dir="/mnt/z/DATASETS/Corrosion_Condition_State_Classification/512x512/Train/images_512",
    mask_dir="/mnt/z/DATASETS/Corrosion_Condition_State_Classification/512x512/Train/mask_512",
    color_to_class=COLOR_TO_CLASS,
    transform=get_train_transform()
)

val_dataset = CorrosionSegmentationDataset(
    image_dir="/mnt/z/DATASETS/Corrosion_Condition_State_Classification/512x512/Test/images_512",
    mask_dir="/mnt/z/DATASETS/Corrosion_Condition_State_Classification/512x512/Test/mask_512",
    color_to_class=COLOR_TO_CLASS,
    transform=get_val_transform()
)

train_loader = DataLoader(
    train_dataset,
    # batch_size=8,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    # batch_size=8,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = build_model(
    num_classes=4
).to(device)

ce_loss = torch.nn.CrossEntropyLoss()

dice_loss = smp.losses.DiceLoss(
    mode="multiclass"
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5,
)

num_epochs = 50

best_miou = 0.0

train_losses = []
val_losses = []
val_ious = []

background_ious = []
mild_ious = []
moderate_ious = []
severe_ious = []

# For early stopping
early_stopping_patience = 10
epochs_without_improvement = 0

for epoch in range(num_epochs):

    # ==========================
    # Training
    # ==========================
    model.train()

    running_loss = 0.0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = (
            ce_loss(outputs, masks)
            + dice_loss(outputs, masks)
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # ==========================
    # Validation
    # ==========================
    model.eval()

    val_iou = 0.0

    #Trying new per class iou
    total_iou = np.zeros(4)
    count = np.zeros(4)

    val_loss = 0
    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            val_iou += compute_iou(
                outputs,
                masks,
                num_classes=4,
            )
            batch_ious = compute_class_iou(
            outputs,
            masks,
            num_classes=4,
            )

            loss = (
                ce_loss(outputs, masks)
                + dice_loss(outputs, masks)
            )

            val_loss += loss.item()

            for cls, iou in enumerate(batch_ious):

                if not np.isnan(iou):
                    total_iou[cls] += iou
                    count[cls] += 1

    val_iou /= len(val_loader)
    scheduler.step(val_iou)
    # class_iou = total_iou / count
    class_iou = np.divide(
        total_iou,
        count,
        out=np.zeros_like(total_iou),
        where=count != 0
    )

    # saving seperate IOU   
    background_ious.append(class_iou[0])
    mild_ious.append(class_iou[1])
    moderate_ious.append(class_iou[2])
    severe_ious.append(class_iou[3])

    # scheduler.step(class_iou)
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    for cls, name in enumerate(CLASS_NAMES):

        print(
            f"{name}: "
            f"{class_iou[cls]:.4f}"
        )

    # ==========================
    # History
    # ==========================
    train_losses.append(avg_train_loss)
    val_ious.append(val_iou)
    # val_ious.append(class_iou)

    # ==========================
    # Save Best Model
    # ==========================
    min_delta = 1e-4
    if val_iou > best_miou + min_delta:

        best_miou = val_iou
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_miou": best_miou,
        }, "best_model_with_optimizer.pth")

        print(
            f"✓ Best model saved "
            f"(mIoU={best_miou:.4f})"
        )
    else:
        epochs_without_improvement += 1

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {avg_train_loss:.4f} "
        f"Val mIoU: {val_iou:.4f}"
    )

    if epochs_without_improvement >= early_stopping_patience:
        print(
            f"\nEarly stopping triggered after "
            f"{epoch+1} epochs."
        )
        break

print(f"\nBest Validation mIoU: {best_miou:.4f}")

import pandas as pd
import matplotlib.pyplot as plt

# Save history
# history = pd.DataFrame({
#     "epoch": range(1, num_epochs + 1),
#     "train_loss": train_losses,
#     "val_miou": val_ious
# })

history = pd.DataFrame({
    "epoch": range(1, num_epochs + 1),

    "train_loss": train_losses,
    "val_loss": val_losses,

    "val_miou": val_ious,

    "background_iou": background_ious,
    "mild_iou": mild_ious,
    "moderate_iou": moderate_ious,
    "severe_iou": severe_ious,
})


history.to_csv(
    "training_history.csv",
    index=False
)

# Loss graph
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.close()

# mIoU graph
plt.figure(figsize=(8, 5))
plt.plot(val_ious)
plt.title("Validation mIoU")
plt.xlabel("Epoch")
plt.ylabel("mIoU")
plt.grid(True)
plt.tight_layout()
plt.savefig("miou_curve.png")
plt.close()

# IoU graphs
plt.figure(figsize=(10,6))

plt.plot(background_ious, label="Background")
plt.plot(mild_ious, label="Mild")
plt.plot(moderate_ious, label="Moderate")
plt.plot(severe_ious, label="Severe")

plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.title("Per-Class IoU")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("class_iou_curve.png")
