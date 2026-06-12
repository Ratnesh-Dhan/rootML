import torch
import segmentation_models_pytorch as smp
# from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from dataset import CorrosionSegmentationDataset
from transforms import (
    get_train_transform,
    get_val_transform,
)
from model import build_model
from utils import compute_iou

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

# full_dataset = CorrosionSegmentationDataset(
#     image_dir="images",
#     mask_dir="masks",
#     color_to_class=COLOR_TO_CLASS
# )

# indices = list(range(len(full_dataset)))

# train_indices, val_indices = train_test_split(
#     indices,
#     test_size=0.2,
#     random_state=42
# )

train_dataset = CorrosionSegmentationDataset(
    image_dir="/mnt/z/DATASETS/Corrosion_Condition_State_Classification/512x512/Train/image_512",
    mask_dir="/mnt/z/DATASETS/Corrosion_Condition_State_Classification/512x512/Train/mask_512",
    color_to_class=COLOR_TO_CLASS,
    transform=get_train_transform()
)

val_dataset = CorrosionSegmentationDataset(
    image_dir="/mnt/z/DATASETS/Corrosion_Condition_State_Classification/512x512/Test/_512",
    mask_dir="/mnt/z/DATASETS/Corrosion_Condition_State_Classification/512x512/Test/mask_512",
    color_to_class=COLOR_TO_CLASS,
    transform=get_val_transform()
)

# train_dataset = Subset(train_dataset, train_indices)
# val_dataset = Subset(val_dataset, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
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
val_ious = []

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
    val_iou /= len(val_loader)
    scheduler.step(val_iou)

    # ==========================
    # History
    # ==========================
    train_losses.append(avg_train_loss)
    val_ious.append(val_iou)

    # ==========================
    # Save Best Model
    # ==========================
    if val_iou > best_miou:

        best_miou = val_iou

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

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {avg_train_loss:.4f} "
        f"Val mIoU: {val_iou:.4f}"
    )

print(f"\nBest Validation mIoU: {best_miou:.4f}")

import pandas as pd
import matplotlib.pyplot as plt

# Save history
history = pd.DataFrame({
    "epoch": range(1, num_epochs + 1),
    "train_loss": train_losses,
    "val_miou": val_ious
})

history.to_csv(
    "training_history.csv",
    index=False
)

# Loss graph
plt.figure(figsize=(8, 5))
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("loss_curve.png")
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
