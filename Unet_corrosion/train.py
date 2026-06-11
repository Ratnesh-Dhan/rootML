import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from dataset import CorrosionSegmentationDataset
from transforms import (
    get_train_transform,
    get_val_transform,
)
from model import build_model
from utils import compute_iou

COLOR_TO_CLASS = {
    (0, 0, 0): 0,        # background
    (0, 0, 128): 1,      # class1
    (0, 128, 0): 2,      # class2
    (0, 128, 128): 3,    # class3
}

full_dataset = CorrosionSegmentationDataset(
    image_dir="images",
    mask_dir="masks",
    color_to_class=COLOR_TO_CLASS
)

indices = list(range(len(full_dataset)))

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42
)

train_dataset = CorrosionSegmentationDataset(
    image_dir="images",
    mask_dir="masks",
    color_to_class=COLOR_TO_CLASS,
    transform=get_train_transform()
)

val_dataset = CorrosionSegmentationDataset(
    image_dir="images",
    mask_dir="masks",
    color_to_class=COLOR_TO_CLASS,
    transform=get_val_transform()
)

train_dataset = Subset(train_dataset, train_indices)
val_dataset = Subset(val_dataset, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
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
)

for epoch in range(50):

    model.train()

    running_loss = 0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = (
            ce_loss(outputs, masks)
            +
            dice_loss(outputs, masks)
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1} "
        f"Loss: {running_loss/len(train_loader):.4f}"
    )

model.eval()

val_iou = 0

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

print(
    f"Validation mIoU: {val_iou:.4f}"
)