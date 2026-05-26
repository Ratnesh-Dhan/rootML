import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import albumentations as A
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import (
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    VAL_IMAGE_DIR,
    VAL_MASK_DIR,
    CHECKPOINT_DIR,
    SANITY_DIR,
    PREDICTION_DIR,
    IMAGE_SIZE,
    NUM_CLASSES,
    ENCODER_NAME,
    ENCODER_WEIGHTS,
    BATCH_SIZE,
    NUM_WORKERS,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    DEVICE,
    COLOR_TO_CLASS,
    CLASS_TO_COLOR,
)

from src.dataset import CorrosionSegmentationDataset
from src.losses import DiceCELoss
from src.metrics import multiclass_iou, per_class_iou
from src.visualize import save_sanity_check, save_prediction_visualization


def get_train_transforms():
    return A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=20,
                border_mode=0,
                value=0,
                mask_value=0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(p=0.4),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.15),
        ]
    )


def get_val_transforms():
    return A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        ]
    )


def build_model():
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
    return model


def save_checkpoint(path, model, optimizer, scaler, epoch, best_iou):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_iou": best_iou,
        "num_classes": NUM_CLASSES,
        "encoder_name": ENCODER_NAME,
    }
    torch.save(checkpoint, path)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()

    running_loss = 0.0
    running_iou = 0.0

    progress = tqdm(loader, desc="Train", leave=False)

    for images, masks, _ in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Required shapes:
        # images: [B,3,H,W]
        # masks: [B,H,W]
        assert masks.ndim == 3, f"Expected masks [B,H,W], got {masks.shape}"

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)

            # Required model output:
            # logits: [B,4,H,W]
            assert logits.shape[1] == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {logits.shape}"
            assert logits.ndim == 4, f"Expected logits [B,C,H,W], got {logits.shape}"

            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_iou = multiclass_iou(logits.detach(), masks, NUM_CLASSES)

        running_loss += loss.item()
        running_iou += batch_iou.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            iou=f"{batch_iou.item():.4f}",
        )

    return running_loss / len(loader), running_iou / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_iou = 0.0
    per_class_totals = {cls: [] for cls in range(NUM_CLASSES)}

    progress = tqdm(loader, desc="Val", leave=False)

    for images, masks, _ in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)

        batch_iou = multiclass_iou(logits, masks, NUM_CLASSES)
        class_ious = per_class_iou(logits, masks, NUM_CLASSES)

        running_loss += loss.item()
        running_iou += batch_iou.item()

        for cls, value in class_ious.items():
            if value is not None:
                per_class_totals[cls].append(value)

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            iou=f"{batch_iou.item():.4f}",
        )

    mean_per_class = {
        cls: sum(values) / len(values) if len(values) > 0 else None
        for cls, values in per_class_totals.items()
    }

    return running_loss / len(loader), running_iou / len(loader), mean_per_class


@torch.no_grad()
def save_validation_predictions(model, loader, device, epoch):
    model.eval()

    images, masks, names = next(iter(loader))
    images = images.to(device)

    logits = model(images)
    preds = torch.argmax(logits, dim=1).cpu()

    max_items = min(4, images.size(0))

    for i in range(max_items):
        save_path = PREDICTION_DIR / f"epoch_{epoch:03d}_{names[i]}.png"
        save_prediction_visualization(
            image_tensor=images[i].cpu(),
            target_mask=masks[i],
            pred_mask=preds[i],
            save_path=save_path,
            class_to_color=CLASS_TO_COLOR,
        )


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    SANITY_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    train_dataset = CorrosionSegmentationDataset(
        image_dir=TRAIN_IMAGE_DIR,
        mask_dir=TRAIN_MASK_DIR,
        color_to_class=COLOR_TO_CLASS,
        transform=get_train_transforms(),
    )

    val_dataset = CorrosionSegmentationDataset(
        image_dir=VAL_IMAGE_DIR,
        mask_dir=VAL_MASK_DIR,
        color_to_class=COLOR_TO_CLASS,
        transform=get_val_transforms(),
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    # Sanity check visualization before training
    sanity_images, sanity_masks, sanity_names = next(iter(train_loader))
    for i in range(min(4, sanity_images.size(0))):
        save_sanity_check(
            image_tensor=sanity_images[i],
            mask_tensor=sanity_masks[i],
            save_path=SANITY_DIR / f"sanity_{i}_{sanity_names[i]}.png",
            class_to_color=CLASS_TO_COLOR,
        )

    print(f"Saved sanity checks to: {SANITY_DIR}")

    model = build_model().to(device)

    criterion = DiceCELoss(dice_weight=1.0, ce_weight=1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        verbose=True,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_iou = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_iou = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
        )

        val_loss, val_iou, val_class_ious = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_iou)

        print(
            f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}"
        )

        print("Per-class IoU:")
        for cls, value in val_class_ious.items():
            print(f"  Class {cls}: {value if value is not None else 'N/A'}")

        save_checkpoint(
            path=CHECKPOINT_DIR / "last.pth",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_iou=best_iou,
        )

        if val_iou > best_iou:
            best_iou = val_iou
            save_checkpoint(
                path=CHECKPOINT_DIR / "best.pth",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_iou=best_iou,
            )
            print(f"Saved new best checkpoint with IoU: {best_iou:.4f}")

        if epoch == 1 or epoch % 5 == 0:
            save_validation_predictions(model, val_loader, device, epoch)

    print("Training complete.")
    print(f"Best validation IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()