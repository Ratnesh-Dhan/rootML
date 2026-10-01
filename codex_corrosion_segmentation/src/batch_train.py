import sys
import csv
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
from src.visualize import (
    save_sanity_check,
    save_prediction_visualization,
)


# ============================================================
# OPTIMIZERS TO TEST
# ============================================================

OPTIMIZERS = [
    "adamw",
    "adam",
    "sgd",
    "rmsprop",
]


# ============================================================
# DATA AUGMENTATION
# ============================================================

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


# ============================================================
# MODEL
# ============================================================

def build_model():
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )

    return model


# ============================================================
# OPTIMIZER
# ============================================================

def build_optimizer(model, optimizer_name):

    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adamw":

        return torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

    elif optimizer_name == "adam":

        return torch.optim.Adam(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

    elif optimizer_name == "sgd":

        return torch.optim.SGD(
            model.parameters(),
            lr=LR,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY,
        )

    elif optimizer_name == "rmsprop":

        return torch.optim.RMSprop(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}"
        )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    best_iou,
    optimizer_name,
):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_iou": best_iou,
        "num_classes": NUM_CLASSES,
        "encoder_name": ENCODER_NAME,
        "optimizer_name": optimizer_name,
    }

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# CSV
# ============================================================

def init_metrics_csv(path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_miou",
                "val_loss",
                "val_miou",
            ]
        )


def save_epoch_metrics(
    path,
    epoch,
    train_loss,
    train_miou,
    val_loss,
    val_miou,
):

    with open(
        path,
        "a",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                epoch,
                f"{train_loss:.6f}",
                f"{train_miou:.6f}",
                f"{val_loss:.6f}",
                f"{val_miou:.6f}",
            ]
        )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
):

    model.train()

    running_loss = 0.0
    running_iou = 0.0

    progress = tqdm(
        loader,
        desc="Train",
        leave=False,
    )

    for images, masks, _ in progress:

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        assert masks.ndim == 3, (
            f"Expected masks [B,H,W], "
            f"got {masks.shape}"
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):

            logits = model(images)

            assert logits.shape[1] == NUM_CLASSES, (
                f"Expected {NUM_CLASSES} classes, "
                f"got {logits.shape}"
            )

            assert logits.ndim == 4, (
                f"Expected logits [B,C,H,W], "
                f"got {logits.shape}"
            )

            loss = criterion(
                logits,
                masks,
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        batch_iou = multiclass_iou(
            logits.detach(),
            masks,
            NUM_CLASSES,
        )

        running_loss += loss.item()

        running_iou += batch_iou.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            iou=f"{batch_iou.item():.4f}",
        )

    return (
        running_loss / len(loader),
        running_iou / len(loader),
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0
    running_iou = 0.0

    per_class_totals = {
        cls: []
        for cls in range(NUM_CLASSES)
    }

    progress = tqdm(
        loader,
        desc="Val",
        leave=False,
    )

    for images, masks, _ in progress:

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):

            logits = model(images)

            loss = criterion(
                logits,
                masks,
            )

        batch_iou = multiclass_iou(
            logits,
            masks,
            NUM_CLASSES,
        )

        class_ious = per_class_iou(
            logits,
            masks,
            NUM_CLASSES,
        )

        running_loss += loss.item()

        running_iou += batch_iou.item()

        for cls, value in class_ious.items():

            if value is not None:

                per_class_totals[cls].append(
                    value
                )

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            iou=f"{batch_iou.item():.4f}",
        )

    mean_per_class = {
        cls: (
            sum(values) / len(values)
            if values
            else None
        )
        for cls, values in per_class_totals.items()
    }

    return (
        running_loss / len(loader),
        running_iou / len(loader),
        mean_per_class,
    )


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

@torch.no_grad()
def save_validation_predictions(
    model,
    loader,
    device,
    epoch,
    prediction_dir,
):

    model.eval()

    images, masks, names = next(
        iter(loader)
    )

    images = images.to(device)

    logits = model(images)

    preds = torch.argmax(
        logits,
        dim=1,
    ).cpu()

    max_items = min(
        4,
        images.size(0),
    )

    for i in range(max_items):

        save_path = (
            prediction_dir
            / f"epoch_{epoch:03d}_{names[i]}.png"
        )

        save_prediction_visualization(
            image_tensor=images[i].cpu(),
            target_mask=masks[i],
            pred_mask=preds[i],
            save_path=save_path,
            class_to_color=CLASS_TO_COLOR,
        )


# ============================================================
# RUN ONE OPTIMIZER EXPERIMENT
# ============================================================

def run_experiment(
    optimizer_name,
    train_loader,
    val_loader,
    device,
):

    print("\n")
    print("=" * 70)
    print(
        f"STARTING OPTIMIZER: "
        f"{optimizer_name.upper()}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Optimizer-specific directories
    # --------------------------------------------------------

    result_dir = (
        CHECKPOINT_DIR
        / optimizer_name
    )

    sanity_dir = (
        result_dir
        / "sanity"
    )

    prediction_dir = (
        result_dir
        / "predictions"
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sanity_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    metrics_file = (
        result_dir
        / "metrics.csv"
    )

    init_metrics_csv(
        metrics_file
    )

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    sanity_images, sanity_masks, sanity_names = next(
        iter(train_loader)
    )

    for i in range(
        min(
            4,
            sanity_images.size(0),
        )
    ):

        save_sanity_check(
            image_tensor=sanity_images[i],
            mask_tensor=sanity_masks[i],
            save_path=(
                sanity_dir
                / f"sanity_{i}_{sanity_names[i]}.png"
            ),
            class_to_color=CLASS_TO_COLOR,
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Build a completely NEW model for every optimizer.
    #
    # This guarantees that Adam doesn't start from AdamW's
    # trained weights, etc.
    # --------------------------------------------------------

    model = build_model().to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = DiceCELoss(
        dice_weight=1.0,
        ce_weight=1.0,
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = build_optimizer(
        model,
        optimizer_name,
    )

    print(
        f"Optimizer : {optimizer_name}"
    )

    print(
        f"LR        : {LR}"
    )

    print(
        f"Weight Decay: {WEIGHT_DECAY}"
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda"
    )

    # --------------------------------------------------------
    # Best metrics
    # --------------------------------------------------------

    best_iou = 0.0
    best_class_ious = None

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print(
            f"\n"
            f"[{optimizer_name.upper()}] "
            f"Epoch {epoch}/{EPOCHS}"
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_iou = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
        )

        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        (
            val_loss,
            val_iou,
            val_class_ious,
        ) = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            val_iou
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        save_epoch_metrics(
            path=metrics_file,
            epoch=epoch,
            train_loss=train_loss,
            train_miou=train_iou,
            val_loss=val_loss,
            val_miou=val_iou,
        )

        # ----------------------------------------------------
        # Console
        # ----------------------------------------------------

        print(
            f"Train Loss : {train_loss:.4f}"
        )

        print(
            f"Train mIoU : {train_iou:.4f}"
        )

        print(
            f"Val Loss   : {val_loss:.4f}"
        )

        print(
            f"Val mIoU   : {val_iou:.4f}"
        )

        print(
            f"LR         : {current_lr:.2e}"
        )

        # ----------------------------------------------------
        # Per-class IoU
        # ----------------------------------------------------

        print(
            "Per-class IoU:"
        )

        for cls, value in val_class_ious.items():

            print(
                f"  Class {cls}: "
                f"{value if value is not None else 'N/A'}"
            )

        # ----------------------------------------------------
        # LAST CHECKPOINT
        # ----------------------------------------------------

        save_checkpoint(
            path=(
                result_dir
                / "last.pth"
            ),
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_iou=best_iou,
            optimizer_name=optimizer_name,
        )

        # ----------------------------------------------------
        # BEST CHECKPOINT
        # ----------------------------------------------------

        if val_iou > best_iou:

            best_iou = val_iou

            best_class_ious = (
                val_class_ious.copy()
            )

            save_checkpoint(
                path=(
                    result_dir
                    / "best.pth"
                ),
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_iou=best_iou,
                optimizer_name=optimizer_name,
            )

            print(
                f"NEW BEST mIoU: "
                f"{best_iou:.4f}"
            )

        # ----------------------------------------------------
        # Prediction visualization
        # ----------------------------------------------------

        if (
            epoch == 1
            or epoch % 5 == 0
        ):

            save_validation_predictions(
                model=model,
                loader=val_loader,
                device=device,
                epoch=epoch,
                prediction_dir=prediction_dir,
            )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n")
    print(
        "=" * 70
    )

    print(
        f"{optimizer_name.upper()} "
        f"TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Best Validation mIoU: "
        f"{best_iou:.4f}"
    )

    # --------------------------------------------------------
    # Final results TXT
    # --------------------------------------------------------

    results_file = (
        result_dir
        / "final_results.txt"
    )

    with open(
        results_file,
        "w",
    ) as f:

        f.write(
            "========== FINAL RESULTS ==========\n"
        )

        f.write(
            f"Optimizer: "
            f"{optimizer_name}\n"
        )

        f.write(
            f"Best Validation mIoU: "
            f"{best_iou:.4f}\n\n"
        )

        if best_class_ious is not None:

            for cls, value in best_class_ious.items():

                if value is not None:

                    f.write(
                        f"Class {cls} IoU: "
                        f"{value:.4f}\n"
                    )

                else:

                    f.write(
                        f"Class {cls} IoU: N/A\n"
                    )

    print(
        f"Metrics: "
        f"{metrics_file}"
    )

    print(
        f"Results: "
        f"{results_file}"
    )

    return {
        "optimizer": optimizer_name,
        "best_miou": best_iou,
        "best_class_ious": best_class_ious,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        DEVICE
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        torch.backends.cudnn.benchmark = True

    # --------------------------------------------------------
    # Base directories
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Dataset
    #
    # Created ONCE and reused for every optimizer.
    # Each optimizer still receives a freshly initialized model.
    # --------------------------------------------------------

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

    print(
        f"Train samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Val samples: "
        f"{len(val_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Run every optimizer
    # --------------------------------------------------------

    all_results = []

    for optimizer_name in OPTIMIZERS:

        result = run_experiment(
            optimizer_name=optimizer_name,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

        all_results.append(
            result
        )

        # ----------------------------------------------------
        # Free GPU memory before next optimizer
        # ----------------------------------------------------

        torch.cuda.empty_cache()

    # ========================================================
    # ALL EXPERIMENTS COMPLETE
    # ========================================================

    print("\n\n")
    print(
        "=" * 70
    )

    print(
        "ALL OPTIMIZER EXPERIMENTS COMPLETE"
    )

    print(
        "=" * 70
    )

    for result in all_results:

        print(
            f"{result['optimizer']:10s} "
            f"-> "
            f"{result['best_miou']:.4f}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()