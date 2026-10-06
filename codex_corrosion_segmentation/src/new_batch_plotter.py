from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader
from tqdm import tqdm


sys.path.append(str(Path(__file__).resolve().parents[1]))


# ============================================================
# CONFIG
# ============================================================

from configs.config import (
    CHECKPOINT_DIR,
    VAL_IMAGE_DIR,
    VAL_MASK_DIR,
    IMAGE_SIZE,
    NUM_CLASSES,
    ENCODER_NAME,
    ENCODER_WEIGHTS,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
    COLOR_TO_CLASS,
)

from src.dataset import CorrosionSegmentationDataset


OPTIMIZERS = [
    "adamw",
    "adam",
    "sgd",
    "rmsprop",
]


# Classes that represent corrosion
# 0 = Background
# 1 = Lite
# 2 = Moderate
# 3 = Severe
CORROSION_CLASSES = {1, 2, 3}


# ============================================================
# DIRECTORIES
# ============================================================

PLOT_DIR = CHECKPOINT_DIR / "plots"
BINARY_EVAL_DIR = CHECKPOINT_DIR / "binary_evaluation"


# ============================================================
# MODEL
# ============================================================

def build_model():
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )

    return model


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_trained_model(
    optimizer_name,
    device,
):
    checkpoint_path = (
        CHECKPOINT_DIR
        / optimizer_name
        / "best.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    print(
        f"\nLoading {optimizer_name.upper()} "
        f"checkpoint..."
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model = build_model()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    print(
        f"Checkpoint epoch : "
        f"{checkpoint.get('epoch', 'N/A')}"
    )

    print(
        f"Original best mIoU: "
        f"{checkpoint.get('best_iou', 'N/A')}"
    )

    return model


# ============================================================
# BINARY CORROSION IoU
# ============================================================

@torch.no_grad()
def calculate_binary_corrosion_iou(
    model,
    loader,
    device,
):
    """
    Evaluate the already-trained 4-class model as a
    binary corrosion segmentation model.

    Original classes:

        0 = Background
        1 = Lite
        2 = Moderate
        3 = Severe

    For binary evaluation:

        0 = Background
        1 = Corrosion

    Lite + Moderate + Severe are merged ONLY here.
    """

    intersection = 0
    union = 0

    for images, masks, _ in tqdm(
        loader,
        desc="Binary IoU",
        leave=False,
    ):

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Model still produces 4 classes
        # ----------------------------------------------------

        logits = model(images)

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        # ----------------------------------------------------
        # Merge Lite + Moderate + Severe
        #
        # 0 -> Background
        # 1,2,3 -> Corrosion
        # ----------------------------------------------------

        predicted_corrosion = torch.zeros_like(
            predictions,
            dtype=torch.bool,
        )

        target_corrosion = torch.zeros_like(
            masks,
            dtype=torch.bool,
        )

        for cls in CORROSION_CLASSES:

            predicted_corrosion |= (
                predictions == cls
            )

            target_corrosion |= (
                masks == cls
            )

        # ----------------------------------------------------
        # IoU
        # ----------------------------------------------------

        batch_intersection = (
            predicted_corrosion
            & target_corrosion
        ).sum().item()

        batch_union = (
            predicted_corrosion
            | target_corrosion
        ).sum().item()

        intersection += batch_intersection
        union += batch_union

    if union == 0:
        return 1.0

    return intersection / union


# ============================================================
# EVALUATE ALL OPTIMIZERS
# ============================================================

def evaluate_binary_iou(
    val_loader,
    device,
):
    results = []

    print("\n")
    print("=" * 70)
    print("BINARY CORROSION EVALUATION")
    print("=" * 70)

    for optimizer_name in OPTIMIZERS:

        model = load_trained_model(
            optimizer_name,
            device,
        )

        binary_iou = calculate_binary_corrosion_iou(
            model=model,
            loader=val_loader,
            device=device,
        )

        print(
            f"{optimizer_name.upper():10s} "
            f"-> Binary Corrosion IoU: "
            f"{binary_iou:.4f}"
        )

        results.append(
            {
                "optimizer": optimizer_name.upper(),
                "binary_corrosion_iou": binary_iou,
            }
        )

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(results)


# ============================================================
# BINARY IoU BAR GRAPH
# ============================================================

def plot_binary_iou(
    results,
):
    plt.figure(
        figsize=(10, 6)
    )

    bars = plt.bar(
        results["optimizer"],
        results["binary_corrosion_iou"],
    )

    plt.xlabel(
        "Optimizer"
    )

    plt.ylabel(
        "Binary Corrosion IoU"
    )

    plt.title(
        "Binary Corrosion IoU by Optimizer"
    )

    # IoU is between 0 and 1
    plt.ylim(
        0,
        1,
    )

    # Value above every bar
    for bar, score in zip(
        bars,
        results["binary_corrosion_iou"],
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height()
            + 0.02,

            f"{score:.4f}",

            ha="center",
            fontsize=11,
        )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    output_path = (
        BINARY_EVAL_DIR
        / "binary_corrosion_iou_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    print(
        f"\n[SAVED] {output_path}"
    )


# ============================================================
# COMBINED SUMMARY
# ============================================================

def create_combined_summary(
    binary_results,
):
    """
    Combine the original 4-class mIoU from metrics.csv
    with the newly calculated binary corrosion IoU.
    """

    rows = []

    for optimizer in OPTIMIZERS:

        csv_path = (
            CHECKPOINT_DIR
            / optimizer
            / "metrics.csv"
        )

        if not csv_path.exists():
            continue

        df = pd.read_csv(
            csv_path
        )

        best_miou = df[
            "val_miou"
        ].max()

        best_epoch = int(
            df.loc[
                df["val_miou"].idxmax(),
                "epoch",
            ]
        )

        binary_row = binary_results[
            binary_results["optimizer"]
            == optimizer.upper()
        ]

        binary_iou = binary_row[
            "binary_corrosion_iou"
        ].iloc[0]

        rows.append(
            {
                "optimizer": optimizer.upper(),
                "best_epoch": best_epoch,
                "4_class_miou": best_miou,
                "binary_corrosion_iou": binary_iou,
            }
        )

    summary = pd.DataFrame(
        rows
    )

    summary = summary.sort_values(
        "binary_corrosion_iou",
        ascending=False,
    )

    output_path = (
        BINARY_EVAL_DIR
        / "optimizer_binary_summary.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    print("\n")
    print("=" * 70)
    print("FINAL OPTIMIZER COMPARISON")
    print("=" * 70)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        f"\n[SAVED] {output_path}"
    )

    return summary


# ============================================================
# ORIGINAL mIoU PLOTS
# ============================================================

def load_results():
    results = {}

    for optimizer in OPTIMIZERS:

        csv_path = (
            CHECKPOINT_DIR
            / optimizer
            / "metrics.csv"
        )

        if not csv_path.exists():
            print(
                f"[WARNING] Missing: {csv_path}"
            )
            continue

        df = pd.read_csv(
            csv_path
        )

        display_name = optimizer.upper()

        results[display_name] = df

        print(
            f"[OK] {display_name:<8} "
            f"{len(df):>3} epochs | "
            f"best mIoU = "
            f"{df['val_miou'].max():.4f}"
        )

    if not results:
        raise RuntimeError(
            "No metrics.csv files found."
        )

    return results


def plot_metric(
    results,
    column,
    ylabel,
    title,
    filename,
):
    plt.figure(
        figsize=(12, 7)
    )

    for optimizer, df in results.items():

        plt.plot(
            df["epoch"],
            df[column],
            linewidth=2,
            label=optimizer,
        )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        PLOT_DIR
        / filename
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    print(
        f"[SAVED] {output_path}"
    )


def plot_best_miou(
    results,
):
    optimizers = []
    best_scores = []

    for optimizer, df in results.items():

        optimizers.append(
            optimizer
        )

        best_scores.append(
            df["val_miou"].max()
        )

    plt.figure(
        figsize=(10, 6)
    )

    bars = plt.bar(
        optimizers,
        best_scores,
    )

    plt.xlabel(
        "Optimizer"
    )

    plt.ylabel(
        "Best Validation mIoU"
    )

    plt.title(
        "Best Validation mIoU by Optimizer"
    )

    plt.ylim(
        0,
        1,
    )

    for bar, score in zip(
        bars,
        best_scores,
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height()
            + 0.015,

            f"{score:.4f}",

            ha="center",
            fontsize=11,
        )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    output_path = (
        PLOT_DIR
        / "best_miou_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    print(
        f"[SAVED] {output_path}"
    )


def create_original_summary(
    results,
):
    rows = []

    for optimizer, df in results.items():

        best_idx = df[
            "val_miou"
        ].idxmax()

        best_row = df.loc[
            best_idx
        ]

        rows.append(
            {
                "optimizer": optimizer,
                "best_epoch": int(
                    best_row["epoch"]
                ),
                "best_val_miou":
                    best_row["val_miou"],
                "train_miou_at_best":
                    best_row["train_miou"],
                "val_loss_at_best":
                    best_row["val_loss"],
                "train_loss_at_best":
                    best_row["train_loss"],
            }
        )

    summary = pd.DataFrame(
        rows
    )

    summary = summary.sort_values(
        "best_val_miou",
        ascending=False,
    )

    output_path = (
        PLOT_DIR
        / "optimizer_summary.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    BINARY_EVAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    val_dataset = CorrosionSegmentationDataset(
        image_dir=VAL_IMAGE_DIR,
        mask_dir=VAL_MASK_DIR,
        color_to_class=COLOR_TO_CLASS,
        transform=__import__(
            "albumentations"
        ).Compose(
            [
                __import__(
                    "albumentations"
                ).Resize(
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                )
            ]
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    # ========================================================
    # PART 1
    # Existing training curves
    # ========================================================

    print("\n")
    print("=" * 70)
    print("LOADING EXISTING TRAINING RESULTS")
    print("=" * 70)

    results = load_results()

    plot_metric(
        results,
        "val_miou",
        "Validation mIoU",
        "Validation mIoU vs Epoch",
        "val_miou_vs_epoch.png",
    )

    plot_metric(
        results,
        "train_miou",
        "Training mIoU",
        "Training mIoU vs Epoch",
        "train_miou_vs_epoch.png",
    )

    plot_metric(
        results,
        "val_loss",
        "Validation Loss",
        "Validation Loss vs Epoch",
        "val_loss_vs_epoch.png",
    )

    plot_metric(
        results,
        "train_loss",
        "Training Loss",
        "Training Loss vs Epoch",
        "train_loss_vs_epoch.png",
    )

    plot_best_miou(
        results
    )

    create_original_summary(
        results
    )

    # ========================================================
    # PART 2
    # NEW: Binary corrosion evaluation
    # ========================================================

    binary_results = evaluate_binary_iou(
        val_loader=val_loader,
        device=device,
    )

    # Save raw binary results
    binary_csv = (
        BINARY_EVAL_DIR
        / "optimizer_binary_iou.csv"
    )

    binary_results.to_csv(
        binary_csv,
        index=False,
    )

    print(
        f"\n[SAVED] {binary_csv}"
    )

    # Plot binary IoU
    plot_binary_iou(
        binary_results
    )

    # Combined summary
    create_combined_summary(
        binary_results
    )

    # ========================================================
    # DONE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("ALL PLOTS AND BINARY EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()