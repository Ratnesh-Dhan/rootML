import os
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

from torch.optim import (
    Adam,
    Adagrad,
    RMSprop,
    Adadelta,
    AdamW,
    NAdam,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix

from model import CNN
from dataLoader import load_dataset


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 64
EPOCHS = 100

BASE_MODEL_DIR = "./models_aug24_2026_100_epochs"
BASE_RESULT_DIR = "./results_aug24_2026_100_epochs"

# EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_PATIENCE = 100

LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# GPU information
# ============================================================

print("=" * 70)
print("DEVICE INFORMATION")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"CUDA version: {torch.version.cuda}"
    )

print()


# ============================================================
# Dataset
# ============================================================

train_loader, validation_loader = load_dataset(
    batch_size=BATCH_SIZE
)

# num_classes = len(
#     train_loader.dataset.dataset.classes
# )

# class_names = (
#     train_loader
#     .dataset
#     .dataset
#     .classes
# )
num_classes = len(train_loader.dataset.classes)
class_names = train_loader.dataset.classes

class_indices = {
    name: index
    for index, name in enumerate(class_names)
}

print("=" * 70)
print("DATASET")
print("=" * 70)

print(f"Classes: {class_names}")
print(f"Class mapping: {class_indices}")

print(
    f"Training samples: "
    f"{len(train_loader.dataset)}"
)

print(
    f"Validation samples: "
    f"{len(validation_loader.dataset)}"
)

print()


# ============================================================
# Optimizers
# ============================================================

def create_optimizer(name, model):

    if name == "Adam":
        return Adam(model.parameters(), lr=0.001)

    elif name == "Adagrad":
        return Adagrad(model.parameters(), lr=0.01)

    elif name == "RMSprop":
        return RMSprop(model.parameters(), lr=0.001)

    elif name == "Adadelta":
        return Adadelta(model.parameters(), lr=1.0)

    elif name == "Nadam":
        return NAdam(model.parameters(), lr=0.001)
        # return torch.optim.NAdam(model.parameters())

    elif name == "AdamW":
        return AdamW(model.parameters(), lr=0.001)

    else:
        raise ValueError(
            f"Unknown optimizer: {name}"
        )


OPTIMIZER_NAMES = [
    "Adam",
    "Adagrad",
    "RMSprop",
    "Adadelta",
    "Nadam",
    "AdamW",
]


# ============================================================
# Training function
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        # Clear previous gradients
        optimizer.zero_grad(
            set_to_none=True
        )

        # Forward pass
        outputs = model(images)

        # Loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        batch_size = labels.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# Validation function
# ============================================================

def validate(
    model,
    loader,
    criterion,
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            batch_size = labels.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += batch_size

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return (
        epoch_loss,
        epoch_accuracy,
        np.array(all_labels),
        np.array(all_predictions),
    )


# ============================================================
# Create directories
# ============================================================

os.makedirs(
    BASE_MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    BASE_RESULT_DIR,
    exist_ok=True
)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# Summary
# ============================================================

summary = {}


# ============================================================
# Optimizer Training Loop
# ============================================================

for optimizer_name in OPTIMIZER_NAMES:

    print("\n")
    print("=" * 70)
    print(
        f"TRAINING WITH {optimizer_name}"
    )
    print("=" * 70)

    model_dir = os.path.join(
        BASE_MODEL_DIR,
        optimizer_name
    )

    result_dir = os.path.join(
        BASE_RESULT_DIR,
        optimizer_name
    )

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    os.makedirs(
        result_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create fresh model
    # --------------------------------------------------------

    model = CNN(
        num_classes=num_classes
    ).to(DEVICE)

    # --------------------------------------------------------
    # Create optimizer
    # --------------------------------------------------------

    optimizer = create_optimizer(
        optimizer_name,
        model
    )

    # --------------------------------------------------------
    # ReduceLROnPlateau
    # --------------------------------------------------------

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_REDUCE_FACTOR,
        patience=LR_REDUCE_PATIENCE,
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = {
        "epoch": [],
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    # --------------------------------------------------------
    # Early stopping variables
    # --------------------------------------------------------

    best_val_loss = float("inf")

    best_epoch = 0

    epochs_without_improvement = 0

    best_model_path = os.path.join(
        model_dir,
        "checkpoint_best_weights.pth"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        epoch_number = epoch + 1

        print(
            f"\nEpoch "
            f"{epoch_number}/{EPOCHS}"
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        (
            val_loss,
            val_accuracy,
            _,
            _,
        ) = validate(
            model,
            validation_loader,
            criterion
        )

        # ----------------------------------------------------
        # Current LR
        # ----------------------------------------------------

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # Store history
        # ----------------------------------------------------

        history["epoch"].append(
            epoch_number
        )

        history["loss"].append(
            train_loss
        )

        history["accuracy"].append(
            train_accuracy
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        history["learning_rate"].append(
            current_lr
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f}"
        )

        print(
            f"Val Loss:   {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        print(
            f"Learning Rate: {current_lr:.8f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_epoch = epoch_number

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch_number,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "class_to_idx": class_indices,
                },
                best_model_path
            )

            print(
                "✓ Best model saved."
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Reduce LR
        # ----------------------------------------------------

        scheduler.step(val_loss)

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                f"\nEarly stopping triggered."
            )

            print(
                f"Best epoch: {best_epoch}"
            )

            break

    # ========================================================
    # Save History CSV
    # ========================================================

    history_df = pd.DataFrame(history)

    history_csv = os.path.join(
        result_dir,
        "training_history.csv"
    )

    history_df.to_csv(
        history_csv,
        index=False
    )

    print(
        f"\n✓ Training history saved:"
        f"\n  {history_csv}"
    )

    # ========================================================
    # Save class mapping
    # ========================================================

    with open(
        os.path.join(
            result_dir,
            "class_indices.json"
        ),
        "w"
    ) as f:

        json.dump(
            class_indices,
            f,
            indent=4
        )

    # ========================================================
    # Individual Accuracy Plot
    # ========================================================

    plt.figure(figsize=(8, 6))

    plt.plot(
        history["epoch"],
        history["accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        history["epoch"],
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title(
        f"{optimizer_name} - Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.grid()

    plt.savefig(
        os.path.join(
            result_dir,
            "accuracy_vs_epochs.png"
        ),
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # Individual Loss Plot
    # ========================================================

    plt.figure(figsize=(8, 6))

    plt.plot(
        history["epoch"],
        history["loss"],
        label="Train Loss"
    )

    plt.plot(
        history["epoch"],
        history["val_loss"],
        label="Validation Loss"
    )

    plt.title(
        f"{optimizer_name} - Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid()

    plt.savefig(
        os.path.join(
            result_dir,
            "loss_vs_epochs.png"
        ),
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # Load BEST model for final evaluation
    # ========================================================

    checkpoint = torch.load(
        best_model_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # ========================================================
    # Final Validation
    # ========================================================

    (
        final_val_loss,
        final_val_accuracy,
        true_labels,
        predicted_classes,
    ) = validate(
        model,
        validation_loader,
        criterion
    )

    print(
        f"\n✓ {optimizer_name} Best Validation Results:"
    )

    print(
        f"  Epoch    : {best_epoch}"
    )

    print(
        f"  Val Loss : {final_val_loss:.4f}"
    )

    print(
        f"  Val Acc  : {final_val_accuracy:.4f}"
    )

    # ========================================================
    # Save final/best model
    # ========================================================

    final_model_path = os.path.join(
        model_dir,
        f"{optimizer_name}_earlystopped_best_epoch{best_epoch}.pth"
    )

    torch.save(
        {
            "epoch": best_epoch,
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_indices,
            "val_loss": final_val_loss,
            "val_accuracy": final_val_accuracy,
        },
        final_model_path
    )

    # ========================================================
    # Classification Report
    # ========================================================

    report = classification_report(
        true_labels,
        predicted_classes,
        target_names=class_names,
        digits=4
    )

    cm = confusion_matrix(
        true_labels,
        predicted_classes
    )

    with open(
        os.path.join(
            result_dir,
            "classification_report.txt"
        ),
        "w"
    ) as f:

        f.write(
            "Classification Report:\n"
        )

        f.write(report)

        f.write(
            "\n\nConfusion Matrix:\n"
        )

        f.write(
            str(cm)
        )

    # ========================================================
    # Confusion Matrix - Counts
    # ========================================================

    plt.figure(
        figsize=(7, 6)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.title(
        f"{optimizer_name} - Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            result_dir,
            "confusion_matrix_counts.png"
        )
    )

    plt.close()

    # ========================================================
    # Confusion Matrix - Percentage
    # ========================================================

    cm_percent = (
        cm.astype(float)
        /
        cm.sum(axis=1, keepdims=True)
        * 100
    )

    np.set_printoptions(
        precision=2
    )

    with open(
        os.path.join(
            result_dir,
            "confusion_matrix_percent.txt"
        ),
        "w"
    ) as f:

        f.write(
            str(cm_percent)
        )

    plt.figure(
        figsize=(7, 6)
    )

    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.title(
        f"{optimizer_name} - Confusion Matrix (%)"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            result_dir,
            "confusion_matrix_percent.png"
        )
    )

    plt.close()

    # ========================================================
    # Summary
    # ========================================================

    summary[optimizer_name] = {
        "best_epoch": best_epoch,
        "val_accuracy": float(
            final_val_accuracy
        ),
        "val_loss": float(
            final_val_loss
        ),
    }


# ============================================================
# Save Summary JSON
# ============================================================

with open(
    os.path.join(
        BASE_RESULT_DIR,
        "summary.json"
    ),
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# Print Summary
# ============================================================

print("\n")
print("=" * 70)
print("OPTIMIZER COMPARISON")
print("=" * 70)

for optimizer_name, result in summary.items():

    print(
        f"{optimizer_name:<10} | "
        f"Best Epoch: {result['best_epoch']:<4} | "
        f"Accuracy: {result['val_accuracy']:.4f} | "
        f"Loss: {result['val_loss']:.4f}"
    )