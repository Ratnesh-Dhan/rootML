
import os
import json
import copy
import random

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

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from model import CNN

from dataloader import (
    get_datasets,
    make_loader,
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 64
EPOCHS = 100

N_SPLITS = 5
SEED = 42

NUM_WORKERS = 4

EARLY_STOPPING_PATIENCE = 12

LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 3
MIN_LR = 1e-6

WEIGHT_DECAY = 1e-4

BASE_MODEL_DIR = "./models_5fold"
BASE_RESULT_DIR = "./results_5fold"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# ============================================================
# DEVICE INFORMATION
# ============================================================

print("=" * 70)
print("DEVICE INFORMATION")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

print()


# ============================================================
# DATASET
# ============================================================

development_dataset, test_dataset = get_datasets()

class_names = development_dataset.classes
class_indices = development_dataset.class_to_idx

num_classes = len(class_names)

if num_classes != 5:
    raise ValueError(
        f"Expected 5 classes, found {num_classes}."
    )

# Extract labels for stratified cross-validation.
development_labels = np.array(
    development_dataset.targets
)

print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(f"Classes: {class_names}")
print(f"Class mapping: {class_indices}")

print(
    f"Development samples: "
    f"{len(development_dataset)}"
)

print(
    f"Test samples: "
    f"{len(test_dataset)}"
)

print()


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

os.makedirs(BASE_MODEL_DIR, exist_ok=True)
os.makedirs(BASE_RESULT_DIR, exist_ok=True)


# ============================================================
# OPTIMIZER CREATION
# ============================================================

def create_optimizer(name, model):

    parameters = model.parameters()

    if name == "Adam":
        return Adam(
            parameters,
            lr=0.001,
            weight_decay=WEIGHT_DECAY
        )

    elif name == "Adagrad":
        return Adagrad(
            parameters,
            lr=0.01,
            weight_decay=WEIGHT_DECAY
        )

    elif name == "RMSprop":
        return RMSprop(
            parameters,
            lr=0.001,
            weight_decay=WEIGHT_DECAY
        )

    elif name == "Adadelta":
        return Adadelta(
            parameters,
            lr=1.0,
            weight_decay=WEIGHT_DECAY
        )

    elif name == "Nadam":
        return NAdam(
            parameters,
            lr=0.001,
            weight_decay=WEIGHT_DECAY
        )

    elif name == "AdamW":
        return AdamW(
            parameters,
            lr=0.001,
            weight_decay=WEIGHT_DECAY
        )

    else:
        raise ValueError(
            f"Unknown optimizer: {name}"
        )


# ============================================================
# TRAINING FUNCTION
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

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size

        predictions = torch.argmax(outputs, dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    return (
        running_loss / total,
        correct / total,
    )


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate(
    model,
    loader,
    criterion,
):

    model.eval()

    running_loss = 0.0
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

            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size
            total += batch_size

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    return (
        running_loss / total,
        np.array(all_labels),
        np.array(all_predictions),
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(true_labels, predictions):

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


# ============================================================
# CROSS-VALIDATION SETUP
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)

criterion = nn.CrossEntropyLoss()

# Results for every optimizer and fold.
all_results = {
    name: [] for name in OPTIMIZER_NAMES
}

# Keep checkpoint paths for the final ensemble.
checkpoint_paths = {
    name: [] for name in OPTIMIZER_NAMES
}


# ============================================================
# FIVE-FOLD CROSS-VALIDATION
# ============================================================

for fold, (train_indices, val_indices) in enumerate(
    skf.split(
        np.zeros(len(development_labels)),
        development_labels
    ),
    start=1
):

    print("\n")
    print("=" * 70)
    print(f"FOLD {fold}/{N_SPLITS}")
    print("=" * 70)

    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")

    fold_seed = SEED + fold



    # --------------------------------------------------------
    # Create common initial weights for this fold.
    # All six optimizers start from these exact weights.
    # --------------------------------------------------------

    seed_everything(fold_seed)

    base_model = CNN(
        num_classes=num_classes
    )

    initial_weights = copy.deepcopy(
        base_model.state_dict()
    )

    del base_model

    # ========================================================
    # OPTIMIZER COMPARISON
    # ========================================================

    for optimizer_name in OPTIMIZER_NAMES:

        print("\n")
        print("-" * 70)

        print(
            f"Fold {fold} | Optimizer: {optimizer_name}"
        )

        print("-" * 70)

        result_dir = os.path.join(
            BASE_RESULT_DIR,
            optimizer_name,
            f"fold_{fold}"
        )

        model_dir = os.path.join(
            BASE_MODEL_DIR,
            optimizer_name,
            f"fold_{fold}"
        )

        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # --------------------------------------------------------
        # Create fold loaders
        # --------------------------------------------------------

        train_loader = make_loader(
            development_dataset,
            indices=train_indices,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            seed=fold_seed,
        )

        validation_loader = make_loader(
            development_dataset,
            indices=val_indices,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            seed=fold_seed,
        )

        # ----------------------------------------------------
        # Reset random state for comparable runs.
        # ----------------------------------------------------

        seed_everything(fold_seed)

        # ----------------------------------------------------
        # Create model and load common initial weights.
        # ----------------------------------------------------

        model = CNN(
            num_classes=num_classes
        )

        model.load_state_dict(initial_weights)

        model = model.to(DEVICE)

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer = create_optimizer(
            optimizer_name,
            model
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=MIN_LR,
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history = {
            "epoch": [],
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }

        best_val_loss = float("inf")
        best_epoch = 0

        epochs_without_improvement = 0

        best_model_path = os.path.join(
            model_dir,
            "best_model.pth"
        )

        # ====================================================
        # EPOCH LOOP
        # ====================================================

        for epoch in range(EPOCHS):

            epoch_number = epoch + 1

            print(
                f"\nEpoch {epoch_number}/{EPOCHS}"
            )

            # -----------------------------------------------
            # Training
            # -----------------------------------------------

            train_loss, train_accuracy = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
            )

            # -----------------------------------------------
            # Validation
            # -----------------------------------------------
            val_loss, val_labels, val_predictions = validate(
                model,
                validation_loader,
                criterion,
            )

            val_accuracy = accuracy_score(
                val_labels,
                val_predictions
            )

            val_accuracy = accuracy_score(
                val_labels,
                val_predictions
            )

            # -----------------------------------------------
            # Learning rate
            # -----------------------------------------------

            current_lr = optimizer.param_groups[0]["lr"]

            # -----------------------------------------------
            # Save history
            # -----------------------------------------------

            history["epoch"].append(epoch_number)
            history["loss"].append(train_loss)
            history["accuracy"].append(train_accuracy)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)
            history["learning_rate"].append(current_lr)

            # -----------------------------------------------
            # Print metrics
            # -----------------------------------------------

            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Train Accuracy: {train_accuracy:.4f}"
            )

            print(
                f"Val Loss: {val_loss:.4f} | "
                f"Val Accuracy: {val_accuracy:.4f}"
            )

            print(
                f"Learning Rate: {current_lr:.8f}"
            )

            # -----------------------------------------------
            # Save best checkpoint
            # -----------------------------------------------

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

                print("Best model saved.")

            else:

                epochs_without_improvement += 1

            # -----------------------------------------------
            # Scheduler
            # -----------------------------------------------

            scheduler.step(val_loss)

            # -----------------------------------------------
            # Early stopping
            # -----------------------------------------------

            if (
                epochs_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):

                print(
                    f"Early stopping at epoch {epoch_number}."
                )

                print(
                    f"Best epoch: {best_epoch}"
                )

                break

        # ====================================================
        # SAVE TRAINING HISTORY
        # ====================================================

        history_df = pd.DataFrame(history)

        history_df.to_csv(
            os.path.join(
                result_dir,
                "training_history.csv"
            ),
            index=False
        )

        # ====================================================
        # PLOT ACCURACY
        # ====================================================

        plt.figure(figsize=(8, 6))

        plt.plot(
            history["epoch"],
            history["accuracy"],
            label="Training Accuracy"
        )

        plt.plot(
            history["epoch"],
            history["val_accuracy"],
            label="Validation Accuracy"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.title(
            f"{optimizer_name} - Fold {fold} Accuracy"
        )

        plt.legend()
        plt.grid()

        plt.savefig(
            os.path.join(
                result_dir,
                "accuracy.png"
            ),
            bbox_inches="tight"
        )

        plt.close()

        # ====================================================
        # PLOT LOSS
        # ====================================================

        plt.figure(figsize=(8, 6))

        plt.plot(
            history["epoch"],
            history["loss"],
            label="Training Loss"
        )

        plt.plot(
            history["epoch"],
            history["val_loss"],
            label="Validation Loss"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.title(
            f"{optimizer_name} - Fold {fold} Loss"
        )

        plt.legend()
        plt.grid()

        plt.savefig(
            os.path.join(
                result_dir,
                "loss.png"
            ),
            bbox_inches="tight"
        )

        plt.close()

        # ====================================================
        # LOAD BEST MODEL
        # ====================================================

        checkpoint = torch.load(
            best_model_path,
            map_location=DEVICE
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # ====================================================
        # FINAL FOLD VALIDATION
        # ====================================================

        (
            final_val_loss,
            true_labels,
            predicted_classes,
        ) = validate(
            model,
            validation_loader,
            criterion,
        )

        metrics = calculate_metrics(
            true_labels,
            predicted_classes
        )

        metrics["fold"] = fold
        metrics["best_epoch"] = best_epoch
        metrics["val_loss"] = float(final_val_loss)

        all_results[optimizer_name].append(metrics)

        checkpoint_paths[optimizer_name].append(
            best_model_path
        )

        print("\nFold results:")

        for key, value in metrics.items():
            print(f"{key}: {value}")

        # ====================================================
        # SAVE FOLD CLASSIFICATION REPORT
        # ====================================================

        report = classification_report(
            true_labels,
            predicted_classes,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )

        with open(
            os.path.join(
                result_dir,
                "classification_report.txt"
            ),
            "w"
        ) as f:

            f.write(report)

        # ====================================================
        # SAVE FOLD CONFUSION MATRIX
        # ====================================================

        cm = confusion_matrix(
            true_labels,
            predicted_classes,
            labels=np.arange(num_classes)
        )

        plt.figure(figsize=(7, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.title(
            f"{optimizer_name} - Fold {fold}"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                result_dir,
                "confusion_matrix.png"
            )
        )

        plt.close()

        del model
        del optimizer
        del scheduler

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# CROSS-VALIDATION SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("CROSS-VALIDATION SUMMARY")
print("=" * 70)

summary = {}

for optimizer_name, fold_results in all_results.items():

    df = pd.DataFrame(fold_results)

    metrics_to_summarize = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "val_loss",
    ]

    summary[optimizer_name] = {}

    for metric in metrics_to_summarize:

        mean_value = float(df[metric].mean())
        std_value = float(df[metric].std(ddof=1))

        summary[optimizer_name][metric] = {
            "mean": mean_value,
            "std": std_value,
        }

    print(f"\n{optimizer_name}")

    for metric in metrics_to_summarize:

        mean_value = summary[optimizer_name][metric]["mean"]
        std_value = summary[optimizer_name][metric]["std"]

        print(
            f"{metric}: {mean_value:.4f} ± {std_value:.4f}"
        )

    # Save individual optimizer fold results.
    df.to_csv(
        os.path.join(
            BASE_RESULT_DIR,
            f"{optimizer_name}_fold_results.csv"
        ),
        index=False
    )


# ============================================================
# SELECT BEST OPTIMIZER
# ============================================================

best_optimizer = max(
    summary,
    key=lambda name: summary[name]["f1_macro"]["mean"]
)

print("\n")
print("=" * 70)
print("SELECTED OPTIMIZER")
print("=" * 70)

print(f"Selected optimizer: {best_optimizer}")

print(
    "Mean CV macro F1: "
    f"{summary[best_optimizer]['f1_macro']['mean']:.4f}"
)


# ============================================================
# SAVE CROSS-VALIDATION SUMMARY
# ============================================================

with open(
    os.path.join(
        BASE_RESULT_DIR,
        "cross_validation_summary.json"
    ),
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


with open(
    os.path.join(
        BASE_RESULT_DIR,
        "selected_optimizer.json"
    ),
    "w"
) as f:

    json.dump(
        {
            "selected_optimizer": best_optimizer,
            "selection_metric": "mean_macro_f1",
            "mean_macro_f1": summary[best_optimizer]["f1_macro"]["mean"],
        },
        f,
        indent=4
    )


# ============================================================
# FINAL TEST EVALUATION USING FIVE-FOLD ENSEMBLE
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

print(
    f"Using five-fold ensemble of {best_optimizer} models."
)

test_loader = make_loader(
    test_dataset,
    indices=None,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    seed=SEED,
)

ensemble_probabilities = []

for fold, checkpoint_path in enumerate(
    checkpoint_paths[best_optimizer],
    start=1
):

    print(f"Loading fold {fold} model...")

    model = CNN(
        num_classes=num_classes
    ).to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    fold_probabilities = []

    with torch.no_grad():

        for images, _ in test_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            fold_probabilities.append(
                probabilities.cpu().numpy()
            )

    fold_probabilities = np.concatenate(
        fold_probabilities,
        axis=0
    )

    ensemble_probabilities.append(
        fold_probabilities
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Average the predictions from all five fold models.
ensemble_probabilities = np.mean(
    np.stack(ensemble_probabilities, axis=0),
    axis=0
)

test_predictions = np.argmax(
    ensemble_probabilities,
    axis=1
)

test_labels = np.array(
    test_dataset.targets
)

# ============================================================
# FINAL TEST METRICS
# ============================================================

test_metrics = calculate_metrics(
    test_labels,
    test_predictions
)

print("\nFinal test results:")

for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")


# ============================================================
# TEST CLASSIFICATION REPORT
# ============================================================

test_report = classification_report(
    test_labels,
    test_predictions,
    target_names=class_names,
    digits=4,
    zero_division=0,
)

with open(
    os.path.join(
        BASE_RESULT_DIR,
        "final_test_classification_report.txt"
    ),
    "w"
) as f:

    f.write(test_report)


# ============================================================
# TEST CONFUSION MATRIX
# ============================================================

test_cm = confusion_matrix(
    test_labels,
    test_predictions,
    labels=np.arange(num_classes)
)

plt.figure(figsize=(7, 6))

sns.heatmap(
    test_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    f"Final Test Confusion Matrix - {best_optimizer} Ensemble"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_RESULT_DIR,
        "final_test_confusion_matrix.png"
    )
)

plt.close()


# ============================================================
# SAVE FINAL TEST RESULTS
# ============================================================

with open(
    os.path.join(
        BASE_RESULT_DIR,
        "final_test_results.json"
    ),
    "w"
) as f:

    json.dump(
        {
            "selected_optimizer": best_optimizer,
            "test_metrics": test_metrics,
            "test_samples": len(test_dataset),
            "ensemble_models": N_SPLITS,
        },
        f,
        indent=4
    )


# ============================================================
# COMPLETED
# ============================================================

print("\n")
print("=" * 70)
print("ALL EXPERIMENTS COMPLETED")
print("=" * 70)

print(f"Selected optimizer: {best_optimizer}")

print("\nFinal test metrics:")

for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

print(f"\nResults saved in: {BASE_RESULT_DIR}")
print(f"Models saved in: {BASE_MODEL_DIR}")
