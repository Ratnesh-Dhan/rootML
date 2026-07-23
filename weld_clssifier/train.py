"""Train and evaluate an MLP classifier for weld defect classification."""

from __future__ import annotations

import json
import logging
import random
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

from config import (
    BEST_MODEL_PATH,
    COMPLETE_PIPELINE_PATH,
    CONFUSION_MATRIX_CSV,
    CONFUSION_MATRIX_PNG,
    CV_SPLITS,
    DATA_PATH,
    EXPECTED_CLASSES,
    FEATURE_COLUMNS,
    GRID_RESULTS_CSV,
    GRID_RESULTS_PNG,
    LABEL_ENCODER_PATH,
    LOSS_CURVE_PNG,
    MAX_ITER,
    METRICS_JSON,
    MODELS_DIR,
    PARAM_GRID,
    PLOTS_DIR,
    PRECISION_RECALL_PNG,
    RANDOM_STATE,
    RESULTS_DIR,
    ROC_CURVE_PNG,
    TARGET_COLUMN,
    TEST_SIZE,
)

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure console logging for training progress."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_random_seeds(seed: int = RANDOM_STATE) -> None:
    """Set reproducibility seeds for Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_directories() -> None:
    """Create output directories if they are missing."""
    for directory in (MODELS_DIR, PLOTS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the Excel dataset and validate the required columns."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Place data.xlsx in the project folder."
        )

    data = pd.read_excel(data_path, engine="openpyxl")
    missing_columns = [
        column
        for column in [*FEATURE_COLUMNS, TARGET_COLUMN]
        if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")

    LOGGER.info("Loaded dataset with %s rows and %s columns.", *data.shape)
    return data


def prepare_features_and_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """Prepare numeric features and encoded class labels."""
    clean_data = data.copy()
    clean_data = clean_data.dropna(subset=[TARGET_COLUMN])

    for column in FEATURE_COLUMNS:
        clean_data[column] = pd.to_numeric(clean_data[column], errors="coerce")

    missing_feature_values = int(clean_data[FEATURE_COLUMNS].isna().sum().sum())
    if missing_feature_values:
        LOGGER.warning(
            "Found %s missing feature values; SimpleImputer will fill them.",
            missing_feature_values,
        )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(clean_data[TARGET_COLUMN].astype(str))
    unknown_classes = sorted(set(label_encoder.classes_) - set(EXPECTED_CLASSES))
    missing_expected = sorted(set(EXPECTED_CLASSES) - set(label_encoder.classes_))

    if unknown_classes:
        LOGGER.warning("Unexpected classes detected: %s", unknown_classes)
    if missing_expected:
        LOGGER.warning("Expected classes absent from dataset: %s", missing_expected)

    return clean_data[FEATURE_COLUMNS], y, label_encoder


def log_class_distribution(y: np.ndarray, label_encoder: LabelEncoder) -> None:
    """Log class counts and simple imbalance diagnostics."""
    class_names = label_encoder.inverse_transform(np.unique(y))
    counts = np.bincount(y, minlength=len(label_encoder.classes_))
    distribution = {
        class_name: int(counts[index])
        for index, class_name in enumerate(label_encoder.classes_)
    }
    LOGGER.info("Class distribution: %s", distribution)

    nonzero_counts = counts[counts > 0]
    if len(nonzero_counts) > 1:
        imbalance_ratio = float(nonzero_counts.max() / nonzero_counts.min())
        LOGGER.info("Class imbalance ratio max/min: %.2f", imbalance_ratio)
        if imbalance_ratio >= 2.0:
            LOGGER.warning("Potential class imbalance detected.")

    LOGGER.debug("Classes present: %s", class_names.tolist())


def build_pipeline() -> Pipeline:
    """Build the sklearn preprocessing and MLP classification pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    max_iter=MAX_ITER,
                    random_state=RANDOM_STATE,
                    early_stopping=False,
                ),
            ),
        ]
    )


def run_grid_search(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
) -> GridSearchCV:
    """Run stratified 5-fold grid search using accuracy."""
    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
        refit=True,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        grid_search.fit(x_train, y_train)

    return grid_search


def save_grid_results(grid_search: GridSearchCV) -> pd.DataFrame:
    """Save GridSearchCV results sorted by rank."""
    results = pd.DataFrame(grid_search.cv_results_)
    results = results.sort_values("rank_test_score")
    results.to_csv(GRID_RESULTS_CSV, index=False)
    LOGGER.info("Saved grid search results to %s.", GRID_RESULTS_CSV)
    return results


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
) -> None:
    """Save confusion matrix plot and CSV values."""
    labels = np.arange(len(label_encoder.classes_))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_df = pd.DataFrame(
        matrix,
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    )
    matrix_df.to_csv(CONFUSION_MATRIX_CSV)

    fig, ax = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=label_encoder.classes_,
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PNG, dpi=300)
    plt.close(fig)


def plot_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_encoder: LabelEncoder,
) -> None:
    """Save one-vs-rest ROC curves for each class."""
    y_bin = label_binarize(y_true, classes=np.arange(len(label_encoder.classes_)))

    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = False
    for class_index, class_name in enumerate(label_encoder.classes_):
        positives = y_bin[:, class_index].sum()
        negatives = len(y_bin) - positives
        if positives == 0 or negatives == 0:
            LOGGER.warning("Skipping ROC for %s; test set lacks both labels.", class_name)
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, class_index], y_score[:, class_index])
        ax.plot(fpr, tpr, linewidth=2, label=class_name)
        plotted = True

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("One-vs-Rest ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right") if plotted else ax.text(0.5, 0.5, "ROC unavailable")
    fig.tight_layout()
    fig.savefig(ROC_CURVE_PNG, dpi=300)
    plt.close(fig)


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_encoder: LabelEncoder,
) -> None:
    """Save one-vs-rest precision-recall curves for each class."""
    y_bin = label_binarize(y_true, classes=np.arange(len(label_encoder.classes_)))

    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = False
    for class_index, class_name in enumerate(label_encoder.classes_):
        positives = y_bin[:, class_index].sum()
        if positives == 0:
            LOGGER.warning("Skipping PR curve for %s; no positives in test set.", class_name)
            continue
        precision, recall, _ = precision_recall_curve(
            y_bin[:, class_index],
            y_score[:, class_index],
        )
        ax.plot(recall, precision, linewidth=2, label=class_name)
        plotted = True

    ax.set_title("One-vs-Rest Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left") if plotted else ax.text(0.5, 0.5, "PR unavailable")
    fig.tight_layout()
    fig.savefig(PRECISION_RECALL_PNG, dpi=300)
    plt.close(fig)


def plot_training_loss(best_pipeline: Pipeline) -> None:
    """Save the MLP training loss curve when available."""
    mlp = best_pipeline.named_steps["mlp"]
    loss_curve = getattr(mlp, "loss_curve_", None)

    fig, ax = plt.subplots(figsize=(8, 5))
    if loss_curve:
        ax.plot(np.arange(1, len(loss_curve) + 1), loss_curve, linewidth=2)
        ax.set_title("Training Loss Curve")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
    else:
        ax.text(
            0.5,
            0.5,
            "Training loss curve is unavailable for the selected solver.",
            ha="center",
            va="center",
            wrap=True,
        )
        ax.set_axis_off()
        LOGGER.warning("Training loss curve unavailable for selected solver.")

    fig.tight_layout()
    fig.savefig(LOSS_CURVE_PNG, dpi=300)
    plt.close(fig)


def plot_grid_results(results: pd.DataFrame) -> None:
    """Save a plot of the best cross-validation accuracy by rank."""
    top_results = results.nsmallest(20, "rank_test_score").copy()
    top_results["model"] = np.arange(1, len(top_results) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        top_results["model"],
        top_results["mean_test_score"],
        yerr=top_results["std_test_score"],
        fmt="o-",
        capsize=4,
    )
    ax.set_title("Top Grid Search Results")
    ax.set_xlabel("Model Rank")
    ax.set_ylabel("Mean CV Accuracy")
    ax.set_xticks(top_results["model"])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(GRID_RESULTS_PNG, dpi=300)
    plt.close(fig)


def evaluate_model(
    best_pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    grid_search: GridSearchCV,
) -> dict[str, Any]:
    """Evaluate the selected model on the held-out test split."""
    y_pred = best_pipeline.predict(x_test)
    y_score = best_pipeline.predict_proba(x_test)
    target_names = label_encoder.classes_.tolist()

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=target_names,
        zero_division=0,
    )

    metrics: dict[str, Any] = {
        "best_hyperparameters": grid_search.best_params_,
        "best_cv_accuracy": float(grid_search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "classification_report": report,
    }

    LOGGER.info("Best hyperparameters: %s", grid_search.best_params_)
    LOGGER.info("Best cross-validation accuracy: %.4f", grid_search.best_score_)
    LOGGER.info("Test accuracy: %.4f", metrics["test_accuracy"])
    LOGGER.info("Precision macro: %.4f", metrics["precision_macro"])
    LOGGER.info("Recall macro: %.4f", metrics["recall_macro"])
    LOGGER.info("F1-score macro: %.4f", metrics["f1_macro"])
    LOGGER.info("Classification report:\n%s", report_text)

    plot_confusion_matrix(y_test, y_pred, label_encoder)
    plot_roc_curves(y_test, y_score, label_encoder)
    plot_precision_recall_curves(y_test, y_score, label_encoder)
    plot_training_loss(best_pipeline)

    with METRICS_JSON.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)
    LOGGER.info("Saved evaluation metrics to %s.", METRICS_JSON)

    return metrics


def save_artifacts(best_pipeline: Pipeline, label_encoder: LabelEncoder) -> None:
    """Save trained model artifacts."""
    joblib.dump(best_pipeline.named_steps["mlp"], BEST_MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    joblib.dump(best_pipeline, COMPLETE_PIPELINE_PATH)
    LOGGER.info("Saved best MLP model to %s.", BEST_MODEL_PATH)
    LOGGER.info("Saved label encoder to %s.", LABEL_ENCODER_PATH)
    LOGGER.info("Saved complete pipeline to %s.", COMPLETE_PIPELINE_PATH)


def main() -> None:
    """Run the complete training, tuning, evaluation, and artifact pipeline."""
    configure_logging()
    set_random_seeds()
    ensure_directories()

    try:
        data = load_dataset()
        x, y, label_encoder = prepare_features_and_target(data)
        log_class_distribution(y, label_encoder)

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
            stratify=y,
        )
        LOGGER.info("Training samples: %s | Test samples: %s", len(x_train), len(x_test))

        pipeline = build_pipeline()
        LOGGER.info("Starting GridSearchCV with stratified %s-fold CV.", CV_SPLITS)
        grid_search = run_grid_search(pipeline, x_train, y_train)
        results = save_grid_results(grid_search)
        plot_grid_results(results)

        best_pipeline = grid_search.best_estimator_
        evaluate_model(best_pipeline, x_test, y_test, label_encoder, grid_search)
        save_artifacts(best_pipeline, label_encoder)
        LOGGER.info("Training pipeline completed successfully.")
    except Exception:
        LOGGER.exception("Training failed.")
        raise


if __name__ == "__main__":
    main()
