"""Run predictions with the saved weld defect MLP pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from config import COMPLETE_PIPELINE_PATH, FEATURE_COLUMNS, LABEL_ENCODER_PATH

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_features(input_path: Path) -> pd.DataFrame:
    """Load feature rows from CSV or Excel and validate columns."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        data = pd.read_excel(input_path, engine="openpyxl")
    elif input_path.suffix.lower() == ".csv":
        data = pd.read_csv(input_path)
    else:
        raise ValueError("Prediction input must be a .csv, .xlsx, or .xls file.")

    missing_columns = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Input is missing feature columns: {missing_columns}")

    return data[FEATURE_COLUMNS]


def predict(input_path: Path, output_path: Path | None = None) -> pd.DataFrame:
    """Predict weld defect labels for the provided feature rows."""
    pipeline: Pipeline = joblib.load(COMPLETE_PIPELINE_PATH)
    label_encoder: LabelEncoder = joblib.load(LABEL_ENCODER_PATH)
    features = load_features(input_path)

    encoded_predictions = pipeline.predict(features)
    predicted_labels = label_encoder.inverse_transform(encoded_predictions)
    probabilities = pipeline.predict_proba(features)

    output = features.copy()
    output["PredictedDefectType"] = predicted_labels
    for index, class_name in enumerate(label_encoder.classes_):
        output[f"Probability_{class_name}"] = probabilities[:, index]

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False)
        LOGGER.info("Saved predictions to %s.", output_path)
    else:
        LOGGER.info("Predictions:\n%s", output.to_string(index=False))

    return output


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Predict weld defect classes.")
    parser.add_argument("input", type=Path, help="CSV or Excel file containing features.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for saving predictions.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    configure_logging()
    args = parse_args()

    try:
        predict(args.input, args.output)
    except Exception:
        LOGGER.exception("Prediction failed.")
        raise


if __name__ == "__main__":
    main()
