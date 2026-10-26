
import json
import argparse
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True,
                        help="CSV containing new steel records")

    parser.add_argument("--model-dir", default=".",
                        help="Directory containing the saved model")

    parser.add_argument("--output", default="predictions.csv",
                        help="Output CSV path")

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Load the model and its metadata.
    model = XGBRegressor()
    model.load_model(str(
        model_dir / "fatigue_strength_xgb.json"
    ))

    metadata = json.loads(
        (model_dir / "fatigue_strength_metadata.json")
        .read_text(encoding="utf-8")
    )

    features = metadata["features"]

    # Load new steel records.
    df = pd.read_csv(args.input)

    # Check that all required input columns exist.
    missing = [col for col in features if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Preserve the exact feature order used during training.
    X = df[features].apply(pd.to_numeric, errors="raise")

    if X.isna().any().any():
        raise ValueError("Input contains missing feature values.")

    # Predict fatigue strength in the target's original units.
    df["Predicted Fatigue Strength (MPa)"] = model.predict(X)

    df.to_csv(args.output, index=False)

    print(f"Predictions saved to: {args.output}")
    print(df[["Predicted Fatigue Strength (MPa)"]])


if __name__ == "__main__":
    main()
