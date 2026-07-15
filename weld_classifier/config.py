"""Configuration for the weld defect classification project."""

from __future__ import annotations

from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5
MAX_ITER = 1000

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.xlsx"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"
RESULTS_DIR = BASE_DIR / "results"

FEATURE_COLUMNS = [
    "CurrentAmp",
    "VoltageV",
    "WeldingSpeedmmmin",
    "GasFlowRate",
    "LIFLockinFrequency",
    "AmplitudeValuemK",
    "PhaseValueo",
    "DefectLength",
    "DefectWidth",
    "DefectDepth",
    "DefectArea",
    "DefectSize",
]

TARGET_COLUMN = "DefectType"
EXPECTED_CLASSES = ["BH", "LF", "LP", "N", "UF"]

PARAM_GRID = {
    "mlp__hidden_layer_sizes": [
        (10,),
        (15,),
        (20,),
        (15, 10),
        (20, 10),
        (30, 15),
    ],
    "mlp__activation": ["relu", "tanh"],
    "mlp__solver": ["adam", "lbfgs"],
    "mlp__alpha": [0.0001, 0.001, 0.01],
}

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
COMPLETE_PIPELINE_PATH = MODELS_DIR / "complete_pipeline.joblib"

GRID_RESULTS_CSV = RESULTS_DIR / "grid_search_results.csv"
METRICS_JSON = RESULTS_DIR / "evaluation_metrics.json"
CONFUSION_MATRIX_CSV = RESULTS_DIR / "confusion_matrix.csv"
TEST_FEATURES_XLSX = RESULTS_DIR / "test_dataset_features.xlsx"
TEST_LABELED_XLSX = RESULTS_DIR / "test_dataset_labeled.xlsx"

CONFUSION_MATRIX_PNG = PLOTS_DIR / "confusion_matrix.png"
ROC_CURVE_PNG = PLOTS_DIR / "roc_curves.png"
PRECISION_RECALL_PNG = PLOTS_DIR / "precision_recall_curves.png"
LOSS_CURVE_PNG = PLOTS_DIR / "training_loss_curve.png"
GRID_RESULTS_PNG = PLOTS_DIR / "grid_search_results.png"
