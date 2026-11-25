# ==========================================================
# SWR PREDICTION USING MULTIPLE MACHINE LEARNING MODELS
# Corrected version with log10(SWR), proper CV, pipelines,
# feature selection after train-test split, SHAP, and plots
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# SKLEARN IMPORTS
# ==========================================================

from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

# ==========================================================
# XGBOOST, CATBOOST, SHAP
# ==========================================================

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import shap

# ==========================================================
# LOAD DATASET
# ==========================================================

# file_path = r"C:\Users\HP\Downloads\ML datasheet_01_06_2026 (1).xlsx"
file_path = r"./ML datasheet_01_06_2026.xlsx"

df = pd.read_excel(file_path, engine="openpyxl")

print("\nDataset Shape:")
print(df.shape)

# Clean column names
df.columns = df.columns.str.strip()

print("\nColumns:")
print(df.columns.tolist())

# ==========================================================
# DEFINE TARGET AND FEATURES
# ==========================================================

target_column = "SWR"

if target_column not in df.columns:
    raise ValueError(f"{target_column} column not found in dataset.")

# Keep only numeric columns
df_numeric = df.select_dtypes(include=np.number)

if target_column not in df_numeric.columns:
    raise ValueError("SWR column is not numeric. Please check the dataset.")

# Remove infinite values
df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)

# Remove rows where target is missing
df_numeric = df_numeric.dropna(subset=[target_column])

# Check SWR values before log transformation
if (df_numeric[target_column] <= 0).any():
    raise ValueError("SWR contains zero or negative values. log10(SWR) cannot be applied.")

# Define X and y
X = df_numeric.drop(columns=[target_column])
y_original = df_numeric[target_column]

# Fill missing input values using column mean
X = X.fillna(X.mean())

# Log transform target
y = np.log10(y_original)

print("\nTarget transformation applied: y = log10(SWR)")
print("Minimum log10(SWR):", y.min())
print("Maximum log10(SWR):", y.max())

# ==========================================================
# TRAIN-TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ==========================================================
# FEATURE SELECTION USING TRAINING DATA ONLY
# ==========================================================

# ---------- Low variance feature removal ----------
variance_selector = VarianceThreshold(threshold=0.01)

X_train_var = variance_selector.fit_transform(X_train)
X_test_var = variance_selector.transform(X_test)

selected_features = X_train.columns[variance_selector.get_support()]

X_train_selected = pd.DataFrame(
    X_train_var,
    columns=selected_features,
    index=X_train.index
)

X_test_selected = pd.DataFrame(
    X_test_var,
    columns=selected_features,
    index=X_test.index
)

print("\nFeatures after low-variance removal:")
print(X_train_selected.columns.tolist())

# ---------- Highly correlated feature removal ----------
corr_matrix = X_train_selected.corr().abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

to_drop = [
    column for column in upper.columns
    if any(upper[column] > 0.90)
]

print("\nDropped highly correlated features:")
print(to_drop)

X_train_final = X_train_selected.drop(columns=to_drop)
X_test_final = X_test_selected.drop(columns=to_drop)

final_features = X_train_final.columns.tolist()

print("\nFinal selected features:")
print(final_features)

# ==========================================================
# CORRELATION HEATMAP OF FINAL TRAINING FEATURES
# ==========================================================

corr_plot_data = X_train_final.copy()
corr_plot_data["log10_SWR"] = y_train

plt.figure(figsize=(18, 14))
sns.heatmap(
    corr_plot_data.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Pearson Correlation"}
)
plt.title("Pearson Correlation Matrix after Feature Selection", fontsize=18, fontweight="bold")
plt.tight_layout()
plt.savefig("Pearson_Correlation_Heatmap.png", dpi=600, bbox_inches="tight")
plt.show()

# ==========================================================
# CROSS-VALIDATION SETUP
# ==========================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ==========================================================
# MODEL DEFINITIONS
# ==========================================================

# Models that require scaling
scaled_models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),

    "Lasso Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.001, max_iter=10000))
    ]),

    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf", C=100, gamma="scale"))
    ])
}

# Tree-based models generally do not require scaling
tree_models = {
    "AdaBoost": AdaBoostRegressor(
        n_estimators=300,
        random_state=42
    ),

    "CatBoost": CatBoostRegressor(
        iterations=500,
        depth=4,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=0,
        random_state=42
    )
}

# ==========================================================
# HYPERPARAMETER TUNING: RANDOM FOREST
# ==========================================================

rf_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions={
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    n_iter=20,
    scoring="r2",
    cv=cv,
    random_state=42,
    n_jobs=-1
)

rf_search.fit(X_train_final, y_train)

rf_model = rf_search.best_estimator_

print("\nBest Random Forest Parameters:")
print(rf_search.best_params_)

# ==========================================================
# HYPERPARAMETER TUNING: XGBOOST
# ==========================================================

xgb_search = RandomizedSearchCV(
    estimator=XGBRegressor(
        objective="reg:squarederror",
        random_state=42
    ),
    param_distributions={
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    },
    n_iter=20,
    scoring="r2",
    cv=cv,
    random_state=42,
    n_jobs=-1
)

xgb_search.fit(X_train_final, y_train)

xgb_model = xgb_search.best_estimator_

print("\nBest XGBoost Parameters:")
print(xgb_search.best_params_)

# Add tuned models
tree_models["Random Forest"] = rf_model
tree_models["XGBoost"] = xgb_model

# Combine all models
models = {}
models.update(scaled_models)
models.update(tree_models)

# ==========================================================
# TRAINING AND EVALUATION
# ==========================================================

results = []
predictions_log = {}
predictions_swr = {}
trained_models = {}

for name, model in models.items():

    print(f"\nTraining model: {name}")

    # Fit model
    model.fit(X_train_final, y_train)

    # Predict log10(SWR)
    y_pred_log = model.predict(X_test_final)

    # Convert log prediction back to original SWR scale
    y_pred_swr = 10 ** y_pred_log
    y_test_swr = 10 ** y_test

    # Save predictions
    predictions_log[name] = y_pred_log
    predictions_swr[name] = y_pred_swr
    trained_models[name] = model

    # Metrics in log scale
    r2_log = r2_score(y_test, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
    mae_log = mean_absolute_error(y_test, y_pred_log)

    # Metrics in original SWR scale
    r2_swr = r2_score(y_test_swr, y_pred_swr)
    rmse_swr = np.sqrt(mean_squared_error(y_test_swr, y_pred_swr))
    mae_swr = mean_absolute_error(y_test_swr, y_pred_swr)

    # Cross-validation R2 in log scale
    cv_r2_scores = cross_val_score(
        model,
        X_train_final,
        y_train,
        cv=cv,
        scoring="r2",
        n_jobs=-1
    )

    cv_r2_mean = cv_r2_scores.mean()
    cv_r2_std = cv_r2_scores.std()

    results.append([
        name,
        r2_log,
        rmse_log,
        mae_log,
        r2_swr,
        rmse_swr,
        mae_swr,
        cv_r2_mean,
        cv_r2_std
    ])

# ==========================================================
# MODEL COMPARISON TABLE
# ==========================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Test_R2_log10",
        "Test_RMSE_log10",
        "Test_MAE_log10",
        "Test_R2_original_SWR",
        "Test_RMSE_original_SWR",
        "Test_MAE_original_SWR",
        "CV_R2_mean_log10",
        "CV_R2_std_log10"
    ]
)

results_df = results_df.sort_values(
    by="Test_R2_log10",
    ascending=False
)

print("\nMODEL COMPARISON:")
print(results_df)

results_df.to_excel("Model_Comparison_Corrected.xlsx", index=False)

# ==========================================================
# BEST MODEL SELECTION
# ==========================================================

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

best_pred_log = predictions_log[best_model_name]
best_pred_swr = predictions_swr[best_model_name]

print("\nBEST MODEL BASED ON TEST R2 IN LOG SCALE:")
print(best_model_name)

# ==========================================================
# PARITY PLOTS: LOG10(SWR)
# ==========================================================

for name, pred_log in predictions_log.items():

    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, pred_log)

    min_v = min(y_test.min(), pred_log.min())
    max_v = max(y_test.max(), pred_log.max())

    plt.plot([min_v, max_v], [min_v, max_v], "r--")

    plt.xlabel("Actual log10(SWR)")
    plt.ylabel("Predicted log10(SWR)")
    plt.title(f"Parity Plot: {name}")
    plt.tight_layout()
    plt.savefig(f"{name}_Parity_log10.png", dpi=600, bbox_inches="tight")
    plt.show()

# ==========================================================
# PARITY PLOTS: ORIGINAL SWR SCALE
# ==========================================================

y_test_swr = 10 ** y_test

for name, pred_swr in predictions_swr.items():

    plt.figure(figsize=(5, 5))
    plt.scatter(y_test_swr, pred_swr)

    min_v = min(y_test_swr.min(), pred_swr.min())
    max_v = max(y_test_swr.max(), pred_swr.max())

    plt.plot([min_v, max_v], [min_v, max_v], "r--")

    plt.xlabel("Actual SWR")
    plt.ylabel("Predicted SWR")
    plt.title(f"Parity Plot in Original Scale: {name}")
    plt.tight_layout()
    plt.savefig(f"{name}_Parity_original_SWR.png", dpi=600, bbox_inches="tight")
    plt.show()

# ==========================================================
# RESIDUAL PLOT FOR BEST MODEL IN LOG SCALE
# ==========================================================

residuals_log = y_test - best_pred_log

plt.figure(figsize=(6, 5))
plt.scatter(best_pred_log, residuals_log)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted log10(SWR)")
plt.ylabel("Residuals")
plt.title(f"Residual Plot: {best_model_name}")
plt.tight_layout()
plt.savefig("Residual_Plot_Best_Model_log10.png", dpi=600, bbox_inches="tight")
plt.show()

# ==========================================================
# FEATURE IMPORTANCE FOR TREE-BASED MODELS
# ==========================================================

imp_df = None

# If best model is a pipeline, extract final estimator
if isinstance(best_model, Pipeline):
    final_estimator = best_model.named_steps["model"]
else:
    final_estimator = best_model

if hasattr(final_estimator, "feature_importances_"):

    imp_df = pd.DataFrame({
        "Feature": final_features,
        "Importance": final_estimator.feature_importances_
    })

    imp_df = imp_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nFeature Importance:")
    print(imp_df)

    imp_df.to_excel("Feature_Importance_Best_Model.xlsx", index=False)

    plt.figure(figsize=(10, 7))
    sns.barplot(
        data=imp_df.head(15),
        x="Importance",
        y="Feature"
    )
    plt.title(f"Feature Importance: {best_model_name}", fontsize=16, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("Feature_Importance_Best_Model.png", dpi=600, bbox_inches="tight")
    plt.show()

else:
    print(f"\nFeature importance is not directly available for {best_model_name}.")

# ==========================================================
# SHAP ANALYSIS
# ==========================================================

try:
    if best_model_name in ["Random Forest", "XGBoost", "CatBoost", "AdaBoost"]:

        print("\nRunning SHAP analysis for best tree-based model...")

        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test_final)

        shap.summary_plot(
            shap_values,
            X_test_final,
            show=True
        )

    else:
        print("\nSHAP skipped because the best model is not a tree-based model.")

except Exception as e:
    print("\nSHAP Error:")
    print(e)

# ==========================================================
# SENSITIVITY ANALYSIS FOR TOP 5 FEATURES
# ==========================================================

if imp_df is not None:

    top_features = imp_df.head(5)["Feature"].tolist()

    for feature in top_features:

        values = np.linspace(
            X_train_final[feature].min(),
            X_train_final[feature].max(),
            100
        )

        temp = pd.DataFrame(
            np.tile(X_train_final.mean().values, (100, 1)),
            columns=final_features
        )

        temp[feature] = values

        pred_log = best_model.predict(temp)
        pred_swr = 10 ** pred_log

        plt.figure(figsize=(6, 4))
        plt.plot(values, pred_swr, linewidth=3)
        plt.xlabel(feature)
        plt.ylabel("Predicted SWR")
        plt.title(f"Sensitivity Analysis: {feature}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"Sensitivity_{feature}.png", dpi=600, bbox_inches="tight")
        plt.show()

else:
    print("\nSensitivity analysis skipped because feature importance is not available.")

# ==========================================================
# SAVE PREDICTIONS
# ==========================================================

pred_df = pd.DataFrame()

pred_df["Actual_log10_SWR"] = y_test.values
pred_df["Actual_SWR"] = 10 ** y_test.values

for model_name in predictions_log:

    clean_name = model_name.replace(" ", "_")

    pred_df[f"{clean_name}_Predicted_log10_SWR"] = predictions_log[model_name]
    pred_df[f"{clean_name}_Predicted_SWR"] = predictions_swr[model_name]

pred_df.to_excel("SWR_Predictions_Corrected.xlsx", index=False)

# ==========================================================
# SAVE FINAL TRAINING FEATURE LIST
# ==========================================================

feature_df = pd.DataFrame({
    "Final_Selected_Features": final_features
})

feature_df.to_excel("Final_Selected_Features.xlsx", index=False)

print("\nALL TASKS COMPLETED SUCCESSFULLY")


# ==========================================================
# SAVE TRAINED MODEL FOR WEB APPLICATION
# ==========================================================

import joblib

# Save the trained model and feature-processing information
model_package = {
    "model": best_model,
    "model_name": best_model_name,
    "original_features": X.columns.tolist(),
    "selected_features": final_features,
    "variance_selector": variance_selector,
    "correlated_features_removed": to_drop,
    "training_feature_means": X_train.mean().to_dict(),
    "target_transform": "log10",
    "target_column": "SWR"
}

joblib.dump(model_package, f"{best_model_name}_SWR_Prediction_Model.joblib")

print("\nMODEL SAVED SUCCESSFULLY!")
print(f"File: {best_model_name}_SWR_Prediction_Model.joblib")
