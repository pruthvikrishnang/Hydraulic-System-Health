"""
Model Training and Cross-Validation Module
==========================================
Implements syllabus requirements for supervised regression:
1. Linear Regression (OLS)
2. Polynomial Regression (Degree 1, 2, 3 feature expansions, bias-variance analysis)
3. Ridge Regression (L2 regularization, objective penalty vs MSE)
4. Tree-based regression baselines (Random Forest, Extra Trees)
5. Principal Component Analysis (PCA) dimensionality reduction experiment
6. Manual Batch Gradient Descent linear model integration
7. Strict train-only cross-validation (5-fold) and held-out test evaluation.
8. Serializes final production model, scaler, and metadata to models/.
"""

from pathlib import Path
import sys
import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data_validation import PROFILE_COLUMNS
from src.gradient_descent import BatchGradientDescentRegressor

MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"

TARGET = "cooler_condition"


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes MSE, RMSE, MAE, and safe R2 (guarded against zero variance)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_t, y_p))

    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    if ss_tot == 0.0:
        r2 = float("nan")  # Undefined mathematically
    else:
        r2 = float(r2_score(y_t, y_p))

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def prepare_feature_subsets(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Constructs well-defined feature subsets:
    1. Statistical_All: All summary features
    2. Statistical_Without_CE_CP: Summary features excluding virtual proxies
    3. Calculus_Enhanced_All: Statistical + Calculus features
    4. Calculus_Enhanced_Without_CE_CP: Combined without virtual proxies
    5. Best_Physical_Predictors: Selected optimal physical sensors
    """
    label_cols = [c for c in PROFILE_COLUMNS if c in df.columns]
    X_all = df.drop(columns=label_cols)

    # Remove zero-variance constant columns
    nunique = X_all.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    X_clean = X_all.drop(columns=constant_cols)

    # Statistical columns
    stat_cols = [c for c in X_clean.columns if not any(k in c for k in ["_integral", "_derivative"])]
    # Calculus columns
    calc_cols = [c for c in X_clean.columns if any(k in c for k in ["_integral", "_derivative"])]

    # Subsets
    subsets = {
        "Statistical_All": X_clean[stat_cols],
        "Statistical_Without_CE_CP": X_clean[[c for c in stat_cols if not c.startswith(("CE_", "CP_"))]],
        "Calculus_Enhanced_All": X_clean,
        "Calculus_Enhanced_Without_CE_CP": X_clean[[c for c in X_clean.columns if not c.startswith(("CE_", "CP_"))]],
    }

    # Best physical predictors (top temperature, flow, and power dynamics)
    best_physical_cols = [
        c for c in subsets["Calculus_Enhanced_Without_CE_CP"].columns
        if c.startswith(("TS1_", "TS2_", "FS1_", "EPS1_", "PS1_"))
    ]
    subsets["Best_Physical_Predictors"] = X_clean[best_physical_cols]

    return subsets


def run_model_training_and_cross_validation(
    data_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main training workflow:
    1. Splits data into 80% train and 20% test (stratified random split)
    2. Runs 5-fold cross-validation on train data across all models & feature sets
    3. Evaluates best candidates on held-out test set
    4. Executes polynomial regression degree analysis (degrees 1, 2, 3)
    5. Executes PCA dimensionality reduction experiment
    6. Fits manual Batch Gradient Descent
    7. Saves final production artifacts.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if data_path is None:
        data_path = Path("data/processed/hydraulic_features_combined.csv")

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    y = df[TARGET].copy()

    feature_subsets = prepare_feature_subsets(df)

    # 80/20 train/test split with fixed random seed
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Save indices for reproducibility
    np.savez(METRICS_DIR / "split_indices.npz", train_idx=train_idx, test_idx=test_idx)

    # Base candidate models
    def get_models() -> Dict[str, Any]:
        return {
            "LinearRegression": Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression())
            ]),
            "Ridge_alpha_1": Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=1.0, random_state=42))
            ]),
            "Ridge_alpha_10": Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=10.0, random_state=42))
            ]),
            "RandomForest": Pipeline([
                ("regressor", RandomForestRegressor(
                    n_estimators=100, max_depth=12, min_samples_leaf=2,
                    max_features=0.8, random_state=42, n_jobs=2
                ))
            ]),
            "ExtraTrees": Pipeline([
                ("regressor", ExtraTreesRegressor(
                    n_estimators=100, min_samples_leaf=2,
                    max_features=0.8, random_state=42, n_jobs=2
                ))
            ]),
        }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    comparison_rows = []
    trained_pipelines = {}

    print("\n--- Starting 5-Fold Cross Validation on Training Set (80%) ---")
    for subset_name, X_subset in feature_subsets.items():
        X_tr = X_subset.iloc[train_idx]
        X_te = X_subset.iloc[test_idx]

        models = get_models()
        for model_name, pipeline in models.items():
            # CV on train set only (NO test leakage)
            cv_scores = cross_val_score(
                pipeline,
                X_tr,
                y_train,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=1
            )
            cv_rmse = float(-cv_scores.mean())
            cv_std = float(cv_scores.std())

            # Fit on full training set, evaluate once on held-out test set
            pipeline.fit(X_tr, y_train)
            test_preds = pipeline.predict(X_te)
            test_metrics = calculate_regression_metrics(y_test, test_preds)

            key = f"{subset_name}__{model_name}"
            trained_pipelines[key] = {
                "pipeline": pipeline,
                "subset_name": subset_name,
                "model_name": model_name,
                "features": list(X_subset.columns),
                "cv_rmse": cv_rmse,
                "test_metrics": test_metrics,
                "test_preds": test_preds
            }

            comparison_rows.append({
                "Feature_Set": subset_name,
                "Model": model_name,
                "Feature_Count": X_subset.shape[1],
                "Train_CV_RMSE": cv_rmse,
                "Train_CV_Std": cv_std,
                "Test_MAE": test_metrics["MAE"],
                "Test_RMSE": test_metrics["RMSE"],
                "Test_R2": test_metrics["R2"],
            })
            print(f"[{subset_name}] {model_name:20s} -> CV RMSE: {cv_rmse:.4f} | Test RMSE: {test_metrics['RMSE']:.4f} | R2: {test_metrics['R2']:.4f}")

    comp_df = pd.DataFrame(comparison_rows).sort_values("Train_CV_RMSE").reset_index(drop=True)
    comp_df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    print("\nModel comparison saved to:", (METRICS_DIR / "model_comparison.csv").resolve())

    # --- EXPERIMENT 1: POLYNOMIAL REGRESSION (Degrees 1, 2, 3) ---
    print("\n--- Running Polynomial Regression Degree Analysis ---")
    poly_features_candidate = [
        "TS1_mean", "TS1_integral", "FS1_mean", "FS1_derivative_mean_abs", "EPS1_mean"
    ]
    # Filter to available columns
    poly_cols = [c for c in poly_features_candidate if c in df.columns]
    X_poly_base_tr = df[poly_cols].iloc[train_idx]
    X_poly_base_te = df[poly_cols].iloc[test_idx]

    poly_results = []
    for deg in [1, 2, 3]:
        poly_pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=deg, include_bias=False)),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=10.0, random_state=42))
        ])
        cv_scores = cross_val_score(
            poly_pipe, X_poly_base_tr, y_train, cv=cv, scoring="neg_root_mean_squared_error"
        )
        cv_rmse = float(-cv_scores.mean())

        poly_pipe.fit(X_poly_base_tr, y_train)
        train_pred = poly_pipe.predict(X_poly_base_tr)
        test_pred = poly_pipe.predict(X_poly_base_te)

        tr_metrics = calculate_regression_metrics(y_train, train_pred)
        te_metrics = calculate_regression_metrics(y_test, test_pred)

        poly_results.append({
            "Polynomial_Degree": deg,
            "Expanded_Feature_Count": poly_pipe.named_steps["poly"].n_output_features_,
            "Train_RMSE": tr_metrics["RMSE"],
            "CV_RMSE": cv_rmse,
            "Test_RMSE": te_metrics["RMSE"],
            "Test_R2": te_metrics["R2"],
        })
        print(f"Degree {deg} (Features: {poly_results[-1]['Expanded_Feature_Count']}) -> Train RMSE: {tr_metrics['RMSE']:.4f} | CV RMSE: {cv_rmse:.4f} | Test RMSE: {te_metrics['RMSE']:.4f}")

    poly_df = pd.DataFrame(poly_results)
    poly_df.to_csv(METRICS_DIR / "polynomial_degree_comparison.csv", index=False)

    # --- EXPERIMENT 2: PRINCIPAL COMPONENT ANALYSIS (PCA) ---
    print("\n--- Running PCA Dimensionality Reduction Experiment ---")
    X_calc_tr = feature_subsets["Calculus_Enhanced_Without_CE_CP"].iloc[train_idx]
    X_calc_te = feature_subsets["Calculus_Enhanced_Without_CE_CP"].iloc[test_idx]

    pca_results = []
    scaler = StandardScaler()
    X_calc_tr_s = scaler.fit_transform(X_calc_tr)
    X_calc_te_s = scaler.transform(X_calc_te)

    # Full PCA to get explained variance curve
    full_pca = PCA().fit(X_calc_tr_s)
    cum_var = np.cumsum(full_pca.explained_variance_ratio_)

    np.savez(
        METRICS_DIR / "pca_variance.npz",
        explained_variance_ratio=full_pca.explained_variance_ratio_,
        cumulative_variance=cum_var
    )

    for n_comp in [2, 5, 10, 20, 30]:
        pca_pipe = Pipeline([
            ("pca", PCA(n_components=n_comp)),
            ("regressor", Ridge(alpha=10.0, random_state=42))
        ])
        cv_scores = cross_val_score(
            pca_pipe, X_calc_tr_s, y_train, cv=cv, scoring="neg_root_mean_squared_error"
        )
        pca_pipe.fit(X_calc_tr_s, y_train)
        pred_te = pca_pipe.predict(X_calc_te_s)
        metrics = calculate_regression_metrics(y_test, pred_te)

        pca_results.append({
            "Components": n_comp,
            "Explained_Variance_Ratio_Sum": float(np.sum(full_pca.explained_variance_ratio_[:n_comp])),
            "CV_RMSE": float(-cv_scores.mean()),
            "Test_RMSE": metrics["RMSE"],
            "Test_R2": metrics["R2"]
        })

    pca_df = pd.DataFrame(pca_results)
    pca_df.to_csv(METRICS_DIR / "pca_experiment.csv", index=False)

    # --- EXPERIMENT 3: RIDGE REGULARIZATION PENALTY DECOMPOSITION ---
    print("\n--- Computing Ridge Regularization Penalty vs Prediction Error ---")
    ridge_model = Ridge(alpha=10.0, random_state=42)
    scaler_ridge = StandardScaler()
    X_tr_s = scaler_ridge.fit_transform(feature_subsets["Calculus_Enhanced_All"].iloc[train_idx])
    X_te_s = scaler_ridge.transform(feature_subsets["Calculus_Enhanced_All"].iloc[test_idx])
    ridge_model.fit(X_tr_s, y_train)
    y_pred_tr = ridge_model.predict(X_tr_s)

    prediction_mse = float(mean_squared_error(y_train, y_pred_tr))
    weights_sq_sum = float(np.sum(ridge_model.coef_ ** 2))
    l2_penalty = float(10.0 * weights_sq_sum)
    total_objective = prediction_mse + l2_penalty

    ridge_breakdown = {
        "alpha": 10.0,
        "prediction_MSE": prediction_mse,
        "sum_of_squared_weights_sum_w2": weights_sq_sum,
        "l2_penalty_term_alpha_sum_w2": l2_penalty,
        "total_ridge_objective": total_objective,
        "penalty_ratio_percent": (l2_penalty / total_objective) * 100.0
    }
    with open(METRICS_DIR / "ridge_penalty_breakdown.json", "w", encoding="utf-8") as f:
        json.dump(ridge_breakdown, f, indent=2)

    # --- EXPERIMENT 4: MANUAL BATCH GRADIENT DESCENT ---
    print("\n--- Training Manual Batch Gradient Descent ---")
    bgd = BatchGradientDescentRegressor(learning_rate=0.03, max_iters=1000, tol=1e-6)
    # Fit on best physical standardized predictors
    scaler_bgd = StandardScaler()
    X_bgd_tr = scaler_bgd.fit_transform(feature_subsets["Best_Physical_Predictors"].iloc[train_idx])
    X_bgd_te = scaler_bgd.transform(feature_subsets["Best_Physical_Predictors"].iloc[test_idx])

    bgd.fit(X_bgd_tr, y_train.to_numpy())
    bgd_pred_te = bgd.predict(X_bgd_te)
    bgd_metrics = calculate_regression_metrics(y_test, bgd_pred_te)
    print(f"Manual BGD Test RMSE: {bgd_metrics['RMSE']:.4f} | R2: {bgd_metrics['R2']:.4f} | Iterations: {bgd.iterations_run_}")

    np.save(METRICS_DIR / "gradient_descent_loss_history.npy", np.array(bgd.loss_history_))

    # --- SELECTION OF FINAL PRODUCTION MODEL ---
    # Based strictly on training CV RMSE
    best_key = min(trained_pipelines.keys(), key=lambda k: trained_pipelines[k]["cv_rmse"])
    best_item = trained_pipelines[best_key]

    print(f"\n==========================================")
    print(f"SELECTED BEST MODEL: {best_item['model_name']}")
    print(f"FEATURE SET: {best_item['subset_name']}")
    print(f"TRAIN CV RMSE: {best_item['cv_rmse']:.4f}")
    print(f"HELD-OUT TEST METRICS: {best_item['test_metrics']}")
    print(f"==========================================")

    # Refit final production model on full dataset
    final_pipeline = best_item["pipeline"]
    X_full = feature_subsets[best_item["subset_name"]]
    final_pipeline.fit(X_full, y)

    # Save final model
    joblib.dump(final_pipeline, MODELS_DIR / "hydraulic_health_model.joblib")

    # Save also a physical-only interpretable Ridge model
    ridge_physical = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=10.0, random_state=42))
    ])
    X_phys_full = feature_subsets["Best_Physical_Predictors"]
    ridge_physical.fit(X_phys_full, y)
    joblib.dump(ridge_physical, MODELS_DIR / "physical_ridge_model.joblib")

    # Save test predictions for evaluation and UI
    test_preds_df = pd.DataFrame({
        "cycle_index": test_idx,
        "actual_cooler_condition": y_test.to_numpy(),
        "predicted_cooler_condition": best_item["test_preds"],
    })
    test_preds_df["absolute_error"] = (
        test_preds_df["actual_cooler_condition"] - test_preds_df["predicted_cooler_condition"]
    ).abs()
    test_preds_df.to_csv(METRICS_DIR / "final_test_predictions.csv", index=False)

    metadata = {
        "target": TARGET,
        "selected_model": best_item["model_name"],
        "feature_set": best_item["subset_name"],
        "feature_columns": best_item["features"],
        "feature_count": len(best_item["features"]),
        "training_cycles": len(train_idx),
        "test_cycles": len(test_idx),
        "train_cv_rmse": float(best_item["cv_rmse"]),
        "held_out_test_metrics": best_item["test_metrics"],
        "bgd_test_metrics": bgd_metrics,
        "ridge_regularization_breakdown": ridge_breakdown,
        "polynomial_experiment_summary": poly_results,
        "pca_experiment_summary": pca_results,
        "best_physical_features": list(X_phys_full.columns),
    }

    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Model training and artifact generation complete!")
    return comp_df, metadata


if __name__ == "__main__":
    run_model_training_and_cross_validation()
