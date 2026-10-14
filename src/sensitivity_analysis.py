"""
Sensitivity Analysis and Partial Derivatives Module
===================================================
Implements syllabus requirements for partial derivatives and sensitivity analysis:
1. Analytical Partial Derivatives for Differentiable Linear Models:
       y_hat = ∑ w_j * z_j + b,  where z_j = (x_j - μ_j) / σ_j
       Standardized sensitivity: ∂f / ∂z_j = w_j
       Raw physical sensitivity: ∂f / ∂x_j = w_j / σ_j
2. Numerical Controlled Perturbation Sensitivity for Non-Differentiable Tree Ensembles:
       S_j ≈ (1 / M) ∑ | (f(x + δ*e_j) - f(x - δ*e_j)) / (2 * δ) |
3. Explicit scientific distinction between model sensitivity and physical causality.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def compute_linear_partial_derivatives(
    weights: np.ndarray,
    feature_names: List[str],
    scaler_stds: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Computes analytical partial derivatives ∂f/∂z_j and ∂f/∂x_j for a linear model.
    Weights are assumed to be trained on standardized inputs (z-scores).
    """
    weights = np.asarray(weights, dtype=np.float64).ravel()
    df = pd.DataFrame({
        "feature": feature_names,
        "standardized_partial_derivative": weights,
        "abs_standardized_sensitivity": np.abs(weights),
    })

    if scaler_stds is not None:
        stds = np.asarray(scaler_stds, dtype=np.float64).ravel()
        # Raw sensitivity: ∂f/∂x_j = w_j / σ_j
        safe_stds = np.where(stds == 0, 1.0, stds)
        df["raw_physical_sensitivity"] = weights / safe_stds
        df["abs_raw_sensitivity"] = np.abs(df["raw_physical_sensitivity"])

    df = df.sort_values("abs_standardized_sensitivity", ascending=False).reset_index(drop=True)
    return df


def compute_tree_perturbation_sensitivity(
    model: Any,
    X_sample: np.ndarray,
    feature_names: List[str],
    epsilon_fraction: float = 0.05
) -> pd.DataFrame:
    """
    Estimates numerical sensitivity for tree ensembles using two-sided finite perturbation:
        ∂f/∂x_j ≈ (f(x + δ e_j) - f(x - δ e_j)) / (2δ)
    where δ = epsilon_fraction * std(x_j).
    """
    X_arr = np.asarray(X_sample, dtype=np.float64)
    n_samples, n_features = X_arr.shape

    feature_stds = np.std(X_arr, axis=0, ddof=1)
    feature_stds = np.where(feature_stds == 0, 1.0, feature_stds)

    mean_sensitivities = []
    max_sensitivities = []

    for j in range(n_features):
        delta = epsilon_fraction * feature_stds[j]
        if delta == 0:
            delta = 1e-4

        X_plus = X_arr.copy()
        X_plus[:, j] += delta

        X_minus = X_arr.copy()
        X_minus[:, j] -= delta

        pred_plus = model.predict(X_plus)
        pred_minus = model.predict(X_minus)

        finite_diff = (pred_plus - pred_minus) / (2.0 * delta)
        mean_sensitivities.append(float(np.mean(np.abs(finite_diff))))
        max_sensitivities.append(float(np.max(np.abs(finite_diff))))

    df = pd.DataFrame({
        "feature": feature_names,
        "mean_perturbation_sensitivity": mean_sensitivities,
        "max_perturbation_sensitivity": max_sensitivities,
    }).sort_values("mean_perturbation_sensitivity", ascending=False).reset_index(drop=True)

    return df


def plot_sensitivity_ranking(
    sensitivity_df: pd.DataFrame,
    metric_col: str,
    title: str = "Feature Sensitivity Analysis (∂f / ∂x)",
    top_n: int = 15,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """Plots top sensitive features."""
    top_df = sensitivity_df.head(top_n).sort_values(metric_col, ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_df["feature"], top_df[metric_col], color="#2ca02c", alpha=0.85)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Sensitivity Magnitude |∂f / ∂x|", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)

    return fig
