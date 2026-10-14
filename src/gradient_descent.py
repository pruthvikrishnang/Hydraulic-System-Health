"""
Manual Batch Gradient Descent for Linear Regression
===================================================
Implements syllabus requirements for manual, first-principles optimization:
1. Linear prediction function: y_hat = X * w + b
2. Mean Squared Error (MSE) objective: J(w, b) = (1 / n) * ||y_hat - y||^2
3. Analytically derived gradients:
       ∂J/∂w = (2 / n) * X^T (y_hat - y)
       ∂J/∂b = (2 / n) * ∑ (y_hat - y)
4. Parameter initialization, learning rate, updates, stopping tolerance,
   loss history tracking, divergence detection, and scikit-learn benchmarking.
"""

from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


class BatchGradientDescentRegressor:
    """
    First-principles Linear Regression trained using Batch Gradient Descent.
    Does NOT use any prebuilt scikit-learn optimizer or solver.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iters: int = 1000,
        tol: float = 1e-6,
        random_state: Optional[int] = 42
    ):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

        self.w_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.loss_history_: List[float] = []
        self.converged_: bool = False
        self.iterations_run_: int = 0
        self.diverged_: bool = False

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediction function: y_hat = X * w + b"""
        if self.w_ is None:
            raise RuntimeError("Model has not been fitted.")
        X_arr = np.asarray(X, dtype=np.float64)
        return np.dot(X_arr, self.w_) + self.b_

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MSE loss function: J = (1 / n) * ∑ (y_pred - y_true)^2"""
        n = len(y_true)
        if n == 0:
            return 0.0
        return float(np.mean((y_pred - y_true) ** 2))

    def compute_gradients(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Computes analytical gradients:
            error = y_hat - y
            ∂J/∂w = (2 / n) * X^T error
            ∂J/∂b = (2 / n) * ∑ error
        """
        n = X.shape[0]
        error = y_pred - y_true
        grad_w = (2.0 / n) * np.dot(X.T, error)
        grad_b = float((2.0 / n) * np.sum(error))
        return grad_w, grad_b

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BatchGradientDescentRegressor":
        """
        Fits linear parameters (w, b) using batch gradient descent.
        Detects divergence if loss explodes or encounters NaN.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X_arr.shape

        # Initialize weights (small values) and intercept (mean of y for fast, stable centering)
        if self.random_state is not None:
            rng = np.random.RandomState(self.random_state)
            self.w_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        else:
            self.w_ = np.zeros(n_features, dtype=np.float64)
        self.b_ = float(np.mean(y_arr))

        self.loss_history_ = []
        self.converged_ = False
        self.diverged_ = False

        prev_loss = float("inf")

        for iteration in range(1, self.max_iters + 1):
            y_pred = self.predict(X_arr)
            loss = self.compute_loss(y_arr, y_pred)

            # Check for numerical instability or divergence
            if np.isnan(loss) or np.isinf(loss) or (len(self.loss_history_) > 0 and loss > self.loss_history_[0] * 1e4):
                self.diverged_ = True
                self.iterations_run_ = iteration
                print(f"[WARNING] Gradient descent diverged at iteration {iteration}! Learning rate {self.learning_rate} is too large.")
                break

            self.loss_history_.append(loss)

            # Check convergence from previous step
            if abs(prev_loss - loss) < self.tol:
                self.converged_ = True
                self.iterations_run_ = iteration
                break

            prev_loss = loss

            # Compute analytical gradients
            grad_w, grad_b = self.compute_gradients(X_arr, y_arr, y_pred)

            # Parameter update step
            self.w_ -= self.learning_rate * grad_w
            self.b_ -= self.learning_rate * grad_b

        if not self.diverged_ and not self.converged_:
            self.iterations_run_ = self.max_iters

        return self


def compare_with_sklearn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    learning_rate: float = 0.05,
    max_iters: int = 2000
) -> Dict[str, Any]:
    """
    Compares manual Batch Gradient Descent against scikit-learn's OLS LinearRegression.
    Inputs MUST be standardized beforehand.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    # 1. Manual Gradient Descent
    bgd = BatchGradientDescentRegressor(learning_rate=learning_rate, max_iters=max_iters, tol=1e-7)
    bgd.fit(X_train, y_train)

    pred_bgd_train = bgd.predict(X_train)
    pred_bgd_test = bgd.predict(X_test)

    # 2. Scikit-Learn OLS (Normal Equation / SVD)
    sk_lr = LinearRegression(fit_intercept=True)
    sk_lr.fit(X_train, y_train)

    pred_sk_train = sk_lr.predict(X_train)
    pred_sk_test = sk_lr.predict(X_test)

    # Metrics
    results = {
        "bgd": {
            "train_mse": float(mean_squared_error(y_train, pred_bgd_train)),
            "test_mse": float(mean_squared_error(y_test, pred_bgd_test)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_bgd_test))),
            "test_r2": float(r2_score(y_test, pred_bgd_test)),
            "intercept": float(bgd.b_),
            "weight_norm": float(np.linalg.norm(bgd.w_)) if not bgd.diverged_ else np.nan,
            "iterations": bgd.iterations_run_,
            "converged": bgd.converged_,
            "diverged": bgd.diverged_,
            "loss_history": bgd.loss_history_,
        },
        "sklearn": {
            "train_mse": float(mean_squared_error(y_train, pred_sk_train)),
            "test_mse": float(mean_squared_error(y_test, pred_sk_test)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_sk_test))),
            "test_r2": float(r2_score(y_test, pred_sk_test)),
            "intercept": float(sk_lr.intercept_),
            "weight_norm": float(np.linalg.norm(sk_lr.coef_)),
        },
        "coefficient_difference_mae": (
            float(np.mean(np.abs(bgd.w_ - sk_lr.coef_))) if not bgd.diverged_ else np.nan
        ),
        "intercept_difference": (
            float(abs(bgd.b_ - sk_lr.intercept_)) if not bgd.diverged_ else np.nan
        ),
    }

    return results


def plot_gradient_descent_convergence(
    loss_history: List[float],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """Generates and saves the loss convergence trajectory plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss_history, color="#1f77b4", lw=2, label="Batch GD Loss J(w,b)")
    ax.set_title("Manual Batch Gradient Descent Convergence Curve", fontsize=13, fontweight="bold")
    ax.set_xlabel("Iteration Step", fontsize=11)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)

    return fig


if __name__ == "__main__":
    # Synthetic verification
    np.random.seed(42)
    X_syn = np.random.randn(200, 5)
    true_w = np.array([2.5, -1.8, 0.5, 3.2, -0.9])
    y_syn = np.dot(X_syn, true_w) + 4.2 + np.random.randn(200) * 0.1

    comp = compare_with_sklearn(X_syn, y_syn, X_syn, y_syn, learning_rate=0.1, max_iters=1000)
    print("Manual BGD Test RMSE:", comp["bgd"]["test_rmse"])
    print("Scikit-Learn Test RMSE:", comp["sklearn"]["test_rmse"])
    print("Coefficient MAE difference:", comp["coefficient_difference_mae"])
