"""
Unit Tests for Manual Batch Gradient Descent Regressor
======================================================
"""

import unittest
import numpy as np
from src.gradient_descent import (
    BatchGradientDescentRegressor,
    compare_with_sklearn,
)


class TestGradientDescent(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n_samples = 150
        self.n_features = 4
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.true_w = np.array([3.0, -2.5, 1.2, 0.8])
        self.true_b = 15.0
        self.y = np.dot(self.X, self.true_w) + self.true_b + np.random.randn(self.n_samples) * 0.05

    def test_loss_computation(self):
        """Verify MSE computation."""
        bgd = BatchGradientDescentRegressor()
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 19.0, 33.0])
        # Errors: +2, -1, +3 -> squared: 4, 1, 9 -> sum=14, mean = 14/3
        loss = bgd.compute_loss(y_true, y_pred)
        self.assertAlmostEqual(loss, 14.0 / 3.0, places=6)

    def test_gradient_derivation_numerical_check(self):
        """
        Validates analytical gradient against two-sided numerical finite difference:
            ∂J/∂w_j ≈ (J(w + ε e_j) - J(w - ε e_j)) / (2ε)
        """
        bgd = BatchGradientDescentRegressor()
        w = np.array([1.5, -0.5, 2.0, -1.0])
        b = 5.0
        y_pred = np.dot(self.X, w) + b

        grad_w_analytic, grad_b_analytic = bgd.compute_gradients(self.X, self.y, y_pred)

        eps = 1e-6
        # Check w gradients
        for j in range(self.n_features):
            w_plus = w.copy()
            w_plus[j] += eps
            loss_plus = bgd.compute_loss(self.y, np.dot(self.X, w_plus) + b)

            w_minus = w.copy()
            w_minus[j] -= eps
            loss_minus = bgd.compute_loss(self.y, np.dot(self.X, w_minus) + b)

            grad_numeric = (loss_plus - loss_minus) / (2.0 * eps)
            self.assertAlmostEqual(grad_w_analytic[j], grad_numeric, places=4)

        # Check b gradient
        loss_b_plus = bgd.compute_loss(self.y, np.dot(self.X, w) + (b + eps))
        loss_b_minus = bgd.compute_loss(self.y, np.dot(self.X, w) + (b - eps))
        grad_b_numeric = (loss_b_plus - loss_b_minus) / (2.0 * eps)
        self.assertAlmostEqual(grad_b_analytic, grad_b_numeric, places=4)

    def test_convergence_on_linear_data(self):
        """BGD must converge to low loss and estimate correct weights."""
        bgd = BatchGradientDescentRegressor(learning_rate=0.08, max_iters=800, tol=1e-8)
        bgd.fit(self.X, self.y)

        self.assertFalse(bgd.diverged_)
        self.assertTrue(bgd.converged_)
        self.assertLess(bgd.loss_history_[-1], 0.05)
        np.testing.assert_allclose(bgd.w_, self.true_w, atol=0.1)
        self.assertAlmostEqual(bgd.b_, self.true_b, delta=0.1)

    def test_divergence_detection(self):
        """Excessively large learning rate must trigger divergence flag, not crash silently."""
        bgd = BatchGradientDescentRegressor(learning_rate=100.0, max_iters=50)
        bgd.fit(self.X, self.y)
        self.assertTrue(bgd.diverged_)


if __name__ == "__main__":
    unittest.main()
