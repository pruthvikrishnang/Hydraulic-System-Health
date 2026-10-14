"""
Unit Tests for Calculus Methods (Integration & Differentiation)
==============================================================
"""

import unittest
import numpy as np
from src.calculus_methods import (
    trapezoidal_integral_1d,
    trapezoidal_integral_2d,
    finite_difference_derivative,
    compute_calculus_features_for_cycle,
)


class TestCalculusMethods(unittest.TestCase):

    def test_trapezoidal_integral_analytical_polynomial(self):
        """
        Analytical integral: ∫[0, 6] t^2 dt = [t^3 / 3]_0^6 = 216 / 3 = 72.0
        """
        dt = 0.001
        t = np.arange(0, 6.0 + dt / 2, dt)
        y = t ** 2
        num_integral = trapezoidal_integral_1d(y, dt)
        self.assertAlmostEqual(num_integral, 72.0, places=3)

    def test_trapezoidal_integral_analytical_sine(self):
        """
        Analytical integral: ∫[0, π] sin(t) dt = [-cos(t)]_0^π = -(-1) - (-1) = 2.0
        """
        dt = 0.0005
        t = np.arange(0, np.pi, dt)
        y = np.sin(t)
        num_integral = trapezoidal_integral_1d(y, dt)
        self.assertAlmostEqual(num_integral, 2.0, places=3)

    def test_trapezoidal_integral_2d_vectorized(self):
        """Verify 2D vectorized trapezoidal rule matches 1D row-by-row."""
        matrix = np.random.randn(10, 500)
        dt = 0.01
        v2d = trapezoidal_integral_2d(matrix, dt)
        v1d = np.array([trapezoidal_integral_1d(row, dt) for row in matrix])
        np.testing.assert_allclose(v2d, v1d, rtol=1e-10)

    def test_finite_difference_derivative_analytical(self):
        """
        Analytical derivative: d/dt(t^2) = 2t. At t=3, deriv=6.0.
        """
        dt = 0.001
        t = np.arange(0, 6.0 + dt / 2, dt)
        y = t ** 2
        dy = finite_difference_derivative(y, dt, method="central")
        idx_3 = int(3.0 / dt) - 1
        self.assertAlmostEqual(dy[idx_3], 6.0, places=2)

    def test_compute_calculus_features_for_cycle(self):
        """Verify cycle dictionary output contains required keys."""
        sample = np.sin(np.linspace(0, 2 * np.pi, 60))
        features = compute_calculus_features_for_cycle(sample, "TS1")
        self.assertIn("TS1_integral", features)
        self.assertIn("TS1_derivative_mean_abs", features)
        self.assertIn("TS1_derivative_max_abs", features)
        self.assertIn("TS1_derivative_std", features)
        self.assertTrue(all(np.isfinite(v) for v in features.values()))


if __name__ == "__main__":
    unittest.main()
