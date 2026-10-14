"""
Unit Tests for Data Preprocessing and Feature Scaling
=====================================================
"""

import unittest
import numpy as np
import pandas as pd
from src.preprocessing import (
    StandardScalerCustom,
    MinMaxScalerCustom,
    detect_and_drop_constant_features,
)


class TestPreprocessing(unittest.TestCase):

    def test_standard_scaler_z_score(self):
        """Verify z = (x - mean) / std produces zero mean and unit variance."""
        np.random.seed(42)
        X = np.random.normal(loc=25.0, scale=4.5, size=(100, 4))
        scaler = StandardScalerCustom()
        Z = scaler.fit_transform(X)

        means = np.mean(Z, axis=0)
        stds = np.std(Z, axis=0, ddof=0)

        np.testing.assert_allclose(means, 0.0, atol=1e-12)
        np.testing.assert_allclose(stds, 1.0, atol=1e-12)

        # Inverse transform
        recovered = scaler.inverse_transform(Z)
        np.testing.assert_allclose(recovered, X, atol=1e-10)

    def test_standard_scaler_constant_feature_safety(self):
        """Constant features (std=0) must not produce NaN or Inf."""
        X = np.array([
            [5.0, 10.0],
            [5.0, 20.0],
            [5.0, 30.0],
        ])
        scaler = StandardScalerCustom()
        Z = scaler.fit_transform(X)
        self.assertFalse(np.isnan(Z).any())
        self.assertFalse(np.isinf(Z).any())
        # First column was constant 5.0 -> (5.0 - 5.0) / 1.0 = 0.0
        np.testing.assert_allclose(Z[:, 0], 0.0)

    def test_minmax_scaler(self):
        """Verify x_scaled in [0, 1] range and inverse transform recovery."""
        X = np.array([
            [10.0, -5.0],
            [20.0, 0.0],
            [30.0, 15.0],
        ])
        scaler = MinMaxScalerCustom(feature_range=(0.0, 1.0))
        X_s = scaler.fit_transform(X)

        self.assertAlmostEqual(float(np.min(X_s)), 0.0)
        self.assertAlmostEqual(float(np.max(X_s)), 1.0)

        recovered = scaler.inverse_transform(X_s)
        np.testing.assert_allclose(recovered, X, atol=1e-10)

    def test_detect_and_drop_constant_features(self):
        """Verify constant columns are detected and dropped correctly."""
        df = pd.DataFrame({
            "col_varying": [1.0, 2.0, 3.0],
            "col_constant": [7.0, 7.0, 7.0],
            "col_binary": [0.0, 1.0, 0.0],
        })
        cleaned, dropped = detect_and_drop_constant_features(df)
        self.assertIn("col_constant", dropped)
        self.assertNotIn("col_constant", cleaned.columns)
        self.assertEqual(cleaned.shape[1], 2)


if __name__ == "__main__":
    unittest.main()
