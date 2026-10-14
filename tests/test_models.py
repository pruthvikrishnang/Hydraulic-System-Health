"""
Unit Tests for Model Persistence, Inference, and Metrics
========================================================
"""

import unittest
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.data_validation import PROFILE_COLUMNS
from src.train_models import calculate_regression_metrics

MODELS_DIR = Path("models")


class TestModels(unittest.TestCase):

    def setUp(self):
        self.model_path = MODELS_DIR / "hydraulic_health_model.joblib"
        self.metadata_path = MODELS_DIR / "model_metadata.json"
        self.data_path = Path("data/processed/hydraulic_features_combined.csv")

    def test_model_and_metadata_exist(self):
        """Verifies trained model file and metadata json exist."""
        self.assertTrue(self.model_path.exists(), "hydraulic_health_model.joblib not found")
        self.assertTrue(self.metadata_path.exists(), "model_metadata.json not found")

    def test_no_target_leakage_in_features(self):
        """CRITICAL: Verifies no target profile columns leak into model input features."""
        with open(self.metadata_path, encoding="utf-8") as f:
            meta = json.load(f)

        features = meta["feature_columns"]
        for target_label in PROFILE_COLUMNS:
            self.assertNotIn(
                target_label,
                features,
                f"DATA LEAKAGE DETECTED! Target '{target_label}' is present in predictor columns!"
            )

    def test_inference_and_bounds(self):
        """Loads model, runs inference on 20 samples, verifies bounds in [0, 100]."""
        model = joblib.load(self.model_path)
        with open(self.metadata_path, encoding="utf-8") as f:
            meta = json.load(f)

        df = pd.read_csv(self.data_path)
        X_sample = df[meta["feature_columns"]].head(20)

        preds = model.predict(X_sample)
        self.assertEqual(len(preds), 20)
        self.assertTrue(np.all(np.isfinite(preds)))
        # Cooler condition percentage should realistically fall between -5 and 105
        self.assertTrue(np.all(preds >= -5.0))
        self.assertTrue(np.all(preds <= 105.0))

    def test_regression_metrics_calculation(self):
        """Tests MSE, RMSE, MAE, and undefined R2 guard."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        metrics_perfect = calculate_regression_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics_perfect["MAE"], 0.0)
        self.assertAlmostEqual(metrics_perfect["RMSE"], 0.0)
        self.assertAlmostEqual(metrics_perfect["R2"], 1.0)

        # Zero variance test target (e.g. chronological split on last block)
        y_constant = np.array([100.0, 100.0, 100.0])
        y_pred_const = np.array([98.0, 99.0, 101.0])
        metrics_zero_var = calculate_regression_metrics(y_constant, y_pred_const)
        self.assertTrue(np.isnan(metrics_zero_var["R2"]))


if __name__ == "__main__":
    unittest.main()
