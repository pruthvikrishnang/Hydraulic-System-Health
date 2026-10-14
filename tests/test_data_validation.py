"""
Unit Tests for Data Validation and Quality Assurance
===================================================
"""

import unittest
import pandas as pd
import numpy as np
from src.data_validation import (
    validate_raw_dataset,
    validate_feature_dataframe,
    validate_sensor_input,
    build_data_dictionary,
    SENSOR_METADATA,
    TARGET_METADATA,
)


class TestDataValidation(unittest.TestCase):

    def test_raw_dataset_validation(self):
        """Validates that raw dataset exists, has 2205 rows, and passes checks."""
        report = validate_raw_dataset()
        self.assertEqual(report["status"], "PASS", f"Raw dataset validation failed: {report['errors']}")
        self.assertEqual(len(report["sensors_validated"]), 17)
        self.assertEqual(report["total_cycles_expected"], 2205)

    def test_feature_dataframe_validation(self):
        """Validates feature matrix has 0 missing values and expected columns."""
        df = pd.read_csv("data/processed/hydraulic_features_combined.csv")
        report = validate_feature_dataframe(df)
        self.assertTrue(report["is_valid"])
        self.assertEqual(report["missing_values"], 0)
        self.assertEqual(report["infinite_values"], 0)
        self.assertEqual(report["row_count"], 2205)

    def test_sensor_input_validation(self):
        """Tests input bounds checker for Streamlit/API."""
        # Valid input
        valid_input = {
            "PS1_mean": 160.0,
            "TS1_mean": 45.0,
            "FS1_mean": 8.5,
            "EPS1_mean": 2500.0,
        }
        is_valid, errors, warnings = validate_sensor_input(valid_input)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(warnings), 0)

        # Invalid input (NaN and out-of-range)
        invalid_input = {
            "PS1_mean": np.nan,
            "TS1_mean": 150.0,  # Physically absurd oil temperature
        }
        is_valid, errors, warnings = validate_sensor_input(invalid_input)
        self.assertFalse(is_valid)
        self.assertIn("NaN", errors[0])
        self.assertGreater(len(warnings), 0)

    def test_data_dictionary_completeness(self):
        """Verifies data dictionary contains all 17 sensors and 5 targets."""
        dict_df = build_data_dictionary()
        self.assertEqual(len(dict_df), 17 + 5)
        self.assertIn("Variable", dict_df.columns)
        self.assertIn("Unit", dict_df.columns)
        self.assertIn("Operating Range", dict_df.columns)


if __name__ == "__main__":
    unittest.main()
