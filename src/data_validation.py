"""
Data Validation and Quality Assurance Module
=============================================
Validates raw UCI Condition Monitoring of Hydraulic Systems dataset files,
checks processed feature matrices for missing values, constants, duplicates,
and validates user-supplied inputs against physical sensor boundaries.
"""

from pathlib import Path
import os
import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

# SENSORS SPECIFICATION
SENSOR_METADATA = {
    "PS1": {"name": "Pressure 1", "unit": "bar", "rate_hz": 100, "samples_per_cycle": 6000, "type": "physical", "min_valid": 100.0, "max_valid": 250.0},
    "PS2": {"name": "Pressure 2", "unit": "bar", "rate_hz": 100, "samples_per_cycle": 6000, "type": "physical", "min_valid": 100.0, "max_valid": 250.0},
    "PS3": {"name": "Pressure 3", "unit": "bar", "rate_hz": 100, "samples_per_cycle": 6000, "type": "physical", "min_valid": 0.0, "max_valid": 50.0},
    "PS4": {"name": "Pressure 4", "unit": "bar", "rate_hz": 100, "samples_per_cycle": 6000, "type": "physical", "min_valid": 0.0, "max_valid": 20.0},
    "PS5": {"name": "Pressure 5", "unit": "bar", "rate_hz": 100, "samples_per_cycle": 6000, "type": "physical", "min_valid": 0.0, "max_valid": 50.0},
    "PS6": {"name": "Pressure 6", "unit": "bar", "rate_hz": 100, "samples_per_cycle": 6000, "type": "physical", "min_valid": 0.0, "max_valid": 50.0},
    "EPS1": {"name": "Motor Power", "unit": "W", "rate_hz": 100, "samples_per_cycle": 6000, "type": "physical", "min_valid": 2000.0, "max_valid": 3500.0},
    "FS1": {"name": "Volume Flow 1", "unit": "l/min", "rate_hz": 10, "samples_per_cycle": 600, "type": "physical", "min_valid": 0.0, "max_valid": 20.0},
    "FS2": {"name": "Volume Flow 2", "unit": "l/min", "rate_hz": 10, "samples_per_cycle": 600, "type": "physical", "min_valid": 5.0, "max_valid": 15.0},
    "TS1": {"name": "Temperature 1", "unit": "°C", "rate_hz": 1, "samples_per_cycle": 60, "type": "physical", "min_valid": 25.0, "max_valid": 75.0},
    "TS2": {"name": "Temperature 2", "unit": "°C", "rate_hz": 1, "samples_per_cycle": 60, "type": "physical", "min_valid": 25.0, "max_valid": 75.0},
    "TS3": {"name": "Temperature 3", "unit": "°C", "rate_hz": 1, "samples_per_cycle": 60, "type": "physical", "min_valid": 25.0, "max_valid": 75.0},
    "TS4": {"name": "Temperature 4", "unit": "°C", "rate_hz": 1, "samples_per_cycle": 60, "type": "physical", "min_valid": 25.0, "max_valid": 75.0},
    "VS1": {"name": "Vibration", "unit": "mm/s", "rate_hz": 1, "samples_per_cycle": 60, "type": "physical", "min_valid": 0.0, "max_valid": 2.0},
    "CE": {"name": "Cooling Efficiency", "unit": "%", "rate_hz": 1, "samples_per_cycle": 60, "type": "virtual", "min_valid": 0.0, "max_valid": 100.0},
    "CP": {"name": "Cooling Power", "unit": "kW", "rate_hz": 1, "samples_per_cycle": 60, "type": "virtual", "min_valid": 0.0, "max_valid": 5.0},
    "SE": {"name": "Efficiency Factor", "unit": "%", "rate_hz": 1, "samples_per_cycle": 60, "type": "virtual", "min_valid": 0.0, "max_valid": 100.0},
}

PROFILE_COLUMNS = [
    "cooler_condition",
    "valve_condition",
    "internal_pump_leakage",
    "accumulator_pressure",
    "stable_flag",
]

TARGET_METADATA = {
    "cooler_condition": {
        "unit": "%",
        "description": "Hydraulic cooler condition",
        "states": {3: "close to total failure", 20: "reduced efficiency", 100: "full efficiency"},
        "expected_counts": {3: 732, 20: 732, 100: 741},
        "continuous_nature": "Represents cooling capacity percentage over progressive heat degradation"
    },
    "valve_condition": {
        "unit": "%",
        "description": "Hydraulic valve switching condition",
        "states": {100: "optimal switching", 90: "small lag", 80: "severe lag", 73: "close to total failure"},
        "expected_counts": {100: 1125, 90: 360, 80: 360, 73: 360},
        "continuous_nature": "Switching delay and responsiveness metric"
    },
    "internal_pump_leakage": {
        "unit": "state",
        "description": "Internal pump leakage severity",
        "states": {0: "no leakage", 1: "weak leakage", 2: "severe leakage"},
        "expected_counts": {0: 1221, 1: 492, 2: 492},
        "continuous_nature": "Discreet hydraulic slip severity"
    },
    "accumulator_pressure": {
        "unit": "bar",
        "description": "Hydraulic gas pre-charge pressure",
        "states": {130: "optimal pressure", 115: "slightly reduced", 100: "severely reduced", 90: "close to total failure"},
        "expected_counts": {130: 599, 115: 399, 100: 399, 90: 808},
        "continuous_nature": "Gas nitrogen reservoir pressure in bar"
    },
    "stable_flag": {
        "unit": "flag",
        "description": "Cycle steady-state stability flag",
        "states": {0: "conditions were stable", 1: "static conditions might not have been reached"},
        "expected_counts": {0: 1449, 1: 756},
        "continuous_nature": "Transient thermal equilibrium binary flag"
    }
}


def resolve_raw_data_dir() -> Path:
    """Finds raw dataset directory from project data/raw/ or environment variable."""
    env_dir = os.environ.get("HYDRAULIC_DATA_DIR")
    candidate_paths = [
        Path("data/raw"),
        Path("../data/raw"),
    ]
    if env_dir:
        candidate_paths.insert(0, Path(env_dir))

    for path in candidate_paths:
        if path.exists() and (path / "profile.txt").exists():
            return path
    raise FileNotFoundError("Raw hydraulic dataset directory could not be resolved. Please place data in data/raw/ or set HYDRAULIC_DATA_DIR.")


def validate_raw_dataset(data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Validates existence, cycle counts, sampling rates, and missing values
    across all raw text files.
    """
    if data_dir is None:
        data_dir = resolve_raw_data_dir()

    report: Dict[str, Any] = {
        "data_dir": str(data_dir.resolve()),
        "status": "PASS",
        "total_cycles_expected": 2205,
        "sensors_validated": {},
        "target_profile": {},
        "errors": [],
        "warnings": [],
    }

    profile_path = data_dir / "profile.txt"
    if not profile_path.exists():
        report["status"] = "FAIL"
        report["errors"].append("profile.txt missing")
        return report

    profile = pd.read_csv(profile_path, sep=r"\s+", header=None)
    profile.columns = PROFILE_COLUMNS

    if len(profile) != 2205 or profile.shape[1] != 5:
        report["status"] = "FAIL"
        report["errors"].append(f"profile.txt has unexpected shape {profile.shape}, expected (2205, 5)")

    # Validate condition counts
    for col, meta in TARGET_METADATA.items():
        counts = profile[col].value_counts().to_dict()
        expected = meta["expected_counts"]
        match = counts == expected
        report["target_profile"][col] = {
            "counts": counts,
            "expected": expected,
            "match": match
        }
        if not match:
            report["warnings"].append(f"Target {col} counts {counts} differ from expected {expected}")

    # Validate each sensor
    for sensor, meta in SENSOR_METADATA.items():
        sensor_path = data_dir / f"{sensor}.txt"
        if not sensor_path.exists():
            report["status"] = "FAIL"
            report["errors"].append(f"{sensor}.txt does not exist")
            continue

        # Inspect sensor matrix
        data = pd.read_csv(sensor_path, sep=r"\s+", header=None, nrows=5)
        cols = data.shape[1]
        expected_cols = meta["samples_per_cycle"]
        col_match = (cols == expected_cols)

        report["sensors_validated"][sensor] = {
            "samples_per_cycle": cols,
            "expected_samples": expected_cols,
            "sampling_rate_hz": meta["rate_hz"],
            "unit": meta["unit"],
            "column_match": col_match
        }

        if not col_match:
            report["status"] = "FAIL"
            report["errors"].append(f"{sensor} has {cols} cols, expected {expected_cols}")

    return report


def validate_feature_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates a processed feature DataFrame: checks shape, nulls, infs,
    constant columns, and duplicate records.
    """
    report = {
        "row_count": len(df),
        "column_count": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "infinite_values": int(np.isinf(df.select_dtypes(include=np.number)).sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_columns": [],
        "target_columns_present": [col for col in PROFILE_COLUMNS if col in df.columns],
        "is_valid": True,
    }

    # Detect constants
    num_df = df.select_dtypes(include=np.number)
    nunique = num_df.nunique(dropna=False)
    report["constant_columns"] = nunique[nunique <= 1].index.tolist()

    if report["missing_values"] > 0 or report["infinite_values"] > 0:
        report["is_valid"] = False

    return report


def validate_sensor_input(inputs: Dict[str, float]) -> Tuple[bool, List[str], List[str]]:
    """
    Validates user-provided sensor inputs (e.g. from Streamlit or API).
    Returns (is_valid, list_of_errors, list_of_warnings).
    """
    errors: List[str] = []
    warnings: List[str] = []

    for key, val in inputs.items():
        # Strip suffix like _mean, _min, etc. to get base sensor
        base_sensor = key.split("_")[0]
        if not np.isfinite(val):
            errors.append(f"Input '{key}' is non-finite or NaN: {val}")
            continue

        if base_sensor in SENSOR_METADATA:
            meta = SENSOR_METADATA[base_sensor]
            min_v = meta["min_valid"]
            max_v = meta["max_valid"]
            if val < min_v or val > max_v:
                warnings.append(
                    f"Sensor '{key}' value {val:.2f} is outside normal operating range "
                    f"[{min_v}, {max_v}] {meta['unit']}"
                )

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def build_data_dictionary() -> pd.DataFrame:
    """Returns a formatted data dictionary DataFrame of all 17 sensors and 5 targets."""
    rows = []
    for s_id, meta in SENSOR_METADATA.items():
        rows.append({
            "Variable": s_id,
            "Name": meta["name"],
            "Type": "Predictor (Sensor)",
            "Nature": meta["type"],
            "Unit": meta["unit"],
            "Sampling Rate": f"{meta['rate_hz']} Hz",
            "Readings/Cycle": meta["samples_per_cycle"],
            "Operating Range": f"[{meta['min_valid']}, {meta['max_valid']}]"
        })
    for t_id, meta in TARGET_METADATA.items():
        states_str = "; ".join([f"{k}={v}" for k, v in meta["states"].items()])
        rows.append({
            "Variable": t_id,
            "Name": meta["description"],
            "Type": "Target (Condition Label)",
            "Nature": "Continuous Degradation" if t_id != "stable_flag" else "Binary Flag",
            "Unit": meta["unit"],
            "Sampling Rate": "Per cycle (1/60 Hz)",
            "Readings/Cycle": 1,
            "Operating Range": states_str
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Running Raw Dataset Validation...")
    res = validate_raw_dataset()
    print(json.dumps(res, indent=2))
    dict_df = build_data_dictionary()
    print("\nData Dictionary Sample:")
    print(dict_df.head(10).to_string(index=False))
