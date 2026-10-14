"""
Feature Engineering Pipeline Module
====================================
Builds comprehensive feature representations from raw 17-sensor time-series:
1. Statistical Features: mean, std, min, max per sensor cycle.
2. Calculus-Based Dynamic Features:
   - Trapezoidal Integral: ∫ s(t) dt (cumulative fluid volume, total work, heat dissipation)
   - Mean Absolute Derivative: mean(|ds/dt|) (dynamic turbulence, rate of temperature rise)
   - Maximum Absolute Derivative: max(|ds/dt|) (peak shock surge, rapid valve transitions)
   - Derivative Standard Deviation: std(ds/dt) (signal fluctuation volatility)
3. Outputs self-contained, validated datasets in data/processed/.
"""

from pathlib import Path
import sys
import gc
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_validation import SENSOR_METADATA, PROFILE_COLUMNS, resolve_raw_data_dir
from src.calculus_methods import SAMPLING_INTERVALS_DT, trapezoidal_integral_2d

PROCESSED_DIR = Path("data/processed")
DATA_DIR = Path("data")


def extract_statistical_features(values: np.ndarray, sensor: str) -> pd.DataFrame:
    """Extracts summary statistical metrics for each cycle."""
    return pd.DataFrame({
        f"{sensor}_mean": values.mean(axis=1),
        f"{sensor}_std": values.std(axis=1, ddof=1),
        f"{sensor}_min": values.min(axis=1),
        f"{sensor}_max": values.max(axis=1),
    })


def extract_calculus_features(values: np.ndarray, sensor: str) -> pd.DataFrame:
    """Extracts calculus-derived dynamic metrics using documented sampling intervals."""
    dt = SAMPLING_INTERVALS_DT.get(sensor, 1.0)

    # 1. Composite Trapezoidal Integral
    integral = trapezoidal_integral_2d(values, dt)

    # 2. Discrete Time Derivative
    diff = np.diff(values, axis=1) / dt
    abs_diff = np.abs(diff)

    mean_abs_diff = abs_diff.mean(axis=1)
    max_abs_diff = abs_diff.max(axis=1)
    std_diff = diff.std(axis=1, ddof=1)

    return pd.DataFrame({
        f"{sensor}_integral": integral,
        f"{sensor}_derivative_mean_abs": mean_abs_diff,
        f"{sensor}_derivative_max_abs": max_abs_diff,
        f"{sensor}_derivative_std": std_diff,
    })


def build_full_feature_datasets(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    force_rebuild: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Processes all 17 raw sensor streams into:
      1. Statistical-only DataFrame
      2. Calculus-only DataFrame
      3. Combined (Statistical + Calculus) DataFrame
    """
    if raw_dir is None:
        raw_dir = resolve_raw_data_dir()
    if output_dir is None:
        output_dir = PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    stat_out = output_dir / "hydraulic_features_statistical.csv"
    calc_out = output_dir / "hydraulic_features_calculus.csv"
    comb_out = output_dir / "hydraulic_features_combined.csv"
    legacy_out = DATA_DIR / "hydraulic_features.csv"

    if not force_rebuild and stat_out.exists() and calc_out.exists() and comb_out.exists():
        print("Feature datasets already exist. Loading cached files...")
        df_stat = pd.read_csv(stat_out)
        df_calc = pd.read_csv(calc_out)
        df_comb = pd.read_csv(comb_out)
        return df_stat, df_calc, df_comb

    print(f"Reading target profile from: {raw_dir / 'profile.txt'}")
    profile = pd.read_csv(raw_dir / "profile.txt", sep=r"\s+", header=None)
    profile.columns = PROFILE_COLUMNS

    n_cycles = len(profile)
    stat_frames: List[pd.DataFrame] = []
    calc_frames: List[pd.DataFrame] = []

    sensors = list(SENSOR_METADATA.keys())

    for sensor in sensors:
        sensor_path = raw_dir / f"{sensor}.txt"
        print(f"Extracting features for {sensor} ({SENSOR_METADATA[sensor]['rate_hz']} Hz)...")

        readings = pd.read_csv(sensor_path, sep=r"\s+", header=None, dtype=np.float32)
        if len(readings) != n_cycles:
            raise ValueError(f"Cycle count mismatch for {sensor}: {len(readings)} vs {n_cycles}")

        values = readings.to_numpy(copy=False)

        stat_features = extract_statistical_features(values, sensor)
        calc_features = extract_calculus_features(values, sensor)

        stat_frames.append(stat_features)
        calc_frames.append(calc_features)

        del readings, values, stat_features, calc_features
        gc.collect()

    X_stat = pd.concat(stat_frames, axis=1)
    X_calc = pd.concat(calc_frames, axis=1)
    del stat_frames, calc_frames
    gc.collect()

    # Combine with targets
    df_stat = pd.concat([X_stat, profile.reset_index(drop=True)], axis=1)
    df_calc = pd.concat([X_calc, profile.reset_index(drop=True)], axis=1)
    df_comb = pd.concat([X_stat, X_calc, profile.reset_index(drop=True)], axis=1)

    print(f"Saving statistical features: {stat_out} (shape: {df_stat.shape})")
    df_stat.to_csv(stat_out, index=False)

    print(f"Saving calculus features: {calc_out} (shape: {df_calc.shape})")
    df_calc.to_csv(calc_out, index=False)

    print(f"Saving combined features: {comb_out} (shape: {df_comb.shape})")
    df_comb.to_csv(comb_out, index=False)

    # Legacy copy for compatibility
    df_stat.to_csv(legacy_out, index=False)

    return df_stat, df_calc, df_comb


if __name__ == "__main__":
    print("Executing feature engineering pipeline...")
    s, c, comb = build_full_feature_datasets(force_rebuild=False)
    print("Pipeline Complete!")
    print(f"Statistical: {s.shape} | Calculus: {c.shape} | Combined: {comb.shape}")
