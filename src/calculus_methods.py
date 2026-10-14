"""
Calculus Methods Module: Numerical Integration and Differentiation
===================================================================
Implements fundamental Calculus principles applied to physical sensor signals:
1. Numerical Integration via the Composite Trapezoidal Rule:
       ∫[0 to T] s(t) dt ≈ ∑_{k=0}^{N-2} ((s_k + s_{k+1}) / 2) * Δt
   Physical meaning: Total energy, cumulative pressure impulse, or accumulated fluid volume.

2. Numerical Differentiation via Finite Differences:
       ds/dt ≈ (s_{k+1} - s_k) / Δt  (forward)
       ds/dt ≈ (s_{k+1} - s_{k-1}) / (2*Δt)  (central)
   Physical meaning: Rate of pressure rise (surge), rate of heating/cooling, volumetric flow acceleration.

3. Calculus Feature Extractor:
   Calculates signal integral, mean absolute derivative, maximum absolute derivative,
   and derivative standard deviation for dynamic cycle characterization.
"""

from typing import Dict, Union, Tuple, Optional
import numpy as np
import pandas as pd

# Sampling intervals Δt in seconds (UCI Hydraulic documentation)
SAMPLING_INTERVALS_DT: Dict[str, float] = {
    # 100 Hz Sensors (Δt = 0.01 s, 6000 points / 60 s cycle)
    "PS1": 0.01, "PS2": 0.01, "PS3": 0.01, "PS4": 0.01, "PS5": 0.01, "PS6": 0.01,
    "EPS1": 0.01,
    # 10 Hz Sensors (Δt = 0.1 s, 600 points / 60 s cycle)
    "FS1": 0.1, "FS2": 0.1,
    # 1 Hz Sensors (Δt = 1.0 s, 60 points / 60 s cycle)
    "TS1": 1.0, "TS2": 1.0, "TS3": 1.0, "TS4": 1.0,
    "VS1": 1.0, "CE": 1.0, "CP": 1.0, "SE": 1.0,
}


def trapezoidal_integral_1d(signal: np.ndarray, dt: float) -> float:
    """
    Computes numerical definite integral of a 1D discrete signal using
    the composite trapezoidal rule:
        ∫ s(t) dt ≈ Δt * (0.5*s_0 + s_1 + s_2 + ... + s_{N-2} + 0.5*s_{N-1})
    """
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) < 2:
        return 0.0
    # Composite trapezoidal formula
    return float(dt * (0.5 * signal[0] + np.sum(signal[1:-1]) + 0.5 * signal[-1]))


def trapezoidal_integral_2d(matrix: np.ndarray, dt: float) -> np.ndarray:
    """
    Vectorized composite trapezoidal integration across rows (cycles).
    matrix shape: (n_cycles, n_time_steps).
    Returns 1D array of integrals of shape (n_cycles,).
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[1] < 2:
        return np.zeros(matrix.shape[0], dtype=np.float64)
    return dt * (0.5 * matrix[:, 0] + np.sum(matrix[:, 1:-1], axis=1) + 0.5 * matrix[:, -1])


def finite_difference_derivative(
    signal: np.ndarray,
    dt: float,
    method: str = "forward"
) -> np.ndarray:
    """
    Calculates numerical time derivative ds/dt using discrete differences.
    Methods:
      - 'forward': (s_{k+1} - s_k) / Δt
      - 'central': (s_{k+1} - s_{k-1}) / (2*Δt)
    """
    signal = np.asarray(signal, dtype=np.float64)
    if method == "forward":
        # Shape along axis is N - 1
        return np.diff(signal, axis=-1) / dt
    elif method == "central":
        if signal.ndim == 1:
            if len(signal) < 3:
                return np.diff(signal) / dt
            return (signal[2:] - signal[:-2]) / (2.0 * dt)
        else:
            if signal.shape[1] < 3:
                return np.diff(signal, axis=1) / dt
            return (signal[:, 2:] - signal[:, :-2]) / (2.0 * dt)
    else:
        raise ValueError(f"Unknown derivative method: {method}")


def compute_calculus_features_for_cycle(
    cycle_readings: np.ndarray,
    sensor_name: str
) -> Dict[str, float]:
    """
    Computes calculus features for a single cycle's 1D time series.
    """
    dt = SAMPLING_INTERVALS_DT.get(sensor_name, 1.0)
    readings = np.asarray(cycle_readings, dtype=np.float64)

    # 1. Definite Integral (Total Area under curve)
    integral = trapezoidal_integral_1d(readings, dt)

    # 2. First Derivative (Rate of change)
    deriv = finite_difference_derivative(readings, dt, method="forward")
    abs_deriv = np.abs(deriv)

    mean_abs_deriv = float(np.mean(abs_deriv)) if len(deriv) > 0 else 0.0
    max_abs_deriv = float(np.max(abs_deriv)) if len(deriv) > 0 else 0.0
    std_deriv = float(np.std(deriv, ddof=1)) if len(deriv) > 1 else 0.0

    return {
        f"{sensor_name}_integral": integral,
        f"{sensor_name}_derivative_mean_abs": mean_abs_deriv,
        f"{sensor_name}_derivative_max_abs": max_abs_deriv,
        f"{sensor_name}_derivative_std": std_deriv,
    }


def compute_calculus_features_for_sensor_matrix(
    sensor_matrix: np.ndarray,
    sensor_name: str
) -> pd.DataFrame:
    """
    Efficiently computes calculus features for all cycles of a given sensor matrix.
    sensor_matrix shape: (2205, N_samples).
    """
    dt = SAMPLING_INTERVALS_DT.get(sensor_name, 1.0)
    matrix = np.asarray(sensor_matrix, dtype=np.float64)

    # Vectorized Trapezoidal Integral
    integrals = trapezoidal_integral_2d(matrix, dt)

    # Vectorized Derivatives
    deriv = np.diff(matrix, axis=1) / dt
    abs_deriv = np.abs(deriv)

    mean_abs_deriv = np.mean(abs_deriv, axis=1)
    max_abs_deriv = np.max(abs_deriv, axis=1)
    std_deriv = np.std(deriv, axis=1, ddof=1)

    df_calc = pd.DataFrame({
        f"{sensor_name}_integral": integrals,
        f"{sensor_name}_derivative_mean_abs": mean_abs_deriv,
        f"{sensor_name}_derivative_max_abs": max_abs_deriv,
        f"{sensor_name}_derivative_std": std_deriv,
    })

    return df_calc


if __name__ == "__main__":
    # Analytical verification: f(t) = t^2 over [0, 6], dt = 0.01
    # Analytical integral: ∫[0, 6] t^2 dt = [t^3 / 3]_0^6 = 216 / 3 = 72.0
    t = np.arange(0, 6.0001, 0.01)
    y = t ** 2
    num_int = trapezoidal_integral_1d(y, 0.01)
    print(f"Analytical Integral: 72.0000 | Numerical Trapezoidal: {num_int:.4f}")

    # Derivative verification: d/dt(t^2) = 2t. At t=3, deriv=6.0
    dy = finite_difference_derivative(y, 0.01, method="central")
    # middle index is near t=3
    idx_3 = int(3.0 / 0.01) - 1
    print(f"Analytical Derivative at t=3: 6.0000 | Numerical Derivative: {dy[idx_3]:.4f}")
