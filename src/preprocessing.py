"""
Data Preprocessing and Feature Scaling Module
==============================================
Implements syllabus concepts for:
1. Z-score Normalization (Standardization): z = (x - mean) / std
2. Min-Max Normalization: x_scaled = (x - min) / (max - min)
3. Safe constant-feature detection and handling (zero-variance guard)
4. Train-only parameter fitting to eliminate data leakage.
"""

from typing import Union, List, Optional, Tuple, Dict
import numpy as np
import pandas as pd


class StandardScalerCustom:
    """
    Standardizes features by removing the mean and scaling to unit variance (Z-score).
    Formula:
        z = (x - mean) / std
    Handles constant features safely by substituting std = 1.0 when std == 0.
    """

    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self.is_fitted: bool = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> "StandardScalerCustom":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        self.mean_ = np.nanmean(X_arr, axis=0)
        self.std_ = np.nanstd(X_arr, axis=0, ddof=0)

        # Protect against zero division on constant features
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_)
        self.is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transforming data.")

        is_df = isinstance(X, pd.DataFrame)
        cols = list(X.columns) if is_df else None
        X_arr = X.to_numpy(dtype=np.float64) if is_df else np.asarray(X, dtype=np.float64)

        Z = (X_arr - self.mean_) / self.std_

        if is_df:
            return pd.DataFrame(Z, columns=cols, index=X.index)
        return Z

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before inverse transform.")

        is_df = isinstance(Z, pd.DataFrame)
        cols = list(Z.columns) if is_df else None
        Z_arr = Z.to_numpy(dtype=np.float64) if is_df else np.asarray(Z, dtype=np.float64)

        X = (Z_arr * self.std_) + self.mean_
        if is_df:
            return pd.DataFrame(X, columns=cols, index=Z.index)
        return X


class MinMaxScalerCustom:
    """
    Transforms features by scaling each feature to a given range [feature_range[0], feature_range[1]].
    Default is [0, 1].
    Formula:
        x_std = (x - min) / (max - min)
        x_scaled = x_std * (max_range - min_range) + min_range
    Handles constant features safely by setting range to 1.0 when max == min.
    """

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0), eps: float = 1e-12):
        self.feature_range = feature_range
        self.eps = eps
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None
        self.data_range_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self.is_fitted: bool = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> "MinMaxScalerCustom":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        self.data_min_ = np.nanmin(X_arr, axis=0)
        self.data_max_ = np.nanmax(X_arr, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_

        # Safe guard for zero range
        self.data_range_ = np.where(self.data_range_ < self.eps, 1.0, self.data_range_)
        self.is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transforming data.")

        is_df = isinstance(X, pd.DataFrame)
        cols = list(X.columns) if is_df else None
        X_arr = X.to_numpy(dtype=np.float64) if is_df else np.asarray(X, dtype=np.float64)

        low, high = self.feature_range
        X_scaled = (X_arr - self.data_min_) / self.data_range_
        X_scaled = X_scaled * (high - low) + low

        if is_df:
            return pd.DataFrame(X_scaled, columns=cols, index=X.index)
        return X_scaled

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before inverse transform.")

        is_df = isinstance(X_scaled, pd.DataFrame)
        cols = list(X_scaled.columns) if is_df else None
        X_arr = X_scaled.to_numpy(dtype=np.float64) if is_df else np.asarray(X_scaled, dtype=np.float64)

        low, high = self.feature_range
        X_std = (X_arr - low) / (high - low)
        X_orig = X_std * self.data_range_ + self.data_min_

        if is_df:
            return pd.DataFrame(X_orig, columns=cols, index=X_scaled.index)
        return X_orig


def detect_and_drop_constant_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Identifies features with zero or near-zero variance and drops them.
    Returns cleaned DataFrame and list of dropped feature names.
    """
    nunique = df.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    cleaned_df = df.drop(columns=constant_cols)
    return cleaned_df, constant_cols


def compare_scaling_methods(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Evaluates Linear / Ridge regression performance across:
    1. Raw (No scaling)
    2. Z-score Standardization
    3. Min-Max Normalization
    Demonstrating that scale-sensitive linear models behave identically or better
    when standardized, while weights become directly interpretable.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, r2_score

    scalers = {
        "No Scaling (Raw)": None,
        "StandardScaler (Z-Score)": StandardScalerCustom(),
        "MinMaxScaler [0, 1]": MinMaxScalerCustom(),
    }

    results = []

    for name, scaler in scalers.items():
        if scaler is None:
            X_tr = X_train.to_numpy(dtype=np.float64)
            X_te = X_test.to_numpy(dtype=np.float64)
        else:
            # Fit strictly on train
            scaler.fit(X_train)
            X_tr = scaler.transform(X_train)
            X_te = scaler.transform(X_test)

        model = Ridge(alpha=10.0, random_state=42)
        model.fit(X_tr, y_train)
        pred = model.predict(X_te)

        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, pred)
        max_weight = np.max(np.abs(model.coef_))
        min_weight = np.min(np.abs(model.coef_))

        results.append({
            "Scaling Method": name,
            "Test RMSE": round(rmse, 4),
            "Test R2": round(r2, 4),
            "Max |Weight|": round(float(max_weight), 4),
            "Min |Weight|": round(float(min_weight), 6),
            "Intercept": round(float(model.intercept_), 4),
        })

    return pd.DataFrame(results)
