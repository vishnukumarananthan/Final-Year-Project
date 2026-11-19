"""
Preprocessing and Data Pipeline Module
Smart Electricity Consumption Anomaly Detection and Appliance Identification System

Features:
- Dynamic column detection and mapping (timestamp, appliances, total consumption)
- Missing value imputation and validation
- Chronological sorting and cleaning
- Appliance baseline calculation (historical mean by hour/day-of-week)
- Feature engineering for Isolation Forest anomaly detection
- Chronological train/test split
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# Common column name patterns for auto-detection
TIMESTAMP_CANDIDATES = [
    "timestamp", "time", "date", "datetime", "date_time", "reading_time", "index"
]
TOTAL_CANDIDATES = [
    "total_consumption_kwh", "total_consumption", "total", "aggregate", "mains",
    "total_kwh", "whole_house", "grid_import"
]
EXCLUDE_COLUMNS = [
    "is_simulated", "id", "index", "unnamed: 0", "label", "anomaly", "is_anomaly"
]


class ElectricityDataPreprocessor:
    """
    Handles end-to-end data ingestion, validation, cleaning, dynamic column detection,
    and feature engineering for appliance-level electricity consumption.
    """

    def __init__(self):
        self.timestamp_col: Optional[str] = None
        self.total_col: Optional[str] = None
        self.appliance_cols: List[str] = []
        self.baseline_profiles: Dict[str, pd.DataFrame] = {}
        self.appliance_stats: Dict[str, Dict[str, float]] = {}
        self.cleaning_summary: Dict[str, any] = {}

    def detect_columns(
        self,
        df: pd.DataFrame,
        custom_mapping: Optional[Dict[str, str]] = None
    ) -> Tuple[str, str, List[str]]:
        """
        Dynamically identify timestamp, total consumption, and appliance columns.
        Supports optional user-specified custom column mapping.
        """
        df_cols_lower = {col.lower().strip(): col for col in df.columns}

        # 1. Detect Timestamp Column
        timestamp_col = None
        if custom_mapping and "timestamp" in custom_mapping and custom_mapping["timestamp"] in df.columns:
            timestamp_col = custom_mapping["timestamp"]
        else:
            for cand in TIMESTAMP_CANDIDATES:
                if cand in df_cols_lower:
                    timestamp_col = df_cols_lower[cand]
                    break

        if not timestamp_col:
            # Try to find string/object/datetime columns that represent dates (avoid numeric columns)
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    continue
                try:
                    sample = df[col].dropna().iloc[:10]
                    if len(sample) > 0 and sample.astype(str).str.contains(r"[\-/: ]").any():
                        pd.to_datetime(sample)
                        timestamp_col = col
                        break
                except Exception:
                    continue

        if not timestamp_col:
            raise ValueError("Could not automatically identify a timestamp column. Please specify column mapping.")

        # 2. Detect Total Consumption Column
        total_col = None
        if custom_mapping and "total" in custom_mapping and custom_mapping["total"] in df.columns:
            total_col = custom_mapping["total"]
        else:
            # Exact match first
            for cand in TOTAL_CANDIDATES:
                if cand in df_cols_lower:
                    total_col = df_cols_lower[cand]
                    break
            # Fuzzy / substring match fallback (e.g. "Total_Mains", "House_Total")
            if not total_col:
                for col in df.columns:
                    c_low = col.lower().strip()
                    if any(cand in c_low for cand in ["total", "aggregate", "mains", "whole_house", "grid_import"]):
                        total_col = col
                        break

        # 3. Detect Appliance Columns (all numeric columns except timestamp and total and exclusions)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        appliance_cols = []
        for col in numeric_cols:
            col_clean = col.lower().strip()
            if col == timestamp_col or col == total_col:
                continue
            if col_clean in EXCLUDE_COLUMNS:
                continue
            appliance_cols.append(col)

        if not appliance_cols:
            raise ValueError("No numeric appliance electricity reading columns found in dataset.")

        # If total column is missing, we will create it as the sum of appliance columns
        self.timestamp_col = timestamp_col
        self.total_col = total_col
        self.appliance_cols = appliance_cols

        return timestamp_col, total_col, appliance_cols

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw dataset:
        - Parse and validate timestamps
        - Sort chronologically
        - Impute missing values
        - Clip negative readings to zero
        - Calculate total consumption if absent
        """
        df = df.copy()

        if self.timestamp_col is None:
            self.detect_columns(df)

        # Parse timestamp
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], errors="coerce")
        null_timestamps = df[self.timestamp_col].isna().sum()
        if null_timestamps > 0:
            df = df.dropna(subset=[self.timestamp_col])

        # Sort chronologically
        df = df.sort_values(by=self.timestamp_col).reset_index(drop=True)

        # Track missing values count before imputation
        missing_counts = {}
        for col in self.appliance_cols:
            if col in df.columns:
                missing_counts[col] = int(df[col].isna().sum())
                # Clip negative values
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].clip(lower=0.0)
                # Impute missing values: forward fill -> backward fill -> 0
                df[col] = df[col].ffill().bfill().fillna(0.0)

        # Handle Total Consumption column
        if self.total_col and self.total_col in df.columns:
            missing_counts[self.total_col] = int(df[self.total_col].isna().sum())
            df[self.total_col] = pd.to_numeric(df[self.total_col], errors="coerce").clip(lower=0.0)
            df[self.total_col] = df[self.total_col].ffill().bfill().fillna(0.0)
        else:
            # Compute total as sum of appliances
            self.total_col = "Total_Consumption_kWh"
            df[self.total_col] = df[self.appliance_cols].sum(axis=1)

        self.cleaning_summary = {
            "total_records": len(df),
            "missing_values_imputed": missing_counts,
            "dropped_invalid_timestamps": int(null_timestamps),
            "start_time": str(df[self.timestamp_col].min()),
            "end_time": str(df[self.timestamp_col].max()),
            "monitored_appliances": self.appliance_cols
        }

        return df

    def compute_baseline_profiles(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Compute normal consumption baseline profiles for each appliance based on
        Hour-of-Day (0-23) and Day-of-Week (0-6).
        This models historical patterns so that normal variations (e.g. night vs day)
        are properly accounted for.
        """
        df = df.copy()
        ts = pd.to_datetime(df[self.timestamp_col])
        hour_series = pd.Series(ts.dt.hour.values, name="hour", index=df.index)
        dow_series = pd.Series(ts.dt.day_of_week.values, name="day_of_week", index=df.index)

        profiles = {}
        stats = {}
        for app in self.appliance_cols:
            grouped = df.groupby([hour_series, dow_series])[app].agg(
                mean="mean",
                median="median",
                std="std"
            ).reset_index()
            grouped.columns = ["hour", "day_of_week", "mean", "median", "std"]
            grouped["std"] = grouped["std"].fillna(0.05)
            profiles[app] = grouped

            # Global and rolling statistics
            series = df[app].astype(float)
            rolling_m = series.rolling(window=24, min_periods=1).mean()
            rolling_s = series.rolling(window=24, min_periods=1).std().fillna(0.05)
            stats[app] = {
                "overall_mean": float(series.mean()),
                "overall_std": float(series.std()),
                "rolling_mean_avg": float(rolling_m.mean()),
                "rolling_std_avg": float(rolling_s.mean())
            }

        self.baseline_profiles = profiles
        self.appliance_stats = stats
        return profiles

    def get_appliance_stats(self, appliance: str) -> Dict[str, float]:
        """Return global rolling baseline statistics for appliance."""
        if appliance in self.appliance_stats:
            return self.appliance_stats[appliance]
        return {
            "overall_mean": 1.0,
            "overall_std": 0.5,
            "rolling_mean_avg": 1.0,
            "rolling_std_avg": 0.5
        }

    def get_expected_baseline(
        self,
        appliance: str,
        timestamp: pd.Timestamp
    ) -> float:
        """
        Look up the expected normal consumption for an appliance at a specific hour and day.
        """
        if appliance not in self.baseline_profiles:
            return 0.0

        profile = self.baseline_profiles[appliance]
        hour = timestamp.hour
        dow = timestamp.day_of_week

        match = profile[(profile["hour"] == hour) & (profile["day_of_week"] == dow)]
        if not match.empty:
            return float(match["median"].iloc[0])
        # Fallback to hourly mean
        hourly_match = profile[profile["hour"] == hour]
        if not hourly_match.empty:
            return float(hourly_match["median"].mean())
        return float(profile["median"].mean())

    def extract_features_for_appliance(
        self,
        df: pd.DataFrame,
        appliance: str
    ) -> pd.DataFrame:
        """
        Extract rich feature set for Isolation Forest anomaly detection for a specific appliance:
        - consumption: raw current reading
        - hour: hour of day (0-23)
        - day_of_week: day of week (0-6)
        - is_weekend: boolean flag
        - rolling_mean_24h: 24-period rolling mean (captures recent baseline)
        - rolling_std_24h: 24-period rolling std (captures volatility)
        - hourly_deviation: difference from expected historical hour-of-day baseline
        - ratio_to_rolling: ratio of reading to 24h rolling average
        """
        ts = pd.to_datetime(df[self.timestamp_col])
        series = df[appliance].astype(float)

        hour = ts.dt.hour
        dow = ts.dt.day_of_week
        is_weekend = (dow >= 5).astype(int)

        rolling_mean = series.rolling(window=24, min_periods=1).mean()
        rolling_std = series.rolling(window=24, min_periods=1).std().fillna(0.01)

        # Baseline deviation
        hourly_mean = series.groupby(hour).transform("mean")
        hourly_dev = series - hourly_mean

        ratio_to_rolling = series / (rolling_mean + 0.01)

        feature_df = pd.DataFrame({
            "consumption": series,
            "hour": hour,
            "day_of_week": dow,
            "is_weekend": is_weekend,
            "rolling_mean_24h": rolling_mean,
            "rolling_std_24h": rolling_std,
            "hourly_deviation": hourly_dev,
            "ratio_to_rolling": ratio_to_rolling
        })

        return feature_df.fillna(0.0)

    def train_test_split_chronological(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split time-series data chronologically (train on past, test on future).
        """
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
        test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
        return train_df, test_df
