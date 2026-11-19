"""
Unit Tests for Preprocessing Pipeline
Smart Electricity Consumption Anomaly Detection and Appliance Identification System
"""

import pytest
import pandas as pd
import numpy as np

from preprocessing import ElectricityDataPreprocessor


def test_column_auto_detection():
    """Test automatic detection of timestamp, total, and appliance columns."""
    df = pd.DataFrame({
        "DateTime": pd.date_range("2024-01-01", periods=24, freq="h"),
        "Air_Conditioner": np.random.uniform(1.0, 3.0, 24),
        "Refrigerator": np.random.uniform(0.1, 0.2, 24),
        "Total_Mains": np.random.uniform(1.5, 4.0, 24),
        "is_simulated": [True] * 24
    })
    preprocessor = ElectricityDataPreprocessor()
    ts_col, total_col, apps = preprocessor.detect_columns(df)

    assert ts_col == "DateTime"
    assert total_col == "Total_Mains"
    assert "Air_Conditioner" in apps
    assert "Refrigerator" in apps
    assert "is_simulated" not in apps  # Excluded metadata column


def test_missing_value_imputation():
    """Test handling and imputation of missing values (NaNs)."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
        "AC_kWh": [1.5, np.nan, 1.7, np.nan, 2.0, 2.2, np.nan, 1.8, 1.9, 2.1],
        "Fridge_kWh": [0.1, 0.12, np.nan, 0.11, 0.13, np.nan, 0.15, 0.14, 0.12, 0.11]
    })
    preprocessor = ElectricityDataPreprocessor()
    df_clean = preprocessor.clean_data(df)

    # Verify no NaNs remain
    assert df_clean["AC_kWh"].isna().sum() == 0
    assert df_clean["Fridge_kWh"].isna().sum() == 0
    # Verify auto-calculation of total consumption
    assert "Total_Consumption_kWh" in df_clean.columns
    assert df_clean["Total_Consumption_kWh"].isna().sum() == 0


def test_negative_value_clipping():
    """Test clipping of negative electricity readings to 0.0."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
        "AC_kWh": [-2.5, 1.0, -0.3, 2.0, -99.0]
    })
    preprocessor = ElectricityDataPreprocessor()
    df_clean = preprocessor.clean_data(df)

    assert (df_clean["AC_kWh"] >= 0.0).all()
    assert df_clean["AC_kWh"].iloc[0] == 0.0


def test_baseline_calculation():
    """Test historical diurnal hourly baseline profiles calculation."""
    timestamps = pd.date_range("2024-01-01", periods=24 * 7, freq="h")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "AC_kWh": np.where(timestamps.hour >= 14, 2.5, 0.2)
    })
    preprocessor = ElectricityDataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    profiles = preprocessor.compute_baseline_profiles(df_clean)

    assert "AC_kWh" in profiles
    profile_ac = profiles["AC_kWh"]
    assert "hour" in profile_ac.columns
    assert "median" in profile_ac.columns

    # Verify afternoon baseline is significantly higher than morning
    afternoon_expected = preprocessor.get_expected_baseline("AC_kWh", pd.Timestamp("2024-01-05 16:00:00"))
    morning_expected = preprocessor.get_expected_baseline("AC_kWh", pd.Timestamp("2024-01-05 06:00:00"))
    assert afternoon_expected > morning_expected


def test_chronological_split():
    """Test chronological train/test splitting."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
        "Appliance": np.arange(100)
    })
    preprocessor = ElectricityDataPreprocessor()
    train_df, test_df = preprocessor.train_test_split_chronological(df, train_ratio=0.7)

    assert len(train_df) == 70
    assert len(test_df) == 30
    # Verify strict time ordering
    assert train_df["timestamp"].max() < test_df["timestamp"].min()
