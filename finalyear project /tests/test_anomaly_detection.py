"""
Unit Tests for Machine Learning Anomaly Detection
Smart Electricity Consumption Anomaly Detection and Appliance Identification System
"""

import os
import pytest
import pandas as pd
import numpy as np

from detect_anomaly import ApplianceAnomalyDetector
from train_model import ApplianceModelTrainer, DEFAULT_DATASET_PATH


@pytest.fixture(scope="session")
def trained_detector():
    """Ensure models are trained and return detector instance."""
    detector = ApplianceAnomalyDetector()
    if not detector.is_loaded:
        trainer = ApplianceModelTrainer()
        trainer.train_models_from_dataset(dataset_path=DEFAULT_DATASET_PATH)
        detector.load_models()
    return detector


def test_models_loaded(trained_detector):
    """Verify models are loaded from disk."""
    assert trained_detector.is_loaded is True
    assert len(trained_detector.models) >= 5
    assert "Air_Conditioner_kWh" in trained_detector.models
    assert "Refrigerator_kWh" in trained_detector.models


def test_normal_reading_detection(trained_detector):
    """Verify normal readings are correctly classified as Normal."""
    # AC normal afternoon reading ~ 2.0 kWh
    res = trained_detector.test_single_reading(
        appliance="Air_Conditioner_kWh",
        timestamp_str="2024-02-10 14:00:00",
        reading_kwh=2.1
    )
    assert res["status"] == "Normal"
    assert res["severity"] == "None"
    assert res["anomaly_score"] >= -0.1


def test_unusually_high_reading_detection(trained_detector):
    """Verify severe high surge is classified as Abnormal and High/Critical severity."""
    # AC surge: 10.5 kWh (normal is ~2 kWh)
    res = trained_detector.test_single_reading(
        appliance="Air_Conditioner_kWh",
        timestamp_str="2024-02-10 14:00:00",
        reading_kwh=10.5
    )
    assert res["status"] == "Abnormal"
    assert res["anomaly_type"] == "High Consumption Surge"
    assert res["severity"] in ["High", "Critical"]
    assert res["difference_kwh"] > 5.0


def test_unseasonal_overnight_high_reading(trained_detector):
    """Verify that an appliance running high at 3 AM is flagged as anomalous."""
    # AC running at 7.5 kWh at 03:00 AM (historical baseline is ~0.05 kWh)
    res = trained_detector.test_single_reading(
        appliance="Air_Conditioner_kWh",
        timestamp_str="2024-02-10 03:00:00",
        reading_kwh=7.5
    )
    assert res["status"] == "Abnormal"
    assert res["difference_kwh"] > 5.0


def test_unusually_low_reading_detection(trained_detector):
    """Verify abnormal low reading (power loss / circuit breaker trip)."""
    # Refrigerator running at 0.001 kWh during peak afternoon
    res = trained_detector.test_single_reading(
        appliance="Refrigerator_kWh",
        timestamp_str="2024-02-10 13:00:00",
        reading_kwh=0.001
    )
    # Check baseline was positive
    assert res["normal_kwh"] > 0.05
    assert res["difference_kwh"] < 0.0


def test_dataset_analysis(trained_detector):
    """Verify batch dataset analysis returns annotated dataframe, anomalies table, and summary."""
    df = pd.read_csv(DEFAULT_DATASET_PATH)
    df_annotated, anom_df, summary = trained_detector.analyze_dataset(df)

    assert not df_annotated.empty
    assert "total_anomalies_detected" in summary
    assert summary["total_anomalies_detected"] > 0
    assert summary["num_monitored_appliances"] >= 5

    # Check fields in anomaly records
    if not anom_df.empty:
        first = anom_df.iloc[0]
        assert "appliance" in first
        assert "normal_kwh" in first
        assert "actual_kwh" in first
        assert "difference_kwh" in first
        assert "timestamp" in first
        assert "status" in first
        assert first["status"] == "Abnormal"
