"""
Unit Tests for Alert Engine, Report Generator, and Input Validation
Smart Electricity Consumption Anomaly Detection and Appliance Identification System
"""

import pytest
import pandas as pd
import numpy as np

from utils.alert_engine import AlertEngine, APPLIANCE_DIAGNOSTICS
from utils.report_generator import ElectricityReportGenerator
from preprocessing import ElectricityDataPreprocessor


# =============================================================================
# ALERT ENGINE TESTS
# =============================================================================

def test_alert_engine_diagnostics_known_appliance():
    """Verify diagnostic explanations and energy saving tips for AC."""
    diag = AlertEngine.get_diagnostics_for_appliance("Air Conditioner", is_high_surge=True)

    assert diag["appliance"] == "Air Conditioner"
    assert len(diag["possible_reasons"]) > 0
    assert len(diag["energy_saving_suggestions"]) > 0
    assert "Disclaimer" in diag["disclaimer"]
    # Verify no false claim of hardware fault
    assert "does not confirm" in diag["disclaimer"].lower()


def test_alert_engine_generic_fallback():
    """Verify generic fallback for unlisted appliance."""
    diag = AlertEngine.get_diagnostics_for_appliance("Industrial Welder", is_high_surge=True)
    assert len(diag["possible_reasons"]) > 0
    assert len(diag["energy_saving_suggestions"]) > 0


def test_build_alert_card():
    """Verify alert card formatting and severity indicators."""
    record = {
        "appliance": "Refrigerator",
        "actual_kwh": 1.25,
        "normal_kwh": 0.15,
        "difference_kwh": 1.10,
        "timestamp": "2024-02-15 14:00:00",
        "severity": "Critical"
    }
    card = AlertEngine.build_alert_card(record)
    assert card["severity"] == "Critical"
    assert card["badge_text"] == "CRITICAL ANOMALY"
    assert card["is_high"] is True
    assert card["difference_kwh"] == 1.10


# =============================================================================
# REPORT GENERATOR TESTS
# =============================================================================

@pytest.fixture
def sample_annotated_data():
    timestamps = pd.date_range("2024-01-01", periods=24 * 35, freq="h")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "Air_Conditioner_kWh": np.random.uniform(0.5, 3.0, len(timestamps)),
        "Refrigerator_kWh": np.random.uniform(0.1, 0.25, len(timestamps)),
        "Total_Consumption_kWh": np.random.uniform(1.0, 4.0, len(timestamps))
    })
    anomalies = pd.DataFrame([
        {
            "timestamp": pd.Timestamp("2024-01-05 14:00:00"),
            "appliance": "Air Conditioner",
            "normal_kwh": 1.5,
            "actual_kwh": 9.5,
            "difference_kwh": 8.0,
            "difference_pct": 533.3,
            "status": "Abnormal",
            "anomaly_type": "High Consumption Surge",
            "severity": "Critical"
        }
    ])
    return df, anomalies


def test_daily_report_generation(sample_annotated_data):
    """Verify daily report compiles correctly."""
    df, anomalies = sample_annotated_data
    gen = ElectricityReportGenerator(df, anomalies)
    rep = gen.generate_daily_report(pd.Timestamp("2024-01-05"))

    assert "total_consumption_kwh" in rep
    assert rep["total_consumption_kwh"] > 0
    assert rep["anomalies_count"] == 1
    assert not rep["appliance_summary_df"].empty


def test_weekly_report_generation(sample_annotated_data):
    """Verify weekly report compiles last 7 days and finds highest consuming day."""
    df, anomalies = sample_annotated_data
    gen = ElectricityReportGenerator(df, anomalies)
    rep = gen.generate_weekly_report(pd.Timestamp("2024-01-15"))

    assert "total_consumption_kwh" in rep
    assert "highest_consuming_day" in rep
    assert "date" in rep["highest_consuming_day"]
    assert len(rep["daily_breakdown_df"]) == 7


def test_monthly_report_generation(sample_annotated_data):
    """Verify monthly report aggregates monthly usage."""
    df, anomalies = sample_annotated_data
    gen = ElectricityReportGenerator(df, anomalies)
    rep = gen.generate_monthly_report("2024-01")

    assert "total_consumption_kwh" in rep
    assert rep["days_monitored"] > 25
    assert not rep["appliance_comparison_df"].empty


# =============================================================================
# INVALID INPUTS & ERROR HANDLING TESTS
# =============================================================================

def test_missing_timestamp_error():
    """Verify exception raised when dataset has no parseable timestamp."""
    df_invalid = pd.DataFrame({
        "app1": [1, 2, 3],
        "app2": [4, 5, 6]
    })
    preprocessor = ElectricityDataPreprocessor()
    with pytest.raises(ValueError, match="Could not automatically identify a timestamp column"):
        preprocessor.detect_columns(df_invalid)


def test_no_numeric_columns_error():
    """Verify exception raised when dataset has no numeric appliance data."""
    df_invalid = pd.DataFrame({
        "timestamp": ["2024-01-01", "2024-01-02"],
        "category": ["A", "B"],
        "status": ["active", "idle"]
    })
    preprocessor = ElectricityDataPreprocessor()
    with pytest.raises(ValueError, match="No numeric appliance electricity reading columns found"):
        preprocessor.detect_columns(df_invalid)
