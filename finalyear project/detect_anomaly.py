"""
Anomaly Detection Engine Module
Smart Electricity Consumption Anomaly Detection and Appliance Identification System

Loads trained Isolation Forest models and identifies appliances exhibiting abnormal consumption.
Computes expected normal baseline, consumption differences, anomaly scores, and severity classifications.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import joblib

from preprocessing import ElectricityDataPreprocessor


MODELS_DIR = "models"


def format_appliance_name(col_name: str) -> str:
    """Format raw column name to friendly display title."""
    clean = col_name.replace("_kWh", "").replace("_kwh", "").replace("_", " ")
    return clean.title()


class ApplianceAnomalyDetector:
    """
    Applies trained Isolation Forest models to detect and characterize electricity consumption anomalies.
    """

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.models: Dict[str, Any] = {}
        self.metadata: Optional[Dict[str, Any]] = None
        self.preprocessor = ElectricityDataPreprocessor()
        self.is_loaded = False
        self.load_models()

    def load_models(self) -> bool:
        """Load trained models and metadata from disk."""
        metadata_path = os.path.join(self.models_dir, "model_metadata.joblib")
        if not os.path.exists(metadata_path):
            self.is_loaded = False
            return False

        try:
            self.metadata = joblib.load(metadata_path)
            self.preprocessor.baseline_profiles = self.metadata.get("baseline_profiles", {})
            self.preprocessor.appliance_stats = self.metadata.get("appliance_stats", {})
            self.preprocessor.timestamp_col = self.metadata.get("timestamp_col", "timestamp")
            self.preprocessor.total_col = self.metadata.get("total_col", "Total_Consumption_kWh")
            self.preprocessor.appliance_cols = self.metadata.get("appliances", [])

            for app in self.preprocessor.appliance_cols:
                safe_app_name = "".join(c if c.isalnum() else "_" for c in app)
                model_file = os.path.join(self.models_dir, f"{safe_app_name}_model.joblib")
                if os.path.exists(model_file):
                    self.models[app] = joblib.load(model_file)

            self.is_loaded = len(self.models) > 0
            return self.is_loaded
        except Exception as e:
            print(f"[!] Error loading models: {e}")
            self.is_loaded = False
            return False

    def analyze_dataset(
        self,
        df: pd.DataFrame,
        custom_mapping: Optional[Dict[str, str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Run anomaly detection across an entire dataset.

        Returns:
        - df_annotated: Cleaned dataset with anomaly flags & baselines per appliance
        - anomalies_df: Table of all detected anomaly events
        - summary: High-level KPI metrics
        """
        # Ensure data is cleaned
        if self.preprocessor.timestamp_col is None:
            self.preprocessor.detect_columns(df, custom_mapping=custom_mapping)

        df_clean = self.preprocessor.clean_data(df)
        if not self.preprocessor.baseline_profiles:
            self.preprocessor.compute_baseline_profiles(df_clean)

        ts_col = self.preprocessor.timestamp_col
        total_col = self.preprocessor.total_col
        appliances = self.preprocessor.appliance_cols

        df_annotated = df_clean.copy()
        anomaly_records: List[Dict[str, Any]] = []

        total_anomalies_count = 0
        appliance_anomaly_counts: Dict[str, int] = {app: 0 for app in appliances}

        for app in appliances:
            model = self.models.get(app)
            features = self.preprocessor.extract_features_for_appliance(df_clean, app)

            if model is not None:
                # Real ML model predictions
                # IsolationForest: 1 = Normal, -1 = Anomaly
                raw_preds = model.predict(features)
                scores = model.decision_function(features)
                is_anomaly = (raw_preds == -1)
            else:
                # Fallback: statistical 3-sigma if a model was not trained for this specific appliance
                app_vals = df_clean[app]
                mean_val = app_vals.mean()
                std_val = max(app_vals.std(), 0.05)
                z_scores = np.abs((app_vals - mean_val) / std_val)
                is_anomaly = (z_scores > 3.0).values
                scores = -z_scores.values

            # Baselines & Differences
            expected_baselines = np.zeros(len(df_clean))
            ts_series = pd.to_datetime(df_clean[ts_col])
            for i, ts in enumerate(ts_series):
                expected_baselines[i] = self.preprocessor.get_expected_baseline(app, ts)

            diffs = df_clean[app].values - expected_baselines

            # Store in annotated DataFrame
            df_annotated[f"{app}_normal"] = np.round(expected_baselines, 3)
            df_annotated[f"{app}_is_anomaly"] = is_anomaly
            df_annotated[f"{app}_anomaly_score"] = np.round(scores, 4)
            df_annotated[f"{app}_diff"] = np.round(diffs, 3)

            # Record individual anomaly events
            for i in np.where(is_anomaly)[0]:
                actual = float(df_clean[app].iloc[i])
                expected = float(expected_baselines[i])
                diff = actual - expected
                score = float(scores[i])
                ts = df_clean[ts_col].iloc[i]

                if diff > 0:
                    atype = "High Consumption Surge"
                else:
                    atype = "Abnormal Low Consumption"

                # Calculate severity
                pct_dev = (diff / (expected + 0.01)) * 100
                abs_pct = abs(pct_dev)
                if abs_pct > 250 or score < -0.15:
                    severity = "Critical"
                elif abs_pct > 120 or score < -0.08:
                    severity = "High"
                elif abs_pct > 50:
                    severity = "Medium"
                else:
                    severity = "Low"

                anomaly_records.append({
                    "timestamp": ts,
                    "appliance_raw": app,
                    "appliance": format_appliance_name(app),
                    "actual_kwh": round(actual, 3),
                    "normal_kwh": round(expected, 3),
                    "difference_kwh": round(diff, 3),
                    "difference_pct": round(pct_dev, 1),
                    "status": "Abnormal",
                    "anomaly_type": atype,
                    "anomaly_score": round(score, 4),
                    "severity": severity
                })
                appliance_anomaly_counts[app] += 1
                total_anomalies_count += 1

        anomalies_df = pd.DataFrame(anomaly_records)
        if not anomalies_df.empty:
            anomalies_df = anomalies_df.sort_values(by="timestamp", ascending=False).reset_index(drop=True)

        # Compute summary KPIs
        total_energy = float(df_clean[total_col].sum())
        appliance_totals = {app: float(df_clean[app].sum()) for app in appliances}
        highest_app = max(appliance_totals, key=appliance_totals.get) if appliance_totals else "None"

        summary = {
            "total_consumption_kwh": round(total_energy, 2),
            "num_monitored_appliances": len(appliances),
            "monitored_appliances": appliances,
            "total_anomalies_detected": total_anomalies_count,
            "anomaly_rate_pct": round(total_anomalies_count / (len(df_clean) * max(len(appliances), 1)) * 100, 2),
            "highest_consuming_appliance": format_appliance_name(highest_app),
            "highest_consuming_kwh": round(appliance_totals.get(highest_app, 0.0), 2),
            "appliance_anomaly_counts": appliance_anomaly_counts,
            "records_count": len(df_clean),
            "start_time": str(df_clean[ts_col].min()),
            "end_time": str(df_clean[ts_col].max())
        }

        return df_annotated, anomalies_df, summary

    def test_single_reading(
        self,
        appliance: str,
        timestamp_str: str,
        reading_kwh: float,
        recent_readings_mean: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Interactive real-time test for a single observation.
        Useful for live demonstrations.
        """
        ts = pd.to_datetime(timestamp_str)
        hour = ts.hour
        dow = ts.day_of_week
        is_weekend = int(dow >= 5)

        expected = self.preprocessor.get_expected_baseline(appliance, ts)
        stats = self.preprocessor.get_appliance_stats(appliance)

        if recent_readings_mean is not None:
            rolling_mean = recent_readings_mean
            rolling_std = max(expected * 0.15, 0.05)
        else:
            rolling_mean = stats.get("rolling_mean_avg", expected if expected > 0 else 1.0)
            rolling_std = stats.get("rolling_std_avg", max(expected * 0.25, 0.1))

        hourly_dev = reading_kwh - expected
        ratio = reading_kwh / (rolling_mean + 0.01)

        feature_row = pd.DataFrame([{
            "consumption": reading_kwh,
            "hour": hour,
            "day_of_week": dow,
            "is_weekend": is_weekend,
            "rolling_mean_24h": rolling_mean,
            "rolling_std_24h": rolling_std,
            "hourly_deviation": hourly_dev,
            "ratio_to_rolling": ratio
        }])

        model = self.models.get(appliance)
        diff = reading_kwh - expected
        pct_dev = (diff / (expected + 0.01)) * 100

        if model is not None:
            pred = model.predict(feature_row)[0]
            score = float(model.decision_function(feature_row)[0])
            # Model prediction: -1 indicates anomaly
            # Validate deviation so that slight variations within normal bounds are not false alarms
            is_abnormal = (pred == -1) and (abs(diff) > max(expected * 0.35, 0.25) or abs(pct_dev) > 50)
        else:
            abs_diff = abs(diff)
            is_abnormal = abs_diff > (expected * 1.5 + 0.2)
            score = -abs_diff

        diff = reading_kwh - expected
        pct_dev = (diff / (expected + 0.01)) * 100

        if is_abnormal:
            status = "Abnormal"
            atype = "High Consumption Surge" if diff > 0 else "Abnormal Low Consumption"
            severity = "Critical" if abs(pct_dev) > 200 else ("High" if abs(pct_dev) > 100 else "Medium")
        else:
            status = "Normal"
            atype = "Normal Operation"
            severity = "None"

        return {
            "appliance": format_appliance_name(appliance),
            "timestamp": str(ts),
            "actual_kwh": round(reading_kwh, 3),
            "normal_kwh": round(expected, 3),
            "difference_kwh": round(diff, 3),
            "difference_pct": round(pct_dev, 1),
            "status": status,
            "anomaly_type": atype,
            "anomaly_score": round(score, 4),
            "severity": severity
        }
