"""
Model Training Module
Smart Electricity Consumption Anomaly Detection and Appliance Identification System

Trains dedicated Isolation Forest anomaly detection models for each individual household appliance.
Saves models and baseline metadata using joblib.
"""

import os
import sys
from typing import Dict, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from preprocessing import ElectricityDataPreprocessor
from dataset.generate_dataset import generate_synthetic_electricity_data


MODELS_DIR = "models"
DEFAULT_DATASET_PATH = "dataset/electricity_data.csv"


class ApplianceModelTrainer:
    """
    Trains and persists separate Isolation Forest models for individual appliances.
    """

    def __init__(self, contamination: float = 0.03, n_estimators: int = 150, random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.preprocessor = ElectricityDataPreprocessor()
        self.models: Dict[str, Pipeline] = {}
        self.training_metrics: Dict[str, Dict[str, Any]] = {}

    def train_models_from_dataset(
        self,
        dataset_path: str = DEFAULT_DATASET_PATH,
        models_dir: str = MODELS_DIR,
        train_ratio: float = 0.8
    ) -> Dict[str, Any]:
        """
        End-to-end training pipeline:
        1. Load or generate dataset
        2. Clean and preprocess
        3. Compute historical baselines
        4. Chronological train/test split
        5. Train Isolation Forest per appliance
        6. Evaluate and save models
        """
        os.makedirs(models_dir, exist_ok=True)

        if not os.path.exists(dataset_path):
            print(f"[!] Dataset not found at {dataset_path}. Generating simulated demonstration dataset...")
            generate_synthetic_electricity_data(output_path=dataset_path)

        print(f"[*] Ingesting dataset from {dataset_path}...")
        df_raw = pd.read_csv(dataset_path)

        # Preprocess and clean data
        df_clean = self.preprocessor.clean_data(df_raw)
        self.preprocessor.compute_baseline_profiles(df_clean)

        appliances = self.preprocessor.appliance_cols
        print(f"[*] Detected {len(appliances)} monitored appliances: {', '.join(appliances)}")

        # Chronological split
        train_df, test_df = self.preprocessor.train_test_split_chronological(df_clean, train_ratio=train_ratio)
        print(f"[*] Chronological Split: Train records = {len(train_df)}, Test records = {len(test_df)}")

        metadata = {
            "appliances": appliances,
            "timestamp_col": self.preprocessor.timestamp_col,
            "total_col": self.preprocessor.total_col,
            "baseline_profiles": self.preprocessor.baseline_profiles,
            "appliance_stats": self.preprocessor.appliance_stats,
            "cleaning_summary": self.preprocessor.cleaning_summary,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "metrics": {}
        }

        print("\n" + "=" * 65)
        print(" TRAINING INDIVIDUAL APPLIANCE ISOLATION FORESTS")
        print("=" * 65)

        for app in appliances:
            print(f"\n[+] Training model for: '{app}'...")

            # Extract features for training
            X_train = self.preprocessor.extract_features_for_appliance(train_df, app)
            X_test = self.preprocessor.extract_features_for_appliance(test_df, app)

            # Build pipeline: StandardScaler + IsolationForest
            iso_forest = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                max_samples="auto",
                random_state=self.random_state,
                n_jobs=1
            )
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("iso_forest", iso_forest)
            ])

            pipeline.fit(X_train)
            self.models[app] = pipeline

            # Save appliance model
            safe_app_name = "".join(c if c.isalnum() else "_" for c in app)
            model_file = os.path.join(models_dir, f"{safe_app_name}_model.joblib")
            joblib.dump(pipeline, model_file)

            # Evaluate on Test Set
            # -1 represents anomaly in scikit-learn IsolationForest, 1 represents normal
            test_preds = pipeline.predict(X_test)
            test_scores = pipeline.decision_function(X_test)

            n_anomalies = int((test_preds == -1).sum())
            anomaly_rate = float(n_anomalies / len(test_preds) * 100)
            avg_score = float(test_scores.mean())

            # Distinguish high vs low anomalies
            test_deviations = X_test["hourly_deviation"]
            high_anomalies = int(((test_preds == -1) & (test_deviations > 0)).sum())
            low_anomalies = int(((test_preds == -1) & (test_deviations < 0)).sum())

            metrics = {
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "test_anomalies": n_anomalies,
                "anomaly_rate_pct": round(anomaly_rate, 2),
                "high_consumption_anomalies": high_anomalies,
                "low_consumption_anomalies": low_anomalies,
                "mean_anomaly_score": round(avg_score, 4),
                "model_path": model_file
            }
            self.training_metrics[app] = metrics
            metadata["metrics"][app] = metrics

            print(f"    - Train Samples: {len(X_train)} | Test Samples: {len(X_test)}")
            print(f"    - Detected Test Anomalies: {n_anomalies} ({anomaly_rate:.2f}%)")
            print(f"      * Unusually High Usage: {high_anomalies}")
            print(f"      * Unusually Low Usage : {low_anomalies}")
            print(f"    - Model saved to: {model_file}")

        # Save central metadata bundle
        metadata_file = os.path.join(models_dir, "model_metadata.joblib")
        joblib.dump(metadata, metadata_file)
        print(f"\n[✓] Central metadata saved to: {metadata_file}")
        print("=" * 65)
        print(" ALL APPLIANCE MODELS TRAINED AND SAVED SUCCESSFULLY")
        print("=" * 65 + "\n")

        return metadata


if __name__ == "__main__":
    trainer = ApplianceModelTrainer(contamination=0.03, n_estimators=150)
    trainer.train_models_from_dataset()
