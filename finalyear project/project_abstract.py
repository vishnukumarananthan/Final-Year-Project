# ============================================================
# AI-Based Smart Electricity Consumption Anomaly Detection
# and Appliance Identification System
# ============================================================
# Project Abstract
# ============================================================

PROJECT_ABSTRACT = """
AI-Based Smart Electricity Consumption Anomaly Detection
and Appliance Identification System

ABSTRACT

Electricity wastage due to undetected appliance malfunctions, door-seal failures,
stuck relays, and unmonitored continuous operation contributes significantly to
domestic energy bills and environmental carbon emissions.

This project presents an AI-powered smart electricity monitoring web application
that continuously analyses sub-metered power consumption data for individual
household appliances. The system applies the Isolation Forest unsupervised
machine learning algorithm — originally proposed by Liu, Ting, and Zhou (2008)
— to identify statistically anomalous consumption patterns without requiring
pre-labelled training data.

The application ingests hourly appliance-level electricity readings (compatible
with the REFIT Electrical Load Measurements Dataset benchmark format), cleans
and preprocesses the data, engineers an eight-dimensional feature vector for
each appliance reading (capturing consumption level, time-of-day, day-of-week
context, 24-hour rolling statistics, and historical deviation), and trains
dedicated Isolation Forest models using chronologically separated training sets.

When an anomaly is detected — such as an air conditioner drawing 9.5 kWh at
3:00 AM when its normal midnight baseline is 0.05 kWh — the system identifies
the responsible appliance, quantifies the deviation against the learned diurnal
baseline profile, classifies the event severity (Critical / High / Medium / Low),
and surfaces actionable diagnostic hypotheses and energy-saving recommendations
through a modern, responsive Streamlit web dashboard.

The system comprises seven functional modules: Home Dashboard, Appliance
Monitoring, Anomaly Detection, Alerts and Suggestions, Electricity Usage Reports
(Daily / Weekly / Monthly with CSV exports), Live Real-Time Anomaly Simulator,
and Dataset Upload with On-Demand Model Retraining.

Empirical evaluation on a 90-day, 2,160-record synthetic benchmark dataset
demonstrates the system's ability to correctly classify normal readings as
normal and injected high-surge and low-draw anomalies as abnormal across six
independently monitored household appliances.

Keywords: Anomaly Detection, Isolation Forest, Smart Metering, Energy Monitoring,
Unsupervised Machine Learning, Streamlit, Appliance Identification
"""

SYSTEM_ARCHITECTURE = """
SYSTEM ARCHITECTURE

Layer 1: Data Ingestion
  · Accepts CSV uploads via Streamlit file uploader
  · Supports REFIT format and any custom appliance-level CSV
  · Automatic timestamp column detection (keyword matching + dtype inference)
  · Automatic total consumption column detection (keyword + substring matching)
  · Remaining numeric columns dynamically assigned as appliance monitors

Layer 2: Data Preprocessing (preprocessing.py)
  · Timestamp parsing and chronological sorting
  · Missing value imputation: forward-fill → back-fill → zero
  · Negative reading clipping (physically impossible power values)
  · Diurnal baseline profile computation (median kWh per hour × day-of-week)
  · Eight-feature vector extraction per appliance reading
  · 80/20 chronological train/test split

Layer 3: Machine Learning (train_model.py)
  · Per-appliance scikit-learn Pipeline: StandardScaler + IsolationForest
  · contamination = 0.03, n_estimators = 150, random_state = 42
  · Models persisted with joblib to models/ directory
  · Metadata bundle (baseline profiles, appliance stats) saved centrally

Layer 4: Anomaly Detection (detect_anomaly.py)
  · Loads persisted models and metadata
  · Applies trained pipeline to new observations
  · Returns IsolationForest prediction (-1=anomaly, +1=normal)
  · Computes deviation against diurnal baseline (kWh and %)
  · Classifies severity based on deviation magnitude and anomaly score
  · Fallback: 3-sigma statistical rule if no model available for appliance

Layer 5: Presentation (app.py + utils/)
  · Streamlit multi-module single-page application
  · Plotly interactive charts (line, scatter, donut, bar)
  · Custom CSS (Plus Jakarta Sans typography, metric cards, alert cards)
  · Alert engine with appliance-specific knowledge base (5 appliances + generic)
  · Report generator (daily / weekly / monthly aggregations + CSV export)
"""

if __name__ == "__main__":
    print(PROJECT_ABSTRACT)
    print(SYSTEM_ARCHITECTURE)
