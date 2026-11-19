# ⚡ AI-Based Smart Electricity Consumption Anomaly Detection and Appliance Identification System

> Final-Year Engineering College Project  
> Technology: Python · Streamlit · Scikit-learn · Pandas · Plotly

---

## 📋 Project Abstract

This system provides an intelligent, AI-powered web dashboard for real-time monitoring of household appliance electricity consumption. Using the **Isolation Forest** unsupervised machine learning algorithm, it learns the normal hourly consumption pattern of each individual appliance and automatically flags statistically unusual readings — both excessive surges and abnormal drops.

The system is built on the **REFIT Electrical Load Measurements** dataset format and ships with a realistic 90-day synthetic benchmark dataset for immediate evaluation without external data dependencies.

---

## 🎯 Key Features

| Feature | Description |
|---|---|
| **Per-Appliance ML Models** | Dedicated Isolation Forest trained for each monitored appliance |
| **Diurnal Baseline Profiling** | 24-hour × 7-day historical normal consumption profiles |
| **Chronological Train/Test Split** | Train on past, evaluate on future (no data leakage) |
| **Multi-Severity Anomaly Classification** | Critical / High / Medium / Low with coloured alerts |
| **Daily / Weekly / Monthly Reports** | Structured electricity audit trails |
| **CSV Upload & Auto Column Detection** | Dynamic mapping for custom real-world datasets |
| **Downloadable CSV Exports** | All reports and anomaly logs downloadable |
| **Live Anomaly Simulator** | Test any reading in real-time against the trained model |
| **Diagnostic Alert Cards** | Possible causes and energy-saving recommendations per appliance |

---

## 🗂️ Project Structure

```
finalyear project/
│
├── app.py                      # Main Streamlit web application (all 7 modules)
├── train_model.py              # Isolation Forest model training pipeline
├── detect_anomaly.py           # Anomaly detection engine using trained models
├── preprocessing.py            # Data ingestion, cleaning, feature engineering
│
├── dataset/
│   ├── generate_dataset.py     # Synthetic dataset generator (90 days, 6 appliances)
│   └── electricity_data.csv    # Pre-generated benchmark dataset (auto-created)
│
├── models/                     # Joblib-persisted trained model files (auto-created)
│   ├── Air_Conditioner_kWh_model.joblib
│   ├── Refrigerator_kWh_model.joblib
│   ├── Washing_Machine_kWh_model.joblib
│   ├── Fan_kWh_model.joblib
│   ├── Lights_kWh_model.joblib
│   ├── Television_kWh_model.joblib
│   └── model_metadata.joblib   # Baseline profiles, appliance stats, metadata
│
├── utils/
│   ├── alert_engine.py         # Diagnostic reasons and energy-saving suggestions
│   ├── report_generator.py     # Daily / Weekly / Monthly report engine
│   └── ui_components.py        # Plotly chart builders and CSS stylesheet
│
├── tests/
│   ├── test_preprocessing.py   # Unit tests: column detection, imputation, baselines
│   ├── test_anomaly_detection.py # Unit tests: model loading, anomaly classification
│   └── test_alert_and_reports.py # Unit tests: alerts, report generation, error handling
│
├── pages/                      # Reserved for multi-page Streamlit expansion
├── .streamlit/
│   └── config.toml             # Streamlit server configuration (headless, port)
├── pytest.ini                  # Pytest root configuration
└── requirements.txt            # Python dependency manifest
```

---

## 🧠 System Architecture

```
CSV Dataset (Uploaded or Default)
         │
         ▼
┌─────────────────────────┐
│  preprocessing.py       │
│  · Auto column detect   │
│  · NaN imputation       │
│  · Negative clipping    │
│  · Chronological sort   │
│  · Baseline profiles    │
│  · Feature engineering  │
└─────────────────────────┘
         │
         ├──── Train (80%) ──────────────────────────────────┐
         │                                                    │
         ▼                                                    ▼
┌─────────────────────────┐                    ┌─────────────────────────┐
│  train_model.py         │                    │  Test / Live Set (20%)  │
│  Per-appliance pipeline:│                    └─────────────────────────┘
│  StandardScaler         │                                   │
│  + IsolationForest      │                                   ▼
│  Saved to models/       │◄───────────────── detect_anomaly.py
└─────────────────────────┘                    · Anomaly score computation
                                               · Severity classification
                                               · Baseline deviation comparison
                                                          │
                                                          ▼
                                              ┌─────────────────────────┐
                                              │      app.py             │
                                              │  Streamlit Modules:     │
                                              │  1. Home Dashboard      │
                                              │  2. Appliance Monitor   │
                                              │  3. Anomaly Detection   │
                                              │  4. Alerts & Suggestions│
                                              │  5. Usage Reports       │
                                              │  6. Live Simulator      │
                                              │  7. Dataset Upload      │
                                              └─────────────────────────┘
```

---

## ⚙️ Installation & Execution (Windows / macOS / Linux)

### Step 1 — Clone or download the project folder

```bash
cd "finalyear project"
```

### Step 2 — Create and activate a Python virtual environment

**Windows (VS Code / Command Prompt):**
```bat
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install all required Python packages

```bash
pip install -r requirements.txt
```

### Step 4 — Generate synthetic dataset (auto-runs if missing)

```bash
python dataset/generate_dataset.py
```

### Step 5 — Train the Isolation Forest models

```bash
python train_model.py
```

Expected output:
```
[*] Ingesting dataset from dataset/electricity_data.csv...
[*] Detected 6 monitored appliances: Air_Conditioner_kWh, Refrigerator_kWh, ...
[*] Chronological Split: Train records = 1728, Test records = 432
[+] Training model for: 'Air_Conditioner_kWh'...  → Saved
...
[✓] ALL APPLIANCE MODELS TRAINED AND SAVED SUCCESSFULLY
```

### Step 6 — Launch the Streamlit web application

```bash
streamlit run app.py
```

Then open: **http://localhost:8501** in your browser.

---

## 🧪 Running Tests

```bash
pytest -v
```

19 unit tests covering preprocessing, anomaly detection, alert engine, and report generation.

---

## 📊 Module Descriptions

### Module 1 — Home Dashboard
Displays five KPI summary cards: total electricity consumed (kWh), monitored appliances count, expected normal consumption baseline, number of Isolation Forest-detected anomalies, and highest-consuming appliance. Includes aggregate consumption timeline and appliance share donut chart. Quick appliance status grid shows green (normal) or red (anomaly detected).

### Module 2 — Appliance Monitoring
Interactive selector for any detected appliance in the uploaded dataset. Timespan filter (7/14/30 days or full). Per-appliance KPI cards (total, average, peak, status). Detailed time-series chart with actual consumption (blue line), expected historical baseline (grey dashed), and Isolation Forest anomaly markers (red dots). 24-hour diurnal profile bar chart.

### Module 3 — Anomaly Detection
Interactive anomaly audit table with appliance filter, severity filter, and type filter (High Surge / Abnormal Low). Bubble scatter map showing anomaly magnitude versus time. All six required fields displayed: Appliance Name, Normal kWh, Actual kWh, Difference (kWh and %), Timestamp, Status. One-click CSV export.

### Module 4 — Alerts and Suggestions
Alert cards with colour-coded severity borders (Critical = red, High = orange, Medium = amber, Low = blue). Each card shows detection timestamp, actual vs. normal draw. Side-by-side: possible investigative reasons (appliance-specific knowledge base) and energy-saving suggestions. Compliance disclaimer on every card confirming the system does not confirm physical faults.

### Module 5 — Electricity Usage Reports
Three tabbed sub-reports:
- **Daily**: Date picker, hourly trend table, appliance-wise breakdown with share%, downloadable CSV
- **Weekly**: 7-day trend, highest-consuming day identification, weekly appliance totals, downloadable CSV
- **Monthly**: Month selector, monthly appliance comparison table, downloadable CSV

### Module 6 — Live Anomaly Simulator
Real-time interactive testing. User selects appliance, date, hour, and enters a custom kWh reading. System displays the historical normal baseline for that exact hour/day, then runs the trained Isolation Forest and returns: Status, Actual, Normal, Difference, Severity classification, and if abnormal — full diagnostic alert card.

### Module 7 — Dataset Upload & Retraining
CSV uploader with dynamic column detection. Timestamp and total column auto-mapped; remaining numeric columns become appliances. Manual column mapping UI for custom datasets. Triggers full retraining on new dataset. Download button for the benchmark dataset.

---

## 🔬 Machine Learning Implementation Details

### Algorithm: Isolation Forest
- **Type**: Unsupervised anomaly detection (no labelled training data required)
- **Principle**: Builds an ensemble of random decision trees. Anomalies are isolated in fewer splits (shorter path length → lower anomaly score)
- **Contamination**: 3% (expected fraction of anomalies in training set)
- **N Estimators**: 150 trees
- **Pipeline**: `StandardScaler → IsolationForest` (scaled feature space)

### Feature Engineering (8 features per appliance)
| Feature | Description |
|---|---|
| `consumption` | Raw kWh reading |
| `hour` | Hour of day (0–23) |
| `day_of_week` | Day of week (0=Mon, 6=Sun) |
| `is_weekend` | Binary weekend flag |
| `rolling_mean_24h` | 24-hour rolling average |
| `rolling_std_24h` | 24-hour rolling standard deviation |
| `hourly_deviation` | Difference from historical hour-of-day mean |
| `ratio_to_rolling` | Current reading ÷ rolling mean |

### Why Isolation Forest over simple thresholds?
Simple threshold rules (e.g., flag > 5 kWh) produce false alarms for high-draw appliances (AC) during peak afternoon hours when elevated consumption is completely normal. The Isolation Forest learns **context-aware** patterns using all 8 features, correctly classifying 2.5 kWh AC consumption at 3:00 AM as anomalous while the same 2.5 kWh at 2:00 PM is normal.

---

## 📦 Requirements

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
joblib>=1.3.0
pytest>=7.4.0
```

---

## ⚠️ Dataset Disclaimer

The bundled `electricity_data.csv` is a **synthetic benchmark dataset** generated for educational and system evaluation purposes. It simulates realistic 90-day, hourly appliance power patterns with deliberately injected anomaly events (compressor surges, motor jams, continuous door-open refrigerator draw, etc.). It does **not** represent real sensor measurements from actual households.

---

## 👩‍💻 Developer Notes for Explanation

**How to explain the Isolation Forest to your evaluator:**
> "The Isolation Forest works by randomly selecting a feature and a split value. Normal data points require many splits to isolate because they appear in dense clusters. Anomalies are sparse and isolated quickly in fewer splits. The anomaly score is inversely proportional to the average path length across all trees. We set the contamination parameter to 3%, meaning the model expects approximately 3 in every 100 readings to be anomalous."

**How to explain the baseline profile:**
> "Before anomaly detection, the system builds a historical diurnal profile — a table of median consumption for every combination of hour-of-day and day-of-week. This becomes the 'expected normal' reference. When the Isolation Forest flags a reading, we compare it against this baseline to quantify the deviation and determine whether it is a high surge or an abnormally low reading."

---

*AI-Based Smart Electricity Consumption Anomaly Detection and Appliance Identification System — Final Year Engineering Project*
