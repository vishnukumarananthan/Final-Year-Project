"""
Smart Electricity Consumption Anomaly Detection and Appliance Identification System
Main Streamlit Web Application

Modules:
1. Home Dashboard
2. Appliance Monitoring
3. Anomaly Detection
4. Alerts and Suggestions
5. Electricity Usage Reports
6. Live Real-Time Simulator
7. Dataset Upload & Model Retraining
"""

import os
import io
import datetime
from typing import Dict, List, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np

from preprocessing import ElectricityDataPreprocessor
from detect_anomaly import ApplianceAnomalyDetector, format_appliance_name
from utils.alert_engine import AlertEngine
from utils.report_generator import ElectricityReportGenerator
from utils.ui_components import (
    get_custom_css,
    create_total_consumption_chart,
    create_appliance_share_donut,
    create_appliance_timeline_with_anomalies,
    create_anomaly_scatter_map,
    create_hourly_profile_chart,
    COLOR_NORMAL,
    COLOR_ABNORMAL,
    COLOR_WARNING
)
from train_model import ApplianceModelTrainer, MODELS_DIR, DEFAULT_DATASET_PATH


# =============================================================================
# STREAMLIT APP CONFIGURATION & STYLING
# =============================================================================
st.set_page_config(
    page_title="Smart Electricity Consumption Anomaly Detector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(get_custom_css(), unsafe_allow_html=True)


# =============================================================================
# SESSION STATE & DATA INITIALIZATION
# =============================================================================
@st.cache_resource
def get_detector() -> ApplianceAnomalyDetector:
    """Load and cache anomaly detection models."""
    detector = ApplianceAnomalyDetector(models_dir=MODELS_DIR)
    return detector


@st.cache_data
def load_default_data() -> pd.DataFrame:
    """Load default dataset from disk."""
    if not os.path.exists(DEFAULT_DATASET_PATH):
        from dataset.generate_dataset import generate_synthetic_electricity_data
        generate_synthetic_electricity_data(output_path=DEFAULT_DATASET_PATH)
    return pd.read_csv(DEFAULT_DATASET_PATH)


# Initialize session state for working dataset
if "raw_df" not in st.session_state:
    st.session_state.raw_df = load_default_data()

if "is_custom_dataset" not in st.session_state:
    st.session_state.is_custom_dataset = False

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = "Default Benchmark Dataset (REFIT-Simulated)"


# Process data with Anomaly Detector
detector = get_detector()

# If models are not trained yet, train them automatically
if not detector.is_loaded:
    with st.spinner("Initial setup: Training appliance Isolation Forest models..."):
        trainer = ApplianceModelTrainer()
        trainer.train_models_from_dataset(dataset_path=DEFAULT_DATASET_PATH)
        detector.load_models()

try:
    df_annotated, anomalies_df, summary = detector.analyze_dataset(st.session_state.raw_df)
    report_gen = ElectricityReportGenerator(
        df_annotated=df_annotated,
        anomalies_df=anomalies_df,
        ts_col=detector.preprocessor.timestamp_col or "timestamp",
        total_col=detector.preprocessor.total_col or "Total_Consumption_kWh"
    )
except Exception as e:
    st.error(f"Error processing dataset: {e}")
    st.stop()


# =============================================================================
# SIDEBAR NAVIGATION & FILTERS
# =============================================================================
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
        <span style="font-size: 2.4rem;">⚡</span>
        <h2 style="margin: 0.3rem 0 0 0; font-size: 1.25rem; font-weight: 800; color: #1E293B;">Smart Power AI</h2>
        <p style="margin: 0; font-size: 0.75rem; color: #64748B; font-weight: 500;">
            Anomaly Detection & Appliance Identification
        </p>
    </div>
    <hr style="margin: 0.8rem 0 1.2rem 0; border: none; border-top: 1px solid #E2E8F0;">
    """,
    unsafe_allow_html=True
)

navigation_choice = st.sidebar.radio(
    "NAVIGATION MENU",
    [
        "📊 Module 1: Home Dashboard",
        "🔌 Module 2: Appliance Monitoring",
        "🎯 Module 3: Anomaly Detection",
        "🚨 Module 4: Alerts & Suggestions",
        "📑 Module 5: Electricity Reports",
        "🧪 Live Anomaly Simulator",
        "⚙️ Dataset Upload & Retraining"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 SYSTEM STATUS")

col_s1, col_s2 = st.sidebar.columns(2)
col_s1.metric("Models Loaded", f"{len(detector.models)} Active")
col_s2.metric("Appliances", f"{len(detector.preprocessor.appliance_cols)}")

st.sidebar.markdown(
    f"""
    <div style="background: #F1F5F9; border-radius: 8px; padding: 0.8rem; font-size: 0.78rem; color: #475569; margin-top: 0.5rem;">
        <b>Current Dataset:</b><br>{st.session_state.dataset_name}<br>
        <b>Records:</b> {len(df_annotated):,} hourly readings<br>
        <b>Timespan:</b> {str(summary['start_time'])[:10]} to {str(summary['end_time'])[:10]}
    </div>
    """,
    unsafe_allow_html=True
)

if st.session_state.is_custom_dataset:
    if st.sidebar.button("🔄 Reset to Default Dataset"):
        st.session_state.raw_df = load_default_data()
        st.session_state.is_custom_dataset = False
        st.session_state.dataset_name = "Default Benchmark Dataset (REFIT-Simulated)"
        st.rerun()

st.sidebar.markdown(
    """
    <div style="font-size: 0.72rem; color: #94A3B8; text-align: center; margin-top: 1.5rem;">
        Final-Year College Engineering Project<br>
        AI-Based Smart Electricity Monitoring System
    </div>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# MODULE 1: HOME DASHBOARD
# =============================================================================
if navigation_choice == "📊 Module 1: Home Dashboard":
    st.markdown(
        """
        <div class="hero-banner">
            <h1>AI-Based Smart Electricity Consumption Dashboard</h1>
            <p>
                Continuous smart-meter monitoring, non-intrusive appliance disaggregation,
                and Isolation Forest anomaly detection for modern energy conservation.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Required Metrics:
    # 1. Total electricity consumption
    # 2. Number of monitored appliances
    # 3. Normal consumption
    # 4. Number of detected anomalies
    # 5. Highest-consuming appliance
    normal_total_estimate = sum(
        float(df_annotated[f"{app}_normal"].sum())
        for app in detector.preprocessor.appliance_cols
        if f"{app}_normal" in df_annotated.columns
    )

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-stripe" style="background: #2563EB;"></div>
                <div class="metric-title">Total Consumption</div>
                <div class="metric-value">{summary['total_consumption_kwh']:,.1f}</div>
                <div class="metric-caption">kWh across whole period</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-stripe" style="background: #0D9488;"></div>
                <div class="metric-title">Monitored Appliances</div>
                <div class="metric-value">{summary['num_monitored_appliances']}</div>
                <div class="metric-caption">Sub-metered circuits</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-stripe" style="background: #10B981;"></div>
                <div class="metric-title">Normal Baseline</div>
                <div class="metric-value">{normal_total_estimate:,.1f}</div>
                <div class="metric-caption">Expected baseline kWh</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi4:
        anomaly_count = summary['total_anomalies_detected']
        badge_style = "background: #EF4444;" if anomaly_count > 0 else "background: #10B981;"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-stripe" style="{badge_style}"></div>
                <div class="metric-title">Detected Anomalies</div>
                <div class="metric-value" style="color: {'#EF4444' if anomaly_count > 0 else '#10B981'};">{anomaly_count}</div>
                <div class="metric-caption">{summary['anomaly_rate_pct']}% of appliance readings</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-stripe" style="background: #F59E0B;"></div>
                <div class="metric-title">Highest Consumer</div>
                <div class="metric-value" style="font-size: 1.35rem; padding-top: 0.35rem;">{summary['highest_consuming_appliance']}</div>
                <div class="metric-caption">{summary['highest_consuming_kwh']:,.1f} kWh total</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # Charts Row
    chart_col1, chart_col2 = st.columns([1.5, 1])

    with chart_col1:
        fig_total = create_total_consumption_chart(
            df_annotated,
            ts_col=detector.preprocessor.timestamp_col,
            total_col=detector.preprocessor.total_col
        )
        st.plotly_chart(fig_total, use_container_width=True)

    with chart_col2:
        app_totals = {
            app: float(df_annotated[app].sum())
            for app in detector.preprocessor.appliance_cols
        }
        fig_donut = create_appliance_share_donut(app_totals)
        st.plotly_chart(fig_donut, use_container_width=True)

    # Appliance Status Quick Grid
    st.markdown("<div class='section-title'>⚡ Real-Time Monitored Appliances Status Overview</div>", unsafe_allow_html=True)

    app_cols = detector.preprocessor.appliance_cols
    grid_cols = st.columns(len(app_cols))

    for i, app in enumerate(app_cols):
        friendly = format_appliance_name(app)
        anom_count = summary["appliance_anomaly_counts"].get(app, 0)
        total_kwh = float(df_annotated[app].sum())

        if anom_count == 0:
            status_badge = '<span class="status-badge status-normal">NORMAL</span>'
            border_color = "#10B981"
        else:
            status_badge = f'<span class="status-badge status-abnormal">{anom_count} ANOMALIES</span>'
            border_color = "#EF4444"

        with grid_cols[i]:
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 4px solid {border_color}; border-radius: 10px; padding: 1rem; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <div style="font-weight: 700; color: #1E293B; font-size: 0.95rem; margin-bottom: 0.3rem;">{friendly}</div>
                    <div style="margin-bottom: 0.5rem;">{status_badge}</div>
                    <div style="font-size: 0.82rem; color: #64748B;">Total: <b>{total_kwh:,.1f}</b> kWh</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# =============================================================================
# MODULE 2: APPLIANCE MONITORING
# =============================================================================
elif navigation_choice == "🔌 Module 2: Appliance Monitoring":
    st.markdown(
        """
        <div class="hero-banner">
            <h1>Module 2: Appliance-Level Electricity Monitoring</h1>
            <p>
                Analyze sub-metered power consumption patterns for individual appliances,
                inspect hourly diurnal behavior, and observe normal operating limits.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    appliances = detector.preprocessor.appliance_cols
    friendly_options = {app: format_appliance_name(app) for app in appliances}

    sel_col1, sel_col2 = st.columns([1.5, 1])
    with sel_col1:
        selected_app = st.selectbox(
            "Select Appliance to Monitor:",
            options=appliances,
            format_func=lambda x: friendly_options[x]
        )
    with sel_col2:
        date_range_choice = st.selectbox(
            "Timespan Filter:",
            ["Full Dataset", "Last 30 Days", "Last 14 Days", "Last 7 Days"]
        )

    # Filter by date range
    df_filtered = df_annotated.copy()
    max_ts = df_filtered[detector.preprocessor.timestamp_col].max()
    if date_range_choice == "Last 7 Days":
        df_filtered = df_filtered[df_filtered[detector.preprocessor.timestamp_col] >= (max_ts - pd.Timedelta(days=7))]
    elif date_range_choice == "Last 14 Days":
        df_filtered = df_filtered[df_filtered[detector.preprocessor.timestamp_col] >= (max_ts - pd.Timedelta(days=14))]
    elif date_range_choice == "Last 30 Days":
        df_filtered = df_filtered[df_filtered[detector.preprocessor.timestamp_col] >= (max_ts - pd.Timedelta(days=30))]

    # Appliance KPI Cards
    app_total_kwh = float(df_filtered[selected_app].sum())
    app_max_kwh = float(df_filtered[selected_app].max())
    app_avg_kwh = float(df_filtered[selected_app].mean())
    app_anom_count = int(df_filtered[f"{selected_app}_is_anomaly"].sum()) if f"{selected_app}_is_anomaly" in df_filtered.columns else 0

    ak1, ak2, ak3, ak4 = st.columns(4)
    with ak1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-stripe" style="background: #2563EB;"></div>
                <div class="metric-title">Total Energy</div>
                <div class="metric-value">{app_total_kwh:,.2f}</div>
                <div class="metric-caption">kWh in selected period</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with ak2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-stripe" style="background: #0D9488;"></div>
                <div class="metric-title">Average Hourly Draw</div>
                <div class="metric-value">{app_avg_kwh:.3f}</div>
                <div class="metric-caption">kWh per active hour</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with ak3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-stripe" style="background: #F59E0B;"></div>
                <div class="metric-title">Peak Reading</div>
                <div class="metric-value">{app_max_kwh:.2f}</div>
                <div class="metric-caption">Maximum hourly spike (kWh)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with ak4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-stripe" style="background: {'#EF4444' if app_anom_count > 0 else '#10B981'};"></div>
                <div class="metric-title">Status</div>
                <div class="metric-value" style="font-size: 1.4rem; padding-top: 0.3rem;">
                    {'⚠️ ABNORMAL' if app_anom_count > 0 else '✅ NORMAL'}
                </div>
                <div class="metric-caption">{app_anom_count} anomalies detected</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)

    # Detailed Time-Series Chart with Anomaly Highlight Markers
    fig_app = create_appliance_timeline_with_anomalies(
        df_filtered,
        appliance_col=selected_app,
        ts_col=detector.preprocessor.timestamp_col,
        normal_col=f"{selected_app}_normal",
        anomaly_col=f"{selected_app}_is_anomaly"
    )
    st.plotly_chart(fig_app, use_container_width=True)

    # Diurnal Pattern and Historical Baseline
    st.markdown("<div class='section-title'>🕒 Historical Diurnal Pattern (24-Hour Cycle)</div>", unsafe_allow_html=True)

    if selected_app in detector.preprocessor.baseline_profiles:
        baseline_df = detector.preprocessor.baseline_profiles[selected_app]
        fig_diurnal = create_hourly_profile_chart(baseline_df, friendly_options[selected_app])
        st.plotly_chart(fig_diurnal, use_container_width=True)
    else:
        st.info("Baseline profile will be computed when models are trained.")


# =============================================================================
# MODULE 3: ANOMALY DETECTION
# =============================================================================
elif navigation_choice == "🎯 Module 3: Anomaly Detection":
    st.markdown(
        """
        <div class="hero-banner">
            <h1>Module 3: Machine Learning Anomaly Detection</h1>
            <p>
                Powered by Isolation Forest. Accurately flags both high surges and abnormal drops
                by comparing actual measurements against learned historical baselines.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if anomalies_df.empty:
        st.success("🎉 No anomalies detected in current dataset. All appliances operating within normal thresholds.")
    else:
        # Filters Bar
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            app_filter = st.selectbox(
                "Filter by Appliance:",
                ["All Appliances"] + list(anomalies_df["appliance"].unique())
            )
        with fcol2:
            severity_filter = st.selectbox(
                "Filter by Severity:",
                ["All Severities", "Critical", "High", "Medium", "Low"]
            )
        with fcol3:
            type_filter = st.selectbox(
                "Filter by Anomaly Type:",
                ["All Types", "High Consumption Surge", "Abnormal Low Consumption"]
            )

        # Apply Filters
        filtered_anomalies = anomalies_df.copy()
        if app_filter != "All Appliances":
            filtered_anomalies = filtered_anomalies[filtered_anomalies["appliance"] == app_filter]
        if severity_filter != "All Severities":
            filtered_anomalies = filtered_anomalies[filtered_anomalies["severity"] == severity_filter]
        if type_filter != "All Types":
            filtered_anomalies = filtered_anomalies[filtered_anomalies["anomaly_type"] == type_filter]

        # Anomaly Scatter Map
        st.plotly_chart(create_anomaly_scatter_map(filtered_anomalies), use_container_width=True)

        st.markdown(
            f"<div class='section-title'>📋 Anomaly Audit Table ({len(filtered_anomalies)} Records)</div>",
            unsafe_allow_html=True
        )

        # Required Fields to Display:
        # - Appliance name
        # - Normal electricity consumption
        # - Actual electricity consumption
        # - Difference in consumption
        # - Date and time of detection
        # - Normal or abnormal status
        display_df = filtered_anomalies[[
            "timestamp", "appliance", "normal_kwh", "actual_kwh",
            "difference_kwh", "difference_pct", "status", "severity", "anomaly_type"
        ]].copy()

        display_df.rename(columns={
            "timestamp": "Date & Time",
            "appliance": "Appliance Name",
            "normal_kwh": "Normal (kWh)",
            "actual_kwh": "Actual (kWh)",
            "difference_kwh": "Difference (kWh)",
            "difference_pct": "Difference (%)",
            "status": "Status",
            "severity": "Severity",
            "anomaly_type": "Anomaly Pattern"
        }, inplace=True)

        # Render styled dataframe
        st.dataframe(
            display_df,
            use_container_width=True,
            height=350,
            column_config={
                "Date & Time": st.column_config.DatetimeColumn("Date & Time", format="YYYY-MM-DD HH:mm"),
                "Normal (kWh)": st.column_config.NumberColumn(format="%.3f kWh"),
                "Actual (kWh)": st.column_config.NumberColumn(format="%.3f kWh"),
                "Difference (kWh)": st.column_config.NumberColumn(format="%+.3f kWh"),
                "Difference (%)": st.column_config.NumberColumn(format="%+.1f%%"),
            }
        )

        # CSV Download Button
        csv_buffer = io.StringIO()
        display_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Anomaly Audit CSV Report",
            data=csv_buffer.getvalue(),
            file_name=f"electricity_anomalies_{datetime.date.today()}.csv",
            mime="text/csv"
        )


# =============================================================================
# MODULE 4: ALERTS AND SUGGESTIONS
# =============================================================================
elif navigation_choice == "🚨 Module 4: Alerts & Suggestions":
    st.markdown(
        """
        <div class="hero-banner">
            <h1>Module 4: Diagnostic Alerts & Energy Conservation</h1>
            <p>
                Actionable diagnostic hypotheses and practical energy-saving suggestions for detected anomalies.
                <br><b>Compliance Note:</b> Explanations represent potential investigative hypotheses only and do not confirm physical electrical faults.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if anomalies_df.empty:
        st.success("✅ No active anomaly alerts! All appliances are consuming electricity within expected normal bounds.")
    else:
        # Filter alerts
        top_apps = ["All Appliances"] + list(anomalies_df["appliance"].unique())
        selected_alert_app = st.selectbox("Select Appliance to View Alerts & Action Plan:", top_apps)

        alert_records = anomalies_df.copy()
        if selected_alert_app != "All Appliances":
            alert_records = alert_records[alert_records["appliance"] == selected_alert_app]

        st.markdown(f"### ⚠️ Showing Top Anomaly Alerts ({min(len(alert_records), 10)} of {len(alert_records)} Total)")

        # Render alert cards
        for idx in range(min(len(alert_records), 10)):
            record = alert_records.iloc[idx].to_dict()
            card = AlertEngine.build_alert_card(record)

            css_class = f"alert-card-{card['severity'].lower()}"

            with st.container():
                st.markdown(
                    f"""
                    <div class="alert-card-container {css_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                            <span style="font-weight: 800; font-size: 1.05rem; color: #1E293B;">
                                {card['headline']}
                            </span>
                            <span class="status-badge" style="background: {card['severity_color']}20; color: {card['severity_color']}; border: 1px solid {card['severity_color']};">
                                {card['badge_text']}
                            </span>
                        </div>
                        <div style="display: flex; gap: 1.8rem; flex-wrap: wrap; font-size: 0.88rem; color: #334155; margin-bottom: 0.8rem;">
                            <div><b>Detection Time:</b> <span class="mono-time">{card['timestamp']}</span></div>
                            <div><b>Actual Draw:</b> <span style="color: #EF4444; font-weight: 700;">{card['actual_kwh']:.3f} kWh</span></div>
                            <div><b>Normal Baseline:</b> <span style="color: #10B981; font-weight: 700;">{card['normal_kwh']:.3f} kWh</span></div>
                            <div><b>Deviation:</b> <span style="font-weight: 700;">{card['difference_kwh']:+.3f} kWh</span></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("#### 🔍 Possible Reasons to Investigate:")
                    for reason in card["possible_reasons"]:
                        st.markdown(f"- {reason}")

                with col_b:
                    st.markdown("#### 💡 Electricity-Saving Suggestions:")
                    for tip in card["saving_suggestions"]:
                        st.markdown(f"- {tip}")

                st.markdown(
                    f"""
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.6rem 0.9rem; font-size: 0.76rem; color: #64748B; margin-top: 0.4rem; margin-bottom: 1.5rem;">
                        {card['disclaimer']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# =============================================================================
# MODULE 5: ELECTRICITY USAGE REPORTS
# =============================================================================
elif navigation_choice == "📑 Module 5: Electricity Reports":
    st.markdown(
        """
        <div class="hero-banner">
            <h1>Module 5: Electricity Usage Reports</h1>
            <p>
                Comprehensive Daily, Weekly, and Monthly electricity usage audits,
                appliance comparisons, and one-click downloadable CSV reports.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    report_tab1, report_tab2, report_tab3 = st.tabs([
        "📅 Daily Report",
        "📆 Weekly Report",
        "📊 Monthly Report"
    ])

    # -------------------------------------------------------------------------
    # 1. DAILY REPORT
    # -------------------------------------------------------------------------
    with report_tab1:
        st.subheader("Daily Electricity Consumption Audit")

        min_d = df_annotated[detector.preprocessor.timestamp_col].min().date()
        max_d = df_annotated[detector.preprocessor.timestamp_col].max().date()

        selected_date = st.date_input(
            "Select Audit Date:",
            value=max_d,
            min_value=min_d,
            max_value=max_d
        )

        daily_report = report_gen.generate_daily_report(pd.Timestamp(selected_date))

        if "error" in daily_report:
            st.warning(daily_report["error"])
        else:
            dr1, dr2, dr3 = st.columns(3)
            dr1.metric("Total Daily Consumption", f"{daily_report['total_consumption_kwh']:.2f} kWh")
            dr2.metric("Recorded Readings", f"{daily_report['readings_count']} hours")
            dr3.metric(
                "Anomalies Detected",
                f"{daily_report['anomalies_count']}",
                delta_color="inverse"
            )

            st.markdown("#### Appliance-Wise Daily Usage Breakdown")
            st.dataframe(
                daily_report["appliance_summary_df"],
                use_container_width=True,
                column_config={
                    "consumption_kwh": st.column_config.NumberColumn("Total kWh", format="%.3f kWh"),
                    "share_pct": st.column_config.NumberColumn("Usage Share (%)", format="%.1f%%"),
                    "peak_hourly_kwh": st.column_config.NumberColumn("Peak Hour (kWh)", format="%.3f kWh")
                }
            )

            st.markdown("#### Hourly Timeline for Selected Day")
            st.dataframe(daily_report["hourly_trend_df"], use_container_width=True, height=250)

            # Daily CSV Download
            csv_d = io.StringIO()
            daily_report["hourly_trend_df"].to_csv(csv_d, index=False)
            st.download_button(
                label=f"📥 Download Daily Report CSV ({selected_date})",
                data=csv_d.getvalue(),
                file_name=f"daily_report_{selected_date}.csv",
                mime="text/csv"
            )

    # -------------------------------------------------------------------------
    # 2. WEEKLY REPORT
    # -------------------------------------------------------------------------
    with report_tab2:
        st.subheader("Weekly Electricity Consumption Audit")

        max_d = df_annotated[detector.preprocessor.timestamp_col].max().date()
        week_end = st.date_input("Select Week Ending Date:", value=max_d, max_value=max_d)

        weekly_report = report_gen.generate_weekly_report(pd.Timestamp(week_end))

        if "error" in weekly_report:
            st.warning(weekly_report["error"])
        else:
            wr1, wr2, wr3 = st.columns(3)
            wr1.metric("7-Day Total Consumption", f"{weekly_report['total_consumption_kwh']:.2f} kWh")
            high_day = weekly_report["highest_consuming_day"]
            wr2.metric("Highest Consuming Day", f"{high_day['day_name']} ({high_day['consumption_kwh']:.1f} kWh)")
            wr3.metric("Weekly Anomalies Count", f"{weekly_report['weekly_anomalies_count']}")

            st.markdown("#### Day-by-Day 7-Day Trend")
            st.dataframe(weekly_report["daily_breakdown_df"], use_container_width=True)

            st.markdown("#### Weekly Appliance Totals")
            st.dataframe(
                weekly_report["weekly_appliance_df"],
                use_container_width=True,
                column_config={
                    "Total_Weekly_kWh": st.column_config.NumberColumn(format="%.2f kWh"),
                    "Share_Pct": st.column_config.NumberColumn(format="%.1f%%"),
                    "Daily_Avg_kWh": st.column_config.NumberColumn(format="%.2f kWh")
                }
            )

            # Weekly CSV Download
            csv_w = io.StringIO()
            weekly_report["daily_breakdown_df"].to_csv(csv_w, index=False)
            st.download_button(
                label="📥 Download Weekly Report CSV",
                data=csv_w.getvalue(),
                file_name=f"weekly_report_{weekly_report['start_date']}_to_{weekly_report['end_date']}.csv",
                mime="text/csv"
            )

    # -------------------------------------------------------------------------
    # 3. MONTHLY REPORT
    # -------------------------------------------------------------------------
    with report_tab3:
        st.subheader("Monthly Electricity Consumption Audit")

        available_months = sorted(
            df_annotated[detector.preprocessor.timestamp_col].dt.to_period("M").astype(str).unique(),
            reverse=True
        )
        selected_month = st.selectbox("Select Target Month:", available_months)

        monthly_report = report_gen.generate_monthly_report(selected_month)

        if "error" in monthly_report:
            st.warning(monthly_report["error"])
        else:
            mr1, mr2, mr3 = st.columns(3)
            mr1.metric("Monthly Total Consumption", f"{monthly_report['total_consumption_kwh']:.2f} kWh")
            mr2.metric("Daily Average Consumption", f"{monthly_report['daily_average_kwh']:.2f} kWh/day")
            mr3.metric("Monthly Anomalies", f"{monthly_report['monthly_anomalies_count']}")

            st.markdown("#### Appliance-Wise Comparison & Energy Share")
            st.dataframe(
                monthly_report["appliance_comparison_df"],
                use_container_width=True,
                column_config={
                    "Total_Monthly_kWh": st.column_config.NumberColumn(format="%.2f kWh"),
                    "Share_Pct": st.column_config.NumberColumn(format="%.1f%%"),
                    "Daily_Average_kWh": st.column_config.NumberColumn(format="%.2f kWh"),
                    "Max_Single_Hour_kWh": st.column_config.NumberColumn(format="%.3f kWh")
                }
            )

            # Monthly CSV Download
            csv_m = io.StringIO()
            monthly_report["daily_trend_df"].to_csv(csv_m, index=False)
            st.download_button(
                label=f"📥 Download Monthly Report CSV ({selected_month})",
                data=csv_m.getvalue(),
                file_name=f"monthly_report_{selected_month}.csv",
                mime="text/csv"
            )


# =============================================================================
# LIVE REAL-TIME ANOMALY SIMULATOR
# =============================================================================
elif navigation_choice == "🧪 Live Anomaly Simulator":
    st.markdown(
        """
        <div class="hero-banner">
            <h1>Live Real-Time Anomaly Simulator</h1>
            <p>
                Test any custom appliance reading in real-time. Verify how the trained Isolation Forest model
                compares the reading with historical hour-of-day baseline patterns.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    appliances = detector.preprocessor.appliance_cols
    sim_app = st.selectbox("Select Appliance to Test:", appliances, format_func=format_appliance_name)

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        sim_date = st.date_input("Simulation Date:", datetime.date.today())
        sim_hour = st.slider("Hour of Day:", min_value=0, max_value=23, value=14, help="0 = Midnight, 14 = 2:00 PM")

    with sim_col2:
        # Pre-populate suggested normal range
        expected_normal = detector.preprocessor.get_expected_baseline(
            sim_app,
            pd.Timestamp(f"{sim_date} {sim_hour:02d}:00:00")
        )
        st.info(f"💡 Historical Normal Baseline for **{format_appliance_name(sim_app)}** at {sim_hour:02d}:00 is approximately **{expected_normal:.3f} kWh**.")

        sim_kwh = st.number_input(
            "Enter Test Electricity Reading (kWh):",
            min_value=0.0,
            max_value=25.0,
            value=float(round(expected_normal * 3.5 if expected_normal > 0 else 5.0, 2)),
            step=0.1
        )

    if st.button("🚀 Analyze Reading with Isolation Forest", type="primary"):
        test_ts = f"{sim_date} {sim_hour:02d}:00:00"
        result = detector.test_single_reading(
            appliance=sim_app,
            timestamp_str=test_ts,
            reading_kwh=sim_kwh
        )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Model Prediction Output")

        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        with res_col1:
            status_color = "#10B981" if result["status"] == "Normal" else "#EF4444"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-stripe" style="background: {status_color};"></div>
                    <div class="metric-title">Classification</div>
                    <div class="metric-value" style="color: {status_color}; font-size: 1.4rem;">{result['status']}</div>
                    <div class="metric-caption">{result['anomaly_type']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with res_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-stripe" style="background: #2563EB;"></div>
                    <div class="metric-title">Actual Reading</div>
                    <div class="metric-value">{result['actual_kwh']:.3f}</div>
                    <div class="metric-caption">kWh input</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with res_col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-stripe" style="background: #0D9488;"></div>
                    <div class="metric-title">Normal Baseline</div>
                    <div class="metric-value">{result['normal_kwh']:.3f}</div>
                    <div class="metric-caption">kWh historical expected</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with res_col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-stripe" style="background: #F59E0B;"></div>
                    <div class="metric-title">Difference</div>
                    <div class="metric-value">{result['difference_kwh']:+.3f}</div>
                    <div class="metric-caption">{result['difference_pct']:+.1f}% deviation</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        if result["status"] == "Abnormal":
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            card = AlertEngine.build_alert_card({
                "appliance": result["appliance"],
                "actual_kwh": result["actual_kwh"],
                "normal_kwh": result["normal_kwh"],
                "difference_kwh": result["difference_kwh"],
                "timestamp": result["timestamp"],
                "severity": result["severity"]
            })

            st.markdown(
                f"""
                <div class="alert-card-container alert-card-{result['severity'].lower()}">
                    <div style="font-weight: 800; font-size: 1.1rem; color: #1E293B; margin-bottom: 0.5rem;">
                        {card['headline']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            ca, cb = st.columns(2)
            with ca:
                st.markdown("#### 🔍 Possible Reasons to Investigate:")
                for r in card["possible_reasons"]:
                    st.markdown(f"- {r}")
            with cb:
                st.markdown("#### 💡 Energy-Saving Recommendations:")
                for s in card["saving_suggestions"]:
                    st.markdown(f"- {s}")


# =============================================================================
# MODULE 7: DATASET UPLOAD & MODEL RETRAINING
# =============================================================================
elif navigation_choice == "⚙️ Dataset Upload & Retraining":
    st.markdown(
        """
        <div class="hero-banner">
            <h1>Dataset Management & Model Retraining</h1>
            <p>
                Upload any appliance-level electricity consumption CSV (REFIT or custom format).
                Dynamic column detection matches timestamps and appliances automatically.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload CSV Dataset:",
        type=["csv"],
        help="Upload an electricity dataset containing a timestamp column and appliance consumption readings."
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully read CSV file: {uploaded_file.name} ({len(uploaded_df):,} rows, {len(uploaded_df.columns)} columns)")

            st.markdown("### Raw Data Preview")
            st.dataframe(uploaded_df.head(5), use_container_width=True)

            # Column Mapping Assistance
            st.markdown("### Dynamic Column Mapping")
            cols = uploaded_df.columns.tolist()

            m_col1, m_col2 = st.columns(2)
            with m_col1:
                detected_ts = next((c for c in cols if any(t in c.lower() for t in ["time", "date"])), cols[0])
                ts_selection = st.selectbox("Timestamp Column:", cols, index=cols.index(detected_ts))

            with m_col2:
                total_candidates = [c for c in cols if "total" in c.lower() or "aggregate" in c.lower()]
                detected_total = total_candidates[0] if total_candidates else "Auto-calculate from appliances"
                total_selection = st.selectbox(
                    "Total Consumption Column (optional):",
                    ["Auto-calculate from appliances"] + cols,
                    index=0 if not total_candidates else cols.index(detected_total) + 1
                )

            st.markdown("#### Detected Monitored Appliances:")
            app_detected = [
                c for c in cols
                if c != ts_selection
                and (total_selection == "Auto-calculate from appliances" or c != total_selection)
                and pd.api.types.is_numeric_dtype(uploaded_df[c])
            ]
            st.write(", ".join([f"`{a}`" for a in app_detected]))

            if st.button("Apply Dataset & Retrain Models", type="primary"):
                with st.spinner("Retraining Isolation Forest models on uploaded dataset..."):
                    # Save uploaded dataset
                    save_path = "dataset/uploaded_electricity_data.csv"
                    uploaded_df.to_csv(save_path, index=False)

                    trainer = ApplianceModelTrainer()
                    trainer.train_models_from_dataset(dataset_path=save_path)

                    detector.load_models()
                    st.session_state.raw_df = uploaded_df
                    st.session_state.is_custom_dataset = True
                    st.session_state.dataset_name = uploaded_file.name

                    st.success("🎉 All models retrained and active! Navigating to Home Dashboard...")
                    st.rerun()

        except Exception as e:
            st.error(f"Failed to process CSV file: {e}")

    st.markdown("---")
    st.markdown("### 📥 Download Benchmark Sample Dataset")
    st.write("Need sample data to evaluate the application? Download the pre-configured 90-day realistic electricity dataset.")

    if os.path.exists(DEFAULT_DATASET_PATH):
        with open(DEFAULT_DATASET_PATH, "rb") as f:
            st.download_button(
                label="📥 Download electricity_data.csv (Benchmark)",
                data=f.read(),
                file_name="electricity_data.csv",
                mime="text/csv"
            )
