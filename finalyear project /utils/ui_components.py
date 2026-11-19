"""
UI Components and Visualizations Module
Smart Electricity Consumption Anomaly Detection and Appliance Identification System

Contains reusable Plotly visualization builders and modern CSS design components.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# Brand Color Palette
COLOR_NORMAL = "#10B981"    # Emerald Green
COLOR_ABNORMAL = "#EF4444"  # Vibrant Crimson Red
COLOR_WARNING = "#F59E0B"   # Amber
COLOR_PRIMARY = "#2563EB"   # Royal Blue
COLOR_ACCENT = "#8B5CF6"    # Purple Accent
COLOR_DARK = "#1E293B"      # Slate 800
COLOR_BG = "#F8FAFC"        # Slate 50


def get_custom_css() -> str:
    """Return polished, responsive CSS stylesheet for Streamlit."""
    return """
    <style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Banner / Hero Styling */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F766E 100%);
        color: #FFFFFF;
        padding: 2rem 2.2rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-banner h1 {
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0;
        color: #F8FAFC;
        letter-spacing: -0.02em;
    }
    .hero-banner p {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-top: 0.4rem;
        margin-bottom: 0;
        line-height: 1.5;
    }

    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
    }
    .metric-card-stripe {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
        letter-spacing: -0.03em;
    }
    .metric-caption {
        font-size: 0.78rem;
        color: #94A3B8;
        margin-top: 0.4rem;
    }

    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .status-normal {
        background-color: #DEF7EC;
        color: #03543F;
        border: 1px solid #31C48D;
    }
    .status-abnormal {
        background-color: #FDE8E8;
        color: #9B1C1C;
        border: 1px solid #F98080;
    }
    .status-warning {
        background-color: #FEF08A;
        color: #713F12;
        border: 1px solid #FACC15;
    }

    /* Alert Banner Card */
    .alert-card-container {
        border-left: 5px solid;
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        border-top: 1px solid #F1F5F9;
        border-right: 1px solid #F1F5F9;
        border-bottom: 1px solid #F1F5F9;
    }
    .alert-card-critical { border-left-color: #EF4444; background: #FFF5F5; }
    .alert-card-high { border-left-color: #F97316; background: #FFF7ED; }
    .alert-card-medium { border-left-color: #F59E0B; background: #FFFBEB; }
    .alert-card-low { border-left-color: #3B82F6; background: #EFF6FF; }

    /* Code & Timestamps */
    .mono-time {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #475569;
        background: #F1F5F9;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
    }

    /* Section Subheaders */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    </style>
    """


# =============================================================================
# PLOTLY CHART BUILDERS
# =============================================================================

def create_total_consumption_chart(df: pd.DataFrame, ts_col: str, total_col: str) -> go.Figure:
    """Create interactive aggregate household electricity consumption timeline."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[ts_col],
        y=df[total_col],
        mode="lines",
        name="Total Consumption (kWh)",
        line=dict(color="#0284C7", width=2.2),
        fill="tozeroy",
        fillcolor="rgba(2, 132, 199, 0.08)",
        hovertemplate="<b>Date/Time:</b> %{x|%Y-%m-%d %H:%M}<br><b>Total Power:</b> %{y:.2f} kWh<extra></extra>"
    ))

    fig.update_layout(
        title="<b>Total Household Electricity Consumption Over Time</b>",
        title_font=dict(size=15, color="#1E293B"),
        xaxis=dict(
            title="Date and Time",
            showgrid=True,
            gridcolor="#F1F5F9",
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(title="Energy (kWh)", showgrid=True, gridcolor="#F1F5F9"),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=380
    )
    return fig


def create_appliance_share_donut(appliance_totals: Dict[str, float]) -> go.Figure:
    """Create modern donut chart showing percentage share of monitored appliances."""
    labels = [k.replace("_kWh", "").replace("_", " ").title() for k in appliance_totals.keys()]
    values = list(appliance_totals.values())

    colors = ["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899", "#14B8A6", "#64748B"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
        textinfo="label+percent",
        textposition="outside",
        hovertemplate="<b>%{label}</b><br>Total: %{value:.2f} kWh<br>Share: %{percent}<extra></extra>"
    )])

    fig.update_layout(
        title="<b>Appliance Electricity Usage Breakdown</b>",
        title_font=dict(size=15, color="#1E293B"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor="#FFFFFF",
        height=380
    )
    return fig


def create_appliance_timeline_with_anomalies(
    df: pd.DataFrame,
    appliance_col: str,
    ts_col: str = "timestamp",
    normal_col: Optional[str] = None,
    anomaly_col: Optional[str] = None
) -> go.Figure:
    """
    Create detailed appliance monitoring line chart with expected baseline band
    and highlighted red markers for Isolation Forest detected anomalies.
    """
    friendly_name = appliance_col.replace("_kWh", "").replace("_", " ").title()
    fig = go.Figure()

    # 1. Expected Baseline (if available)
    if normal_col and normal_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df[ts_col],
            y=df[normal_col],
            mode="lines",
            name="Normal Expected Baseline",
            line=dict(color="#94A3B8", width=1.5, dash="dash"),
            hovertemplate="<b>Baseline:</b> %{y:.3f} kWh<extra></extra>"
        ))

    # 2. Actual Consumption
    fig.add_trace(go.Scatter(
        x=df[ts_col],
        y=df[appliance_col],
        mode="lines",
        name="Actual Consumption",
        line=dict(color="#2563EB", width=2),
        hovertemplate="<b>Actual:</b> %{y:.3f} kWh<extra></extra>"
    ))

    # 3. Anomaly Markers (Isolation Forest Predictions)
    if anomaly_col and anomaly_col in df.columns:
        anom_mask = df[anomaly_col] == True
        anom_points = df[anom_mask]

        if not anom_points.empty:
            fig.add_trace(go.Scatter(
                x=anom_points[ts_col],
                y=anom_points[appliance_col],
                mode="markers",
                name="Detected Anomaly (Abnormal)",
                marker=dict(
                    color=COLOR_ABNORMAL,
                    size=9,
                    symbol="circle",
                    line=dict(color="#FFFFFF", width=1.5)
                ),
                hovertemplate="<b>ANOMALY ALERT!</b><br><b>Time:</b> %{x|%Y-%m-%d %H:%M}<br><b>Actual:</b> %{y:.3f} kWh<extra></extra>"
            ))

    fig.update_layout(
        title=f"<b>{friendly_name} Consumption & Anomaly Detection</b>",
        title_font=dict(size=15, color="#1E293B"),
        xaxis=dict(title="Timestamp", showgrid=True, gridcolor="#F1F5F9"),
        yaxis=dict(title="Electricity (kWh)", showgrid=True, gridcolor="#F1F5F9"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=400
    )
    return fig


def create_anomaly_scatter_map(anomalies_df: pd.DataFrame) -> go.Figure:
    """Create bubble scatter plot of all anomaly occurrences across time."""
    if anomalies_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Anomaly Events Detected", height=300)
        return fig

    color_map = {
        "Critical": "#EF4444",
        "High": "#F97316",
        "Medium": "#F59E0B",
        "Low": "#3B82F6"
    }

    fig = px.scatter(
        anomalies_df,
        x="timestamp",
        y="difference_kwh",
        color="severity",
        size="actual_kwh",
        hover_data=["appliance", "normal_kwh", "difference_pct", "anomaly_type"],
        color_discrete_map=color_map,
        title="<b>Anomaly Occurrences: Magnitude & Severity Timeline</b>"
    )

    fig.update_layout(
        xaxis=dict(title="Detection Timestamp", showgrid=True, gridcolor="#F1F5F9"),
        yaxis=dict(title="Difference from Normal (kWh)", showgrid=True, gridcolor="#F1F5F9"),
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=380
    )
    return fig


def create_hourly_profile_chart(baseline_df: pd.DataFrame, appliance_name: str) -> go.Figure:
    """Plot average 24-hour diurnal profile for an appliance."""
    hourly_avg = baseline_df.groupby("hour")["median"].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly_avg["hour"],
        y=hourly_avg["median"],
        marker_color="#3B82F6",
        name="Median kWh",
        hovertemplate="Hour %{x}:00: <b>%{y:.3f} kWh</b><extra></extra>"
    ))

    fig.update_layout(
        title=f"<b>24-Hour Diurnal Pattern: {appliance_name}</b>",
        xaxis=dict(title="Hour of Day (0 = 12 AM, 23 = 11 PM)", tickmode="linear", tick0=0, dtick=2),
        yaxis=dict(title="Expected Normal kWh", showgrid=True, gridcolor="#F1F5F9"),
        margin=dict(l=30, r=20, t=50, b=30),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=300
    )
    return fig
