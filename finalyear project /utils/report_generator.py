"""
Electricity Usage Reports Engine
Module 5 of Smart Electricity Consumption Anomaly Detection and Appliance Identification System

Generates structured Daily, Weekly, and Monthly electricity usage reports,
aggregations, appliance comparisons, anomaly audit trails, and downloadable CSV exports.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np


class ElectricityReportGenerator:
    """
    Compiles operational electricity usage reports and downloadable exports.
    """

    def __init__(self, df_annotated: pd.DataFrame, anomalies_df: pd.DataFrame, ts_col: str = "timestamp", total_col: str = "Total_Consumption_kWh"):
        self.df = df_annotated.copy()
        self.anomalies_df = anomalies_df.copy()
        self.ts_col = ts_col
        self.total_col = total_col

        self.df[self.ts_col] = pd.to_datetime(self.df[self.ts_col])
        if not self.anomalies_df.empty:
            self.anomalies_df["timestamp"] = pd.to_datetime(self.anomalies_df["timestamp"])

        # Determine appliance columns
        exclude = ["timestamp", "total_consumption_kwh", "is_simulated", "total"]
        self.appliance_cols = [
            c for c in self.df.columns
            if pd.api.types.is_numeric_dtype(self.df[c])
            and not any(x in c.lower() for x in ["_normal", "_is_anomaly", "_anomaly_score", "_diff"])
            and c.lower() not in exclude
            and c != self.total_col
        ]

    # =========================================================================
    # 1. DAILY REPORT
    # =========================================================================
    def generate_daily_report(self, target_date: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        """
        Generate Daily Electricity Usage Report:
        - Daily electricity consumption (kWh)
        - Appliance-wise usage
        - Daily anomaly detection events
        - Hourly trend dataframe
        """
        if target_date is None:
            target_date = self.df[self.ts_col].max().normalize()
        else:
            target_date = pd.to_datetime(target_date).normalize()

        day_start = target_date
        day_end = target_date + pd.Timedelta(days=1)

        mask = (self.df[self.ts_col] >= day_start) & (self.df[self.ts_col] < day_end)
        day_df = self.df[mask].copy().sort_values(by=self.ts_col)

        if day_df.empty:
            return {"error": f"No electricity reading records found for date {target_date.strftime('%Y-%m-%d')}."}

        total_daily_kwh = float(day_df[self.total_col].sum())

        # Appliance-wise breakdown
        app_breakdown = {}
        for app in self.appliance_cols:
            val = float(day_df[app].sum())
            pct = (val / total_daily_kwh * 100) if total_daily_kwh > 0 else 0.0
            app_breakdown[app] = {
                "appliance": app.replace("_kWh", "").replace("_", " ").title(),
                "consumption_kwh": round(val, 3),
                "share_pct": round(pct, 1),
                "peak_hourly_kwh": round(float(day_df[app].max()), 3)
            }

        app_summary_df = pd.DataFrame(list(app_breakdown.values()))
        app_summary_df = app_summary_df.sort_values(by="consumption_kwh", ascending=False).reset_index(drop=True)

        # Daily anomaly events
        if not self.anomalies_df.empty:
            anom_mask = (self.anomalies_df["timestamp"] >= day_start) & (self.anomalies_df["timestamp"] < day_end)
            day_anomalies = self.anomalies_df[anom_mask].copy().reset_index(drop=True)
        else:
            day_anomalies = pd.DataFrame()

        # Hourly trend for export
        hourly_export = day_df[[self.ts_col, self.total_col] + self.appliance_cols].copy()
        hourly_export = hourly_export.rename(columns={self.ts_col: "Timestamp", self.total_col: "Total_House_kWh"})

        return {
            "report_type": "Daily Electricity Usage Report",
            "date": target_date.strftime("%Y-%m-%d"),
            "total_consumption_kwh": round(total_daily_kwh, 2),
            "readings_count": len(day_df),
            "anomalies_count": len(day_anomalies),
            "appliance_summary_df": app_summary_df,
            "anomalies_df": day_anomalies,
            "hourly_trend_df": hourly_export
        }

    # =========================================================================
    # 2. WEEKLY REPORT
    # =========================================================================
    def generate_weekly_report(self, end_date: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        """
        Generate Weekly Electricity Usage Report:
        - Last seven days' total consumption (kWh)
        - Day-by-day consumption and identification of highest-consuming day
        - Weekly anomaly history
        """
        if end_date is None:
            end_date = self.df[self.ts_col].max().normalize() + pd.Timedelta(days=1)
        else:
            end_date = pd.to_datetime(end_date).normalize() + pd.Timedelta(days=1)

        start_date = end_date - pd.Timedelta(days=7)

        mask = (self.df[self.ts_col] >= start_date) & (self.df[self.ts_col] < end_date)
        week_df = self.df[mask].copy().sort_values(by=self.ts_col)

        if week_df.empty:
            return {"error": f"No electricity reading records found for week ending {end_date.strftime('%Y-%m-%d')}."}

        total_weekly_kwh = float(week_df[self.total_col].sum())

        # Daily breakdown for the 7 days
        week_df["Date"] = week_df[self.ts_col].dt.date
        daily_group = week_df.groupby("Date")[self.total_col].sum().reset_index()
        daily_group = daily_group.rename(columns={self.total_col: "Total_kWh"})
        daily_group["Day_Name"] = pd.to_datetime(daily_group["Date"]).dt.day_name()
        daily_group["Total_kWh"] = daily_group["Total_kWh"].round(2)

        # Identify highest consuming day
        highest_day_row = daily_group.loc[daily_group["Total_kWh"].idxmax()]
        highest_day_date = str(highest_day_row["Date"])
        highest_day_name = str(highest_day_row["Day_Name"])
        highest_day_kwh = float(highest_day_row["Total_kWh"])

        # Weekly appliance totals
        app_totals = {}
        for app in self.appliance_cols:
            kwh = float(week_df[app].sum())
            app_totals[app] = {
                "Appliance": app.replace("_kWh", "").replace("_", " ").title(),
                "Total_Weekly_kWh": round(kwh, 2),
                "Share_Pct": round(kwh / total_weekly_kwh * 100, 1) if total_weekly_kwh > 0 else 0.0,
                "Daily_Avg_kWh": round(kwh / 7.0, 2)
            }
        weekly_app_df = pd.DataFrame(list(app_totals.values())).sort_values(by="Total_Weekly_kWh", ascending=False).reset_index(drop=True)

        # Weekly anomaly history
        if not self.anomalies_df.empty:
            anom_mask = (self.anomalies_df["timestamp"] >= start_date) & (self.anomalies_df["timestamp"] < end_date)
            week_anomalies = self.anomalies_df[anom_mask].copy().reset_index(drop=True)
        else:
            week_anomalies = pd.DataFrame()

        return {
            "report_type": "Weekly Electricity Usage Report",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": (end_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "total_consumption_kwh": round(total_weekly_kwh, 2),
            "highest_consuming_day": {
                "date": highest_day_date,
                "day_name": highest_day_name,
                "consumption_kwh": highest_day_kwh
            },
            "daily_breakdown_df": daily_group,
            "weekly_appliance_df": weekly_app_df,
            "weekly_anomalies_count": len(week_anomalies),
            "weekly_anomalies_df": week_anomalies
        }

    # =========================================================================
    # 3. MONTHLY REPORT
    # =========================================================================
    def generate_monthly_report(self, target_year_month: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate Monthly Electricity Usage Report:
        - Total monthly consumption (kWh)
        - Appliance-wise comparison & share %
        - Daily trend across the month
        - Monthly anomaly summary
        """
        self.df["YearMonth"] = self.df[self.ts_col].dt.to_period("M").astype(str)

        if target_year_month is None:
            target_year_month = str(self.df["YearMonth"].max())

        month_df = self.df[self.df["YearMonth"] == target_year_month].copy().sort_values(by=self.ts_col)

        if month_df.empty:
            return {"error": f"No electricity reading records found for month {target_year_month}."}

        total_monthly_kwh = float(month_df[self.total_col].sum())
        total_days = month_df[self.ts_col].dt.date.nunique()

        # Appliance-wise comparison
        app_comparison = []
        for app in self.appliance_cols:
            kwh = float(month_df[app].sum())
            share = (kwh / total_monthly_kwh * 100) if total_monthly_kwh > 0 else 0.0
            app_comparison.append({
                "Appliance": app.replace("_kWh", "").replace("_", " ").title(),
                "Total_Monthly_kWh": round(kwh, 2),
                "Share_Pct": round(share, 1),
                "Daily_Average_kWh": round(kwh / max(total_days, 1), 2),
                "Max_Single_Hour_kWh": round(float(month_df[app].max()), 3)
            })
        monthly_app_df = pd.DataFrame(app_comparison).sort_values(by="Total_Monthly_kWh", ascending=False).reset_index(drop=True)

        # Daily profile
        month_df["Date"] = month_df[self.ts_col].dt.date
        daily_trend = month_df.groupby("Date")[[self.total_col] + self.appliance_cols].sum().reset_index()
        daily_trend = daily_trend.rename(columns={self.total_col: "Total_kWh"})

        # Monthly anomaly history
        if not self.anomalies_df.empty:
            anom_copy = self.anomalies_df.copy()
            anom_copy["YearMonth"] = anom_copy["timestamp"].dt.to_period("M").astype(str)
            month_anomalies = anom_copy[anom_copy["YearMonth"] == target_year_month].copy().reset_index(drop=True)
        else:
            month_anomalies = pd.DataFrame()

        return {
            "report_type": "Monthly Electricity Usage Report",
            "month": target_year_month,
            "total_consumption_kwh": round(total_monthly_kwh, 2),
            "days_monitored": total_days,
            "daily_average_kwh": round(total_monthly_kwh / max(total_days, 1), 2),
            "appliance_comparison_df": monthly_app_df,
            "daily_trend_df": daily_trend,
            "monthly_anomalies_count": len(month_anomalies),
            "monthly_anomalies_df": month_anomalies
        }
