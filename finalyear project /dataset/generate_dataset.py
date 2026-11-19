"""
Synthetic Electricity Load Dataset Generator
Simulates realistic household appliance-level electricity consumption compatible with
the REFIT Electrical Load Measurements benchmark format for academic demonstration.

NOTE: This is a synthetic dataset generated for educational and development purposes.
It simulates realistic appliance power patterns and controlled anomalies.
"""

import os
import numpy as np
import pandas as pd


def generate_synthetic_electricity_data(
    start_date: str = "2024-01-01 00:00:00",
    days: int = 90,
    output_path: str = "dataset/electricity_data.csv",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic appliance-level electricity consumption data.

    Appliances:
    - Air_Conditioner_kWh: High seasonal/diurnal power, runs mostly afternoons/evenings
    - Refrigerator_kWh: Continuous compressor cycling throughout the day
    - Washing_Machine_kWh: Sporadic operation during mornings and weekends
    - Fan_kWh: Low steady load during warm hours and night
    - Lights_kWh: Peaks in morning and evening
    - Television_kWh: Evening entertainment peaks
    - Total_Consumption_kWh: Aggregate household load including baseload
    """
    np.random.seed(seed)
    timestamps = pd.date_range(start=start_date, periods=days * 24, freq="h")
    n = len(timestamps)

    hours = timestamps.hour.values
    days_of_week = timestamps.dayofweek.values  # 0=Monday, 6=Sunday
    is_weekend = (days_of_week >= 5).astype(int)

    # 1. Air Conditioner (kWh per hour)
    # Peak during afternoon (12:00 - 18:00) and evening (19:00 - 22:00)
    ac_base = np.where(
        (hours >= 12) & (hours <= 18),
        2.2 + 0.6 * np.sin((hours - 12) / 6.0 * np.pi),
        np.where((hours >= 19) & (hours <= 23), 1.4, 0.05)
    )
    ac_noise = np.random.normal(0, 0.15, n)
    ac = np.clip(ac_base + ac_noise, 0.0, 4.0)

    # 2. Refrigerator (kWh per hour)
    # Cyclical duty cycle with slight peak during meal preparation hours (door openings)
    fridge_door_effect = np.where(np.isin(hours, [7, 8, 12, 13, 19, 20]), 0.06, 0.0)
    fridge_cycle = 0.12 + 0.05 * np.sin(np.arange(n) * 2 * np.pi / 4.0)  # 4-hour cycle
    fridge_noise = np.random.normal(0, 0.02, n)
    fridge = np.clip(fridge_cycle + fridge_door_effect + fridge_noise, 0.05, 0.35)

    # 3. Washing Machine (kWh per hour)
    # Sporadic runs: usually Saturday/Sunday morning (9:00 - 12:00) or Wednesday morning
    wm = np.zeros(n)
    for i in range(n):
        h = hours[i]
        dow = days_of_week[i]
        # Wash cycle probability
        if (dow in [5, 6] and 9 <= h <= 12) or (dow == 2 and 10 <= h <= 11):
            if np.random.rand() < 0.65:
                wm[i] = np.random.uniform(1.2, 2.2)
        elif np.random.rand() < 0.03:
            wm[i] = np.random.uniform(0.8, 1.6)

    # 4. Fan (kWh per hour)
    # Runs during night and warm afternoons
    fan_base = np.where(
        (hours >= 22) | (hours <= 6),
        0.065,
        np.where((hours >= 13) & (hours <= 17), 0.055, 0.02)
    )
    fan_noise = np.random.normal(0, 0.008, n)
    fan = np.clip(fan_base + fan_noise, 0.01, 0.09)

    # 5. Lights (kWh per hour)
    # Morning (6:00 - 8:00) and evening (18:00 - 23:00)
    lights_base = np.where(
        (hours >= 18) & (hours <= 23),
        0.28 + 0.08 * np.sin((hours - 18) / 5.0 * np.pi),
        np.where((hours >= 6) & (hours <= 8), 0.12, 0.015)
    )
    lights_noise = np.random.normal(0, 0.02, n)
    lights = np.clip(lights_base + lights_noise, 0.01, 0.45)

    # 6. Television (kWh per hour)
    # Evening entertainment (19:00 - 23:00) plus weekend afternoons
    tv_base = np.where(
        (hours >= 19) & (hours <= 23),
        0.18,
        np.where(is_weekend & (hours >= 14) & (hours <= 18), 0.15, 0.01)
    )
    tv_noise = np.random.normal(0, 0.02, n)
    tv = np.clip(tv_base + tv_noise, 0.0, 0.30)

    # --- INJECT REALISTIC ANOMALIES ---
    # Appliance 1: Air Conditioner compressor failure / stuck relay -> severe power surge (8.5 - 10.2 kWh)
    ac[24 * 12 + 15] = 9.85  # Day 13, 15:00
    ac[24 * 28 + 3] = 7.45   # Day 29, 03:00 AM (unseasonal overnight continuous high draw)
    ac[24 * 45 + 14] = 9.20  # Day 46, 14:00
    ac[24 * 67 + 16] = 8.90  # Day 68, 16:00
    ac[24 * 82 + 2] = 8.10   # Day 83, 02:00 AM

    # Appliance 2: Refrigerator door left open or refrigerant leak (continuous high draw 0.9 - 1.4 kWh)
    for h in range(6):
        fridge[24 * 20 + 13 + h] = 1.15 + np.random.uniform(0.0, 0.2)  # Day 21 afternoon leak
    # Refrigerator tripped breaker / motor dead (abnormally low ~ 0.0 kWh for 8 hours)
    for h in range(8):
        fridge[24 * 55 + 8 + h] = 0.001

    # Appliance 3: Washing Machine motor jam / continuous run overnight
    wm[24 * 33 + 2] = 3.95  # Day 34, 02:00 AM abnormal run
    wm[24 * 61 + 10] = 4.20  # Day 62, 10:00 AM mechanical surge

    # Appliance 4: Fan bearing friction overload
    fan[24 * 40 + 14] = 0.42  # Day 41, 14:00 (normally 0.05)
    fan[24 * 75 + 15] = 0.38  # Day 76, 15:00

    # Appliance 5: Lights left turned on during daytime
    for h in range(5):
        lights[24 * 50 + 10 + h] = 1.65  # Day 51 high wattage spotlights left on in day

    # Standby baseload (routers, clock, standby electronics)
    baseload = 0.05 + np.random.normal(0, 0.005, n)

    # Total consumption is the aggregate sum
    total_consumption = ac + fridge + wm + fan + lights + tv + baseload

    # Inject a few missing values (NaNs) to demonstrate data cleaning
    missing_indices = np.random.choice(n, size=15, replace=False)
    for idx in missing_indices[:5]:
        ac[idx] = np.nan
    for idx in missing_indices[5:10]:
        fridge[idx] = np.nan
    for idx in missing_indices[10:]:
        total_consumption[idx] = np.nan

    df = pd.DataFrame({
        "timestamp": timestamps,
        "Air_Conditioner_kWh": np.round(ac, 3),
        "Refrigerator_kWh": np.round(fridge, 3),
        "Washing_Machine_kWh": np.round(wm, 3),
        "Fan_kWh": np.round(fan, 3),
        "Lights_kWh": np.round(lights, 3),
        "Television_kWh": np.round(tv, 3),
        "Total_Consumption_kWh": np.round(total_consumption, 3),
        "is_simulated": True
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated synthetic electricity dataset with {len(df)} records at '{output_path}'.")
    return df


if __name__ == "__main__":
    generate_synthetic_electricity_data()
