"""
Alert and Energy Conservation Suggestions Engine
Module 4 of Smart Electricity Consumption Anomaly Detection and Appliance Identification System

Provides diagnostic explanations and tailored energy-saving recommendations
whenever an appliance exhibits abnormal consumption patterns.

CRITICAL COMPLIANCE NOTICE:
The suggestions provided are informational hypotheses based on behavioral pattern deviations.
The system does not claim to confirm physical electrical hardware faults.
"""

from typing import Dict, List, Any, Optional


# Appliance Diagnostic Knowledge Base
APPLIANCE_DIAGNOSTICS = {
    "air conditioner": {
        "high_reasons": [
            "Clogged air filters restricting airflow, forcing the compressor to run continuously.",
            "Refrigerant leak or low gas levels causing inefficient heat exchange and prolonged cooling cycles.",
            "Thermostat temperature set excessively low (e.g., below 20°C / 68°F).",
            "Doors or windows left open during operation, letting warm external air enter continuously.",
            "Compressor short-cycling or failing contactor relay failing to disengage.",
            "Outdoor condensing unit fins coated in dust, dirt, or debris preventing heat dissipation."
        ],
        "low_reasons": [
            "Tripped dedicated miniature circuit breaker (MCB) or blown fuse.",
            "Thermostat sensor disconnect or faulty ambient temperature sensor reading zero load.",
            "Compressor failed to engage due to a defective start capacitor.",
            "Loose power cord connection or timer scheduled off."
        ],
        "energy_saving_tips": [
            "Set the thermostat to the recommended energy-efficient setting of 24°C - 26°C (75°F - 78°F).",
            "Clean or wash reusable air filters every 2 to 4 weeks to maintain unrestricted airflow.",
            "Schedule annual HVAC maintenance and verify refrigerant pressure levels.",
            "Use ceiling fans in conjunction with the AC to circulate cooled air effectively.",
            "Install blackout curtains or thermal blinds on sun-facing windows to minimize solar heat gain."
        ]
    },
    "refrigerator": {
        "high_reasons": [
            "Damaged, cracked, or worn magnetic door gasket seals allowing ambient room air inside.",
            "Refrigerator or freezer door accidentally left slightly ajar or blocked by food items.",
            "Excessive frost buildup on the evaporator cooling coils obstructing heat transfer.",
            "Condenser coils coated in thick household dust, forcing the compressor to work twice as hard.",
            "Hot or warm food placed directly inside without allowing it to cool to room temperature first.",
            "Malfunctioning defrost timer or defrost heater remaining permanently engaged."
        ],
        "low_reasons": [
            "Power supply interruption or loose plug in the wall socket.",
            "Compressor motor thermal overload protector cut off.",
            "Faulty temperature control thermostat failing to trigger the compressor motor.",
            "Defective starter relay on the compressor."
        ],
        "energy_saving_tips": [
            "Test the door seal with the 'dollar bill test': close the door on a slip of paper; if it slips out easily, replace the gasket.",
            "Maintain refrigerator temperature at 3°C - 5°C (37°F - 40°F) and freezer at -18°C (0°F).",
            "Vacuum the condenser coils at the rear or bottom of the unit once every 6 months.",
            "Keep the refrigerator at least 70% full to act as thermal ballast, but do not block internal air vents.",
            "Allow warm food to reach room temperature before placing it inside."
        ]
    },
    "washing machine": {
        "high_reasons": [
            "Overloaded wash drum exceeding recommended capacity, stressing the electric drive motor.",
            "Water heating element running continuously if using high-temperature wash cycles (e.g., 60°C - 90°C).",
            "Stuck mechanical programmer or digital controller stuck on continuous rinse/drain loop.",
            "Drive motor bearing friction or foreign objects jammed in the drainage pump."
        ],
        "low_reasons": [
            "Interrupted wash cycle due to water inlet pressure failure.",
            "Machine paused or waiting for drainage timeout.",
            "Tripped residual current device (RCD) or power socket switch turned off."
        ],
        "energy_saving_tips": [
            "Wash full loads whenever practical to maximize water and electricity efficiency.",
            "Wash everyday garments with cold water (30°C or cold tap); up to 80% of wash electricity heats water.",
            "Select the 'Eco Mode' or high-efficiency short spin cycle for lightly soiled clothing.",
            "Clean the lint filter and check the drain pump filter monthly to avoid motor strain."
        ]
    },
    "fan": {
        "high_reasons": [
            "Dry or worn motor bearings causing severe mechanical friction and resistive heating.",
            "Bent or unbalanced blades causing aerodynamic drag and motor overwork.",
            "Faulty running capacitor causing excessive phase lag and higher current draw.",
            "Unit left running continuously at top speed for prolonged unmonitored periods."
        ],
        "low_reasons": [
            "Defective speed regulator / potentiometer with high internal contact resistance.",
            "Blown fuse or open circuit in wall regulator switch.",
            "Motor thermal cutoff activated due to overheating."
        ],
        "energy_saving_tips": [
            "Turn off ceiling and pedestal fans when leaving the room; fans cool people through evaporative breeze, not empty rooms.",
            "Clean dust accumulation from blades regularly to maintain optimal aerodynamic lift.",
            "Lubricate bearings periodically with lightweight machine oil if squeaking occurs.",
            "Consider upgrading older AC induction fans to energy-efficient BLDC (Brushless DC) motor fans which save up to 60% electricity."
        ]
    },
    "lights": {
        "high_reasons": [
            "High-wattage incandescent, halogen, or flood fixtures inadvertently left illuminated during bright daylight hours.",
            "Defective smart switch, timer, or photocell daylight sensor failing to disengage fixtures.",
            "Wiring short or partial leakage in older electrical conduit.",
            "Multiple fixtures left turned on in unoccupied sections of the property."
        ],
        "low_reasons": [
            "Burned-out bulb, LED driver failure, or loose fixture socket connection.",
            "Tripped localized lighting sub-circuit breaker."
        ],
        "energy_saving_tips": [
            "Transition legacy incandescent and halogen lamps to high-efficacy LED lighting (consuming up to 85% less power).",
            "Install passive infrared (PIR) occupancy motion sensors in corridors, bathrooms, and utility closets.",
            "Make use of natural ambient daylight by keeping curtains open during daytime hours.",
            "Utilize smart home scheduling to ensure outdoor security lighting switches off at sunrise."
        ]
    },
    "television": {
        "high_reasons": [
            "TV left running with display active when no one is present (e.g. overnight background noise).",
            "Screen brightness, backlight, and contrast set to maximum 'Vivid' or 'Dynamic Store Demo' mode.",
            "Connected gaming consoles, audio receivers, or streaming boxes failing to enter sleep state.",
            "Faulty internal switch-mode power supply (SMPS) drawing excessive current."
        ],
        "low_reasons": [
            "Standby deep-sleep power mode correctly engaged.",
            "Wall plug disconnected or master power strip toggled off."
        ],
        "energy_saving_tips": [
            "Enable 'Eco Mode' or ambient light sensor which automatically dims the backlight in dark rooms.",
            "Set an automatic sleep timer (e.g. 60-90 minutes) if prone to falling asleep with the TV on.",
            "Turn off master wall switches to avoid parasitic standby (vampire load) power draw.",
            "Turn off connected peripheral devices (amplifiers, consoles) when not actively in use."
        ]
    }
}

# Generic fallback for unlisted appliances in user-uploaded datasets
GENERIC_DIAGNOSTICS = {
    "high_reasons": [
        "Continuous unmonitored operation beyond typical scheduled duty cycle.",
        "Internal heating element, compressor, or high-draw drive motor failing to cycle off.",
        "Mechanical resistance, dry bearings, or clogged air/fluid passageways.",
        "Degraded electrical insulation or aging internal components operating with lower efficiency."
    ],
    "low_reasons": [
        "Loss of utility power, disconnected mains cord, or tripped miniature circuit breaker (MCB).",
        "Internal safety thermal cutoff triggered.",
        "Timer switch or standby mode active."
    ],
    "energy_saving_tips": [
        "Unplug or disconnect appliances from wall outlets when not in active use to curb phantom vampire draw.",
        "Inspect equipment regularly for abnormal heating, unusual humming sounds, or burning odors.",
        "Ensure appliances have adequate ventilation clearance around vents and heat sinks.",
        "Consult manufacturer maintenance schedules to ensure peak operating efficiency."
    ]
}


class AlertEngine:
    """
    Evaluates anomaly events, classifies operational severity, and generates
    safe diagnostic hypotheses and actionable electricity-saving recommendations.
    """

    @staticmethod
    def get_diagnostics_for_appliance(appliance_name: str, is_high_surge: bool = True) -> Dict[str, Any]:
        """
        Retrieve potential investigative reasons and energy-saving tips for an appliance.
        """
        app_clean = appliance_name.lower().replace("_", " ").replace("kwh", "").strip()

        # Fuzzy match to knowledge base
        matched_key = None
        for key in APPLIANCE_DIAGNOSTICS:
            if key in app_clean or app_clean in key:
                matched_key = key
                break

        if matched_key:
            data = APPLIANCE_DIAGNOSTICS[matched_key]
        else:
            data = GENERIC_DIAGNOSTICS

        reasons = data["high_reasons"] if is_high_surge else data["low_reasons"]
        tips = data["energy_saving_tips"]

        return {
            "appliance": appliance_name,
            "anomaly_nature": "Unusually High Consumption" if is_high_surge else "Unusually Low Consumption",
            "possible_reasons": reasons,
            "energy_saving_suggestions": tips,
            "disclaimer": (
                "⚠️ Diagnostic Disclaimer: The explanations listed above are potential operational "
                "hypotheses derived from consumption anomaly patterns. The system does not confirm "
                "physical electrical hardware faults. Always observe electrical safety precautions."
            )
        }

    @staticmethod
    def build_alert_card(anomaly_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a comprehensive, display-ready alert object for a detected anomaly event.
        """
        app_name = anomaly_record.get("appliance", "Appliance")
        diff_kwh = anomaly_record.get("difference_kwh", 0.0)
        actual_kwh = anomaly_record.get("actual_kwh", 0.0)
        normal_kwh = anomaly_record.get("normal_kwh", 0.0)
        timestamp = anomaly_record.get("timestamp", "Unknown time")
        severity = anomaly_record.get("severity", "Medium")

        is_high = diff_kwh >= 0
        diag = AlertEngine.get_diagnostics_for_appliance(app_name, is_high_surge=is_high)

        # Visual indicator styling
        if severity == "Critical":
            color = "#EF4444"  # Vibrant Red
            badge_text = "CRITICAL ANOMALY"
            icon = "🚨"
        elif severity == "High":
            color = "#F97316"  # Orange
            badge_text = "HIGH PRIORITY"
            icon = "⚠️"
        elif severity == "Medium":
            color = "#F59E0B"  # Amber
            badge_text = "MODERATE NOTICE"
            icon = "⚡"
        else:
            color = "#3B82F6"  # Blue
            badge_text = "INFORMATIONAL"
            icon = "ℹ️"

        headline = (
            f"{icon} Abnormal High Consumption Detected: {app_name}"
            if is_high else
            f"{icon} Abnormal Low Consumption Detected: {app_name}"
        )

        return {
            "headline": headline,
            "appliance": app_name,
            "timestamp": str(timestamp),
            "actual_kwh": actual_kwh,
            "normal_kwh": normal_kwh,
            "difference_kwh": diff_kwh,
            "severity": severity,
            "severity_color": color,
            "badge_text": badge_text,
            "is_high": is_high,
            "possible_reasons": diag["possible_reasons"],
            "saving_suggestions": diag["energy_saving_suggestions"],
            "disclaimer": diag["disclaimer"]
        }
