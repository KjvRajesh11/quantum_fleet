"""
Physics-Informed Feature Engineering Pipeline for Marine Telemetry & Fuel Prediction.
Combines raw AIS telemetry and NOAA oceanographic/meteorological inputs with
first-principles naval hydrodynamics to create rich physics-grounded feature vectors.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union
from app.physics.hydrodynamics import compute_total_resistance
from app.physics.power_model import compute_required_shaft_power

def calculate_speed_through_water(
    v_sog_knots: float,
    ship_heading_deg: float,
    current_speed_knots: float = 0.0,
    current_direction_deg: float = 0.0
) -> float:
    """
    Resolves ocean current vector against vessel course over ground to derive Speed Through Water (V_stw).
    V_stw_vector = V_sog_vector - V_current_vector
    """
    v_sog_ms = v_sog_knots * 0.514444
    v_curr_ms = current_speed_knots * 0.514444

    psi_ship = math.radians(ship_heading_deg)
    psi_curr = math.radians(current_direction_deg)

    # Ship SOG components
    u_ship = v_sog_ms * math.sin(psi_ship)
    v_ship = v_sog_ms * math.cos(psi_ship)

    # Current flow components
    u_curr = v_curr_ms * math.sin(psi_curr)
    v_curr = v_curr_ms * math.cos(psi_curr)

    # Relative speed through water components
    u_stw = u_ship - u_curr
    v_stw = v_ship - v_curr

    v_stw_ms = math.sqrt(u_stw ** 2 + v_stw ** 2)
    return float(v_stw_ms / 0.514444)

def extract_physics_features(
    v_sog_knots: float,
    ship_heading_deg: float,
    draft_m: float,
    trim_m: float,
    displacement_tons: float,
    l_wl: float,
    beam: float,
    c_b: float,
    transverse_area_m2: float,
    lateral_area_m2: float,
    mcr_power_kw: float,
    wind_speed_knots: float,
    wind_direction_deg: float,
    wave_hs_m: float,
    wave_tp_sec: float,
    wave_direction_deg: float,
    current_speed_knots: float = 0.0,
    current_direction_deg: float = 0.0,
    fuel_type: str = "VLSFO"
) -> Dict[str, float]:
    """
    Extracts complete physics-informed feature dictionary for ML inference or training.
    """
    # 1. Kinematic conversions
    v_stw_knots = calculate_speed_through_water(
        v_sog_knots=v_sog_knots,
        ship_heading_deg=ship_heading_deg,
        current_speed_knots=current_speed_knots,
        current_direction_deg=current_direction_deg
    )
    v_stw_ms = v_stw_knots * 0.514444
    v_sog_ms = v_sog_knots * 0.514444
    wind_speed_ms = wind_speed_knots * 0.514444

    # Relative wave angle (0 = head waves, 180 = following)
    rel_wave_deg = abs((wave_direction_deg - ship_heading_deg + 180.0) % 360.0 - 180.0)

    # Seawater volumetric displacement
    disp_m3 = displacement_tons / 1.025

    # 2. Naval Hydrodynamics
    res = compute_total_resistance(
        v_stw=v_stw_ms,
        v_sog=v_sog_ms,
        l_wl=l_wl,
        beam=beam,
        draft=draft_m,
        c_b=c_b,
        displacement_m3=disp_m3,
        ship_heading_deg=ship_heading_deg,
        wind_speed_ms=wind_speed_ms,
        wind_direction_deg=wind_direction_deg,
        transverse_area_m2=transverse_area_m2,
        lateral_area_m2=lateral_area_m2,
        hs_wave=wave_hs_m,
        tp_wave=wave_tp_sec,
        wave_angle_deg=rel_wave_deg
    )

    # 3. Propulsion Shaft Power
    pwr = compute_required_shaft_power(
        r_total_n=res["r_total_n"],
        v_stw=v_stw_ms,
        c_b=c_b,
        length_wl=l_wl,
        beam=beam,
        draft=draft_m,
        mcr_power_kw=mcr_power_kw,
        fuel_type=fuel_type
    )

    # 4. Assembled Features
    features = {
        # Telemetry & Operating Conditions
        "v_sog_knots": float(v_sog_knots),
        "v_stw_knots": float(v_stw_knots),
        "draft_m": float(draft_m),
        "trim_m": float(trim_m),
        "displacement_tons": float(displacement_tons),
        
        # Hull Geometry
        "l_wl": float(l_wl),
        "beam": float(beam),
        "c_b": float(c_b),
        "froude_number": float(res["fn"]),
        "wetted_surface_m2": float(res["wetted_surface_m2"]),

        # Weather & Sea State
        "wind_speed_knots": float(wind_speed_knots),
        "wind_rel_angle_deg": float(res["wind_breakdown"]["gamma_rel"]),
        "wave_hs_m": float(wave_hs_m),
        "wave_tp_sec": float(wave_tp_sec),
        "wave_rel_angle_deg": float(rel_wave_deg),
        "current_speed_knots": float(current_speed_knots),

        # Physics-Engine Outputs (PINN Prior Channels)
        "r_frictional_kn": float(res["r_frictional_n"] / 1000.0),
        "r_wave_kn": float(res["r_wave_n"] / 1000.0),
        "r_wind_kn": float(res["r_wind_n"] / 1000.0),
        "r_total_kn": float(res["r_total_kn"]),
        "p_effective_kw": float(pwr["p_effective_kw"]),
        "p_shaft_kw": float(pwr["p_shaft_kw"]),
        "engine_load_ratio": float(pwr["engine_load_ratio"]),
        "physics_baseline_fuel_tons_day": float(pwr["fuel_rate_tons_per_day"])
    }

    return features

FEATURE_COLUMNS = [
    "v_sog_knots",
    "v_stw_knots",
    "draft_m",
    "trim_m",
    "displacement_tons",
    "l_wl",
    "beam",
    "c_b",
    "froude_number",
    "wetted_surface_m2",
    "wind_speed_knots",
    "wind_rel_angle_deg",
    "wave_hs_m",
    "wave_tp_sec",
    "wave_rel_angle_deg",
    "current_speed_knots",
    "r_frictional_kn",
    "r_wave_kn",
    "r_wind_kn",
    "r_total_kn",
    "p_effective_kw",
    "p_shaft_kw",
    "engine_load_ratio",
    "physics_baseline_fuel_tons_day"
]
