"""
Unit tests for Naval Hydrodynamic Resistance, Propulsion Power, and LightGBM PINN Prediction.
"""

import pytest
import numpy as np
from app.physics.hydrodynamics import (
    compute_reynolds_number,
    compute_frictional_resistance,
    compute_wave_resistance,
    compute_wind_resistance,
    compute_total_resistance
)
from app.physics.power_model import (
    estimate_propulsive_efficiencies,
    compute_sfoc_g_kwh,
    compute_required_shaft_power
)
from app.ml.feature_engineering import calculate_speed_through_water, extract_physics_features
from app.ml.predictor import FuelPredictor

def test_reynolds_number():
    re = compute_reynolds_number(v_stw=7.72, l_wl=285.0) # ~15 knots
    assert re > 1e8, "Reynolds number for merchant vessel must be turbulent (> 10^8)"

def test_hydrodynamic_resistance_components():
    res = compute_total_resistance(
        v_stw=7.72, # m/s (~15 knots)
        v_sog=7.72,
        l_wl=285.0,
        beam=45.0,
        draft=18.0,
        c_b=0.84,
        displacement_m3=200000.0,
        ship_heading_deg=90.0,
        wind_speed_ms=10.0,
        wind_direction_deg=90.0,
        transverse_area_m2=850.0,
        lateral_area_m2=3800.0,
        hs_wave=1.5,
        tp_wave=8.0,
        wave_angle_deg=0.0
    )
    assert res["r_frictional_n"] > 0.0
    assert res["r_wave_n"] > 0.0
    assert res["r_wind_n"] >= 0.0
    assert res["r_total_n"] == res["r_frictional_n"] + res["r_wave_n"] + res["r_wind_n"]
    assert res["r_total_kn"] > 100.0, "Total resistance for 180k DWT at 15 kts should be hundreds of kN"

def test_speed_through_water_current_vector():
    # Head current of 2 knots should increase STW
    stw = calculate_speed_through_water(
        v_sog_knots=14.0,
        ship_heading_deg=90.0,
        current_speed_knots=2.0,
        current_direction_deg=270.0 # opposing
    )
    assert stw > 14.0, "Opposing ocean current must increase speed through water"

def test_sfoc_parabolic_curve():
    # Optimal load around 75% MCR should have lowest SFOC
    sfoc_opt = compute_sfoc_g_kwh(0.75)
    sfoc_low = compute_sfoc_g_kwh(0.25)
    sfoc_high = compute_sfoc_g_kwh(1.00)
    assert sfoc_opt < sfoc_low, "Low engine load must have higher SFOC than 75% MCR"
    assert sfoc_opt <= sfoc_high, "100% MCR load must have higher or equal SFOC than 75% MCR"

def test_fuel_predictor_monotonicity():
    predictor = FuelPredictor.get_instance()
    
    # Speed 12 knots vs 15 knots
    pred_12 = predictor.predict(
        v_sog_knots=12.0,
        ship_heading_deg=90.0,
        draft_m=16.0,
        trim_m=0.0,
        displacement_tons=170000.0,
        l_wl=285.0,
        beam=45.0,
        c_b=0.84,
        transverse_area_m2=850.0,
        lateral_area_m2=3800.0,
        mcr_power_kw=18500.0
    )
    pred_15 = predictor.predict(
        v_sog_knots=15.0,
        ship_heading_deg=90.0,
        draft_m=16.0,
        trim_m=0.0,
        displacement_tons=170000.0,
        l_wl=285.0,
        beam=45.0,
        c_b=0.84,
        transverse_area_m2=850.0,
        lateral_area_m2=3800.0,
        mcr_power_kw=18500.0
    )

    assert pred_15["predicted_fuel_tons_day"] > pred_12["predicted_fuel_tons_day"], \
        "Hydrodynamic fuel consumption must be strictly monotonic with respect to vessel speed"
