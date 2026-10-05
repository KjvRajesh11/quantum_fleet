"""
Pydantic Schemas for Voyages, Route Waypoints, and Fuel Prediction Telemetry.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.schemas.vessel import VesselType, FuelType

class RouteWaypoint(BaseModel):
    name: str = Field(default="Waypoint", example="Suez Canal South Entry")
    latitude: float = Field(..., ge=-90.0, le=90.0, example=27.8)
    longitude: float = Field(..., ge=-180.0, le=180.0, example=34.2)
    distance_to_next_nm: float = Field(..., gt=0.0, example=180.0)
    heading_deg: float = Field(default=90.0, ge=0.0, le=360.0)
    wind_speed_knots: float = Field(default=12.0, ge=0.0)
    wind_dir_deg: float = Field(default=45.0, ge=0.0, le=360.0)
    wave_hs_m: float = Field(default=1.2, ge=0.0)
    wave_tp_sec: float = Field(default=7.5, ge=1.0)
    wave_dir_deg: float = Field(default=50.0, ge=0.0, le=360.0)
    current_speed_knots: float = Field(default=0.4, ge=0.0)
    current_dir_deg: float = Field(default=180.0, ge=0.0, le=360.0)

class FuelPredictRequest(BaseModel):
    speed_sog_knots: float = Field(..., ge=1.0, le=35.0, example=14.5)
    ship_heading_deg: float = Field(default=90.0, ge=0.0, le=360.0)
    draft_m: float = Field(..., ge=2.0, le=25.0, example=15.2)
    trim_m: float = Field(default=0.0, ge=-5.0, le=5.0)
    displacement_tons: float = Field(..., gt=500.0, example=165000.0)
    vessel_class: VesselType = Field(default=VesselType.BULK_CARRIER)
    l_wl: float = Field(default=285.0, gt=10.0)
    beam: float = Field(default=45.0, gt=2.0)
    c_b: float = Field(default=0.84, ge=0.50, le=0.95)
    mcr_power_kw: float = Field(default=18500.0, gt=500.0)
    transverse_area_m2: float = Field(default=850.0)
    lateral_area_m2: float = Field(default=3800.0)
    wind_speed_knots: float = Field(default=14.0, ge=0.0)
    wind_direction_deg: float = Field(default=45.0, ge=0.0, le=360.0)
    wave_hs_m: float = Field(default=1.5, ge=0.0)
    wave_tp_sec: float = Field(default=8.0, ge=1.0)
    wave_direction_deg: float = Field(default=45.0, ge=0.0, le=360.0)
    current_speed_knots: float = Field(default=0.5, ge=0.0)
    current_direction_deg: float = Field(default=180.0, ge=0.0, le=360.0)
    fuel_type: FuelType = Field(default=FuelType.VLSFO)

class FuelPredictResponse(BaseModel):
    predicted_fuel_tons_day: float
    predicted_fuel_tons_hour: float
    physics_baseline_tons_day: float
    confidence_score: float
    effective_power_kw: float
    shaft_power_kw: float
    engine_load_ratio: float
    total_resistance_kn: float
    speed_through_water_knots: float
    features_snapshot: Dict[str, Any]
