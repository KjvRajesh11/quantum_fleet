"""
Pydantic Schemas for Vessel Specifications, Telemetry, and Digital Twin State.
"""

from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field

class VesselType(str, Enum):
    BULK_CARRIER = "Bulk Carrier"
    CONTAINER_SHIP = "Container Ship"
    TANKER = "Tanker"
    LNG_CARRIER = "LNG Carrier"
    GENERAL_CARGO = "General Cargo"

class FuelType(str, Enum):
    VLSFO = "VLSFO"
    LNG = "LNG"
    GREEN_METHANOL = "Green Methanol"
    GREEN_AMMONIA = "Green Ammonia"
    GREEN_HYDROGEN = "Green Hydrogen"

class VesselBase(BaseModel):
    name: str = Field(..., example="Quantum Fleet Pioneer")
    imo_number: str = Field(..., example="IMO9876543")
    vessel_type: VesselType = Field(default=VesselType.BULK_CARRIER)
    dwt: float = Field(..., gt=100.0, description="Deadweight Tonnage (DWT)", example=180000.0)
    gross_tonnage: Optional[float] = Field(default=95000.0, description="Gross Tonnage (GT)")
    length_waterline_m: float = Field(..., gt=10.0, example=285.0)
    beam_m: float = Field(..., gt=2.0, example=45.0)
    design_draft_m: float = Field(..., gt=1.0, example=18.0)
    block_coefficient_cb: float = Field(..., ge=0.50, le=0.92, example=0.84)
    mcr_power_kw: float = Field(..., gt=500.0, example=18500.0)
    speed_min_knots: float = Field(default=9.0, ge=5.0)
    speed_max_knots: float = Field(default=16.0, le=30.0)
    transverse_area_m2: float = Field(default=850.0)
    lateral_area_m2: float = Field(default=3800.0)
    primary_fuel: FuelType = Field(default=FuelType.VLSFO)

class VesselCreate(VesselBase):
    pass

class VesselState(BaseModel):
    vessel_id: str
    current_speed_sog_knots: float
    current_draft_m: float
    current_trim_m: float
    current_displacement_tons: float
    latitude: float
    longitude: float
    destination_port: str
    eta_hours_remaining: float
    fuel_type: FuelType = FuelType.VLSFO
    current_cii_grade: str = "C"

class VesselResponse(VesselBase):
    id: str
    current_state: Optional[VesselState] = None

    class Config:
        from_attributes = True
