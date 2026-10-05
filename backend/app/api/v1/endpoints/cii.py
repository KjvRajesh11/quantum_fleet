"""
FastAPI Decarbonization and IMO CII Gate Endpoints.
Implements IMO MEPC.338(76) Carbon Intensity Indicator ratings and Well-to-Wake LCA lifecycle accounting.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, List
from app.decarbonization.cii_calculator import evaluate_cii_rating, calculate_attained_cii
from app.decarbonization.wtw_lca import compare_multi_fuel_pathways, compute_wtw_emissions
from app.core.database import list_fleet

router = APIRouter()

class CIICalculateRequest(BaseModel):
    vessel_type: str = Field(default="Bulk Carrier", example="Bulk Carrier")
    capacity_dwt: float = Field(..., gt=100.0, example=180000.0)
    fuel_consumption_tons: float = Field(..., gt=0.0, example=280.0)
    distance_travelled_nm: float = Field(..., gt=1.0, example=4500.0)
    fuel_type: str = Field(default="VLSFO", example="VLSFO")
    evaluation_year: int = Field(default=2026, ge=2023, le=2030)

class WTWCompareRequest(BaseModel):
    baseline_vlsfo_tons: float = Field(default=350.0, gt=0.0, example=350.0)
    carbon_tax_usd_per_ton: float = Field(default=100.0, ge=0.0, example=100.0)

@router.post(
    "/calculate",
    summary="Compute Official IMO CII Grade & Reduction Target",
    description="Evaluates attained CII AER against IMO MEPC reference lines and returns Grades A through E."
)
async def calculate_cii(request: CIICalculateRequest):
    try:
        attained = calculate_attained_cii(
            fuel_consumption_tons=request.fuel_consumption_tons,
            distance_nm=request.distance_travelled_nm,
            capacity=request.capacity_dwt,
            fuel_type=request.fuel_type
        )

        rating = evaluate_cii_rating(
            vessel_type=request.vessel_type,
            capacity=request.capacity_dwt,
            attained_cii=attained,
            year=request.evaluation_year
        )

        return rating
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CII calculation failed: {str(e)}"
        )

@router.post(
    "/wtw-compare",
    summary="Well-to-Wake Multi-Fuel Decarbonization LCA",
    description="Analyzes lifecycle GHG emissions, methane slip, and OPEX across VLSFO, LNG, Methanol, Ammonia, and H2."
)
async def compare_fuel_pathways(request: WTWCompareRequest):
    try:
        results = compare_multi_fuel_pathways(
            baseline_vlsfo_tons=request.baseline_vlsfo_tons,
            carbon_tax_usd_per_ton=request.carbon_tax_usd_per_ton
        )
        return {
            "baseline_vlsfo_tons": request.baseline_vlsfo_tons,
            "carbon_tax_usd_per_ton": request.carbon_tax_usd_per_ton,
            "pathways_comparison": results
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"WTW LCA comparison failed: {str(e)}"
        )

@router.get(
    "/fleet-overview",
    summary="Fleet-Wide CII Decarbonization Monitor",
    description="Returns CII distribution and compliance breakdown for all monitored vessels."
)
async def get_fleet_cii_overview():
    try:
        vessels = list_fleet()
        grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
        non_compliant_count = 0

        for v in vessels:
            g = v.get("current_cii_grade", "C")
            grade_counts[g] = grade_counts.get(g, 0) + 1
            if g in ["D", "E"]:
                non_compliant_count += 1

        compliance_rate_pct = ((len(vessels) - non_compliant_count) / max(len(vessels), 1)) * 100.0

        return {
            "total_vessels": len(vessels),
            "compliance_rate_pct": round(compliance_rate_pct, 1),
            "grade_distribution": grade_counts,
            "non_compliant_vessels_requiring_reoptimization": non_compliant_count,
            "vessels": vessels
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch fleet overview: {str(e)}"
        )
