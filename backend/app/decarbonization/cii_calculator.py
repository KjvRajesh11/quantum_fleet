"""
IMO Carbon Intensity Indicator (CII) Rating Engine.
Implements IMO MEPC.336(76), MEPC.337(76), MEPC.338(76), MEPC.352(78), and MEPC.353(78).
Calculates:
- Reference CII line: CII_ref = a * Capacity^(-c)
- Required CII: CII_req = CII_ref * (1 - Z / 100)
- Attained Annual Efficiency Ratio (AER): CII_attained = sum(Fuel_i * Cf_i) / (Distance * Capacity)
- Operational Rating: Grades A, B, C, D, E based on IMO boundary vectors d1..d4
- Flags non-compliant vessels (D and E) for mandatory speed/route re-optimization
- Transition optimization delta: exactly how many tons of CO2 or speed reduction needed to attain Grade A/B/C.
"""

from typing import Dict, Any, Optional
from enum import Enum
from app.core.config import settings

class CIIGrade(str, Enum):
    A = "A" # Major superior
    B = "B" # Minor superior
    C = "C" # Moderate (Compliant)
    D = "D" # Minor inferior (Action plan required)
    E = "E" # Inferior (Immediate re-optimization mandated)

# IMO MEPC Reference Line Parameters: CII_ref = a * (Capacity)^(-c)
# Capacity is DWT for Bulk, Tanker, Container; Gross Tonnage (GT) for LNG Carrier
IMO_REFERENCE_LINE_COEFFICIENTS = {
    "Bulk Carrier": {"a": 4745.0, "c": 0.400, "capacity_type": "DWT"},
    "Container Ship": {"a": 1984.0, "c": 0.489, "capacity_type": "DWT"},
    "Tanker": {"a": 5247.0, "c": 0.461, "capacity_type": "DWT"},
    "LNG Carrier": {"a": 9.827e7, "c": 1.314, "capacity_type": "GT"},
    "General Cargo": {"a": 31948.0, "c": 0.742, "capacity_type": "DWT"},
    "Gas Carrier": {"a": 14405.0, "c": 0.783, "capacity_type": "DWT"}
}

# Boundary vectors (d1, d2, d3, d4) per IMO MEPC.353(78)
IMO_BOUNDARY_VECTORS = {
    "Bulk Carrier": {"d1": 0.86, "d2": 0.94, "d3": 1.06, "d4": 1.18},
    "Container Ship": {"d1": 0.83, "d2": 0.93, "d3": 1.07, "d4": 1.19},
    "Tanker": {"d1": 0.82, "d2": 0.93, "d3": 1.08, "d4": 1.20},
    "LNG Carrier": {"d1": 0.89, "d2": 0.95, "d3": 1.06, "d4": 1.13},
    "General Cargo": {"d1": 0.83, "d2": 0.94, "d3": 1.06, "d4": 1.19},
    "Gas Carrier": {"d1": 0.85, "d2": 0.95, "d3": 1.06, "d4": 1.18}
}

# IMO Carbon conversion factor Cf (t-CO2 / t-fuel) MEPC.308(73)
IMO_CARBON_CONVERSION_CF = {
    "VLSFO": 3.114,
    "LSMGO": 3.206,
    "HFO": 3.114,
    "LNG": 2.750,
    "Green Methanol": 0.05, # Net-zero biogenic lifecycle credit
    "Green Ammonia": 0.00,
    "Green Hydrogen": 0.00
}

# Official IMO Annual Reduction Rates Z (%) relative to 2019 baseline
IMO_REDUCTION_FACTORS = {
    2023: 5.0,
    2024: 7.0,
    2025: 9.0,
    2026: 11.0, # SIH2026 current benchmark
    2027: 13.0,
    2028: 15.0,
    2029: 17.5,
    2030: 20.0
}

def calculate_reference_cii(vessel_type: str, capacity: float) -> float:
    """
    Computes baseline CII_ref = a * Capacity^(-c)
    """
    coef = IMO_REFERENCE_LINE_COEFFICIENTS.get(vessel_type, IMO_REFERENCE_LINE_COEFFICIENTS["Bulk Carrier"])
    a = coef["a"]
    c = coef["c"]
    safe_cap = max(capacity, 1000.0)
    return a * (safe_cap ** (-c))

def calculate_required_cii(vessel_type: str, capacity: float, year: int = 2026) -> float:
    """
    Computes Required CII for specified operational year:
    CII_req = CII_ref * (1 - Z / 100)
    """
    cii_ref = calculate_reference_cii(vessel_type, capacity)
    z = IMO_REDUCTION_FACTORS.get(year, 11.0)
    return cii_ref * (1.0 - z / 100.0)

def calculate_attained_cii(
    fuel_consumption_tons: float,
    distance_nm: float,
    capacity: float,
    fuel_type: str = "VLSFO"
) -> float:
    """
    Calculates attained AER in g CO2 / (capacity_unit * nm):
    CII_attained = (Fuel * Cf * 10^6) / (Distance * Capacity)
    """
    if distance_nm <= 0.1 or capacity <= 1.0:
        return 0.0
    
    cf = IMO_CARBON_CONVERSION_CF.get(fuel_type, 3.114)
    co2_grams = fuel_consumption_tons * cf * 1e6
    return co2_grams / (distance_nm * capacity)

def evaluate_cii_rating(
    vessel_type: str,
    capacity: float,
    attained_cii: float,
    year: int = 2026
) -> Dict[str, Any]:
    """
    Determines vessel CII grade (A to E) against official IMO boundary thresholds:
    - Superior: Grade A (Attained < d1 * Req)
    - Minor Superior: Grade B (d1 <= Attained < d2 * Req)
    - Moderate: Grade C (d2 <= Attained < d3 * Req)
    - Minor Inferior: Grade D (d3 <= Attained < d4 * Req) -> Requires action plan
    - Inferior: Grade E (Attained >= d4 * Req) -> Requires mandatory speed/route re-optimization
    """
    v_type = vessel_type if vessel_type in IMO_REFERENCE_LINE_COEFFICIENTS else "Bulk Carrier"
    cii_ref = calculate_reference_cii(v_type, capacity)
    cii_req = calculate_required_cii(v_type, capacity, year)
    
    bounds = IMO_BOUNDARY_VECTORS.get(v_type, IMO_BOUNDARY_VECTORS["Bulk Carrier"])
    exp_d1 = bounds["d1"] * cii_req
    exp_d2 = bounds["d2"] * cii_req
    exp_d3 = bounds["d3"] * cii_req
    exp_d4 = bounds["d4"] * cii_req

    if attained_cii <= 0.001:
        grade = CIIGrade.A
    elif attained_cii < exp_d1:
        grade = CIIGrade.A
    elif attained_cii < exp_d2:
        grade = CIIGrade.B
    elif attained_cii < exp_d3:
        grade = CIIGrade.C
    elif attained_cii < exp_d4:
        grade = CIIGrade.D
    else:
        grade = CIIGrade.E

    requires_reoptimization = grade in [CIIGrade.D, CIIGrade.E]

    # Ratio of attained to required
    cii_ratio = attained_cii / max(cii_req, 1e-4)

    # Reduction target needed to reach Grade C (compliant boundary) and Grade A (superior)
    target_c = exp_d3 * 0.99
    target_a = exp_d1 * 0.95
    
    reduction_pct_to_c = max(0.0, ((attained_cii - target_c) / attained_cii) * 100.0) if attained_cii > 0 else 0.0
    reduction_pct_to_a = max(0.0, ((attained_cii - target_a) / attained_cii) * 100.0) if attained_cii > 0 else 0.0

    return {
        "vessel_type": v_type,
        "capacity": float(capacity),
        "evaluation_year": year,
        "attained_cii": float(attained_cii),
        "required_cii": float(cii_req),
        "reference_cii": float(cii_ref),
        "cii_ratio": float(cii_ratio),
        "grade": grade.value,
        "requires_reoptimization": requires_reoptimization,
        "boundaries": {
            "A_upper": float(exp_d1),
            "B_upper": float(exp_d2),
            "C_upper": float(exp_d3),
            "D_upper": float(exp_d4)
        },
        "reduction_target_pct_to_grade_c": float(reduction_pct_to_c),
        "reduction_target_pct_to_grade_a": float(reduction_pct_to_a),
        "status_description": (
            "Fully Compliant (Grade A - Superior Efficiency)" if grade == CIIGrade.A else
            "Compliant (Grade B - High Efficiency)" if grade == CIIGrade.B else
            "Compliant Baseline (Grade C - Meets Minimum IMO Standard)" if grade == CIIGrade.C else
            "Warning: Non-Compliant (Grade D - Corrective Action Required)" if grade == CIIGrade.D else
            "CRITICAL: Non-Compliant (Grade E - Mandatory Speed Re-optimization & Route Correction)"
        )
    }
