"""
FastAPI Route Speed Profiling and Fleet Decarbonization Optimization Endpoints.
Executes vectorized Quantum-Behaved Particle Swarm Optimization (QPSO).
"""

from fastapi import APIRouter, HTTPException, status
from typing import List, Dict, Any
from app.schemas.optimization import FleetOptimizeRequest, FleetOptimizeResponse
from app.optimization.qpso_solver import QPSOSolver
from app.core.database import list_fleet

router = APIRouter()

# Benchmark Maritime Corridors for Fleet Evaluation
PREDEFINED_MARITIME_ROUTES = {
    "rotterdam_singapore": {
        "name": "Rotterdam to Singapore via Suez Canal",
        "description": "High-traffic intercontinental container & tanker route through North Sea, Gibraltar, Mediterranean, Suez, Red Sea, and Malacca Strait.",
        "eta_baseline_hours": 360.0,
        "waypoints": [
            {"name": "Rotterdam Maasvlakte", "latitude": 51.95, "longitude": 4.02, "distance_to_next_nm": 320.0, "heading_deg": 225.0, "wind_speed_knots": 18.0, "wave_hs_m": 2.2, "wave_tp_sec": 8.0},
            {"name": "English Channel West", "latitude": 49.80, "longitude": -4.50, "distance_to_next_nm": 640.0, "heading_deg": 205.0, "wind_speed_knots": 15.0, "wave_hs_m": 2.8, "wave_tp_sec": 9.0},
            {"name": "Bay of Biscay & Finisterre", "latitude": 43.50, "longitude": -9.30, "distance_to_next_nm": 580.0, "heading_deg": 185.0, "wind_speed_knots": 22.0, "wave_hs_m": 3.4, "wave_tp_sec": 10.5},
            {"name": "Strait of Gibraltar", "latitude": 35.95, "longitude": -5.60, "distance_to_next_nm": 850.0, "heading_deg": 95.0, "wind_speed_knots": 12.0, "wave_hs_m": 1.4, "wave_tp_sec": 6.5},
            {"name": "Central Mediterranean (Malta Channel)", "latitude": 36.10, "longitude": 14.50, "distance_to_next_nm": 940.0, "heading_deg": 110.0, "wind_speed_knots": 10.0, "wave_hs_m": 1.1, "wave_tp_sec": 6.0},
            {"name": "Port Said (Suez North Entry)", "latitude": 31.26, "longitude": 32.30, "distance_to_next_nm": 105.0, "heading_deg": 160.0, "wind_speed_knots": 8.0, "wave_hs_m": 0.5, "wave_tp_sec": 4.0},
            {"name": "Red Sea Central", "latitude": 22.50, "longitude": 38.20, "distance_to_next_nm": 620.0, "heading_deg": 155.0, "wind_speed_knots": 14.0, "wave_hs_m": 1.6, "wave_tp_sec": 6.8},
            {"name": "Bab-el-Mandeb Strait", "latitude": 12.60, "longitude": 43.30, "distance_to_next_nm": 480.0, "heading_deg": 105.0, "wind_speed_knots": 16.0, "wave_hs_m": 1.8, "wave_tp_sec": 7.2},
            {"name": "Gulf of Aden & Arabian Sea Entry", "latitude": 12.20, "longitude": 51.50, "distance_to_next_nm": 1150.0, "heading_deg": 85.0, "wind_speed_knots": 12.0, "wave_hs_m": 1.5, "wave_tp_sec": 7.0},
            {"name": "Southern Tip of India (Cape Comorin)", "latitude": 6.00, "longitude": 79.50, "distance_to_next_nm": 980.0, "heading_deg": 98.0, "wind_speed_knots": 10.0, "wave_hs_m": 1.2, "wave_tp_sec": 6.5},
            {"name": "Malacca Strait Entry", "latitude": 5.60, "longitude": 95.30, "distance_to_next_nm": 490.0, "heading_deg": 125.0, "wind_speed_knots": 8.0, "wave_hs_m": 0.8, "wave_tp_sec": 5.0},
            {"name": "Port of Singapore (Tuas)", "latitude": 1.25, "longitude": 103.65, "distance_to_next_nm": 0.0, "heading_deg": 125.0, "wind_speed_knots": 6.0, "wave_hs_m": 0.4, "wave_tp_sec": 4.0}
        ]
    },
    "shanghai_los_angeles": {
        "name": "Shanghai to Los Angeles Transpacific Corridor",
        "description": "Great circle deep-sea container route across the North Pacific ocean.",
        "eta_baseline_hours": 280.0,
        "waypoints": [
            {"name": "Shanghai Yangshan Deepwater", "latitude": 30.62, "longitude": 122.06, "distance_to_next_nm": 420.0, "heading_deg": 75.0, "wind_speed_knots": 14.0, "wave_hs_m": 1.8, "wave_tp_sec": 7.0},
            {"name": "East China Sea / Tokara Strait", "latitude": 29.80, "longitude": 129.50, "distance_to_next_nm": 850.0, "heading_deg": 65.0, "wind_speed_knots": 16.0, "wave_hs_m": 2.2, "wave_tp_sec": 8.0},
            {"name": "Off Honshu / Kuroshio Current", "latitude": 34.50, "longitude": 144.00, "distance_to_next_nm": 1450.0, "heading_deg": 60.0, "wind_speed_knots": 20.0, "wave_hs_m": 3.2, "wave_tp_sec": 10.0},
            {"name": "North Pacific Apex (Aleutian South)", "latitude": 42.00, "longitude": 175.00, "distance_to_next_nm": 1600.0, "heading_deg": 95.0, "wind_speed_knots": 24.0, "wave_hs_m": 3.8, "wave_tp_sec": 11.5},
            {"name": "Sub-Arctic Approach", "latitude": 38.50, "longitude": -140.00, "distance_to_next_nm": 820.0, "heading_deg": 115.0, "wind_speed_knots": 15.0, "wave_hs_m": 2.1, "wave_tp_sec": 8.5},
            {"name": "Port of Los Angeles (San Pedro)", "latitude": 33.74, "longitude": -118.26, "distance_to_next_nm": 0.0, "heading_deg": 115.0, "wind_speed_knots": 10.0, "wave_hs_m": 1.1, "wave_tp_sec": 6.5}
        ]
    },
    "ras_tanura_mumbai": {
        "name": "Ras Tanura to Mumbai (JNPA) Crude Oil Shuttle",
        "description": "Strategic Arabian Gulf crude tanker route through Strait of Hormuz to India.",
        "eta_baseline_hours": 96.0,
        "waypoints": [
            {"name": "Ras Tanura Terminal", "latitude": 26.65, "longitude": 50.16, "distance_to_next_nm": 280.0, "heading_deg": 110.0, "wind_speed_knots": 12.0, "wave_hs_m": 0.9, "wave_tp_sec": 5.5},
            {"name": "Strait of Hormuz", "latitude": 26.35, "longitude": 56.40, "distance_to_next_nm": 190.0, "heading_deg": 140.0, "wind_speed_knots": 14.0, "wave_hs_m": 1.2, "wave_tp_sec": 6.0},
            {"name": "Gulf of Oman", "latitude": 24.50, "longitude": 58.80, "distance_to_next_nm": 480.0, "heading_deg": 115.0, "wind_speed_knots": 11.0, "wave_hs_m": 1.4, "wave_tp_sec": 6.8},
            {"name": "Central Arabian Sea", "latitude": 20.80, "longitude": 66.50, "distance_to_next_nm": 340.0, "heading_deg": 105.0, "wind_speed_knots": 10.0, "wave_hs_m": 1.3, "wave_tp_sec": 6.2},
            {"name": "Jawaharlal Nehru Port (Mumbai)", "latitude": 18.95, "longitude": 72.95, "distance_to_next_nm": 0.0, "heading_deg": 90.0, "wind_speed_knots": 8.0, "wave_hs_m": 0.7, "wave_tp_sec": 5.0}
        ]
    }
}

@router.post(
    "/fleet",
    response_model=FleetOptimizeResponse,
    summary="Optimize Route Speed Profile via Vectorized QPSO",
    description="Finds Pareto-optimal speed profiles minimizing Fuel OPEX, WtW Emissions, and Schedule Delay in <45s."
)
async def optimize_fleet_route(request: FleetOptimizeRequest):
    try:
        # Prepare vessel configuration dict
        vessel_cfg = {
            "vessel_type": request.vessel.vessel_type.value,
            "dwt": request.vessel.dwt,
            "l_wl": request.vessel.length_waterline_m,
            "beam": request.vessel.beam_m,
            "draft": request.vessel.design_draft_m,
            "c_b": request.vessel.block_coefficient_cb,
            "mcr_power_kw": request.vessel.mcr_power_kw,
            "speed_min_knots": request.vessel.speed_min_knots,
            "speed_max_knots": request.vessel.speed_max_knots,
            "transverse_area": request.vessel.transverse_area_m2,
            "lateral_area": request.vessel.lateral_area_m2,
            "displacement_tons": request.vessel.dwt * 1.15
        }

        # Route legs (pairs of waypoints)
        route_legs = []
        for i in range(len(request.waypoints) - 1):
            w_curr = request.waypoints[i]
            w_next = request.waypoints[i + 1]
            dist = w_curr.distance_to_next_nm if w_curr.distance_to_next_nm > 0 else 100.0
            route_legs.append({
                "origin": w_curr.name,
                "destination": w_next.name,
                "distance_nm": dist,
                "heading_deg": w_curr.heading_deg,
                "wind_speed_knots": w_curr.wind_speed_knots,
                "wind_dir_deg": w_curr.wind_dir_deg,
                "wave_hs_m": w_curr.wave_hs_m,
                "wave_tp_sec": w_curr.wave_tp_sec,
                "wave_dir_deg": w_curr.wave_dir_deg,
                "current_speed_knots": w_curr.current_speed_knots,
                "current_dir_deg": w_curr.current_dir_deg
            })

        if len(route_legs) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least 2 waypoints are required to form an optimization corridor."
            )

        solver_params = request.solver_params
        solver = QPSOSolver(
            swarm_size=solver_params.swarm_size,
            max_iterations=solver_params.max_iterations,
            alpha_start=solver_params.alpha_start,
            alpha_end=solver_params.alpha_end,
            max_archive_size=solver_params.max_archive_size
        )

        optimization_result = solver.optimize_route(
            vessel_config=vessel_cfg,
            route_legs=route_legs,
            eta_deadline_hours=request.eta_deadline_hours,
            target_fuel=request.target_fuel.value,
            carbon_tax_usd_per_ton=request.carbon_tax_usd_per_ton
        )

        return optimization_result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"QPSO optimization failed: {str(e)}"
        )

@router.get("/sample-routes", summary="Get Predefined Maritime Corridors")
async def get_sample_routes():
    """Returns curated international shipping routes for rapid test benchmarking."""
    return PREDEFINED_MARITIME_ROUTES

@router.get("/fleet", summary="List Monitored Fleet")
async def get_fleet():
    """Returns real-time fleet digital twins and positions from DuckDB."""
    return list_fleet()
