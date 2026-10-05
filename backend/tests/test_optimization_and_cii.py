"""
Unit tests for QPSO Delta-Potential Solver, Non-Dominated Sorting, and IMO CII Decarbonization Boundaries.
"""

import pytest
import numpy as np
from app.optimization.pareto import dominates, fast_non_dominated_sort, compute_crowding_distance
from app.optimization.qpso_solver import QPSOSolver
from app.decarbonization.cii_calculator import (
    calculate_reference_cii,
    calculate_required_cii,
    calculate_attained_cii,
    evaluate_cii_rating,
    CIIGrade
)
from app.decarbonization.wtw_lca import compute_wtw_emissions, compare_multi_fuel_pathways

def test_pareto_dominance():
    # Minimization: [cost, emissions, delay]
    sol_a = np.array([100.0, 50.0, 2.0])
    sol_b = np.array([120.0, 60.0, 3.0])
    sol_c = np.array([90.0, 70.0, 1.0])

    assert dominates(sol_a, sol_b), "A should dominate B because all objectives are lower"
    assert not dominates(sol_b, sol_a), "B cannot dominate A"
    assert not dominates(sol_a, sol_c), "A and C are non-dominating trade-offs"

def test_fast_non_dominated_sort():
    objs = np.array([
        [10.0, 20.0],
        [15.0, 15.0],
        [20.0, 10.0],
        [25.0, 25.0] # dominated by all
    ])
    fronts = fast_non_dominated_sort(objs)
    assert set(fronts[0]) == {0, 1, 2}, "Rank 1 front must contain non-dominated points"
    assert fronts[1] == [3], "Rank 2 front must contain the dominated point"

def test_qpso_delta_potential_convergence():
    vessel = {
        "vessel_type": "Bulk Carrier",
        "dwt": 180000.0,
        "l_wl": 285.0,
        "beam": 45.0,
        "draft": 18.0,
        "c_b": 0.84,
        "mcr_power_kw": 18500.0,
        "speed_min_knots": 10.0,
        "speed_max_knots": 15.0,
        "transverse_area": 850.0,
        "lateral_area": 3800.0,
        "displacement_tons": 207000.0
    }
    legs = [
        {"distance_nm": 150.0, "heading_deg": 90.0, "wind_speed_knots": 12.0, "wave_hs_m": 1.2, "wave_tp_sec": 7.0},
        {"distance_nm": 180.0, "heading_deg": 100.0, "wind_speed_knots": 14.0, "wave_hs_m": 1.5, "wave_tp_sec": 7.5}
    ]

    solver = QPSOSolver(swarm_size=20, max_iterations=15, seed=42)
    res = solver.optimize_route(vessel, legs, eta_deadline_hours=30.0)

    assert "pareto_front_3d" in res
    assert len(res["pareto_front_3d"]) > 0
    assert res["runtime_seconds"] < 45.0, "QPSO execution must complete in <45s"
    assert res["recommended_compromise"]["metrics"]["total_distance_nm"] == 330.0

def test_imo_cii_grading_boundaries():
    dwt = 180000.0
    ref_cii = calculate_reference_cii("Bulk Carrier", dwt)
    req_cii = calculate_required_cii("Bulk Carrier", dwt, 2026)

    # 1. Attained far below required => Grade A
    eval_a = evaluate_cii_rating("Bulk Carrier", dwt, req_cii * 0.75, 2026)
    assert eval_a["grade"] == CIIGrade.A.value
    assert not eval_a["requires_reoptimization"]

    # 2. Attained slightly above required => Grade C or D
    eval_c = evaluate_cii_rating("Bulk Carrier", dwt, req_cii * 1.02, 2026)
    assert eval_c["grade"] == CIIGrade.C.value

    # 3. Attained far above required => Grade E
    eval_e = evaluate_cii_rating("Bulk Carrier", dwt, req_cii * 1.25, 2026)
    assert eval_e["grade"] == CIIGrade.E.value
    assert eval_e["requires_reoptimization"], "Grade E vessels must be flagged for mandatory re-optimization"

def test_wtw_lca_methane_slip_impact():
    # Comparing low-slip Diesel vs high-slip Otto LNG
    otto = compute_wtw_emissions(fuel_mass_tons=100.0, fuel_pathway="LNG_Otto")
    diesel = compute_wtw_emissions(fuel_mass_tons=100.0, fuel_pathway="LNG_Diesel")

    assert otto["ttw_methane_slip_tons_co2e"] > diesel["ttw_methane_slip_tons_co2e"], \
        "Otto-cycle LNG engine must have higher methane slip than high-pressure Diesel cycle"
    assert otto["wtw_total_tons_co2e"] > diesel["wtw_total_tons_co2e"]
