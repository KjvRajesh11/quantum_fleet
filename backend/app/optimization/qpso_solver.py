"""
Vectorized Quantum-Behaved Particle Swarm Optimization (QPSO) Delta-Potential Solver.
Formulated for maritime fleet route speed profiling, fuel selection, and multi-objective decarbonization.
Minimizes:
  f1: Total OPEX Cost ($) = Fuel Bunker Cost + Carbon Pricing
  f2: Total Well-to-Wake Lifecycle GHG Emissions (tons CO2e)
  f3: Schedule Delay (hours beyond target ETA)
Enforces:
  - Hull power limits (P_shaft <= 100% MCR)
  - Speed envelope bounds [V_min, V_max]
  - Hard ETA windows with penalty functions
  - IMO CII boundary compliance (penalizing Grade D and E transitions)
"""

import math
import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from app.physics.hydrodynamics import compute_total_resistance
from app.physics.power_model import compute_required_shaft_power
from app.decarbonization.cii_calculator import evaluate_cii_rating, calculate_attained_cii
from app.decarbonization.wtw_lca import compute_wtw_emissions, LIFECYCLE_EMISSION_FACTORS
from app.optimization.pareto import (
    dominates,
    fast_non_dominated_sort,
    compute_crowding_distance,
    extract_pareto_front,
    calculate_hypervolume
)

class QPSOSolver:
    """
    Vectorized Delta-Potential Quantum-Behaved Particle Swarm Optimization (QPSO).
    Uses quantum mechanical wave equation to eliminate velocity vector parameters,
    providing global exploration without premature convergence.
    """
    def __init__(
        self,
        swarm_size: int = 50,
        max_iterations: int = 100,
        alpha_start: float = 1.0,
        alpha_end: float = 0.4,
        max_archive_size: int = 80,
        seed: Optional[int] = 42
    ):
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.max_archive_size = max_archive_size
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def optimize_route(
        self,
        vessel_config: Dict[str, Any],
        route_legs: List[Dict[str, Any]],
        eta_deadline_hours: float,
        target_fuel: str = "VLSFO",
        carbon_tax_usd_per_ton: float = 100.0,
        cii_penalty_weight: float = 5000.0,
        delay_penalty_usd_per_hour: float = 2500.0
    ) -> Dict[str, Any]:
        """
        Executes multi-objective QPSO optimization over route leg speeds:
        Decision vector x: [v_1, v_2, ..., v_L] in knots.
        """
        start_time = time.perf_counter()
        num_legs = len(route_legs)
        if num_legs == 0:
            raise ValueError("Route must contain at least one waypoint leg.")

        # Speed bounds
        v_min = float(vessel_config.get("speed_min_knots", 10.0))
        v_max = float(vessel_config.get("speed_max_knots", 22.0))
        dim = num_legs

        # Vectorized bounds
        lb = np.full(dim, v_min)
        ub = np.full(dim, v_max)

        # 1. Swarm Initialization: uniform sampling in [lb, ub]
        swarm_x = np.random.uniform(lb, ub, size=(self.swarm_size, dim))
        pbest_x = np.copy(swarm_x)
        
        # Evaluate initial swarm
        pbest_objs = np.zeros((self.swarm_size, 3))
        for i in range(self.swarm_size):
            pbest_objs[i] = self._evaluate_solution(
                speed_profile=swarm_x[i],
                vessel_config=vessel_config,
                route_legs=route_legs,
                eta_deadline_hours=eta_deadline_hours,
                target_fuel=target_fuel,
                carbon_tax_usd=carbon_tax_usd_per_ton,
                cii_penalty_weight=cii_penalty_weight,
                delay_penalty_usd=delay_penalty_usd_per_hour
            )

        # External Pareto Archive
        archive_x, archive_objs = extract_pareto_front(pbest_x, pbest_objs, self.max_archive_size)

        # Convergence tracking
        history = []

        # 2. Main QPSO Iterative Loop
        for it in range(self.max_iterations):
            # Adaptive contraction-expansion parameter alpha(t) decreasing linearly
            alpha = self.alpha_start - (self.alpha_start - self.alpha_end) * (it / max(1, self.max_iterations - 1))

            # Mean Best Position (mbest) across all personal bests: mbest = 1/N * sum(p_i)
            mbest = np.mean(pbest_x, axis=0)

            # Select global leader g_d from Pareto archive for each particle
            leader_indices = np.random.randint(0, len(archive_x), size=self.swarm_size)
            g_leaders = archive_x[leader_indices]

            # Vectorized Quantum Local Attractor: p_{i,d} = phi * p_{i,d} + (1 - phi) * g_d
            phi = np.random.uniform(0.0, 1.0, size=(self.swarm_size, dim))
            p_attractor = phi * pbest_x + (1.0 - phi) * g_leaders

            # Quantum Wave Spatial Update via Delta-Potential:
            # x_{i,d} = p_{i,d} +/- alpha * |mbest_d - x_{i,d}| * ln(1 / u)
            u = np.random.uniform(1e-7, 1.0, size=(self.swarm_size, dim))
            signs = np.where(np.random.uniform(0.0, 1.0, size=(self.swarm_size, dim)) > 0.5, 1.0, -1.0)
            delta = alpha * np.abs(mbest - swarm_x) * np.log(1.0 / u)

            swarm_x = p_attractor + signs * delta

            # Clamp positions to feasible speed bounds
            swarm_x = np.clip(swarm_x, lb, ub)

            # Evaluate new swarm positions
            current_objs = np.zeros((self.swarm_size, 3))
            for i in range(self.swarm_size):
                current_objs[i] = self._evaluate_solution(
                    speed_profile=swarm_x[i],
                    vessel_config=vessel_config,
                    route_legs=route_legs,
                    eta_deadline_hours=eta_deadline_hours,
                    target_fuel=target_fuel,
                    carbon_tax_usd=carbon_tax_usd_per_ton,
                    cii_penalty_weight=cii_penalty_weight,
                    delay_penalty_usd=delay_penalty_usd_per_hour
                )

                # Personal best update (Pareto dominance)
                if dominates(current_objs[i], pbest_objs[i]):
                    pbest_x[i] = swarm_x[i]
                    pbest_objs[i] = current_objs[i]
                elif not dominates(pbest_objs[i], current_objs[i]):
                    # Mutually non-dominating: replace with 50% probability to maintain diversity
                    if np.random.rand() > 0.5:
                        pbest_x[i] = swarm_x[i]
                        pbest_objs[i] = current_objs[i]

            # Merge current personal bests with Pareto archive
            combined_x = np.vstack([archive_x, pbest_x])
            combined_objs = np.vstack([archive_objs, pbest_objs])
            archive_x, archive_objs = extract_pareto_front(combined_x, combined_objs, self.max_archive_size)

            # Record iteration history (min cost, min emissions, min delay, current alpha)
            min_cost = float(np.min(archive_objs[:, 0]))
            min_emiss = float(np.min(archive_objs[:, 1]))
            min_delay = float(np.min(archive_objs[:, 2]))
            history.append({
                "iteration": it + 1,
                "alpha": float(round(alpha, 4)),
                "best_cost_usd": float(round(min_cost, 2)),
                "best_emissions_tons": float(round(min_emiss, 2)),
                "best_delay_hours": float(round(min_delay, 2)),
                "archive_size": int(len(archive_x))
            })

        runtime_sec = time.perf_counter() - start_time

        # Calculate reference point for hypervolume: 1.2x worst observed objectives
        ref_pt = np.max(archive_objs, axis=0) * 1.25
        hv = calculate_hypervolume(archive_objs, ref_pt)

        # Select representative recommended compromise solution (minimum normalized distance to ideal point)
        norm_objs = (archive_objs - np.min(archive_objs, axis=0)) / np.maximum(np.ptp(archive_objs, axis=0), 1e-4)
        dist_to_utopia = np.linalg.norm(norm_objs, axis=1)
        best_idx = int(np.argmin(dist_to_utopia))

        recommended_speed_profile = archive_x[best_idx].tolist()
        recommended_metrics = self._detailed_solution_breakdown(
            speed_profile=archive_x[best_idx],
            vessel_config=vessel_config,
            route_legs=route_legs,
            eta_deadline_hours=eta_deadline_hours,
            target_fuel=target_fuel,
            carbon_tax_usd=carbon_tax_usd_per_ton
        )

        # Format Pareto solutions for 3D visualization
        pareto_front_points = []
        for idx in range(len(archive_x)):
            pareto_front_points.append({
                "cost_usd": float(round(archive_objs[idx, 0], 2)),
                "emissions_tons_co2e": float(round(archive_objs[idx, 1], 2)),
                "delay_hours": float(round(archive_objs[idx, 2], 2)),
                "speeds_knots": [round(float(s), 2) for s in archive_x[idx]]
            })

        return {
            "algorithm": "Quantum-Behaved Particle Swarm Optimization (QPSO)",
            "runtime_seconds": float(round(runtime_sec, 4)),
            "iterations_executed": self.max_iterations,
            "swarm_size": self.swarm_size,
            "pareto_archive_size": len(archive_x),
            "hypervolume_indicator": float(round(hv, 4)),
            "convergence_history": history,
            "recommended_compromise": {
                "speed_profile_knots": [round(s, 2) for s in recommended_speed_profile],
                "metrics": recommended_metrics
            },
            "pareto_front_3d": pareto_front_points
        }

    def _evaluate_solution(
        self,
        speed_profile: np.ndarray,
        vessel_config: Dict[str, Any],
        route_legs: List[Dict[str, Any]],
        eta_deadline_hours: float,
        target_fuel: str,
        carbon_tax_usd: float,
        cii_penalty_weight: float,
        delay_penalty_usd: float
    ) -> np.ndarray:
        """
        Evaluates objective function vector [f1, f2, f3]:
        f1 = Total Cost (fuel + carbon + delay + penalties)
        f2 = Total WtW GHG (tons CO2e)
        f3 = Arrival Delay (hours)
        """
        total_time_hours = 0.0
        total_fuel_tons = 0.0
        total_distance_nm = 0.0
        penalty = 0.0

        mcr_power = float(vessel_config.get("mcr_power_kw", 25000.0))

        for leg_idx, speed in enumerate(speed_profile):
            leg = route_legs[leg_idx]
            dist_nm = float(leg.get("distance_nm", 100.0))
            total_distance_nm += dist_nm

            # Time on leg = distance / speed
            time_hours = dist_nm / max(speed, 1.0)
            total_time_hours += time_hours

            # Hydrodynamics
            v_sog_ms = speed * 0.514444
            res = compute_total_resistance(
                v_stw=v_sog_ms,
                v_sog=v_sog_ms,
                l_wl=float(vessel_config.get("l_wl", 250.0)),
                beam=float(vessel_config.get("beam", 40.0)),
                draft=float(vessel_config.get("draft", 14.0)),
                c_b=float(vessel_config.get("c_b", 0.78)),
                displacement_m3=float(vessel_config.get("displacement_tons", 100000.0)) / 1.025,
                ship_heading_deg=float(leg.get("heading_deg", 90.0)),
                wind_speed_ms=float(leg.get("wind_speed_knots", 10.0)) * 0.514444,
                wind_direction_deg=float(leg.get("wind_dir_deg", 0.0)),
                transverse_area_m2=float(vessel_config.get("transverse_area", 1000.0)),
                lateral_area_m2=float(vessel_config.get("lateral_area", 4000.0)),
                hs_wave=float(leg.get("wave_hs_m", 1.0)),
                tp_wave=float(leg.get("wave_tp_sec", 7.0)),
                wave_angle_deg=float(leg.get("wave_dir_deg", 0.0))
            )

            # Power & Fuel
            pwr = compute_required_shaft_power(
                r_total_n=res["r_total_n"],
                v_stw=v_sog_ms,
                c_b=float(vessel_config.get("c_b", 0.78)),
                length_wl=float(vessel_config.get("l_wl", 250.0)),
                beam=float(vessel_config.get("beam", 40.0)),
                draft=float(vessel_config.get("draft", 14.0)),
                mcr_power_kw=mcr_power,
                fuel_type=target_fuel
            )

            leg_fuel_tons = (pwr["fuel_rate_tons_per_day"] / 24.0) * time_hours
            total_fuel_tons += leg_fuel_tons

            # Engine Overpower constraint penalty
            if pwr["p_shaft_kw"] > mcr_power:
                overpower_ratio = (pwr["p_shaft_kw"] - mcr_power) / mcr_power
                penalty += overpower_ratio * 15000.0

        # WtW Emissions & Economics
        wtw_res = compute_wtw_emissions(total_fuel_tons, target_fuel, carbon_tax_usd)
        fuel_and_tax_cost = wtw_res["total_opex_usd"]
        wtw_emissions_tons = wtw_res["wtw_total_tons_co2e"]

        # Schedule Delay
        delay_hours = max(0.0, total_time_hours - eta_deadline_hours)
        delay_cost = delay_hours * delay_penalty_usd

        # IMO CII Rating Gate & Penalty for Grade D/E
        dwt = float(vessel_config.get("dwt", 100000.0))
        v_type = vessel_config.get("vessel_type", "Bulk Carrier")
        attained_aer = calculate_attained_cii(total_fuel_tons, total_distance_nm, dwt, target_fuel)
        cii_eval = evaluate_cii_rating(v_type, dwt, attained_aer)

        if cii_eval["grade"] in ["D", "E"]:
            # Penalty proportional to excess over Grade C upper boundary
            excess_ratio = (attained_aer - cii_eval["boundaries"]["C_upper"]) / max(cii_eval["required_cii"], 1e-4)
            penalty += max(0.0, excess_ratio) * cii_penalty_weight

        # Objective vector:
        f1_cost = fuel_and_tax_cost + delay_cost + penalty
        f2_emissions = wtw_emissions_tons + (penalty / 500.0)
        f3_delay = delay_hours

        return np.array([f1_cost, f2_emissions, f3_delay])

    def _detailed_solution_breakdown(
        self,
        speed_profile: np.ndarray,
        vessel_config: Dict[str, Any],
        route_legs: List[Dict[str, Any]],
        eta_deadline_hours: float,
        target_fuel: str,
        carbon_tax_usd: float
    ) -> Dict[str, Any]:
        """Provides full operational breakdown of a candidate speed profile."""
        total_time_hours = 0.0
        total_fuel_tons = 0.0
        total_distance_nm = 0.0
        leg_details = []

        mcr = float(vessel_config.get("mcr_power_kw", 25000.0))

        for leg_idx, speed in enumerate(speed_profile):
            leg = route_legs[leg_idx]
            dist_nm = float(leg.get("distance_nm", 100.0))
            total_distance_nm += dist_nm
            leg_time = dist_nm / max(speed, 1.0)
            total_time_hours += leg_time

            v_sog_ms = speed * 0.514444
            res = compute_total_resistance(
                v_stw=v_sog_ms,
                v_sog=v_sog_ms,
                l_wl=float(vessel_config.get("l_wl", 250.0)),
                beam=float(vessel_config.get("beam", 40.0)),
                draft=float(vessel_config.get("draft", 14.0)),
                c_b=float(vessel_config.get("c_b", 0.78)),
                displacement_m3=float(vessel_config.get("displacement_tons", 100000.0)) / 1.025,
                ship_heading_deg=float(leg.get("heading_deg", 90.0)),
                wind_speed_ms=float(leg.get("wind_speed_knots", 10.0)) * 0.514444,
                wind_direction_deg=float(leg.get("wind_dir_deg", 0.0)),
                transverse_area_m2=float(vessel_config.get("transverse_area", 1000.0)),
                lateral_area_m2=float(vessel_config.get("lateral_area", 4000.0)),
                hs_wave=float(leg.get("wave_hs_m", 1.0)),
                tp_wave=float(leg.get("wave_tp_sec", 7.0)),
                wave_angle_deg=float(leg.get("wave_dir_deg", 0.0))
            )

            pwr = compute_required_shaft_power(
                r_total_n=res["r_total_n"],
                v_stw=v_sog_ms,
                c_b=float(vessel_config.get("c_b", 0.78)),
                length_wl=float(vessel_config.get("l_wl", 250.0)),
                beam=float(vessel_config.get("beam", 40.0)),
                draft=float(vessel_config.get("draft", 14.0)),
                mcr_power_kw=mcr,
                fuel_type=target_fuel
            )

            leg_fuel = (pwr["fuel_rate_tons_per_day"] / 24.0) * leg_time
            total_fuel_tons += leg_fuel

            leg_details.append({
                "leg_index": leg_idx + 1,
                "origin": leg.get("origin", f"WPT-{leg_idx}"),
                "destination": leg.get("destination", f"WPT-{leg_idx+1}"),
                "distance_nm": round(dist_nm, 1),
                "speed_knots": round(float(speed), 2),
                "duration_hours": round(leg_time, 2),
                "shaft_power_kw": round(pwr["p_shaft_kw"], 1),
                "engine_load_pct": round(pwr["engine_load_ratio"] * 100.0, 1),
                "fuel_consumed_tons": round(leg_fuel, 2),
                "sea_resistance_kn": round(res["r_total_kn"], 1)
            })

        wtw_res = compute_wtw_emissions(total_fuel_tons, target_fuel, carbon_tax_usd)
        dwt = float(vessel_config.get("dwt", 100000.0))
        v_type = vessel_config.get("vessel_type", "Bulk Carrier")
        attained_aer = calculate_attained_cii(total_fuel_tons, total_distance_nm, dwt, target_fuel)
        cii_eval = evaluate_cii_rating(v_type, dwt, attained_aer)

        delay_hours = max(0.0, total_time_hours - eta_deadline_hours)

        return {
            "total_distance_nm": round(total_distance_nm, 1),
            "voyage_duration_hours": round(total_time_hours, 2),
            "eta_deadline_hours": round(eta_deadline_hours, 2),
            "delay_hours": round(delay_hours, 2),
            "total_fuel_tons": round(total_fuel_tons, 2),
            "bunker_cost_usd": round(wtw_res["bunker_cost_usd"], 2),
            "carbon_tax_usd": round(wtw_res["carbon_tax_cost_usd"], 2),
            "total_opex_cost_usd": round(wtw_res["total_opex_usd"], 2),
            "wtw_emissions_tons_co2e": round(wtw_res["wtw_total_tons_co2e"], 2),
            "attained_cii": round(attained_aer, 3),
            "required_cii": round(cii_eval["required_cii"], 3),
            "cii_grade": cii_eval["grade"],
            "cii_status": cii_eval["status_description"],
            "legs": leg_details
        }
