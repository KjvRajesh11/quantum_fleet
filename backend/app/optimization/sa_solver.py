"""
Multi-Objective Simulated Annealing (MOSA) Solver.
Benchmark baseline implementing Metropolis-Hastings acceptance probability
under an exponential cooling schedule.
"""

import time
import math
import numpy as np
from typing import Dict, Any, List, Optional
from app.optimization.pareto import (
    dominates,
    extract_pareto_front,
    calculate_hypervolume
)
from app.optimization.qpso_solver import QPSOSolver

class SASolver:
    """
    Multi-Objective Simulated Annealing with Archive Preservation.
    """
    def __init__(
        self,
        t_initial: float = 100.0,
        t_final: float = 0.05,
        cooling_rate: float = 0.95,
        iterations_per_temp: int = 5,
        seed: Optional[int] = 42
    ):
        self.t_initial = t_initial
        self.t_final = t_final
        self.cooling_rate = cooling_rate
        self.iters_per_t = iterations_per_temp
        if seed is not None:
            np.random.seed(seed)
        self._evaluator = QPSOSolver()

    def optimize_route(
        self,
        vessel_config: Dict[str, Any],
        route_legs: List[Dict[str, Any]],
        eta_deadline_hours: float,
        target_fuel: str = "VLSFO",
        carbon_tax_usd_per_ton: float = 100.0
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        dim = len(route_legs)
        v_min = float(vessel_config.get("speed_min_knots", 10.0))
        v_max = float(vessel_config.get("speed_max_knots", 22.0))

        lb = np.full(dim, v_min)
        ub = np.full(dim, v_max)

        # Initial solution
        current_x = np.random.uniform(lb, ub)
        current_obj = self._evaluator._evaluate_solution(
            current_x, vessel_config, route_legs, eta_deadline_hours, target_fuel, carbon_tax_usd_per_ton, 5000.0, 2500.0
        )

        archive_x = [np.copy(current_x)]
        archive_objs = [np.copy(current_obj)]

        temp = self.t_initial
        history = []
        step_count = 0

        while temp > self.t_final:
            for _ in range(self.iters_per_t):
                step_count += 1
                # Gaussian perturbation
                step_size = (ub - lb) * 0.15 * (temp / self.t_initial)
                neighbor_x = np.clip(current_x + np.random.normal(0, step_size), lb, ub)
                neighbor_obj = self._evaluator._evaluate_solution(
                    neighbor_x, vessel_config, route_legs, eta_deadline_hours, target_fuel, carbon_tax_usd_per_ton, 5000.0, 2500.0
                )

                # Acceptance criterion
                if dominates(neighbor_obj, current_obj):
                    current_x = np.copy(neighbor_x)
                    current_obj = np.copy(neighbor_obj)
                    archive_x.append(np.copy(neighbor_x))
                    archive_objs.append(np.copy(neighbor_obj))
                elif not dominates(current_obj, neighbor_obj):
                    # Non-dominating: accept
                    current_x = np.copy(neighbor_x)
                    current_obj = np.copy(neighbor_obj)
                    archive_x.append(np.copy(neighbor_x))
                    archive_objs.append(np.copy(neighbor_obj))
                else:
                    # Deterioration: Metropolis acceptance probability
                    delta_e = (neighbor_obj[0] - current_obj[0]) / max(current_obj[0], 1.0)
                    prob = math.exp(-max(0.0, delta_e) / max(temp / 10.0, 1e-4))
                    if np.random.rand() < prob:
                        current_x = np.copy(neighbor_x)
                        current_obj = np.copy(neighbor_obj)

            # Prune archive
            if len(archive_x) > 100:
                p_x, p_o = extract_pareto_front(np.array(archive_x), np.array(archive_objs), 50)
                archive_x = list(p_x)
                archive_objs = list(p_o)

            temp *= self.cooling_rate
            history.append({
                "iteration": step_count,
                "temperature": float(round(temp, 4)),
                "current_cost_usd": float(round(current_obj[0], 2)),
                "current_emissions_tons": float(round(current_obj[1], 2))
            })

        runtime_sec = time.perf_counter() - start_time
        pareto_x, pareto_objs = extract_pareto_front(np.array(archive_x), np.array(archive_objs), 50)
        ref_pt = np.max(pareto_objs, axis=0) * 1.25
        hv = calculate_hypervolume(pareto_objs, ref_pt)

        return {
            "algorithm": "Simulated Annealing (MOSA)",
            "runtime_seconds": float(round(runtime_sec, 4)),
            "iterations_executed": step_count,
            "pareto_archive_size": len(pareto_x),
            "hypervolume_indicator": float(round(hv, 4)),
            "convergence_history": history,
            "best_cost_usd": float(np.min(pareto_objs[:, 0])),
            "best_emissions_tons": float(np.min(pareto_objs[:, 1]))
        }
