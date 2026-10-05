"""
Multi-Objective Genetic Algorithm (GA / NSGA-II Baseline) for Fleet Route Optimization.
Benchmark comparison baseline against QPSO.
Features:
- Simulated Binary Crossover (SBX)
- Polynomial Mutation
- Fast Non-Dominated Sorting and Crowding Distance selection
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional
from app.optimization.pareto import (
    dominates,
    fast_non_dominated_sort,
    compute_crowding_distance,
    extract_pareto_front,
    calculate_hypervolume
)
from app.optimization.qpso_solver import QPSOSolver

class GASolver:
    """
    Standard Real-Coded Multi-Objective Genetic Algorithm (NSGA-II).
    """
    def __init__(
        self,
        population_size: int = 50,
        max_generations: int = 100,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.15,
        eta_c: float = 20.0,
        eta_m: float = 20.0,
        seed: Optional[int] = 42
    ):
        self.pop_size = population_size
        self.max_generations = max_generations
        self.p_c = crossover_prob
        self.p_m = mutation_prob
        self.eta_c = eta_c
        self.eta_m = eta_m
        if seed is not None:
            np.random.seed(seed)
        self._evaluator = QPSOSolver() # Reuse hydrodynamic physics evaluator

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

        # 1. Initialize population
        pop = np.random.uniform(lb, ub, size=(self.pop_size, dim))
        objs = np.zeros((self.pop_size, 3))
        for i in range(self.pop_size):
            objs[i] = self._evaluator._evaluate_solution(
                pop[i], vessel_config, route_legs, eta_deadline_hours, target_fuel, carbon_tax_usd_per_ton, 5000.0, 2500.0
            )

        history = []

        # 2. Main GA loop
        for gen in range(self.max_generations):
            # Tournament selection & offspring generation
            offspring = []
            for _ in range(self.pop_size // 2):
                p1_idx = np.random.randint(0, self.pop_size)
                p2_idx = np.random.randint(0, self.pop_size)
                parent1 = pop[p1_idx]
                parent2 = pop[p2_idx]

                # SBX Crossover
                if np.random.rand() < self.p_c:
                    u = np.random.rand(dim)
                    beta = np.where(u <= 0.5, (2.0 * u) ** (1.0 / (self.eta_c + 1.0)), (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta_c + 1.0)))
                    c1 = 0.5 * ((1.0 + beta) * parent1 + (1.0 - beta) * parent2)
                    c2 = 0.5 * ((1.0 - beta) * parent1 + (1.0 + beta) * parent2)
                else:
                    c1, c2 = np.copy(parent1), np.copy(parent2)

                # Polynomial Mutation
                for child in [c1, c2]:
                    if np.random.rand() < self.p_m:
                        r = np.random.rand(dim)
                        delta = np.where(r < 0.5, (2.0 * r) ** (1.0 / (self.eta_m + 1.0)) - 1.0, 1.0 - (2.0 * (1.0 - r)) ** (1.0 / (self.eta_m + 1.0)))
                        child += delta * (ub - lb) * 0.1
                    offspring.append(np.clip(child, lb, ub))

            offspring = np.array(offspring)[:self.pop_size]
            off_objs = np.zeros((len(offspring), 3))
            for i in range(len(offspring)):
                off_objs[i] = self._evaluator._evaluate_solution(
                    offspring[i], vessel_config, route_legs, eta_deadline_hours, target_fuel, carbon_tax_usd_per_ton, 5000.0, 2500.0
                )

            # (mu + lambda) environmental selection
            combined_pop = np.vstack([pop, offspring])
            combined_objs = np.vstack([objs, off_objs])

            fronts = fast_non_dominated_sort(combined_objs)
            new_pop_indices = []
            for front in fronts:
                if len(new_pop_indices) + len(front) <= self.pop_size:
                    new_pop_indices.extend(front)
                else:
                    needed = self.pop_size - len(new_pop_indices)
                    dist = compute_crowding_distance(combined_objs, front)
                    sorted_front = [front[idx] for idx in np.argsort(dist)[::-1]]
                    new_pop_indices.extend(sorted_front[:needed])
                    break

            pop = combined_pop[new_pop_indices]
            objs = combined_objs[new_pop_indices]

            history.append({
                "iteration": gen + 1,
                "best_cost_usd": float(round(np.min(objs[:, 0]), 2)),
                "best_emissions_tons": float(round(np.min(objs[:, 1]), 2)),
                "best_delay_hours": float(round(np.min(objs[:, 2]), 2))
            })

        runtime_sec = time.perf_counter() - start_time
        pareto_x, pareto_objs = extract_pareto_front(pop, objs, 50)
        ref_pt = np.max(pareto_objs, axis=0) * 1.25
        hv = calculate_hypervolume(pareto_objs, ref_pt)

        return {
            "algorithm": "Genetic Algorithm (NSGA-II)",
            "runtime_seconds": float(round(runtime_sec, 4)),
            "iterations_executed": self.max_generations,
            "population_size": self.pop_size,
            "pareto_archive_size": len(pareto_x),
            "hypervolume_indicator": float(round(hv, 4)),
            "convergence_history": history,
            "best_cost_usd": float(np.min(pareto_objs[:, 0])),
            "best_emissions_tons": float(np.min(pareto_objs[:, 1]))
        }
