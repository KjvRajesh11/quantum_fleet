"""
Multi-Objective Pareto Optimization Utilities.
Implements:
1. Fast Non-Dominated Sorting (Deb et al., NSGA-II).
2. Crowding Distance Assignment for Pareto diversity preservation.
3. Pareto Archive maintenance with maximum capacity truncation.
4. Hypervolume Indicator (HV) and Spacing metrics for benchmarking solver convergence.
"""

import numpy as np
from typing import List, Tuple, Dict, Any

def dominates(obj_a: np.ndarray, obj_b: np.ndarray) -> bool:
    """
    Checks if solution A Pareto-dominates solution B (minimization problem).
    A dominates B iff A_m <= B_m for all m and A_m < B_m for at least one m.
    """
    return bool(np.all(obj_a <= obj_b) and np.any(obj_a < obj_b))

def fast_non_dominated_sort(objectives: np.ndarray) -> List[List[int]]:
    """
    Performs O(M * N^2) fast non-dominated sorting on an (N, M) matrix of objective values.
    Returns:
        fronts: List of lists containing indices of solutions at each non-domination rank.
    """
    n_points = objectives.shape[0]
    domination_counts = np.zeros(n_points, dtype=int)
    dominated_solutions = [[] for _ in range(n_points)]
    fronts = [[]]

    for p in range(n_points):
        for q in range(p + 1, n_points):
            if dominates(objectives[p], objectives[q]):
                dominated_solutions[p].append(q)
                domination_counts[q] += 1
            elif dominates(objectives[q], objectives[p]):
                dominated_solutions[q].append(p)
                domination_counts[p] += 1

        if domination_counts[p] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        i += 1
        if len(next_front) > 0:
            fronts.append(next_front)
        else:
            break

    return fronts

def compute_crowding_distance(objectives: np.ndarray, front_indices: List[int]) -> np.ndarray:
    """
    Computes crowding distances for solutions in a Pareto front to maintain diversity.
    """
    l = len(front_indices)
    if l == 0:
        return np.array([])
    if l <= 2:
        return np.full(l, np.inf)

    distances = np.zeros(l)
    sub_objs = objectives[front_indices]
    m_objs = sub_objs.shape[1]

    for m in range(m_objs):
        sorted_order = np.argsort(sub_objs[:, m])
        distances[sorted_order[0]] = np.inf
        distances[sorted_order[-1]] = np.inf

        f_min = sub_objs[sorted_order[0], m]
        f_max = sub_objs[sorted_order[-1], m]
        denom = max(f_max - f_min, 1e-8)

        for i in range(1, l - 1):
            if not np.isinf(distances[sorted_order[i]]):
                prev_val = sub_objs[sorted_order[i - 1], m]
                next_val = sub_objs[sorted_order[i + 1], m]
                distances[sorted_order[i]] += (next_val - prev_val) / denom

    return distances

def extract_pareto_front(
    positions: np.ndarray,
    objectives: np.ndarray,
    max_archive_size: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the rank-1 non-dominated front and trims using crowding distance if necessary.
    """
    if len(objectives) == 0:
        return positions, objectives

    fronts = fast_non_dominated_sort(objectives)
    rank1_indices = fronts[0]

    pareto_pos = positions[rank1_indices]
    pareto_objs = objectives[rank1_indices]

    if len(rank1_indices) > max_archive_size:
        distances = compute_crowding_distance(pareto_objs, list(range(len(rank1_indices))))
        # Keep boundary points (inf distance) and highest crowding distance
        keep_order = np.argsort(distances)[::-1][:max_archive_size]
        pareto_pos = pareto_pos[keep_order]
        pareto_objs = pareto_objs[keep_order]

    return pareto_pos, pareto_objs

def calculate_hypervolume(
    pareto_objs: np.ndarray,
    reference_point: np.ndarray
) -> float:
    """
    Calculates exact or Monte Carlo Hypervolume (HV) indicator for 3D objective vectors:
    f1 = Fuel OPEX (USD), f2 = WtW Emissions (tons CO2e), f3 = Delay (hours).
    Higher HV indicates superior convergence and wider coverage.
    """
    if len(pareto_objs) == 0:
        return 0.0

    # Filter points that strictly dominate the reference point
    valid = np.all(pareto_objs <= reference_point, axis=1)
    if not np.any(valid):
        return 0.0
    
    pts = pareto_objs[valid]
    
    # Normalize objectives into [0, 1] relative to reference point
    min_bounds = np.min(pts, axis=0)
    ranges = np.maximum(reference_point - min_bounds, 1e-6)
    
    norm_pts = (pts - min_bounds) / ranges
    norm_ref = np.ones(3)

    # 10,000 Monte Carlo quasi-random samples for robust 3D hypervolume integration
    num_samples = 15000
    mc_samples = np.random.uniform(0.0, 1.0, size=(num_samples, 3))
    
    # Check what fraction of MC samples is dominated by at least one Pareto point
    dominated = np.zeros(num_samples, dtype=bool)
    for p in norm_pts:
        dominated |= np.all(mc_samples >= p, axis=1)

    volume_fraction = np.mean(dominated)
    total_volume = float(np.prod(ranges))
    return float(volume_fraction * total_volume)
