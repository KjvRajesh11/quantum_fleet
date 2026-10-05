"""
Pydantic Schemas for QPSO Fleet Optimization, Multi-Objective Pareto Fronts, and Benchmarking.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.schemas.vessel import VesselBase, FuelType
from app.schemas.voyage import RouteWaypoint

class QPSOParameters(BaseModel):
    swarm_size: int = Field(default=50, ge=10, le=200, example=50)
    max_iterations: int = Field(default=80, ge=10, le=500, example=80)
    alpha_start: float = Field(default=1.0, ge=0.5, le=1.5, example=1.0)
    alpha_end: float = Field(default=0.4, ge=0.1, le=0.8, example=0.4)
    max_archive_size: int = Field(default=60, ge=10, le=200)

class FleetOptimizeRequest(BaseModel):
    vessel: VesselBase
    waypoints: List[RouteWaypoint]
    eta_deadline_hours: float = Field(..., gt=1.0, example=120.0)
    target_fuel: FuelType = Field(default=FuelType.VLSFO)
    carbon_tax_usd_per_ton: float = Field(default=100.0, ge=0.0)
    solver_params: Optional[QPSOParameters] = Field(default_factory=QPSOParameters)

class ParetoPoint(BaseModel):
    cost_usd: float
    emissions_tons_co2e: float
    delay_hours: float
    speeds_knots: List[float]

class FleetOptimizeResponse(BaseModel):
    algorithm: str
    runtime_seconds: float
    iterations_executed: int
    swarm_size: int
    pareto_archive_size: int
    hypervolume_indicator: float
    convergence_history: List[Dict[str, Any]]
    recommended_compromise: Dict[str, Any]
    pareto_front_3d: List[ParetoPoint]

class BenchmarkRequest(BaseModel):
    vessel: Optional[VesselBase] = None
    waypoints: Optional[List[RouteWaypoint]] = None
    eta_deadline_hours: float = Field(default=96.0, gt=1.0)
    swarm_size: int = Field(default=40, ge=10, le=100)
    iterations: int = Field(default=50, ge=10, le=150)
    algorithms: List[str] = Field(default=["QPSO", "GA", "SA", "MILP"])

class BenchmarkAlgorithmResult(BaseModel):
    algorithm: str
    runtime_seconds: float
    hypervolume: float
    best_cost_usd: float
    best_emissions_tons: float
    iterations: int
    convergence_curve: List[Dict[str, Any]]

class BenchmarkResponse(BaseModel):
    benchmark_id: str
    test_conditions: Dict[str, Any]
    results: List[BenchmarkAlgorithmResult]
    fastest_algorithm: str
    highest_quality_algorithm: str
    qpso_speedup_factor_vs_ga: float
    summary: str
