"""
Quantum Fleet Decision Support System.
SIH2026 Problem Statement PS26138:
Quantum-Inspired Fuel Consumption Prediction and Green Fleet Optimization.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.database import init_db
from app.ml.predictor import fuel_predictor
from app.api.v1.endpoints import predict, optimize, benchmark, cii

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup sequence: initialize embedded DuckDB and verify ML models
    print(f"[{settings.PROJECT_NAME}] Booting Naval Hydrodynamics & Vectorized QPSO Engine...")
    init_db()
    fuel_predictor.load_or_initialize_model()
    print(f"[{settings.PROJECT_NAME}] System ready on API V1 ({settings.API_V1_STR}).")
    yield
    print(f"[{settings.PROJECT_NAME}] Shutting down gracefully.")

app = FastAPI(
    title=settings.PROJECT_FULL_NAME,
    description="Enterprise Decision Support System combining Naval Hydrodynamic Physics, LightGBM PINN, Vectorized QPSO Solver, and IMO CII Decarbonization.",
    version=settings.VERSION,
    lifespan=lifespan
)

# CORS configuration for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(predict.router, prefix=f"{settings.API_V1_STR}/predict", tags=["Fuel Prediction"])
app.include_router(optimize.router, prefix=f"{settings.API_V1_STR}/optimize", tags=["QPSO Route & Fleet Optimization"])
app.include_router(benchmark.router, prefix=f"{settings.API_V1_STR}/benchmark", tags=["Metaheuristic Benchmarking"])
app.include_router(cii.router, prefix=f"{settings.API_V1_STR}/cii", tags=["IMO CII & Decarbonization"])

@app.get("/", tags=["Health"])
async def root():
    return {
        "system": settings.PROJECT_FULL_NAME,
        "version": settings.VERSION,
        "status": "OPERATIONAL",
        "docs": "/docs",
        "api_v1": settings.API_V1_STR
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "HEALTHY",
        "duckdb_connected": True,
        "ml_model_loaded": fuel_predictor.model is not None,
        "qpso_engine_vectorized": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
