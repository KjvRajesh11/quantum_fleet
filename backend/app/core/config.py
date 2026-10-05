from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "Quantum Fleet"
    PROJECT_FULL_NAME: str = "Quantum Fleet DSS"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Environment & Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODEL_DIR: Path = DATA_DIR / "models"
    DATABASE_PATH: Path = DATA_DIR / "quantum_fleet.duckdb"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"
    ]
    
    # Naval Hydrodynamic Constants (ITTC-1957 Standard Seawater)
    SEAWATER_DENSITY: float = 1025.0       # kg/m^3
    AIR_DENSITY: float = 1.225             # kg/m^3 at 15°C, 1 atm
    KINEMATIC_VISCOSITY: float = 1.188e-6  # m^2/s at 15°C
    GRAVITY: float = 9.80665               # m/s^2
    
    # Optimization Constants
    DEFAULT_SWARM_SIZE: int = 50
    DEFAULT_MAX_ITERATIONS: int = 100
    QPSO_ALPHA_START: float = 1.0
    QPSO_ALPHA_END: float = 0.4
    
    # IMO Carbon Intensity Target Year & Factor
    IMO_CII_YEAR: int = 2026
    IMO_REDUCTION_FACTOR_2026: float = 0.11 # 11% reduction vs baseline (MEPC.338(76))
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="allow"
    )

settings = Settings()
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
