"""
DuckDB Embedded High-Performance Analytical Database Integration for Quantum Fleet.
Manages fleet inventory, voyage records, optimization benchmarks, and CII operational logs.
"""

import duckdb
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.core.config import settings

def get_db_connection() -> duckdb.DuckDBPyConnection:
    """Returns thread-safe connection to DuckDB."""
    db_path = str(settings.DATABASE_PATH)
    conn = duckdb.connect(db_path)
    return conn

def init_db():
    """Initializes schema tables and seeds benchmark vessels if empty."""
    conn = get_db_connection()
    
    # 1. Vessels Table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS vessels (
        id VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL,
        imo_number VARCHAR UNIQUE NOT NULL,
        vessel_type VARCHAR NOT NULL,
        dwt DOUBLE NOT NULL,
        gross_tonnage DOUBLE,
        length_waterline_m DOUBLE NOT NULL,
        beam_m DOUBLE NOT NULL,
        design_draft_m DOUBLE NOT NULL,
        block_coefficient_cb DOUBLE NOT NULL,
        mcr_power_kw DOUBLE NOT NULL,
        speed_min_knots DOUBLE NOT NULL,
        speed_max_knots DOUBLE NOT NULL,
        transverse_area_m2 DOUBLE,
        lateral_area_m2 DOUBLE,
        primary_fuel VARCHAR NOT NULL,
        current_status VARCHAR DEFAULT 'Underway',
        current_lat DOUBLE DEFAULT 12.5,
        current_lon DOUBLE DEFAULT 68.2,
        current_speed_knots DOUBLE DEFAULT 14.2,
        current_cii_grade VARCHAR DEFAULT 'C'
    );
    """)

    # 2. Optimization History
    conn.execute("""
    CREATE TABLE IF NOT EXISTS optimization_runs (
        id VARCHAR PRIMARY KEY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        vessel_id VARCHAR,
        algorithm VARCHAR NOT NULL,
        runtime_seconds DOUBLE,
        total_distance_nm DOUBLE,
        total_fuel_tons DOUBLE,
        total_opex_usd DOUBLE,
        wtw_co2e_tons DOUBLE,
        delay_hours DOUBLE,
        cii_grade VARCHAR,
        hypervolume DOUBLE
    );
    """)

    # Seed Default Fleet if empty
    vessel_count = conn.execute("SELECT COUNT(*) FROM vessels;").fetchone()[0]
    if vessel_count == 0:
        conn.execute("""
        INSERT INTO vessels VALUES
        ('vessel-01', 'Quantum Fleet Pioneer', 'IMO9845123', 'Bulk Carrier', 180000.0, 93000.0, 285.0, 45.0, 18.0, 0.84, 18500.0, 9.0, 15.5, 850.0, 3800.0, 'VLSFO', 'Underway', 18.92, 72.83, 13.8, 'C'),
        ('vessel-02', 'Indus Galaxy', 'IMO9912044', 'Container Ship', 195000.0, 210000.0, 388.0, 59.0, 16.0, 0.65, 58000.0, 12.0, 22.5, 1600.0, 9200.0, 'LNG', 'Underway', 1.28, 103.85, 18.2, 'B'),
        ('vessel-03', 'Varuna Titan', 'IMO9765401', 'Tanker', 115000.0, 62000.0, 240.0, 42.0, 15.0, 0.82, 14200.0, 10.0, 15.0, 720.0, 3200.0, 'VLSFO', 'Underway', 25.10, 56.40, 12.5, 'D'),
        ('vessel-04', 'Dakshin Horizon', 'IMO9934812', 'LNG Carrier', 95000.0, 112000.0, 290.0, 45.8, 12.5, 0.74, 28000.0, 11.0, 19.5, 1100.0, 5100.0, 'Green Methanol', 'Underway', 22.30, 91.80, 16.0, 'A'),
        ('vessel-05', 'Ganga Express', 'IMO9681129', 'Container Ship', 140000.0, 150000.0, 366.0, 51.2, 15.2, 0.67, 45000.0, 11.0, 21.0, 1400.0, 7800.0, 'VLSFO', 'Underway', 31.23, 121.47, 17.5, 'E');
        """)
        print("[Database] Seeded 5 state-of-the-art merchant vessels into DuckDB.")

    conn.close()

def list_fleet() -> List[Dict[str, Any]]:
    """Retrieves all vessels in fleet."""
    conn = get_db_connection()
    df = conn.execute("SELECT * FROM vessels").fetchdf()
    conn.close()
    return df.to_dict(orient="records")

def get_vessel_by_id(vessel_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves specific vessel by ID."""
    conn = get_db_connection()
    df = conn.execute("SELECT * FROM vessels WHERE id = ?", [vessel_id]).fetchdf()
    conn.close()
    if len(df) == 0:
        return None
    return df.to_dict(orient="records")[0]
