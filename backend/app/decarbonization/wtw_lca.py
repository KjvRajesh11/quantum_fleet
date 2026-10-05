"""
Well-to-Wake (WtW) Lifecycle Greenhouse Gas (GHG) Accounting Engine.
Implements:
1. Well-to-Tank (WtT) upstream extraction, production, transport, bunkering GHG emissions.
2. Tank-to-Wake (TtW) direct vessel combustion and fugitive/slip emissions (CH4, N2O).
3. Well-to-Wake (WtW) total footprint in g CO2e / MJ and tons CO2e.
4. Methane slip analysis for LNG propulsion architectures (Otto-cycle vs Diesel ME-GI).
5. Comprehensive Multi-Fuel Decarbonization Pathway Matrix.
"""

from typing import Dict, Any, List

# Lifecycle Greenhouse Gas factors in g CO2e / MJ of fuel energy (EU RED II / FuelEU Maritime / IMO LCA Guidelines)
LIFECYCLE_EMISSION_FACTORS = {
    "VLSFO": {
        "fuel_name": "Very Low Sulphur Fuel Oil (0.5% S)",
        "lhv_mj_kg": 42.7,
        "wtt_g_co2e_mj": 13.5,       # Refining, transport, bunkering
        "ttw_combustion_g_co2e_mj": 73.5, # Direct combustion
        "slip_g_co2e_mj": 0.0,
        "base_cost_usd_ton": 680.0,
        "description": "Conventional fossil maritime fuel baseline."
    },
    "LNG_Otto": {
        "fuel_name": "Liquefied Natural Gas (Low-Pressure Otto Engine)",
        "lhv_mj_kg": 50.0,
        "wtt_g_co2e_mj": 18.5,       # Gas extraction, liquefaction, cryogenic sea transit
        "ttw_combustion_g_co2e_mj": 55.0, # Clean combustion CO2
        "slip_g_co2e_mj": 46.2,      # High methane slip (~3.1% unburnt CH4, GWP20=29.8)
        "base_cost_usd_ton": 760.0,
        "description": "Dual-fuel low pressure 4-stroke / 2-stroke Otto engine with significant methane slip."
    },
    "LNG_Diesel": {
        "fuel_name": "Liquefied Natural Gas (High-Pressure Diesel ME-GI)",
        "lhv_mj_kg": 50.0,
        "wtt_g_co2e_mj": 18.5,
        "ttw_combustion_g_co2e_mj": 55.0,
        "slip_g_co2e_mj": 3.0,       # Low methane slip (~0.2% slip with high-pressure direct injection)
        "base_cost_usd_ton": 810.0,
        "description": "High-pressure diesel cycle ME-GI engine minimizing methane slip."
    },
    "Green Methanol": {
        "fuel_name": "E-Methanol (Direct Air Capture + Green H2)",
        "lhv_mj_kg": 19.9,
        "wtt_g_co2e_mj": -58.0,      # Biogenic / DAC atmospheric carbon capture credit
        "ttw_combustion_g_co2e_mj": 69.0, # Released on combustion (net circular)
        "slip_g_co2e_mj": 0.0,
        "base_cost_usd_ton": 1150.0,
        "description": "Renewable synthetic e-methanol synthesized with DAC and renewable electricity."
    },
    "Green Ammonia": {
        "fuel_name": "Green Ammonia (Renewable Haber-Bosch)",
        "lhv_mj_kg": 18.6,
        "wtt_g_co2e_mj": 9.5,        # Electrolysis + Air Separation + Haber-Bosch synthesis
        "ttw_combustion_g_co2e_mj": 0.0,  # Zero-carbon molecule (NH3)
        "slip_g_co2e_mj": 1.8,       # Minor nitrous oxide (N2O) formation in SCR catalyst
        "base_cost_usd_ton": 980.0,
        "description": "Carbon-free fuel carrier requiring pressurized or refrigerated storage and De-NOx SCR."
    },
    "Green Hydrogen": {
        "fuel_name": "Liquid Green Hydrogen (Cryogenic LH2)",
        "lhv_mj_kg": 120.0,
        "wtt_g_co2e_mj": 6.0,        # PEM electrolysis + cryogenic liquefaction
        "ttw_combustion_g_co2e_mj": 0.0,
        "slip_g_co2e_mj": 0.0,
        "base_cost_usd_ton": 2600.0,
        "description": "Ultra-clean fuel cell / combustion with zero tailpipe carbon emissions."
    }
}

def compute_wtw_emissions(
    fuel_mass_tons: float,
    fuel_pathway: str = "VLSFO",
    carbon_tax_usd_per_ton: float = 100.0 # EU ETS / IMO Global GHG Pricing mechanism
) -> Dict[str, Any]:
    """
    Computes Well-to-Wake (WtW) lifecycle GHG accounting:
    - Fuel energy demand (GJ)
    - WtT emissions (tons CO2e)
    - TtW emissions (tons CO2e) including combustion and slip
    - Total WtW emissions (tons CO2e)
    - Net fuel cost + Carbon Tax OPEX (USD)
    """
    pathway = fuel_pathway if fuel_pathway in LIFECYCLE_EMISSION_FACTORS else "VLSFO"
    profile = LIFECYCLE_EMISSION_FACTORS[pathway]

    lhv = profile["lhv_mj_kg"]
    # Total chemical energy (MJ) = mass (kg) * LHV (MJ/kg)
    fuel_energy_mj = fuel_mass_tons * 1000.0 * lhv
    fuel_energy_gj = fuel_energy_mj / 1000.0

    # Emission totals in grams converted to metric tons (1 ton = 10^6 grams)
    wtt_tons_co2e = (fuel_energy_mj * profile["wtt_g_co2e_mj"]) / 1e6
    ttw_combustion_tons = (fuel_energy_mj * profile["ttw_combustion_g_co2e_mj"]) / 1e6
    slip_tons_co2e = (fuel_energy_mj * profile["slip_g_co2e_mj"]) / 1e6
    ttw_total_tons_co2e = ttw_combustion_tons + slip_tons_co2e

    wtw_total_tons_co2e = wtt_tons_co2e + ttw_total_tons_co2e

    # OPEX Economics
    bunker_cost_usd = fuel_mass_tons * profile["base_cost_usd_ton"]
    # Carbon tax applied to Tank-to-Wake taxable emissions under IMO / EU ETS rules
    carbon_tax_cost_usd = max(0.0, ttw_total_tons_co2e) * carbon_tax_usd_per_ton
    total_opex_usd = bunker_cost_usd + carbon_tax_cost_usd

    # Intensity in g CO2e / MJ
    wtw_intensity_g_mj = (
        profile["wtt_g_co2e_mj"] + profile["ttw_combustion_g_co2e_mj"] + profile["slip_g_co2e_mj"]
    )

    return {
        "pathway": pathway,
        "fuel_name": profile["fuel_name"],
        "fuel_mass_tons": float(fuel_mass_tons),
        "fuel_energy_gj": float(fuel_energy_gj),
        "wtt_emissions_tons_co2e": float(wtt_tons_co2e),
        "ttw_combustion_tons_co2": float(ttw_combustion_tons),
        "ttw_methane_slip_tons_co2e": float(slip_tons_co2e),
        "ttw_total_tons_co2e": float(ttw_total_tons_co2e),
        "wtw_total_tons_co2e": float(wtw_total_tons_co2e),
        "wtw_intensity_g_co2e_per_mj": float(wtw_intensity_g_mj),
        "bunker_cost_usd": float(bunker_cost_usd),
        "carbon_tax_cost_usd": float(carbon_tax_cost_usd),
        "total_opex_usd": float(total_opex_usd)
    }

def compare_multi_fuel_pathways(
    baseline_vlsfo_tons: float,
    carbon_tax_usd_per_ton: float = 100.0
) -> List[Dict[str, Any]]:
    """
    Compares all alternative fuel pathways against a baseline VLSFO voyage energy requirement.
    Preserves total deliverable propulsive energy while recalculating fuel mass and lifecycle emissions.
    """
    vlsfo_lhv = LIFECYCLE_EMISSION_FACTORS["VLSFO"]["lhv_mj_kg"]
    required_energy_mj = baseline_vlsfo_tons * 1000.0 * vlsfo_lhv

    comparison_results = []
    baseline_wtw = None

    for pathway, profile in LIFECYCLE_EMISSION_FACTORS.items():
        lhv_target = profile["lhv_mj_kg"]
        # Required mass of alternative fuel for equivalent energy
        target_mass_tons = (required_energy_mj / (lhv_target * 1000.0))
        res = compute_wtw_emissions(target_mass_tons, pathway, carbon_tax_usd_per_ton)

        if pathway == "VLSFO":
            baseline_wtw = res["wtw_total_tons_co2e"]
            res["emissions_reduction_pct"] = 0.0
        else:
            reduction = ((baseline_wtw - res["wtw_total_tons_co2e"]) / max(baseline_wtw, 1e-3)) * 100.0
            res["emissions_reduction_pct"] = float(round(reduction, 2))

        comparison_results.append(res)

    return comparison_results
