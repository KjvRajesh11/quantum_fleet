"""
Naval Hydrodynamic Resistance Engine for Quantum Fleet.
Implements:
1. ITTC-1957 skin friction correlation with Holtrop form factor & roughness allowance.
2. Holtrop & Mennen (1982/1984) wave resistance formulation + STAWAVE spectral wave drag.
3. Isherwood (1972) / Fujiwara aerodynamic wind resistance estimation.
"""

import math
from typing import Dict, Any, Tuple
from app.core.config import settings

def compute_reynolds_number(v_stw: float, l_wl: float, nu: float = settings.KINEMATIC_VISCOSITY) -> float:
    """
    Computes Reynolds number: Re = (V_stw * L_wl) / nu
    v_stw: Speed through water (m/s)
    l_wl: Length on waterline (m)
    nu: Kinematic viscosity of seawater (m^2/s)
    """
    if v_stw <= 0.0 or l_wl <= 0.0:
        return 1e5
    return (v_stw * l_wl) / nu

def compute_frictional_resistance(
    v_stw: float,
    l_wl: float,
    beam: float,
    draft: float,
    c_b: float,
    displacement_m3: float,
    c_m: float = 0.98,
    c_wp: float = 0.85,
    roughness_ks_um: float = 150.0,
    rho: float = settings.SEAWATER_DENSITY,
    nu: float = settings.KINEMATIC_VISCOSITY
) -> Dict[str, float]:
    """
    ITTC-1957 skin friction line with Holtrop-Mennen form factor (1+k1) and Bowden roughness allowance.
    Returns:
        r_f: Frictional resistance in Newtons (N)
        c_f: ITTC friction coefficient
        form_factor: (1 + k1)
        wetted_surface: S in m^2
    """
    if v_stw <= 0.01:
        return {"r_f": 0.0, "c_f": 0.0, "form_factor": 1.0, "wetted_surface": 1.0}

    # 1. Wetted surface area (Holtrop-Mennen approximation)
    # S = L * (2*T + B) * sqrt(C_M) * (0.453 + 0.4425*C_B - 0.2862*C_M - 0.00346*(B/T) + 0.3696*C_WP)
    b_t_ratio = beam / max(draft, 0.5)
    s_factor = 0.453 + 0.4425 * c_b - 0.2862 * c_m - 0.00346 * b_t_ratio + 0.3696 * c_wp
    s_factor = max(s_factor, 0.35)
    wetted_surface = l_wl * (2.0 * draft + beam) * math.sqrt(c_m) * s_factor
    wetted_surface = max(wetted_surface, 1.0)

    # 2. Reynolds number & ITTC-1957 skin friction
    re = compute_reynolds_number(v_stw, l_wl, nu)
    log_re = math.log10(max(re, 1e4))
    c_f0 = 0.075 / ((log_re - 2.0) ** 2)

    # 3. Form factor (1 + k1) via Holtrop & Mennen
    # Approximation based on hull fullness
    l_r = l_wl * (1.0 - c_b + (0.06 * c_b * (0.8 / max(c_b, 0.1)))) # run length estimate
    l_r = max(l_r, l_wl * 0.2)
    c_p = c_b / max(c_m, 0.7)
    
    # 1 + k1 Holtrop regression parameter
    c14 = 1.0 + 0.011 * 0.0 # conventional stern
    try:
        k1_term = 0.93 + 0.487118 * c14 * ((beam / l_wl) ** 1.06806) * \
                  ((draft / l_wl) ** 0.46106) * ((l_wl / l_r) ** 0.121563) * \
                  (((l_wl ** 3) / max(displacement_m3, 100.0)) ** 0.36486) * \
                  ((1.0 - c_p) ** -0.604247)
        form_factor = max(1.05, min(k1_term, 1.45))
    except (ValueError, ZeroDivisionError, OverflowError):
        form_factor = 1.18 # robust fallback for merchant hulls

    # 4. Hull surface roughness allowance Delta C_F (Bowden-Norrbin / ITTC standard)
    ks_m = roughness_ks_um * 1e-6
    delta_cf = max(0.0, 0.044 * ((ks_m / l_wl) ** (1.0 / 3.0) - 10.0 * (re ** (-1.0 / 3.0))) + 0.000125)
    delta_cf = min(delta_cf, 0.0006)

    # 5. Frictional resistance force
    total_cf = (c_f0 * form_factor) + delta_cf
    r_f = 0.5 * rho * wetted_surface * (v_stw ** 2) * total_cf

    return {
        "r_f": float(r_f),
        "c_f": float(c_f0),
        "form_factor": float(form_factor),
        "wetted_surface": float(wetted_surface),
        "re": float(re),
        "delta_cf": float(delta_cf)
    }

def compute_wave_resistance(
    v_stw: float,
    l_wl: float,
    beam: float,
    draft: float,
    displacement_m3: float,
    c_b: float,
    hs_wave: float = 0.0,
    tp_wave: float = 8.0,
    wave_angle_deg: float = 0.0, # 0 = head sea, 180 = following sea
    rho: float = settings.SEAWATER_DENSITY,
    g: float = settings.GRAVITY
) -> Dict[str, float]:
    """
    Holtrop & Mennen (1982) calm water wave-making resistance + STAWAVE spectral wave added resistance.
    Returns:
        r_w: Calm water wave resistance (N)
        r_aw: Added resistance in irregular seaway (N)
        r_wave_total: Total wave resistance (N)
        fn: Froude number
    """
    if v_stw <= 0.01:
        return {"r_w": 0.0, "r_aw": 0.0, "r_wave_total": 0.0, "fn": 0.0}

    # 1. Froude Number Fn = V_stw / sqrt(g * L_wl)
    fn = v_stw / math.sqrt(g * l_wl)

    # 2. Holtrop & Mennen Calm Water Wave Resistance
    c_p = c_b / 0.98
    c1 = 2223105.0 * ((beam / l_wl) ** 3.78613) * ((draft / beam) ** 1.07961) * ((1.0 - c_p) ** 1.37465)
    c1 = max(0.5, min(c1, 15.0))
    c2 = 1.0 # bulbous bow attenuation factor
    c5 = 1.0 - 0.8 * (0.0 / max(beam * draft * c_b, 1.0)) # transom factor

    # Exponents m1, d, lambda
    m1 = 0.0140407 * (l_wl / draft) - 1.75254 * (displacement_m3 ** (1.0 / 3.0) / l_wl) - 4.79323 * (beam / l_wl) - 0.5
    m1 = max(-2.5, min(m1, 0.0))
    d = -0.9
    lambda_factor = 1.446 * c_p - 0.03 * (l_wl / beam)
    lambda_factor = max(0.5, min(lambda_factor, 1.2))

    # Calm water wave drag
    try:
        exp_term = math.exp(m1 * (fn ** d) - 0.1 * math.cos(lambda_factor * (fn ** -2.0)))
        r_w = c1 * c2 * c5 * displacement_m3 * rho * g * exp_term
        # Bound physically to reasonable naval proportions
        r_w = max(0.0, min(r_w, 0.5 * rho * (beam * draft) * (v_stw ** 2) * 1.5))
    except (ValueError, OverflowError):
        # Semi-empirical cubic Froude fallback
        r_w = 0.5 * rho * (beam * draft) * (v_stw ** 2) * (fn ** 2.5) * 0.15

    # 3. Added Wave Resistance in Irregular Seaway (STAWAVE-1 / Kwon Modified Spectral Drag)
    # R_aw = 1/16 * rho * g * Hs^2 * B * sqrt(B / L_wl) * transfer_function(wave_angle, Fn)
    if hs_wave > 0.05:
        rad_angle = math.radians(wave_angle_deg)
        # Heading directivity: Head seas (0 deg) exert maximum resistance; following seas (180 deg) reduce to ~10%
        heading_coef = max(0.05, 0.5 * (1.0 + math.cos(rad_angle)))
        
        # Wavelength approximation from peak period: lambda_w = g * Tp^2 / (2 * pi)
        lambda_w = (g * (tp_wave ** 2)) / (2.0 * math.pi)
        tuning_ratio = l_wl / max(lambda_w, 1.0)
        
        # Peak resonance occurs when ship length is close to wavelength (L/lambda ~ 1.0 - 1.3)
        resonance_amp = math.exp(-0.5 * (((tuning_ratio - 1.15) / 0.4) ** 2))
        raw_base = (1.0 / 16.0) * rho * g * (hs_wave ** 2) * beam * math.sqrt(beam / l_wl)
        r_aw = raw_base * (0.8 + 1.2 * resonance_amp) * heading_coef * (1.0 + 2.0 * fn)
    else:
        r_aw = 0.0

    return {
        "r_w": float(r_w),
        "r_aw": float(r_aw),
        "r_wave_total": float(r_w + r_aw),
        "fn": float(fn)
    }

def compute_wind_resistance(
    v_sog: float,
    ship_heading_deg: float,
    wind_speed_ms: float,
    wind_direction_deg: float, # meteorological: direction wind is blowing from
    transverse_area_m2: float,
    lateral_area_m2: float,
    length_overall: float,
    beam: float,
    rho_air: float = settings.AIR_DENSITY
) -> Dict[str, float]:
    """
    Isherwood (1972) / Blendermann Wind Resistance Formulation.
    Computes apparent wind vector and aerodynamic drag force on ship superstructure & cargo.
    Returns:
        r_wind: Aerodynamic longitudinal drag force in Newtons (N)
        v_rel: Apparent wind speed (m/s)
        gamma_rel: Apparent wind angle relative to vessel bow (degrees, 0 = head wind)
        c_x: Longitudinal drag coefficient
    """
    # 1. Resolve true wind and ship motion to find apparent wind vector
    # Ship heading in radians
    psi_ship = math.radians(ship_heading_deg)
    # Wind angle from which it blows
    psi_wind = math.radians(wind_direction_deg)

    # Vessel velocity components (m/s)
    u_ship = v_sog * math.sin(psi_ship)
    v_ship = v_sog * math.cos(psi_ship)

    # True wind components (wind blowing towards opposite angle)
    u_true = -wind_speed_ms * math.sin(psi_wind)
    v_true = -wind_speed_ms * math.cos(psi_wind)

    # Relative wind seen by vessel
    u_rel = u_true - u_ship
    v_rel = v_true - v_ship

    v_rel_mag = math.sqrt(u_rel ** 2 + v_rel ** 2)

    if v_rel_mag <= 0.1:
        return {"r_wind": 0.0, "v_rel": 0.0, "gamma_rel": 0.0, "c_x": 0.0}

    # Apparent wind angle relative to ship bow (0 to 180 degrees)
    # Ship bow vector is (sin(psi), cos(psi))
    dot_prod = (u_rel * u_ship + v_rel * v_ship) if v_sog > 0.01 else (u_rel * math.sin(psi_ship) + v_rel * math.cos(psi_ship))
    # Relative bearing
    alpha_rel = math.atan2(u_rel, v_rel) - psi_ship
    # Normalize to [-pi, pi]
    alpha_rel = (alpha_rel + math.pi) % (2.0 * math.pi) - math.pi
    gamma_rel_deg = abs(math.degrees(alpha_rel))

    # 2. Isherwood / Fujiwara Longitudinal Aerodynamic Drag Coefficient C_X(gamma)
    # C_X = -a0 + a1 * cos(gamma) + a2 * cos(2*gamma) + ...
    # Standard merchant profile (Container / Bulker / Tanker)
    rad_gamma = math.radians(gamma_rel_deg)
    
    # Area ratio parameters
    area_ratio = lateral_area_m2 / max(transverse_area_m2, 10.0)
    b_l_ratio = beam / max(length_overall, 10.0)
    
    # Isherwood regression approximation for C_X
    # Head wind (0 deg): CX ~ 0.85 to 1.05; Beam wind (90 deg): CX ~ 0.0 to 0.1; Following wind (180 deg): CX ~ -0.6 to -0.8
    c_x0 = 0.92 + 0.05 * (area_ratio / 6.0)
    c_x180 = -0.65 - 0.04 * (area_ratio / 6.0)
    
    c_x = c_x0 * math.cos(rad_gamma)
    if gamma_rel_deg > 90.0:
        c_x = c_x180 * (math.sin(rad_gamma - math.pi / 2.0) ** 2) * (-1.0 if math.cos(rad_gamma) < 0 else 1.0)
        c_x = min(c_x, 0.0)
    else:
        c_x = max(c_x, 0.0)

    # 3. Longitudinal Wind Force R_wind = 0.5 * rho_air * A_T * V_rel^2 * C_X
    r_wind = 0.5 * rho_air * transverse_area_m2 * (v_rel_mag ** 2) * c_x

    return {
        "r_wind": float(r_wind),
        "v_rel": float(v_rel_mag),
        "gamma_rel": float(gamma_rel_deg),
        "c_x": float(c_x)
    }

def compute_total_resistance(
    v_stw: float,
    v_sog: float,
    l_wl: float,
    beam: float,
    draft: float,
    c_b: float,
    displacement_m3: float,
    ship_heading_deg: float,
    wind_speed_ms: float,
    wind_direction_deg: float,
    transverse_area_m2: float,
    lateral_area_m2: float,
    hs_wave: float = 0.0,
    tp_wave: float = 8.0,
    wave_angle_deg: float = 0.0
) -> Dict[str, Any]:
    """
    Master Naval Hydrodynamic Resistance Engine:
    R_total = R_frictional + R_wave + R_wind
    """
    fric = compute_frictional_resistance(
        v_stw=v_stw,
        l_wl=l_wl,
        beam=beam,
        draft=draft,
        c_b=c_b,
        displacement_m3=displacement_m3
    )

    wave = compute_wave_resistance(
        v_stw=v_stw,
        l_wl=l_wl,
        beam=beam,
        draft=draft,
        displacement_m3=displacement_m3,
        c_b=c_b,
        hs_wave=hs_wave,
        tp_wave=tp_wave,
        wave_angle_deg=wave_angle_deg
    )

    wind = compute_wind_resistance(
        v_sog=v_sog,
        ship_heading_deg=ship_heading_deg,
        wind_speed_ms=wind_speed_ms,
        wind_direction_deg=wind_direction_deg,
        transverse_area_m2=transverse_area_m2,
        lateral_area_m2=lateral_area_m2,
        length_overall=l_wl * 1.03,
        beam=beam
    )

    r_f = fric["r_f"]
    r_wave = wave["r_wave_total"]
    r_wind = max(0.0, wind["r_wind"]) # Forward drag only adds to required thrust

    r_total = r_f + r_wave + r_wind

    return {
        "r_frictional_n": float(r_f),
        "r_wave_n": float(r_wave),
        "r_wind_n": float(r_wind),
        "r_total_n": float(r_total),
        "r_total_kn": float(r_total / 1000.0),
        "wetted_surface_m2": fric["wetted_surface"],
        "fn": wave["fn"],
        "frictional_breakdown": fric,
        "wave_breakdown": wave,
        "wind_breakdown": wind
    }
