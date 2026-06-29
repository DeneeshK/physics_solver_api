"""
graph_builder/registry.py

THE canonical variable registry — the single source of truth for every physical
quantity the solver knows. Each quantity is ONE node with ONE symbol, globally.

Why global symbol uniqueness is the whole game
----------------------------------------------
The solver's neighbor-walk treats "two equations share a symbol" as "these
equations connect". The old graph overloaded symbols — `T` meant temperature
AND period AND tension; `C` meant capacitance AND heat capacity; `Q` meant heat
AND charge; `a`/`s` were reused as collision velocities. So the walk wired a
thermodynamics equation to an SHM equation through a bogus shared `T`, and the
LLM (correctly) rejected the nonsense — which looked like "selection failure".

Here, every distinct quantity gets a distinct symbol. Shared symbol now always
means a real physical bridge. Student/LLM notation (`F_net`, `mu_k`, `lambda`,
`Vrms`, `Q` for charge…) is captured as ALIASES and normalized to the canonical
symbol before resolution — so Stage 1 can write naturally and still land here.

Collisions resolved (old → canonical):
  charge Q            → q        (Q reserved for heat)
  resistance R        → R        ; projectile range → Rng
  capacitance C/C_cap → C        ; heat capacity → C_heat ; specific heat → c_sp
  temperature T       → T        ; period → T_p ; tension → T_F ; half-life → T_half
  weight (force) W    → F_g      (W reserved for work/energy)
  pressure/power/momentum → P (pressure) / P_pow (power) / p (momentum)
  voltage V           → V        ; volume → Vol
  internal-energy ΔU  → DeltaU   (never DeltaV)
  area A / amplitude  → A (area) ; amplitude → A_amp
  moment of inertia   → I_mom    (I reserved for current)
  frequency f         → f        ; focal length → f_lens
  refractive index    → n_ref    ; moles → n ; turns/count → N
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Var:
    sym: str                 # canonical symbol (globally unique)
    name: str                # plain-English name (used in prompts + embedding)
    dim: str                 # dimension string, dim.py notation ('MLT-2', 'L', '1')
    unit: str                # SI unit ('N', 'm', '')
    aliases: tuple = ()      # Stage-1 / student notations that map here
    signed: bool = False     # may take negative values (root selection keeps sign)
    desc: str = ""           # short gloss for retrieval context


# Each entry is authored once. `aliases` is what makes Stage-1 output land.
_VARS: list[Var] = [
    # ── Mechanics: kinematics & dynamics ──────────────────────────────────────
    Var("t", "time", "T", "s", aliases=("time", "duration", "t_time")),
    Var("s", "displacement / path length", "L", "m",
        aliases=("d_s", "distance_travelled", "path"), signed=True),
    Var("x", "position / displacement", "L", "m", aliases=("x_pos", "displacement"), signed=True),
    Var("h", "height", "L", "m", aliases=("height", "altitude", "depth_h")),
    Var("r", "radius / distance from centre", "L", "m", aliases=("radius", "R_radius")),
    Var("d", "distance / separation", "L", "m", aliases=("distance", "separation", "gap")),
    Var("L", "length", "L", "m", aliases=("l", "length", "L_len")),
    Var("u", "initial velocity", "LT-1", "m/s", aliases=("v_initial", "v_i", "u_i", "v0"), signed=True),
    Var("v", "final / instantaneous velocity", "LT-1", "m/s",
        aliases=("v_final", "v_f", "speed", "velocity", "v_max", "vmax"), signed=True),
    # Two-body collision velocities — distinct quantities, never collapsed.
    Var("u1", "initial velocity of first body", "LT-1", "m/s", aliases=("u_1",), signed=True),
    Var("u2", "initial velocity of second body", "LT-1", "m/s", aliases=("u_2",), signed=True),
    Var("v1", "final velocity of first body", "LT-1", "m/s", aliases=("v_1",), signed=True),
    Var("v2", "final velocity of second body", "LT-1", "m/s", aliases=("v_2",), signed=True),
    Var("a", "acceleration", "LT-2", "m/s^2",
        aliases=("accel", "a_acc", "a_c", "a_cent", "a_centripetal"), signed=True),
    Var("g", "acceleration due to gravity", "LT-2", "m/s^2", aliases=("g_acc",)),
    Var("Rng", "projectile range", "L", "m", aliases=("range", "R_range", "horizontal_range")),
    Var("theta", "angle", "1", "rad", aliases=("angle", "theta_angle", "phi")),

    # ── Mass / force / momentum / energy ──────────────────────────────────────
    Var("m", "mass", "M", "kg", aliases=("mass",)),
    Var("m1", "mass of first body", "M", "kg", aliases=("m_1",)),
    Var("m2", "mass of second body", "M", "kg", aliases=("m_2",)),
    Var("M", "mass of large/second body (e.g. planet)", "M", "kg", aliases=("M_big", "M_planet")),
    Var("F", "force", "MLT-2", "N", aliases=("F_net", "F_applied", "Fnet", "force"), signed=True),
    Var("F_c", "centripetal force", "MLT-2", "N", aliases=("Fc", "F_centripetal")),
    Var("F_g", "weight (gravitational force)", "MLT-2", "N", aliases=("W_weight", "weight", "Wt")),
    Var("T_F", "tension", "MLT-2", "N", aliases=("tension", "T_tension")),
    Var("N", "normal reaction force", "MLT-2", "N", aliases=("N_normal", "normal_force")),
    Var("p", "linear momentum", "MLT-1", "kg*m/s", aliases=("momentum", "p_mom"), signed=True),
    Var("J", "impulse", "MLT-1", "N*s", aliases=("impulse",)),
    Var("K", "kinetic energy", "ML2T-2", "J", aliases=("KE", "E_k", "Ek", "E_kin", "T_ke")),
    Var("U", "potential energy", "ML2T-2", "J", aliases=("PE", "E_p", "Ep", "E_pot")),
    Var("W", "work / energy", "ML2T-2", "J", aliases=("work", "E", "energy"), signed=True),
    # 'P' is conventional for BOTH pressure and power. Power carries 'P' as a
    # unit-disambiguated alias (unit 'W' → P_pow, 'Pa' → P pressure).
    Var("P_pow", "power", "ML2T-3", "W", aliases=("power", "P_power", "P")),
    Var("eff", "efficiency", "1", "", aliases=("eta", "efficiency")),

    # ── Rotational ────────────────────────────────────────────────────────────
    Var("omega", "angular velocity", "T-1", "rad/s", aliases=("angular_velocity", "w", "omega_ang")),
    Var("alpha", "angular acceleration", "T-2", "rad/s^2", aliases=("angular_accel",)),
    Var("tau", "torque", "ML2T-2", "N*m", aliases=("torque", "moment")),
    # 'I' is conventional for BOTH current and moment of inertia. Moment of
    # inertia carries 'I' as a unit-disambiguated alias (unit 'kg*m^2' → I_mom,
    # 'A' → I current).
    Var("I_mom", "moment of inertia", "ML2", "kg*m^2",
        aliases=("I_inertia", "moment_of_inertia", "MOI", "I")),
    Var("Lang", "angular momentum", "ML2T-1", "kg*m^2/s", aliases=("L_ang", "angular_momentum")),
    Var("A_amp", "amplitude", "L", "m", aliases=("amplitude", "A0", "x_max")),

    # ── Circular / SHM / gravitation ──────────────────────────────────────────
    # 'T' is the conventional symbol for BOTH temperature and period. Period
    # carries it as a unit-disambiguated alias (unit 's' → T_p, 'K' → T).
    Var("T_p", "time period", "T", "s", aliases=("period", "T_period", "time_period", "T")),
    Var("f", "frequency", "T-1", "Hz", aliases=("frequency", "nu", "freq")),
    Var("G", "universal gravitational constant", "M-1L3T-2", "N*m^2/kg^2", aliases=("G_const",)),
    Var("k_spring", "spring constant", "MT-2", "N/m", aliases=("k", "spring_constant", "k_s")),

    # ── Fluids / elasticity / surface tension ─────────────────────────────────
    Var("rho", "density", "ML-3", "kg/m^3", aliases=("density", "rho_block", "rho_water",
                                                       "rho_fluid", "rho_object", "rho_liquid")),
    Var("P", "pressure", "ML-1T-2", "Pa", aliases=("pressure", "P_excess", "P_gauge", "P_pressure")),
    # 'V' is conventional for BOTH voltage and volume. Volume carries 'V' as a
    # unit-disambiguated alias (unit 'm^3' → Vol, 'V' → V voltage).
    Var("Vol", "volume", "L3", "m^3", aliases=("volume", "V_vol", "V")),
    Var("A", "area", "L2", "m^2", aliases=("area", "A_area")),
    Var("Y", "Young's modulus", "ML-1T-2", "Pa", aliases=("young_modulus", "E_young")),
    Var("B_bulk", "bulk modulus", "ML-1T-2", "Pa", aliases=("bulk_modulus",)),
    Var("stress", "stress", "ML-1T-2", "Pa", aliases=("sigma_stress", "sigma")),
    Var("strain", "strain", "1", "", aliases=("strain_e",)),
    Var("sigma_st", "surface tension", "MT-2", "N/m",
        aliases=("surface_tension", "gamma_st", "S_st", "sigma")),
    Var("eta_visc", "coefficient of viscosity", "ML-1T-1", "Pa*s", aliases=("viscosity",)),

    # ── Friction ──────────────────────────────────────────────────────────────
    Var("mu", "coefficient of friction", "1", "", aliases=("mu_k", "mu_s", "muk", "mus", "friction_coeff")),
    Var("f_fr", "friction force", "MLT-2", "N", aliases=("friction", "f_friction", "F_fr")),

    # ── Thermal / thermodynamics ──────────────────────────────────────────────
    Var("T", "temperature", "K", "K", aliases=("temperature", "temp")),
    Var("T_hot", "hot reservoir temperature", "K", "K", aliases=("T_h", "T1_temp")),
    Var("T_cold", "cold reservoir temperature", "K", "K", aliases=("T_c", "T2_temp")),
    Var("DeltaT", "temperature change", "K", "K", aliases=("delta_T", "dT"), signed=True),
    Var("Q", "heat energy", "ML2T-2", "J", aliases=("heat", "Q_heat")),
    Var("DeltaU", "change in internal energy", "ML2T-2", "J", aliases=("dU", "delta_U", "internal_energy"), signed=True),
    Var("c_sp", "specific heat capacity", "L2T-2K-1", "J/(kg*K)", aliases=("specific_heat", "c_specific", "s_heat")),
    Var("C_heat", "heat capacity", "ML2T-2K-1", "J/K", aliases=("heat_capacity",)),
    Var("n", "amount of substance (moles)", "N", "mol", aliases=("n_mol", "moles", "n_moles")),
    Var("R_gas", "universal gas constant", "ML2T-2N-1K-1", "J/(mol*K)", aliases=("R_g",)),
    Var("k_th", "thermal conductivity", "MLT-3K-1", "W/(m*K)", aliases=("thermal_conductivity",)),

    # ── Electricity & magnetism ───────────────────────────────────────────────
    # 'Q' is the conventional symbol for BOTH heat and charge. Charge carries
    # 'Q' as a unit-disambiguated alias (unit 'C' → q, 'J' → Q heat).
    Var("q", "electric charge", "AT", "C", aliases=("charge", "Q_charge", "Q")),
    Var("q1", "first electric charge", "AT", "C", aliases=("q_1",)),
    Var("q2", "second electric charge", "AT", "C", aliases=("q_2",)),
    Var("I", "electric current", "A", "A", aliases=("current", "Irms", "I_rms", "i")),
    Var("V", "potential difference / voltage", "ML2T-3A-1", "V", aliases=("voltage", "Vrms", "V_rms", "pd", "emf_v")),
    Var("R", "resistance", "ML2T-3A-2", "ohm",
        aliases=("resistance", "R_e", "R_res", "R_eq", "Req", "R_equivalent")),
    Var("R1", "first resistance", "ML2T-3A-2", "ohm", aliases=("R_1",)),
    Var("R2", "second resistance", "ML2T-3A-2", "ohm", aliases=("R_2",)),
    Var("rho_r", "resistivity", "ML3T-3A-2", "ohm*m", aliases=("resistivity",)),
    Var("emf", "electromotive force", "ML2T-3A-1", "V", aliases=("e_mf", "E_emf", "epsilon_emf", "e")),
    Var("C", "capacitance", "M-1L-2T4A2", "F",
        aliases=("capacitance", "C_cap", "C_eq", "Ceq", "C_equivalent")),
    Var("C1", "first capacitance", "M-1L-2T4A2", "F", aliases=("C_1",)),
    Var("C2", "second capacitance", "M-1L-2T4A2", "F", aliases=("C_2",)),
    # 'E' is conventional for BOTH energy (→ W, unit J) and electric field
    # (→ E_field, unit V/m). Both carry 'E'; unit disambiguates.
    Var("E_field", "electric field strength", "MLT-3A-1", "V/m", aliases=("E_e", "electric_field", "E")),
    Var("k_e", "Coulomb constant", "ML3T-4A-2", "N*m^2/C^2", aliases=("k_coulomb",)),
    Var("epsilon_0", "permittivity of free space", "M-1L-3T4A2", "C^2/(N*m^2)", aliases=("eps0", "epsilon0")),
    Var("Phi", "magnetic flux", "ML2T-2A-1", "Wb", aliases=("flux", "magnetic_flux")),
    Var("B", "magnetic field", "MT-2A-1", "T", aliases=("magnetic_field", "B_field")),
    Var("mu_0", "permeability of free space", "MLT-2A-2", "T*m/A", aliases=("mu0",)),
    Var("X_L", "inductive reactance", "ML2T-3A-2", "ohm", aliases=("XL",)),
    Var("X_C", "capacitive reactance", "ML2T-3A-2", "ohm", aliases=("XC",)),
    Var("Z", "impedance", "ML2T-3A-2", "ohm", aliases=("impedance",)),
    Var("L_ind", "inductance", "ML2T-2A-2", "H", aliases=("inductance", "L_inductance")),

    # ── Waves / sound / optics ────────────────────────────────────────────────
    Var("lambda_", "wavelength", "L", "m", aliases=("lambda", "wavelength", "wave_length")),
    Var("v_wave", "wave / sound speed", "LT-1", "m/s", aliases=("wave_speed", "c_sound", "v_sound")),
    Var("f1", "first frequency", "T-1", "Hz", aliases=("f_1",)),
    Var("f2", "second frequency", "T-1", "Hz", aliases=("f_2",)),
    Var("f_beat", "beat frequency", "T-1", "Hz", aliases=("fb", "beat_frequency")),
    Var("f_src", "source frequency", "T-1", "Hz", aliases=("f_source", "f0_src")),
    Var("f_obs", "observed frequency", "T-1", "Hz", aliases=("f_observed", "f_apparent")),
    Var("v_obs", "observer speed", "LT-1", "m/s", aliases=("v_observer",)),
    Var("v_src", "source speed", "LT-1", "m/s", aliases=("v_source",)),
    Var("f_lens", "focal length", "L", "m", aliases=("focal_length", "f_focal")),
    Var("v_img", "image distance", "L", "m", aliases=("v_image", "image_distance", "v_i_img"), signed=True),
    Var("u_obj", "object distance", "L", "m", aliases=("u_object", "object_distance"), signed=True),
    Var("mag", "magnification", "1", "", aliases=("magnification", "m_mag")),
    Var("n_ref", "refractive index", "1", "", aliases=("refractive_index", "mu_r", "n_index")),
    Var("P_lens", "lens power", "L-1", "D", aliases=("power_lens", "dioptre")),
    Var("beta_fr", "fringe width", "L", "m", aliases=("fringe_width", "beta")),
    Var("D_slit", "slit-to-screen distance", "L", "m", aliases=("D", "screen_distance")),
    Var("d_slit", "slit separation", "L", "m", aliases=("slit_separation", "d_double")),

    # ── Modern / nuclear ──────────────────────────────────────────────────────
    Var("h_planck", "Planck constant", "ML2T-1", "J*s", aliases=("h", "planck")),
    Var("E_ph", "photon energy", "ML2T-2", "J", aliases=("photon_energy", "E_photon")),
    Var("Kmax", "maximum photoelectron kinetic energy", "ML2T-2", "J", aliases=("K_max", "KE_max")),
    Var("W0", "work function", "ML2T-2", "J", aliases=("work_function", "phi_work")),
    Var("c", "speed of light", "LT-1", "m/s", aliases=("c_light", "speed_of_light")),
    Var("N0", "initial number of nuclei", "1", "", aliases=("N_initial",)),
    Var("N_t", "number of nuclei at time t", "1", "", aliases=("N", "N_remaining")),
    Var("lambda_d", "decay constant", "T-1", "s^-1", aliases=("lambda_decay", "decay_constant", "lambda")),
    Var("T_half", "half-life", "T", "s", aliases=("half_life", "t_half")),
    Var("Act", "activity", "T-1", "Bq", aliases=("activity", "A_activity")),
    Var("Act0", "initial activity", "T-1", "Bq", aliases=("A0_activity", "activity_initial")),
    Var("nq", "principal quantum number", "1", "", aliases=("n_quantum", "n_q")),

    # ══════════════ v9 EXPANSION — JEE/NEET coverage additions ══════════════
    # ── Mechanics extras ──────────────────────────────────────────────────────
    Var("v_avg", "average velocity / speed", "LT-1", "m/s", aliases=("average_velocity", "v_average"), signed=True),
    Var("T_flight", "time of flight", "T", "s", aliases=("time_of_flight", "t_flight")),
    Var("H_max", "maximum height", "L", "m", aliases=("max_height", "H_proj", "h_max")),
    Var("R3", "third resistance", "ML2T-3A-2", "ohm", aliases=("R_3",)),
    Var("C3", "third capacitance", "M-1L-2T4A2", "F", aliases=("C_3",)),
    Var("m3", "mass of third body", "M", "kg", aliases=("m_3",)),
    Var("k_eff", "effective / equivalent spring constant", "MT-2", "N/m", aliases=("k_equivalent", "k_total")),
    # ── Gravitation extras ────────────────────────────────────────────────────
    Var("g0", "surface gravitational acceleration", "LT-2", "m/s^2", aliases=("g_surface", "g_s")),
    Var("R_E", "planet / Earth radius", "L", "m", aliases=("R_planet", "R_earth", "radius_planet")),
    Var("v_orb", "orbital velocity", "LT-1", "m/s", aliases=("orbital_velocity", "v_orbital")),
    Var("v_esc", "escape velocity", "LT-1", "m/s", aliases=("escape_velocity",)),
    # ── Rotational extras ─────────────────────────────────────────────────────
    Var("omega0", "initial angular velocity", "T-1", "rad/s", aliases=("omega_initial", "w0")),
    # ── Fluids extras ─────────────────────────────────────────────────────────
    Var("A1", "first cross-sectional area", "L2", "m^2", aliases=("A_1",)),
    Var("A2", "second cross-sectional area", "L2", "m^2", aliases=("A_2",)),
    Var("Q_flow", "volume flow rate", "L3T-1", "m^3/s", aliases=("flow_rate", "volumetric_flow")),
    Var("v_term", "terminal velocity", "LT-1", "m/s", aliases=("terminal_velocity",)),
    # ── Thermodynamics extras ─────────────────────────────────────────────────
    Var("Cp", "molar heat capacity at constant pressure", "ML2T-2N-1K-1", "J/(mol*K)", aliases=("C_p", "molar_cp")),
    Var("Cv", "molar heat capacity at constant volume", "ML2T-2N-1K-1", "J/(mol*K)", aliases=("C_v", "molar_cv")),
    Var("gamma_ad", "adiabatic index (ratio of specific heats)", "1", "", aliases=("gamma", "heat_capacity_ratio")),
    Var("v_rms", "root-mean-square molecular speed", "LT-1", "m/s", aliases=("rms_speed", "v_rms_speed")),
    # ── Capacitor / current extras ────────────────────────────────────────────
    Var("K_diel", "dielectric constant (relative permittivity)", "1", "", aliases=("dielectric_constant", "kappa", "K_dielectric", "epsilon_r")),
    Var("r_int", "internal resistance", "ML2T-3A-2", "ohm", aliases=("internal_resistance", "r_internal")),
    Var("v_d", "drift velocity", "LT-1", "m/s", aliases=("drift_velocity", "v_drift")),
    Var("n_e", "free-electron number density", "L-3", "m^-3", aliases=("electron_density", "number_density")),
    # ── Optics extras ─────────────────────────────────────────────────────────
    Var("n1", "refractive index of first medium", "1", "", aliases=("n_1", "mu1")),
    Var("n2", "refractive index of second medium", "1", "", aliases=("n_2", "mu2")),
    Var("theta1", "angle of incidence", "1", "rad", aliases=("theta_i", "angle_incidence", "i_angle")),
    Var("theta2", "angle of refraction", "1", "rad", aliases=("theta_r", "angle_refraction", "r_angle")),
    Var("theta_c", "critical angle", "1", "rad", aliases=("critical_angle",)),
    # ── Modern / atomic extras ────────────────────────────────────────────────
    Var("E_n", "energy of the nth orbit / level", "ML2T-2", "J", aliases=("E_level", "energy_level"), signed=True),
    Var("V_stop", "stopping potential", "ML2T-3A-1", "V", aliases=("stopping_potential", "V_0")),
    Var("f0_th", "threshold frequency", "T-1", "Hz", aliases=("threshold_frequency", "nu_0")),
    # Physical constants used by the new equations. Their VALUES live in
    # config.IMPLICIT_CONSTANTS_CATALOG; registering the SYMBOL here lets the
    # compiler validate equations that use them (and they are in
    # config.PHYSICAL_CONSTANTS, so the resolver never chases them).
    Var("k_B", "Boltzmann constant", "ML2T-2K-1", "J/K", aliases=("boltzmann", "k_boltzmann")),
    Var("e_charge", "elementary charge", "AT", "C", aliases=("elementary_charge", "q_e")),
]


# ── Derived lookups ───────────────────────────────────────────────────────────
VARS: dict[str, Var] = {}
ALIASES: dict[str, str] = {}
_ALIAS_DIM: dict[str, list] = {}   # alias -> [(dim_string, canonical)] for dim-aware

def _register():
    for v in _VARS:
        if v.sym in VARS:
            raise ValueError(f"duplicate canonical symbol: {v.sym}")
        VARS[v.sym] = v
        for al in v.aliases:
            # An alias may legitimately map to different canonicals depending on
            # dimension (e.g. 'lambda' → wavelength L, but in nuclear context →
            # decay constant T-1). Record per-dimension so canonicalization can
            # disambiguate; keep a default (first-registered) in ALIASES.
            _ALIAS_DIM.setdefault(al, []).append((v.dim, v.sym))
            ALIASES.setdefault(al, v.sym)

_register()


def canonical(sym: str, dim: str = "") -> str:
    """Map a Stage-1 symbol to its canonical registry symbol. If `dim` is given
    and the alias is dimension-ambiguous, pick the canonical whose dimension
    matches. Returns the input unchanged if it is already canonical/unknown."""
    if sym in VARS:
        return sym
    options = _ALIAS_DIM.get(sym)
    if not options:
        return sym
    if dim and len(options) > 1:
        from graph_builder.dim import parse_dim_string
        want = parse_dim_string(dim)
        for d, canon in options:
            if parse_dim_string(d) == want:
                return canon
    return options[0][1]
