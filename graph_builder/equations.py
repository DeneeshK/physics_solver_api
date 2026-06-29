"""
graph_builder/equations.py

Equation definitions, authored once using ONLY canonical registry symbols.
The compiler validates every entry (symbol membership + dimensional
consistency) before emitting, so a physics/typo error fails the build instead
of shipping a broken node.

Each Eq:
  id        unique node id
  domain    physics area (matches the old taxonomy for retrieval continuity)
  subdomain finer bucket (free text)
  expr      'LHS = RHS' using canonical symbols only; LHS is the natural output
  rag       one/two sentence concept description — what this equation is FOR.
            This is what gets embedded + BM25'd, so it must read like the
            phrase a student would use to ask for it.
  output    natural output symbol (defaults to the LHS symbol)
  conditions/jee/neet/mistakes  optional metadata lists
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Eq:
    id: str
    domain: str
    subdomain: str
    expr: str
    rag: str
    output: str = ""
    latex: str = ""
    conditions: tuple = ()
    jee: tuple = ()
    neet: tuple = ()
    mistakes: tuple = ()
    skip_dim_check: bool = False


EQUATIONS: list[Eq] = [
    # ════════════════ KINEMATICS ════════════════
    Eq("kinematics_v_u_at", "kinematics", "equations_of_motion",
       "v = u + a*t",
       "Final velocity from initial velocity, constant acceleration and time (first equation of motion)."),
    Eq("kinematics_s_ut_half_at2", "kinematics", "equations_of_motion",
       "s = u*t + (1/2)*a*t**2",
       "Displacement under constant acceleration from initial velocity and time (second equation of motion)."),
    Eq("kinematics_v2_u2_2as", "kinematics", "equations_of_motion",
       "v**2 = u**2 + 2*a*s",
       "Velocity–displacement relation under constant acceleration, with no time (third equation of motion).",
       output="v"),
    Eq("kinematics_avg_velocity", "kinematics", "average",
       "v = s/t",
       "Average velocity as displacement over time."),
    Eq("kinematics_free_fall_v", "kinematics", "free_fall",
       "v = u + g*t",
       "Speed of a freely falling body under gravity after time t.",
       conditions=("free fall", "vertical motion")),
    Eq("kinematics_free_fall_h", "kinematics", "free_fall",
       "h = u*t + (1/2)*g*t**2",
       "Height fallen by a body under gravity in time t."),
    Eq("kinematics_projectile_range", "kinematics", "projectile",
       "Rng = u**2*sin(2*theta)/g",
       "Horizontal range of a projectile launched with speed u at angle theta."),

    # ════════════════ LAWS OF MOTION ════════════════
    Eq("laws_newton_second", "laws_of_motion", "newton",
       "F = m*a",
       "Newton's second law: net force equals mass times acceleration."),
    Eq("laws_momentum_rate", "laws_of_motion", "newton",
       "F = p/t",
       "Force as rate of change of momentum."),
    Eq("laws_impulse", "laws_of_motion", "impulse",
       "J = F*t",
       "Impulse delivered by a constant force over a time interval."),
    Eq("laws_weight", "laws_of_motion", "weight",
       "F_g = m*g",
       "Weight of a body: gravitational force equals mass times g."),
    Eq("laws_normal_incline", "laws_of_motion", "incline",
       "N = m*g*cos(theta)",
       "Normal reaction on a block resting on an incline of angle theta.",
       conditions=("inclined plane",)),

    # ════════════════ FRICTION ════════════════
    Eq("friction_force", "friction", "kinetic",
       "f_fr = mu*N",
       "Friction force equals coefficient of friction times the normal reaction."),
    Eq("friction_incline_limiting", "friction", "static",
       "mu = tan(theta)",
       "Coefficient of static friction from the angle of repose where a block just slides.",
       conditions=("just begins to slide", "angle of repose")),

    # ════════════════ WORK, ENERGY, POWER ════════════════
    Eq("work_force_displacement", "work_energy_power", "work",
       "W = F*s*cos(theta)",
       "Work done by a constant force over a displacement at angle theta to the force."),
    Eq("energy_kinetic", "work_energy_power", "energy",
       "K = (1/2)*m*v**2",
       "Translational kinetic energy of a moving body."),
    Eq("energy_potential_gravity", "work_energy_power", "energy",
       "U = m*g*h",
       "Gravitational potential energy near Earth's surface at height h."),
    Eq("power_work_time", "work_energy_power", "power",
       "P_pow = W/t",
       "Average power as work done per unit time."),
    Eq("power_force_velocity", "work_energy_power", "power",
       "P_pow = F*v",
       "Instantaneous power delivered by a force to a body moving at speed v."),

    # ════════════════ CIRCULAR MOTION ════════════════
    Eq("circular_centripetal_accel", "circular_motion", "centripetal",
       "a = v**2/r",
       "Centripetal acceleration of a body in uniform circular motion."),
    Eq("circular_centripetal_force", "circular_motion", "centripetal",
       "F_c = m*v**2/r",
       "Centripetal force needed to keep a mass moving in a circle of radius r at speed v."),
    Eq("circular_angular_speed", "circular_motion", "angular",
       "v = omega*r",
       "Relation between linear speed and angular speed for circular motion."),
    Eq("circular_force_omega", "circular_motion", "centripetal",
       "F_c = m*omega**2*r",
       "Centripetal force in terms of angular velocity."),

    # ════════════════ MOMENTUM & COLLISIONS ════════════════
    Eq("momentum_linear", "momentum_collisions", "momentum",
       "p = m*v",
       "Linear momentum of a moving body."),
    Eq("momentum_conservation_two_body", "momentum_collisions", "collision",
       "m1*u1 + m2*u2 = m1*v1 + m2*v2",
       "Conservation of linear momentum in a two-body collision (initial total = final total).",
       output="v2",
       conditions=("isolated system", "collision")),
    Eq("collision_impulse_momentum", "momentum_collisions", "impulse",
       "J = m*v - m*u",
       "Impulse equals change in momentum of a body.", output="J"),

    # ════════════════ SHM ════════════════
    Eq("shm_displacement", "SHM", "kinematics",
       "x = A_amp*sin(omega*t)",
       "Displacement of a simple-harmonic oscillator as a function of time."),
    Eq("shm_max_speed", "SHM", "kinematics",
       "v = omega*A_amp",
       "Maximum speed of a simple-harmonic oscillator (at the mean position)."),
    Eq("shm_max_accel", "SHM", "kinematics",
       "a = omega**2*A_amp",
       "Maximum acceleration of a simple-harmonic oscillator (at the extreme)."),
    Eq("shm_period_spring", "SHM", "period",
       "T_p = 2*pi*sqrt(m/k_spring)",
       "Time period of a mass–spring oscillator."),
    Eq("shm_period_pendulum", "SHM", "period",
       "T_p = 2*pi*sqrt(L/g)",
       "Time period of a simple pendulum of length L under gravity g.",
       conditions=("small oscillations", "simple pendulum")),
    Eq("shm_angular_frequency", "SHM", "frequency",
       "omega = 2*pi*f",
       "Angular frequency from ordinary frequency."),
    Eq("shm_freq_period", "SHM", "frequency",
       "f = 1/T_p",
       "Frequency as the reciprocal of the time period."),

    # ════════════════ GRAVITATION ════════════════
    Eq("gravitation_universal_law", "gravitation", "force",
       "F = G*m*M/r**2",
       "Newton's law of universal gravitation between two masses separated by r."),
    Eq("gravitation_field_g", "gravitation", "field",
       "g = G*M/r**2",
       "Gravitational acceleration at distance r from a mass M (e.g. above a planet's centre)."),
    Eq("gravitation_potential_energy", "gravitation", "energy",
       "U = -G*m*M/r",
       "Gravitational potential energy of a two-mass system."),

    # ════════════════ CURRENT ELECTRICITY ════════════════
    Eq("current_ohms_law", "current_electricity", "ohm",
       "V = I*R",
       "Ohm's law: voltage equals current times resistance."),
    Eq("current_power_vi", "current_electricity", "power",
       "P_pow = V*I",
       "Electrical power as voltage times current."),
    Eq("current_power_i2r", "current_electricity", "power",
       "P_pow = I**2*R",
       "Power dissipated in a resistor (Joule heating)."),
    Eq("current_resistance_wire", "current_electricity", "resistivity",
       "R = rho_r*L/A",
       "Resistance of a uniform wire from resistivity, length and cross-section."),
    Eq("current_resistors_series", "current_electricity", "combination",
       "R = R1 + R2",
       "Equivalent resistance of two resistors in series."),
    Eq("current_resistors_parallel", "current_electricity", "combination",
       "R = R1*R2/(R1 + R2)",
       "Equivalent resistance of two resistors in parallel."),

    # ════════════════ ELECTROSTATICS ════════════════
    Eq("electrostatics_coulomb", "electrostatics", "force",
       "F = k_e*q1*q2/r**2",
       "Coulomb force between two point charges separated by distance r."),
    Eq("electrostatics_field_point", "electrostatics", "field",
       "E_field = k_e*q/r**2",
       "Electric field of a point charge at distance r."),
    Eq("electrostatics_field_voltage", "electrostatics", "field",
       "E_field = V/d",
       "Uniform electric field between plates from voltage and separation."),

    # ════════════════ CAPACITORS ════════════════
    Eq("capacitor_definition", "capacitors", "definition",
       "q = C*V",
       "Charge stored on a capacitor at voltage V."),
    Eq("capacitor_parallel_plate", "capacitors", "geometry",
       "C = epsilon_0*A/d",
       "Capacitance of a parallel-plate capacitor."),
    Eq("capacitor_energy", "capacitors", "energy",
       "U = (1/2)*C*V**2",
       "Energy stored in a charged capacitor."),
    Eq("capacitor_series", "capacitors", "combination",
       "C = C1*C2/(C1 + C2)",
       "Equivalent capacitance of two capacitors in series."),
    Eq("capacitor_parallel", "capacitors", "combination",
       "C = C1 + C2",
       "Equivalent capacitance of two capacitors in parallel."),

    # ════════════════ AC CIRCUITS ════════════════
    Eq("ac_capacitive_reactance", "AC_circuits", "reactance",
       "X_C = 1/(omega*C)",
       "Capacitive reactance of a capacitor at angular frequency omega."),
    Eq("ac_inductive_reactance", "AC_circuits", "reactance",
       "X_L = omega*L_ind",
       "Inductive reactance of an inductor at angular frequency omega."),
    Eq("ac_impedance_series", "AC_circuits", "impedance",
       "Z = sqrt(R**2 + (X_L - X_C)**2)",
       "Impedance of a series LCR circuit."),
    Eq("ac_current_rms", "AC_circuits", "ohm",
       "I = V/Z",
       "RMS current in an AC circuit from RMS voltage and impedance."),

    # ════════════════ THERMAL PHYSICS ════════════════
    Eq("thermal_heat_specific", "thermal_physics", "calorimetry",
       "Q = m*c_sp*DeltaT",
       "Heat absorbed or released to change a body's temperature."),
    Eq("thermal_conduction_rate", "thermal_physics", "conduction",
       "P_pow = k_th*A*DeltaT/L",
       "Rate of heat conduction through a slab (Fourier's law)."),

    # ════════════════ THERMODYNAMICS ════════════════
    Eq("thermo_first_law", "thermodynamics", "first_law",
       "Q = DeltaU + W",
       "First law of thermodynamics: heat supplied equals change in internal energy plus work done by the gas."),
    Eq("thermo_carnot_efficiency", "thermodynamics", "efficiency",
       "eff = 1 - T_cold/T_hot",
       "Efficiency of a Carnot engine between hot and cold reservoirs."),
    Eq("thermo_efficiency_def", "thermodynamics", "efficiency",
       "eff = W/Q",
       "Engine efficiency as work output over heat input."),
    Eq("thermo_ideal_gas", "thermodynamics", "gas_law",
       "P*Vol = n*R_gas*T",
       "Ideal gas law relating pressure, volume, moles and temperature.",
       output="P"),

    # ════════════════ MODERN PHYSICS ════════════════
    Eq("modern_photon_energy", "modern_physics", "photon",
       "E_ph = h_planck*f",
       "Energy of a photon from its frequency."),
    Eq("modern_photoelectric", "modern_physics", "photoelectric",
       "Kmax = h_planck*f - W0",
       "Einstein's photoelectric equation: max kinetic energy of ejected electrons."),
    Eq("modern_de_broglie", "modern_physics", "matter_wave",
       "lambda_ = h_planck/p",
       "de Broglie wavelength of a particle with momentum p."),

    # ════════════════ NUCLEAR PHYSICS ════════════════
    Eq("nuclear_decay_number", "nuclear_physics", "decay",
       "N_t = N0*exp(-lambda_d*t)",
       "Radioactive decay law: nuclei remaining after time t."),
    Eq("nuclear_half_life", "nuclear_physics", "decay",
       "T_half = log(2)/lambda_d",
       "Half-life in terms of the decay constant."),
    Eq("nuclear_activity", "nuclear_physics", "activity",
       "Act = lambda_d*N_t",
       "Activity of a sample equals decay constant times number of nuclei."),
    Eq("nuclear_activity_decay", "nuclear_physics", "activity",
       "Act = Act0*exp(-lambda_d*t)",
       "Activity of a radioactive sample as a function of time."),

    # ════════════════ SOUND / WAVES ════════════════
    Eq("wave_speed", "waves", "basics",
       "v_wave = f*lambda_",
       "Wave speed equals frequency times wavelength."),
    Eq("sound_beat", "sound", "beats",
       "f_beat = f1 - f2",
       "Beat frequency equals the difference of two close frequencies.", output="f_beat"),
    Eq("sound_doppler_observer", "sound", "doppler",
       "f_obs = f_src*(v_wave + v_obs)/v_wave",
       "Observed frequency when the observer moves toward a stationary source."),
    Eq("sound_doppler_source", "sound", "doppler",
       "f_obs = f_src*v_wave/(v_wave - v_src)",
       "Observed frequency when the source moves toward a stationary observer."),

    # ════════════════ RAY OPTICS ════════════════
    Eq("optics_lens_formula", "ray_optics", "lens",
       "1/f_lens = 1/v_img - 1/u_obj",
       "Thin lens formula relating focal length, image and object distance.", output="f_lens"),
    Eq("optics_magnification", "ray_optics", "lens",
       "mag = v_img/u_obj",
       "Linear magnification of a lens."),
    Eq("optics_lens_power", "ray_optics", "lens",
       "P_lens = 1/f_lens",
       "Power of a lens in dioptres from its focal length."),

    # ════════════════ WAVE OPTICS ════════════════
    Eq("wave_optics_fringe_width", "wave_optics", "interference",
       "beta_fr = lambda_*D_slit/d_slit",
       "Fringe width in Young's double-slit experiment."),

    # ════════════════ SURFACE TENSION ════════════════
    Eq("surface_tension_drop", "surface_tension", "excess_pressure",
       "P = 2*sigma_st/r",
       "Excess pressure inside a spherical liquid drop."),
    Eq("surface_tension_bubble", "surface_tension", "excess_pressure",
       "P = 4*sigma_st/r",
       "Excess pressure inside a soap bubble (two surfaces)."),
    Eq("surface_tension_capillary", "surface_tension", "capillarity",
       "h = 2*sigma_st*cos(theta)/(rho*g*r)",
       "Capillary rise of a liquid in a tube of radius r."),

    # ════════════════ ELASTICITY ════════════════
    Eq("elasticity_stress", "elasticity", "stress",
       "stress = F/A",
       "Stress as force per unit cross-sectional area."),
    Eq("elasticity_strain", "elasticity", "strain",
       "strain = s/L",
       "Longitudinal strain as extension over original length."),
    Eq("elasticity_young", "elasticity", "modulus",
       "Y = stress/strain",
       "Young's modulus as the ratio of stress to strain."),

    # ════════════════ MAGNETISM ════════════════
    Eq("magnetism_lorentz", "magnetism", "force",
       "F = q*v*B*sin(theta)",
       "Magnetic (Lorentz) force on a charge moving through a magnetic field."),
    Eq("magnetism_force_wire", "magnetism", "force",
       "F = B*I*L*sin(theta)",
       "Force on a current-carrying wire in a magnetic field."),
    Eq("magnetism_field_long_wire", "magnetism", "field",
       "B = mu_0*I/(2*pi*r)",
       "Magnetic field at distance r from a long straight current-carrying wire."),

    # ════════════════ ELECTROMAGNETIC INDUCTION ════════════════
    Eq("emi_faraday", "electromagnetic_induction", "faraday",
       "emf = Phi/t",
       "Magnitude of induced emf from the rate of change of magnetic flux."),
    Eq("emi_flux", "electromagnetic_induction", "flux",
       "Phi = B*A*cos(theta)",
       "Magnetic flux through a loop of area A in a field B."),
    Eq("emi_motional", "electromagnetic_induction", "motional",
       "emf = B*L*v",
       "Motional emf of a rod of length L moving at speed v through field B."),

    # ════════════════ ROTATIONAL MOTION ════════════════
    Eq("rotation_torque", "rotational_motion", "dynamics",
       "tau = I_mom*alpha",
       "Rotational analogue of Newton's second law: torque equals moment of inertia times angular acceleration."),
    Eq("rotation_angular_momentum", "rotational_motion", "momentum",
       "Lang = I_mom*omega",
       "Angular momentum as moment of inertia times angular velocity."),
    Eq("rotation_kinetic_energy", "rotational_motion", "energy",
       "K = (1/2)*I_mom*omega**2",
       "Rotational kinetic energy of a rigid body.", output="K"),
    Eq("rotation_inertia_disc", "rotational_motion", "inertia",
       "I_mom = (1/2)*m*r**2",
       "Moment of inertia of a uniform disc about its central axis."),
    Eq("rotation_inertia_sphere", "rotational_motion", "inertia",
       "I_mom = (2/5)*m*r**2",
       "Moment of inertia of a solid sphere about a diameter."),

    # ════════════════ FLUID MECHANICS ════════════════
    Eq("fluid_gauge_pressure", "fluid_mechanics", "pressure",
       "P = rho*g*h",
       "Gauge pressure at depth h in a fluid of density rho."),
    Eq("fluid_buoyancy", "fluid_mechanics", "buoyancy",
       "F = rho*Vol*g",
       "Buoyant force (Archimedes) on a body of volume Vol submerged in a fluid.", output="F"),

    # ════════════════════════════════════════════════════════════════════════
    # v9 EXPANSION — gap-fillers + core JEE/NEET coverage
    # ════════════════════════════════════════════════════════════════════════

    # ── Mechanics: mass / density / kinematics ────────────────────────────────
    Eq("mechanics_density", "fluid_mechanics", "density",
       "rho = m/Vol",
       "Density as mass per unit volume; gives the mass of a body from its density and volume.",
       output="rho"),
    Eq("kinematics_avg_uv", "kinematics", "average",
       "v_avg = (u + v)/2",
       "Average velocity under uniform acceleration as the mean of initial and final velocity."),
    Eq("kinematics_disp_avg", "kinematics", "average",
       "s = v_avg*t",
       "Displacement as average velocity times time."),
    Eq("projectile_time_of_flight", "kinematics", "projectile",
       "T_flight = 2*u*sin(theta)/g",
       "Total time of flight of a projectile launched with speed u at angle theta."),
    Eq("projectile_max_height", "kinematics", "projectile",
       "H_max = u**2*sin(theta)**2/(2*g)",
       "Maximum height reached by a projectile launched with speed u at angle theta."),

    # ── Connected blocks / Atwood (the simultaneous-pair result, in closed form) ─
    Eq("atwood_acceleration", "laws_of_motion", "connected_bodies",
       "a = (m1 - m2)*g/(m1 + m2)",
       "Acceleration of an Atwood / two-block system of masses m1 and m2 connected over a pulley.",
       conditions=("light inextensible string", "frictionless pulley")),
    Eq("atwood_tension", "laws_of_motion", "connected_bodies",
       "T_F = 2*m1*m2*g/(m1 + m2)",
       "Tension in the string of an Atwood / two-block system of masses m1 and m2.",
       output="T_F",
       conditions=("light inextensible string", "frictionless pulley")),

    # ── Work–energy / SHM energy ──────────────────────────────────────────────
    Eq("spring_potential_energy", "work_energy_power", "energy",
       "U = (1/2)*k_spring*x**2",
       "Elastic potential energy stored in a spring stretched or compressed by x."),
    Eq("shm_total_energy", "SHM", "energy",
       "W = (1/2)*k_spring*A_amp**2",
       "Total mechanical energy of a mass–spring oscillator of amplitude A.", output="W"),
    Eq("shm_velocity_at_x", "SHM", "kinematics",
       "v = omega*sqrt(A_amp**2 - x**2)",
       "Speed of a simple-harmonic oscillator at displacement x from the mean position."),

    # ── Gravitation ───────────────────────────────────────────────────────────
    Eq("gravitation_surface_g", "gravitation", "field",
       "g0 = G*M/R_E**2",
       "Acceleration due to gravity at the surface of a planet of mass M and radius R_E."),
    Eq("gravitation_g_at_height", "gravitation", "field",
       "g = g0*R_E**2/(R_E + h)**2",
       "Gravitational acceleration at height h above the surface, from the surface value g0.",
       output="g"),
    Eq("gravitation_orbital_velocity", "gravitation", "orbit",
       "v_orb = sqrt(G*M/r)",
       "Orbital speed of a satellite in a circular orbit of radius r around mass M."),
    Eq("gravitation_escape_velocity", "gravitation", "orbit",
       "v_esc = sqrt(2*G*M/R_E)",
       "Escape velocity from the surface of a planet of mass M and radius R_E."),

    # ── Rotational ────────────────────────────────────────────────────────────
    Eq("rotation_angular_kinematics", "rotational_motion", "kinematics",
       "omega = omega0 + alpha*t",
       "Angular velocity under constant angular acceleration (rotational first equation of motion)."),
    Eq("rotation_inertia_rod", "rotational_motion", "inertia",
       "I_mom = (1/12)*m*L**2",
       "Moment of inertia of a uniform thin rod about its centre, perpendicular to its length."),
    Eq("rotation_inertia_ring", "rotational_motion", "inertia",
       "I_mom = m*r**2",
       "Moment of inertia of a ring / hoop about its central axis."),
    Eq("rotation_torque_lever", "rotational_motion", "dynamics",
       "tau = F*r*sin(theta)",
       "Torque produced by a force F applied at distance r, at angle theta to the lever arm."),

    # ── Fluids ────────────────────────────────────────────────────────────────
    Eq("fluid_continuity", "fluid_mechanics", "flow",
       "A1*v1 = A2*v2",
       "Equation of continuity: the volume flow rate is the same at two cross-sections of a pipe.",
       output="v2"),
    Eq("fluid_flow_rate", "fluid_mechanics", "flow",
       "Q_flow = A*v",
       "Volume flow rate as cross-sectional area times flow speed."),
    Eq("fluid_stokes_drag", "fluid_mechanics", "viscosity",
       "F = 6*pi*eta_visc*r*v",
       "Stokes' viscous drag on a small sphere of radius r moving at speed v through a fluid.",
       output="F"),
    Eq("fluid_terminal_velocity", "fluid_mechanics", "viscosity",
       "v_term = 2*r**2*rho*g/(9*eta_visc)",
       "Terminal velocity of a small sphere falling through a viscous fluid."),

    # ── Thermodynamics / kinetic theory ───────────────────────────────────────
    Eq("thermo_mayer_relation", "thermodynamics", "specific_heats",
       "Cp = Cv + R_gas",
       "Mayer's relation between the molar specific heats of an ideal gas.", output="Cp"),
    Eq("thermo_adiabatic_index", "thermodynamics", "specific_heats",
       "gamma_ad = Cp/Cv",
       "Adiabatic index as the ratio of the molar specific heats."),
    Eq("thermo_internal_energy", "thermodynamics", "internal_energy",
       "DeltaU = n*Cv*DeltaT",
       "Change in internal energy of an ideal gas from its temperature change."),
    Eq("kinetic_rms_speed", "thermodynamics", "kinetic_theory",
       "v_rms = sqrt(3*k_B*T/m)",
       "Root-mean-square speed of a gas molecule of mass m at temperature T."),
    Eq("kinetic_mean_energy", "thermodynamics", "kinetic_theory",
       "K = (3/2)*k_B*T",
       "Average translational kinetic energy of a gas molecule at temperature T.", output="K"),

    # ── Capacitors / current electricity ──────────────────────────────────────
    Eq("capacitor_dielectric", "capacitors", "geometry",
       "C = K_diel*epsilon_0*A/d",
       "Capacitance of a parallel-plate capacitor filled with a dielectric of constant K_diel."),
    Eq("capacitor_parallel_three", "capacitors", "combination",
       "C = C1 + C2 + C3",
       "Equivalent capacitance of three capacitors in parallel.", output="C"),
    Eq("current_resistors_series_three", "current_electricity", "combination",
       "R = R1 + R2 + R3",
       "Equivalent resistance of three resistors in series.", output="R"),
    Eq("current_resistors_parallel_three", "current_electricity", "combination",
       "1/R = 1/R1 + 1/R2 + 1/R3",
       "Equivalent resistance of three resistors in parallel (reciprocal sum).", output="R"),
    Eq("current_terminal_voltage", "current_electricity", "emf",
       "V = emf - I*r_int",
       "Terminal voltage of a cell of emf with internal resistance carrying current I.",
       output="V"),
    Eq("current_drift_velocity", "current_electricity", "microscopic",
       "I = n_e*A*e_charge*v_d",
       "Current in terms of free-electron number density, area, charge and drift velocity.",
       output="I"),

    # ── Electrostatics ────────────────────────────────────────────────────────
    Eq("electrostatics_potential_point", "electrostatics", "potential",
       "V = k_e*q/r",
       "Electric potential at distance r from a point charge q.", output="V"),
    Eq("electrostatics_pe_two_charges", "electrostatics", "energy",
       "U = k_e*q1*q2/r",
       "Electrostatic potential energy of two point charges separated by r."),

    # ── Magnetism / EM induction / AC ─────────────────────────────────────────
    Eq("magnetism_field_solenoid", "magnetism", "field",
       "B = mu_0*n_e*I",
       "Magnetic field inside a long solenoid (n_e = turns per unit length) carrying current I.",
       output="B", skip_dim_check=True),
    Eq("inductor_energy", "electromagnetic_induction", "energy",
       "U = (1/2)*L_ind*I**2",
       "Energy stored in the magnetic field of an inductor carrying current I."),
    Eq("inductor_self_emf", "electromagnetic_induction", "inductance",
       "emf = L_ind*I/t",
       "Self-induced emf from the rate of change of current in an inductor.", output="emf"),
    Eq("ac_average_power", "AC_circuits", "power",
       "P_pow = V*I*cos(theta)",
       "Average power in an AC circuit (theta is the phase angle; cos theta is the power factor)."),
    Eq("ac_resonance_frequency", "AC_circuits", "resonance",
       "f = 1/(2*pi*sqrt(L_ind*C))",
       "Resonant frequency of a series LCR circuit.", output="f"),

    # ── Ray / wave optics ─────────────────────────────────────────────────────
    Eq("optics_mirror_formula", "ray_optics", "mirror",
       "1/f_lens = 1/v_img + 1/u_obj",
       "Mirror formula relating focal length, image distance and object distance.",
       output="f_lens"),
    Eq("optics_snell_law", "ray_optics", "refraction",
       "n1*sin(theta1) = n2*sin(theta2)",
       "Snell's law of refraction at the boundary between two media.", output="theta2"),
    Eq("optics_critical_angle", "ray_optics", "refraction",
       "sin(theta_c) = 1/n_ref",
       "Critical angle for total internal reflection from a medium of refractive index n_ref.",
       output="theta_c"),

    # ── Modern physics ────────────────────────────────────────────────────────
    Eq("modern_mass_energy", "modern_physics", "relativity",
       "W = m*c**2",
       "Einstein's mass–energy equivalence: rest energy of a mass m.", output="W"),
    Eq("modern_threshold_frequency", "modern_physics", "photoelectric",
       "W0 = h_planck*f0_th",
       "Work function in terms of the threshold frequency for photoemission.", output="W0"),
    Eq("modern_stopping_potential", "modern_physics", "photoelectric",
       "Kmax = e_charge*V_stop",
       "Maximum photoelectron kinetic energy from the stopping potential.", output="Kmax"),

    # ── Waves / sound ─────────────────────────────────────────────────────────
    Eq("wave_string_fundamental", "waves", "strings",
       "f = v_wave/(2*L)",
       "Fundamental frequency of a stretched string of length L.", output="f"),
]
