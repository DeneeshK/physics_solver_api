#!/usr/bin/env python3
"""
scripts/build_complete_exemplars.py
Generates scripts/rag_text_exemplars.json containing ALL 182 hand-authored
concept-level rag_texts.

After running this, the existing scripts/apply_exemplars_only.py will pick
up every one of them — no LLM batch generator needed, no Groq rate-limit
exposure.
"""
import json
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Notes on graph-content quirks I worked around:
#
# Several equations are stored with variable schemes that don't fully match
# standard physics notation (the suffix-disambiguation scheme presses kinematic
# letters into unrelated roles):
#   - kinematics_relative_velocity: 'v = u - a' — a is meant as 2nd-body velocity
#   - momentum_collisions_*: a, s used as 2nd-body final velocities
#   - sound_beat_frequency: 'f = u - v' — u, v are the two frequencies
#   - sound_doppler_*: 'f' on both sides (LHS is observed, RHS is source)
#   - thermodynamics_first_law: 'Q = DeltaV + W' — DeltaV here means ΔU
#   - thermodynamics_carnot_efficiency: 'eta = 1 - t/T' — t, T are T_cold, T_hot
#   - thermodynamics_isothermal_work: log() should be log(Vf/Vi); single V
#   - modern_physics_rydberg_formula: R_g used, but should be R_Rydberg
#   - nuclear_physics_binding_energy / q_value: DeltaV used, but should be Δm
#   - current_electricity_resistivity_temp: R_e on both sides — meant as R(T)/R0
#   - work_energy_power_efficiency: P/W — should be P_out/P_in
#
# Per the user's request, the rag_text describes the equation's INTENDED
# concept faithfully — the LLM picks the right node by concept; SymPy
# execution may need attention separately in v7.3 (graph-content cleanup).
# ─────────────────────────────────────────────────────────────────────────────

EXEMPLARS: dict[str, dict] = {}

def add(eid: str, concept_name: str, rag_text: str) -> None:
    EXEMPLARS[eid] = {"concept_name": concept_name, "rag_text": rag_text.strip()}

# ═════════════════════════════════════════════════════════════════════════════
# CORE 16 — same as v7.1 (kept here so this file is the single source of truth)
# ═════════════════════════════════════════════════════════════════════════════

add("laws_of_motion_newton_second_law",
    "Newton's Second Law of Motion",
    "Newton's Second Law of Motion: the foundational dynamics relationship. Net force on a body equals its mass times its acceleration. This is the universal force–motion link — applicable to any body undergoing acceleration in any setting, whether the cause of the force is contact, friction, tension, applied push, or anything else. The force here is the net force; the equation does not specify its origin. Use this whenever the question asks about the relationship between a body's motion and the force producing it. Mass and acceleration may not appear directly in the question — mass can come from density times volume, acceleration can come from kinematic relations among initial velocity, final velocity, displacement, or time. Conceptually distinct from equations that describe a specific physical mechanism: buoyancy (Archimedes), gravitational pull between bodies (Newton's gravitation), electrostatic attraction (Coulomb), spring restoring force (Hooke). Those compute particular forces of identified physical origin; F = m*a connects net force to motion.")

add("fluid_mechanics_buoyant_force",
    "Archimedes' Principle (Buoyant Force)",
    "Archimedes' Principle for the buoyant force on an immersed body. A body in contact with a fluid medium experiences an upward force equal to the weight of fluid it displaces. This equation describes a specific physical mechanism that exists only when a body is in a fluid — submerged in water, floating on liquid, suspended in air treated as a fluid. The concept is fluid-pushing-back-on-body, not the body's own dynamics. The density is the fluid's density and the volume is the body's submerged volume. The question's story must include a fluid medium: 'submerged in water', 'floating in oil', 'displaces fluid', 'in a liquid'. If the question concerns a body in motion outside any fluid context, this is the wrong concept — density and volume in the data are there to give the body's mass for use in dynamics, not its buoyancy. Conceptually distinct from F = m*a (general dynamics), weight (mg), and from hydrostatic pressure (P = rho*g*h, which is fluid pressure at depth, not force on a body).")

add("fluid_mechanics_hydrostatic_pressure",
    "Hydrostatic Pressure at Depth",
    "Hydrostatic pressure in a static fluid column. The pressure at a depth h below the free surface of a fluid at rest equals the fluid's density times g times h. This concept describes pressure within a fluid at rest, not force on a body. Apply when the question asks about pressure at a given depth, the pressure difference between two depths, the additional pressure from a fluid column, or the gauge pressure at the bottom of a tank. The h here is the vertical depth below the free surface — a static configuration, not a displacement of motion. Conceptually distinct from the buoyant force (which is the net upward force on a body in fluid) and from atmospheric/applied pressures that are added on top of this gauge pressure. Used when the question's story involves a fluid at rest and the variable of interest is pressure (Pa) rather than a force on a body (N).")

add("kinematics_v2_u2_2as",
    "Time-Free Kinematic Relation",
    "Kinematic relation for uniformly accelerated 1D motion when time is not needed or known. Links the change in velocity squared directly to acceleration and displacement. Apply whenever a body undergoes constant acceleration and any three of {initial velocity, final velocity, acceleration, displacement} are accessible to find the fourth, with time unknown or irrelevant. Covers free fall (acceleration = g, downward), braking to a stop (negative acceleration), launching upward and decelerating under gravity, a vehicle accelerating along a straight road. The displacement is the body's total travel along its direction of motion; if the question describes a height risen or fallen, or a distance traveled, that quantity IS the displacement — same physical thing, motion context. A member of the constant-acceleration kinematic family; pairs with v = u + a*t (time-known) and s = u*t + (1/2)*a*t^2 (final velocity unknown). Use this one specifically when time does not appear in either the givens or the unknowns.")

add("kinematics_v_u_at",
    "Time-Velocity Kinematic Relation",
    "Kinematic relation linking initial velocity, final velocity, acceleration, and time for uniformly accelerated 1D motion — without involving displacement. Apply when a body undergoes constant acceleration over a known time interval and any three of {initial velocity, final velocity, acceleration, time} are accessible to find the fourth. Use this when time IS a relevant variable in the problem (given or asked), and displacement is unknown or irrelevant. Common scenarios: 'after t seconds, what is the velocity', 'how long to reach a final speed', 'a body decelerates at constant rate — time to stop'. Same kinematic family as v^2 = u^2 + 2*a*s (used when time is absent) and s = u*t + (1/2)*a*t^2 (used when final velocity is absent). Choose this when time is in scope.")

add("electrostatics_coulomb_law",
    "Coulomb's Law (Electrostatic Force Between Point Charges)",
    "Coulomb's Law: the electrostatic force between two point charges. Two charges at a distance r exert an electrostatic force on each other proportional to the product of their charges and inversely proportional to the square of their separation. Apply when the question concerns the electrical interaction force between charged bodies treated as points — two charges, two charged particles, two charged spheres at separation r. The concept is electric charges attract or repel each other across space; the force here is specifically the electrostatic Coulomb force, distinct from gravitational attraction (same inverse-square form but different mechanism), from net force in F = m*a (this is a specific cause of force), and from electric force on a charge in a field (F = q*E, which already presupposes the field's existence). The constant epsilon_0 is the permittivity of free space, present because the force is computed in vacuum (or air, as approximation). Used when the question involves two charges and their mutual force.")

add("gravitation_universal_law",
    "Newton's Law of Universal Gravitation",
    "Newton's Law of Universal Gravitation: the gravitational force between two masses. Two point masses (or spherically symmetric masses treated as points) exert a mutual gravitational attraction proportional to the product of their masses and inversely proportional to the square of their separation. Apply when the question concerns gravitational attraction between bodies at significant scales — planet and satellite, Earth and Moon, Sun and Earth, two astronomical bodies. The concept is mass attracts mass across space; uses G, the universal gravitational constant, NOT g (the local gravitational acceleration at Earth's surface). Distinct from weight (W = m*g, which uses local g for surface gravity) and from F = m*a (general dynamics). Distinct from Coulomb's law in concept (mass-mass attraction, not charge-charge), though the equation has the same inverse-square form. Choose this when the scenario involves astronomical bodies or two masses at non-negligible separation; choose W = m*g for objects near Earth's surface.")

add("current_electricity_power_electric",
    "Electrical Power (Volt-Ampere Form)",
    "Electrical power dissipated or delivered in a circuit element. Power equals voltage across the element times current through it. This is the most general form of electrical power and applies to any circuit element: resistor, bulb, motor, source. Apply when the question gives voltage and current (or makes either of them findable from other circuit data) and asks for power dissipated, power consumed, power delivered, or rate of energy use in a circuit. Conceptually power-as-V*I; alternative forms (P = I^2*R, P = V^2/R) are derived by substituting Ohm's law and apply specifically when the element is purely resistive. Distinct from mechanical power (P = F*v, P = W/t), from radiative power, and from intensity. Used when the question is about a circuit and the unknown is power (W or J/s).")

add("current_electricity_ohms_law",
    "Ohm's Law",
    "Ohm's Law: the relationship between voltage, current, and resistance for an ohmic conductor. The voltage across a resistor equals its resistance times the current through it. Apply when the question concerns an electrical circuit with a resistor (or ohmic conductor) and any two of {voltage, current, resistance} are accessible to find the third. The concept is voltage-drives-current-through-resistance for ohmic materials — the linear V-I relationship that defines resistance. The resistance symbol in this graph is R_e (electrical resistance) to distinguish from other R uses (radius, gas constant); the underlying physics is the same. Used when the scenario involves a resistor and a circuit, regardless of whether the resistor is given the symbol R, R_e, or named explicitly. Distinct from electrical power (P = V*I) and from the EMF of a source (EMF = V + I*r_internal), which deals with a source's terminal voltage.")

add("modern_physics_photon_energy",
    "Photon Energy (Planck-Einstein Relation)",
    "Photon energy: the energy of a single photon equals Planck's constant times its frequency. This is the foundational quantum relation between particle (energy) and wave (frequency) descriptions of light. Apply when the question concerns light, electromagnetic radiation, or a photon, and the unknown is the photon's energy (or, by rearrangement, its frequency or Planck's constant in a problem testing the relation). The frequency symbol used in modern_physics is nu, not f — same physical quantity (frequency), different conventional label in this domain. If the question gives a wavelength instead of a frequency, pair this with the wave relation c = nu*lambda (where lambda is wavelength) to bridge. Distinct from the kinetic energy of a particle (K = (1/2)*m*v^2) and from photoelectric work-function energy (which uses this relation but adds the work function). Used in photoelectric effect, photon-matter interaction, atomic transition, blackbody radiation contexts.")

add("work_energy_power_gravitational_potential_mgh",
    "Gravitational Potential Energy (Near Earth's Surface)",
    "Gravitational potential energy of a body at height h above a reference, near Earth's surface (uniform gravitational field approximation). U = m*g*h. Apply when the question concerns the stored energy of a raised body in a roughly-uniform gravitational field — a block on a shelf, a ball at the top of a building, water in an elevated tank, a pendulum at its high point. The h here is height above the chosen reference (often ground or starting level), a static configuration of the body, distinct from displacement-during-motion. The concept is gravitational potential energy of position — stored, not kinetic. Pairs with kinetic energy (K = (1/2)*m*v^2) in energy-conservation problems. Distinct from the full Newton-gravitation potential energy (U = -G*M*m/r, which applies on astronomical scales). Used when the question is about energy of position near Earth's surface and asks for U, m, g, or h.")

add("work_energy_power_kinetic_energy",
    "Kinetic Energy of a Translating Body",
    "Kinetic energy of a body in translational motion: K = (1/2)*m*v^2. The energy a body possesses by virtue of its motion. Apply when the question concerns the energy a moving body carries, the energy it loses on stopping, the energy it gains on accelerating from rest, or energy conservation in a system with translating bodies. The v here is the body's speed (magnitude of velocity). The concept is energy-of-motion. Pairs with gravitational PE (U = m*g*h) and with the work-energy theorem (W_net = DeltaK) in conservation problems. Distinct from rotational kinetic energy (K = (1/2)*I*omega^2, used when the body's spin matters) and from momentum (p = m*v, which is a vector quantity, not energy). Used when the question is about energy and a body is moving translationally.")

add("magnetism_lorentz_force",
    "Lorentz Magnetic Force on a Moving Charge",
    "Magnetic force on a moving charge: a charge q moving with velocity v through a magnetic field B experiences a force perpendicular to both, with magnitude q*v*B*sin(theta) where theta is the angle between velocity and field. Apply when the question concerns a charged particle moving through a magnetic field — electron in a CRT, proton in a cyclotron, charged droplet in a Millikan-style setup, particle in a mass spectrometer. The concept is moving-charge-in-magnetic-field; the force is zero when motion is parallel to field (sin(theta) = 0), maximum when perpendicular. Distinct from the electrostatic Coulomb force (which acts on stationary charges, no velocity needed) and from the force on a current-carrying wire (F = B*I*L, which is the integrated version of this same physics over a wire's worth of moving charges). Used when the scenario involves a single moving charged particle in a magnetic field.")

add("circular_motion_centripetal_force",
    "Centripetal Force for Uniform Circular Motion",
    "Centripetal force required for uniform circular motion. A body moving in a circle of radius r at speed v experiences an inward (center-pointing) net force of magnitude m*v^2/r. This equation gives the net force REQUIRED to keep the body on its circular path; the actual physical source of that force (tension, gravity, friction, normal force, electrostatic attraction, etc.) is supplied by the scenario. Apply when the question concerns circular motion — a car on a curve, a satellite in orbit, a ball on a string in a circle, an electron in a Bohr orbit. The concept is the kinematic requirement for circular motion; the equation does not invent a force, it tells you how strong the inward force MUST be. Pairs with whatever provides the force in the scenario — set m*v^2/r equal to the appropriate force (mg, T, qvB, GMm/r^2, etc.) to solve. Distinct from F = m*a (which is the same physics, applied to circular motion's inward acceleration v^2/r, but written in centripetal-specific form here for clarity).")

add("shm_period_spring",
    "Period of a Spring-Mass Oscillator",
    "Period of simple harmonic motion for a mass on a spring. The time for one complete oscillation is 2*pi*sqrt(m/k), depending only on the mass and spring constant — independent of the amplitude of oscillation. Apply when the question concerns a mass undergoing SHM on a spring (or any restoring force linear in displacement) and the unknown is the period, frequency (reciprocal of period), angular frequency, mass, or spring constant. The concept is the natural oscillation period of a linear oscillator. Distinct from the pendulum period (T = 2*pi*sqrt(L/g), which depends on length and gravity, not mass). Period independence from amplitude is a signature property of true SHM — if amplitude affects the period in the scenario, this is anharmonic motion and a different concept applies.")

add("ray_optics_lens_formula",
    "Thin Lens Formula (Object-Image-Focal-Length Relation)",
    "Thin lens formula relating object distance, image distance, and focal length for a thin lens. 1/f_lens = 1/v_i - 1/u_o, using the Cartesian sign convention. Apply when the question concerns image formation by a thin lens (converging or diverging) and the unknown is the image distance, the object distance, the focal length, or the type/position of the image. The concept is lens imaging geometry. The symbols u_o (object distance) and v_i (image distance) use the suffix scheme to disambiguate from kinematic u, v; signs follow the convention (distances measured from lens, with light direction as positive). Distinct from the mirror formula (1/f = 1/v + 1/u, sign difference reflects the different geometry) and from the lens-maker's equation (which gives focal length from the lens's own physical properties, not from imaging geometry). Used when the question is about how a lens images an object.")

# ═════════════════════════════════════════════════════════════════════════════
# AC_circuits (7)
# ═════════════════════════════════════════════════════════════════════════════

add("ac_circuits_capacitive_reactance",
    "Capacitive Reactance in an AC Circuit",
    "Capacitive reactance: the effective opposition a capacitor offers to alternating current at angular frequency omega. X_C = 1/(omega*C_cap). Apply when the question concerns a capacitor in an AC circuit and asks for the reactance, or relates voltage and current across a capacitor in steady-state AC. The concept is frequency-dependent opposition by a capacitor — high at low frequencies (DC blocks), low at high frequencies. Distinct from resistance (frequency-independent), from inductive reactance (X_L = omega*L, which rises with frequency), and from impedance (Z, which combines all three in an LCR circuit). Used in single-element capacitor circuits and as one term inside the LCR impedance.")

add("ac_circuits_impedance_lcr",
    "Total Impedance of a Series LCR Circuit",
    "Impedance of a series LCR (inductor-capacitor-resistor) circuit at angular frequency omega: Z = sqrt(R^2 + (X_L - X_C)^2). The total effective opposition to AC current, combining resistive and reactive elements vectorially. Apply when the question concerns a series AC circuit with R, L, and C together and the unknown is the impedance, the peak current, or the phase angle. The concept is vector combination of resistance and net reactance. Distinct from individual reactances (X_L, X_C) and from pure resistance. At resonance (X_L = X_C), impedance reduces to R — see resonance frequency formula. Used in AC circuit analysis when more than one type of element is present.")

add("ac_circuits_inductive_reactance",
    "Inductive Reactance in an AC Circuit",
    "Inductive reactance: the effective opposition an inductor offers to alternating current at angular frequency omega. X_L = omega*L_ind. Apply when the question concerns an inductor in an AC circuit and asks for reactance, or relates voltage and current across an inductor in steady-state AC. The concept is frequency-dependent opposition by an inductor — zero at DC (short circuit), rises linearly with frequency. Distinct from resistance (frequency-independent), from capacitive reactance (X_C = 1/(omega*C), which falls with frequency), and from impedance (which combines all three in an LCR circuit). Used in single-element inductor circuits and as one term inside the LCR impedance.")

add("ac_circuits_power_factor",
    "AC Power with Power Factor",
    "Average AC power delivered to a circuit element with non-zero phase angle phi between voltage and current. P = V*I*cos(phi), where V and I are RMS values. The cos(phi) factor is called the power factor. Apply when the question concerns real (average) power dissipated in an AC circuit and the voltage and current are not in phase — e.g. a circuit containing inductive or capacitive elements alongside resistance. The concept is that only the in-phase component of current dissipates real power; the out-of-phase (reactive) component carries energy back and forth without net dissipation. Distinct from pure resistive AC power (where cos(phi) = 1, reducing to P = V*I) and from instantaneous power. Used in AC circuits with mixed components.")

add("ac_circuits_resonance_frequency",
    "Resonance Frequency of a Series LCR Circuit",
    "Resonance angular frequency of a series LCR circuit: omega = 1/sqrt(L*C). At this frequency, inductive and capacitive reactances cancel exactly (X_L = X_C), the circuit's impedance is at its minimum (pure resistance R), and the current amplitude is maximized for a given source voltage. Apply when the question concerns LCR resonance — finding the frequency at which the circuit resonates, peak current, or maximum response. The concept is the natural frequency where energy oscillates losslessly between L and C. Distinct from the impedance formula at arbitrary frequency (which uses X_L, X_C generally). Used in tuned circuits, radio receivers, filter design.")

add("ac_circuits_rms_current",
    "RMS Value of Sinusoidal AC Current",
    "Root-mean-square (RMS) value of a sinusoidal AC current: I_rms = I_0 / sqrt(2), where I_0 is the peak amplitude. The concept of RMS current gives the equivalent DC current that would dissipate the same average power in a resistor. Apply when the question gives a peak current and asks for the RMS value (or vice versa), or when comparing AC and DC power capacity. Distinct from the average current of a sinusoid (which is zero over a full cycle) and from the rectified average. The factor sqrt(2) comes from time-averaging the square of the sinusoid. Used wherever AC currents are quoted in 'effective' terms.")

add("ac_circuits_rms_voltage",
    "RMS Value of Sinusoidal AC Voltage",
    "Root-mean-square (RMS) value of a sinusoidal AC voltage: V_rms = V_0 / sqrt(2), where V_0 is the peak amplitude. RMS voltage is the equivalent DC voltage that would deliver the same average power across a resistor. Apply when the question gives a peak voltage and asks for RMS (or vice versa), or when AC voltages are quoted as 'mains' or 'effective' values (e.g. household 220 V is RMS, not peak). Distinct from the average voltage over a cycle (zero for a pure sinusoid). The sqrt(2) factor comes from time-averaging the square of the sinusoid. Used wherever AC voltages are quoted in standard 'effective' form.")

# ═════════════════════════════════════════════════════════════════════════════
# SHM (6)
# ═════════════════════════════════════════════════════════════════════════════

add("shm_acceleration",
    "Acceleration in Simple Harmonic Motion",
    "Acceleration of a body in simple harmonic motion as a function of displacement from equilibrium: a = -omega^2 * x. The negative sign indicates the acceleration always points toward equilibrium (restoring), and the magnitude is proportional to displacement. Apply when the question concerns SHM and relates the acceleration of the oscillator to its instantaneous position. The concept is restoring-force-proportional-to-displacement, the defining property of SHM. Distinct from kinematic acceleration in linear motion (constant a) and from circular motion acceleration (always v^2/r, constant magnitude). Used to identify motions as SHM and to compute peak acceleration at amplitude.")

add("shm_angular_frequency",
    "Angular Frequency of a Spring-Mass Oscillator",
    "Angular frequency of SHM for a mass on a spring: omega = sqrt(k/m). Apply when the question concerns the rate of oscillation of a spring-mass system and the unknown is omega (or k or m, by rearrangement). The angular frequency relates to the period (T = 2*pi/omega) and ordinary frequency (f = omega/(2*pi)). The concept is the natural rate of oscillation, set entirely by the restoring stiffness and the inertia. Distinct from the angular frequency of a pendulum (omega = sqrt(g/L)) and from forced-oscillation driving frequency. Used in any spring-mass SHM problem.")

add("shm_displacement",
    "Displacement in Simple Harmonic Motion (Sinusoidal Form)",
    "Time-domain displacement of an SHM oscillator: x = A*sin(omega*t + phi), where A is amplitude, omega is angular frequency, t is time, and phi is the initial phase. Apply when the question gives or asks for the position of an oscillator at a specific time, identifying amplitude, phase, period, or instantaneous displacement. The concept is sinusoidal time-evolution of position — the universal solution form for an SHM oscillator. Distinct from the time-domain velocity (x derivative, cosine form) and from displacement in uniform motion (linear in t). Used to read off and manipulate SHM time-series.")

add("shm_energy_total",
    "Total Mechanical Energy of an SHM Oscillator",
    "Total mechanical energy of a spring-mass SHM oscillator: E = (1/2)*k*A^2, where k is the spring constant and A is the amplitude. The energy is conserved across the oscillation, exchanging between kinetic and potential forms but summing to this constant. Apply when the question concerns the total energy of an oscillator, peak kinetic energy (which equals total E at x=0), or peak potential energy (equals total E at x=A). The concept is amplitude-determined energy of an oscillator. Distinct from the kinetic energy at a specific position (which varies with x) and from energy of damped or driven oscillators (which decay or are sourced). Used in oscillator energy-conservation problems.")

add("shm_period_pendulum",
    "Period of a Simple Pendulum",
    "Period of a simple pendulum: T = 2*pi*sqrt(L/g), where L is the length and g is local gravitational acceleration. The period depends only on length and gravity — independent of mass and (for small angles) amplitude. Apply when the question concerns a pendulum's period, frequency, length, or local g (e.g. 'a clock pendulum loses time when moved to the Moon'). The concept is the natural oscillation period of a gravity-driven pendulum. Distinct from the spring-mass period (T = 2*pi*sqrt(m/k), depends on mass). Approximation: valid for small swing angles where sin(theta) ≈ theta; large-angle pendulum is anharmonic.")

add("shm_velocity",
    "Velocity in Simple Harmonic Motion (Position-Dependent Form)",
    "Instantaneous speed of an SHM oscillator at displacement x from equilibrium: v = omega*sqrt(A^2 - x^2). Apply when the question asks for the speed of an SHM oscillator at a specific position (not a specific time) — peak speed at x=0 is omega*A, speed is zero at x=±A. The concept is position-determined kinetic state of an oscillator. Distinct from the time-domain velocity (omega*A*cos(omega*t + phi), which depends on time) and from the kinematic v = u + a*t (linear, not oscillatory). Used to bridge position and speed within one SHM cycle without needing time.")

# ═════════════════════════════════════════════════════════════════════════════
# capacitors (7)
# ═════════════════════════════════════════════════════════════════════════════

add("capacitors_capacitance_definition",
    "Capacitance (Definition: Charge per Voltage)",
    "Capacitance: the ability of a body or arrangement to store electric charge per unit voltage. C = q/V. Apply whenever the question relates the charge stored on a capacitor to the voltage across it, or asks for capacitance from charge-voltage data. This is the DEFINING relationship of capacitance — applies to any capacitor regardless of geometry. Distinct from geometric formulas for specific capacitor shapes (parallel-plate C = epsilon_0*A/d, with dielectric C = k*epsilon_0*A/d), which compute C from physical properties. Used as the foundational link between charge, voltage, and capacitance in any capacitor problem.")

add("capacitors_electric_field_plate",
    "Electric Field Between Parallel Capacitor Plates",
    "Electric field strength in the uniform region between the plates of a parallel-plate capacitor: E = V/d, where V is the voltage across the plates and d is the plate separation. Apply when the question concerns the field strength inside a parallel-plate capacitor, or relates field, voltage, and gap. The concept is uniform-field-from-voltage-over-distance in a parallel-plate geometry. Distinct from the field of a point charge (E = q/(4*pi*epsilon_0*r^2), inverse-square) and from non-uniform fields in irregular geometries. Used when computing the field a charged particle experiences inside a parallel-plate region.")

add("capacitors_energy_capacitor",
    "Energy Stored in a Charged Capacitor",
    "Energy stored in a charged capacitor: U = (1/2)*C*V^2, where C is capacitance and V is the voltage across it. Equivalent forms (by Q = CV) are U = (1/2)*Q*V and U = Q^2/(2*C). Apply when the question concerns the energy stored in a capacitor, energy released on discharge, or energy redistribution among capacitors. The concept is electrostatic energy stored in the electric field between the plates. Distinct from the energy stored in an inductor (U = (1/2)*L*I^2, which uses current) and from instantaneous power dissipation in a circuit. Used in capacitor energy and discharge problems.")

add("capacitors_parallel_equivalent",
    "Equivalent Capacitance of Capacitors in Parallel",
    "Equivalent capacitance of n identical capacitors connected in parallel: C_eq = n*C, where C is the capacitance of one. (More generally, capacitances in parallel add directly: C_eq = C_1 + C_2 + ...) Apply when the question concerns combining capacitors in parallel, asking for the equivalent capacitance or designing a target capacitance with available units. The concept is voltage-shared, charge-summed combination. Distinct from series combination (where 1/C_eq = sum of 1/C, which DECREASES the equivalent capacitance) and from resistor combinations (parallel resistors use 1/R_eq = sum of 1/R — opposite rule from capacitors).")

add("capacitors_parallel_plate",
    "Capacitance of a Parallel-Plate Capacitor in Vacuum",
    "Capacitance of a parallel-plate capacitor with plate area A and separation d, in vacuum (or air, as approximation): C = epsilon_0*A/d. Apply when the question gives plate geometry and asks for the capacitance, or relates geometry to capacitance. The concept is geometric capacitance — set entirely by area, separation, and the permittivity of the medium. Distinct from C = q/V (the definitional relation independent of geometry) and from C = k*epsilon_0*A/d (the same geometry with a dielectric, larger by a factor k). Used in computing capacitor values from device geometry.")

add("capacitors_series_equivalent",
    "Equivalent Capacitance of Capacitors in Series",
    "Equivalent capacitance of n identical capacitors in series: C_eq = C/n. (More generally, in series 1/C_eq = sum of 1/C_i, which always gives a smaller equivalent than any individual capacitor.) Apply when the question concerns combining capacitors in series, asking for the equivalent capacitance, voltage division across them, or charge (same charge passes through every series capacitor). The concept is charge-shared, voltage-summed combination — opposite rule from parallel. Distinct from parallel combination (which adds capacitances directly) and from resistor combinations (resistors in series add, opposite from capacitors).")

add("capacitors_with_dielectric",
    "Parallel-Plate Capacitance with a Dielectric",
    "Capacitance of a parallel-plate capacitor filled with a dielectric of relative permittivity k: C = k*epsilon_0*A/d. Apply when the question concerns inserting a dielectric, computing the new capacitance, or comparing with the air-filled value (which is larger by the factor k). The concept is dielectric enhancement — the dielectric polarizes and reduces the effective field, allowing more charge at the same voltage. Distinct from the vacuum case (C = epsilon_0*A/d, no k) and from a capacitor partially filled with dielectric (geometric subcases). Used in problems involving dielectric insertion or comparison.")

# ═════════════════════════════════════════════════════════════════════════════
# circular_motion (5)
# ═════════════════════════════════════════════════════════════════════════════

add("circular_motion_angular_speed",
    "Angular Speed from Tangential Speed and Radius",
    "Angular speed of a body in circular motion in terms of its tangential (linear) speed v and radius r: omega = v/r. Apply when the question relates linear and angular speeds in circular motion, or converts between rotational and translational descriptions. The concept is the geometric link between how fast a body sweeps angle and how fast its position changes along the circumference. Distinct from angular speed defined as theta/t (which is the time-derivative form) and from frequency (f = omega/(2*pi)). Used in circular motion problems whenever both v and r appear.")

add("circular_motion_centripetal_acceleration",
    "Centripetal Acceleration in Uniform Circular Motion",
    "Centripetal acceleration of a body moving in a circle of radius r at tangential speed v: a = v^2/r, directed inward (toward the center). Apply when the question concerns the acceleration of a body in uniform circular motion — independent of cause, this is the kinematic requirement of circular trajectory. The concept is the inward acceleration any object on a circular path must have. Distinct from the centripetal FORCE (F = m*v^2/r, this acceleration times mass) and from tangential acceleration (which changes the speed, not the direction). Used when the question asks for the acceleration of an object in circular motion, regardless of what's providing the force.")

add("circular_motion_conical_pendulum",
    "Period of a Conical Pendulum",
    "Period of a conical pendulum (a mass swinging in a horizontal circle on a string at angle theta from vertical): T = 2*pi*sqrt(r/(g*tan(theta))), where r is the radius of the horizontal circle. Apply when the question concerns a conical pendulum and asks for the period, the angle, or the radius. The concept is uniform circular motion sustained by the horizontal component of string tension. Distinct from the simple pendulum (which oscillates in a plane, T = 2*pi*sqrt(L/g)) and from a body on a banked turn (similar geometry but rolling-friction-driven). Used in conical-pendulum / horizontal-circle problems with a tilted support.")

add("circular_motion_frequency_period",
    "Frequency–Period Reciprocal Relationship",
    "Frequency and period of any periodic motion are reciprocals: f = 1/T. Apply whenever the question gives one and asks for the other — period of a wave from its frequency, frequency of an oscillator from its period, etc. The concept is the basic kinematic reciprocity: number of cycles per second equals one divided by seconds per cycle. Distinct from angular frequency (omega = 2*pi*f, which expresses the same rate in radians per second). Used universally across waves, oscillations, circular motion, AC circuits, anything periodic.")

add("circular_motion_period_speed",
    "Period of Uniform Circular Motion from Speed and Radius",
    "Period of a body in uniform circular motion: T = 2*pi*r/v, where r is the circle's radius and v is the tangential speed. Apply when the question asks for the time to complete one revolution in circular motion and gives speed and radius (or asks for radius/speed given period). The concept is circumference divided by speed — geometric time-for-one-loop. Distinct from rotational period defined via angular velocity (T = 2*pi/omega) and from SHM period (which uses physical restoring properties, not geometric distance). Used in uniform circular motion problems.")

# ═════════════════════════════════════════════════════════════════════════════
# current_electricity (6)
# ═════════════════════════════════════════════════════════════════════════════

add("current_electricity_current_definition",
    "Electric Current as Charge per Unit Time",
    "Electric current, by definition: the rate of flow of charge. I = q/t. Apply whenever the question relates the total charge that has flowed through a conductor in a given time to the current, or asks for either quantity given the other two. The concept is the definitional link between current and charge flow. Distinct from drift velocity formulas (which decompose I in terms of microscopic carrier properties) and from Ohm's law (V = I*R, which links current to voltage and resistance). Used as the foundational definition of current in any circuit problem.")

add("current_electricity_drift_velocity",
    "Current from Carrier Drift Velocity (Microscopic Form)",
    "Microscopic expression for current in a conductor: I = n*q*A*v, where n is the number density of charge carriers, q is the charge per carrier, A is the conductor's cross-sectional area, and v is the drift velocity. Apply when the question concerns drift velocity, carrier density, or microscopic current physics. The concept is the macroscopic current expressed in terms of microscopic carrier motion. Distinct from the macroscopic definition I = q/t (no geometry, no carrier breakdown) and from Ohm's law. Used in conductor physics, semiconductor problems, or any question that probes the inside of the wire.")

add("current_electricity_joule_heating",
    "Joule Heating in a Resistor",
    "Heat dissipated in a resistor carrying current I for time t: Q = I^2*R*t. Apply when the question concerns the heat produced by a resistor or any ohmic conductor — heating coils, fuses, transmission losses. The concept is electrical energy dissipated as heat (power times time). Distinct from instantaneous power P = I^2*R (which is the rate, not the total) and from electrical work moving charge through a potential difference (which encompasses non-resistive contexts too). Used in heating, energy-loss, and thermal problems involving current flow.")

add("current_electricity_resistance_wire",
    "Resistance of a Wire from Resistivity, Length, and Area",
    "Resistance of a uniform wire: R = rho*L/A, where rho is the material's resistivity, L is the wire length, and A is its cross-sectional area. Apply when the question concerns wire geometry and asks for resistance, or compares resistances of wires of different shapes. The concept is geometric resistance — proportional to length, inversely proportional to area, with resistivity as the material-specific factor. Distinct from Ohm's law V=I*R (which relates voltage, current, resistance — doesn't tell you what R is) and from temperature-dependence formulas. Used in problems about wire design, resistance scaling, gauge effects.")

add("current_electricity_resistivity_temp",
    "Temperature Dependence of Resistance (Linear Model)",
    "Linear temperature-coefficient model for the resistance of a conductor: R(T) = R_0*(1 + alpha*DeltaT), where R_0 is the resistance at the reference temperature, alpha is the material's temperature coefficient, and DeltaT is the temperature change. Apply when the question concerns how a conductor's resistance changes with temperature — most metals rise with T, semiconductors fall. The concept is first-order linear temperature dependence of resistivity. Distinct from the basic resistivity formula (R = rho*L/A, geometry-driven, fixed T) and from non-linear thermistor behaviors. Used in thermal-resistance problems.")

add("current_electricity_terminal_voltage",
    "Terminal Voltage of a Source with Internal Resistance",
    "Terminal voltage across a real battery or source with EMF and internal resistance r, delivering current I: V = EMF - I*r. Apply when the question concerns the actual voltage across a battery's terminals when current is flowing (smaller than EMF), or when distinguishing open-circuit EMF from on-load terminal voltage. The concept is the voltage drop internal to the source. Distinct from the bare Ohm's law applied to an external resistor and from the open-circuit EMF (where I=0). Used in problems with non-ideal sources, internal resistance, or battery discharge.")

# ═════════════════════════════════════════════════════════════════════════════
# elasticity (6)
# ═════════════════════════════════════════════════════════════════════════════

add("elasticity_bulk_modulus",
    "Bulk Modulus (Volumetric Elasticity)",
    "Bulk modulus B of a material under uniform pressure change P producing volumetric strain: B = P/strain. Apply when the question concerns a material's resistance to volumetric compression — gases, liquids, solids under uniform pressure. The concept is volumetric stiffness. Distinct from Young's modulus (longitudinal stress over longitudinal strain, for stretching/compressing in one direction) and from shear modulus (for shear deformations). Used in compressibility problems, sound speed in solids, and bulk-elasticity contexts.")

add("elasticity_elastic_potential",
    "Elastic Potential Energy from Force and Extension",
    "Elastic potential energy stored when a force F has produced extension x: U = (1/2)*F*x. For a Hookean spring this equals (1/2)*k*x^2, and the two forms are equivalent. Apply when the question asks for the energy stored in a stretched (or compressed) elastic element. The concept is the work done in stretching the element, which becomes stored potential energy. Distinct from the kinetic energy of a moving body and from gravitational PE. Used in spring and elastic-deformation energy problems.")

add("elasticity_spring_hooke",
    "Hooke's Law for Spring Force",
    "Hooke's Law: the restoring force of an ideal spring is proportional to its displacement from natural length, F = k*x, where k is the spring constant. Apply when the question concerns a spring's force at a given extension or compression, the spring constant of a stretched spring, or the displacement under a given load. The concept is the linear restoring force, which is what makes the spring an SHM-producing system. Distinct from Newton's second law (which connects force to acceleration, not to displacement). Used in spring problems, SHM setups, and elastic-deformation analyses.")

add("elasticity_strain",
    "Longitudinal Strain (Fractional Length Change)",
    "Longitudinal strain: the fractional change in length of a body under stress, strain = DeltaL/L, dimensionless. Apply when the question concerns how much a material has stretched or compressed relative to its original length. The concept is the geometric measure of deformation. Distinct from stress (force per area, the cause) and from Young's modulus (stress/strain, the material's stiffness). Used as a fundamental quantity in elasticity problems; pairs with stress.")

add("elasticity_stress",
    "Stress (Force per Unit Area)",
    "Longitudinal stress on a body: the force applied per unit cross-sectional area, stress = F/A, units Pa. Apply when the question concerns the internal force-per-area in a stretched or compressed material — wire under tension, rod under compression. The concept is the cause-side of deformation, complementary to strain (the effect). Distinct from pressure (which is the same units but typically describes fluid pressure isotropically) and from Young's modulus. Used in elasticity problems as the load measure.")

add("elasticity_young_modulus",
    "Young's Modulus (Longitudinal Elasticity)",
    "Young's modulus Y of a material: the ratio of longitudinal stress to longitudinal strain, Y = stress/strain, a material-specific stiffness constant. Apply when the question relates how much a material deforms under a given load — e.g. how much a wire stretches under tension, what load causes a given strain. The concept is longitudinal stiffness, distinct from bulk modulus (volumetric stiffness) and shear modulus (shape-distortion stiffness). Larger Y means a stiffer material. Used in problems involving stretching wires, rods, beams, etc.")

# ═════════════════════════════════════════════════════════════════════════════
# electromagnetic_induction (6)
# ═════════════════════════════════════════════════════════════════════════════

add("electromagnetic_induction_energy_inductor",
    "Energy Stored in an Inductor's Magnetic Field",
    "Energy stored in an inductor carrying current I: U = (1/2)*L*I^2. Apply when the question concerns the magnetic energy stored in an inductor's field, energy released on collapse of the current, or LC circuit energy. The concept is magnetic-field energy storage. Distinct from capacitor energy (U = (1/2)*C*V^2, electric-field storage) and from resistive heat dissipation (which is energy lost, not stored). Used in inductor energy and LC oscillation problems.")

add("electromagnetic_induction_faraday_law",
    "Faraday's Law of Induction (EMF from Changing Flux)",
    "Faraday's Law: the EMF induced in a loop equals the negative time-rate of change of magnetic flux through it, EMF = -dPhi/dt. Apply when the question concerns electromagnetic induction — a loop in a changing magnetic field, a moving conductor, a transformer, a generator. The concept is changing-flux-induces-EMF, the foundational law of electromagnetic induction. The negative sign reflects Lenz's law (induced EMF opposes the change). Distinct from motional EMF (EMF = B*L*v, which is a special case for a conductor moving through a uniform field) and from self/mutual induction (which are special cases involving changing currents). Used in any induction problem.")

add("electromagnetic_induction_flux",
    "Magnetic Flux Through a Surface",
    "Magnetic flux Phi through a flat surface of area A in a uniform field B at angle theta to the surface normal: Phi = B*A*cos(theta), with units Wb. Apply when the question asks for the magnetic flux through a coil or loop, given field strength, area, and orientation. The concept is the geometric scalar 'how much field penetrates the surface'. Distinct from EMF (the time-derivative of flux) and from the field B itself (a vector quantity). Used as the foundational quantity in any Faraday's-law calculation.")

add("electromagnetic_induction_motional_emf",
    "Motional EMF of a Conductor Moving in a Magnetic Field",
    "EMF induced across a straight conductor of length L moving with velocity v perpendicular to a uniform magnetic field B: EMF = B*L*v. Apply when the question concerns a rod sliding on rails in a magnetic field, a wing of a moving aircraft, or any straight conductor moving across field lines. The concept is the motion-driven version of Faraday's law — changing flux due to the moving conductor sweeping area through the field. Distinct from the general Faraday EMF (a more general formulation) and from inductor self-EMF. Used in motional-EMF problems involving moving conductors.")

add("electromagnetic_induction_mutual_induction",
    "Mutual Induction Between Two Coils",
    "EMF induced in one coil due to a changing current in a neighboring coil: EMF = -M*dI/dt, where M is the mutual inductance between the coils. Apply when the question concerns transformers, coupled coils, or any setup where one coil's changing current induces voltage in another. The concept is current-in-one-coil drives flux in the other, and the changing flux induces EMF. Distinct from self-inductance (EMF in the SAME coil due to its own changing current) and from Faraday's law in general. Used in transformer and coupled-circuit problems.")

add("electromagnetic_induction_self_inductance",
    "Self-Induction in a Coil",
    "EMF induced in a coil due to a change in its own current: EMF = -L*dI/dt, where L is the coil's self-inductance. Apply when the question concerns the back-EMF of an inductor as its current changes — switching an inductor on or off, oscillations in an LC circuit, transients in an RL circuit. The concept is changing-current-in-a-coil induces opposing EMF in the same coil. Distinct from mutual inductance (between two coils) and from motional EMF (mechanical motion). Used in inductor transient and AC problems.")

# ═════════════════════════════════════════════════════════════════════════════
# electrostatics (6)
# ═════════════════════════════════════════════════════════════════════════════

add("electrostatics_dipole_moment",
    "Electric Dipole Moment",
    "Electric dipole moment magnitude: p = q*d, where q is the magnitude of each charge and d is the separation between them. The dipole moment is a vector pointing from the negative to the positive charge. Apply when the question concerns a pair of equal and opposite point charges separated by a small distance, asking for the dipole moment, or computing the field and torque a dipole experiences. The concept is the strength of charge separation. Distinct from individual point-charge fields or potentials. Used as the foundational quantity in dipole physics.")

add("electrostatics_dipole_torque",
    "Torque on an Electric Dipole in a Uniform Field",
    "Torque on an electric dipole p placed in a uniform external electric field E at angle theta to the field: tau = p*E*sin(theta). Apply when the question concerns a dipole in an electric field — its tendency to align with the field, the work done in rotating it. The concept is the dipole experiences a couple that tends to align it with the field. Maximum torque at theta=90 degrees, zero at theta=0 or 180 (aligned or anti-aligned). Distinct from net force on a dipole in a non-uniform field (which is a different calculation) and from torque on a current loop in a magnetic field (similar form, different physics). Used in dipole-in-field problems.")

add("electrostatics_electric_field_force",
    "Electric Field as Force per Unit Charge",
    "Electric field strength at a point, defined as the force a test charge would experience per unit of its charge: E = F/q. Apply when the question relates field and force on a charge — a charge in a known field experiences force F = q*E (rearranging), or the field at a point is found from the force on a test charge. The concept is the DEFINITION of the electric field. Distinct from the field of a point charge (which gives E from a SOURCE charge's geometry) and from the dipole field. Used as the foundational link between field and force.")

add("electrostatics_electric_potential",
    "Electric Potential of a Point Charge",
    "Electric potential V at distance r from a point charge q in vacuum: V = q/(4*pi*epsilon_0*r). Apply when the question concerns the potential due to a point source charge — its value at a point, the work done in bringing another charge from infinity to that point. The concept is the scalar potential of a point-charge source. Distinct from the electric field of a point charge (a vector, with 1/r^2 dependence) and from potential energy of two charges (q*V, with two charges in it). Used in problems involving the potential created by isolated charges.")

add("electrostatics_point_charge_field",
    "Electric Field of a Point Charge",
    "Electric field strength at distance r from an isolated point charge q in vacuum: E = q/(4*pi*epsilon_0*r^2). The field points radially outward from a positive charge, inward toward a negative one. Apply when the question concerns the field set up by a point charge (or a charged sphere treated as a point) at a specific distance. The concept is the inverse-square field of a point source. Distinct from Coulomb's law for force between TWO charges (q1*q2/(4*pi*epsilon_0*r^2)) and from the field inside a parallel-plate capacitor (uniform E=V/d). Used in problems about a single source charge's effect on its surroundings.")

add("electrostatics_potential_energy_two_charges",
    "Electrostatic Potential Energy of Two Point Charges",
    "Electrostatic potential energy of a pair of point charges q1 and q2 separated by distance r in vacuum: U = q1*q2/(4*pi*epsilon_0*r). Apply when the question concerns the energy required to assemble two charges from infinity to their current separation, or the energy released as they fly apart. The concept is the mutual electrostatic energy of a configured charge pair. Distinct from the potential due to a single charge (q/(4*pi*epsilon_0*r), no second charge) and from kinetic energy considerations. Used in charge-pair energy problems and atomic-physics binding-energy reasoning.")

# ═════════════════════════════════════════════════════════════════════════════
# fluid_mechanics (5 — buoyant + hydrostatic already covered above)
# ═════════════════════════════════════════════════════════════════════════════

add("fluid_mechanics_bernoulli",
    "Bernoulli's Equation (Energy Conservation in Fluid Flow)",
    "Bernoulli's Equation: along a streamline in a steady, incompressible, non-viscous fluid flow, the quantity P + rho*g*h + (1/2)*rho*v^2 is constant. Apply when the question concerns the relation between pressure, height, and speed at different points along a fluid streamline — flow through a pipe of varying cross-section, water shooting from an orifice, lift on an airfoil. The concept is conservation of mechanical energy per unit volume in fluid flow. Distinct from hydrostatic pressure (no flow, only the rho*g*h term), the continuity equation (mass conservation), and Stokes' law (viscous drag). Used in flow problems with two points whose pressures or speeds need relating.")

add("fluid_mechanics_continuity",
    "Continuity Equation for Incompressible Fluid Flow",
    "Continuity equation for steady, incompressible flow: A*v = constant along the streamline (A is cross-sectional area, v is flow speed). Apply when the question concerns a fluid flowing through a pipe of varying cross-section — flow speed increases where the pipe narrows, decreases where it widens. The concept is mass conservation in incompressible flow. Distinct from Bernoulli's equation (which adds pressure and height relationships) and from pressure formulas. Used in pipe-flow, nozzle, and venturi problems.")

add("fluid_mechanics_pressure",
    "Pressure as Force per Unit Area (General Definition)",
    "Pressure: the force applied per unit area perpendicular to a surface, P = F/A. Apply whenever the question requires converting between a force and the pressure it creates over an area, or vice versa. The concept is the general definition of pressure, valid for solids, liquids, gases, or any contact. Distinct from hydrostatic pressure (P = rho*g*h, a specific formula for fluid columns), atmospheric pressure (a specific value), and stress (similar form, but stress is usually internal in a solid). Used in any problem mixing forces and areas, especially at fluid-solid or solid-solid contact.")

add("fluid_mechanics_stokes_law",
    "Stokes' Law for Viscous Drag on a Small Sphere",
    "Viscous drag force on a small sphere of radius r moving at speed v through a fluid of viscosity eta, in the low-Reynolds limit: F = 6*pi*eta*r*v. Apply when the question concerns drag on a slowly-moving sphere in viscous fluid — falling raindrops at terminal velocity, oil drops in Millikan's experiment, particles sedimenting in a liquid. The concept is laminar viscous drag, linear in speed. Distinct from inertial drag (quadratic in speed, applies at high Reynolds number) and from buoyancy. Used to calculate terminal velocities and viscous transport.")

add("fluid_mechanics_terminal_velocity",
    "Terminal Velocity of a Sphere Falling Through a Viscous Fluid",
    "Terminal velocity of a small sphere of radius r and density rho falling through a viscous fluid of density rho_fluid and viscosity eta: v = (2*r^2*g*(rho - rho_fluid))/(9*eta). Apply when the question concerns the steady-state speed of a sphere in viscous fluid where gravity is balanced by buoyancy and viscous drag — a falling raindrop in air, an oil drop in a Millikan apparatus, a sphere sinking in oil. The concept is Stokes-drag-balanced-by-net-gravity. Distinct from free-fall velocity (no fluid) and from non-terminal transients. Used when terminal velocity is asked or implied.")

# ═════════════════════════════════════════════════════════════════════════════
# friction (6)
# ═════════════════════════════════════════════════════════════════════════════

add("friction_banked_road_max_speed",
    "Maximum Safe Speed on a Banked Curve with Friction",
    "Maximum safe speed on a banked curve of radius r at angle theta with coefficient of friction mu: v = sqrt(r*g*(mu + tan(theta))/(1 - mu*tan(theta))). Apply when the question concerns the highest speed a vehicle can sustain on a banked turn before slipping outward. The concept is combined banking + friction providing centripetal acceleration; friction acts inward, supplementing the horizontal component of the normal force. Distinct from a frictionless banked curve (mu = 0, simplifies to v = sqrt(r*g*tan(theta))) and from a flat road with friction (no bank angle). Used in road-design and curved-track problems.")

add("friction_incline_no_slip",
    "Angle of Repose (Maximum Frictionless-Sliding Incline)",
    "Angle of repose: the maximum incline angle at which a stationary object on an inclined surface will not slip, given coefficient of static friction mu: mu = tan(theta). At this angle, static friction is at its maximum and exactly balances the component of gravity along the slope. Apply when the question concerns whether an object on an incline will slip, the steepest stable angle, or the friction coefficient measured by tilting an incline until slipping begins. The concept is the threshold-tilt for static friction. Distinct from kinetic friction problems (motion-time situations) and from banked-curve formulas (which combine friction with circular motion). Used in incline and angle-of-repose problems.")

add("friction_kinetic_friction",
    "Kinetic Friction Force",
    "Force of kinetic (sliding) friction: f_k = mu_k * N, where mu_k is the coefficient of kinetic friction and N is the normal force pressing the surfaces together. The kinetic friction force opposes the direction of sliding and has a fixed magnitude (for a given normal force) once sliding begins. Apply when the question concerns a body already sliding across a surface and asks for the friction force, deceleration due to friction, or distance traveled before stopping. The concept is sliding-friction-proportional-to-normal-force. Distinct from static friction (which varies up to mu_s*N until slipping starts) and from viscous drag (which depends on speed). Used in kinetic-friction problems involving sliding.")

add("friction_pulling_force_horizontal",
    "Minimum Horizontal Force to Maintain Motion on a Surface",
    "Horizontal force required to pull a body of mass m at constant velocity along a surface with friction coefficient mu: F = mu*m*g. Apply when the question concerns the steady-state pulling force needed to overcome friction on a level surface, with no acceleration. The concept is friction-balanced horizontal pull. Distinct from accelerated-motion problems (which add m*a) and from pulling on inclines (which involves the incline angle). Used in straightforward friction-balance setups.")

add("friction_static_limit",
    "Maximum Static Friction Force",
    "Maximum static friction force: f_s_max = mu_s * N, where mu_s is the coefficient of static friction and N is the normal force. Static friction takes any value up to this maximum to keep an object from slipping; the formula gives the threshold above which motion begins. Apply when the question concerns whether an object will start to slip under an applied force, the minimum force required to start motion, or static-friction limits. The concept is the slipping-threshold force. Distinct from kinetic friction (constant value, applies once slipping starts) and from the angle-of-repose form (specific to inclines). Used in threshold-of-motion problems.")

add("friction_stopping_distance",
    "Stopping Distance under Kinetic Friction (No Other Forces)",
    "Distance a body of initial speed u travels before stopping on a surface with friction coefficient mu, with no other forces: s = u^2/(2*mu*g). Derived from kinematics with deceleration a = mu*g. Apply when the question concerns a body sliding to a stop on a rough surface — a sliding block, a car skidding to a halt. The concept is friction-driven kinematic stopping. Distinct from brake-applied stopping (which gives a different deceleration) and from energy-method stopping problems (equivalent but framed differently). Used in friction stopping problems.")

# ═════════════════════════════════════════════════════════════════════════════
# gravitation (6 — universal_law already done above)
# ═════════════════════════════════════════════════════════════════════════════

add("gravitation_escape_speed",
    "Escape Velocity from a Gravitating Body",
    "Escape speed from the surface of a body of mass M at radius r: v_esc = sqrt(2*G*M/r). The minimum launch speed for a projectile to escape the gravitational pull and reach infinity with zero residual speed. Apply when the question concerns the minimum launch speed to escape a planet, moon, or sun. The concept is gravitational binding energy converted to launch KE. Distinct from orbital speed (v_orb = sqrt(G*M/r), which is sqrt(2) smaller — the speed to circle, not escape) and from terminal velocity. Used in gravitational-escape problems.")

add("gravitation_field",
    "Gravitational Field of a Mass at Distance",
    "Gravitational field strength at distance r from a point mass M: g = G*M/r^2, directed toward the mass. Apply when the question concerns the gravitational field of a planet, moon, or star at a given point, or the local g at altitude or at a different planet's surface. The concept is the force-per-unit-test-mass field of a gravitating body. Distinct from the gravitational FORCE between two bodies (m*g of the test mass), from local Earth-surface g (a specific value of this formula at Earth's surface), and from gravitational potential (a different quantity). Used in gravitational-field problems.")

add("gravitation_kepler_third",
    "Kepler's Third Law (Orbital Period vs. Radius)",
    "Kepler's third law: the square of an orbital period is proportional to the cube of the orbit's semi-major axis (radius for circular orbit), T^2 = C*r^3 with constant C. Apply when the question compares orbital periods and radii of different bodies orbiting the same central body (planets around the sun, moons of Jupiter), or relates one set of orbital data to another. The concept is the universal period-radius scaling in central-force orbits. Distinct from the explicit orbital-period formula (T = 2*pi*sqrt(r^3/(G*M)), which gives the absolute value with constants) — Kepler's third is the proportionality between two systems. Used in orbital comparison problems.")

add("gravitation_orbital_period",
    "Orbital Period of a Circular Orbit",
    "Period of a circular orbit at radius r around a body of mass M: T = 2*pi*sqrt(r^3/(G*M)). Apply when the question asks for the period of a satellite, moon, or planet in circular orbit, with the central body's mass and orbital radius known. The concept is the absolute orbital period from gravitational physics. Distinct from Kepler's third law (the proportionality form, without G*M) and from orbital speed. Used to compute the actual time for one revolution in orbital mechanics.")

add("gravitation_orbital_speed",
    "Orbital Speed of a Circular Orbit",
    "Tangential speed of a body in circular orbit at radius r around a body of mass M: v = sqrt(G*M/r). Apply when the question concerns the speed of a satellite, planet, or moon in circular orbit. The concept is gravity-providing-centripetal force balanced at the orbital radius. Distinct from escape speed (sqrt(2) larger — speed to escape, not orbit) and from surface speeds of rotation. Used in orbital-speed problems and satellite mechanics.")

add("gravitation_potential_energy",
    "Gravitational Potential Energy of Two Masses",
    "Gravitational potential energy of two masses M and m at separation r: U = -G*M*m/r, taken as zero at infinity. Apply when the question concerns the gravitational binding energy of orbits, escape calculations from a planet, or comparison of energies at different orbital radii. The concept is the negative-binding-energy of attracted masses. Distinct from the near-Earth-surface form U = m*g*h (a local linearization valid only for small heights compared to Earth radius) and from kinetic energy. Used in gravitational binding, escape, and orbital-energy problems.")

# ═════════════════════════════════════════════════════════════════════════════
# kinematics (8 — v2_u2_2as and v_u_at done above)
# ═════════════════════════════════════════════════════════════════════════════

add("kinematics_average_velocity",
    "Average Velocity (General Definition)",
    "Average velocity over any motion: total displacement divided by total time, v_avg = s/t. This is the GENERAL definition — it applies to any motion in one dimension, whether uniform, uniformly accelerated, or arbitrary. Not restricted to constant-speed motion. Apply when the question gives total displacement and total time (or equivalently, two of these three quantities) and asks for the third, without needing to know the motion's detailed time-evolution. The concept is the defining ratio of displacement to time-interval. Distinct from instantaneous velocity (the derivative, at a specific instant) and from the constant-acceleration average velocity formula s = ((u+v)/2)*t (which is a SPECIAL case requiring uniform acceleration). Used as the foundational definition in kinematics.")

add("kinematics_free_fall_velocity",
    "Free-Fall Velocity under Constant Gravity",
    "Velocity of a body in vertical motion under constant gravity, taking upward as positive: v = u - g*t. Apply when the question concerns a body thrown upward, dropped, or moving vertically under gravity alone, and asks for velocity at a given time given the initial velocity. The concept is the time-velocity kinematic relation applied specifically to vertical motion with acceleration -g (downward). Distinct from the general v = u + a*t (the same physics with a generic acceleration sign and direction). For 'dropped' or 'released' problems, u = 0 and v = -g*t (downward speed grows linearly). Used in vertical-motion problems involving time.")

add("kinematics_projectile_max_height",
    "Maximum Height of a Projectile",
    "Maximum vertical height reached by a projectile launched at angle theta with initial speed u, under constant gravity: H = u^2*sin^2(theta)/(2*g). Apply when the question asks for the peak vertical position a projectile reaches during its flight. The concept is the highest point in projectile motion, where vertical velocity is zero. Distinct from the range (horizontal distance to landing) and the time of flight. Used in projectile-motion problems asking for maximum height. For vertical-only launches (theta = 90 degrees), reduces to H = u^2/(2*g).")

add("kinematics_projectile_range",
    "Range of a Projectile Launched and Landing at the Same Height",
    "Horizontal range of a projectile launched from ground level at angle theta with initial speed u, landing at the same height under constant gravity: R = u^2*sin(2*theta)/g. Apply when the question concerns the horizontal distance a projectile travels before returning to its launch height. The concept is the horizontal distance of projectile motion under uniform gravity, with launch and landing at the same level. Distinct from maximum height (vertical peak) and from problems where launch and landing are at different elevations. Maximum range is at theta = 45 degrees. Used in standard projectile-range problems.")

add("kinematics_projectile_time",
    "Total Time of Flight of a Projectile",
    "Total time a projectile spends in the air, launched with initial speed u at angle theta, returning to its launch height under constant gravity: t = 2*u*sin(theta)/g. Apply when the question asks for how long a projectile is in the air or the time to return to launch level. The concept is the total time of flight for a same-launch-and-landing-height projectile, determined by the vertical velocity component. Distinct from time to maximum height (half this value) and from problems with landing at a different height. Used in projectile time-of-flight problems.")

add("kinematics_relative_velocity",
    "Relative Velocity Between Two Bodies",
    "Relative velocity of one body with respect to another: v_AB = v_A - v_B (sign convention: positive if A moves in the chosen positive direction faster than B). Apply when the question asks about the velocity of one object as seen from another moving object — overtaking, approach speed, two cars on a highway, an observer on a moving train viewing another moving body. The concept is the difference of velocities in a chosen reference frame. Distinct from absolute (ground-frame) velocities and from acceleration-related kinematics. Used in relative-motion problems with two moving bodies.")

add("kinematics_s_avg_velocity",
    "Displacement from Average Velocity for Uniform Acceleration",
    "Displacement for uniformly accelerated 1D motion expressed via the average of initial and final velocities: s = ((u+v)/2)*t. Apply when the question gives initial and final velocities and a time interval and asks for displacement, under constant-acceleration conditions. The concept is that for constant acceleration, the average velocity equals (u+v)/2, so displacement equals that average times the time. Distinct from the general v_avg = s/t (which applies to ANY motion, not just constant a) and from v^2 = u^2 + 2*a*s (time-free, no t). Used in constant-acceleration problems where u, v, t are known and acceleration itself is not needed.")

add("kinematics_s_ut_half_at2",
    "Displacement under Constant Acceleration with Time",
    "Displacement of a body under uniform acceleration as a function of initial velocity, acceleration, and time: s = u*t + (1/2)*a*t^2. Apply when the question gives initial velocity, acceleration, and time, and asks for displacement (or any one of these when the others are known). The concept is the constant-acceleration displacement-time relation. Distinct from v^2 = u^2 + 2*a*s (time-absent, displacement-and-velocity), from v = u + a*t (no displacement), and from s = ((u+v)/2)*t (uses final velocity instead of acceleration). Used when displacement and time are in scope but final velocity is not directly involved.")

# ═════════════════════════════════════════════════════════════════════════════
# laws_of_motion (7 — newton_second_law done above)
# ═════════════════════════════════════════════════════════════════════════════

add("general_density_definition",
    "Density (Mass per Unit Volume)",
    "Density of a body: mass per unit volume, rho = m/V. Apply whenever the question relates a body's mass, volume, and density — finding mass from density and volume, finding volume from mass and density, identifying material from density. The concept is the defining ratio that characterizes how much mass occupies a given volume. Distinct from weight density (rho*g), pressure (force per area), and specific gravity (a ratio). This equation is a critical bridge: in dynamics problems giving density and volume rather than mass, this is the path to compute mass for use in F=ma, work-energy, etc.")

add("laws_of_motion_apparent_weight_down",
    "Apparent Weight in a Downward-Accelerating Frame",
    "Apparent weight (normal force) on a body of mass m in a system that is accelerating downward at rate a: N = m*(g - a). Apply when the question concerns the apparent weight in a downward-accelerating elevator, a falling lift, or a person in free-fall (where N = 0). The concept is reduced normal force when the supporting surface accelerates downward with the body. At a = g (free fall) the apparent weight is zero — weightlessness. Distinct from upward-acceleration form (N = m*(g + a)) and from regular weight on a stationary surface (N = m*g). Used in accelerating-frame normal-force problems.")

add("laws_of_motion_apparent_weight_up",
    "Apparent Weight in an Upward-Accelerating Frame",
    "Apparent weight (normal force) on a body of mass m in a system accelerating upward at rate a: N = m*(g + a). Apply when the question concerns an elevator accelerating upward, a rocket lifting off, or any setup where the supporting surface accelerates upward. The concept is increased normal force when the supporting surface accelerates upward. Distinct from downward-acceleration form (N = m*(g - a)) and from regular weight at rest (N = m*g). Used in accelerating-frame problems where weight feels heavier.")

add("laws_of_motion_impulse",
    "Impulse from Constant Force and Time",
    "Impulse from a constant force applied for a time interval: J = F*t. Impulse is a vector with the units N*s (= kg*m/s, same as momentum). Apply when the question gives a constant force and the duration over which it acts and asks for the impulse delivered. The concept is the time-integrated force, equivalent to the change in momentum it produces (impulse-momentum theorem). Distinct from work (force times distance, energy) and from average force (impulse / time). Used in collision and force-time problems.")

add("laws_of_motion_impulse_momentum",
    "Impulse-Momentum Theorem",
    "Impulse equals change in momentum: J = Delta_p = m*(v - u). Apply when the question relates the impulse delivered to a body and the resulting change in its velocity (or momentum). The concept is the direct equivalence between impulse and momentum change — Newton's 2nd law integrated over time. Distinct from impulse computed as F*t (which assumes a constant force) and from the work-energy theorem (force integrated over distance, gives energy change). Used in collision and momentum-change problems.")

add("laws_of_motion_momentum_rate",
    "Newton's Second Law in Momentum-Rate Form",
    "Newton's second law in its general form: net force equals the time-rate of change of momentum, F = dp/dt. For constant mass this reduces to F = m*a. Apply when the question concerns variable-mass systems (rockets, conveyor belts loading material, a chain falling onto a scale) where Newton's force-equals-mass-times-acceleration form is incomplete. The concept is the more general form of the 2nd law, valid even when mass changes. Distinct from F = m*a (the constant-mass special case) and from impulse-momentum (the time-integrated form). Used when the system's mass is not constant.")

add("laws_of_motion_weight",
    "Weight Force (Near Earth's Surface)",
    "Weight force on a body of mass m near Earth's surface: W = m*g, directed downward. Apply when the question asks for the weight of an object, or relates mass and weight near Earth's surface. The concept is the gravitational force on a body in a uniform gravitational field. Distinct from mass (an intrinsic property, unit kg) and from apparent weight (which can differ from mg if the supporting frame is accelerating). Also distinct from the universal Newton's law of gravitation (which gives weight on other planets or at altitude via F = G*M*m/r^2). Used in weight-related problems.")

# ═════════════════════════════════════════════════════════════════════════════
# magnetism (6 — lorentz_force done above)
# ═════════════════════════════════════════════════════════════════════════════

add("magnetism_circular_radius_charge",
    "Radius of Circular Motion of a Charge in a Magnetic Field",
    "Radius of the circular orbit of a charged particle of mass m, charge q, speed v, moving perpendicular to a uniform magnetic field B: r = m*v/(q*B). Apply when the question concerns a charged particle's circular path in a magnetic field — electron beams, cyclotrons, mass spectrometers. The concept is the radius where the magnetic Lorentz force exactly provides the centripetal acceleration. Distinct from the cyclotron frequency (the rate at which the particle goes around) and from straight-line motion in fields. Used in cyclotron, mass-spectrometer, and charge-in-field problems.")

add("magnetism_cyclotron_frequency",
    "Cyclotron Frequency",
    "Cyclotron frequency of a charged particle (mass m, charge q) in a magnetic field B: f = q*B/(2*pi*m). Apply when the question concerns the rotation frequency of a charged particle in a uniform magnetic field — cyclotrons, ion traps. The concept is that the orbital frequency is independent of the particle's speed (or orbit radius), depending only on q, B, m. Distinct from the orbital radius (which DOES depend on speed) and from other resonance frequencies. Used in cyclotron-resonance problems.")

add("magnetism_field_long_wire",
    "Magnetic Field of a Long Straight Current-Carrying Wire",
    "Magnetic field magnitude at perpendicular distance r from a long, straight wire carrying steady current I: B = mu_0*I/(2*pi*r). The field forms concentric circles around the wire (right-hand rule for direction). Apply when the question asks for the magnetic field at a point near a current-carrying wire. The concept is the Biot-Savart result for the canonical case of a long straight current. Distinct from the field inside a solenoid (uniform, depends on turns per length and current) and from the field of a current loop. Used in problems concerning the field near a single straight wire.")

add("magnetism_field_solenoid",
    "Magnetic Field Inside an Ideal Solenoid",
    "Magnetic field magnitude inside a long ideal solenoid: B = mu_0*n*I, where n is the number of turns per unit length and I is the current. The field is uniform and parallel to the axis inside, approximately zero outside. Apply when the question concerns the field inside a solenoid (e.g. setting up a uniform field for a Hall-effect or motional-EMF experiment). The concept is the uniform interior field of a tightly-wound coil. Distinct from a single current loop (which has a non-uniform field) and from a long straight wire (whose field falls off with distance). Used in solenoid-field problems.")

add("magnetism_force_wire",
    "Force on a Current-Carrying Wire in a Magnetic Field",
    "Force on a wire of length L carrying current I in a uniform magnetic field B at angle theta to the field: F = B*I*L*sin(theta). Apply when the question concerns the force a magnetic field exerts on a straight conductor carrying current — motors, current-balances, force on bus bars. The concept is the integrated Lorentz force on the moving charges in the wire. Distinct from the Lorentz force on a single charge (per-particle form) and from torque on a current loop (which integrates this over a loop). Used in motor and wire-in-field problems.")

add("magnetism_torque_loop",
    "Torque on a Current Loop in a Magnetic Field",
    "Torque on a flat current loop of N turns, area A, carrying current I, in a uniform magnetic field B at angle theta between the field and the loop's normal: tau = N*I*A*B*sin(theta). Apply when the question concerns the torque on a current loop in a magnetic field — moving-coil galvanometers, electric motors, magnetic moment in field. The concept is the rotational effect of the magnetic force on a current loop, equivalent to torque on a magnetic dipole. Distinct from the force on a straight wire and from torque-free configurations. Used in motor and galvanometer problems.")

# ═════════════════════════════════════════════════════════════════════════════
# modern_physics (6 — photon_energy done above)
# ═════════════════════════════════════════════════════════════════════════════

add("modern_physics_bohr_radius",
    "Bohr Radius for Hydrogen Atom Stationary States",
    "Radius of the n-th allowed orbit of an electron in a hydrogen-like atom in the Bohr model. The smallest (n=1) orbit gives the Bohr radius. Apply when the question concerns the radius of an electron's orbit in a hydrogen atom (or hydrogenic ion) in the Bohr model. The concept is the quantized stationary-state orbit radius derived from angular-momentum quantization. Distinct from the hydrogen energy levels (a separate quantity) and from atomic radii measured in chemistry. Used in atomic-physics problems about orbit geometry in the Bohr model.")

add("modern_physics_de_broglie",
    "de Broglie Wavelength of a Particle",
    "de Broglie wavelength of a particle with momentum p: lambda = h_planck/p. Apply when the question concerns the wave nature of a particle — wavelength of an electron in a beam, wavelength of a thermal neutron, wave-particle duality experiments (Davisson-Germer). The concept is matter-as-wave: any particle has a wavelength inversely proportional to its momentum. Distinct from the photon's wavelength derived from its energy (E = h*nu, with c = nu*lambda) and from classical wavelengths. Used in matter-wave problems.")

add("modern_physics_hydrogen_energy",
    "Energy Levels of the Hydrogen Atom",
    "Energy of the n-th level of a hydrogen atom: E_n = -13.6/n^2 eV. Apply when the question concerns electron energy in a stationary state of hydrogen, or photon energy for a transition between two levels (E_photon = E_n2 - E_n1). The concept is the discrete energy spectrum of the hydrogen atom, with the ground state at -13.6 eV and levels approaching zero at infinity. Distinct from the Rydberg formula (which gives wavelengths of emission lines from these levels) and from photon energy in general. Used in hydrogen-atom spectroscopy and transition problems.")

add("modern_physics_mass_energy",
    "Mass-Energy Equivalence",
    "Einstein's mass-energy relation: E = m*c^2. The rest energy of a body of mass m equals its mass times c squared. Apply when the question concerns mass-energy conversion — pair production, nuclear binding energy, particle annihilation, mass defect in nuclear reactions. The concept is that mass is a form of energy. Distinct from the kinetic energy of motion (separate from rest energy in special relativity) and from photon energy. Used in nuclear and particle-physics energy problems.")

add("modern_physics_photoelectric_equation",
    "Einstein's Photoelectric Equation",
    "Maximum kinetic energy of photoelectrons emitted from a metal of work function W_0, illuminated by light of frequency nu: K_max = h*nu - W_0. Below the threshold frequency (h*nu < W_0), no photoelectrons are emitted regardless of intensity. Apply when the question concerns the photoelectric effect — threshold frequency, stopping potential, maximum KE of emitted electrons. The concept is energy conservation in single-photon absorption by an electron in the metal. Distinct from the bare photon energy formula (E = h*nu, no work function) and from classical EM-wave intensity-driven emission (which doesn't fit data). Used in photoelectric-effect problems.")

add("modern_physics_rydberg_formula",
    "Rydberg Formula for Hydrogen Spectral Lines",
    "Rydberg formula for the wavelengths of emission/absorption lines in hydrogen: 1/lambda = R*(1/n1^2 - 1/n2^2), where R is the Rydberg constant and n1, n2 are the lower and upper quantum numbers of the transition. (The graph notation reuses u and v as the two quantum numbers and R_g as the Rydberg constant — read these as quantum numbers and Rydberg constant, not velocities or gas constant.) Apply when the question concerns line spectra of hydrogen — Lyman, Balmer, Paschen series — and asks for the wavelength of a specific transition. The concept is the discrete spectrum of allowed photons. Distinct from the photon energy formula directly (E = h*c/lambda, which converts wavelength to energy without referencing levels) and from the hydrogen energy-level formula. Used in atomic-spectroscopy problems.")

# ═════════════════════════════════════════════════════════════════════════════
# momentum_collisions (7)
# ═════════════════════════════════════════════════════════════════════════════

add("momentum_collisions_coefficient_restitution",
    "Coefficient of Restitution",
    "Coefficient of restitution e: the ratio of relative speed of separation after a collision to relative speed of approach before, e = v_separation / u_approach. Apply when the question concerns how 'elastic' a collision is — e=1 perfectly elastic, e=0 perfectly inelastic (stick together), in between for real collisions. The concept is a phenomenological measure of collision elasticity. Distinct from energy-conservation analysis (which directly checks kinetic energy preservation) and from momentum conservation (which always holds in isolated collisions regardless of e). Used in real-collision problems involving bounce, ball drops, etc.")

add("momentum_collisions_conservation_two_body",
    "Conservation of Linear Momentum in a Two-Body Collision",
    "Conservation of linear momentum in an isolated collision between two bodies: total momentum before equals total momentum after. m1*u1 + m2*u2 = m1*v1 + m2*v2. (The graph's variable scheme uses m, M for the two masses and u, v, a, s as the four velocities — see graph notes; concept is the standard two-body momentum conservation.) Apply when the question concerns a collision between two bodies in an isolated system — billiard balls, particles in physics, vehicles colliding. The concept is total momentum conservation, valid regardless of whether the collision is elastic or inelastic. Distinct from elastic-collision-specific formulas (which add KE conservation as a second constraint) and from impulse-momentum theorem applied to a single body. Used in any two-body-collision problem.")

add("momentum_collisions_elastic_collision_v1",
    "Final Velocity of First Body in 1D Elastic Collision",
    "Final velocity of the first body after a one-dimensional elastic collision between bodies of masses m1, m2 with initial velocities u1, u2: v1 = ((m1-m2)*u1 + 2*m2*u2)/(m1+m2). Derived from simultaneous conservation of momentum and kinetic energy. Apply when the question concerns a 1D elastic collision and asks for the post-collision velocity of body 1. The concept is the elastic-collision result for the first body. Distinct from the perfectly-inelastic case (where bodies stick) and from the general momentum conservation (one equation, two unknowns). Used in elastic-collision problems for body 1.")

add("momentum_collisions_elastic_collision_v2",
    "Final Velocity of Second Body in 1D Elastic Collision",
    "Final velocity of the second body after a one-dimensional elastic collision between bodies of masses m1, m2 with initial velocities u1, u2: v2 = ((m2-m1)*u2 + 2*m1*u1)/(m1+m2). Symmetric partner to the v1 formula. Apply when the question concerns a 1D elastic collision and asks for the post-collision velocity of body 2. The concept is the elastic-collision result for the second body. Distinct from the v1 formula (which gives body 1's velocity) and from the perfectly-inelastic case. Used in elastic-collision problems for body 2.")

add("momentum_collisions_impulse_change_momentum",
    "Impulse Equals Mass Times Change in Velocity",
    "Impulse on a body of mass m undergoing velocity change from u to v: J = m*(v - u). This is the impulse-momentum theorem applied to a single body. Apply when the question relates the impulse delivered to a body's velocity change — bounce of a ball, kick of a soccer player, sudden push. The concept is the direct equivalence between impulse and momentum change. Distinct from impulse = F*t (which assumes constant force, and gives the same impulse via the time-integrated route) and from work-energy theorem (force times distance, energy not momentum). Used in collision problems where velocity changes.")

add("momentum_collisions_linear_momentum",
    "Linear Momentum (Definition)",
    "Linear momentum of a body of mass m moving at velocity v: p = m*v. A vector quantity, with the same direction as velocity. Apply whenever the question asks for the momentum of a moving body, or relates momentum to mass and velocity. The concept is the defining product of mass and velocity. Distinct from kinetic energy (a scalar, (1/2)*m*v^2) and from impulse (the change in momentum, or force times time). Used as the foundational quantity in any momentum problem.")

add("momentum_collisions_perfectly_inelastic_velocity",
    "Common Final Velocity in a Perfectly Inelastic Collision",
    "Final common velocity after a perfectly inelastic collision between bodies of masses m1, m2 with initial velocities u1, u2: v = (m1*u1 + m2*u2)/(m1 + m2). Apply when the question concerns a collision where the two bodies stick together after impact and asks for the common post-collision speed. The concept is total-momentum-conservation applied with the constraint that the bodies share one final velocity. Distinct from elastic-collision formulas (which give different final velocities for each body) and from energy considerations (KE is NOT conserved in inelastic collisions — energy is lost to heat, sound, deformation). Used in perfectly-inelastic-collision problems.")

# ═════════════════════════════════════════════════════════════════════════════
# nuclear_physics (8)
# ═════════════════════════════════════════════════════════════════════════════

add("nuclear_physics_activity",
    "Activity of a Radioactive Sample",
    "Activity (decay rate) of a radioactive sample containing N nuclei with decay constant lambda: A = lambda*N. Activity is the number of disintegrations per second, measured in becquerels (Bq) or curies (Ci). Apply when the question relates the activity of a sample to its number of nuclei present and the decay constant. The concept is the instantaneous decay rate. Distinct from the radioactive-decay equation (which gives N as a function of time) and from half-life or mean-life formulas. Used in radioactivity problems where activity is asked.")

add("nuclear_physics_binding_energy",
    "Nuclear Binding Energy from Mass Defect",
    "Binding energy of a nucleus from its mass defect: E_B = (Delta_m)*c^2, where Delta_m is the difference between the sum of constituent nucleon masses and the actual nuclear mass. Apply when the question concerns nuclear binding energy, mass defect, or nuclear stability. The concept is mass-energy equivalence applied to nuclear binding — the missing mass is the energy holding the nucleus together. Distinct from the Q-value of a nuclear reaction (a related but distinct application) and from photon energy. Used in nuclear-physics binding-energy problems.")

add("nuclear_physics_decay_constant_half_life",
    "Decay Constant from Half-Life",
    "Decay constant lambda from half-life T_half: lambda = ln(2)/T_half. The decay constant is the inverse-time rate at which a radioactive sample's population decays exponentially. Apply when the question concerns converting half-life to decay constant or vice versa. The concept is the parameter that controls the rate of exponential decay. Distinct from mean life (= 1/lambda) and from the activity formula. Used in radioactivity problems involving half-life.")

add("nuclear_physics_half_life",
    "Half-Life from Decay Constant",
    "Half-life of a radioactive sample from decay constant lambda: T_half = ln(2)/lambda. Half-life is the time for half the nuclei in a sample to decay. Apply when the question concerns finding half-life from decay constant or relating these to mean life. The concept is the time-scale of exponential decay measured by half-population. Distinct from mean life (the e-folding time, longer than half-life by a factor 1/ln(2)) and from activity formulas. Used in radioactivity half-life problems.")

add("nuclear_physics_mass_number",
    "Mass Number as Sum of Protons and Neutrons",
    "Mass number A of a nucleus: A = Z + N, where Z is the number of protons (atomic number) and N is the number of neutrons. Apply when the question relates a nucleus's mass number to its proton and neutron count, or asks for the neutron count from atomic number and mass number. The concept is the count of nucleons in a nucleus. Distinct from atomic mass in amu (a measured quantity, close to but not exactly A) and from atomic number alone. Used in nuclear-composition problems.")

add("nuclear_physics_mean_life",
    "Mean Life of a Radioactive Nucleus",
    "Mean life (expected lifetime) of a radioactive nucleus: tau = 1/lambda. Apply when the question concerns the average time a nucleus survives before decaying. The concept is the e-folding time of exponential decay — the time after which a fraction 1/e remains. Distinct from half-life (the time for 50% to decay; tau = T_half / ln(2)) and from decay constant. Used in radioactivity mean-life problems.")

add("nuclear_physics_q_value",
    "Q-Value of a Nuclear Reaction",
    "Q-value of a nuclear reaction from mass defect: Q = (Delta_m)*c^2, where Delta_m is the difference between rest masses of reactants and products. Apply when the question concerns the energy released (Q > 0, exothermic) or absorbed (Q < 0, endothermic) in a nuclear reaction. The concept is mass-energy equivalence applied to reaction energetics. Distinct from binding energy of a single nucleus (which is one application of mass-energy) and from kinetic-energy considerations of reaction products. Used in nuclear-reaction Q-value problems.")

add("nuclear_physics_radioactive_decay",
    "Radioactive Decay Law (Number Remaining vs. Time)",
    "Number of nuclei remaining after time t in a radioactive sample with initial count N_0 and decay constant lambda: N(t) = N_0 * exp(-lambda*t). Apply when the question asks how much of a radioactive sample remains (or has decayed) after a specified time, given the decay constant or half-life. The concept is exponential decay of a population. Distinct from the activity formula (A = lambda*N, the instantaneous decay rate) and from half-life (a single point on this curve). Used in radioactive-decay problems involving time dependence.")

# ═════════════════════════════════════════════════════════════════════════════
# ray_optics (6 — lens_formula done above)
# ═════════════════════════════════════════════════════════════════════════════

add("ray_optics_critical_angle",
    "Critical Angle for Total Internal Reflection",
    "Critical angle theta_c for total internal reflection: sin(theta_c) = 1/mu, where mu is the refractive index of the denser medium (going from a denser medium to a less-dense one). Above the critical angle, all light is reflected internally. Apply when the question concerns total internal reflection — fiber optics, prisms, mirages, light trapped in glass. The concept is the angle of incidence beyond which refraction is impossible. Distinct from ordinary Snell's law (which applies below the critical angle) and from the lens-maker formula. Used in TIR and fiber-optic problems.")

add("ray_optics_lens_maker",
    "Lens-Maker's Equation",
    "Focal length of a thin lens from its geometry and refractive index: 1/f = (mu_r - 1)*(1/R_1 - 1/R_2), where mu_r is the lens material's refractive index relative to its surroundings, and R_1, R_2 are the radii of curvature of the two surfaces (with sign convention). Apply when the question concerns the focal length of a lens given its physical properties (curvature, material). The concept is geometric and material determination of focal length. Distinct from the lens formula (1/f = 1/v - 1/u, which uses imaging geometry, not lens properties) and from the mirror formula. Used in lens-design problems.")

add("ray_optics_magnification",
    "Magnification of a Lens or Mirror Image",
    "Magnification of an image formed by a lens or mirror: m = v_image / u_object. The sign of m indicates upright (positive) or inverted (negative) image, and the magnitude gives the size ratio (>1 enlarged, <1 reduced). Apply when the question concerns the size or orientation of an image — magnifying glass, camera, microscope. The concept is the size ratio of image to object. Distinct from the focal length and from the lens formula. Used when the question asks about image size.")

add("ray_optics_mirror_formula",
    "Mirror Formula (Object-Image-Focal-Length for Mirrors)",
    "Mirror formula for spherical mirrors relating object distance, image distance, and focal length: 1/f = 1/v + 1/u (Cartesian sign convention). Apply when the question concerns image formation by a concave or convex mirror — where the image is, how large, real or virtual. The concept is the geometric imaging equation for mirrors. Distinct from the LENS formula (1/f = 1/v - 1/u, opposite sign on 1/u) and from lens-maker's equation. Used in mirror-imaging problems.")

add("ray_optics_power_lens",
    "Optical Power of a Lens",
    "Optical power of a lens: P = 1/f (in diopters when f is in meters). Apply when the question concerns the strength of a lens or when summing powers of multiple lenses in contact (which add: P_total = P_1 + P_2 + ...). The concept is the inverse focal length, useful for combination calculations. Distinct from the focal length itself (just its reciprocal) and from the lens-maker's equation. Used in problems about combinations of lenses, eyeglass prescriptions.")

add("ray_optics_snells_law",
    "Snell's Law (Refraction at an Interface)",
    "Snell's law: at a refracting interface, the angles of incidence and refraction relate as mu_r = sin(theta_i)/sin(theta_t), where mu_r is the relative refractive index of the second medium with respect to the first. Apply when the question concerns light bending at an interface — air-glass, glass-water, prism refraction, fish-out-of-water apparent depth. The concept is the law of refraction. Distinct from the critical angle (a derived special case for total internal reflection) and from reflection laws. Used in all refraction problems.")

# ═════════════════════════════════════════════════════════════════════════════
# rotational_motion (8)
# ═════════════════════════════════════════════════════════════════════════════

add("rotational_motion_angular_acceleration",
    "Angular Acceleration as Rate of Change of Angular Velocity",
    "Angular acceleration: rate of change of angular velocity, alpha = d(omega)/dt, for constant alpha equivalently alpha = (omega - omega_0)/t. Apply when the question relates how angular velocity changes over time — a spinning disc speeding up, a wheel braking. The concept is the rotational analog of linear acceleration. Distinct from centripetal acceleration (which is geometric, not rotational rate) and from tangential acceleration (which equals r*alpha at the rim of a spinning body). Used in rotational kinematics problems.")

add("rotational_motion_angular_momentum",
    "Angular Momentum of a Rotating Body",
    "Angular momentum L of a rigid body rotating about a fixed axis with moment of inertia I and angular velocity omega: L = I*omega. Apply when the question concerns angular momentum of a spinning body, conservation of angular momentum, or rotational counterparts of linear momentum problems. The concept is the rotational analog of linear momentum. Distinct from linear momentum (p = m*v) and from torque (which changes angular momentum over time). Used in rotational momentum conservation problems — figure skater pulling arms in, etc.")

add("rotational_motion_angular_velocity",
    "Angular Velocity from Angle and Time (Uniform Rotation)",
    "Angular velocity from total angular displacement and time, for uniform rotation: omega = theta/t. Apply when the question concerns uniform rotational motion and gives total angle and time, asking for angular velocity, or vice versa. The concept is the rotational analog of v = s/t. Distinct from instantaneous angular velocity in non-uniform motion and from the angular speed formula omega = v/r (which connects rotational and linear motion). Used in uniform-rotation problems.")

add("rotational_motion_parallel_axis",
    "Parallel-Axis Theorem for Moment of Inertia",
    "Parallel-axis theorem (graph stores the additional term): the moment of inertia about an axis parallel to one through the center of mass, displaced by distance d, is I = I_cm + M*d^2. Apply when the question requires the moment of inertia about an axis other than through the center of mass — a rod rotated about its end, a disc rotated about a point on its rim. The concept is geometric shift of the rotation axis. Distinct from the perpendicular-axis theorem (a different relation for planar bodies) and from the standard center-of-mass moments. Used to compute I about non-central axes.")

add("rotational_motion_rotational_ke",
    "Rotational Kinetic Energy",
    "Kinetic energy of a rigid body rotating about a fixed axis: K = (1/2)*I*omega^2. Apply when the question concerns the energy stored in rotation — a spinning flywheel, a rolling wheel (combined with translational KE), a rotating planet. The concept is the rotational analog of (1/2)*m*v^2. Distinct from translational KE (which uses mass and linear speed) and from the rotational energy of a deformable body. Used in rotational energy and energy-conservation problems involving spin.")

add("rotational_motion_rotational_kinematics_omega",
    "Final Angular Velocity from Constant Angular Acceleration",
    "Angular velocity after constant angular acceleration alpha for time t, starting from rest: omega = alpha*t. (More generally, omega = omega_0 + alpha*t, but the graph stores the special-case form.) Apply when the question concerns uniformly accelerated rotation starting from rest, asking for angular velocity after a given time, or angular acceleration from a velocity change. The concept is the rotational counterpart of v = u + a*t. Distinct from omega = theta/t (which is uniform-rotation average) and from non-uniform angular acceleration problems. Used in rotational kinematics.")

add("rotational_motion_torque",
    "Torque from Force at a Lever Arm",
    "Torque produced by a force F applied at a perpendicular distance r from a pivot, with angle theta between the force vector and the lever arm: tau = r*F*sin(theta). Apply when the question concerns the rotational effect of a force — a wrench turning a bolt, a door opened by a push, a seesaw with masses. The concept is the cross-product magnitude of position and force, giving the rotational tendency. Distinct from net torque (sum of all torques on a body) and from torque on a current loop in a magnetic field (similar form, different physics). Used in static-equilibrium and torque-balance problems.")

add("rotational_motion_torque_inertia",
    "Newton's Second Law for Rotation",
    "Newton's second law in rotational form: net torque equals moment of inertia times angular acceleration, tau = I*alpha. Apply when the question concerns the angular acceleration of a rigid body under applied torques — a pulley with mass spinning under tension, a wheel braking, a yo-yo unwinding. The concept is the rotational analog of F = m*a. Distinct from the linear F = m*a (for translational motion) and from angular-momentum conservation problems. Used in rotational dynamics problems.")

# ═════════════════════════════════════════════════════════════════════════════
# sound (6)
# ═════════════════════════════════════════════════════════════════════════════

add("sound_beat_frequency",
    "Beat Frequency Between Two Close Frequencies",
    "Beat frequency produced when two sound waves of nearby frequencies f1, f2 superpose: f_beat = |f1 - f2|. Apply when the question concerns the periodic loudness modulation heard when two close-but-not-identical tones play together — tuning instruments, ultrasonic mixing. The concept is the audible difference frequency from superposition of two sinusoids. (The graph's variable scheme uses u and v as the two frequencies — read as f1, f2.) Distinct from Doppler shifts (which involve a moving source or observer) and from harmonic structure. Used in beat-frequency problems.")

add("sound_doppler_observer_moving",
    "Doppler Shift: Observer Moving Relative to Stationary Source",
    "Observed frequency when an observer moves with speed u toward a stationary source emitting at frequency f, with sound speed v in the medium: f_obs = f * (v + u)/v. Negative u (observer moving away) gives a lower frequency. Apply when the question concerns the apparent change in pitch from a moving observer — a person on a bicycle approaching a stationary speaker, etc. The concept is the Doppler effect for a moving observer with a stationary source. Distinct from the moving-source case (a different formula) and from the general source-and-observer-both-moving case. Used in Doppler-with-moving-observer problems.")

add("sound_doppler_source_moving",
    "Doppler Shift: Source Moving Relative to Stationary Observer",
    "Observed frequency when a source moving at speed u emits frequency f and the observer is stationary, with sound speed v: f_obs = f * v / (v - u) for a source approaching, with (v + u) in the denominator for a source receding. Apply when the question concerns the apparent change in pitch from a moving source — a passing ambulance siren, a train whistle approaching/receding. The concept is the Doppler effect for a moving source with a stationary observer. Distinct from the moving-observer case (different formula structure) and from the general source-and-observer-both-moving case. Used in Doppler-with-moving-source problems.")

add("sound_intensity_level",
    "Sound Intensity Level in Decibels",
    "Sound intensity level in decibels (dB), relative to a reference intensity I_0 (typically 1e-12 W/m^2 at 1 kHz): beta = 10*log10(I/I_0). Apply when the question relates physical sound intensity (W/m^2) to perceptual loudness on the decibel scale. The concept is the logarithmic decibel scaling of intensity. Distinct from intensity itself (the physical W/m^2) and from sound pressure level (a related but pressure-based scale). Used in dB-conversion and acoustic-loudness problems.")

add("sound_open_pipe_frequency",
    "Standing Wave Frequencies in an Open-Open Pipe",
    "Resonant frequencies of an open-open organ pipe of length L with sound speed v: f_n = n*v/(2*L), n = 1, 2, 3, ... — fundamental at n=1, harmonics at n>1. Apply when the question concerns standing waves in a pipe open at both ends — flutes, recorders, open organ pipes. The concept is integer-harmonic standing waves with displacement antinodes at both open ends. Distinct from closed-pipe resonances (odd harmonics only, f_n = n*v/(4*L), at n=1,3,5...) and from string standing waves (similar form but different physical realization). Used in open-pipe-resonance problems.")

add("sound_sound_speed_gas",
    "Speed of Sound in a Gas",
    "Speed of sound in an ideal gas: v = sqrt(gamma*P/rho), where gamma is the adiabatic index (ratio of specific heats), P is the pressure, and rho is the density. Equivalently, v = sqrt(gamma*R*T/M) using ideal gas law substitution. Apply when the question concerns the speed of sound in air or another gas, given its pressure, density, or temperature. The concept is the Newton-Laplace formula for sound speed in a compressible gas, treating compression as adiabatic. Distinct from speed of sound in solids (which uses Young's modulus) or liquids (bulk modulus). Used in sound-speed-in-gas problems.")

# ═════════════════════════════════════════════════════════════════════════════
# surface_tension (5)
# ═════════════════════════════════════════════════════════════════════════════

add("surface_tension_capillary_rise",
    "Capillary Rise in a Narrow Tube",
    "Height of liquid rise in a capillary tube of radius r in a liquid of density rho and surface tension sigma, with contact angle theta: h = 2*sigma*cos(theta)/(rho*g*r). Apply when the question concerns liquid rising (or falling, for non-wetting) in a thin tube, plant capillaries, soil water transport. The concept is surface-tension-driven liquid column height. Distinct from hydrostatic pressure (which doesn't involve a meniscus) and from buoyancy. Used in capillary-rise problems.")

add("surface_tension_excess_pressure_bubble",
    "Excess Pressure Inside a Soap Bubble",
    "Excess pressure inside a soap bubble (two surfaces): P = 4*sigma/r. Apply when the question concerns the pressure difference between the inside and outside of a soap bubble. The concept is that a soap bubble has two surfaces (inner and outer), so the excess pressure is twice that of a single liquid drop. Distinct from the excess pressure of a single-surface liquid drop (P = 2*sigma/r) and from hydrostatic pressure. Used in soap-bubble problems.")

add("surface_tension_excess_pressure_drop",
    "Excess Pressure Inside a Liquid Drop",
    "Excess pressure inside a single-surface liquid drop (a small sphere of liquid in air, not a bubble): P = 2*sigma/r. Apply when the question concerns the pressure inside a liquid droplet, raindrop, or any single-surface curved liquid mass. The concept is surface tension-driven pressure increase in a curved liquid surface. Distinct from a soap bubble (which has two surfaces, hence P = 4*sigma/r). Used in liquid-drop problems.")

add("surface_tension_surface_energy",
    "Surface Energy",
    "Surface energy of a liquid film of area A with surface tension sigma: W = sigma*A. Apply when the question concerns the work required to create a new surface area against surface tension — splitting a drop into smaller drops, expanding a soap film. The concept is energy per unit area associated with creating new surface. Distinct from the force form of surface tension (force per unit length) and from elastic energies. Used in problems involving surface-area changes.")

add("surface_tension_surface_force",
    "Surface Tension Force on a Boundary",
    "Force exerted by surface tension across a contact line of length L: F = sigma*L, where sigma is the surface tension coefficient. Apply when the question concerns the force a liquid surface exerts along an edge — a needle floating on water, an insect walking on water, a wire frame supporting a soap film. The concept is force per unit boundary length from surface tension. Distinct from surface energy (energy per area) and from pressure. Used in problems involving the force from a liquid's surface.")

# ═════════════════════════════════════════════════════════════════════════════
# thermal_physics (6)
# ═════════════════════════════════════════════════════════════════════════════

add("thermal_physics_heat_capacity",
    "Heat Capacity (Lumped Form)",
    "Heat required to change a body's temperature by Delta_T: Q = C*Delta_T, where C is the body's heat capacity (J/K). Apply when the question concerns heating or cooling a body whose heat capacity is given as a single property (not material specific heat times mass). The concept is the body's lumped thermal capacity. Distinct from specific heat (which is per unit mass, c, with Q = m*c*Delta_T) and from latent heat (which involves phase change at constant temperature). Used in calorimetry problems where total heat capacity is given.")

add("thermal_physics_latent_heat",
    "Latent Heat (Heat for Phase Change)",
    "Heat required for a phase change in mass m at constant temperature: Q = m*L, where L is the latent heat (per unit mass) of fusion (solid-liquid) or vaporization (liquid-gas). Apply when the question concerns ice melting, water boiling, or any phase transition. The concept is heat that converts phase without changing temperature. Distinct from sensible heating (Q = m*c*Delta_T, which changes T) and from heat capacity formulas. Used in problems with phase changes (melting, boiling, condensation, freezing).")

add("thermal_physics_linear_expansion",
    "Linear Thermal Expansion",
    "Change in length of a solid body with temperature change Delta_T: Delta_L = L_0*alpha*Delta_T, where alpha is the linear thermal expansion coefficient. Apply when the question concerns thermal stretching of a rod, wire, or beam — bridge expansion joints, bimetallic strips, length-measurement corrections. The concept is first-order linear thermal expansion. Distinct from volumetric expansion (Delta_V = V*beta*Delta_T, beta ≈ 3*alpha for isotropic solids) and from elastic-deformation expansion. Used in thermal-stretching problems.")

add("thermal_physics_newton_cooling",
    "Newton's Law of Cooling",
    "Newton's law of cooling: the rate of temperature change of a body is proportional to its excess over the surrounding temperature. Often expressed as a temperature decay: Delta_T(t) = Delta_T_0 * exp(-k*t), where k is a cooling constant. Apply when the question concerns how a hot body cools over time in a constant-temperature environment. The concept is exponential approach to surroundings temperature. Distinct from radiative cooling (Stefan-Boltzmann, T^4 dependence) and from conductive heat transfer formulas. Used in cooling-curve problems.")

add("thermal_physics_specific_heat",
    "Specific Heat (Material Property)",
    "Heat needed to change a mass m of a substance by Delta_T degrees, given the substance's specific heat c (J/kg/K): Q = m*c*Delta_T. Apply when the question concerns heating or cooling a body of known mass and known specific heat — water in a calorimeter, a metal block heated. The concept is material-specific thermal capacity per unit mass. Distinct from heat capacity (lumped C, no separate mass) and from latent heat (phase change at constant T). Used in calorimetry problems.")

add("thermal_physics_volume_expansion",
    "Volumetric Thermal Expansion",
    "Change in volume of a body with temperature change: Delta_V = V_0*beta*Delta_T, where beta is the volumetric thermal expansion coefficient. Apply when the question concerns thermal expansion of fluids or volumetric expansion of solids — liquid in a thermometer, gas in a balloon, hot oil overflowing. The concept is first-order volumetric thermal expansion. For isotropic solids, beta ≈ 3*alpha (linear coefficient). Distinct from linear expansion (length change) and from ideal-gas expansion formulas. Used in volumetric-expansion problems.")

# ═════════════════════════════════════════════════════════════════════════════
# thermodynamics (7)
# ═════════════════════════════════════════════════════════════════════════════

add("thermodynamics_adiabatic_relation",
    "Adiabatic P-V Relation for Ideal Gas",
    "For an ideal gas undergoing an adiabatic process (no heat exchange): P*V^gamma = constant, where gamma is the adiabatic index. Apply when the question concerns a gas compressed or expanded rapidly enough to prevent heat flow — sound waves, fast piston motion, atmospheric adiabatic processes. The concept is the P-V relation for a reversible adiabatic process. Distinct from isothermal (P*V = constant), isobaric (P = constant), and isochoric (V = constant) processes. Used in adiabatic-process problems.")

add("thermodynamics_carnot_efficiency",
    "Carnot Efficiency (Maximum Heat-Engine Efficiency)",
    "Maximum efficiency of a heat engine operating between hot reservoir at T_h and cold reservoir at T_c (both in kelvin): eta = 1 - T_c/T_h. Apply when the question concerns the theoretical maximum efficiency of a heat engine, or a Carnot cycle specifically. The concept is the absolute upper limit on heat-engine efficiency, set by the temperature ratio. Distinct from real-engine efficiency (which is always less) and from refrigerator coefficient of performance. Used in Carnot-cycle and efficiency-limit problems.")

add("thermodynamics_efficiency_engine",
    "Efficiency of a Heat Engine (Work Done over Heat Input)",
    "Efficiency of a heat engine: eta = W/Q_h, where W is the net work output and Q_h is the heat absorbed from the hot reservoir. Apply when the question concerns the real efficiency of a heat engine, asking for any of {eta, W, Q_h} given the others. The concept is the practical fraction of input heat converted to useful work. Distinct from Carnot efficiency (the theoretical maximum) and from the coefficient of performance for refrigerators. Used in any heat-engine efficiency calculation.")

add("thermodynamics_first_law",
    "First Law of Thermodynamics (Energy Conservation)",
    "First law of thermodynamics: the heat added to a system equals the change in its internal energy plus the work it does, Q = Delta_U + W. (The graph's variable scheme uses DeltaV in place of DeltaU here — read as 'change in internal energy', not change in volume.) Apply when the question concerns energy bookkeeping in a thermodynamic process — gas in a cylinder being heated and doing work, isothermal expansion, adiabatic compression. The concept is energy conservation for a thermodynamic system. Distinct from the second law (entropy considerations) and from specific-process formulas. Used as the foundational equation in any thermodynamics problem.")

add("thermodynamics_ideal_gas",
    "Ideal Gas Law",
    "Ideal gas law: P*V = n*R*T, where P is pressure, V is volume, n is moles of gas, R is the universal gas constant, T is absolute temperature (kelvin). Apply when the question concerns the state of an ideal gas — relating pressure, volume, temperature, and amount, in any equilibrium configuration. The concept is the equation of state for an ideal gas. Distinct from van der Waals (which adds real-gas corrections) and from process-specific relations (isothermal, adiabatic, etc., which are particular paths through P-V-T space). Used in any ideal-gas problem.")

add("thermodynamics_isothermal_work",
    "Work Done in Isothermal Expansion/Compression of an Ideal Gas",
    "Work done by n moles of ideal gas in an isothermal (constant temperature) process from initial volume V_i to final volume V_f: W = n*R*T*ln(V_f/V_i). Apply when the question concerns the work done by or on a gas held at constant temperature during expansion or compression. The concept is the integral of P dV at constant T, using the ideal gas law. Distinct from adiabatic work (no heat exchange), isobaric work (W = P*Delta_V, constant pressure), and the bare work-energy relation. Used in isothermal-process problems.")

add("thermodynamics_work_constant_pressure",
    "Work at Constant Pressure",
    "Work done by a gas in a process at constant pressure: W = P*Delta_V, where Delta_V is the volume change. Apply when the question concerns a gas expanding or being compressed at constant pressure — heating a gas in an open piston cylinder, atmospheric processes at uniform pressure. The concept is the simplest case of work done by an expanding gas — pressure times volume change. Distinct from isothermal work (constant T, uses logarithm) and from adiabatic work (no heat exchange, different relation). Used in isobaric-process problems.")

# ═════════════════════════════════════════════════════════════════════════════
# wave_optics (7)
# ═════════════════════════════════════════════════════════════════════════════

add("wave_optics_constructive_interference",
    "Constructive Interference Condition",
    "Constructive interference occurs when the path difference between two coherent waves equals an integer number of wavelengths: Delta_L = n*lambda, n = 0, 1, 2, .... Apply when the question concerns where bright fringes occur in a two-slit or thin-film interference pattern. The concept is the path-difference condition for waves to reinforce each other. Distinct from destructive interference (Delta_L = (n+1/2)*lambda, dark fringes) and from path-difference formulas tied to geometry. Used in interference-position problems.")

add("wave_optics_destructive_interference",
    "Destructive Interference Condition",
    "Destructive interference occurs when the path difference between two coherent waves is a half-integer multiple of the wavelength: Delta_L = (n+1/2)*lambda, n = 0, 1, 2, .... Apply when the question concerns where dark fringes occur in a two-slit or thin-film interference pattern. The concept is the path-difference condition for waves to cancel. Distinct from constructive interference (Delta_L = n*lambda, bright fringes). Used in dark-fringe-position problems.")

add("wave_optics_malus_law",
    "Malus's Law (Polarization Transmission)",
    "Intensity of light transmitted through a polarizer when the incident polarization makes angle theta with the transmission axis: I = I_0 * cos^2(theta). Apply when the question concerns polarized light passing through one or more polarizers — sunglasses, polaroid filters, optical experiments with polarization. The concept is intensity loss in polarization transmission. Distinct from Brewster's angle (which deals with polarization upon reflection) and from interference. Used in polarization problems involving an analyzer.")

add("wave_optics_path_difference",
    "Path Difference in a Two-Slit Geometry",
    "Geometric path difference between two slit-paths to a screen point at angle theta: Delta_L = d*sin(theta), where d is the slit separation. Apply when the question concerns the geometric path-difference in Young's double-slit or similar two-source setups. The concept is the geometric quantity that, combined with constructive/destructive conditions, gives the positions of fringes. Distinct from the constructive/destructive conditions themselves (which combine this with the wavelength) and from path differences in thin-film geometries. Used in two-slit geometry calculations.")

add("wave_optics_polarization_brewster",
    "Brewster's Angle for Complete Polarization on Reflection",
    "Brewster's angle theta_B at which reflected light is completely polarized: tan(theta_B) = mu_r, where mu_r is the refractive index of the second medium relative to the first. Apply when the question concerns the angle of incidence at which reflected light is fully polarized — anti-glare polarizing-filter design, polarizing sunglasses. The concept is the unique incidence angle where the reflected ray and refracted ray are perpendicular, eliminating one polarization component from the reflection. Distinct from the critical angle (TIR) and from Snell's law (refraction). Used in Brewster-angle problems.")

add("wave_optics_single_slit_minima",
    "Single-Slit Diffraction Minima",
    "Positions of minima in single-slit diffraction: a*sin(theta) = n*lambda, n = 1, 2, 3, ..., where a is the slit width. Apply when the question concerns dark fringes in a single-slit diffraction pattern. (The graph uses d as the slit-width symbol — read as a.) The concept is the diffraction-minima condition from single-slit interference. Distinct from double-slit fringe patterns and from constructive-interference conditions. Used in single-slit diffraction problems.")

add("wave_optics_young_fringe_width",
    "Fringe Width in Young's Double-Slit Experiment",
    "Spacing between adjacent fringes in Young's double-slit experiment: beta = lambda*D/d, where lambda is wavelength, D is distance to screen, d is slit separation. Apply when the question asks for the fringe width or spacing between bright (or dark) fringes in a double-slit pattern. The concept is the geometric spacing of fringes from the interference geometry. Distinct from the constructive/destructive interference conditions (which give positions of single fringes) and from single-slit diffraction widths. Used in fringe-width problems.")

# ═════════════════════════════════════════════════════════════════════════════
# waves (7)
# ═════════════════════════════════════════════════════════════════════════════

add("waves_angular_frequency",
    "Angular Frequency from Ordinary Frequency",
    "Angular frequency from ordinary frequency: omega = 2*pi*f. Apply when the question relates angular frequency and ordinary frequency, or converts between them. The concept is the radians-per-second form of frequency, used in wave equations and oscillation formulas. Distinct from frequency f itself (cycles per second) and from period T = 1/f. Used universally where wave or oscillation math is written in terms of omega.")

add("waves_progressive_wave",
    "Progressive Wave Function",
    "Displacement of a point on a 1D progressive sinusoidal wave: y(x,t) = A*sin(k*x - omega*t + phi), where A is amplitude, k = 2*pi/lambda is wave number, omega is angular frequency, and phi is the initial phase. The minus sign represents a wave moving in +x direction. Apply when the question asks for the displacement at a particular position and time, or for amplitude, wavelength, frequency, or speed of a wave. The concept is the canonical 1D traveling-wave function. Distinct from standing-wave functions (which are products of separate space and time sinusoids) and from non-sinusoidal pulses. Used in traveling-wave time/space problems.")

add("waves_standing_wave_pipe_closed",
    "Standing Wave Frequencies in a Closed Pipe",
    "Resonant frequencies of a pipe closed at one end, open at the other, of length L with sound speed v: f_n = n*v/(4*L), n = 1, 3, 5, ... (odd integers only). The fundamental is n=1, harmonics at n=3,5,7,.... Apply when the question concerns standing waves in a pipe closed at one end — clarinet, closed organ pipe. The concept is odd-harmonic standing waves with a displacement node at the closed end. Distinct from open-open pipe resonances (all harmonics, f_n = n*v/(2*L)) and from string standing waves. Used in closed-pipe-resonance problems.")

add("waves_standing_wave_string",
    "Standing Wave Frequencies on a Stretched String",
    "Resonant frequencies of a stretched string of length L fixed at both ends with wave speed v: f_n = n*v/(2*L), n = 1, 2, 3, .... Fundamental at n=1; harmonics at n=2,3,4,.... Apply when the question concerns standing waves on a string — guitar, violin, sonometer. The concept is the integer-harmonic standing-wave pattern with nodes at both fixed ends. Distinct from the open-pipe sound resonance formula (same shape, but the wave speed v is the sound speed in air, here v is the wave speed on the string) and from closed-pipe (odd harmonics only). Used in string-resonance problems.")

add("waves_string_speed",
    "Wave Speed on a Stretched String",
    "Speed of a transverse wave on a stretched string of tension T and linear mass density mu (mass per length): v = sqrt(T/mu). Apply when the question concerns the wave speed on a string given its tension and mass density (or asks for one given the other two). The concept is wave speed set by the restoring force (tension) and inertia (mass density). Distinct from speed of sound in air or other media. Used in string-wave problems.")

add("waves_wave_number",
    "Wave Number from Wavelength",
    "Wave number from wavelength: k = 2*pi/lambda. The wave number is the radians-per-unit-length measure of how rapidly the wave's phase advances in space. Apply when the question relates wave number and wavelength, or uses k in a wave equation like y(x,t) = A*sin(k*x - omega*t). The concept is the spatial-frequency analog of angular frequency (omega = 2*pi*f, which is the temporal version). Distinct from wavelength lambda itself (the spatial period, units of meters) and from angular frequency (which measures temporal repetition). Used in wave equations and dispersion-relation problems.")

add("waves_wave_speed",
    "Wave Speed from Frequency and Wavelength",
    "Speed of a wave: v = f*lambda. Apply when the question relates wave speed to frequency and wavelength — any periodic wave in any medium, including sound, light, water waves, and waves on a string. The concept is the universal wave relation: speed equals frequency times wavelength. It links the temporal property (frequency) to the spatial property (wavelength) through the medium's propagation speed. Distinct from medium-specific speed formulas — wave on a string (v = sqrt(T/mu)), sound in a gas (v = sqrt(gamma*P/rho)), light in vacuum (c) — which determine the speed from physical properties of the medium. This relation, by contrast, just connects f, lambda, and v without needing to compute v from medium properties. Used in nearly every wave problem to bridge frequency and wavelength.")

# ═════════════════════════════════════════════════════════════════════════════
# work_energy_power (6 — gravitational_PE, kinetic_energy already done)
# ═════════════════════════════════════════════════════════════════════════════

add("work_energy_power_efficiency",
    "Efficiency (Power or Energy Output over Input)",
    "Efficiency: ratio of useful output to total input — eta = output / input, expressed as a fraction or percentage. (The graph stores it in P/W form; read this as 'output / input' generally.) Apply when the question concerns the fraction of input energy or power that becomes useful output — engines, motors, electrical devices. The concept is the dimensionless ratio of useful to total. Distinct from Carnot efficiency (a specific theoretical limit for heat engines) and from absolute power or energy quantities. Used in efficiency-calculation problems.")

add("work_energy_power_power_average",
    "Average Power (Work over Time)",
    "Average power: total work done divided by the time taken, P = W/t. Apply when the question gives total work and total time (or relates two of {power, work, time}) and asks for the third, under conditions where the instantaneous power may vary but only the average matters. The concept is the time-averaged rate of energy transfer over a finite interval. Distinct from instantaneous power (P = F*v, applies at a specific moment when force and velocity are known together) and from electrical power formulas (P = V*I, applies in circuits). Same dimensions and units as instantaneous power but conceptually different — averaging vs. instant. Used in mechanical-power problems where total work and elapsed time are the natural givens.")

add("work_energy_power_power_instantaneous",
    "Instantaneous Mechanical Power",
    "Instantaneous mechanical power delivered by a force F to a body moving at velocity v in the direction of the force: P = F*v. Apply when the question asks for the instantaneous power at a particular speed — a car at cruising velocity against air drag, a winch pulling a load. The concept is the instant rate of energy transfer by a force acting on a moving body. Distinct from average power (P = W/t, time-averaged) and from electrical power (P = V*I, in circuits). Used in instantaneous-power problems involving force and velocity together.")

add("work_energy_power_spring_energy",
    "Elastic Potential Energy of a Spring (Hookean)",
    "Elastic potential energy stored in a Hookean spring stretched or compressed by displacement x from equilibrium: U = (1/2)*k*x^2. Apply when the question concerns the energy stored in a spring at a given deformation. The concept is restoring-force-derived spring PE. Distinct from gravitational PE (m*g*h) and from kinetic energy. In SHM problems, pairs with kinetic energy in energy conservation. Used in spring-energy and SHM-energy problems.")

add("work_energy_power_work_constant_force",
    "Work Done by a Constant Force on a Displaced Body",
    "Work done by a constant force F on a body undergoing displacement s at angle theta between the force and displacement vectors: W = F*s*cos(theta). Apply when the question gives a constant force, the displacement of the body, and the angle between them, and asks for the work done. The concept is the dot product of force and displacement (component of force along motion). Distinct from the work-energy theorem (which relates work to KE change) and from non-constant-force work (which would require integration). Used in mechanical-work problems with constant force.")

add("work_energy_power_work_energy_theorem",
    "Work-Energy Theorem",
    "Work-energy theorem: the net work done on a body equals its change in kinetic energy, W_net = Delta_K. Apply when the question relates the net work done on a body to its initial and final speeds, or asks for the work required to accelerate (or decelerate) a body to a given speed. The concept is the direct equivalence between net work and change in kinetic energy. Distinct from energy conservation involving potential energy (which extends this to non-conservative cases) and from impulse-momentum theorem (which is about momentum, not energy). Used in problems where speed change and work are linked.")

# ═════════════════════════════════════════════════════════════════════════════
# Final
# ═════════════════════════════════════════════════════════════════════════════

OUT_PATH = Path(__file__).parent / "rag_text_exemplars.json"

if __name__ == "__main__":
    out = {
        "version": "v7.1.1-all-exemplars-001",
        "principle": (
            "Each rag_text is a unique conceptual identifier for the equation. "
            "The concept distinguishes it from other equations even when symbols, "
            "variables, or surface scenarios overlap. The LLM matches concept-to-"
            "concept: it identifies the physics concept from the question's story, "
            "generates a concept-level search query, and ChromaDB retrieves by "
            "conceptual similarity, not by keyword overlap."
        ),
        "exemplars": EXEMPLARS,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {len(EXEMPLARS)} exemplars to {OUT_PATH}")
    print(f"  Total characters: {sum(len(v['rag_text']) for v in EXEMPLARS.values())}")
    print(f"  Avg per equation: {sum(len(v['rag_text']) for v in EXEMPLARS.values()) // len(EXEMPLARS)} chars")
