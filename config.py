"""
config.py  —  Central configuration for Physics Solver
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
MAIN_GRAPH_PATH = DATA_DIR / "physics_equation_graph_final.json"
CHROMA_DIR      = str(BASE_DIR / "chroma_db")
BM25_INDEX_PATH = str(BASE_DIR / "bm25_index.pkl")

# ── ChromaDB (optional, kept for legacy) ──────────────────────────────────────
COLLECTION_NAME = "physics_equations"
EMBED_MODEL     = "BAAI/bge-large-en-v1.5"
HYBRID_ALPHA    = 0.6
RAG_TOP_K       = 5
GRAPH_HOPS      = 1

# ── Frontier resolver ─────────────────────────────────────────────────────────
MAX_CHAIN_DEPTH = 6   # max frontier resolution rounds

# Physical constants — never treated as unknowns; always excluded from
# the frontier (the resolver will never try to "solve for" any of these).
PHYSICAL_CONSTANTS = {
    'g', 'pi', 'c', 'h_planck', 'k_B', 'R_g',
    'NA', 'epsilon_0', 'mu_0', 'e_charge', 'G',
}

# True universal constants — context-independent, same value everywhere,
# always safe to surface as "already known" without per-question judgment.
# 'g' is deliberately EXCLUDED: it is Earth-surface gravitational
# acceleration, not a constant of nature (compare to G, which is universal).
# g must only become available when Stage 1 actually judges the scenario
# implies it (e.g. "free fall", "dropped", "projectile") — never by default.
UNIVERSAL_CONSTANTS = PHYSICAL_CONSTANTS - {'g'}

# Equations containing these symbols are conservation-law forms —
# cannot be rearranged for a specific numerical value, skip during resolution.
NON_SOLVABLE_SYMBOLS = {'constant'}

# ── Groq API ──────────────────────────────────────────────────────────────────
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
MODEL_FAST       = "llama-3.1-8b-instant"      # Stage 1 parse + Stage 2 selection
MODEL_SMART      = "llama-3.3-70b-versatile"   # Stage 4 narration + Stage 5 distractors
GROQ_TEMPERATURE = 0.1

# ── Implicit constants catalog ─────────────────────────────────────────────────
# Stage 1 uses this to inject constants that are implied by the scenario
# (e.g. "in vacuum" → epsilon_0; "free fall" → g).
# symbol → {value (SI), unit, name, dimension, cue: scenario keywords}
IMPLICIT_CONSTANTS_CATALOG: dict[str, dict] = {
    "g": {
        "value": 9.8, "unit": "m/s^2",
        "name": "gravitational acceleration", "dimension": "LT-2",
        "cue": "free fall, dropped, thrown, vertical, weight, gravity",
    },
    "G": {
        "value": 6.674e-11, "unit": "N·m^2/kg^2",
        "name": "universal gravitational constant", "dimension": "M-1L3T-2",
        "cue": "gravitation, orbit, planetary, universal law of gravitation",
    },
    "c": {
        "value": 3e8, "unit": "m/s",
        "name": "speed of light", "dimension": "LT-1",
        "cue": "vacuum, light, electromagnetic wave, photon, relativistic, speed of light",
    },
    "epsilon_0": {
        "value": 8.854e-12, "unit": "C^2/(N·m^2)",
        "name": "permittivity of free space", "dimension": "M-1L-3T4A2",
        "cue": "vacuum, free space, Coulomb, electric field in vacuum, capacitance",
    },
    "mu_0": {
        "value": 1.257e-6, "unit": "T·m/A",
        "name": "permeability of free space", "dimension": "MLT-2A-2",
        "cue": "vacuum, magnetic, Ampere law, solenoid in vacuum",
    },
    "h_planck": {
        "value": 6.626e-34, "unit": "J·s",
        "name": "Planck constant", "dimension": "ML2T-1",
        "cue": "photon energy, quantum, de Broglie, photoelectric, wavelength of photon",
    },
    "k_B": {
        "value": 1.38e-23, "unit": "J/K",
        "name": "Boltzmann constant", "dimension": "ML2T-2K-1",
        "cue": "thermal energy, gas, kinetic theory, temperature",
    },
    "R_g": {
        "value": 8.314, "unit": "J/(mol·K)",
        "name": "universal gas constant", "dimension": "ML2T-2K-1mol-1",
        "cue": "ideal gas, molar, PV=nRT",
    },
    "NA": {
        "value": 6.022e23, "unit": "mol^-1",
        "name": "Avogadro number", "dimension": "mol-1",
        "cue": "moles, molecules, atoms per mole",
    },
    "e_charge": {
        "value": 1.6e-19, "unit": "C",
        "name": "elementary charge", "dimension": "AT",
        "cue": "electron charge, proton charge, eV, elementary charge",
    },
    "pi": {
        "value": 3.141592653589793, "unit": "",
        "name": "pi", "dimension": "dimensionless",
        "cue": "circle, sphere, cylinder, angular, period",
    },
}
