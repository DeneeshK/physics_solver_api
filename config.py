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

# ── ChromaDB retrieval ────────────────────────────────────────────────────────
# Restored in v7: ChromaDB landing is the *initial* equation-finding step,
# per the original architecture brief. Once landed, frontier expansion uses
# the deterministic symbol→equations index in graph_loader, NOT ChromaDB.
COLLECTION_NAME    = "physics_equations"
EMBED_MODEL        = "BAAI/bge-large-en-v1.5"
BGE_QUERY_PREFIX   = "Represent this sentence for searching relevant passages: "
HYBRID_ALPHA       = 0.6
RAG_TOP_K          = 5     # Stage 2 landing: top-K seed equations from ChromaDB
GRAPH_HOPS         = 1     # legacy expand_neighbors hop count (unused by frontier_resolver)

# Feature flag: ChromaDB landing is additive — when enabled and the index
# exists, semantic-matched candidates are unioned with symbol-matched
# candidates for the initial target. When disabled or the index isn't built,
# v6 behavior (pure symbol lookup) is preserved exactly. This is the safety
# property: enabling Chroma can only ADD candidates the LLM can see; it
# never removes one v6 would have shown.
ENABLE_CHROMA_LANDING = os.getenv("ENABLE_CHROMA_LANDING", "auto").lower()
# "auto" → use Chroma if the index exists on disk, otherwise silently fall back
# "true" / "1" / "yes" → require Chroma; raise if the index isn't there
# "false" / "0" / "no" → never use Chroma (force v6 behavior)

# ── Frontier resolver ─────────────────────────────────────────────────────────
MAX_CHAIN_DEPTH = 6   # max frontier resolution rounds

# Safety valve: rough token budget (chars // 4) for a single batched round's
# candidate payload. If domain-filtered candidates for a round still estimate
# above this, frontier_resolver splits that round into sequential per-symbol
# calls instead of one batched call, rather than risking an oversized request.
MAX_CANDIDATES_TOKENS_PER_ROUND = 4000

# Physical constants — never treated as unknowns; always excluded from
# the frontier. The frontier_resolver checks this set when deciding what
# variables it still needs to solve for; sympy_executor uses the catalog
# below for numeric substitution.
#
# v7 reconciliation note: this set must contain EVERY symbol the graph
# actually uses for a physical constant. v6 missed the underscoreless
# forms ('epsilon0', 'mu0') that the graph file uses, which meant the
# frontier would try to "solve for" epsilon0 in any electrostatics
# question. Both forms are listed here in case any equation file still
# uses the old name; the graph itself has been standardized to the
# underscored form ('epsilon_0', 'mu_0') in v7.
PHYSICAL_CONSTANTS = {
    'g', 'pi', 'c',
    'h_planck', 'k_B', 'R_g', 'NA', 'N_A',
    'epsilon_0', 'epsilon0',
    'mu_0', 'mu0',
    'e_charge',
    'G',
}

# True universal constants — context-independent, same value everywhere,
# always safe to surface as "already known" without per-question judgment.
# 'g' is deliberately EXCLUDED (Earth-surface, not a constant of nature).
UNIVERSAL_CONSTANTS = {
    'pi', 'c', 'G',
    'epsilon_0', 'mu_0',
    'h_planck', 'k_B', 'R_g', 'NA',
    'e_charge',
}

# Equations containing these symbols are conservation-law forms —
# cannot be rearranged for a specific numerical value, skip during resolution.
NON_SOLVABLE_SYMBOLS = {'constant'}

# ── Groq API ──────────────────────────────────────────────────────────────────
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
MODEL_FAST       = "llama-3.1-8b-instant"      # Stage 1 parse + (default) Stage 2 selection
MODEL_SMART      = "llama-3.3-70b-versatile"   # Stage 4 narration + Stage 5 distractors

# v7.1.2: Stage 2 (round_selector) model is now overridable. The default is
# MODEL_FAST for cost/latency, but concept-level rag_texts (v7.1+) demand
# more reasoning than the 8B model reliably delivers. If you see
# `llm_omitted_item` or `llm_decision_none` fallbacks where the right
# equation was clearly in the candidates, switch to MODEL_SMART:
#
#   STAGE2_MODEL=llama-3.3-70b-versatile  (set in .env)
#
# Trade-off: ~5–10x slower, ~5x cost. But the v6 8B-friendly architecture
# was relying on cheap pattern-matching against templated rag_texts, and
# that escape hatch is gone in v7.1+ by design.
import os
STAGE2_MODEL     = os.getenv("STAGE2_MODEL", MODEL_FAST)

# v7.1.3: Stage 2 batching strategy. The default is "auto" — single-item
# rounds use one batched call (cheap, fast), multi-item rounds split into
# one LLM call per frontier item.
#
# Why this exists: chained problems (e.g. F=ma where m comes from rho*V and
# a comes from kinematics) require Round 1+ to ask about multiple symbols
# at once. The 8B fast model handles the first item well, then loses the
# second item off its attention budget — producing `llm_omitted_item`
# events. Per-item batching prevents this because each call asks about
# exactly one symbol.
#
# Modes:
#   "auto"   — single if len(round_data) > 1, else all (default)
#   "all"    — one LLM call covering all items (v7.1.2 and earlier behavior)
#   "single" — one LLM call per item, regardless of count
STAGE2_BATCH_MODE = os.getenv("STAGE2_BATCH_MODE", "auto")
GROQ_TEMPERATURE = 0.1

# ── Implicit constants catalog ─────────────────────────────────────────────────
# Stage 1 uses this to inject constants that are implied by the scenario
# (e.g. "in vacuum" → epsilon_0; "free fall" → g).
# symbol → {value (SI), unit, name, dimension, cue: scenario keywords}
#
# IMPORTANT: the keys here are the CANONICAL names. The graph has been
# standardized in v7 to use these same names. If you ever add a new
# physical constant, add it here AND ensure the graph uses the same symbol.
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
        "name": "universal gas constant", "dimension": "ML2T-2N-1Theta-1",
        "cue": "ideal gas, molar, PV=nRT",
    },
    "NA": {
        "value": 6.022e23, "unit": "mol^-1",
        "name": "Avogadro number", "dimension": "N-1",
        "cue": "moles, molecules, atoms per mole",
    },
    "e_charge": {
        "value": 1.6e-19, "unit": "C",
        "name": "elementary charge", "dimension": "AT",
        "cue": "electron charge, proton charge, eV, elementary charge",
    },
    "pi": {
        "value": 3.141592653589793, "unit": "",
        "name": "pi", "dimension": "1",
        "cue": "circle, sphere, cylinder, angular, period",
    },
}
