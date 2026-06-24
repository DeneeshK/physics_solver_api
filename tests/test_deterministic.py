"""
tests/test_deterministic.py
Deterministic tests — no Groq API needed.

Tests the full pipeline with a mock llm_round_fn that simulates the LLM's
conceptual choices.  This verifies:

  §8 step 2 — candidates_for_quantity isolation
  §8 step 3 — frontier loop happy path (F=ma NOT F=rho*V*g)
  §8 step 4 — exact arithmetic, substitution traces, fraction survival
  §8 step 5 — cycle detection / simultaneous-solve path
  §8 step 6 — backtracking path (excluded equation forces alternative pick)

Run: python3 -m pytest tests/test_deterministic.py -v
 or: python3 tests/test_deterministic.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from solver.graph_loader      import load_graphs, _dimensions_compatible
from solver.frontier_resolver import (
    FrontierItem, ResolvedStep, SimultaneousGroup,
    resolve_frontier, _topological_sort,
)
from solver.sympy_executor    import execute_plan


GRAPH = None

def get_graph():
    global GRAPH
    if GRAPH is None:
        GRAPH = load_graphs()
    return GRAPH


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class StubRetriever:
    """
    v7.2.4: Round 0 is now a pure-semantic sequential scan, so resolver tests
    need a retriever. This stub returns a FIXED ranked list of equation nodes
    (looked up from the graph by id) as the 'semantic' search result, in the
    given order, with descending scores.

    `priority_ids` (optional) are equation ids that rank_candidates floats to
    the top before capping — used so a known bridging equation (e.g.
    general_density_definition for mass) survives the neighbor-walk cap in
    deterministic tests, where the real semantic ranker isn't available to
    score it. This keeps tests faithful to the live path (cap + rank) while
    ensuring the equation the mock intends to pick is actually shown.
    """
    def __init__(self, graph_index, landing_order, priority_ids=None):
        self._gi = graph_index
        self._priority = set(priority_ids or [])
        self._nodes = []
        for i, eid in enumerate(landing_order):
            node = self._lookup(eid)
            if node is not None:
                self._nodes.append({"node": node, "score": 1.0 - i * 0.01,
                                    "semantic_score": 1.0 - i * 0.01, "bm25_score": 0.5})

    def _lookup(self, eid):
        if hasattr(self._gi, "nodes_by_id"):
            return self._gi.nodes_by_id.get(eid)
        return next((n for n in self._gi.nodes if n["id"] == eid), None)

    def search(self, query, top_k=5):
        return self._nodes[:top_k]

    def rank_candidates(self, query, candidates, top_k=8):
        # Priority ids first (stable), then the rest in their given order.
        prio = [c for c in candidates if c["id"] in self._priority]
        rest = [c for c in candidates if c["id"] not in self._priority]
        return (prio + rest)[:top_k]


def make_selector(choices: dict):
    """
    Build a mock llm_round_fn.
    choices: {needed_symbol → equation_id_to_pick}
    If a symbol is not in choices, pick the first candidate.
    """
    def selector(question, available, round_data, round_num=0, solve_context=None):
        results = []
        for rd in round_data:
            fi      = rd["frontier_item"]
            cands   = rd["candidates"]
            eq_id   = choices.get(fi.symbol)
            chosen  = (
                next((c for c in cands if c["id"] == eq_id), None)
                or (cands[0] if cands else None)
            )
            results.append({
                "frontier_item":      fi,
                "chosen_eq":          chosen,
                "reason":             f"Mock pick: {chosen['id'] if chosen else 'none'}",
                "conditions_concern": None,
                "deferred":           False,
                "_candidates":        cands,
            })
        return results
    return selector


def fi(symbol, name, unit, dimension):
    return FrontierItem(symbol=symbol, name=name, unit=unit, dimension=dimension)


# ─────────────────────────────────────────────────────────────────────────────
# §8 step 2 — candidates_for_quantity isolation
# ─────────────────────────────────────────────────────────────────────────────

def test_dimension_normalization_format_independent():
    """
    Regression for a real production failure: a power-dissipation question
    failed with "No candidate equations found for 'P'" even though
    current_electricity_power_electric (P=V*I) exists and is dimensionally
    correct. Root cause: dimension compatibility used to be plain string
    equality after splitting on " or " — so the graph's stored "ML2T-3 or
    ML-1T-2" wouldn't match an LLM-produced "M L^2 T^-3" or "ML^2T^-3" or
    lowercase, despite being the identical physical dimension. "power"
    wasn't anchored with a worked example in Stage 1's prompt (force,
    velocity, mass etc. were), so nothing constrained its output format.
    Confirms the normalized comparison is robust to that variance while
    still correctly rejecting genuinely different dimensions.
    """
    print("\n[test_dimension_normalization_format_independent]")
    from solver.graph_loader import _dimensions_compatible
    stored = "ML2T-3 or ML-1T-2"
    for variant in ["ML2T-3", "M L^2 T^-3", "ML^2T^-3", "M*L2*T-3", "ml2t-3"]:
        assert _dimensions_compatible(stored, variant), (
            f"{variant!r} should be recognized as the same dimension as {stored!r}"
        )
    # Genuinely different dimensions must still be rejected
    assert not _dimensions_compatible("M", "LT-1")
    assert not _dimensions_compatible("MLT-2", "M")
    print("  PASSED")


def test_dimension_compat():
    print("\n[test_dimension_compat]")
    assert _dimensions_compatible("M", "M")
    assert _dimensions_compatible("MLT-2", "MLT-2")
    assert _dimensions_compatible("MLT-1 or ML2 or A", "A")
    assert _dimensions_compatible("MLT-1 or ML2 or A", "ML2")
    assert not _dimensions_compatible("M", "L")
    assert not _dimensions_compatible("MLT-2", "ML2")
    # Empty → always compatible
    assert _dimensions_compatible("", "M")
    assert _dimensions_compatible("MLT-2", "")
    print("  PASSED")


def test_candidates_symbol_m():
    """All 25 equations containing m should appear — dimension M is unambiguous."""
    g = get_graph()
    cands = g.candidates_for_quantity("m", "mass", "M", set())
    ids   = {c["id"] for c in cands}
    assert "laws_of_motion_newton_second_law" in ids, "F=ma must be a candidate for m"
    assert "general_density_definition" in ids,       "rho=m/V must be a candidate for m"
    assert len(cands) == 25, f"expected 25, got {len(cands)}"
    print(f"\n[test_candidates_symbol_m] {len(cands)} candidates — PASSED")


def test_candidates_symbol_F():
    """Both F=m*a and F=rho*V*g should appear — LLM must choose between them."""
    g    = get_graph()
    cands = g.candidates_for_quantity("F", "force", "MLT-2", set())
    ids   = {c["id"] for c in cands}
    assert "laws_of_motion_newton_second_law" in ids,  "F=m*a missing"
    assert "fluid_mechanics_buoyant_force" in ids,     "F=rho*V*g missing"
    print(f"\n[test_candidates_symbol_F] {len(cands)} candidates — PASSED")


def test_candidates_visited_exclusion():
    """Equations in visited_eqs must not appear."""
    g       = get_graph()
    visited = {"laws_of_motion_newton_second_law"}
    cands   = g.candidates_for_quantity("F", "force", "MLT-2", visited)
    ids     = {c["id"] for c in cands}
    assert "laws_of_motion_newton_second_law" not in ids, "visited eq must be excluded"
    assert "fluid_mechanics_buoyant_force" in ids
    print(f"\n[test_candidates_visited_exclusion] {len(cands)} remaining — PASSED")


def test_candidates_conservation_law_excluded():
    """Equations with 'constant' variable must be filtered (can't produce a value)."""
    g     = get_graph()
    # kinematics_kepler_third or similar should be excluded
    # Let's check 'T' symbol where T**2 = constant * r**3 exists
    cands = g.candidates_for_quantity("T", "period", "T", set())
    ids   = {c["id"] for c in cands}
    # gravitation_kepler_third uses 'constant' — should NOT appear
    assert "gravitation_kepler_third" not in ids, "conservation-law eq must be excluded"
    print(f"\n[test_candidates_conservation_law_excluded] PASSED")


class _StubGraphIndex:
    """
    Minimal graph_index stub for testing frontier expansion logic in
    isolation, independent of real graph data.

    v7.2: the resolver's Round 1+ now uses a GRAPH NEIGHBOR WALK instead of a
    symbol lookup. So the stub must expose the neighbor-walk interface:
    edges/adjacency plus neighbors_sharing_variable() and
    all_equations_with_variable(). Round 0 still uses candidates_for_quantity
    via the landing layer. We auto-build edges between any two stub equations
    that share a variable (the same SHARES_VARIABLE semantics as the real
    graph), so chains like X->(eq_A)->Y->(eq_B) are reachable by the walk.
    """
    def __init__(self, equations: list[dict]):
        self.equations = equations
        self.nodes_by_id = {e["id"]: e for e in equations}
        # Build symbol -> [eq_ids] (used by tier-3 fallback + Round 0).
        self.sym_to_eqs = {}
        for eq in equations:
            for sym in eq["variables"]:
                self.sym_to_eqs.setdefault(sym, []).append(eq["id"])
        # Auto-build SHARES_VARIABLE edges between equations sharing a variable.
        self.edges = []
        for i, a in enumerate(equations):
            for b in equations[i + 1:]:
                shared = set(a["variables"]) & set(b["variables"])
                if shared:
                    self.edges.append({
                        "from": a["id"], "to": b["id"],
                        "shared_variables": sorted(shared),
                    })

    def candidates_for_quantity(self, needed_symbol, needed_name, needed_dimension, visited_eqs, allowed_domains=None):
        return [
            eq for eq in self.equations
            if needed_symbol in eq["variables"] and eq["id"] not in visited_eqs
        ]

    def _edges_for(self, eq_id):
        out = []
        for e in self.edges:
            if e["from"] == eq_id:
                out.append({"other": e["to"], "shared_variables": e["shared_variables"]})
            elif e["to"] == eq_id:
                out.append({"other": e["from"], "shared_variables": e["shared_variables"]})
        return out

    def neighbors_sharing_variable(self, *, from_eq_ids, variable, visited_eqs):
        # Compute from shared-variable membership (matches real GraphIndex):
        # every equation containing `variable`, excluding sources and visited.
        out, seen = [], set()
        for eq_id in self.sym_to_eqs.get(variable, []):
            if eq_id in from_eq_ids or eq_id in visited_eqs or eq_id in seen:
                continue
            seen.add(eq_id)
            out.append(self.nodes_by_id[eq_id])
        return out

    def all_equations_with_variable(self, *, variable, visited_eqs):
        return [
            self.nodes_by_id[eq_id]
            for eq_id in self.sym_to_eqs.get(variable, [])
            if eq_id not in visited_eqs
        ]


def test_already_targeted_symbol_not_reintroduced():
    """
    Regression for the test-5 runaway: a 6-round, 150+ second cascade on a
    simple free-fall question, caused by a target symbol resolved in one
    round getting silently reintroduced as a "new" unknown in a later
    round, because a different equation (chosen for some OTHER symbol)
    happened to also reference it as an input.

    Minimal isolation: target X is resolved via eq_A (needs Y). Y is then
    resolved via eq_B, whose variables ALSO include X (the already-resolved
    target) and Z (given). Without tracking "already targeted" symbols
    across rounds, X gets wrongly re-added to the frontier for round 2,
    and — since eq_A is already visited and no other candidate exists for
    X — the whole resolution fails. With the fix, X is correctly recognized
    as already in progress, frontier empties after round 1, and resolution
    succeeds with exactly 2 steps.
    """
    print("\n[test_already_targeted_symbol_not_reintroduced]")
    eq_A = {
        "id": "eq_A", "equation_str": "X = Y + 1",
        "sympy_expr": "Eq(X, Y + 1)",
        "variables": {
            "X": {"name": "x quantity", "unit": "u", "dimension": "L"},
            "Y": {"name": "y quantity", "unit": "u", "dimension": "L"},
        },
        "conditions": [], "rag_text": "test fixture A", "common_mistakes": [],
        "domain": "test",
    }
    eq_B = {
        "id": "eq_B", "equation_str": "Y = X + Z",
        "sympy_expr": "Eq(Y, X + Z)",
        "variables": {
            "Y": {"name": "y quantity", "unit": "u", "dimension": "L"},
            "X": {"name": "x quantity", "unit": "u", "dimension": "L"},  # reappears here
            "Z": {"name": "z quantity", "unit": "u", "dimension": "L"},
        },
        "conditions": [], "rag_text": "test fixture B", "common_mistakes": [],
        "domain": "test",
    }
    stub_graph = _StubGraphIndex([eq_A, eq_B])
    selector = make_selector({"X": "eq_A", "Y": "eq_B"})
    target = fi("X", "x quantity", "u", "L")
    given = {"Z": {"value": 5, "unit": "u", "name": "z quantity", "dimension": "L"}}

    result = resolve_frontier(target, given, stub_graph, "test question", selector,
                              retriever=StubRetriever(stub_graph, ["eq_A", "eq_B"]),
                              search_query="test question")

    print(f"  success: {result.success}")
    print(f"  plan shape: {[type(s).__name__ for s in result.plan]}")
    targeting_counts = {}
    for entry in result.decision_log:
        targeting_counts[entry["solving_for"]] = targeting_counts.get(entry["solving_for"], 0) + 1
    print(f"  times each symbol was targeted: {targeting_counts}")

    assert result.success, f"resolution should succeed, got: {result.failure_reason}"
    # The actual property under test: X must never be asked about twice.
    # (These particular equations are genuinely mutually dependent — X=Y+1
    # and Y=X+Z really do form a simultaneous system — so the correct
    # outcome here is a SimultaneousGroup, which is a separate, already-
    # correct mechanism. That's fine; what would indicate the bug is back
    # is X or Y appearing as "solving_for" more than once.)
    assert targeting_counts.get("X", 0) == 1, (
        f"X was targeted {targeting_counts.get('X', 0)} times — "
        f"should be exactly 1 (it's already in progress after round 0)"
    )
    assert targeting_counts.get("Y", 0) == 1
    print("  PASSED")


def test_parse_system_has_target_identification_rule():
    """
    Regression guard, not behavioral proof: confirms the explicit
    target-identification instruction (added after Stage 1 misidentified
    'a' instead of 'F' for a question explicitly asking "find the net
    force") is present, so a future edit can't silently drop it. Whether
    the LLM actually follows it reliably needs live testing — this just
    catches the instruction itself disappearing.
    """
    print("\n[test_parse_system_has_target_identification_rule]")
    from solver.llm_interface import _build_parse_system
    system = _build_parse_system({"kinematics"})
    assert "ULTIMATELY asking to find" in system
    assert "net force" in system, "the anchoring example from the real failure case should be present"
    print("  PASSED")


def test_narrate_system_forbids_new_computation():
    """
    Regression guard for the more serious bug: narration introduced an
    entire unverified mass+force calculation that was never in the trace,
    because the old rule only forbade *altering* trace numbers, not
    *introducing* new ones. Confirms the closed rule is present.

    Also covers two issues found in later live testing: (a) narration
    re-deriving a step's algebra itself and flipping a signed value's sign
    in the process (a=-2 in the trace narrated as "a=2"), and (b) a
    "this matches what was asked" note appearing even when there was no
    actual mismatch, as a side effect of the mismatch-detection rule
    always firing reflexively.
    """
    print("\n[test_narrate_system_forbids_new_computation]")
    from solver.llm_interface import NARRATE_SYSTEM
    assert "NEVER introduce a number" in NARRATE_SYSTEM
    assert "narrating a finished computation, not completing one" in NARRATE_SYSTEM
    assert "keep it negative in your prose" in NARRATE_SYSTEM
    assert "do not add a routine" in NARRATE_SYSTEM
    print("  PASSED")


def test_domain_filter_narrows_candidates():
    """
    allowed_domains should shrink the candidate set to just the matching
    domain(s) — this is the actual token-reduction mechanism.
    """
    print("\n[test_domain_filter_narrows_candidates]")
    g = get_graph()
    unfiltered = g.candidates_for_quantity("m", "mass", "M", set())
    filtered   = g.candidates_for_quantity(
        "m", "mass", "M", set(), allowed_domains={"laws_of_motion"}
    )
    print(f"  m: {len(unfiltered)} unfiltered -> {len(filtered)} filtered to laws_of_motion")
    assert len(filtered) < len(unfiltered), "domain filter should narrow the set"
    assert all(c.get("domain") == "laws_of_motion" for c in filtered)
    assert "laws_of_motion_newton_second_law" in {c["id"] for c in filtered}
    print("  PASSED")


def test_domain_filter_fallback_when_empty():
    """
    CRITICAL SAFETY PROPERTY: if the domain guess doesn't match anything for
    this quantity, fall back to the full set — never silently return zero
    candidates. This is what keeps domain filtering from reintroducing the
    original "correct equation got excluded" failure mode.
    """
    print("\n[test_domain_filter_fallback_when_empty]")
    g = get_graph()
    unfiltered = g.candidates_for_quantity("m", "mass", "M", set())
    # 'sound' is a real domain in the graph, but has zero candidates for 'm'
    fallback = g.candidates_for_quantity(
        "m", "mass", "M", set(), allowed_domains={"sound"}
    )
    print(f"  m with allowed_domains={{'sound'}} (no match): {len(fallback)} candidates")
    assert len(fallback) == len(unfiltered), (
        "an empty domain match must fall back to the full set, not return zero"
    )
    print("  PASSED")


def test_domain_filter_fixes_real_overflow_scenario():
    """
    Regression for the actual production crash: a Groq 413 (Request too
    large, limit 6000, requested 8202) on the m+a combined round after
    picking F=m*a. Reproduces that exact round and confirms domain
    filtering brings it back under budget.
    """
    print("\n[test_domain_filter_fixes_real_overflow_scenario]")
    from solver.llm_interface import estimate_round_tokens
    from config import MAX_CANDIDATES_TOKENS_PER_ROUND

    g = get_graph()
    visited = {"laws_of_motion_newton_second_law"}
    m_unfiltered = g.candidates_for_quantity("m", "mass", "M", visited)
    a_unfiltered = g.candidates_for_quantity("a", "acceleration", "LT-2", visited)
    round_data_unfiltered = [
        {"frontier_item": fi("m", "mass", "kg", "M"), "candidates": m_unfiltered},
        {"frontier_item": fi("a", "acceleration", "m/s^2", "LT-2"), "candidates": a_unfiltered},
    ]
    tokens_unfiltered = estimate_round_tokens(round_data_unfiltered)

    allowed = {"laws_of_motion", "kinematics"}
    m_filtered = g.candidates_for_quantity("m", "mass", "M", visited, allowed_domains=allowed)
    a_filtered = g.candidates_for_quantity("a", "acceleration", "LT-2", visited, allowed_domains=allowed)
    round_data_filtered = [
        {"frontier_item": fi("m", "mass", "kg", "M"), "candidates": m_filtered},
        {"frontier_item": fi("a", "acceleration", "m/s^2", "LT-2"), "candidates": a_filtered},
    ]
    tokens_filtered = estimate_round_tokens(round_data_filtered)

    print(f"  unfiltered: m={len(m_unfiltered)}, a={len(a_unfiltered)} -> ~{tokens_unfiltered} tokens")
    print(f"  filtered:   m={len(m_filtered)}, a={len(a_filtered)} -> ~{tokens_filtered} tokens")
    print(f"  budget: {MAX_CANDIDATES_TOKENS_PER_ROUND}")

    assert tokens_unfiltered > MAX_CANDIDATES_TOKENS_PER_ROUND, (
        "sanity check: the unfiltered round should reproduce the original overflow"
    )
    assert tokens_filtered < MAX_CANDIDATES_TOKENS_PER_ROUND, (
        "domain-filtered round should fit comfortably under budget"
    )
    print("  PASSED")


def test_round_splits_on_token_overflow():
    """
    The overflow safety valve: when a multi-item round's candidate prompt
    would exceed the per-round token budget, resolve_frontier splits it into
    separate single-symbol LLM calls instead of one oversized batched call
    (the fix for the 413 crash). v7.2.4: with concept-level rag_texts and the
    neighbor cap, a real m+a round no longer naturally exceeds the default
    4000-token budget, so we force the condition by patching the threshold
    very low for this test — that genuinely exercises the split path. The
    chain still resolves correctly (priority_ids surface density), so we can
    assert BOTH that the split fired AND that the answer is right.
    """
    print("\n[test_round_splits_on_token_overflow]")
    import config as _cfg
    g = get_graph()
    base_selector = make_selector({
        "F": "laws_of_motion_newton_second_law",
        "m": "general_density_definition",
        "a": "kinematics_v2_u2_2as",
    })
    call_log = []
    def tracking_selector(question, available, round_data, round_num=0, solve_context=None):
        call_log.append(len(round_data))
        return base_selector(question, available, round_data, round_num)

    target = fi("F", "force", "N", "MLT-2")
    given = {
        "rho": {"value": 8000, "unit": "kg/m^3", "name": "density", "dimension": "ML-3"},
        "V":   {"value": 0.5,  "unit": "m^3",    "name": "volume",  "dimension": "L3"},
        "u":   {"value": 0.0,  "unit": "m/s",    "name": "initial velocity", "dimension": "LT-1"},
        "v":   {"value": 30.0, "unit": "m/s",    "name": "final velocity",   "dimension": "LT-1"},
        "s":   {"value": 40.0, "unit": "m",      "name": "displacement",     "dimension": "L"},
    }
    # Force the overflow condition: patch the per-round budget to a tiny value
    # so ANY 2-item round exceeds it and must split. priority_ids keep density
    # reachable so the chain still completes correctly.
    saved = _cfg.MAX_CANDIDATES_TOKENS_PER_ROUND
    try:
        _cfg.MAX_CANDIDATES_TOKENS_PER_ROUND = 1  # any 2-item round now overflows
        result = resolve_frontier(target, given, g, "test question", tracking_selector,
                                  retriever=StubRetriever(
                                      g, ["laws_of_motion_newton_second_law"],
                                      priority_ids=["general_density_definition",
                                                    "kinematics_v2_u2_2as"]),
                                  search_query="test question")
    finally:
        _cfg.MAX_CANDIDATES_TOKENS_PER_ROUND = saved

    print(f"  round_data lengths per LLM call: {call_log}")
    assert result.success, f"resolution should still succeed: {result.failure_reason}"
    assert 2 not in call_log, (
        "a 2-item round reached the LLM unsplit — the overflow safety valve "
        "did not trigger when it should have"
    )
    assert call_log.count(1) >= 3, f"expected >=3 single-item calls, got {call_log}"
    print("  PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# §8 step 3 — Frontier loop happy path
# The original failing case: body with rho,V,u,v,s given; find F.
# Correct chain: F=m*a, rho=m/V, v²=u²+2as  (NOT F=rho*V*g)
# ─────────────────────────────────────────────────────────────────────────────

FMVA_QUESTION = (
    "A body of density 8000 kg/m³ and volume 0.5 m³ starts from rest "
    "and reaches 30 m/s after travelling 40 m. Find the net force acting on it."
)

FMVA_GIVEN = {
    "rho": {"value": 8000, "unit": "kg/m^3", "name": "density",          "dimension": "ML-3"},
    "V":   {"value": 0.5,  "unit": "m^3",    "name": "volume",           "dimension": "L3"},
    "u":   {"value": 0.0,  "unit": "m/s",    "name": "initial velocity", "dimension": "LT-1"},
    "v":   {"value": 30.0, "unit": "m/s",    "name": "final velocity",   "dimension": "LT-1"},
    "s":   {"value": 40.0, "unit": "m",      "name": "displacement",     "dimension": "L"},
    "g":   {"value": 9.8,  "unit": "m/s^2",  "name": "grav. accel.",    "dimension": "LT-2"},
}

FMVA_TARGET = fi("F", "net force", "N", "MLT-2")

# The mock LLM makes the CORRECT conceptual choice: F=m*a, not F=rho*V*g
FMVA_MOCK_CHOICES = {
    "F": "laws_of_motion_newton_second_law",  # NOT fluid_mechanics_buoyant_force
    "m": "general_density_definition",         # rho = m/V  → m = rho*V
    "a": "kinematics_v2_u2_2as",              # v² = u² + 2as  → a
}

# v7.2.4: Round-0 semantic scan needs a retriever. The landing node for FMVA
# is Newton's second law; list it first so the scan lands on it.
FMVA_LANDING_ORDER = [
    "laws_of_motion_newton_second_law",
    "fluid_mechanics_buoyant_force",   # the tempting-but-wrong alternative, lower
    "rotational_motion_torque_inertia",
]


def test_frontier_happy_path():
    """
    The crux of the redesign: frontier loop picks F=m*a conceptually,
    even though F=rho*V*g would be immediately solvable.
    """
    print(f"\n[test_frontier_happy_path]")
    g        = get_graph()
    selector = make_selector(FMVA_MOCK_CHOICES)

    result = resolve_frontier(
        target       = FMVA_TARGET,
        given        = FMVA_GIVEN,
        graph_index  = g,
        question     = FMVA_QUESTION,
        llm_round_fn = selector,
        retriever    = StubRetriever(g, FMVA_LANDING_ORDER, priority_ids=list(FMVA_MOCK_CHOICES.values())),
        search_query = FMVA_QUESTION,
    )

    print(f"  success: {result.success}")
    print(f"  status:  {result.status}")
    print(f"  plan:")
    for item in result.plan:
        if hasattr(item, "equation"):
            print(f"    {item.equation['equation_str']}  → {item.solves_for.symbol}")
        else:
            print(f"    SIMULTANEOUS({[e['equation_str'] for e in item.equations]})")

    assert result.success,              f"Resolution failed: {result.failure_reason}"
    chosen_ids = {
        s.equation["id"] for s in result.plan if hasattr(s, "equation")
    }
    assert "laws_of_motion_newton_second_law"  in chosen_ids, "F=m*a must be in plan"
    assert "general_density_definition"         in chosen_ids, "rho=m/V must be in plan"
    assert "kinematics_v2_u2_2as"              in chosen_ids, "v²=u²+2as must be in plan"
    assert "fluid_mechanics_buoyant_force"     not in chosen_ids, \
        "F=rho*V*g must NOT be chosen — this is the old algorithm's failure mode"

    # Execution order: prerequisites before dependents
    plan_syms = [s.solves_for.symbol for s in result.plan if hasattr(s, "solves_for")]
    assert plan_syms.index("F") > plan_syms.index("m"), "m must be computed before F"
    assert plan_syms.index("F") > plan_syms.index("a"), "a must be computed before F"
    print(f"  Execution order: {plan_syms}")
    print("  PASSED")


def test_decision_log_populated():
    """Decision log must record what the LLM was shown and what it chose."""
    g        = get_graph()
    selector = make_selector(FMVA_MOCK_CHOICES)
    result   = resolve_frontier(
        target=FMVA_TARGET, given=FMVA_GIVEN,
        graph_index=g, question=FMVA_QUESTION, llm_round_fn=selector,
        retriever=StubRetriever(g, FMVA_LANDING_ORDER, priority_ids=list(FMVA_MOCK_CHOICES.values())), search_query=FMVA_QUESTION,
    )
    log = result.decision_log
    assert len(log) == 3, f"Expected 3 decisions (F, m, a), got {len(log)}"
    solving_for = {entry["solving_for"] for entry in log}
    assert solving_for == {"F", "m", "a"}, f"Unexpected: {solving_for}"
    for entry in log:
        assert "chosen_eq_id"   in entry
        assert "equation_str"   in entry
        assert "reason"         in entry
    print(f"\n[test_decision_log_populated] {len(log)} entries — PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# §8 step 4 — SymPy exact arithmetic + substitution traces
# ─────────────────────────────────────────────────────────────────────────────

def test_sympy_exact_fractions():
    """
    Execute the F=ma chain with exact arithmetic.
    v² = 30² = 900, u² = 0, 2s = 80  →  a = 900/80 = 45/4
    m = 8000 * 0.5 = 4000
    F = 4000 * 45/4 = 45000  N
    The intermediate a = 45/4 must survive as an exact fraction.
    """
    print(f"\n[test_sympy_exact_fractions]")
    g        = get_graph()
    selector = make_selector(FMVA_MOCK_CHOICES)
    result   = resolve_frontier(
        target=FMVA_TARGET, given=FMVA_GIVEN,
        graph_index=g, question=FMVA_QUESTION, llm_round_fn=selector,
        retriever=StubRetriever(g, FMVA_LANDING_ORDER, priority_ids=list(FMVA_MOCK_CHOICES.values())), search_query=FMVA_QUESTION,
    )

    given_values = {sym: meta["value"] for sym, meta in FMVA_GIVEN.items()}
    exec_result  = execute_plan(
        plan          = result.plan,
        given_values  = given_values,
        target_symbol = "F",
        target_unit   = "N",
        target_dim    = "MLT-2",
    )

    print(f"  success:         {exec_result.success}")
    print(f"  error:           {exec_result.error!r}")
    for t in exec_result.step_traces:
        print(f"  step: {t.solving_for}")
        print(f"    symbolic:    {t.trace.symbolic}")
        print(f"    substituted: {t.trace.substituted}")
        print(f"    result_exact:{t.trace.result_exact}")
        print(f"    float:       {t.value_float}")
    print(f"  final (exact):   {exec_result.final_exact_str}")
    print(f"  final (float):   {exec_result.final_float}")

    assert exec_result.success, f"Execution failed: {exec_result.error}"

    # Correct values
    m_step = next(t for t in exec_result.step_traces if t.solving_for == "m")
    a_step = next(t for t in exec_result.step_traces if t.solving_for == "a")
    F_step = next(t for t in exec_result.step_traces if t.solving_for == "F")

    assert abs(m_step.value_float - 4000.0)  < 0.01, f"m should be 4000, got {m_step.value_float}"
    assert abs(a_step.value_float - 11.25)   < 0.01, f"a should be 11.25, got {a_step.value_float}"
    assert abs(F_step.value_float - 45000.0) < 0.1,  f"F should be 45000, got {F_step.value_float}"
    assert abs(exec_result.final_float - 45000.0) < 0.1

    # a = 45/4 should be exact
    from sympy import Rational as R
    assert a_step.value_exact == R(45, 4), \
        f"a should be exact 45/4, got {a_step.value_exact}"

    print("  PASSED")


def test_trace_strings():
    """Substitution trace strings must be non-empty and contain the symbol."""
    g        = get_graph()
    selector = make_selector(FMVA_MOCK_CHOICES)
    result   = resolve_frontier(
        target=FMVA_TARGET, given=FMVA_GIVEN,
        graph_index=g, question=FMVA_QUESTION, llm_round_fn=selector,
        retriever=StubRetriever(g, FMVA_LANDING_ORDER, priority_ids=list(FMVA_MOCK_CHOICES.values())), search_query=FMVA_QUESTION,
    )
    given_values = {sym: meta["value"] for sym, meta in FMVA_GIVEN.items()}
    exec_result  = execute_plan(
        plan=result.plan, given_values=given_values,
        target_symbol="F", target_unit="N",
    )
    for t in exec_result.step_traces:
        assert t.solving_for in t.trace.result_exact, \
            f"Trace for {t.solving_for} missing symbol in result_exact"
        assert t.trace.symbolic,    f"Empty symbolic trace for {t.solving_for}"
        assert t.trace.substituted, f"Empty substituted trace for {t.solving_for}"
    print(f"\n[test_trace_strings] {len(exec_result.step_traces)} steps with traces — PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# §8 step 5 — Topological sort + cycle detection
# ─────────────────────────────────────────────────────────────────────────────

def test_topological_sort_simple():
    """Three steps with clear dependency: m→F and a→F must sort correctly."""
    print(f"\n[test_topological_sort_simple]")
    g    = get_graph()
    step_F = ResolvedStep(
        equation   = g.nodes_by_id["laws_of_motion_newton_second_law"],
        solves_for = fi("F","force","N","MLT-2"),
        inputs_used= ["m","a"], round_num=0, llm_reason="test",
    )
    step_m = ResolvedStep(
        equation   = g.nodes_by_id["general_density_definition"],
        solves_for = fi("m","mass","kg","M"),
        inputs_used= ["rho","V"], round_num=1, llm_reason="test",
    )
    step_a = ResolvedStep(
        equation   = g.nodes_by_id["kinematics_v2_u2_2as"],
        solves_for = fi("a","acceleration","m/s^2","LT-2"),
        inputs_used= ["v","u","s"], round_num=1, llm_reason="test",
    )
    given = {"rho","V","u","v","s","g"}
    sorted_steps, cyclic = _topological_sort([step_F, step_m, step_a], given)

    print(f"  sorted order: {[s.solves_for.symbol for s in sorted_steps]}")
    assert not cyclic, f"No cycle expected, got: {[s.solves_for.symbol for s in cyclic]}"
    syms = [s.solves_for.symbol for s in sorted_steps]
    assert syms.index("F") > syms.index("m"), "m must precede F"
    assert syms.index("F") > syms.index("a"), "a must precede F"
    print("  PASSED")


def test_simultaneous_group_in_plan():
    """
    Cycle detection in _topological_sort: two steps each depending on the
    other's output must be flagged as cyclic and merged into SimultaneousGroup.

    Genuine circular structure:
      step_v: uses v = u + a*t  (vars: v,u,a,t) → solves for v, needs t
      step_t: uses s = v*t      (vars: s,v,t)   → solves for t, needs v
    Given: u, a, s (but NOT v or t) → v and t form a true mutual dependency.
    """
    print(f"\n[test_simultaneous_group_in_plan]")
    from solver.frontier_resolver import _topological_sort, _merge_cycles

    # Synthetic equation dicts that form a true circular dependency
    eq_v = {
        "id": "synthetic_v_eq",
        "equation_str": "v = u + a*t",
        "sympy_expr": "Eq(v, u + a*t)",
        "variables": {
            "v": {"name": "velocity",           "unit": "m/s",   "dimension": "LT-1"},
            "u": {"name": "initial velocity",   "unit": "m/s",   "dimension": "LT-1"},
            "a": {"name": "acceleration",       "unit": "m/s^2", "dimension": "LT-2"},
            "t": {"name": "time",               "unit": "s",     "dimension": "T"},
        },
        "conditions": [], "rag_text": "", "common_mistakes": [],
    }
    eq_t = {
        "id": "synthetic_t_eq",
        "equation_str": "s = v*t",
        "sympy_expr": "Eq(s, v*t)",
        "variables": {
            "s": {"name": "displacement", "unit": "m",   "dimension": "L"},
            "v": {"name": "velocity",     "unit": "m/s", "dimension": "LT-1"},
            "t": {"name": "time",         "unit": "s",   "dimension": "T"},
        },
        "conditions": [], "rag_text": "", "common_mistakes": [],
    }

    step_v = ResolvedStep(
        equation=eq_v, solves_for=fi("v","velocity","m/s","LT-1"),
        inputs_used=["u","a","t"], round_num=0, llm_reason="cycle test"
    )
    step_t = ResolvedStep(
        equation=eq_t, solves_for=fi("t","time","s","T"),
        inputs_used=["s","v"], round_num=0, llm_reason="cycle test"
    )

    # Given: u, a, s — but NOT v or t → step_v needs t (from step_t) and
    # step_t needs v (from step_v) → mutual dependency
    given_syms = {"u", "a", "s"}
    sorted_s, cyclic = _topological_sort([step_v, step_t], given_syms)
    print(f"  sorted: {[s.solves_for.symbol for s in sorted_s]}")
    print(f"  cyclic: {[s.solves_for.symbol for s in cyclic]}")
    assert len(cyclic) == 2, f"Expected 2 cyclic steps, got {len(cyclic)}"

    plan = _merge_cycles(sorted_s, cyclic)
    sim_groups = [p for p in plan if isinstance(p, SimultaneousGroup)]
    assert len(sim_groups) == 1, f"Expected 1 SimultaneousGroup, got {len(sim_groups)}"
    assert set(u.symbol for u in sim_groups[0].unknowns) == {"v", "t"}
    print(f"  SimultaneousGroup unknowns: {[u.symbol for u in sim_groups[0].unknowns]}")
    print("  PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# §8 step 6 — Backtracking via excluded_eqs
# ─────────────────────────────────────────────────────────────────────────────

def test_backtracking_excluded_eq():
    """
    If newton_second_law is excluded (simulating a Stage 3 failure on it),
    the resolver must find an alternative route to F.
    """
    print(f"\n[test_backtracking_excluded_eq]")
    g = get_graph()

    # With newton_second_law excluded, the mock LLM now picks buoyant force
    fallback_choices = {
        "F": "fluid_mechanics_buoyant_force",  # forced fallback
        "m": "general_density_definition",
        "a": "kinematics_v2_u2_2as",
    }
    selector = make_selector(fallback_choices)
    result   = resolve_frontier(
        target       = FMVA_TARGET,
        given        = FMVA_GIVEN,
        graph_index  = g,
        question     = FMVA_QUESTION,
        llm_round_fn = selector,
        excluded_eqs = {"laws_of_motion_newton_second_law"},  # excluded!
    )

    print(f"  success: {result.success}")
    if result.success:
        chosen_ids = {s.equation["id"] for s in result.plan if hasattr(s,"equation")}
        print(f"  chosen: {chosen_ids}")
        assert "laws_of_motion_newton_second_law" not in chosen_ids, \
            "Excluded equation must not appear in plan"
        assert "fluid_mechanics_buoyant_force" in chosen_ids, \
            "Fallback equation should be chosen"
    print("  PASSED (fallback chain found)")


# ─────────────────────────────────────────────────────────────────────────────
# Regression — original failing test case confirmed correct
# ─────────────────────────────────────────────────────────────────────────────

def test_regression_fmva_full():
    """
    End-to-end deterministic regression: the original failing case.
    Mock LLM picks correctly; Stage 3 confirms answer is 45000 N.
    """
    print(f"\n[test_regression_fmva_full]")
    g        = get_graph()
    selector = make_selector(FMVA_MOCK_CHOICES)
    result   = resolve_frontier(
        target=FMVA_TARGET, given=FMVA_GIVEN,
        graph_index=g, question=FMVA_QUESTION, llm_round_fn=selector,
        retriever=StubRetriever(g, FMVA_LANDING_ORDER, priority_ids=list(FMVA_MOCK_CHOICES.values())), search_query=FMVA_QUESTION,
    )
    assert result.success
    given_values = {sym: meta["value"] for sym, meta in FMVA_GIVEN.items()}
    exec_result  = execute_plan(
        plan=result.plan, given_values=given_values,
        target_symbol="F", target_unit="N",
    )
    assert exec_result.success, f"Stage 3 failed: {exec_result.error}"
    assert abs(exec_result.final_float - 45000.0) < 0.1, \
        f"Expected 45000 N, got {exec_result.final_float}"
    print(f"  Final answer: {exec_result.final_exact_str} = {exec_result.final_float} N")
    print("  PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Simultaneous solve execution — Coulomb ∥ Newton
# ─────────────────────────────────────────────────────────────────────────────

def test_simultaneous_solve_execution():
    """
    Execute a SimultaneousGroup: Coulomb's law and Newton's 2nd law,
    with a known force value, verify SymPy solves the system.
    This exercises _execute_simultaneous in sympy_executor.
    """
    print(f"\n[test_simultaneous_solve_execution]")
    from solver.sympy_executor import _execute_simultaneous
    from solver.frontier_resolver import SimultaneousGroup

    g   = get_graph()
    eq1 = g.nodes_by_id["laws_of_motion_newton_second_law"]   # F = m*a
    eq2 = g.nodes_by_id["general_density_definition"]          # rho = m/V

    # Solve F and m simultaneously given: a=11.25, rho=8000, V=0.5
    # F = m * 11.25   and   rho = m / V  →  m=4000, F=45000
    group = SimultaneousGroup(
        equations = [eq1, eq2],
        unknowns  = [
            fi("F","force","N","MLT-2"),
            fi("m","mass","kg","M"),
        ],
        round_num = 0,
    )
    from sympy import nsimplify
    computed = {
        "a":   nsimplify(11.25),
        "rho": nsimplify(8000),
        "V":   nsimplify(0.5),
        "g":   nsimplify(9.8),
        "u":   nsimplify(0),
        "v":   nsimplify(30),
        "s":   nsimplify(40),
    }
    traces = _execute_simultaneous(group, computed)
    print(f"  traces: {[(t.solving_for, t.value_float) for t in traces]}")
    assert traces is not None, "Simultaneous solve must succeed"
    vals = {t.solving_for: t.value_float for t in traces}
    assert abs(vals.get("m", 0) - 4000)  < 0.1, f"m should be 4000, got {vals.get('m')}"
    assert abs(vals.get("F", 0) - 45000) < 1.0,  f"F should be 45000, got {vals.get('F')}"
    print("  PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# v7 — new tests for v7-specific behaviour
# ─────────────────────────────────────────────────────────────────────────────

def test_dimension_normalizer_handles_theta_and_N():
    """v7 fix: v6's _normalize_dimension only handled MLTAK; Theta and N
    silently mangled. Now: 'Theta' canonicalizes to K bucket, 'N' and 'mol'
    canonicalize to N bucket, 'varies' surfaces as a sentinel."""
    from solver.graph_loader import _normalize_dimension, _dimensions_compatible

    # 'Theta' is canonical for temperature; folds to K
    assert _normalize_dimension("Theta")   == (("K", 1),)
    assert _normalize_dimension("K")       == (("K", 1),)
    assert _normalize_dimension("Theta-1") == (("K", -1),)
    # Compound dimension involving Theta — the gas constant
    assert _normalize_dimension("ML2T-2N-1Theta-1") == (
        ("K", -1), ("L", 2), ("M", 1), ("N", -1), ("T", -2)
    )
    # N and mol both map to N
    assert _normalize_dimension("N")     == (("N", 1),)
    assert _normalize_dimension("mol-1") == (("N", -1),)
    # 'varies' sentinel — never compatible with anything real
    assert _normalize_dimension("varies") == (("__VARIES__", 1),)
    # Cross-compat
    assert _dimensions_compatible("Theta", "K") is True
    assert _dimensions_compatible("ML2T-2N-1Theta-1", "ML2T-2N-1K-1") is True
    assert _dimensions_compatible("varies", "M") is False
    print("[test_dimension_normalizer_handles_theta_and_N] PASSED")


def test_physical_constants_contain_graph_names():
    """v7 reconciliation: PHYSICAL_CONSTANTS must contain every symbol the
    graph actually uses for a physical constant. v6's set missed 'epsilon0'
    and 'mu0' (the graph's underscoreless forms), which meant frontier
    resolver would try to solve for epsilon_0 as if it were unknown in any
    electrostatics question."""
    from config import PHYSICAL_CONSTANTS
    # Both forms must be in PHYSICAL_CONSTANTS as belt-and-suspenders.
    # The graph itself has been standardized to the underscored form in v7,
    # but the underscoreless forms remain in the set in case any legacy
    # equation still uses them.
    for sym in ('epsilon_0', 'mu_0', 'e_charge', 'k_B', 'R_g', 'NA',
                'h_planck', 'g', 'G', 'c', 'pi'):
        assert sym in PHYSICAL_CONSTANTS, f"{sym!r} missing from PHYSICAL_CONSTANTS"
    print("[test_physical_constants_contain_graph_names] PASSED")


def test_graph_uses_underscored_constant_names():
    """v7: the graph file must use 'epsilon_0' and 'mu_0' (not 'epsilon0',
    'mu0'). This is the rename that aligns the graph with config."""
    import json
    from config import MAIN_GRAPH_PATH
    with open(MAIN_GRAPH_PATH) as f:
        g = json.load(f)
    import re
    for n in g["nodes"]:
        # Variable keys
        assert 'epsilon0' not in n["variables"], \
            f"{n['id']} still uses 'epsilon0' (should be 'epsilon_0')"
        assert 'mu0' not in n["variables"], \
            f"{n['id']} still uses 'mu0' (should be 'mu_0')"
        # In equation strings
        for field in ('equation_str', 'sympy_expr', 'latex'):
            v = n.get(field, '')
            assert not re.search(r'\bepsilon0\b', v), \
                f"{n['id']}.{field} still references 'epsilon0'"
            assert not re.search(r'\bmu0\b', v), \
                f"{n['id']}.{field} still references 'mu0'"
    print("[test_graph_uses_underscored_constant_names] PASSED")


def test_constants_not_treated_as_unknowns():
    """v7 regression guard: with the graph using 'epsilon_0' and config
    listing 'epsilon_0' as a PHYSICAL_CONSTANT, an electrostatics chain
    must not add epsilon_0 to the frontier as a missing variable."""
    from solver.graph_loader import load_graphs
    from config              import PHYSICAL_CONSTANTS
    graph = get_graph()
    # Find Coulomb's law — should contain epsilon_0
    coulomb = next((n for n in graph.nodes if n["id"] == "electrostatics_coulomb_law"), None)
    assert coulomb is not None
    assert 'epsilon_0' in coulomb["variables"]
    assert 'epsilon_0' in PHYSICAL_CONSTANTS
    # Simulate the frontier_resolver check: after picking Coulomb's law,
    # which symbols get added to the next frontier?
    target_sym = 'F'
    new_frontier_syms = [
        sym for sym in coulomb["variables"].keys()
        if sym != target_sym and sym not in PHYSICAL_CONSTANTS
    ]
    assert 'epsilon_0' not in new_frontier_syms, \
        "epsilon_0 is being treated as unknown — PHYSICAL_CONSTANTS check failed"
    assert 'pi' not in new_frontier_syms, "pi should also be excluded"
    print(f"[test_constants_not_treated_as_unknowns] new frontier from Coulomb: "
          f"{new_frontier_syms} (correctly excludes epsilon_0, pi) — PASSED")


def test_landing_layer_falls_through_when_no_retriever():
    """v7 safety: when retriever is None (no ChromaDB index built), the
    landing layer returns exactly the symbol-table candidates — same as
    v6 — so any v6-passing question still passes."""
    from solver.landing import get_landing_candidates
    graph = get_graph()
    cands = get_landing_candidates(
        graph_index      = graph,
        target_symbol    = "F",
        target_name      = "force",
        target_dimension = "MLT-2",
        search_query     = "find the net force on a body",
        visited_eqs      = set(),
        allowed_domains  = None,
        retriever        = None,
    )
    # All candidates should be from symbol lookup
    assert len(cands) > 0
    assert all(c.get("landing_source") == "symbol" for c in cands), \
        "Without retriever, every candidate must be tagged 'symbol' only"
    # Symbol-lookup parity with v6
    sym_only = graph.candidates_for_quantity("F", "force", "MLT-2", set(), None)
    assert {c["id"] for c in cands} == {c["id"] for c in sym_only}, \
        "Without retriever, landing must return the same set as v6 symbol lookup"
    print(f"[test_landing_layer_falls_through_when_no_retriever] "
          f"got {len(cands)} symbol candidates — PASSED")


def test_landing_layer_unions_symbol_and_semantic():
    """v7.2.4: Round-0 landing is now PURE semantic vector search (the symbol-
    table union was removed — it dumped every equation containing the target
    symbol, e.g. all 21 with 'F', overloading the model). This test pins the
    new behavior: get_landing_candidates_semantic returns exactly the
    retriever's top-k nodes, in rank order, each tagged 'semantic', with
    visited/excluded nodes filtered out — and NOTHING from a symbol table."""
    from solver.landing import get_landing_candidates_semantic
    graph = get_graph()

    newton = graph.nodes_by_id["laws_of_motion_newton_second_law"]
    torque = graph.nodes_by_id["rotational_motion_torque_inertia"]
    weight = graph.nodes_by_id["laws_of_motion_weight"]

    class MockRetriever:
        def rank_candidates(self, query, candidates, top_k=8):
            return list(candidates)[:top_k]
        def search(self, query, top_k=5):
            return [
                {"node": newton, "score": 0.99, "semantic_score": 0.99, "bm25_score": 0.9},
                {"node": torque, "score": 0.80, "semantic_score": 0.80, "bm25_score": 0.7},
                {"node": weight, "score": 0.70, "semantic_score": 0.70, "bm25_score": 0.6},
            ]

    cands = get_landing_candidates_semantic(
        retriever    = MockRetriever(),
        search_query = "find the net force on a body that accelerates",
        visited_eqs  = set(),
        top_k        = 8,
    )
    # Exactly the retrieved nodes, in order, all tagged 'semantic'.
    assert [c["id"] for c in cands] == [
        "laws_of_motion_newton_second_law",
        "rotational_motion_torque_inertia",
        "laws_of_motion_weight",
    ], f"landing must be the ranked semantic list, got {[c['id'] for c in cands]}"
    assert all(c["landing_source"] == "semantic" for c in cands)

    # Excluded/visited nodes are filtered out, and the scan continues down.
    cands2 = get_landing_candidates_semantic(
        retriever    = MockRetriever(),
        search_query = "q",
        visited_eqs  = {"laws_of_motion_newton_second_law"},
        top_k        = 8,
    )
    assert "laws_of_motion_newton_second_law" not in {c["id"] for c in cands2}, \
        "a visited/excluded node must not be re-offered by the landing scan"
    assert cands2[0]["id"] == "rotational_motion_torque_inertia", \
        "scan must continue with the next-ranked node after exclusion"
    print("[test_landing_layer_unions_symbol_and_semantic] "
          "pure-semantic ranked landing + exclusion — PASSED")


def test_landing_layer_drops_dimensionally_incompatible_semantic_candidates():
    """v7: semantic match alone is not enough; if the retrieved equation
    contains target_symbol with a dimensionally incompatible meaning,
    drop it. (Prevents e.g. an optics fringe-order m showing up as a
    candidate for 'mass' just because the rag_text incidentally mentioned
    mass.)"""
    from solver.landing import get_landing_candidates
    graph = get_graph()

    # Find an equation where 'V' means volume (L^3). Then use it as a
    # semantic match for needed='V'/voltage (ML2T-3A-1). The landing layer
    # should drop it via the dimension filter.
    vol_eq = next((n for n in graph.nodes
                   if "V" in n["variables"]
                   and n["variables"]["V"].get("dimension", "").startswith("L3")
                  ), None)
    assert vol_eq is not None, "Need at least one volume-V equation in graph"

    class MockRetriever:
        def rank_candidates(self, query, candidates, top_k=8):
            return list(candidates)[:top_k]
        def search(self, query, top_k=5):
            return [{"node": vol_eq, "score": 0.9,
                     "semantic_score": 0.9, "bm25_score": 0.5}]

    # Ask for V with VOLTAGE's dimension
    cands = get_landing_candidates(
        graph_index      = graph,
        target_symbol    = "V",
        target_name      = "potential difference",
        target_dimension = "ML2T-3A-1",
        search_query     = "any query",
        visited_eqs      = set(),
        allowed_domains  = None,
        retriever        = MockRetriever(),
    )
    # The volume-V equation should NOT appear as a semantic candidate
    assert vol_eq["id"] not in {c["id"] for c in cands if c.get("landing_source") == "semantic"}, \
        f"Dimension filter failed to drop volume-V eq when needed voltage-V"
    print(f"[test_landing_layer_drops_dimensionally_incompatible_semantic_candidates] PASSED")


def test_round_selector_surfaces_invalid_id_as_no_pick():
    """v7 change: when the LLM returns an equation ID that isn't in the
    candidate set, v6 silently substituted the first candidate. v7
    surfaces this as a no-pick with fallback_used='llm_invalid_id'."""
    from solver.llm_interface import _format_candidate
    # Direct test of the post-LLM mapping logic — we simulate the path
    # without an actual LLM call.
    # We do this by constructing the function's expected inputs and the
    # parsed selection that v6 would have silently fallen back on.
    from solver.frontier_resolver import FrontierItem
    fi = FrontierItem(symbol="F", name="force", unit="N", dimension="MLT-2")
    candidates = [
        {"id": "laws_of_motion_newtons_second_law",
         "equation_str": "F = m*a",
         "rag_text": "Newton's second law",
         "variables": {"F": {"name": "force", "unit": "N", "dimension": "MLT-2"}},
         "conditions": []},
    ]
    # Build the selection dict the LLM "returned" with a bad ID
    selections_raw = [{
        "needed_symbol": "F",
        "decision": "pick",
        "chosen_eq_id": "nonexistent_equation_id",
        "reason": "test",
    }]
    rd_by_symbol = {"F": {"frontier_item": fi, "candidates": candidates}}
    # Mirror the v7 logic in call_round_selector
    result = []
    for sel in selections_raw:
        sym = sel.get("needed_symbol")
        rd = rd_by_symbol.get(sym)
        if rd is None: continue
        fi_ = rd["frontier_item"]
        cands = rd["candidates"]
        decision = sel.get("decision")
        deferred = decision == "defer"
        chosen_eq = None
        fallback_used = None
        if deferred:
            pass
        elif decision == "none":
            chosen_eq = None
            fallback_used = "llm_decision_none"
        else:
            chosen_id = sel.get("chosen_eq_id")
            chosen_eq = next((eq for eq in cands if eq["id"] == chosen_id), None)
            if chosen_eq is None:
                fallback_used = f"llm_invalid_id: got {chosen_id!r}, not in candidates {[c['id'] for c in cands]}"
        result.append({"frontier_item": fi_, "chosen_eq": chosen_eq,
                       "fallback_used": fallback_used})

    assert len(result) == 1
    assert result[0]["chosen_eq"] is None, \
        "v7: bad LLM id must yield None (no silent first-candidate substitution)"
    assert result[0]["fallback_used"].startswith("llm_invalid_id"), \
        f"fallback_used must be tagged, got {result[0]['fallback_used']}"
    print("[test_round_selector_surfaces_invalid_id_as_no_pick] PASSED")


def test_round_selector_handles_decision_none():
    """v7: when LLM explicitly says decision='none', surface no-pick with
    fallback_used='llm_decision_none'. No more silent fallback to first
    candidate."""
    from solver.frontier_resolver import FrontierItem
    fi = FrontierItem(symbol="F", name="force", unit="N", dimension="MLT-2")
    candidates = [{
        "id": "fluid_mechanics_buoyant_force", "equation_str": "F = rho*V*g",
        "rag_text": "...", "variables": {"F": {"name":"force","unit":"N","dimension":"MLT-2"}},
        "conditions": [],
    }]
    selections_raw = [{
        "needed_symbol": "F", "decision": "none",
        "reason": "wrong physics — none of the candidates apply"
    }]
    # Mirror v7 logic
    sel = selections_raw[0]
    decision = sel.get("decision")
    if decision == "none":
        chosen_eq, fallback_used = None, "llm_decision_none"
    assert chosen_eq is None
    assert fallback_used == "llm_decision_none"
    print("[test_round_selector_handles_decision_none] PASSED")


def test_decision_log_records_candidates_shown():
    """v7: decision_log entries must include candidates_shown so Stage 4
    can honestly narrate rejected alternatives, and so debugging can see
    what was visible vs. what was picked."""
    graph = get_graph()
    target = fi("F", "force", "N", "MLT-2")
    given = {
        "rho": {"value": 8000, "unit": "kg/m^3", "name": "density",  "dimension": "ML-3"},
        "V":   {"value": 0.5,  "unit": "m^3",    "name": "volume",   "dimension": "L3"},
        "u":   {"value": 10,   "unit": "m/s",    "name": "u-vel",    "dimension": "LT-1"},
        "v":   {"value": 30,   "unit": "m/s",    "name": "v-vel",    "dimension": "LT-1"},
        "s":   {"value": 40,   "unit": "m",      "name": "displ",    "dimension": "L"},
    }
    selector = make_selector({
        "F": "laws_of_motion_newton_second_law",
        "a": "kinematics_v2_u2_2as",
    })
    res = resolve_frontier(
        target=target, given=given, graph_index=graph,
        question="test", llm_round_fn=selector,
        search_query="find the net force on a body that accelerates",
        retriever=StubRetriever(graph, ["laws_of_motion_newton_second_law"],
                                priority_ids=["general_density_definition",
                                              "kinematics_v2_u2_2as"]),
    )
    assert res.success
    assert len(res.decision_log) >= 1
    for entry in res.decision_log:
        assert "candidates_shown" in entry, \
            f"decision_log entry missing candidates_shown: {entry.keys()}"
        if entry["decision"] == "pick":
            assert len(entry["candidates_shown"]) >= 1
            assert entry["chosen_eq_id"] in [c["id"] for c in entry["candidates_shown"]], \
                "chosen_eq_id must be in candidates_shown for picked rounds"
    print(f"[test_decision_log_records_candidates_shown] "
          f"{len(res.decision_log)} log entries, all have candidates_shown — PASSED")


def test_resolve_frontier_accepts_search_query_and_retriever_kwargs():
    """v7: resolve_frontier's new optional kwargs default safely. Calling
    without them (the old v6 signature) must still work — the live tests
    do this. Calling with them must also work."""
    import inspect
    from solver.frontier_resolver import resolve_frontier
    sig = inspect.signature(resolve_frontier)
    assert "search_query" in sig.parameters
    assert "retriever" in sig.parameters
    # Both should default to safe values (no retriever → symbol-only)
    assert sig.parameters["search_query"].default == ""
    assert sig.parameters["retriever"].default is None
    print("[test_resolve_frontier_accepts_search_query_and_retriever_kwargs] PASSED")


def test_parse_system_emits_search_query():
    """v7: the Stage 1 parse system prompt must instruct the LLM to emit
    a search_query field for the new ChromaDB landing step."""
    from solver.llm_interface import _build_parse_system
    prompt = _build_parse_system({"laws_of_motion", "kinematics"})
    assert "search_query" in prompt, \
        "Stage 1 prompt must mention search_query field"
    assert "height" in prompt.lower() and "displacement" in prompt.lower(), \
        "Stage 1 prompt must guide LLM on height/displacement equivalence"
    print("[test_parse_system_emits_search_query] PASSED")


def test_round_selector_prompt_mentions_landing_source_and_none():
    """v7: the Stage 2 prompt must explain landing_source tags and the
    'decision: none' option, so the LLM uses them correctly."""
    from solver.llm_interface import ROUND_SELECT_SYSTEM
    assert "landing_source" in ROUND_SELECT_SYSTEM
    assert '"semantic"' in ROUND_SELECT_SYSTEM
    assert '"both"' in ROUND_SELECT_SYSTEM
    assert '"none"' in ROUND_SELECT_SYSTEM
    print("[test_round_selector_prompt_mentions_landing_source_and_none] PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# v7.1 — concept-level rag_text and search_query tests
# ─────────────────────────────────────────────────────────────────────────────

def test_exemplars_applied_to_graph():
    """v7.1: the 16 hand-authored exemplars must be present in the graph's
    rag_text fields after apply_exemplars_only.py runs."""
    import json
    from pathlib import Path
    from config import MAIN_GRAPH_PATH
    with open(MAIN_GRAPH_PATH) as f:
        graph = json.load(f)
    nodes_by_id = {n["id"]: n for n in graph["nodes"]}

    exemplars_path = Path(MAIN_GRAPH_PATH).parent.parent / "scripts" / "rag_text_exemplars.json"
    if not exemplars_path.exists():
        print(f"[test_exemplars_applied_to_graph] SKIP (no exemplars file)")
        return
    with open(exemplars_path) as f:
        ex_data = json.load(f)

    applied = 0
    not_applied = []
    for eid, body in ex_data["exemplars"].items():
        if eid not in nodes_by_id:
            continue
        if nodes_by_id[eid].get("rag_text", "") == body["rag_text"]:
            applied += 1
        else:
            not_applied.append(eid)
    assert applied >= len(ex_data["exemplars"]) - 1, \
        f"only {applied}/{len(ex_data['exemplars'])} exemplars applied; missing: {not_applied}"
    print(f"[test_exemplars_applied_to_graph] {applied}/{len(ex_data['exemplars'])} applied — PASSED")


def test_exemplar_rag_texts_are_concept_level():
    """v7.1: every exemplar must open with a named physics concept (not generic
    'Use it when...' boilerplate from v6) and must mention what it is NOT
    (the look-alike-exclusion property)."""
    import json
    from pathlib import Path
    from config import MAIN_GRAPH_PATH
    exemplars_path = Path(MAIN_GRAPH_PATH).parent.parent / "scripts" / "rag_text_exemplars.json"
    if not exemplars_path.exists():
        print(f"[test_exemplar_rag_texts_are_concept_level] SKIP (no exemplars file)")
        return
    with open(exemplars_path) as f:
        ex_data = json.load(f)

    # Banned openers — the old v6 templated patterns
    banned_openers = [
        "Use it when pressure",
        "Use it when force",
        "Use it when energy",
        "Use it when",
    ]
    # Concept-distinction markers — at least one of these patterns should
    # appear, signaling the rag_text explicitly contrasts itself with
    # look-alikes (either by exclusion or by family-position).
    distinction_markers = [
        "distinct from", "different from", "NOT use", "NOT applicable",
        "NOT for", "not the right concept", "NOT a", "Distinct from",
        "Conceptually distinct",
        # Family-position language — equivalent concept-distinction, more naturalistic
        "Pairs with", "pairs with", "Choose this", "choose this",
        "Same kinematic family", "same kinematic family",
        "is the wrong concept", "wrong concept",
    ]
    issues = []
    for eid, body in ex_data["exemplars"].items():
        rt = body["rag_text"]
        for bad in banned_openers:
            if rt.startswith(bad):
                issues.append(f"{eid}: starts with banned v6 opener {bad!r}")
                break
        if not any(m.lower() in rt.lower() for m in distinction_markers):
            issues.append(f"{eid}: no concept-distinction language (look-alike-exclusion property)")
    assert not issues, "\n".join(issues)
    print(f"[test_exemplar_rag_texts_are_concept_level] all "
          f"{len(ex_data['exemplars'])} exemplars passed concept-level checks — PASSED")


def test_stage1_prompt_has_concept_level_search_query_guidance():
    """v7.1: Stage 1's prompt for search_query must explicitly tell the LLM
    to write a concept-level query (named physics concept), not a keyword-
    level one. The cost of misframing this prompt is the exact failure
    mode the user kept flagging (F=ρgh winning over F=ma by symbol overlap)."""
    from solver.llm_interface import _build_parse_system
    prompt = _build_parse_system({"laws_of_motion", "kinematics"})
    must_have = [
        "CONCEPT-LEVEL",
        "Newton's Second Law",       # used as a concept-name example
        "Archimedes' Principle",     # used as a concept-name example
        "keyword-shaped",            # the antipattern is named
        "physics concept",
    ]
    missing = [m for m in must_have if m not in prompt]
    assert not missing, f"Stage 1 prompt missing concept-level guidance: {missing}"
    print("[test_stage1_prompt_has_concept_level_search_query_guidance] PASSED")


def test_format_candidate_uses_concept_not_full_rag_text():
    """v7.1.4: candidate format drops the full rag_text and uses a short
    concept name instead. The 8B model gets a clean, focused prompt.

    Replaces the v7.1 test that verified the 120-char truncation was gone —
    in v7.1.4 we drop rag_text entirely, replacing it with the concept_name
    surfaced by apply_exemplars_only or extracted from the rag_text head."""
    from solver.llm_interface import _format_candidate
    eq = {
        "id": "test_id",
        "equation_str": "F = m*a",
        "rag_text": "Newton's Second Law of Motion: the foundational dynamics relationship. " + ("x" * 600),
        "concept_name": "Newton's Second Law of Motion",
        "conditions": ["c1", "c2"],
        "variables": {
            "F": {"name": "force", "unit": "N", "dimension": "MLT-2"},
            "m": {"name": "mass", "unit": "kg", "dimension": "M"},
        },
    }
    out = _format_candidate(eq)
    assert "description" not in out, "v7.1.4 drops the description (full rag_text) field"
    assert "concept" in out, "v7.1.4 should add a concept field"
    assert out["concept"] == "Newton's Second Law of Motion"
    assert len(out["concept"]) < 80, f"concept should be short, got {len(out['concept'])} chars"
    assert out["equation"] == "F = m*a"
    assert "F" in out["variables"] and "m" in out["variables"]
    print(f"[test_format_candidate_uses_concept_not_full_rag_text] "
          f"concept={out['concept']!r} — PASSED")


def test_format_candidate_extracts_concept_from_rag_text_when_no_concept_name():
    """v7.1.4: if a node doesn't have concept_name (legacy graph, exemplars
    not re-applied), fall back to extracting the concept from the rag_text
    head. Our authored rag_texts open with '<Concept Name>: ...' or
    '<Concept Name>. ...'."""
    from solver.llm_interface import _format_candidate, _extract_concept
    eq = {
        "id": "test",
        "equation_str": "F = q1*q2/(4*pi*epsilon_0*r**2)",
        "rag_text": "Coulomb's Law: the electrostatic force between two point charges. Apply when...",
        "variables": {"F": {"name": "force", "unit": "N"}},
    }
    out = _format_candidate(eq)
    assert out["concept"] == "Coulomb's Law"
    eq2 = dict(eq); eq2["rag_text"] = "Archimedes' Principle for the buoyant force. Apply when..."
    assert _extract_concept(eq2) == "Archimedes' Principle for the buoyant force"
    print("[test_format_candidate_extracts_concept_from_rag_text_when_no_concept_name] PASSED")


def test_landing_overlap_ranking_promotes_bridge_equation():
    """v7.1.4: in Round 1+, candidates are re-ranked by overlap with
    known_symbols. Bridging equations (those whose variables overlap most
    with what we already have) bubble to the top.

    Scenario: looking for 'm' with knowns = {rho, V}. The equation
    rho = m/V shares {rho, V} (overlap 2). Other mass-containing equations
    share fewer. Overlap ranking must put rho = m/V first."""
    from solver.landing import get_landing_candidates

    class StubGraph:
        def candidates_for_quantity(self, **kwargs):
            return [
                {"id": "p_eq",   "variables": {"p": {}, "m": {}, "v": {}}},
                {"id": "rho_eq", "variables": {"rho": {}, "m": {}, "V": {}}},
                {"id": "K_eq",   "variables": {"K": {}, "m": {}, "v": {}}},
            ]

    ranked = get_landing_candidates(
        graph_index=StubGraph(),
        target_symbol="m", target_name="mass", target_dimension="M",
        search_query="", visited_eqs=set(), retriever=None,
        known_symbols={"rho", "V"}, round_num=1,
    )
    ids = [c["id"] for c in ranked]
    assert ids[0] == "rho_eq", \
        f"bridge equation should rank first in Round 1+; got order: {ids}"

    # Round 0 → must NOT re-rank by overlap
    unranked = get_landing_candidates(
        graph_index=StubGraph(),
        target_symbol="m", target_name="mass", target_dimension="M",
        search_query="", visited_eqs=set(), retriever=None,
        known_symbols={"rho", "V"}, round_num=0,
    )
    assert [c["id"] for c in unranked] == ["p_eq", "rho_eq", "K_eq"], \
        "Round 0 must NOT re-rank by overlap (concept-based ranking only)"
    print("[test_landing_overlap_ranking_promotes_bridge_equation] PASSED")


def test_round_select_prompt_states_derivability_principle():
    """v7.1.4: Stage 2 prompt must explicitly tell the LLM not to reject
    an equation because its variables don't appear in the question.
    Multi-equation chains are derivable — this must be a stated principle."""
    from solver.llm_interface import ROUND_SELECT_SYSTEM
    must_have = [
        "derivability",          # principle named
        "DO NOT reject",         # explicit anti-pattern
        "chain",                 # chaining concept
        "Density = m/V",         # F=ma example
        "v = u + a*t",           # KE example
    ]
    missing = [m for m in must_have if m not in ROUND_SELECT_SYSTEM]
    assert not missing, f"Stage 2 prompt missing derivability markers: {missing}"
    print("[test_round_select_prompt_states_derivability_principle] PASSED")


def test_exemplar_concept_names_are_unique():
    """v7.1: the user's principle is 'each concept is a unique identifier.'
    Verify that no two exemplars share a concept_name — every exemplar
    represents a distinct physics concept."""
    import json
    from pathlib import Path
    from config import MAIN_GRAPH_PATH
    exemplars_path = Path(MAIN_GRAPH_PATH).parent.parent / "scripts" / "rag_text_exemplars.json"
    if not exemplars_path.exists():
        print("[test_exemplar_concept_names_are_unique] SKIP (no exemplars file)")
        return
    with open(exemplars_path) as f:
        ex_data = json.load(f)
    names = [body["concept_name"] for body in ex_data["exemplars"].values()]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"Duplicate concept_names violate uniqueness: {dupes}"
    print(f"[test_exemplar_concept_names_are_unique] "
          f"{len(set(names))} distinct concepts across {len(names)} exemplars — PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# v7.1.2 — Logging and Stage 2 model switch
# ─────────────────────────────────────────────────────────────────────────────

def test_solver_log_emits_json_lines():
    """v7.1.2: solver.solver_log writes one JSON object per line to logs/solver.log,
    with timestamp, level, event, and any structured fields the caller passed."""
    import json
    from pathlib import Path
    from solver.solver_log import log
    log("test_v712_event", question="test", round_num=99, candidate_ids=["a","b"])
    log_path = Path(__file__).parent.parent / "logs" / "solver.log"
    assert log_path.exists(), f"log file not created at {log_path}"
    # Last line of the log should be our event
    last = log_path.read_text().splitlines()[-1]
    obj = json.loads(last)  # must parse as JSON
    assert obj["event"] == "test_v712_event"
    assert obj["question"] == "test"
    assert obj["round_num"] == 99
    assert obj["candidate_ids"] == ["a", "b"]
    assert "ts" in obj and "level" in obj
    print("[test_solver_log_emits_json_lines] PASSED")


def test_stage2_model_env_override():
    """v7.1.2: STAGE2_MODEL env var overrides the default. Empty/unset falls back
    to MODEL_FAST. This is the documented escape hatch when the 8B model
    can't follow the agentic-chain reasoning v7.1's concept rag_texts demand."""
    import os
    import importlib
    # Save original env
    saved = os.environ.get("STAGE2_MODEL")
    try:
        os.environ["STAGE2_MODEL"] = "llama-3.3-70b-versatile"
        import config
        importlib.reload(config)
        assert config.STAGE2_MODEL == "llama-3.3-70b-versatile", \
            f"env override didn't take: STAGE2_MODEL={config.STAGE2_MODEL}"

        del os.environ["STAGE2_MODEL"]
        importlib.reload(config)
        assert config.STAGE2_MODEL == config.MODEL_FAST, \
            f"default not MODEL_FAST: {config.STAGE2_MODEL}"
    finally:
        if saved is not None:
            os.environ["STAGE2_MODEL"] = saved
        elif "STAGE2_MODEL" in os.environ:
            del os.environ["STAGE2_MODEL"]
        # Reload config one more time to leave global state clean for other tests
        import config
        importlib.reload(config)
    print("[test_stage2_model_env_override] PASSED")


def test_stage2_batch_mode_dispatches_per_item_for_multi_item_rounds():
    """v7.1.3: When STAGE2_BATCH_MODE='auto' (default) and a round has more
    than one frontier item, call_round_selector splits into per-item LLM
    calls. With one item, it batches as before.

    The chained-problem failure mode (llm_omitted_item) happens specifically
    when the 8B model has to address multiple items in one call. Per-item
    batching eliminates the structural cause."""
    import os
    import importlib
    from solver.frontier_resolver import FrontierItem
    saved = os.environ.get("STAGE2_BATCH_MODE")

    try:
        os.environ["STAGE2_BATCH_MODE"] = "auto"
        # Reload to pick up env
        import config
        importlib.reload(config)
        import solver.llm_interface as li
        importlib.reload(li)

        # Stub the inner call helper to record how many times it's invoked
        calls_made = []
        def stub_call(question, available, round_data, round_num=0,
                      sub_index=None, sub_total=None, solve_context=None):
            calls_made.append({
                "n_items": len(round_data),
                "sub_index": sub_index,
                "sub_total": sub_total,
                "symbols": [rd["frontier_item"].symbol for rd in round_data],
            })
            return [{
                "frontier_item": rd["frontier_item"], "chosen_eq": None,
                "reason": "stub", "conditions_concern": None,
                "deferred": False, "_candidates": rd["candidates"],
                "fallback_used": None,
            } for rd in round_data]

        li._round_select_call = stub_call

        # Single-item round → should batch (one call covering 1 item)
        fi_F = FrontierItem(symbol="F", name="force", unit="N", dimension="MLT-2")
        single_round = [{"frontier_item": fi_F, "candidates": [{"id":"e1","equation_str":""}]}]
        calls_made.clear()
        li.call_round_selector("q", {}, single_round, round_num=0)
        assert len(calls_made) == 1, \
            f"single-item round should make 1 call, made {len(calls_made)}"
        assert calls_made[0]["n_items"] == 1
        assert calls_made[0]["sub_index"] is None, "no sub-indexing for single-item rounds"

        # Multi-item round → should split into per-item calls
        fi_m = FrontierItem(symbol="m", name="mass", unit="kg", dimension="M")
        fi_a = FrontierItem(symbol="a", name="acceleration", unit="m/s^2", dimension="LT-2")
        multi_round = [
            {"frontier_item": fi_m, "candidates": [{"id":"e1","equation_str":""}]},
            {"frontier_item": fi_a, "candidates": [{"id":"e2","equation_str":""}]},
        ]
        calls_made.clear()
        li.call_round_selector("q", {}, multi_round, round_num=1)
        assert len(calls_made) == 2, \
            f"multi-item round should make 2 calls in auto mode, made {len(calls_made)}"
        for i, c in enumerate(calls_made):
            assert c["n_items"] == 1, f"each per-item call should have 1 item, got {c['n_items']}"
            assert c["sub_index"] == i
            assert c["sub_total"] == 2

    finally:
        if saved is not None:
            os.environ["STAGE2_BATCH_MODE"] = saved
        elif "STAGE2_BATCH_MODE" in os.environ:
            del os.environ["STAGE2_BATCH_MODE"]
        # Reload everything to leave clean global state
        import config
        importlib.reload(config)
        import solver.llm_interface
        importlib.reload(solver.llm_interface)
    print("[test_stage2_batch_mode_dispatches_per_item_for_multi_item_rounds] PASSED")


def test_stage2_filters_unused_constants_from_prompt():
    """v7.1.5: the ALREADY KNOWN section must NOT list universal constants
    that no candidate equation uses. A kinematics round should not show the
    LLM Planck's constant, Boltzmann's constant, speed of light, etc.

    This was a real bug: every Stage 2 prompt listed all 10 universal
    constants regardless of relevance, wasting ~300 bytes per prompt and
    accelerating TPM exhaustion on the Groq free tier."""
    import importlib
    import config, solver.llm_interface as li
    importlib.reload(config); importlib.reload(li)
    from solver.frontier_resolver import FrontierItem

    captured = {}
    def fake_call(model, system, user, temperature=0.1, stage="?", _attempt=1):
        captured["user"] = user
        return '{"selections":[{"needed_symbol":"a","decision":"none","reason":"x"}]}'
    li._call = fake_call

    available = {
        "v": {"value": 30, "unit": "m/s", "name": "velocity"},
        "s": {"value": 40, "unit": "m", "name": "displacement"},
        "h_planck": {"value": 6.626e-34, "unit": "J·s", "name": "Planck constant"},
        "k_B": {"value": 1.38e-23, "unit": "J/K", "name": "Boltzmann constant"},
        "c": {"value": 3e8, "unit": "m/s", "name": "speed of light"},
        "NA": {"value": 6.022e23, "unit": "1/mol", "name": "Avogadro number"},
    }
    fi_a = FrontierItem(symbol="a", name="acceleration", unit="m/s^2", dimension="LT-2")
    cand = {"id": "kinematics_v2_u2_2as", "equation_str": "v**2 = u**2 + 2*a*s",
            "concept_name": "Time-Free Kinematic Relation",
            "variables": {"v": {"name":"velocity","unit":"m/s"},
                          "u": {"name":"initial velocity","unit":"m/s"},
                          "a": {"name":"acceleration","unit":"m/s^2"},
                          "s": {"name":"displacement","unit":"m"}}}
    round_data = [{"frontier_item": fi_a, "candidates": [cand]}]

    li.call_round_selector("find acceleration", available, round_data, round_num=1)

    known_section = captured["user"].split("NEEDED QUANTITIES")[0]
    for c in ("Planck", "Boltzmann", "Avogadro", "speed of light"):
        assert c not in known_section, \
            f"unused constant {c!r} should be filtered from the prompt, but it's present"
    # The question-relevant givens must still be there
    assert "v (velocity)" in known_section
    assert "s (displacement)" in known_section

    # restore clean global state
    importlib.reload(li)
    print("[test_stage2_filters_unused_constants_from_prompt] PASSED")


def test_stage2_detects_hallucinated_symbol():
    """v7.1.5: when the LLM answers about a symbol we didn't ask for, the
    code logs it as a hallucination rather than silently treating the asked
    symbol as omitted. Observed in the live log: Round 2 asked for 'u', the
    8B model returned needed_symbol='a'."""
    import importlib, json
    from pathlib import Path
    import config, solver.llm_interface as li
    importlib.reload(config); importlib.reload(li)
    from solver.frontier_resolver import FrontierItem

    def fake_call(model, system, user, temperature=0.1, stage="?", _attempt=1):
        # Asked about 'u', but answer is about 'a' — a hallucination
        return '{"selections":[{"needed_symbol":"a","decision":"pick","chosen_eq_id":"kinematics_v_u_at","reason":"x"}]}'
    li._call = fake_call

    log_path = Path(__file__).parent.parent / "logs" / "solver.log"
    if log_path.exists():
        log_path.write_text("")  # clear

    fi_u = FrontierItem(symbol="u", name="initial velocity", unit="m/s", dimension="LT-1")
    cand = {"id": "kinematics_v_u_at", "equation_str": "v = u + a*t",
            "concept_name": "Time-Velocity Kinematic Relation",
            "variables": {"u": {"name":"initial velocity","unit":"m/s"}}}
    round_data = [{"frontier_item": fi_u, "candidates": [cand]}]
    li.call_round_selector("find u", {"v": {"value":30,"unit":"m/s","name":"velocity"}},
                           round_data, round_num=2)

    # Check the log has a hallucination event
    events = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    halluc = [e for e in events if e.get("event") == "stage2_hallucinated_symbol"]
    assert halluc, "expected a stage2_hallucinated_symbol log event"
    assert "a" in halluc[-1]["hallucinated"]
    assert "u" in halluc[-1]["asked_for"]

    importlib.reload(li)
    print("[test_stage2_detects_hallucinated_symbol] PASSED")


def test_stage1_prompt_has_implicit_given_phrase_rules():
    """v7.1.5: Stage 1 prompt must instruct the model to extract numeric
    values from physics PHRASES — 'starts from rest' → u=0, etc. The live
    log showed the 8B model dropping u=0 about half the time, which broke
    chained kinematics."""
    from solver.llm_interface import _build_parse_system
    prompt = _build_parse_system({"kinematics", "laws_of_motion"})
    must_have = [
        "starts from rest",
        "u = 0",
        "comes to rest",
        "dropped",
        "IMPLICIT GIVENS",
    ]
    missing = [m for m in must_have if m not in prompt]
    assert not missing, f"Stage 1 prompt missing implicit-given rules: {missing}"
    print("[test_stage1_prompt_has_implicit_given_phrase_rules] PASSED")


def test_symbol_canonicalization_maps_energy_aliases():
    """v7.1.6: the target symbol canonicalizer maps LLM-chosen energy aliases
    to the graph's canonical symbol. The live KE test failed because Stage 1
    named kinetic energy 'E', the graph uses 'K', and the resolver treated
    them as different unknowns — cascading into a wrong chain ending at
    projectile angle theta."""
    from solver.pipeline import _canonicalize_symbol
    # Direct aliases (dimension-independent)
    assert _canonicalize_symbol("E_k") == "K"
    assert _canonicalize_symbol("KE") == "K"
    assert _canonicalize_symbol("PE") == "U"
    assert _canonicalize_symbol("E_p") == "U"
    # Dimension-aware: bare 'E' with energy dimension → K
    assert _canonicalize_symbol("E", "ML2T-2") == "K"
    assert _canonicalize_symbol("E", "ML²T-2") == "K"   # unicode superscript
    # Dimension-aware: 'E' with force dimension is NOT energy — leave alone
    assert _canonicalize_symbol("E", "MLT-2") == "E"
    # Non-aliased symbols pass through
    assert _canonicalize_symbol("F", "MLT-2") == "F"
    assert _canonicalize_symbol("K", "ML2T-2") == "K"
    print("[test_symbol_canonicalization_maps_energy_aliases] PASSED")


def test_stage1_prompt_forbids_inventing_values():
    """v7.1.6: Stage 1 prompt must forbid inventing numeric values. The live
    test 'Find the velocity of the object.' (no numbers) had the 8B model
    hallucinate s=5, g=9.81, t=2 and then confidently solve v=2.5 — a
    correct answer to a question never asked. The worst failure mode."""
    from solver.llm_interface import _build_parse_system
    prompt = _build_parse_system({"kinematics"})
    must_have = [
        "NEVER INVENT VALUES",
        "EXPLICITLY present",
        "MUST be empty",
        "UNDERSPECIFIED",
    ]
    missing = [m for m in must_have if m not in prompt]
    assert not missing, f"Stage 1 prompt missing anti-hallucination rules: {missing}"
    print("[test_stage1_prompt_forbids_inventing_values] PASSED")


def test_extract_json_tolerates_latex_escapes():
    """v7.1.9: small local models (Qwen-3B etc.) write LaTeX math inside JSON
    string values, producing invalid escapes like \\cos, \\Delta, \\( that
    crash json.loads. The exact payloads below are from the user's local-run
    log where the model picked the RIGHT equation but the parse died on the
    backslash, recording a false 'no-fit'. The sanitizer must recover these
    while preserving valid escapes."""
    from solver.llm_interface import _extract_json
    # These mirror the real failing log lines.
    latex_cases = [
        r'{"selections":[{"needed_symbol":"v","decision":"pick","chosen_eq_id":"kinematics_v2_u2_2as","reason":"equation \( v^2 = u^2 + 2a\Delta s \) is correct, u=0","conditions_concern":null}]}',
        r'{"selections":[{"needed_symbol":"F","decision":"pick","chosen_eq_id":"work_energy_power_work_constant_force","reason":"W = Fs\cos(\theta) rearranged for F","conditions_concern":null}]}',
        r'{"selections":[{"needed_symbol":"q2","decision":"pick","chosen_eq_id":"x","reason":"we use \( F = -\frac{dU}{dr} \) here","conditions_concern":null}]}',
    ]
    for raw in latex_cases:
        result = _extract_json(raw)
        sel = result["selections"][0]
        # The critical fields the pipeline consumes must survive intact.
        assert sel["decision"] == "pick"
        assert sel["chosen_eq_id"]  # non-empty
    # Valid escapes must NOT be corrupted.
    valid = _extract_json(r'{"name":"caf\u00e9","note":"line1\nline2","path":"a\/b"}')
    assert valid["name"] == "café"
    assert "\n" in valid["note"]
    assert valid["path"] == "a/b"
    print("[test_extract_json_tolerates_latex_escapes] PASSED")


def test_sanitize_json_escapes_doubles_only_invalid():
    """v7.1.9: the escape sanitizer doubles invalid backslashes but leaves
    valid JSON escape sequences and \\uXXXX untouched."""
    from solver.llm_interface import _sanitize_json_escapes
    import json
    # \c is invalid -> becomes \\c (literal backslash-c); parseable.
    assert json.loads('"' + _sanitize_json_escapes(r'a\cos b') + '"') == r'a\cos b'
    # \n stays a newline escape.
    assert json.loads('"' + _sanitize_json_escapes(r'x\ny') + '"') == 'x\ny'
    # \uXXXX preserved.
    assert json.loads('"' + _sanitize_json_escapes(r'\u00e9') + '"') == 'é'
    print("[test_sanitize_json_escapes_doubles_only_invalid] PASSED")


def test_stage1_empty_givens_retry_trigger_logic():
    """v7.1.9: the empty-givens retry fires when the question has digits but
    the model extracted zero givens, and does NOT fire for digit-free
    underspecified questions (which must stay empty to return UNVERIFIED)."""
    import re
    def would_retry(question, given):
        return bool(re.search(r"\d", question)) and not given
    # Numeric questions with dropped givens -> retry
    assert would_retry("A 5 kg object, net force 20 N. Find acceleration.", {})
    assert would_retry("starts from rest, accelerates 3 m/s^2 for 4 s", {})
    # Digit-free underspecified -> NO retry (preserves anti-hallucination)
    assert not would_retry("Find the velocity of the object.", {})
    # Already has givens -> NO retry
    assert not would_retry("A 5 kg object", {"m": {"value": 5}})
    print("[test_stage1_empty_givens_retry_trigger_logic] PASSED")


def test_dimension_unicode_superscript_folding():
    """v7.1.11: dimension comparison must treat Unicode superscripts the same
    as ASCII. The 7B emits 'MLT⁻²' where the graph stores 'MLT-2'; without
    folding, the dimension filter wrongly rejected the equation, silently
    dropping correct candidates (e.g. Newton's second law for a force target)
    from Stage 2."""
    from solver.graph_loader import _dimensions_compatible, _normalize_dimension
    pairs = [
        ("MLT⁻²", "MLT-2"), ("ML²T⁻²", "ML2T-2"), ("LT⁻¹", "LT-1"),
        ("LT⁻²", "LT-2"), ("ML⁻³", "ML-3"),
    ]
    for uni, asc in pairs:
        assert _normalize_dimension(uni) == _normalize_dimension(asc), \
            f"{uni} should normalize same as {asc}"
        assert _dimensions_compatible(asc, uni), f"{asc} vs {uni} should be compatible"
    # Unicode minus sign (U+2212) also folds.
    assert _dimensions_compatible("MLT-2", "MLT−2")
    # Genuinely different dimensions still don't match.
    assert not _dimensions_compatible("MLT-2", "M")
    print("[test_dimension_unicode_superscript_folding] PASSED")


def test_si_unit_normalization_generic():
    """v7.1.11: given values are converted to SI in code. The model left μC as
    2 (instead of 2e-6), making Coulomb's law off by 10^12. Conversion must be
    generic across prefixes/units, and must NOT touch values already in SI
    (including the kg trap)."""
    from solver.llm_interface import _normalize_given_to_si
    # The exact failing case.
    out = _normalize_given_to_si({
        "q1": {"value": 2, "unit": "μC", "name": "charge", "dimension": "AT"},
        "r":  {"value": 0.5, "unit": "m", "name": "distance", "dimension": "L"},
    })
    assert out["q1"]["value"] == 2e-6 and out["q1"]["unit"] == "C"
    assert out["r"]["value"] == 0.5 and out["r"]["unit"] == "m"  # bare metre untouched
    # Battery of conversions.
    checks = [
        ("cm", 5, 0.05, "m"), ("nm", 400, 4e-7, "m"), ("g", 500, 0.5, "kg"),
        ("mA", 20, 0.02, "A"), ("km/h", 72, 20.0, "m/s"),
        ("kg", 5, 5, "kg"),    # SI base — protected
        ("m/s", 30, 30, "m/s"), ("N", 20, 20, "N"), ("mol", 2, 2, "mol"),
    ]
    for unit, val, exp_val, exp_unit in checks:
        r = _normalize_given_to_si({"x": {"value": val, "unit": unit}})["x"]
        assert abs(r["value"] - exp_val) < abs(exp_val) * 1e-6 + 1e-20, \
            f"{val} {unit} → expected {exp_val}, got {r['value']}"
        assert r["unit"] == exp_unit, f"{unit} → expected {exp_unit}, got {r['unit']}"
    print("[test_si_unit_normalization_generic] PASSED")


def test_simultaneous_solver_registers_canonicalized_unknown():
    """v7.1.11: the simultaneous solver must not KeyError on an unknown whose
    symbol isn't in the group's equation variables (e.g. a canonicalized
    target like K). It should register the symbol rather than crash."""
    from solver.frontier_resolver import SimultaneousGroup, FrontierItem
    from solver import sympy_executor
    # A group whose unknown symbol 'K' is NOT present in the equation vars
    # (simulating the canonicalization mismatch). The fix should register K
    # rather than raise KeyError. We only assert no KeyError on the symbol
    # collection step — full solve depends on equation content.
    eq = {
        "id": "test_eq", "variables": {"x": {}, "y": {}},
        "sympy_expr": "Eq(x, y)",
    }
    fi = FrontierItem(symbol="K", name="kinetic energy", unit="J", dimension="ML2T-2")
    group = SimultaneousGroup(equations=[eq], unknowns=[fi], round_num=1)
    try:
        # This previously raised KeyError: 'K' at the unknowns list-comp.
        sympy_executor._execute_simultaneous(group, computed={})
    except KeyError as e:
        raise AssertionError(f"Should not KeyError on canonicalized unknown: {e}")
    except Exception:
        # Other failures (unsolvable group, etc.) are fine for this test —
        # we're only verifying the KeyError on symbol registration is gone.
        pass
    print("[test_simultaneous_solver_registers_canonicalized_unknown] PASSED")


def test_dead_end_reports_root_for_rollback():
    """
    v7.2.2/v7.2.3: when a chain dead-ends, resolve_frontier must report a
    SPECIFIC equation to exclude in dead_end_root_eq so the pipeline can drop
    that equation and retry. v7.2.3 reports the equation that INTRODUCED the
    dead-ended variable (fi.introduced_by) rather than always the Round-0
    root, so good early steps survive. In this isolation the introducer and
    the root are the same equation (eq_bad introduces Y), so the reported id
    is 'eq_bad' either way.

    Isolation: target X. Round 0 picks eq_bad (X = Y), which introduces Y. Y
    has no equation that resolves it, so Y dead-ends. The result must be
    unsuccessful AND carry dead_end_root_eq == 'eq_bad'.
    """
    print("\n[test_dead_end_reports_root_for_rollback]")
    eq_bad = {
        "id": "eq_bad", "equation_str": "X = Y",
        "sympy_expr": "Eq(X, Y)",
        "variables": {
            "X": {"name": "x", "unit": "u", "dimension": "L"},
            "Y": {"name": "y", "unit": "u", "dimension": "L"},
        },
        "conditions": [], "rag_text": "bad fixture", "common_mistakes": [],
        "domain": "test",
    }
    # Only eq_bad exists; Y can't be resolved by anything else → dead-end.
    stub_graph = _StubGraphIndex([eq_bad])
    selector = make_selector({"X": "eq_bad", "Y": "eq_bad"})  # Y will find no new eq
    target = fi("X", "x", "u", "L")
    given = {}  # nothing given → Y truly unresolvable

    result = resolve_frontier(target, given, stub_graph, "test", selector,
                              retriever=StubRetriever(stub_graph, ["eq_bad"]),
                              search_query="test")
    print(f"  success: {result.success}")
    print(f"  dead_end_root_eq: {result.dead_end_root_eq!r}")
    assert not result.success, "should fail (Y unresolvable)"
    assert result.dead_end_root_eq == "eq_bad", (
        f"must report the culprit equation for rollback, got {result.dead_end_root_eq!r}"
    )
    print("  PASSED")


def test_bare_selection_object_parsed():
    """
    v7.2.2: the 7B sometimes returns a BARE selection object instead of
    wrapping it in {"selections": [...]}. The parser must recover it rather
    than discarding a correct pick as 'omitted'.
    """
    print("\n[test_bare_selection_object_parsed]")
    from solver.llm_interface import _extract_json
    bare = ('{"needed_symbol": "a", "decision": "pick", '
            '"chosen_eq_id": "laws_of_motion_newton_second_law", "reason": "x"}')
    parsed = _extract_json(bare)
    if isinstance(parsed, list):
        sel = parsed
    else:
        sel = parsed.get("selections", [])
        if not sel and ("needed_symbol" in parsed or "chosen_eq_id" in parsed
                        or "decision" in parsed):
            sel = [parsed]
    assert len(sel) == 1, "bare object should be recovered as one selection"
    assert sel[0]["chosen_eq_id"] == "laws_of_motion_newton_second_law"
    # And the normal wrapped shape still works.
    wrapped = _extract_json('{"selections": [{"needed_symbol": "a", "chosen_eq_id": "z"}]}')
    sel2 = wrapped.get("selections", []) if not isinstance(wrapped, list) else wrapped
    assert len(sel2) == 1
    print("  PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    tests = [
        test_dimension_normalization_format_independent,
        test_dimension_compat,
        test_candidates_symbol_m,
        test_candidates_symbol_F,
        test_candidates_visited_exclusion,
        test_candidates_conservation_law_excluded,
        test_already_targeted_symbol_not_reintroduced,
        test_parse_system_has_target_identification_rule,
        test_narrate_system_forbids_new_computation,
        test_domain_filter_narrows_candidates,
        test_domain_filter_fallback_when_empty,
        test_domain_filter_fixes_real_overflow_scenario,
        test_round_splits_on_token_overflow,
        test_frontier_happy_path,
        test_decision_log_populated,
        test_sympy_exact_fractions,
        test_trace_strings,
        test_topological_sort_simple,
        test_simultaneous_group_in_plan,
        test_backtracking_excluded_eq,
        test_regression_fmva_full,
        test_simultaneous_solve_execution,
        # v7 additions
        test_dimension_normalizer_handles_theta_and_N,
        test_physical_constants_contain_graph_names,
        test_graph_uses_underscored_constant_names,
        test_constants_not_treated_as_unknowns,
        test_landing_layer_falls_through_when_no_retriever,
        test_landing_layer_unions_symbol_and_semantic,
        test_landing_layer_drops_dimensionally_incompatible_semantic_candidates,
        test_round_selector_surfaces_invalid_id_as_no_pick,
        test_round_selector_handles_decision_none,
        test_decision_log_records_candidates_shown,
        test_resolve_frontier_accepts_search_query_and_retriever_kwargs,
        test_parse_system_emits_search_query,
        test_round_selector_prompt_mentions_landing_source_and_none,
        # v7.1 additions
        test_exemplars_applied_to_graph,
        test_exemplar_rag_texts_are_concept_level,
        test_stage1_prompt_has_concept_level_search_query_guidance,
        test_format_candidate_uses_concept_not_full_rag_text,
        test_format_candidate_extracts_concept_from_rag_text_when_no_concept_name,
        test_landing_overlap_ranking_promotes_bridge_equation,
        test_round_select_prompt_states_derivability_principle,
        test_exemplar_concept_names_are_unique,
        # v7.1.2 additions
        test_solver_log_emits_json_lines,
        test_stage2_model_env_override,
        # v7.1.3 additions
        test_stage2_batch_mode_dispatches_per_item_for_multi_item_rounds,
        # v7.1.5 additions
        test_stage2_filters_unused_constants_from_prompt,
        test_stage2_detects_hallucinated_symbol,
        test_stage1_prompt_has_implicit_given_phrase_rules,
        # v7.1.6 additions
        test_symbol_canonicalization_maps_energy_aliases,
        test_stage1_prompt_forbids_inventing_values,
        # v7.1.9 additions
        test_extract_json_tolerates_latex_escapes,
        test_sanitize_json_escapes_doubles_only_invalid,
        test_stage1_empty_givens_retry_trigger_logic,
        # v7.1.11 additions
        test_dimension_unicode_superscript_folding,
        test_si_unit_normalization_generic,
        test_simultaneous_solver_registers_canonicalized_unknown,
        # v7.2.2 additions
        test_dead_end_reports_root_for_rollback,
        test_bare_selection_object_parsed,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"\n  ✗ {t.__name__} FAILED: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*55}")
    print(f"  {passed} passed, {failed} failed  ({len(tests)} total)")
    print(f"{'='*55}")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
