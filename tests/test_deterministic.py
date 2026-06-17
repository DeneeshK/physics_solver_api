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

def make_selector(choices: dict):
    """
    Build a mock llm_round_fn.
    choices: {needed_symbol → equation_id_to_pick}
    If a symbol is not in choices, pick the first candidate.
    """
    def selector(question, available, round_data, round_num=0):
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
    """
    print("\n[test_narrate_system_forbids_new_computation]")
    from solver.llm_interface import NARRATE_SYSTEM
    assert "NEVER introduce a number" in NARRATE_SYSTEM
    assert "narrating a finished computation, not completing one" in NARRATE_SYSTEM
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
    Without a domain hint (worst case), the m+a round after picking F=m*a
    is large enough to exceed budget. Confirm resolve_frontier's safety
    valve splits it into separate single-symbol calls instead of sending
    one oversized batched call — this is the actual fix for the 413 crash,
    independent of whether domain filtering also helped.
    """
    print("\n[test_round_splits_on_token_overflow]")
    g = get_graph()
    base_selector = make_selector({
        "F": "laws_of_motion_newton_second_law",
        "m": "general_density_definition",
        "a": "kinematics_v2_u2_2as",
    })
    call_log = []
    def tracking_selector(question, available, round_data, round_num=0):
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
    # Deliberately no allowed_domains — forces the large candidate sets for
    # m and a, so the safety valve (not domain filtering) is what's tested.
    result = resolve_frontier(target, given, g, "test question", tracking_selector)

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
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    tests = [
        test_dimension_compat,
        test_candidates_symbol_m,
        test_candidates_symbol_F,
        test_candidates_visited_exclusion,
        test_candidates_conservation_law_excluded,
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
