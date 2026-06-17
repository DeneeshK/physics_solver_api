"""
tests/test_pipeline_integration.py
Pipeline-level tests with Stage 1/2/4/5 monkeypatched — no API key needed.

These specifically cover three defects found during a design review:

1. `{{...}}` outside an f-string in llm_interface.py constructed a SET
   containing a dict (unhashable) -> TypeError on every call that injected
   an implicit constant. Confirmed crashing 100% of the time 'g' needed
   injecting, which was almost every question (see #2).

2. llm_interface.py force-injected 'g' into `given` for EVERY question
   regardless of relevance ("most JEE/NEET problems need it"), contradicting
   the brief's Stage-1-scenario-judgment design and polluting Stage 2's
   "already known" context for unrelated domains (electrostatics, optics...).
   Fixed by splitting PHYSICAL_CONSTANTS into UNIVERSAL_CONSTANTS (always
   available, context-independent: pi, c, G, epsilon_0...) vs 'g' (only
   available when Stage 1 actually judges the scenario implies it).

3. pipeline.py only backtracked when the LLM *itself* flagged a
   conditions_concern. A confidently-wrong-but-unflagged pick that fails
   in Stage 3 (no real solution, dimension mismatch, etc.) used to fall
   straight through to UNVERIFIED with no retry. Fixed with a bounded
   retry loop keyed on ANY Stage 3 failure, using the newly-added
   ExecutionTrace.failed_eq_ids to know exactly what to exclude.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_no_dict_literal_crash_on_implicit_injection():
    """
    Regression for bug #1+#2: Stage 1 flags 'g' as implicit (free-fall
    scenario) -> must inject cleanly with no TypeError, using single braces.
    """
    print("\n[test_no_dict_literal_crash_on_implicit_injection]")
    import solver.llm_interface as li

    def fake_call(model, system, user, temperature=0.1):
        return json.dumps({
            "given": {"h": {"value": 20, "unit": "m", "name": "height", "dimension": "L"}},
            "unknown": {"symbol": "v", "name": "final velocity", "unit": "m/s", "dimension": "LT-1"},
            "implicit_constants": ["g"],
        })
    original_call = li._call
    li._call = fake_call
    try:
        result = li.parse_question("A ball is dropped from 20 m. Find its final velocity.")
    finally:
        li._call = original_call

    assert "g" in result["given"], "g should be injected — Stage 1 flagged it"
    assert result["given"]["g"]["value"] == 9.8
    assert result["given"]["g"]["unit"] == "m/s^2"
    print("  No TypeError. g injected correctly via single-brace dict literal.")
    print("  PASSED")


def test_g_not_force_injected_for_unrelated_domain():
    """
    Regression for bug #2: a non-mechanics question where Stage 1 does NOT
    flag 'g' must NOT have g appear in `given` at all.
    """
    print("\n[test_g_not_force_injected_for_unrelated_domain]")
    import solver.llm_interface as li

    def fake_call(model, system, user, temperature=0.1):
        return json.dumps({
            "given": {
                "q1": {"value": 2e-6, "unit": "C", "name": "first charge", "dimension": "AT"},
                "q2": {"value": 3e-6, "unit": "C", "name": "second charge", "dimension": "AT"},
                "r":  {"value": 0.5, "unit": "m", "name": "separation", "dimension": "L"},
            },
            "unknown": {"symbol": "F", "name": "force", "unit": "N", "dimension": "MLT-2"},
            "implicit_constants": ["epsilon_0"],
        })
    original_call = li._call
    li._call = fake_call
    try:
        result = li.parse_question(
            "Two charges 2uC and 3uC are 0.5m apart in vacuum. Find the force."
        )
    finally:
        li._call = original_call

    assert "g" not in result["given"], \
        "g must NOT be force-injected for an electrostatics problem"
    assert "epsilon_0" in result["given"], "epsilon_0 should be injected (Stage 1 flagged it)"
    print(f"  given keys: {list(result['given'].keys())}")
    print("  PASSED")


def test_universal_constants_excludes_g():
    """UNIVERSAL_CONSTANTS must contain true constants but exclude g."""
    print("\n[test_universal_constants_excludes_g]")
    from config import UNIVERSAL_CONSTANTS, PHYSICAL_CONSTANTS
    assert "g" not in UNIVERSAL_CONSTANTS
    assert "g" in PHYSICAL_CONSTANTS  # still excluded from frontier-item eligibility
    for true_const in ("c", "G", "epsilon_0", "pi", "h_planck"):
        assert true_const in UNIVERSAL_CONSTANTS, f"{true_const} should be universal"
    print("  PASSED")


def test_backtrack_triggers_on_unflagged_failure():
    """
    Regression for bug #3: a confidently-wrong equation pick (NO
    conditions_concern flagged) that fails in Stage 3 must still trigger
    backtracking — not just LLM-self-flagged provisional picks.

    Synthetic "F = sqrt(-m)" is offered as a candidate alongside the real
    ones; the mock selector picks it on attempt 1 (no concern raised), Stage
    3 fails (no real solution), and the pipeline must retry with it excluded.
    """
    print("\n[test_backtrack_triggers_on_unflagged_failure]")
    import solver.pipeline as pl
    from solver.graph_loader import GraphIndex, load_graphs

    WRONG_EQ = {
        "id": "synthetic_bad_force_eq",
        "equation_str": "F = sqrt(-m)",
        "sympy_expr": "Eq(F, sqrt(-1*m))",
        "variables": {
            "F": {"name": "force", "unit": "N", "dimension": "MLT-2"},
            "m": {"name": "mass",  "unit": "kg", "dimension": "M"},
        },
        "conditions": [], "rag_text": "test fixture", "common_mistakes": [],
    }

    def fake_parse(question, valid_domains=None):
        return {
            "given": {
                "rho": {"value": 8000, "unit": "kg/m^3", "name": "density", "dimension": "ML-3"},
                "V":   {"value": 0.5,  "unit": "m^3",    "name": "volume",  "dimension": "L3"},
                "u":   {"value": 0.0,  "unit": "m/s",    "name": "initial velocity", "dimension": "LT-1"},
                "v":   {"value": 30.0, "unit": "m/s",    "name": "final velocity",   "dimension": "LT-1"},
                "s":   {"value": 40.0, "unit": "m",      "name": "displacement",     "dimension": "L"},
            },
            "unknown": {"symbol": "F", "name": "net force", "unit": "N", "dimension": "MLT-2"},
            "implicit_constants": [],
            "likely_domains": ["laws_of_motion", "kinematics"],
        }

    def fake_selector(question, available, round_data, round_num=0):
        results = []
        for rd in round_data:
            fi, cands = rd["frontier_item"], rd["candidates"]
            if fi.symbol == "F" and any(c["id"] == "synthetic_bad_force_eq" for c in cands):
                chosen, reason = WRONG_EQ, "Confidently wrong, no concern flagged."
            elif fi.symbol == "F":
                chosen = next((c for c in cands if c["id"] == "laws_of_motion_newton_second_law"), cands[0])
                reason = "Correct pick after backtrack."
            elif fi.symbol == "m":
                chosen = next((c for c in cands if c["id"] == "general_density_definition"), cands[0])
                reason = "rho = m/V"
            elif fi.symbol == "a":
                chosen = next((c for c in cands if c["id"] == "kinematics_v2_u2_2as"), cands[0])
                reason = "v^2 = u^2 + 2as"
            else:
                chosen, reason = (cands[0] if cands else None), "default"
            results.append({
                "frontier_item": fi, "chosen_eq": chosen, "reason": reason,
                "conditions_concern": None,  # never flagged — the whole point of this test
                "deferred": False, "_candidates": cands,
            })
        return results

    orig_candidates = GraphIndex.candidates_for_quantity
    def patched_candidates(self, needed_symbol, needed_name, needed_dimension, visited_eqs, allowed_domains=None):
        real = orig_candidates(self, needed_symbol, needed_name, needed_dimension, visited_eqs, allowed_domains)
        if needed_symbol == "F" and "synthetic_bad_force_eq" not in visited_eqs:
            return [WRONG_EQ] + real
        return real

    # Patch
    pl.parse_question        = fake_parse
    pl.call_round_selector   = fake_selector
    pl.narrate_from_trace    = lambda **kw: "mock explanation"
    pl.generate_distractors  = lambda **kw: [{"value": i, "unit": "N", "mistake": "x"} for i in (1, 2, 3)]
    GraphIndex.candidates_for_quantity = patched_candidates

    try:
        graph  = load_graphs()
        solver = pl.PhysicsSolver(graph)
        resp = solver.solve(
            "A body of density 8000 kg/m^3 and volume 0.5 m^3 starts from rest "
            "and reaches 30 m/s after 40 m. Find the net force."
        )

        print(f"  confidence:  {resp.confidence}")
        print(f"  final_value: {resp.final_value}")
        print(f"  chain:       {resp.chain_summary}")

        attempts_seen   = sorted({e["attempt"] for e in resp.decision_log})
        bad_eq_entries  = [e for e in resp.decision_log if e.get("chosen_eq_id") == "synthetic_bad_force_eq"]

        assert resp.error == "",            f"Should have succeeded, got: {resp.error}"
        assert resp.confidence == "HIGH"
        assert abs(resp.final_value - 45000.0) < 0.1
        assert len(attempts_seen) >= 2,     "decision_log must show more than one attempt"
        assert len(bad_eq_entries) == 1,    "wrong equation should appear exactly once (the abandoned attempt)"
        assert bad_eq_entries[0]["attempt"] == 1
        assert "synthetic_bad_force_eq" not in resp.chain_summary[-1]
        print("  PASSED")
    finally:
        # Restore — avoid leaking the monkeypatch into other tests
        GraphIndex.candidates_for_quantity = orig_candidates


def run_all():
    tests = [
        test_no_dict_literal_crash_on_implicit_injection,
        test_g_not_force_injected_for_unrelated_domain,
        test_universal_constants_excludes_g,
        test_backtrack_triggers_on_unflagged_failure,
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
