"""
tests/test_live.py
Full end-to-end tests requiring a GROQ_API_KEY.

These verify that the real LLM (llama-3.1-8b-instant for selection,
llama-3.3-70b-versatile for narration) makes the correct conceptual
choices and that the complete 5-stage pipeline produces valid answers.

Run: GROQ_API_KEY=<key> python3 tests/test_live.py
  or: python3 tests/test_live.py  (will skip all tests if no key)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def _has_key():
    from dotenv import load_dotenv
    load_dotenv()
    return bool(os.getenv("GROQ_API_KEY", ""))


# ── Test cases ────────────────────────────────────────────────────────────────

TESTS = [
    # ── Regression: original failing case ─────────────────────────────────────
    {
        "name": "F=ma  (NOT F=ρVg) — original failing case",
        "question": (
            "A body of density 8000 kg/m³ and volume 0.5 m³ starts from rest "
            "and reaches 30 m/s after travelling 40 m. Find the net force."
        ),
        "expected_answer":   45000.0,
        "tolerance_pct":     1.0,
        "expect_in_chain":   ["laws_of_motion_newton_second_law",
                              "general_density_definition",
                              "kinematics_v2_u2_2as"],
        "must_not_be_in_chain": ["fluid_mechanics_buoyant_force"],
    },

    # ── Implicit constant injection ────────────────────────────────────────────
    {
        "name": "Coulomb's law (ε₀ implicit, 'in vacuum')",
        "question": (
            "Two point charges of 2 μC and 3 μC are placed 0.5 m apart in vacuum. "
            "Calculate the electrostatic force between them."
        ),
        # F = kq1q2/r²,  k=1/(4πε₀)≈9e9
        # F = 9e9 * 2e-6 * 3e-6 / (0.5)² = 9e9 * 6e-12 / 0.25 = 0.216 N
        "expected_answer":  0.216,
        "tolerance_pct":    2.0,
        "expect_in_chain":  ["electrostatics_coulomb_law"],
    },

    # ── Simple 1-step ─────────────────────────────────────────────────────────
    {
        "name": "F = ma (direct, all given)",
        "question": "A 5 kg object experiences a net force of 20 N. Find its acceleration.",
        "expected_answer":  4.0,
        "tolerance_pct":    0.5,
        "expect_in_chain":  ["laws_of_motion_newton_second_law"],
    },

    # ── 2-step chain ──────────────────────────────────────────────────────────
    {
        "name": "Kinetic energy (compute v first from kinematics)",
        "question": (
            "A 2 kg object starts from rest and accelerates at 3 m/s² for 4 s. "
            "Find its kinetic energy."
        ),
        # v = u + at = 0 + 3*4 = 12  →  KE = ½mv² = ½*2*144 = 144 J
        "expected_answer":  144.0,
        "tolerance_pct":    1.0,
        "expect_in_chain":  [],   # just verify answer
    },

    # ── No valid chain (genuinely missing info) ────────────────────────────────
    {
        "name": "Missing information — should return UNVERIFIED",
        "question": "Find the velocity of the object.",   # no given values at all
        "expected_confidence": "UNVERIFIED",
        "expected_answer": None,  # don't check numerical answer
    },
]


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_live_tests():
    if not _has_key():
        print("GROQ_API_KEY not set — skipping all live tests.")
        print("Set GROQ_API_KEY=<key> to run end-to-end tests.")
        return True

    from solver.graph_loader import load_graphs
    from solver.pipeline     import PhysicsSolver

    graph  = load_graphs()
    solver = PhysicsSolver(graph)

    passed = failed = skipped = 0

    for test in TESTS:
        name = test["name"]
        print(f"\n{'─'*60}")
        print(f"TEST: {name}")
        print(f"Q:    {test['question'][:80]}...")

        try:
            resp = solver.solve(test["question"])

            print(f"  confidence:    {resp.confidence}")
            print(f"  answer:        {resp.final_value} {resp.final_unit}")
            print(f"  answer_exact:  {resp.final_value_exact}")
            print(f"  chain:         {resp.chain_summary}")
            if resp.error:
                print(f"  error:         {resp.error}")

            # Check expected_confidence
            exp_conf = test.get("expected_confidence")
            if exp_conf and resp.confidence != exp_conf:
                raise AssertionError(
                    f"Confidence {resp.confidence!r} ≠ expected {exp_conf!r}"
                )

            # Check numerical answer
            exp_ans = test.get("expected_answer")
            if exp_ans is not None:
                tol = test.get("tolerance_pct", 2.0) / 100.0
                diff = abs(resp.final_value - exp_ans) / max(abs(exp_ans), 1e-12)
                if diff > tol:
                    raise AssertionError(
                        f"Answer {resp.final_value} differs from expected "
                        f"{exp_ans} by {diff*100:.1f}% (tolerance {tol*100}%)"
                    )

            # Check equations in chain — use only the WINNING attempt's
            # entries. decision_log now accumulates across every backtrack
            # attempt (tagged "attempt": N) so abandoned/corrected picks stay
            # visible for narration. That's good for transparency, but it
            # means a naive "appears anywhere in decision_log" check would
            # wrongly fail a case where the system picked wrong, caught it
            # via Stage 3 failure, and self-corrected — exactly the behavior
            # we want, not a regression of the original bug.
            if resp.decision_log:
                final_attempt = max(e.get("attempt", 1) for e in resp.decision_log)
                chain_ids = {
                    e.get("chosen_eq_id", "") for e in resp.decision_log
                    if e.get("attempt", 1) == final_attempt
                }
            else:
                chain_ids = set()

            for must_have in test.get("expect_in_chain", []):
                if must_have not in chain_ids:
                    raise AssertionError(
                        f"Expected equation {must_have!r} in chain, got: {chain_ids}"
                    )

            for must_not in test.get("must_not_be_in_chain", []):
                if must_not in chain_ids:
                    raise AssertionError(
                        f"Equation {must_not!r} must NOT be in chain (old algorithm failure mode)"
                    )

            print(f"  ✓ PASSED")
            passed += 1

        except Exception as e:
            import traceback
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Live tests: {passed} passed, {failed} failed  ({len(TESTS)} total)")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    ok = run_live_tests()
    sys.exit(0 if ok else 1)
