# Changelog v7.2.3 — Agentic working memory + culprit-specific rollback

Two fixes addressing the two real problems the v7.2.2 run exposed: (1) the
model picking a conflicting equation for a sub-variable because each decision
was made blind to the rest of the solve, and (2) the rollback excluding the
wrong equation (the Round-0 root) instead of the actual culprit deeper in the
chain.

## Problem 1: each sub-decision was amnesiac → conflicting picks

The density→force question completed its chain but computed F = 0. Why: to get
mass, the model chose F = m*g (weight) instead of Density = m/V. Since F = m*a
was already the chain for force, having BOTH F = m*a and F = m*g forces a = g —
a contradiction that collapses to F = 0. The model picked weight because, when
asked only "find m", weight DOES contain m and looked fine. It had no awareness
that F was the ultimate goal or that a was being solved separately.

Fix — a SMALL agentic working memory carried into each Stage 2 decision
(the user's design: "the LLM should be aware of the steps we took — small,
small things"). Each call now shows:

    SOLVING FOR (ultimate goal): F (net force)
    SOLUTION SO FAR (already chosen — do not re-derive or contradict these):
      • F ← F = m*a
      • a ← v**2 = u**2 + 2*a*s
    SYMBOLS ALREADY BEING SOLVED (do NOT pick an equation that re-introduces
    or conflicts with these): F, a

Only COMMITTED steps go in — not dead-ends, not rejected attempts — so it stays
tiny (a few lines). A new system-prompt principle (Decision principle 6)
instructs the model to use it: never pick a candidate that re-introduces a
symbol already being solved if it creates a circular/conflicting constraint;
the F=m*g-for-mass trap is given as the worked example. The model is told to
choose an equation that resolves the needed symbol WITHOUT dragging the goal or
an already-solved symbol back in (Density = m/V for mass).

This is concept/meaning-based rejection by the LLM — no hardcoding, no
symbol/dimension gate. The graph still hands over candidates; the model now has
the context to reject the conflicting one itself.

Code:
  - solver/llm_interface.py: `_round_select_call` and `call_round_selector`
    take a `solve_context` dict; a compact working-memory block is rendered
    into the prompt. New Decision principle 6 in ROUND_SELECT_SYSTEM.
  - solver/frontier_resolver.py: builds solve_context (goal, committed steps,
    being-solved symbols) each round and passes it through llm_round_fn.

## Problem 2: rollback excluded the root, not the culprit

v7.2.2's rollback, on a dead-end, excluded the Round-0 root equation. But the
dead-end is usually caused by a pick DEEPER in the chain, not the root. Banning
the root threw away a good starting equation — which broke the kinetic-energy
chain (the good K = 0.5*m*v^2 root got excluded because a deeper step
dead-ended, and the retry then had no kinetic-energy equation and the model
hallucinated a fake id).

Fix — on a dead-end while chasing `fi`, exclude the equation that INTRODUCED
fi (fi.introduced_by), i.e. the specific deeper pick that led down the dead
branch, rather than the Round-0 root. The retry re-picks THAT step while the
good root and good earlier steps stay intact. Falls back to the root only when
fi has no recorded introducer (it was the main unknown itself).

Code:
  - solver/frontier_resolver.py: dead_end_root_eq is now fi.introduced_by
    (culprit) with root as fallback.

## What did NOT change

  - The neighbor-walk, ranking, SI conversion, unicode-dimension folding,
    bare-selection parsing — all intact.
  - No hardcoding; the LLM remains the only thing deciding fit, now with
    working-memory context.

## Tests

59 deterministic tests pass.

## VERIFIED vs NEEDS-YOUR-MACHINE

VERIFIED in sandbox:
  - 59 tests pass.
  - The working-memory block renders correctly and compactly (confirmed by
    capturing the actual Stage 2 prompt for the F=ma chain: goal, the two
    committed steps, and "SYMBOLS ALREADY BEING SOLVED: F, a" all present).

NEEDS YOUR RUN TO CONFIRM (the 7B's actual behavior can't be tested here):
  - That the 7B USES the working memory to reject F=m*g for mass and instead
    pick Density = m/V — fixing the density→force = 0 bug. The context is now
    in front of it and the system prompt tells it what to do; whether the 7B
    obeys is what the run shows. This model has ignored instructions before
    (it hallucinated an equation id last run), so this raises the odds but is
    not a guarantee.
  - That culprit-exclusion lets the kinetic-energy chain survive its dead-end
    and complete (144 J) instead of regressing.

## Deployment

```bash
tar -xzf physics_solver_v7_2_3_tar.gz
cd physics_solver_v7
python tests/test_deterministic.py        # expect: 59 passed
python tests/test_live.py                  # the 5
```

Watch the Stage 2 prompts in the log — they now carry the SOLUTION SO FAR
block. And watch whether the density→force case picks Density = m/V for mass
(success) or F = m*g (the old conflict).

No re-ingest. Embedder CPU, 7B resident.

## Still open (honest)

  - Constant bloat (R_g into unrelated questions) — still present, cosmetic.
  - The underspecified "find velocity" case still burns all retries before
    correctly failing (~100s). It ends UNVERIFIED (correct) but slowly; a
    fast-path "truly nothing given" check could short-circuit it. Not done
    yet — it's a performance issue, not a correctness one.
