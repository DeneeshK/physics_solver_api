# Changelog v7.1.6 — Symbol canonicalization + anti-hallucination guard

The v7.1.5 live run jumped from 1/5 to 3/5, with both original failing
cases (F=ma chained, Coulomb's) passing clean and no rate-limit crash.
v7.1.6 fixes the two remaining failures, both of which the log pinned
to specific, non-model-capacity bugs.

## Live results that motivated v7.1.6

```
F=ma (NOT F=ρVg)        ✓ PASSED   45000 N
Coulomb's law           ✓ PASSED   0.2157 N
F = ma direct           ✓ PASSED   4.0 m/s²
Kinetic energy chain    ✗ FAILED   (theta dead-end)
"Find the velocity"     ✗ FAILED   (HIGH instead of UNVERIFIED)
```

## Fix 1 — Kinetic energy chain: symbol canonicalization (E/E_k → K)

The KE test failed with "LLM rejected all candidates for 'theta' in round
3". Tracing the log:

  - Stage 1 named the unknown 'E' (kinetic energy, dimension ML²T-2)
  - The graph's kinetic-energy equation is `K = 0.5*m*v**2` — it uses 'K'
  - The resolver treated 'E' and 'K' as DIFFERENT unknowns
  - Round 0 picked the KE equation for 'E', which produced 'K', a "new"
    unknown
  - Round 1 picked `W = K` (work-energy theorem) for K → dragged in 'W'
  - Round 2 picked `W = F*s*cos(theta)` for W → dragged in F, s, theta
  - Round 3: theta only has projectile candidates → LLM correctly says
    "none" → dead end

The LLM reasoning was correct at every step. The bug was symbol identity:
'E' and 'K' are the same physical quantity, but the system didn't know it.

Fix: a symbol canonicalization layer (`_canonicalize_symbol` in
pipeline.py, maps in config.py). The target symbol from Stage 1 is
normalized to the graph's canonical symbol before resolution:

  - Direct aliases (dimension-independent): E_k, Ek, KE, E_kin → K;
    E_p, Ep, PE, E_pot → U
  - Dimension-aware aliases: bare 'E' is ambiguous (energy? field? EMF?),
    so it's resolved by the dimension Stage 1 reported — 'E' with
    dimension ML2T-2 → K. 'E' with force dimension (MLT-2) is left alone.
  - Unicode superscripts (ML²T-2) are folded to ASCII (ML2T-2) for the
    dimension match, since Stage 1 emits both forms.

After canonicalization, the KE target is 'K', Round 0 sees
`K = 0.5*m*v**2` directly, picks it, needs only 'v' (derivable from
kinematics), and the chain completes in 2 rounds instead of wandering
into projectile motion.

The map is deliberately conservative — only unambiguous same-quantity,
same-dimension aliases. A wrong alias mapping would be worse than none.
Adding more aliases is a config.py edit, no code change.

## Fix 2 — "Find the velocity": anti-hallucination guard

The question "Find the velocity of the object." has NO numbers. But Stage
1 hallucinated given values:

  - given = {s: 5, g: 9.81, t: 2}  ← none of these are in the question

The pipeline then "correctly" solved v = s/t = 2.5 m/s with HIGH
confidence — a confident, fabricated answer to a question that was never
asked. This is the most dangerous failure mode: not a wrong answer to a
real question, but a real-looking answer to an invented question.

Fix: a "NEVER INVENT VALUES" section at the top of the Stage 1 prompt
rules:

  - Extract ONLY numeric values explicitly present in the question text,
    or that follow from an unambiguous phrase ("starts from rest" → u=0).
  - If the question has NO numeric data, "given" MUST be empty {}.
  - A question with no givens is UNDERSPECIFIED — return the empty set and
    let the solver report it can't proceed. That's the honest outcome.
  - Do not carry over numbers from physics seen before.

With an empty given set, the resolver has nothing to work with and
returns UNVERIFIED — the correct behavior the test expects.

## A note on what these two bugs have in common

Both are Stage 1 (the 8B parser) behaviors, and both are about the
boundary between what the question says and what the system assumes:

  - Fix 1: the system assumed 'E' ≠ 'K' when they're the same thing
    (under-assuming identity)
  - Fix 2: the model assumed values that weren't there (over-assuming
    data)

Neither is a reasoning-depth problem. Both are addressed by being
explicit — a canonicalization map and a prompt rule.

## Deployment

```bash
tar -xzf physics_solver_v7_1_6_tar.gz
cd physics_solver_v7
pip install -r requirements.txt

python tests/test_deterministic.py        # expect: 51 passed

# No re-ingest needed (no rag_text changes)
python tests/test_live.py
```

Expected after v7.1.6: all 5 live tests pass. The KE chain should resolve
via K = 0.5*m*v² → v from kinematics; "Find the velocity" should return
UNVERIFIED.

## A caveat on the KE fix

Canonicalization gets the target to 'K', so Round 0 sees the right
equation. There's a residual risk the LLM could still pick `W = K`
(work-energy theorem) over `K = 0.5*m*v²` at Round 0. The concept-level
prompt should prevent this — "Kinetic Energy of a Translating Body" is a
far better concept match for "find the kinetic energy" than "Work-energy
theorem" — but if the live test shows it still happening, the next step
is to deprioritize degenerate identity equations like `W = K` in landing
(they're rarely the right entry point). Watch the Round 0
stage2_item_decision for the KE test.

## Files changed in v7.1.6

  - `config.py`: added SYMBOL_ALIASES and SYMBOL_ALIASES_BY_DIMENSION maps
  - `solver/pipeline.py`: added `_canonicalize_symbol`; applied it to the
    target symbol after Stage 1 parse, with a target_symbol_canonicalized
    log event
  - `solver/llm_interface.py`: added "NEVER INVENT VALUES" section to the
    Stage 1 prompt
  - `tests/test_deterministic.py`: 2 new tests (canonicalization,
    anti-hallucination prompt rule)
  - `CHANGELOG_v7_1_6.md`: this file

## Known remaining items (v7.3 graph cleanup)

The `W = K` work-energy-theorem equation is a degenerate identity that
can trap chains. Other graph-content issues flagged earlier
(kinematics_relative_velocity = v - a, momentum conservation symbol
scheme, thermodynamics DeltaV-for-DeltaU, Rydberg R_g, etc.) remain for
the v7.3 graph cleanup pass.
