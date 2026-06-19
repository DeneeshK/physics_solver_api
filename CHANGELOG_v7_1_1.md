# Changelog v7.1.1 — Complete hand-authored rag_text corpus

v7.1.1 is the practical conclusion of the v7.1 rewrite. The shipped
`scripts/rag_text_exemplars.json` now contains **all 182 equations**,
hand-authored end to end. No LLM batch is needed — and won't fail on
Groq free-tier TPM caps the way v7.1's regenerate script did.

## Why this changed from v7.1

The v7.1 design relied on two paths:
  1. 16 hand-authored exemplars as the gold standard.
  2. An LLM batch generator (`scripts/regenerate_rag_texts.py`) for the
     remaining 166, using the 16 as in-context demonstrations.

Path 2 ran into a real constraint on user-side deployment: the Groq free
tier's tokens-per-minute cap halts the batch at ~29 equations. The
generator handled that with backoff, but the resulting partial corpus
(some equations on the new style, others still on the v6 templates) is a
worse state than either extreme.

v7.1.1 collapses both paths into one: every rag_text is hand-authored.
The batch generator is kept in the package for future incremental
additions (e.g. when new equations get added to the graph), with the
v7.1.1 prompt and validator improvements already integrated. But for
the current 182, no generator run is needed.

## The corpus

  - 182 / 182 equations covered
  - 109,146 characters total, 599 average per equation
  - Every rag_text passes the v7.1.1 validator (length, banned phrases,
    no word-concatenation patterns)
  - Every concept_name is unique across all 182 entries (the
    "unique-identifier" principle the user emphasized)

## Validator improvements in v7.1.1

The v7.1 validator over-rejected a few legitimate cases. Two adjustments:

  - **Removed `"this equation"` from BANNED_PHRASES**. It was meant to
    catch generic LLM filler like "This equation describes..." but it
    also flagged legitimate references like "this equation gives the net
    force required..." in the buoyancy and centripetal exemplars.
    Leaving `"as mentioned"`, `"the formula above"`, `"step by step"`,
    etc. — those remain clearer filler-tells.

  - **Added word-concatenation detector**. The 70B model occasionally
    drops spaces during streaming, producing tokens like
    `finalvelocities`, `accelerationwhen`, `variableacceleration`. The
    validator now rejects:
      - Any unbroken lowercase run ≥ 17 chars (longer than any standard
        English physics word — `electromagnetic`=15, `characteristic`=14)
      - Common merged-word patterns: a word ending in `-ation`, `-ity`,
        `-ence`, `-ance`, `-tion`, `-sion` immediately followed by a
        common function-word starter (`when`, `with`, `and`, `of`, ...)

  Both prevent the generator from silently letting cosmetic typos
  through if it's run again in the future.

## SYSTEM_PROMPT improvements in v7.1.1

Three new rules in the regenerate script's prompt, addressing the
specific quality issues observed in the v7.1 dry-run:

  8. **VERIFY THE PHYSICS BEFORE DESCRIBING**. The equation_str alone
     isn't always enough to determine scope of applicability. Example:
     `v = s/t` is the GENERAL definition of average velocity, not just
     for uniform motion. The dry-run had narrowed this to "uniform
     motion only" — incorrect. The new rule explicitly asks the LLM to
     check whether the equation is a general/defining form or a special
     case, and write accordingly.

  9. **WORD SPACING**. Explicit instruction to use single-space
     separation between every word, with examples of common merges to
     avoid (`finalvelocity`, `constantacceleration`).

  10. **LOOK-ALIKE COMPARISONS MUST BE TO REAL EQUATIONS**. The dry-run
      had one rag_text "distinct from `s = ut + at`, which incorrectly
      omits the one-half factor" — but `s = ut + at` isn't a real
      equation in the graph, it's a hypothetical wrong form. New rule
      requires look-alikes to be real equations from the JEE/NEET
      curriculum.

## Graph-content quirks worked around in the rag_texts

Several equations are stored with variable schemes that don't fully
match standard physics notation. The rag_texts describe the **intended**
concept faithfully, with explicit notes when the stored form differs
from standard convention. List for v7.3 graph-content cleanup:

  - `kinematics_relative_velocity`: stored as `v = u - a` — `a` is meant
    as second-body velocity, not acceleration
  - `momentum_collisions_*`: variables `a`, `s` used as 2nd-body final
    velocities (kinematic letters pressed into duty)
  - `sound_beat_frequency`: `f = u - v` — `u`, `v` are the two
    frequencies, not velocities
  - `sound_doppler_*`: `f` on both sides; LHS is observed frequency, RHS
    is source frequency
  - `thermodynamics_first_law`: `Q = DeltaV + W` — `DeltaV` here means
    `ΔU` (internal energy), not volume change
  - `thermodynamics_carnot_efficiency`: `eta = 1 - t/T` — `t`, `T` are
    cold and hot reservoir temperatures
  - `thermodynamics_isothermal_work`: `log(V)` should be `log(Vf/Vi)`
  - `modern_physics_rydberg_formula`: uses `R_g` (gas constant symbol)
    but the formula needs Rydberg constant; also `u`, `v` are quantum
    numbers, not velocities
  - `nuclear_physics_binding_energy` / `q_value`: `DeltaV` used but
    should be `Δm` (mass defect)
  - `current_electricity_resistivity_temp`: `R_e` on both sides — meant
    as `R(T) = R_0*(1+alpha*ΔT)`
  - `work_energy_power_efficiency`: stored as `P/W` — should be
    `P_out/P_in` or `W_out/W_in`

Per the user's request, these are flagged for v7.3 graph cleanup; the
rag_texts describe the intended concepts so retrieval still picks the
right node.

## Files added / changed in v7.1.1

  - `scripts/rag_text_exemplars.json` — now contains all 182 entries
    (was 16 in v7.1)
  - `scripts/build_complete_exemplars.py` — the builder script
    containing every rag_text in source-controlled Python form. Running
    it regenerates the JSON.
  - `scripts/regenerate_rag_texts.py` — kept for future incremental
    additions; SYSTEM_PROMPT and validator improvements integrated
  - `scripts/apply_exemplars_only.py` — unchanged from v7.1; now applies
    all 182 exemplars instead of 16
  - `tests/test_deterministic.py` — adapts automatically (uses dynamic
    `len(ex_data['exemplars'])`); test names already in place from v7.1
  - `CHANGELOG_v7_1_1.md` — this file

## Deployment

After git restore to before your v7.1 changes:

```bash
tar -xzf physics_solver_v7_1_1_tar.gz
cd physics_solver_v7
pip install -r requirements.txt

# 1. Create .env with your Groq key
cp .env.template .env
nano .env

# 2. Apply all 182 hand-authored exemplars (no LLM needed)
python scripts/apply_exemplars_only.py

# 3. Verify deterministic tests
python tests/test_deterministic.py        # expect: 40 passed, 0 failed
                                           # (with 182/182 applied,
                                           #  182 distinct concepts)

# 4. Rebuild ChromaDB over the new rag_texts (~10 min first time)
python -m solver.ingest --reingest        # --reingest is required —
                                           # without it the script will
                                           # refuse to rebuild over an
                                           # existing index

# 5. Run live stress tests
python tests/test_live.py
```

No Groq batch run. No TPM cap. No partial coverage.

## What v7.1.1 still does not yet do

  - **Stage 4 narration** (planned for v7.2): use `candidates_shown` from
    the decision log to make narration template-driven rather than
    re-asking the LLM to quote its own trace.

  - **Graph content cleanup** (v7.3): all the variable-scheme issues
    flagged above. The rag_texts describe intended concepts, so
    retrieval is correct; but SymPy execution on those nodes may be
    affected by the stored scheme. Separate workstream.

  - **SIGNED_SYMBOLS** still hardcoded in `sympy_executor.py` (v7.3).

  - **Symbol-table union in landing layer**: still active. Once
    ChromaDB landing proves out on live tests, the symbol-table layer
    can be stripped — but only after evaluation, not preemptively.
