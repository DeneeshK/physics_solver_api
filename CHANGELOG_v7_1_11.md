# Changelog v7.1.11 — Three generic plumbing fixes (units, dimensions, executor)

After v7.1.10's 7B migration, the 5-question live run isolated three failures
that are NOT the model and NOT specific to those questions. Each fix is generic
— it works for every unit, every dimension, and every symbol, not the five test
cases.

## The user's diagnosis was correct

The user noticed F=ma-direct PASSED but F=ma-chain FAILED with "none (it went
to rotational motion)", even though both should use the same equation. They
suspected "some hard coding or forced patching" was dropping the right
equation. That was exactly right — see Bug 3.

## Bug 3 (the one the user flagged) — Unicode dimensions silently drop candidates

In the F=ma-chain case, retrieval returned `laws_of_motion_newton_second_law`
at score 1.0, but by the time candidates reached Stage 2, only
`rotational_motion_torque_inertia` survived — Newton's second law was filtered
out. Root cause: the 7B emitted the force dimension as `MLT⁻²` (Unicode
superscripts) while the graph stores `MLT-2` (ASCII). The dimension-
compatibility check compared them as strings via `_normalize_dimension`, which
stripped `^`/`*`/spaces but did NOT handle Unicode superscript digits. So
`MLT⁻²` parsed as MLT (all exponent 1) ≠ `MLT-2`, the filter judged them
incompatible, and dropped the equation.

In F=ma-DIRECT, the model happened to emit ASCII `LT-2`, which matched — which
is why the same equation worked there. Pure formatting luck.

Fix: `_normalize_dimension` now folds Unicode superscript digits (⁰¹²³…),
Unicode superscript +/- (⁺⁻), and the Unicode MINUS SIGN (U+2212) to ASCII
before parsing. `MLT⁻²` and `MLT-2` now normalize identically. Generic across
every dimension and equation. This was the "forced patching" the user suspected
— a string comparison that silently rejected correct, concept-matched
candidates.

## Bug 1 — Microcoulombs (and all prefixed units) not converted to SI

Coulomb's law was picked perfectly and the chain was correct, but the answer
was off by 10^12: Stage 1 stored `q1: value 2, unit "μC"` instead of
`2e-6 C`, so SymPy computed with 2 coulombs. The Stage 1 prompt asks the model
to output SI, but the 7B is unreliable about it.

Unit conversion is deterministic arithmetic — it belongs in code, not the
LLM's head (and doing it in code honors the design principle that the LLM
shouldn't compute). Fix: `_normalize_given_to_si` runs after Stage 1 parsing.
It detects SI prefixes (μ, m, k, c, n, p, G, M, …) on the leading symbol and a
table of common compound/non-SI units (g→kg, cm→m, nm→m, km/h→m/s, eV→J, …)
and applies the correct factor in code. Conservative: unknown units are left
untouched (no guessing), and SI base units are protected from decomposition
(the kg trap — `kg` is NOT kilo+gram). Generic across every prefixed unit and
every question — this fixes all the μC/nm/mm/cm/g questions in the 89-bank, not
just Coulomb.

## Bug 2 — Executor KeyError on canonicalized symbols

The KE chain crashed with `KeyError: 'K'` in `sympy_executor._execute_
simultaneous`. The unknowns list-comp looked up `fi.symbol` ('K', the
canonicalized target) in `all_var_syms`, which was built only from the
equations' literal variable names. When the canonicalized target symbol isn't
among them, KeyError.

Fix: register any unknown symbol not already in `all_var_syms` (create the
SymPy symbol) rather than crashing. Generic — handles any canonicalized
(E_k→K, PE→U, KE→K) or otherwise-unlisted unknown, not one specific symbol.

## What these fixes do and don't claim

These are PLUMBING fixes — they remove three mechanical defects that were
discarding correct work. They do not change the architecture, and they are not
tuned to the five test questions. After these:
  - Coulomb's law should compute correctly (μC converted).
  - The KE chain won't crash (symbol registered) — though whether it gets the
    right K depends on the model picking a complete chain, which is a separate
    (model) question.
  - F=ma-chain's right equation will reach Stage 2 (dimension matches), so the
    model can pick it.

The remaining open question is purely about the model's chain-building quality
on multi-step problems, which the 89-bank + evaluate_bank cross-tab will
quantify.

## Files changed in v7.1.11

  - `solver/graph_loader.py` — `_normalize_dimension` folds Unicode
    superscripts/minus to ASCII.
  - `solver/llm_interface.py` — `_normalize_given_to_si` (new) + call in
    `parse_question` after extraction, before constant injection.
  - `solver/sympy_executor.py` — `_execute_simultaneous` registers unlisted
    unknown symbols instead of KeyError.
  - `tests/test_deterministic.py` — 3 new tests (Unicode dimension folding,
    SI normalization battery, canonicalized-unknown registration).

## Deployment

```bash
tar -xzf physics_solver_v7_1_11_tar.gz
cd physics_solver_v7
python tests/test_deterministic.py        # expect: 57 passed
python tests/test_live.py                  # re-run the 5
```

No re-ingest needed (no rag_text changes). Embedder stays on CPU, 7B stays
resident — v7.1.10 settings unchanged.
