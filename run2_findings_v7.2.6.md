# v7.2.6 — Run-2 Log Analysis (post Fix 1+2)

Run: 70/78 questions executed before a hard SymPy hang on Q70 forced a manual stop.
Source of truth: `logs/solver.log` (8,597 lines, analyzed by streaming aggregation —
never loaded whole). Per-run terminal tally from the log:

- **19 solve_success**, **47 solve_error**, **4 no-terminal (hung)**.
- Eval console showed ~21 ✓ because it also counts correct *refusals* of
  underspecified questions (Q22 "Find the electric field at a point", Q36 "Find
  the velocity of the object", etc.) as passes.
- Baseline was 17/78. The symbol/unit fixes helped, but most questions have a
  SECOND blocker beyond symbols, so the solve-count gain is modest.

The fixes confirmed firing: `given_symbols_canonicalized` ×10,
`target_symbol_canonicalized` ×10, `stage1_si_normalized` ×13.

---

## Bugs FIXED this session (code)

1. **SymPy runaway hang** — `sympy_executor.py` `_execute_simultaneous`/`_execute_step`.
   `solve()`/`evalf()` spun >15 min on a spurious transcendental simultaneous
   group (Q70 Young's double-slit; traceback deep in polynomial factorization
   inside `checksol`). `try/except` can't catch runaway CPU. Added a SIGALRM
   wall-clock guard (`_time_limit`, `SOLVE_TIMEOUT_S=8`) around every `solve()`
   and `float(sympy_N())`. On timeout the step fails → resolver backtracks.
   Likely also unblocks the other 3 hangs (Q7, Q24, Q26).

2. **Stage 1 crash: `'list' object has no attribute 'get'`** (Q30 satellite, Q43
   straight-wire) — the 7B sometimes emits `"unknown"` as a LIST; `parse_question`
   then crashed at the `log(...unknown.get("symbol")...)` line, losing the whole
   question. Added shape-coercion of `unknown`/`given` right after parse.

3. **My own Fix-1 alias errors** (config.py):
   - `K_max→K` was WRONG: graph photoelectric node is `Kmax = h_planck*nu - W0`
     (symbol `Kmax`). Corrected `K_max/KE_max → Kmax`.
   - Removed the pre-existing `E (energy dim) → K` rule: it rewrote photon energy
     (`E = h_planck*nu`, graph uses `E`) to kinetic `K` and broke every
     photon-energy question (Q44).

---

## Dominant remaining blocker: GRAPH DATA-QUALITY BUGS

The single biggest discovery this run. Many equations EXIST but carry variable
names that are wrong, collide with other quantities, or make a chain unsolvable.
The LLM is mostly judging correctly; the graph is feeding it broken symbols.

| Node | Equation as stored | Problem |
|---|---|---|
| `thermodynamics_first_law` | `Q = DeltaV + W` | internal-energy change named **`DeltaV`** (reads as ΔVolume). Q68 asks ΔU → never matches. |
| `sound_beat_frequency` | `f = u - v` | beat freq from two frequencies named **`u`,`v`** (velocity symbols). Collides everywhere. |
| `sound_doppler_source_moving` | `f = f*v/(v - u)` | **degenerate**: same symbol `f` on both sides; source/observed freq not distinguished. Unusable. |
| `sound_doppler_observer_moving` | `f = f*(v + u)/v` | same degeneracy. |
| `momentum_collisions_conservation_two_body` | `m*u + M*v = m*a + M*s` | final velocities named **`a`,`s`** (accel/displacement). |
| `kinematics_relative_velocity` | `v = u - a` | relative velocity uses **`a`** as a velocity. |
| moment of inertia (disc/sphere) | — | appears **MISSING** (Q57 found no inertia equation). |
| heat conduction (Fourier) | — | **MISSING** (Q65 `Q_dot`, no node). |
| capacitance | `C` vs `C_cap` split | `C=q/V`, parallel-plate use `C`; reactance/equivalence use `C_cap`. Chains can't bridge (Q1,6,7,8). |

These are why "the right equation was found" still fails downstream.

---

## Remaining failure buckets (47 errors + 4 hangs)

**A. Target symbol mismatch — equation EXISTS under a different graph symbol
(alias-fixable, but several are dimension-/context-dependent so need care):**
- Q18 `e` → graph `emf` (faraday_law)
- Q45 `K_max` → `Kmax` (FIXED)
- Q52 `λ` (unicode) → `lambda_decay`
- Q53/54 `A_final`/`A` → `A` (activity exists)
- Q55/56 `s'`/`f` → `v_i`/`f_lens` (lens_formula)
- Q60 `f_observed` → `f`, Q61 `fb` → `f`
- Q63 `P_excess` → `P` (excess_pressure)
- Q66 `eta` → `eta` (exists; downstream needs t,T)
- Q16 `sigma` → `stress` (BUT `sigma` is surface tension elsewhere → dimension-aware only)

**B. Unicode / prime symbols Stage 1 emits that no alias matches:**
`λ` (Q52), `ΔU` (Q68), `W'` (Q31), `s'` (Q55), `ΔU`/`Δ`-forms. Need a
normalization pass (greek→ascii, strip primes) before canonicalization.

**C. Genuinely missing equations:**
- moment of inertia (disc, sphere, rod) + `tau = I*alpha`, `KE_rot = ½Iω²` (Q57,58)
- heat conduction `Q_dot = k*A*ΔT/L` (Q65)
- parallel resistance `1/R_eq = Σ1/R_i` (Q14); general series capacitance (Q8)
- SHM `v_max = ω*A` (Q5)

**D. `llm_invalid_id` — model knows the concept, emits a wrong id string:**
Q15 `ohms_law` (real id `current_electricity_ohms_law`). A fuzzy-id fallback
recovers these.

**E. C↔C_cap capacitance split** — Q1, Q6, Q7, Q8 (see graph-bug table).

**F. Multi-step downstream dead-ends (rb=6 exhausted):** Q19, Q25, Q41, Q46,
Q50, Q59, Q67 — chain picked a root then couldn't bridge a sub-symbol, often
because of a graph-bug symbol (e.g. two-body `a`/`s`, relative-velocity `a`).

---

## Recommended next order

1. **Re-run the eval** with the 3 code fixes (no re-ingest needed) — confirms the
   hang is gone and Q30/43/44/45 recover.
2. **Graph data cleanup** (biggest lever, needs re-ingest after):
   rename `DeltaV→DeltaU` in first_law; fix beat/Doppler frequency symbols;
   fix two-body `a,s→v1f,v2f` and relative-velocity `a→v_rel`; standardize
   capacitance to one symbol; add the missing equations (inertia, conduction,
   parallel R, v_max).
3. **Unicode/prime symbol normalization** in Stage 1 post-processing.
4. **Dimension-aware target aliases** for bucket A (sigma→stress, e→emf, etc.).
5. **Fuzzy-id fallback** for bucket D.
