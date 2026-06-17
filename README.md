# JEE/NEET Physics Solver — v2 (5-Stage Pipeline)

AI-assisted physics problem solver for Indian competitive-exam students.
The student types a free-text question; the system identifies the correct
equation chain, computes an exact answer with SymPy, and produces a
step-by-step student explanation plus 3 wrong MCQ distractors.

## Key design principle (the reason this version exists)

**Equation selection is a conceptual decision, made by the LLM.**

The v1 algorithm scored candidates by "how many variables are already
known" — a numeric heuristic. On the original failing test case
(body with ρ, V, u, v, s given; find F), it picked `F = ρVg`
(buoyant force) because ρ, V, g were all directly available — even though
the question described a *kinematics* scenario with no fluid involved.

The redesign fixes this by making the LLM choose among pre-filtered
candidates based on what the question is *physically describing*, not
on which equation happens to be numerically convenient.

---

## 5-Stage Pipeline

```
Question text
   │
[Stage 1] Parse (LLM)
   │  → given values+units+dimensions, target symbol+quantity,
   │    implicit constants (ε₀ if "in vacuum", G if "gravitation", etc.)
   ▼
[Stage 2] Frontier Resolution (LLM + deterministic filtering)
   │  Round-based loop:
   │   1. Deterministic: candidates_for_quantity() filters sym_to_eqs[symbol]
   │      by dimension compatibility (§5 symbol-collision guardrail)
   │   2. LLM: batched call picks ONE equation per needed quantity based
   │      on the physical scenario, NOT on numeric completeness
   │   3. Topological sort → execution order; cycle detection → SimultaneousGroup
   ▼
[Stage 3] SymPy Assembly & Solve (deterministic)
   │  → exact Rational arithmetic throughout, substitution traces,
   │    multiple-root filtering, simultaneous-group solving
   ▼
[Stage 4] Narrate (LLM)
   │  → describes the already-correct trace in student-friendly language
   │    (never touches numbers — can't corrupt fractions)
   ▼
[Stage 5] Distractors (LLM)
   → 3 wrong MCQ options based on common_mistakes from equation nodes
```

---

## Directory structure

```
physics_solver/
├── config.py                   ← paths, constants, Groq models,
│                                  IMPLICIT_CONSTANTS_CATALOG
├── data/
│   └── physics_equation_graph_final.json  ← 182 equations (unchanged)
├── solver/
│   ├── graph_loader.py         ← GraphIndex + candidates_for_quantity()
│   ├── frontier_resolver.py    ← Stage 2: FrontierItem, ResolvedStep,
│   │                              SimultaneousGroup, resolve_frontier()
│   ├── llm_interface.py        ← Stage 1 parse, Stage 2 round selector,
│   │                              Stage 4 narration, Stage 5 distractors
│   ├── sympy_executor.py       ← Stage 3: exact arithmetic, traces,
│   │                              simultaneous groups, root filtering
│   ├── pipeline.py             ← PhysicsSolver (wires all 5 stages)
│   └── backward_chain.py       ← legacy (replaced by frontier_resolver)
├── api/
│   └── main.py                 ← FastAPI endpoints
└── tests/
    ├── test_deterministic.py   ← 14 tests, no API key needed
    └── test_live.py            ← end-to-end tests (requires GROQ_API_KEY)
```

---

## Quick start

```bash
cp .env.template .env
# edit .env and set GROQ_API_KEY=gsk_...

pip install groq sympy python-dotenv fastapi uvicorn pydantic

# Run deterministic tests (no API key needed)
python3 tests/test_deterministic.py

# Run live end-to-end tests
GROQ_API_KEY=<key> python3 tests/test_live.py

# Start the API server
uvicorn api.main:app --reload --port 8000
```

---

## What the graph stores (no changes from v1)

Each of the 182 equation nodes contains:
- `equation_str`, `sympy_expr` — the equation in two forms
- `variables` — `{symbol: {name, unit, dimension}}` per variable
- `conditions` — when the equation is valid
- `rag_text` — natural-language description (used in Stage 2 LLM prompt)
- `common_mistakes` — used for Stage 5 distractor generation

---

## Symbol-collision guardrail (§5 of brief)

`candidates_for_quantity(symbol, name, dimension, visited)` in `GraphIndex`:

1. Looks up `sym_to_eqs[symbol]` (all equations containing that letter).
2. **Filters by dimension** — keeps only equations where the variable
   for that symbol has a compatible dimension to what's needed.
   This is the *deterministic* layer: it removes representation errors
   (e.g. optics "m" = fringe order, dimensionless) before the LLM sees them.
3. Skips `visited_eqs` and conservation-law forms (`'constant'` variable).

Same-dimension-different-name cases (e.g. "radius" vs "separation
distance", both dimension L; or "I" which the graph stores with the
combined dimension "MLT-1 or ML2 or A") still pass through — these are
genuine conceptual choices and belong to the LLM.

---

## Keeping a single round's prompt within the model's token limit

In production, a real run hit a Groq 413 (`Request too large... limit
6000, requested 8202`) on `llama-3.1-8b-instant`. Root cause: after
picking F=m·a, the next round needs to resolve `m` and `a` together —
and showing every dimension-compatible candidate for both (24 + 12 in
this case, with full equation/description/conditions/variables metadata
each) measured at ~8,500 tokens by itself, before the system prompt or
question even get added. Three fixes, applied together:

1. **Domain filtering, with a hard fallback.** Every node already carries
   a `domain` field (`laws_of_motion`, `fluid_mechanics`, `electrostatics`,
   ... 26 total). Stage 1 now also outputs `likely_domains` — 1–3 domains
   it judges the question involves, at zero extra LLM calls since Stage 1
   already reads the question. `candidates_for_quantity()` takes this as
   an optional `allowed_domains` filter: if narrowing to those domains
   leaves at least one candidate, use the narrowed set; if it would leave
   zero (the domain guess missed), return the **full** set instead. That
   fallback is the load-bearing safety property — domain filtering can
   only reduce what's shown by default, never make something permanently
   unreachable, which is what kept this from reintroducing the original
   retrieval-can-silently-exclude-the-right-answer failure mode.
   Measured effect on the actual failing case: m+a's candidates drop from
   24+12 to 4+6, and the round's estimated cost drops from ~8,500 to
   ~1,900 tokens.
2. **Formatting trims, free on top of that.** Compact JSON instead of
   pretty-printed (saves ~33% of bytes for zero information loss), and
   each candidate's `variables` list drops symbols already shown in
   "ALREADY KNOWN" instead of repeating them. `conditions` was considered
   for trimming too but left at 2 entries — domain filtering already
   provides enough headroom that cutting it wasn't necessary, and it's a
   real (if probably small) signal for selection quality.
3. **A split-on-overflow safety valve, as a backstop, not the main fix.**
   `frontier_resolver` estimates a round's token cost before sending it
   (`llm_interface.estimate_round_tokens`); if it's still over
   `config.MAX_CANDIDATES_TOKENS_PER_ROUND` even after domain filtering,
   the round is sent as separate sequential single-symbol calls instead
   of one batched call. Costs a few extra calls only on the rare round
   that needs it — guarantees no future combination of busy domains can
   reproduce this crash.

---

## LLM call budget per question (typical 2–4-equation problem)

| Stage | Call | Model |
|-------|------|-------|
| 1 | parse (1×) | llama-3.1-8b-instant |
| 2 | round selector (1–3×, one per round) | llama-3.1-8b-instant |
| 4 | narrate (1×) | llama-3.3-70b-versatile |
| 5 | distractors (1×) | llama-3.1-8b-instant |

Total: 4–7 calls for a clean run. If Stage 3 fails and backtracking
fires, add one more Stage-2 round per retry attempt (bounded at 3 total
attempts) — so a worst-case run costs more, but only on the failure path,
not by default. Does **not** scale with graph size — only with the
conceptual depth (number of frontier rounds) of the specific problem.

---

## Implicit constants: universal vs. context-dependent

`config.py` splits `PHYSICAL_CONSTANTS` (never treated as a frontier
unknown — the resolver will never try to "solve for" one of these) into:

- **`UNIVERSAL_CONSTANTS`** — π, c, G, ε₀, μ₀, h, k_B, R, Nₐ, e. True
  constants of nature: same value regardless of scenario, so they're
  always available without needing Stage 1 to reason about them.
- **`g`** — deliberately excluded from the universal tier. Earth-surface
  gravitational acceleration is *not* a constant of nature (compare to G,
  which is); it's only relevant when the scenario is actually about
  gravity near Earth's surface. `g` enters `given` exactly the same way
  any other context-dependent implicit constant does — Stage 1 flags it
  from scenario cues ("free fall", "dropped", "projectile") — and never
  by default. An earlier version force-injected `g` for every question
  ("most JEE/NEET problems need it"); that's been removed, since it
  polluted Stage 2's context for unrelated domains and contradicted the
  brief's actual Stage-1-judgment design.

Stage 3's `CONSTANT_VALUES` (in `sympy_executor.py`) is a separate,
intentionally-unconditional safety net: it's built from the full catalog
including `g`, but it's never shown to any LLM — it just guarantees Stage
3 always has a numeric value ready if a *chosen* equation happens to need
one. No bias risk there; only the Stage-2-facing `given_full` needed the
universal/contextual split.

---

## Backtracking on Stage 3 failure

`pipeline.py` retries (up to `MAX_BACKTRACK_ATTEMPTS = 3` total attempts)
whenever Stage 3 fails — no real solution, dimension mismatch, ambiguous
roots — excluding the offending equation(s) and re-running Stage 2's LLM
call fresh. This triggers on **any** Stage 3 failure, not only ones the
LLM itself flagged with a `conditions_concern`: a confidently-wrong pick
that never raises a flag fails Stage 3 the same way a self-doubted one
does, and both need to retry. `decision_log` accumulates across every
attempt (tagged `attempt: N`) so an abandoned wrong pick stays visible
for narration instead of disappearing once corrected.

**What this can't catch:** backtracking only fires when Stage 3 produces
a detectable mathematical failure. A wrong equation that's still
dimensionally valid and numerically solvable — the original F=ma vs.
F=ρVg case — produces a real, correctly-dimensioned, *different* number.
SymPy has no way to know that's wrong; there is no deterministic check
for "mathematically fine, physically the wrong equation." For that
category, correctness depends entirely on Stage 2's conceptual judgment
being right the first time — backtracking is a second line of defense
for a specific failure mode, not a general-purpose correctness guarantee.

---

## Tested edge cases

| Case | Status |
|------|--------|
| F=ma vs F=ρVg (original failure) | ✓ mock-verified; live test ready |
| Symbol collision dimension filter | ✓ 14 deterministic tests pass |
| Exact fraction survival (a=45/4) | ✓ SymPy Rational preserved |
| Conservation-law equation excluded | ✓ |
| Topological sort (prereqs before dependents) | ✓ |
| Cycle detection → SimultaneousGroup | ✓ |
| Backtracking via excluded_eqs | ✓ |
| Simultaneous SymPy solve | ✓ |
| No API key → live tests gracefully skip | ✓ |
| Implicit-constant dict injection doesn't crash (was: `{{...}}` → unhashable set) | ✓ |
| `g` NOT force-injected for an unrelated domain (electrostatics) | ✓ |
| Universal constants (c, G, ε₀...) correctly exclude `g` | ✓ |
| Backtrack triggers on an unflagged (non-provisional) Stage 3 failure | ✓ |
| Domain filter narrows candidates correctly | ✓ |
| Domain filter falls back to full set when narrowing would empty it | ✓ |
| Real production overflow scenario (m+a, ~8500 tok) fixed by domain filter | ✓ |
| Oversized round splits into sequential single-symbol calls | ✓ |

---

## Known limitation: graph text is templated, not hand-written per equation

`rag_text` and `conditions` give the Stage 2 LLM most of its disambiguating
signal beyond the raw variable list. Auditing the highest-collision-risk
equations directly (force/mass/energy clusters) found real templating: the
sentence *"The key assumption is that all quantities belong to the same
body or interaction. Do not use it when data from different bodies,
intervals, or directions are mixed."* appears verbatim across F=ma, F=mg,
and Newton's law of gravitation — three different physical setups sharing
one generic caveat. `fluid_mechanics_hydrostatic_pressure` and
`fluid_mechanics_buoyant_force` share an identical opening sentence too,
and `conditions` attaches "flow is steady for flow equations" to buoyant
force, which isn't really a flow equation.

The specific signal — variable names/units, domain category — is accurate
and not templated, and is probably enough for an LLM to disambiguate most
cases correctly when combined with the question text and the dimension
filter already applied upstream. But the "do not use it when..." guidance
is generic per-category rather than per-equation, which is weakest exactly
where it matters most: genuinely borderline cases in the same domain.
This wasn't fixed in this pass — it's a data-authoring task, not a code
bug, and the scope (rewrite all 182, or just the ~30–40 highest-risk
force/mass/energy equations) is a real cost/benefit call rather than
something to decide unilaterally.

