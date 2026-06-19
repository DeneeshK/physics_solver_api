# Changelog

## v7.0 — Foundation + ChromaDB landing scaffold (current)

This is the first of a planned multi-step refactor against the v6 audit.
v7.0 is the **foundation** pass: every change is deterministic-testable
without a Groq API key, and every v6 deterministic test continues to pass.

### Changes

**Constants reconciliation (data + config)**

- Graph: renamed `epsilon0` → `epsilon_0`, `mu0` → `mu_0` across all 8
  affected nodes (variable keys, `equation_str`, `sympy_expr`, `latex`, and
  edge `shared_variables`). The graph and `config.PHYSICAL_CONSTANTS` are
  now consistent on these symbols.
- `config.PHYSICAL_CONSTANTS` rewritten to include the actual symbols
  the graph uses (both underscored and underscoreless forms as belt-and-
  suspenders). Adds `UNIVERSAL_CONSTANTS` for symbols safe to treat as
  context-independent universal constants — used in pipeline's
  context-aware constant injection.
- Why it matters: v6 silently treated `epsilon_0` as an unknown in any
  electrostatics question because the graph's `epsilon0` (no underscore)
  did not match the config's `epsilon_0` (with underscore). The pipeline
  would either chase it as a missing variable (cascading), or fail to
  substitute its numeric value in SymPy.

**Dimension normalizer fix (`solver/graph_loader.py`)**

- The v6 regex `[MLTAK](-?\d*)` only handled M, L, T, A, K — silently
  mangling `Theta` (temperature) and `N` (moles), and decaying `varies`
  to `('A', 1)`.
- v7 tokenizer recognizes `Theta`, `K`, `mol`, `N`, plus the MLTAK set.
  `Theta` canonicalizes to the K bucket; `mol` canonicalizes to N.
  `varies` returns a synthetic `('__VARIES__', 1)` sentinel that never
  matches any real dimension.
- Why it matters: any equation involving the universal gas constant, gas
  laws, thermal-physics relations, or molar quantities was using
  dimensions the normalizer silently scrambled. With v6 those equations
  worked by accident (both sides of the comparison broke the same way);
  with v7 they work because they're parsed correctly.

**ChromaDB landing — restored as additive layer**

- `solver/retrieval.py`: fixed missing `BGE_QUERY_PREFIX` import (the
  v6 module couldn't even be imported), added `Retriever.try_load()`
  classmethod for graceful absence of the index.
- New `solver/landing.py`: unifies symbol-table lookup with optional
  ChromaDB semantic lookup. Returns a candidate list with each
  candidate tagged `landing_source` ∈ {`symbol`, `semantic`, `both`}.
- `solver/frontier_resolver.py`: round 0 now uses the landing layer.
  Rounds 1+ keep symbol-only behavior (we're chasing specific variables
  in already-chosen equations).
- **Safety property**: when no retriever is loaded, the landing layer
  returns exactly the symbol-table candidates v6 would have shown.
  Enabling Chroma can only *add* candidates the LLM sees; it never
  removes one. This is why no v6-passing question can regress in v7.0
  with Chroma off.
- Feature-flagged via `ENABLE_CHROMA_LANDING` env var:
  `auto` (default), `true`, or `false`.

**Stage 1 prompt — search_query field and softer symbol rules**

- New required `search_query` field: a single-sentence scenario
  description for ChromaDB retrieval. Question-relevant nouns and
  category, no numbers.
- Removed the rigid "h=height" rule. Replaced with context-aware
  guidance: in a motion question, a height through which something
  moves IS a displacement (`s`), not `h`. Reserve `h` for static
  configurations (gravitational PE in isolation, hydrostatic depth,
  fluid columns). This is one of the two root causes of the v6
  free-fall failure.
- Added `N` (amount of substance) to the dimension formula examples.

**Stage 2 prompt — landing_source awareness, explicit no-pick**

- Each candidate's `landing_source` is surfaced to the LLM so it can
  read context — "both" = strongest signal — without it being a
  forced ranking criterion (the physics still decides).
- New `decision: "none"` option: if no candidate physically fits, the
  LLM can say so. v6 forced a pick regardless; v7 surfaces this as a
  no-pick that the resolver records cleanly.

**Silent fallback elimination (`solver/llm_interface.py`)**

- v6's `call_round_selector` had two silent fallbacks:
  1. LLM returns an equation ID not in the candidate set → silently
     substitute the first candidate.
  2. LLM omits a frontier item entirely → silently pick the first
     candidate.
- Both are removed. Both now surface as `chosen_eq=None` with a
  diagnostic `fallback_used` field (`llm_invalid_id`, `llm_omitted_item`,
  or `llm_decision_none`).
- The frontier resolver records the diagnostic, returns a clean
  `UNVERIFIED` result with a descriptive `failure_reason`, and the
  pipeline's existing backtracking can react.

**Decision log — candidates_shown**

- Each `decision_log` entry now includes:
  - `candidates_shown`: the full candidate list the LLM saw (id +
    equation_str + landing_source per item)
  - `fallback_used`: any diagnostic from the LLM call
  - `decision`: `pick` / `defer` / `none`
- Stage 4 narration will be updated in v7.2 to honestly cite rejected
  alternatives using this data.

### Files changed
- `config.py` — rewritten (constant reconciliation, ChromaDB flag)
- `data/physics_equation_graph_final.json` — `epsilon0` → `epsilon_0`,
  `mu0` → `mu_0` in 8 nodes
- `solver/graph_loader.py` — new dimension tokenizer
- `solver/retrieval.py` — fixed import, added `try_load`
- `solver/landing.py` — NEW
- `solver/llm_interface.py` — Stage 1 + Stage 2 prompts, post-LLM logic
- `solver/frontier_resolver.py` — search_query/retriever params, log update
- `solver/pipeline.py` — Retriever init, search_query plumbing
- `tests/test_deterministic.py` — 13 new tests for v7 behavior

### Deployment

**To enable ChromaDB landing** (recommended):

```bash
# 1. Install dependencies (no change from v6)
pip install -r requirements.txt

# 2. Build the index — ~10 minutes, downloads ~1.3 GB BGE model on first run
python -m solver.ingest

# 3. Start server. ChromaDB landing auto-enables when the index exists.
python -m uvicorn api.main:app --port 8000
```

**To force v6 behavior** (symbol-only landing, no Chroma):

```bash
export ENABLE_CHROMA_LANDING=false
python -m uvicorn api.main:app --port 8000
```

### What v7.0 does NOT yet fix

- **rag_text quality.** The graph's rag_texts are still templated
  boilerplate (one shared opening per domain, generic variable-by-variable
  enumeration). ChromaDB embedding is over this content, so retrieval
  quality is bounded by that. Fixing this is v7.1 — a batch-generator
  script that rewrites all 182 rag_texts with proper equation-specific
  physics descriptions using your Groq key.
- **Stage 4 narration** doesn't yet *use* `candidates_shown` — v7.2.
- **`SIGNED_SYMBOLS` and per-equation variable name accuracy** — v7.3.

### Verifying

Run the deterministic test suite (no API key required):

```bash
export ENABLE_CHROMA_LANDING=false
python tests/test_deterministic.py
```

Expected: `35 passed, 0 failed (35 total)`.

### Live test verification (requires your Groq key)

I do not have your Groq key in my sandbox, so I cannot run the live
stress tests myself. After deploying v7.0:

```bash
export GROQ_API_KEY=your_key
export ENABLE_CHROMA_LANDING=auto   # or 'false' for parity with v6
python tests/test_live.py
```

Compare results against your v6 baseline. With Chroma OFF, v7.0 should
match v6 on every passing case (the landing layer falls through to
symbol-only). With Chroma ON, the power-dissipation case should
continue to pass (it did in v6 after the dimension normalizer was added),
and the free-fall case may or may not start passing — that depends on
whether the templated rag_texts give the LLM enough signal to pick the
right kinematic equation. That's the v7.1 job.

If you see any v6-passing question regress on v7.0 with Chroma OFF,
that's a bug in this delivery — please tell me before I move to v7.1.
