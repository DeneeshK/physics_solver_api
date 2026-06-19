# Changelog v7.1 — Concept-level rag_text rewrite

v7.1 builds on v7.0 (which restored ChromaDB landing as scaffolding) by
giving that scaffolding actual content to retrieve over: rag_texts that
are concept-level identifiers, not domain boilerplate.

## The principle

Per the user's design decision: **the rag_text is the unique conceptual
identifier for an equation node**. Each equation represents a distinct
physics concept (Newton's Second Law, Archimedes' Principle, Coulomb's
Law, Time-Free Kinematic Relation, Photon Energy, etc.) and the rag_text
must express that concept clearly enough that:

  1. ChromaDB embedding over the rag_text creates a concept-distinguishing
     vector — not a domain-clustered, near-identical one.
  2. An LLM reading the rag_text alongside the question can make a
     **physics judgment** about applicability — even when the equation's
     symbols don't literally appear in the question.

The matching happens **concept-to-concept on both sides**. Stage 1's
`search_query` is also lifted to the concept level — "Newton's second
law applied to find net force on a body undergoing acceleration", not
"find force given density volume velocity displacement". This kills
the F=ρgh-wins-over-F=ma failure mode at its root: surface symbol
overlap can no longer beat conceptual fit, because retrieval and the
LLM's pick both operate above the symbol layer.

## What v7.1 ships

### 16 hand-authored exemplar rag_texts (`scripts/rag_text_exemplars.json`)

Each ~600–1100 chars, written to the principle. Coverage spans the
highest-collision domains in your audit:

  - `laws_of_motion_newton_second_law` — Newton's Second Law of Motion
  - `fluid_mechanics_buoyant_force` — Archimedes' Principle (Buoyant Force)
  - `fluid_mechanics_hydrostatic_pressure` — Hydrostatic Pressure at Depth
  - `kinematics_v2_u2_2as` — Time-Free Kinematic Relation
  - `kinematics_v_u_at` — Time-Velocity Kinematic Relation
  - `electrostatics_coulomb_law` — Coulomb's Law
  - `gravitation_universal_law` — Newton's Law of Universal Gravitation
  - `current_electricity_power_electric` — Electrical Power (Volt-Ampere)
  - `current_electricity_ohms_law` — Ohm's Law
  - `modern_physics_photon_energy` — Photon Energy (Planck-Einstein)
  - `work_energy_power_gravitational_potential_mgh` — Gravitational PE
  - `work_energy_power_kinetic_energy` — Kinetic Energy of Translation
  - `magnetism_lorentz_force` — Lorentz Magnetic Force
  - `circular_motion_centripetal_force` — Centripetal Force
  - `shm_period_spring` — SHM Period of Spring-Mass Oscillator
  - `ray_optics_lens_formula` — Thin Lens Formula

Properties enforced by the deterministic tests (5 new in v7.1):
  - Each opens with a named physics concept (no v6 "Use it when..." boilerplate).
  - Each contains explicit concept-distinction language against look-alikes.
  - All 16 concept names are unique (no two exemplars share an identity).

### Apply-only script (`scripts/apply_exemplars_only.py`)

Applies the 16 exemplars to the graph immediately. No LLM call, no Groq
key required. Use this to get gold-standard exemplars into the graph as
the first step:

```bash
python scripts/apply_exemplars_only.py             # apply
python scripts/apply_exemplars_only.py --dry-run   # show diff first
```

The script backs up the original graph to `data/physics_equation_graph_final.json.bak`
unless you pass `--no-backup`.

### Batch generator (`scripts/regenerate_rag_texts.py`)

For the remaining ~166 equations, this script uses your Groq API key to
generate concept-level rag_texts in the same style as the exemplars.

How it works:
  - Loads the 16 hand-authored exemplars.
  - For each non-exemplar equation: builds a prompt with 3 random
    exemplars as in-context demonstrations, the equation's structured
    data (id, domain, subdomain, equation_str, variables, conditions,
    common_mistakes, jee_chapters), and the system prompt that defines
    the concept-level style.
  - Calls `llama-3.3-70b-versatile` (Stage-4-grade model — quality over
    cost on this one-shot per-equation generation).
  - Validates the output against:
      - length 400–1500 chars,
      - no banned phrases ("this equation", "Use it when force, mass,
        weight, or contact between bodies controls the motion", etc. —
        these are old v6 templating tells and generic LLM filler),
      - parses as JSON.
  - One retry per equation if validation fails. Failures are reported
    at the end, NOT silently skipped — those equations keep their
    existing rag_text and you decide whether to hand-author them.
  - Backs up the original graph before writing.

Cost estimate: 166 calls × ~1500 output tokens × $0.0008/1k = roughly
$0.20 per full regeneration run. Time: ~5 minutes including rate limits.

Usage:

```bash
export GROQ_API_KEY=your_key

# Dry-run on 5 equations to validate output quality before committing
python scripts/regenerate_rag_texts.py --dry-run 5

# Real run on everything not in exemplars
python scripts/regenerate_rag_texts.py

# Re-run just specific failed equations
python scripts/regenerate_rag_texts.py --ids electrostatics_xyz,wave_optics_abc
```

### Stage 1 prompt — concept-level search_query

The `search_query` field in Stage 1's output is now explicitly required
to be at the physics-concept level, with worked examples of GOOD
(concept-named) and BAD (keyword-listed) queries inline in the prompt.

This is the matching counterpart to the concept-level rag_text. Both
sides operate above the symbol layer; ChromaDB matches concept-to-concept.

### Stage 2 — 120-char rag_text truncation removed

In v6, `_format_candidate` truncated each candidate's `rag_text` to 120
characters before showing it to the LLM. That cap was appropriate when
rag_texts were templated boilerplate (the first sentence was the same
across all equations in a domain — truncating threw away no
disambiguation value). With concept-level rag_texts (~600–1100 chars),
the full text is what the LLM needs to judge fit. The cap is gone in
v7.1.

## Required follow-up actions

After applying v7.1:

```bash
# 1. Apply the exemplars
python scripts/apply_exemplars_only.py

# 2. Verify deterministic tests still pass (no Groq key needed)
python tests/test_deterministic.py        # expect: 40 passed, 0 failed

# 3. Rebuild ChromaDB index over the new rag_texts
#    (the embeddings are text-derived — they need to be regenerated
#    whenever rag_text changes)
python -m solver.ingest

# 4. (optional, recommended) Run the batch generator on the other ~166
#    equations
python scripts/regenerate_rag_texts.py --dry-run 5  # spot-check first
python scripts/regenerate_rag_texts.py               # full run

# 5. After step 4, rebuild ChromaDB AGAIN over the now-fully-regenerated graph
python -m solver.ingest

# 6. Run your live stress tests
python tests/test_live.py
```

## Files added in v7.1

- `scripts/rag_text_exemplars.json` — the 16 hand-authored exemplars
- `scripts/apply_exemplars_only.py` — apply exemplars (no LLM)
- `scripts/regenerate_rag_texts.py` — batch generator (uses Groq)
- `CHANGELOG_v7_1.md` — this file

## Files changed in v7.1

- `solver/llm_interface.py` — Stage 1 search_query guidance,
  Stage 2 rag_text truncation removed
- `data/physics_equation_graph_final.json` — 16 rag_texts now hold the
  hand-authored exemplars (other ~166 still hold the v6 templated text
  until you run the batch generator)
- `tests/test_deterministic.py` — 5 new tests

## What v7.1 does NOT yet do (planned for later)

- The remaining ~166 equations still have v6 templated rag_texts in the
  shipped graph file. The batch generator regenerates them, but I can't
  run it here without your Groq key. After your batch run, you'll have
  the full v7.1 corpus.
- Stage 4 narration is not yet updated to use `candidates_shown` from
  the decision log (planned for v7.2).
- The `SIGNED_SYMBOLS` hardcoded set and per-equation variable-name
  templating ("power or pressure" combined names) still exist (v7.3).
- The `momentum_collisions_conservation_two_body` equation has its
  variable scheme wrong (`m*u + M*v = m*a + M*s` mis-uses kinematic
  symbols as second-body velocities). Flagging for graph-content
  cleanup separately.
