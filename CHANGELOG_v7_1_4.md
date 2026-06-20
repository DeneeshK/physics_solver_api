# Changelog v7.1.4 — Compact candidate format + knowns-overlap ranking + derivability principle

v7.1.4 implements the architecture you described: strip the full rag_text
out of Stage 2 prompts, keep only the short concept name + equation +
variable meanings, and explicitly tell the LLM to think in terms of
**derivability through chains** rather than direct symbol matching.

## What changed and why

**1. Compact Stage 2 candidate format**

Before (v7.1.3):
```json
{
  "id": "laws_of_motion_newton_second_law",
  "equation": "F = m*a",
  "description": "<full 600-char rag_text>",
  "conditions": [...],
  "variables": {...}
}
```

After (v7.1.4):
```json
{
  "id": "laws_of_motion_newton_second_law",
  "equation": "F = m*a",
  "concept": "Newton's Second Law of Motion",
  "variables": {...},
  "conditions": [...]   // only when non-empty
}
```

For a round with 5 candidates per item × 2 items, that's ~6000 chars of
rag_text removed from each Stage 2 prompt. The 8B model now sees a clean,
focused prompt and uses its own physics knowledge for nuance — which it has,
because Newton's Second Law and Coulomb's Law are in its training data.

The "concept" field is the equation's UNIQUE PHYSICS IDENTIFIER ("Newton's
Second Law of Motion", "Archimedes' Principle", "Time-Free Kinematic
Relation"). One short label per equation, written by hand for all 182
equations during the v7.1.1 corpus build. The candidate format pulls it
from `node["concept_name"]` if `apply_exemplars_only.py` wrote it, else
extracts it from the rag_text head as a fallback.

**2. Derivability principle in Stage 2 prompt**

The new ROUND_SELECT_SYSTEM prompt explicitly tells the LLM:

> DO NOT reject an equation because its variables don't appear in the
> question's given values. The pipeline solves problems in CHAINS. An
> equation is the right pick if its needed variables can be derived from
> what's known — directly or through other equations in subsequent rounds.

With worked examples:
- F = m*a is the right pick even when ρ, V are given (not m): m derivable from ρ×V
- K = ½mv² is the right pick even when v is not given: v derivable from v = u + a*t

This addresses the failure mode you identified — chained problems failing
not because retrieval missed the right equation, but because the LLM was
rejecting it for "variables not in the question." The prompt now names
this anti-pattern and tells the LLM to think about derivability instead.

**3. Knowns-overlap ranking in Round 1+**

When Round 1 looks for `m` and `ρ=m/V` exists in the candidates alongside
`p = m*v` and `K = ½mv²`, the bridging equation gets ranked first by
overlap with `known_symbols`.

Implementation: when `round_num > 0` and `known_symbols` is non-empty,
`get_landing_candidates` re-sorts candidates by:
- Number of overlapping variables with `known_symbols` (descending)
- Stable secondary order (preserves the original retrieval order as a
  tie-breaker)

Round 0 keeps the original ranking — there, the concept (not overlap) is
what determines fit. Applying overlap ranking in Round 0 would push e.g.
F = ρVg above F = ma in a kinematics problem just because ρ, V happened
to be given for mass derivation downstream.

Round 1+ also now goes through `get_landing_candidates` (was using raw
`candidates_for_quantity` before). The unified path makes the overlap
ranking apply uniformly.

## What's NOT in v7.1.4

You asked about an agentic retry loop within selection — pick, try, fail,
reject, try next. I'm deferring that to v7.1.5 if needed. Reasoning:

- Each retry costs a Groq call and burns TPM
- We already have backtracking at the **execution** level — if SymPy can't
  resolve a chain because we picked the wrong equation in Round N, the
  pipeline backtracks and tries again with the failed equation excluded
- v7.1.4's compact format + derivability principle should let the LLM
  pick correctly on the first attempt for the test cases that were failing

If v7.1.4 still has failures after live tests, I'll build the in-selection
retry loop as v7.1.5. Diagnose first, then add complexity if warranted.

## Deployment

```bash
tar -xzf physics_solver_v7_1_4_tar.gz
cd physics_solver_v7
pip install -r requirements.txt

# 1. Apply exemplars — v7.1.4 also writes concept_name to each node
python scripts/apply_exemplars_only.py

# 2. Verify deterministic tests
python tests/test_deterministic.py             # expect: 46 passed

# 3. ChromaDB does NOT need rebuilding (no rag_text changes — embeddings
#    are still based on the same text the LLM was seeing before; just
#    the LLM's prompt is what changed)

# 4. Live tests — both failing cases (F=ma chained, KE chained) should
#    now pass
python tests/test_live.py

# 5. If anything still fails, diagnose:
python scripts/diagnose_question.py "<the failing question>"
# The log will show the new compact prompt structure under
# stage2_round_entry. Check that "concept" is set on each candidate.
```

If F=ma-with-density-volume still fails: send me the
`stage2_item_decision` log line — I want to see what the LLM said in
the `reason` field. The diagnostic should be richer now that the prompt
is cleaner.

## Files changed in v7.1.4

  - `solver/llm_interface.py` — `_format_candidate` outputs `concept`
    not `description`; added `_extract_concept` helper for fallback;
    `ROUND_SELECT_SYSTEM` rewritten with derivability principle and
    concept-based decision rules
  - `solver/landing.py` — `get_landing_candidates` accepts
    `known_symbols` and `round_num`; re-ranks by overlap when
    `round_num > 0`
  - `solver/frontier_resolver.py` — passes `known_symbols`+`round_num`
    to landing in both Round 0 and Round 1+ branches; Round 1+ now also
    goes through `get_landing_candidates` (no semantic step in Round 1+)
  - `scripts/apply_exemplars_only.py` — writes `concept_name` to each
    node alongside `rag_text`
  - `tests/test_deterministic.py` — replaces the v7.1
    "no_longer_truncates_rag_text" test with four new tests covering
    the compact format, concept extraction, overlap ranking, and
    derivability prompt
  - `CHANGELOG_v7_1_4.md` — this file
