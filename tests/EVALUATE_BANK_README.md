# Evaluation Harness — `tests/evaluate_bank.py`

Runs a JSON question bank through the solver and reports WHERE failures happen,
grouped by domain, reasoning_type, and failure stage. This is how you stop
fixing individual questions and start fixing failure categories.

## Quick start

```bash
# 1. Put your generated bank here (one file, or many you concatenate):
#    questions/question_bank.json
#    (or questions/batch_kinematics.json, batch_laws_of_motion.json, ...)

# 2. Make sure your .env points where you want (local Ollama or Groq).
#    For a 100-question run, use local — no token-per-minute limit.

# 3. Run it:
python tests/evaluate_bank.py questions/question_bank.json

# Or run several batch files at once:
python tests/evaluate_bank.py questions/batch_*.json

# Smoke-test on the first 10:
python tests/evaluate_bank.py questions/question_bank.json --limit 10

# Save a detailed per-question JSON report:
python tests/evaluate_bank.py questions/question_bank.json --report results.json
```

## What you get

A console report with four sections:

1. **By reasoning type** (worst first) — does multi_step_chain fail more than
   single_step? This tests the core hypothesis that failures cluster by
   STRUCTURE, not domain.
2. **By domain** (worst first) — which physics areas are weak.
3. **By difficulty** — do failures track difficulty or structure?
4. **Failures by stage** + **cross-tab** — the actionable part. For every
   failure it attributes a STAGE:
   - `stage1_parse` — parser dropped givens or mis-typed the unknown
   - `retrieval` — search didn't surface a usable equation
   - `stage2_selection` — the LLM omitted/none'd/invalid'd the pick
   - `execution` — SymPy couldn't compute despite a chain
   - `answer_mismatch` — produced a number, but wrong
   - `hallucinated_answer` — fabricated an answer to an underspecified question
   - `rate_limit` — provider rate limit interrupted (infra, not logic)

The cross-tab (reasoning_type × failure_stage) is the key diagnostic: it shows
which failure stage hits which reasoning type, so you know exactly what to fix.

## How failure-stage attribution works

After each `solver.solve()`, the harness reads the slice of `logs/solver.log`
that solve produced and inspects the events (`stage1_parsed`, `retrieval_search`,
`stage2_item_omitted`, `stage2_item_decision` with fallback codes,
`stage2_hallucinated_symbol`, `is_rate_limit`). It attributes the failure to the
earliest stage that broke. This is approximate but reliable enough to reveal
clustering — which is what matters for deciding where to invest fixing effort.

## Bank format

See `QUESTION_BANK_PROMPT.md`. Each question needs at least: `id`, `domain`,
`reasoning_type`, `question`, `expected_answer` (null for underspecified),
`answer_tolerance_pct`. The harness tolerates a `{"questions": [...]}` wrapper
or a bare JSON array.

## Underspecified questions

Questions with `reasoning_type: "underspecified"` and `expected_answer: null`
PASS only if the solver refuses (UNVERIFIED confidence or no value). If the
solver fabricates an answer, it's marked `hallucinated_answer` — directly
testing the anti-hallucination guard across many phrasings.

## Reading the result

- Failures concentrated in ONE stage across many domains → systematic bug, fix
  once.
- `stage1_parse` heavy → prompt-level parser fix.
- `stage2_selection` heavy on chains → the concept-aware variable-mapping work.
- `retrieval` heavy → tune rag_texts for those equations, or re-ingest
  (`scripts/verify_rag.py --check-sync`).
- `answer_mismatch` heavy → execution/units; check the v7.3 graph-content quirks.
- Any `hallucinated_answer` → Stage 1 anti-hallucination prompt needs that
  phrasing.
```
