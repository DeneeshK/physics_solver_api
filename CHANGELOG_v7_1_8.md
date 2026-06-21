# Changelog v7.1.8 — Evaluation harness (failure-surface analysis)

v7.1.8 adds `tests/evaluate_bank.py`, a harness that runs a JSON question
bank through the solver and reports WHERE failures happen — grouped by
domain, reasoning_type, and failure stage. This is the tool that converts
"fix five questions" into "fix the systematic failure modes."

## Why this exists

Across v7.1.1 → v7.1.6 we fixed five live-test questions one at a time. That
is the "modeling for five questions" trap: a system tuned to pass exactly the
tests we've seen, with no signal on whether it generalizes. The only way out
is to run many questions (50-100+) across domains and SEE the failure surface.

The local-LLM migration (v7.1.7) removed the Groq token-per-minute ceiling
that made bulk runs impractical. v7.1.8 is the harness that uses that
capability.

## What it does

  - Reads `questions/question_bank.json` (or any bank file(s)/glob), format
    per `QUESTION_BANK_PROMPT.md`.
  - Runs each question through the same `PhysicsSolver` the live tests use —
    so whichever provider .env selects (local or Groq) is what runs.
  - Compares answers within each question's `answer_tolerance_pct`.
  - For `underspecified` questions (expected_answer null), PASSES only if the
    solver refuses (UNVERIFIED / no value) — directly testing the
    anti-hallucination guard across many phrasings.
  - After each solve, reads that solve's slice of `logs/solver.log` and
    attributes any failure to a STAGE: stage1_parse, retrieval,
    stage2_selection, execution, answer_mismatch, hallucinated_answer, or
    rate_limit.

## The report

Four sections:
  1. By reasoning type (worst first) — tests the "failures cluster by
     structure not domain" hypothesis.
  2. By domain (worst first).
  3. By difficulty.
  4. Failures by stage + a reasoning_type × failure_stage CROSS-TAB — the
     actionable diagnostic showing which stage breaks which reasoning type.

Plus interpretation hints mapping each dominant failure stage to the fix it
implies.

## How failure-stage attribution works

The harness inspects the log events each solve emits (`stage1_parsed` with
n_given, `retrieval_search`, `stage2_item_omitted`, `stage2_item_decision`
fallback codes, `stage2_hallucinated_symbol`, `is_rate_limit`) and attributes
to the earliest stage that broke. Approximate but reliable enough to reveal
clustering — which is the whole point.

## Usage

```bash
python tests/evaluate_bank.py questions/question_bank.json
python tests/evaluate_bank.py questions/batch_*.json
python tests/evaluate_bank.py questions/question_bank.json --limit 10
python tests/evaluate_bank.py questions/question_bank.json --report results.json
```

Exit code 0 if all pass, else 1 (for scripting/CI).

## Files added in v7.1.8

  - `tests/evaluate_bank.py` — the harness
  - `tests/EVALUATE_BANK_README.md` — usage and interpretation guide
  - `questions/` — directory for bank files (create your bank here)
  - `CHANGELOG_v7_1_8.md` — this file

## No code changes to the solver

v7.1.8 is purely additive. The deterministic suite is unchanged (51 passed).
The harness only reads the solver's public response object and its log — it
does not modify any pipeline behavior.

## The intended workflow from here

  1. Generate the bank (QUESTION_BANK_PROMPT.md in a fresh chat), save to
     questions/.
  2. Run evaluate_bank.py.
  3. Read the cross-tab. Identify the 2-3 dominant failure stages.
  4. Fix by CATEGORY:
     - stage1_parse heavy → parser prompt fix
     - stage2_selection heavy on chains → the concept-aware variable-mapping
       work (the big change discussed but deliberately deferred until
       measured)
     - retrieval heavy → rag_text tuning / re-ingest
     - answer_mismatch heavy → v7.3 graph-content quirks
  5. Re-run the bank. MEASURE whether the category fix moved the needle across
     all questions — not just the five we started with.
