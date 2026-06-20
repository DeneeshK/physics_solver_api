# Changelog v7.1.2 — Observability, RAG diagnostics, Stage 2 model switch

v7.1.2 addresses the question that came up after running v7.1.1 live tests:
"I can see two tests fail with `llm_omitted_item`, but I can't tell WHY.
Is my RAG even working? Is the rate limit being hit? Where's the failure?"

The answer is logging, diagnostic tooling, and an escape hatch for the
most likely root cause.

## What was actually wrong (hypothesis to verify with the new logging)

Both failing tests in the v7.1.1 live run share the same diagnostic:

  - `F=ma (NOT F=ρVg)` — original failing case — `LLM rejected all 4
    candidate(s) for 'm' in round 1 (llm_omitted_item)`
  - `Kinetic energy (compute v first from kinematics)` — `LLM rejected all
    6 candidate(s) for 'v' in round 1 (llm_omitted_item)`

`llm_omitted_item` is the v7 honest-failure code: it means Stage 2's LLM
returned valid JSON, but the JSON didn't include a selection for the
symbol we asked about. In v6 this would have been silently masked by a
"pick the first candidate" fallback — producing a confident wrong answer.
v7's design intentionally surfaces it.

Both failures involve **agentic chains** — the LLM has to recognize:
  - "I need m, density+volume are given, so rho=m/V is the bridge"
  - "I need v, initial velocity+acceleration+time are given, so v=u+a*t"

This is reasoning, not pattern-matching. **Stage 2 uses MODEL_FAST
(llama-3.1-8b-instant)**. The 8B model is fine at lookup-style matching
against templated rag_texts (v6's regime), but the v7.1's ~600-char
concept-level rag_texts demand more reasoning capacity than 8B reliably
delivers — especially when the question requires chaining multiple
equations.

v7.1.2 doesn't force a model change. It makes the model switchable via
env var and adds the logging needed to confirm or reject this hypothesis.

## Structured logging end-to-end

New: `solver/solver_log.py`. Every meaningful event in the solver writes
one JSON object per line to `logs/solver.log`. The format is
one-event-per-line so you can:

```bash
tail -f logs/solver.log                                  # live stream
grep '"event":"stage2_item_omitted"' logs/solver.log     # find omissions
jq 'select(.fallback_used)' logs/solver.log              # find every fallback
jq 'select(.is_rate_limit)' logs/solver.log              # find every rate limit
```

What gets logged automatically (no code change needed by you):

  - **`llm_request`** / **`llm_response`** — every Groq call, with model,
    stage, latency, prompt/response previews, and token usage.
  - **`llm_error`** — every exception from Groq, with `is_rate_limit`
    pre-flagged so you see TPM caps at a glance.
  - **`stage1_entry`** / **`stage1_parsed`** — what Stage 1 received and
    what it produced (given, unknown, implicit constants, search_query).
  - **`retrieval_search`** — every ChromaDB hybrid search, with the
    query, top-k IDs, and the semantic/BM25 score breakdown. This is the
    log line to inspect when answering "did retrieval find the right
    equation?".
  - **`stage2_round_entry`** — every Stage 2 round: what symbols are
    being requested, what candidates were given, and which landing route
    surfaced each candidate (`symbol`, `semantic`, or `both`).
  - **`stage2_llm_selections_received`** — what symbols the LLM
    addressed vs. what we asked about. Catches the `llm_omitted_item`
    case at the source.
  - **`stage2_item_decision`** — per-item outcome: pick / defer / none /
    invalid_id, with the LLM's stated reason.
  - **`stage2_item_omitted`** — the LLM didn't address this item at all.
  - **`solve_entry`** / **`solve_attempt`** / **`solve_success`** /
    **`solve_error`** — pipeline-level lifecycle.

Modes:
  - `SOLVER_LOG=off` — ERROR-level only (minimal log)
  - default       — full structured logging, prompt/response truncated
                    at 4000 chars per line
  - `SOLVER_LOG_VERBOSE=1` — no truncation; full prompts and responses
                             captured (large logs, useful for deep debug)

Rotates at 10 MB, keeps last 5 files. No external dependencies — pure
stdlib `logging`.

## Diagnostic scripts

**`scripts/verify_rag.py`** — answers "is my RAG actually working?"

```bash
# 1. Verify ChromaDB is in sync with the graph file (catches stale-embedding bugs)
python scripts/verify_rag.py --check-sync

# 2. Run a retrieval query and see what comes back
python scripts/verify_rag.py --query "Newton's second law for a body undergoing acceleration"

# 3. For a given equation, see its nearest semantic neighbors
#    (i.e. what the LLM is most likely to confuse it with)
python scripts/verify_rag.py --neighbors laws_of_motion_newton_second_law

# 4. Run all checks with canned stress-test queries
python scripts/verify_rag.py --all
```

The `--check-sync` mode is the one to run first. If you applied new
rag_texts but didn't re-ingest, ChromaDB has STALE embeddings and is
retrieving against text the LLM never sees. The script catches that and
tells you to run `python -m solver.ingest --reingest`.

**`scripts/diagnose_question.py`** — runs one question with full
verbosity, then prints a human-readable per-stage breakdown.

```bash
python scripts/diagnose_question.py "A body of density 8000 kg/m^3 and volume 0.5 m^3 starts from rest and reaches 30 m/s over 40 m. Find the net force."
```

Output is structured by stage:
  - Stage 1: parse output (given, unknown, search_query)
  - Retrieval: top-k results for each search, with scores
  - Stage 2: per round, what was asked, what each candidate was, what
    the LLM picked or rejected and WHY
  - Errors: any exceptions or rate-limit hits
  - Final: success/error, chain summary

Truncates the log file at start, so each diagnose run is isolated. Use
`--keep-log` if you want to preserve previous runs.

## The Stage 2 escape hatch: `STAGE2_MODEL` env var

`config.STAGE2_MODEL` defaults to `MODEL_FAST` (llama-3.1-8b-instant)
but is now overridable:

```ini
# .env
STAGE2_MODEL=llama-3.3-70b-versatile
```

When to flip it:

  - You run `scripts/diagnose_question.py` and see
    `stage2_item_omitted` or `stage2_item_decision` with
    `fallback_used="llm_invalid_id"` events, AND
  - Inspection of the candidate list shows the right equation WAS
    present (e.g. `general_density_definition` for the F=ma case), AND
  - The semantic retrieval log shows the equation made it into the
    top-k.

In that scenario the model isn't being failed by retrieval or by
prompt — it's failing on reasoning. Switching to the 70B model addresses
exactly this.

Trade-off: ~5–10x slower per question, ~5x cost per call. But the v7.1
design (concept-level rag_texts, agentic chains, no pattern-matching
escape) is fundamentally incompatible with the 8B model's reasoning
ceiling. The v6 architecture was implicitly engineered around the 8B
model's strengths; v7.1+ deliberately demands more.

## Files added in v7.1.2

  - `solver/solver_log.py` — structured logger
  - `scripts/verify_rag.py` — RAG sync + retrieval diagnostics
  - `scripts/diagnose_question.py` — per-question full-trace runner
  - `CHANGELOG_v7_1_2.md` — this file
  - `logs/` directory created automatically on first run

## Files changed in v7.1.2

  - `solver/llm_interface.py` — every `_call` instrumented with
    request/response logging including stage label, latency, rate-limit
    detection. Stage 2 now uses `STAGE2_MODEL` (defaults to MODEL_FAST).
    Per-round and per-item Stage 2 events logged.
  - `solver/pipeline.py` — solve entry, per-attempt, success/error
    events.
  - `solver/retrieval.py` — every ChromaDB search logged with query +
    scored results.
  - `config.py` — added `STAGE2_MODEL` env override.
  - `.env.template` — documents `STAGE2_MODEL`, `SOLVER_LOG`,
    `SOLVER_LOG_VERBOSE`.
  - `tests/test_deterministic.py` — 2 new tests for the logger and the
    Stage 2 model override.

## Required follow-up

This is the run-loop after applying v7.1.2:

```bash
# 1. Apply the patch and rebuild (no rag_text changes; ChromaDB doesn't need re-ingest)
tar -xzf physics_solver_v7_1_2_tar.gz
cd physics_solver_v7
pip install -r requirements.txt

# 2. Verify deterministic tests
python tests/test_deterministic.py             # expect: 42 passed

# 3. FIRST: verify RAG is actually in sync (this is the question you asked)
python scripts/verify_rag.py --check-sync
# Expected output: "✅ Sync OK — ChromaDB is consistent with the graph file."
# If you see "STALE INDEX", run: python -m solver.ingest --reingest

# 4. Run the failing test under the diagnostic
python scripts/diagnose_question.py "A body of density 8000 kg/m^3 and volume 0.5 m^3 starts from rest and reaches 30 m/s over 40 m. Find the net force."

# Read the output: did the retrieval surface general_density_definition?
# Did Stage 2 see it in the candidates? What was the LLM's reason for
# rejecting? If the equation was in the candidates but rejected, this
# confirms the 8B-model hypothesis.

# 5. If the hypothesis is confirmed, flip the Stage 2 model:
echo 'STAGE2_MODEL=llama-3.3-70b-versatile' >> .env

# 6. Re-run live tests
python tests/test_live.py
```

The diagnostic output from step 4 is the artifact to send back if you
want my help interpreting it — paste the per-stage breakdown and I can
read it directly.
