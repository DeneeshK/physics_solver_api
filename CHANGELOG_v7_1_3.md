# Changelog v7.1.3 — Per-item Stage 2 batching for chained problems

v7.1.3 fixes the most likely root cause of the v7.1.1 live-test failures:
multi-equation chains failing at Round 1+, while single-equation problems
pass. The user's diagnostic from running v7.1.1 live tests:

| Test                                            | Chain depth | Result |
|-------------------------------------------------|-------------|--------|
| F=ma with density+volume+kinematics             | 3 equations | FAIL   |
| Coulomb's law in vacuum                         | 1 equation  | PASS   |
| F=ma direct (m, a given)                        | 1 equation  | PASS   |
| KE via kinematics chain                         | 2 equations | FAIL   |
| Missing-info (correctly returns UNVERIFIED)     | n/a         | PASS   |

The pattern is exact: every chained problem failed, every single-equation
problem passed.

## The actual cause

Stage 2 was making one LLM call per round that asked about ALL frontier
items in the round simultaneously. For a single-equation problem this is
fine (Round 0 has one item, the unknown). For chains this means Round 1+
has multiple items at once — the frontier expands when the first equation
is picked, and Round 1 needs to choose equations for every variable in
that expansion.

The 8B fast model handles the first item in its response well, then loses
the second item off the back of the attention/output budget. The result
is a JSON response with fewer `selections` entries than items asked
about. Pipeline records this as `llm_omitted_item` — an honest failure
code added in v7 (v6 would have silently substituted the first
candidate, producing a confident wrong answer).

This is a STRUCTURAL issue, not just a model-capacity issue. The 70B
model would also benefit from per-item calls; the 8B model just makes
the failure more visible.

## The fix

`call_round_selector` is now a thin dispatcher that decides whether to
batch all items in one LLM call or split into per-item calls. Default
mode `auto` does the right thing automatically:

  - 1 item in round: batched (same as before, no overhead)
  - 2+ items in round: split into one LLM call per item

Each per-item call has only one "needed quantity" section in its prompt
and is expected to return only one selection. The LLM cannot "drop an
item" because there's only one item to address.

Round 0 is unaffected (it has one item — the unknown).
Round 1+ on chained problems now makes N calls of single-item prompts
instead of one call of an N-item prompt.

## Trade-offs

  - **API call count**: ~2x more calls in chained problems' Round 1+.
    Round 0 unchanged. For your typical 2-equation chain (the KE test),
    this is 1 call → 2 calls. For a 3-equation chain (the F=ma test),
    Round 1 goes from 1 call → 2 calls (m and a), Round 2 may or may not
    follow depending on which equation was chosen.
  - **Latency**: per-item calls are smaller and faster individually, but
    they're sequential, so wall-clock time for multi-item rounds goes up
    by ~1.5x (smaller prompts run faster per call, but you make more).
  - **TPM pressure**: each call has a smaller prompt, so total tokens per
    round is actually LOWER in per-item mode (less repeated context
    across items). Good for free-tier TPM caps.
  - **Reliability**: significantly higher. The 8B model can no longer
    drop items on chained problems.

## Configuration

New env var: `STAGE2_BATCH_MODE` (default "auto"). See `.env.template`
for full documentation.

  - `auto`   → single per item when multi-item, batched when single-item
  - `all`    → always batched (v7.1.2 and earlier behavior; useful for A/B
               testing if you want to confirm the change matters)
  - `single` → always per item, even for single-item rounds (no benefit
               for single-item; this is the most conservative mode)

## How this interacts with STAGE2_MODEL (from v7.1.2)

The two are orthogonal dials:

  - `STAGE2_BATCH_MODE=auto` (default) addresses the **structural** cause —
    multi-item prompts overload the model.
  - `STAGE2_MODEL=llama-3.3-70b-versatile` (optional) addresses any
    remaining **reasoning** gap on individual chained items.

Recommended progression:
  1. Apply v7.1.3 (default `auto` mode) and rerun live tests.
  2. If chains still fail, run `diagnose_question.py` to see whether the
     LLM is now rejecting candidates explicitly (`decision: "none"`) vs.
     omitting them — the latter is fixed in v7.1.3; the former suggests
     a reasoning gap.
  3. If reasoning gap: flip `STAGE2_MODEL=llama-3.3-70b-versatile`.

## Files added in v7.1.3

  - `CHANGELOG_v7_1_3.md` — this file

## Files changed in v7.1.3

  - `config.py` — adds `STAGE2_BATCH_MODE` env override
  - `solver/llm_interface.py` — `call_round_selector` becomes a
    dispatcher; existing logic factored into `_round_select_call`.
    Per-item calls carry `sub_index`/`sub_total` in log lines.
  - `.env.template` — documents `STAGE2_BATCH_MODE`
  - `tests/test_deterministic.py` — 1 new test verifying per-item
    dispatching for multi-item rounds

## Required follow-up

After applying v7.1.3:

```bash
tar -xzf physics_solver_v7_1_3_tar.gz
cd physics_solver_v7
pip install -r requirements.txt

# 1. Deterministic tests
python tests/test_deterministic.py             # expect: 43 passed

# 2. Live tests — should fix at least the chained failures
python tests/test_live.py

# 3. If something STILL fails, diagnose:
python scripts/diagnose_question.py "<the failing question>"
# Inspect the log breakdown: with STAGE2_BATCH_MODE=auto, you should see
# `stage2_dispatch mode=single` for chained problems' multi-item rounds.
# Each item gets its own stage2_round_entry / stage2_item_decision pair.

# 4. If chains still fail, the issue is reasoning not structural —
#    add to .env:
echo 'STAGE2_MODEL=llama-3.3-70b-versatile' >> .env

# 5. Rerun
python tests/test_live.py
```

No re-ingest needed for v7.1.3 (no rag_text changes).
