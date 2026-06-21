# Changelog v7.1.7 — Local LLM support (Ollama) with zero-edit provider switch

v7.1.7 makes the solver run against a LOCAL model (Ollama on your RTX 3050)
or Groq cloud, switchable entirely through .env — no code edits to change
provider. This removes the Groq free-tier token-per-minute ceiling that was
crashing bulk test runs, so you can run 50-100+ questions to find the real
failure surface.

## The one-line switch

The provider is chosen by `LLM_BASE_URL`:
  - SET (e.g. http://localhost:11434/v1) → local OpenAI-compatible server
  - UNSET / blank → Groq cloud (original behavior, unchanged)

Both backends expose the identical `.chat.completions.create(...)` surface,
so nothing in the 5-stage pipeline changes between them. All logging, the
429-retry, batching, canonicalization — all identical.

## What changed in code (2 files)

**config.py**
  - Added `LLM_BASE_URL` and `LLM_API_KEY` env vars.
  - `MODEL_FAST`, `MODEL_SMART`, `STAGE2_MODEL` now default to the local
    tuned model name `physics-solver-llm` (created by setup_local_llm.sh),
    and remain fully env-overridable. To use Groq, set them to the Groq
    model strings in .env.

**solver/llm_interface.py**
  - Replaced the hardcoded `client = Groq(api_key=...)` with a provider
    switch: if `LLM_BASE_URL` is set, build `OpenAI(base_url=..., api_key=...)`
    (works with Ollama/LM Studio/vLLM); else build `Groq(...)` as before.
  - Added a one-time `llm_provider_selected` log event so the trace shows
    which backend served a run.
  - The 429-retry logic is unchanged and provider-agnostic (it catches
    generic Exception; locally there are no 429s so it simply never fires).

## Verification

The full 51-test deterministic suite passes in BOTH modes:
  - LOCAL  (LLM_BASE_URL set): 51 passed
  - GROQ   (LLM_BASE_URL unset): 51 passed

No regression in either direction.

## New helper files shipped

  - `.env.local`  — ready-to-use local config (rename to .env)
  - `.env.cloud`  — ready-to-use Groq config (rename to .env, add key)
  - `.env.template` — updated with both options documented

(The setup_local_llm.sh script, QUESTION_BANK_PROMPT.md, and
CONNECT_SOLVER_TO_LOCAL.md were delivered separately.)

## Requirements note

Local mode needs the `openai` Python package:
    pip install openai
It usually ships as a Groq dependency already. Groq mode needs `groq` (already
present). Both can be installed; only the configured one is imported at runtime.

## Why a 3B local model, not 8B (recap)

On 8GB VRAM, a 3B Q4 model leaves ~3-4GB for KV cache; an 8B Q4 model leaves
almost none and spills to system RAM, killing throughput. The pipeline does
many small JUDGMENT calls, not heavy reasoning, so 3B is sufficient. A
reasoning model (DeepSeek-R1-distill) is the wrong tool here — it emits long
<think> traces that fight the JSON parser and slow every call. Default:
Qwen2.5-3B-Instruct.

## What this unblocks

The whole point: run a 50-100 question bank with no quota limit, get the
failure surface grouped by reasoning_type and failure stage, and fix
SYSTEMATIC failure modes instead of individual questions. The evaluation
harness is the next build.
