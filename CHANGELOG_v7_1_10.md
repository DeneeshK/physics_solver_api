# Changelog v7.1.10 — 7B model for both stages, VRAM-managed

After v7.1.9 fixed the two local-model plumbing bugs (LaTeX-in-JSON parse,
empty givens), the 5-question live run isolated the remaining failure cleanly:
the 3B model passes Stage 1 (parsing) but fails Stage 2 (equation selection).
It picks near-miss equations (e.g. mgh for kinetic energy, F=mg instead of
F=ma) and drifts off the symbol it was asked about (asked for "v", answers
about "a"). That is a reasoning-capacity ceiling, not a code bug. This release
moves to a stronger model for the selection work and manages the 8GB VRAM
budget so it fits.

## The decision: ONE 7B model for BOTH stages, kept resident

  - Model: Qwen2.5-7B-Instruct (Q4_K_M). ~4.7GB weights — meaningfully
    stronger reasoning than 3B, still fits 8GB with the embedder on CPU.
  - Used for BOTH Stage 1 and Stage 2, not split. Two models can't coexist
    in 8GB, so a 3B-for-Stage-1 + 7B-for-Stage-2 split would force
    evict-and-reload on every question (several seconds each × 89). One
    resident model is faster and far more stable. The 7B does Stage 1's
    easy parsing job fine — it's already loaded.
  - Kept resident the whole run via keep-alive, so no reload churn across
    the 89-question bank.

## VRAM management (the 8GB math)

  Qwen2.5-7B Q4_K_M weights      ~4.7 GB
  KV cache @ num_ctx 4096        ~0.8-1 GB
  CUDA/driver/display            ~0.5-1 GB
  ----------------------------------------
  Total                          ~6-6.7 GB  → fits 8GB with headroom
  BGE embedder ON GPU            +1.3 GB    → would push to the ceiling

So the embedder is moved to CPU. It runs ONCE per question (embedding the
search query), so CPU costs almost nothing in wall-clock time but frees the
~1.3GB the 7B needs to stay resident and stable.

## Code changes

**solver/retrieval.py** — the embedding model is pinned to a configurable
device, default CPU:
```python
_embed_device = os.getenv("EMBED_DEVICE", "cpu")
self.model = SentenceTransformer(EMBED_MODEL, device=_embed_device)
```
Override with `EMBED_DEVICE=cuda` if you ever want it back on GPU.

**solver/pipeline.py** — between-question memory cleanup WITHOUT unloading the
model. `_free_inter_question_memory()` runs at the top of every `solve()`:
```python
gc.collect()
torch.cuda.empty_cache(); torch.cuda.ipc_collect()  # if torch+CUDA present
```
This clears transient per-question allocations and the embedder's CUDA cache
(if on GPU) while leaving the LLM weights resident — the model is held in the
Ollama server's own process, so this never touches them. Safe when torch is
absent (embedder on CPU).

**solver/llm_interface.py** — Stage 2 prompt hardened with the principle the
user emphasized and the 3B violated:
  - "Decision principle 6: answer ONLY about the symbol you are asked." The
    single most repeated 3B failure was drifting off the asked symbol (asked
    for v, answered about a). The prompt now explicitly forbids this with the
    exact failure as a worked counter-example.
  - Reinforced "you are NOT deriving or computing — stay a concept-matcher,
    not a calculator", with "a = F/m = 20/5 = 4"-style derivation called out
    as wrong. The LLM matches concepts; SymPy does all math. (This was always
    the design; the 3B kept half-deriving in its reasoning and grabbing the
    wrong equation as a result.)

## New / updated tooling

**setup_local_llm.sh** — now defaults to Qwen2.5-7B-Instruct (Q4_K_M),
configures keep-alive so the model stays resident, and prints `.env` lines
including `EMBED_DEVICE=cpu`. Variants: `7b-q3` (tighter quant), `3b`
(fallback).

**scripts/check_vram.py** (new) — run BEFORE the 89-question bank to confirm
the fit: checks nvidia-smi, loads the model, reports VRAM used/free, and warns
if `ollama ps` shows the model spilled partially to CPU (the slow case).

**.env.local** — updated: 7B model for all three model vars, `EMBED_DEVICE=cpu`.

## Why the architecture did NOT change

The user's design is unchanged and correct: the LLM is a concept-matcher and
a stateful agent, never a calculator; SymPy does all arithmetic. The 3B
failures were the model being too weak to FOLLOW that design (it half-derived
and picked near-miss equations), not the design being wrong. The fix is a model
capable of holding the discipline the design requires.

## Deployment

```bash
# 1. Set up the 7B model (downloads ~4.7GB, creates physics-solver-llm)
./setup_local_llm.sh

# 2. Put the printed .env lines in your .env (includes EMBED_DEVICE=cpu)
cp .env.local .env    # already has the right values

# 3. CONFIRM THE VRAM FIT before committing to a long run
python scripts/check_vram.py
#    Want: model answers, "100% GPU" placement, >0.5GB free.
#    If it shows partial CPU: try ./setup_local_llm.sh 7b-q3

# 4. Re-run the 5 live tests
python tests/test_live.py

# 5. If 4-5/5 pass, run the bank
python tests/evaluate_bank.py questions/question_bank.json --report results.json
```

No re-ingest needed (no rag_text changes). 54 deterministic tests pass.

## What to expect

The 7B should hold the asked symbol and pick exact concepts (F=ma not F=mg,
K=½mv² not mgh). If it does, the 5 should largely pass and the bank becomes
meaningful. If specific questions still fail, the evaluate_bank cross-tab will
show whether they cluster (systematic, fixable) or are the known v7.3
graph-content quirks (Doppler, beats, thermo sign conventions).
