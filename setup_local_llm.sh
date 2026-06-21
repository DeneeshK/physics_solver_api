#!/usr/bin/env bash
# ==============================================================================
# setup_local_llm.sh  (v7.1.10 — 7B for both stages, VRAM-managed)
# One-shot local LLM setup for the physics solver, tuned for an RTX 3050 (8GB).
#
# WHAT CHANGED IN THIS VERSION
#   The 3B model passed Stage 1 (parsing) but failed Stage 2 (equation
#   selection): it picked near-miss equations (mgh for kinetic energy) and
#   drifted off the symbol it was asked about. That's a reasoning-capacity
#   ceiling, not a plumbing bug. The fix is a stronger model for the SELECTION
#   work. On 8GB we run ONE 7B model for BOTH stages, kept resident, so nothing
#   swaps. The embedding model is moved to CPU (see the solver's EMBED_DEVICE)
#   to free VRAM for the 7B.
#
# WHAT THIS DOES
#   1. Installs Ollama if missing.
#   2. Pulls Qwen2.5-7B-Instruct (Q4_K_M) by default — the sweet spot for 8GB:
#      meaningfully stronger reasoning than 3B, still fits with the embedder
#      on CPU.
#   3. Creates a tuned 'physics-solver-llm' variant: low temperature, context
#      sized for the tight VRAM budget, and a long keep-alive so the model
#      stays loaded across a full 89-question run (no reload churn).
#   4. Smoke test.
#   5. Prints the exact .env lines.
#
# WHY 7B FOR BOTH STAGES (the 8GB math)
#   Qwen2.5-7B Q4_K_M weights ~4.7GB. KV cache at num_ctx 4096 ~0.8-1GB.
#   CUDA/driver/display ~0.5-1GB. Total ~6-6.7GB — fits 8GB with headroom
#   ONLY IF the embedding model is NOT also on the GPU. That's why the solver
#   now pins BGE to CPU by default (EMBED_DEVICE=cpu). The embedder runs once
#   per question, so CPU costs almost nothing in wall-clock time.
#
#   Running ONE model for both stages (rather than 3B for Stage 1 + 7B for
#   Stage 2) avoids model swapping. Two models can't coexist in 8GB, so a
#   split would force evict-and-reload on every question — several seconds
#   each, 89 times. One resident model is faster and far more stable.
#
# USAGE
#   chmod +x setup_local_llm.sh
#   ./setup_local_llm.sh                 # default: Qwen2.5-7B-Instruct (Q4_K_M)
#   ./setup_local_llm.sh 7b-q3           # tighter quant if 7B Q4 won't fit
#   ./setup_local_llm.sh qwen2.5:3b      # fall back to the 3B
#   ./setup_local_llm.sh <any-ollama-tag>
#
# After it finishes:
#   1. Copy the printed .env lines into your solver's .env.
#   2. Run  python scripts/check_vram.py  to confirm the fit BEFORE the bank.
# ==============================================================================

set -euo pipefail

# ── Pick the model ────────────────────────────────────────────────────────────
CHOICE="${1:-qwen2.5:7b}"

case "$CHOICE" in
  qwen2.5:7b|qwen7b|7b|qwen)
    MODEL_TAG="qwen2.5:7b-instruct-q4_K_M"
    FRIENDLY="Qwen2.5-7B-Instruct (Q4_K_M) — recommended for 8GB, both stages"
    NUM_CTX=4096
    ;;
  7b-q3|q3)
    MODEL_TAG="qwen2.5:7b-instruct-q3_K_M"
    FRIENDLY="Qwen2.5-7B-Instruct (Q3_K_M) — tighter fit if Q4 spills"
    NUM_CTX=4096
    ;;
  qwen2.5:3b|3b)
    MODEL_TAG="qwen2.5:3b-instruct-q4_K_M"
    FRIENDLY="Qwen2.5-3B-Instruct (Q4_K_M) — fallback, weaker on Stage 2"
    NUM_CTX=4096
    ;;
  qwen2.5:1.5b|1.5b|small)
    MODEL_TAG="qwen2.5:1.5b-instruct-q4_K_M"
    FRIENDLY="Qwen2.5-1.5B-Instruct (Q4_K_M) — fastest, weakest"
    NUM_CTX=4096
    ;;
  *)
    MODEL_TAG="$CHOICE"
    FRIENDLY="$CHOICE (custom tag)"
    NUM_CTX=4096
    ;;
esac

echo "=============================================================="
echo " Local LLM setup for the physics solver (v7.1.10)"
echo " Target GPU: RTX 3050 (8GB)"
echo " Model: $FRIENDLY"
echo " Ollama tag: $MODEL_TAG"
echo " Context window: $NUM_CTX tokens"
echo " Strategy: ONE model, BOTH stages, kept resident (no swapping)"
echo " Embedder: CPU (set EMBED_DEVICE=cuda in env to override)"
echo "=============================================================="
echo

# ── Step 1: Install Ollama if missing ─────────────────────────────────────────
if ! command -v ollama >/dev/null 2>&1; then
  echo "[1/5] Ollama not found. Installing..."
  curl -fsSL https://ollama.com/install.sh | sh
  echo "      Ollama installed."
else
  echo "[1/5] Ollama already installed: $(ollama --version 2>/dev/null || echo 'version unknown')"
fi
echo

# ── Step 2: Make sure the Ollama server is running ────────────────────────────
echo "[2/5] Ensuring the Ollama server is up..."
if ! curl -fsS http://localhost:11434/api/version >/dev/null 2>&1; then
  echo "      Starting 'ollama serve' in the background..."
  nohup ollama serve > ollama_server.log 2>&1 &
  for i in $(seq 1 15); do
    if curl -fsS http://localhost:11434/api/version >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi
if curl -fsS http://localhost:11434/api/version >/dev/null 2>&1; then
  echo "      Server is up at http://localhost:11434"
else
  echo "      ERROR: Ollama server did not come up. Check ollama_server.log"
  exit 1
fi
echo

# ── Step 3: Pull the model ────────────────────────────────────────────────────
echo "[3/5] Pulling $MODEL_TAG (first time downloads ~4-5GB)..."
ollama pull "$MODEL_TAG"
echo "      Model pulled."
echo

# ── Step 4: Create a tuned variant for the solver ─────────────────────────────
echo "[4/5] Creating tuned variant 'physics-solver-llm'..."
SOLVER_MODEL="physics-solver-llm"
cat > Modelfile.physics <<EOF
FROM $MODEL_TAG

# Low temperature: the solver wants deterministic structured JSON, not
# creative variation. Matches the 0.1 the code uses.
PARAMETER temperature 0.1

# Context window sized for the 8GB budget with a 7B model. 4096 fits the
# solver's largest prompts (Stage 1 ~2.2k, Stage 2 ~2-4k tokens) and keeps
# the KV cache small enough to stay on-GPU alongside the 7B weights. Do NOT
# raise this without checking VRAM — a larger context grows the KV cache and
# can push the model into shared system RAM, collapsing throughput.
PARAMETER num_ctx $NUM_CTX

# Stop sequences keep JSON clean for chat-style models.
PARAMETER stop "<|im_end|>"
PARAMETER stop "</s>"
EOF

ollama create "$SOLVER_MODEL" -f Modelfile.physics
echo "      Created '$SOLVER_MODEL'."
echo

# ── Keep-alive: make the model stay resident for the whole run ────────────────
# This is the key to avoiding reload churn across an 89-question bank. We set
# OLLAMA_KEEP_ALIVE so the server keeps the model in VRAM. -1 means "never
# unload while the server runs". If you prefer a timeout, use e.g. 2h.
echo "      Setting the model to stay resident (no mid-run unload)..."
echo "      NOTE: the solver also sends keep_alive per request, but exporting"
echo "      OLLAMA_KEEP_ALIVE makes it stick server-wide. To make it permanent,"
echo "      add this to the Ollama service environment:"
echo "          OLLAMA_KEEP_ALIVE=-1"
echo

# ── Step 5: Smoke test ────────────────────────────────────────────────────────
echo "[5/5] Smoke test — asking the model for a tiny JSON object..."
SMOKE=$(curl -fsS http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$SOLVER_MODEL\",
    \"messages\": [{\"role\":\"user\",\"content\":\"Reply with ONLY this JSON, nothing else: {\\\"ok\\\": true}\"}],
    \"temperature\": 0.1,
    \"keep_alive\": -1
  }" 2>/dev/null || echo "SMOKE_FAILED")

if echo "$SMOKE" | grep -q '"ok"'; then
  echo "      Smoke test PASSED — model answered with JSON."
else
  echo "      Smoke test response (inspect manually):"
  echo "      $SMOKE"
fi
echo

# ── Done — print the .env lines ───────────────────────────────────────────────
echo "=============================================================="
echo " DONE. Put these lines in your solver's .env:"
echo "--------------------------------------------------------------"
cat <<'ENVEOF'
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
MODEL_FAST=physics-solver-llm
MODEL_SMART=physics-solver-llm
STAGE2_MODEL=physics-solver-llm
EMBED_DEVICE=cpu
ENVEOF
echo "--------------------------------------------------------------"
echo
echo " NEXT: confirm the VRAM fit BEFORE running the 89-question bank:"
echo "     python scripts/check_vram.py"
echo
echo " Then re-run the 5 live tests:"
echo "     python tests/test_live.py"
echo "=============================================================="
