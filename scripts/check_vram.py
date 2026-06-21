"""
scripts/check_vram.py
Confirm the local 7B model fits in VRAM alongside the (CPU) embedder BEFORE
committing to a long run. Run this after setup_local_llm.sh and before the
89-question bank.

What it checks:
  1. nvidia-smi is available and reports your GPU + free/used VRAM.
  2. The Ollama server is up.
  3. The 'physics-solver-llm' model loads and answers.
  4. After loading, how much VRAM is actually used vs free — and whether
     Ollama reports the model as fully on GPU (not spilled to system RAM).
  5. The embedder device the solver will use (should be CPU).

Usage:
  python scripts/check_vram.py
"""
import os
import sys
import json
import subprocess
import urllib.request

OLLAMA = "http://localhost:11434"
MODEL = os.getenv("STAGE2_MODEL", "physics-solver-llm")


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        return f"(failed: {e})"


def check_nvidia_smi():
    print("=" * 64)
    print(" 1. GPU / VRAM (nvidia-smi)")
    print("=" * 64)
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader"])
    if "failed" in out or not out.strip():
        print("  Could not read nvidia-smi. Is the NVIDIA driver installed?")
        return None
    name, total, used, free = [x.strip() for x in out.strip().split(",")]
    print(f"  GPU:        {name}")
    print(f"  Total VRAM: {total}")
    print(f"  Used:       {used}")
    print(f"  Free:       {free}")
    return {"total": total, "used": used, "free": free}


def check_server():
    print("\n" + "=" * 64)
    print(" 2. Ollama server")
    print("=" * 64)
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/version", timeout=5) as r:
            v = json.loads(r.read())
        print(f"  Server up. Version: {v.get('version','?')}")
        return True
    except Exception as e:
        print(f"  Server NOT reachable at {OLLAMA} ({e})")
        print("  Start it with:  ollama serve")
        return False


def load_and_test_model():
    print("\n" + "=" * 64)
    print(f" 3. Load + test model: {MODEL}")
    print("=" * 64)
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user",
                      "content": "Reply with ONLY this JSON: {\"ok\": true}"}],
        "temperature": 0.1,
        "keep_alive": -1,
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read())
        content = resp["choices"][0]["message"]["content"]
        print(f"  Model answered: {content.strip()[:80]}")
        return True
    except Exception as e:
        print(f"  Model did NOT answer ({e})")
        return False


def check_model_placement():
    print("\n" + "=" * 64)
    print(" 4. Is the model fully on GPU? (ollama ps)")
    print("=" * 64)
    out = _run(["ollama", "ps"])
    print(out if out.strip() else "  (ollama ps returned nothing)")
    # ollama ps shows a PROCESSOR column: "100% GPU" is what we want.
    if "GPU" in out and "CPU" in out:
        print("  ⚠ WARNING: model is PARTIALLY on CPU — it spilled out of VRAM.")
        print("    This will be slow. Options: use the Q3 quant")
        print("    (./setup_local_llm.sh 7b-q3), lower num_ctx, or close other")
        print("    GPU users (browser, desktop effects).")
    elif "100% GPU" in out or ("GPU" in out and "CPU" not in out):
        print("  ✓ Model appears fully on GPU.")


def check_embed_device():
    print("\n" + "=" * 64)
    print(" 5. Embedder device (should be CPU to save VRAM)")
    print("=" * 64)
    dev = os.getenv("EMBED_DEVICE", "cpu")
    print(f"  EMBED_DEVICE = {dev}")
    if dev == "cpu":
        print("  ✓ Embedder on CPU — frees ~1.3GB VRAM for the model.")
    else:
        print("  ⚠ Embedder on GPU — this steals VRAM from the 7B model.")
        print("    On 8GB, set EMBED_DEVICE=cpu in your .env.")


def main():
    before = check_nvidia_smi()
    if not check_server():
        sys.exit(1)
    ok = load_and_test_model()
    check_model_placement()
    after = check_nvidia_smi_quiet()
    check_embed_device()

    print("\n" + "=" * 64)
    print(" SUMMARY")
    print("=" * 64)
    if before and after:
        print(f"  VRAM free before load: {before['free']}")
        print(f"  VRAM free after load:  {after['free']}")
    if ok:
        print("  ✓ Model loads and answers. If placement shows 100% GPU and")
        print("    you have >0.5GB free, you're clear to run the bank.")
    else:
        print("  ✗ Model did not answer — fix this before the bank run.")
    print("=" * 64)


def check_nvidia_smi_quiet():
    out = _run(["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader"])
    if "failed" in out or not out.strip():
        return None
    total, used, free = [x.strip() for x in out.strip().split(",")]
    return {"total": total, "used": used, "free": free}


if __name__ == "__main__":
    main()
