# Changelog v7.1.5 — Four mechanical bug fixes from live-test log analysis

The v7.1.4 live test produced a detailed log. Reading it carefully showed
the architecture is correct — the F=ma chained problem PASSED (45000 N,
HIGH confidence, exact chain: F=ma → m from rho/V → a from kinematics).
The failures were four mechanical bugs, now fixed.

## What the log proved

At timestamp 22:54:12–22:54:46, the F=ma-with-density-volume problem
solved correctly:
  - Round 0 picked F = m*a by concept ("Newton's Second Law of Motion
    matches the physics scenario of a body accelerating from rest")
  - Round 1 picked rho = m/V for mass, v^2 = u^2 + 2as for acceleration
  - SymPy resolved the chain to 45000 N

The concept-matching and derivability reasoning the architecture is built
on both worked. The bugs below are mechanical, not architectural.

## Bug 1 — Stage 1 non-determinism on "starts from rest"

Same question, two runs, two different parses:
  - Run A: given = [rho, V, u, v, s, g]   (u=0 extracted from "starts from rest")
  - Run B: given = [rho, V, v, s, g]      (u DROPPED)

When u is dropped, the solver later opens a round hunting for u, which
cascades into wrong picks. Fix: Stage 1 prompt now has an explicit
"IMPLICIT GIVENS" section instructing the model to extract numeric values
from physics phrases:
  - "starts from rest" / "released from rest" / "dropped" → u = 0
  - "comes to rest" / "comes to a stop" → v = 0
  - "from a height of H" in motion → s = H AND u = 0
  - "uniform motion" / "constant velocity" → a = 0

The prompt explicitly notes that the downstream solver treats
missing-symbol and known-zero-value DIFFERENTLY: known-zero unblocks
chains; missing forces an extra round that often fails.

## Bug 2 — Bloated constants in every Stage 2 prompt

Every Stage 2 prompt listed all ~10 universal constants (Planck's
constant, Boltzmann's constant, speed of light, Avogadro's number, etc.)
in the ALREADY KNOWN section — even for a pure kinematics question. This
wasted ~300 bytes per prompt and accelerated TPM exhaustion.

Fix: the ALREADY KNOWN section now filters out any universal constant
that no candidate equation in the current round actually uses. Constants
remain in `available` for SymPy substitution; they're just not shown to
the LLM when irrelevant. A kinematics round now shows 3 lines (v, s, g)
instead of 16.

Question-given values are always shown. 'g' is always shown when present
(it only enters `available` when Stage 1 flags it as scenario-implied).

## Bug 3 — One rate limit killed the entire test run

In the live run, test #3 hit a Groq 429 TPM cap and the pipeline raised,
which aborted every subsequent test. Fix: `_call` now parses Groq's
"Please try again in Xms" hint, sleeps that duration (capped at 30s, +0.5s
buffer), and retries up to 3 times. A single TPM hit now just slows the
run instead of crashing it.

The `llm_error` log event now includes `retry_after_s` and `will_retry`
so you can see backoff behavior in the log.

## Bug 4 — 8B model answers about the wrong symbol (hallucination)

The live log Round 2 asked for 'u' (initial velocity); the 8B model
returned a selection with `needed_symbol: "a"`. The previous code silently
treated 'u' as omitted, hiding the real problem.

Fix: detection + logging. When the LLM responds about a symbol we didn't
ask for, a `stage2_hallucinated_symbol` event is logged with what was
asked vs. what the model responded about. This doesn't auto-correct the
hallucination (that needs the in-selection retry loop, deferred to a
future version) but it makes the diagnostic unambiguous — you'll see
"hallucinated" in the log instead of a misleading "omitted".

## What v7.1.5 does NOT fix (deferred)

**The KE test failure (E_k vs K symbol mismatch).** Stage 1 names kinetic
energy `E_k`; the graph's `work_energy_power_kinetic_energy` equation uses
`K`. When that equation is picked for E_k, the resolver treats K as a
separate unknown needing its own chain — which cascades into work-energy-
theorem, then displacement, then projectile angle, ending in failure.

This is a graph-content / symbol-canonicalization issue, same class as
the malformed equations flagged for v7.3. Fixing it properly means either:
  (a) canonicalizing energy symbols (E_k → K) in Stage 1 or a mapping
      layer, or
  (b) the broader v7.3 graph cleanup.

It's NOT fixed in v7.1.5 because it needs careful graph surgery, not a
prompt tweak. Flagged prominently for v7.3.

## Deployment

```bash
tar -xzf physics_solver_v7_1_5_tar.gz
cd physics_solver_v7
pip install -r requirements.txt

python tests/test_deterministic.py        # expect: 49 passed

# No re-ingest needed (no rag_text changes)
python tests/test_live.py
```

With the constants filter + 429 retry, the live run should no longer
crash mid-suite, and chained kinematics should parse u=0 reliably.

## Files changed in v7.1.5

  - `solver/llm_interface.py`:
    - `_call`: 429 retry with backoff parsing Groq's retry-after hint
    - Stage 2 prompt builder: filter unused universal constants from
      ALREADY KNOWN
    - Stage 2 response handler: detect + log hallucinated symbols
    - Stage 1 prompt: IMPLICIT GIVENS section (phrase → numeric extraction)
  - `tests/test_deterministic.py`: 3 new tests (constants filter,
    hallucination detection, implicit-given prompt rules)
  - `CHANGELOG_v7_1_5.md`: this file
