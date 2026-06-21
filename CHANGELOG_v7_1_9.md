# Changelog v7.1.9 — Local-model robustness: LaTeX-in-JSON + empty-givens retry

The first local (Ollama / Qwen2.5-3B) run dropped to 1/5 — worse than Groq's
3/5. The log showed this was NOT an architecture regression. The concept
matching, retrieval, and the v7.1.6 canonicalization all worked
(`target_symbol_canonicalized: KE -> K` fired correctly, Coulomb's law was
picked correctly). Two local-model output-quality bugs were discarding correct
work. Both are fixed here.

## Bug 1 (critical) — Local models write LaTeX inside JSON, breaking the parser

The Qwen model writes its reasoning with LaTeX math in the "reason" field:
`W = Fs\cos(\theta)`, `\( v^2 = u^2 + 2a\Delta s \)`, `F = -\frac{dU}{dr}`.
Those backslashes are INVALID JSON escapes (`\c`, `\D`, `\(`, `\f`-as-frac),
so `json.loads` throws "Invalid \escape" and the ENTIRE selection is
discarded — recorded as a false `no-fit`.

The model was choosing the RIGHT equation every time this happened
(`electrostatics_coulomb_law`, `work_energy_power_kinetic_energy`) and being
thrown away on a formatting technicality. Groq's Llama never wrote LaTeX, so
this bug was invisible on cloud and only surfaced locally.

Fix: `_extract_json` now sanitizes invalid backslash escapes before parsing
(`_sanitize_json_escapes`). Any backslash that isn't the start of a valid JSON
escape (`\" \\ \/ \b \f \n \r \t \uXXXX`) is doubled, turning `\cos` into a
literal backslash in the string value. Content is preserved; JSON becomes
valid. Valid escapes and `\uXXXX` are left untouched. Verified against the
exact failing payloads from the user's log.

Defense in depth: the Stage 2 prompt now also instructs the model to write
plain ASCII math in the reason field (no LaTeX) — so fewer invalid escapes are
emitted in the first place. The sanitizer is the safety net.

## Bug 2 (important) — Local model returns empty "given" for numeric questions

The 3B model returned `"given": {}` for EVERY test question — even
"A 5 kg object experiences a net force of 20 N" which plainly has numbers. It
described the unknown correctly but skipped extracting the inputs. With no
givens, no chain can resolve. The v7.1.5/v7.1.6 prompt fixes that worked on
Groq's Llama don't land as reliably on the smaller local model.

Fix: an empty-givens retry. If the question text contains digits but the model
extracted zero givens, `parse_question` retries ONCE with a sharpened
instruction forcing population of "given". It only fires in this specific
failure case, so it costs nothing on the happy path.

Crucially, genuinely underspecified questions ("Find the velocity of the
object.") contain NO digits, so they do NOT trigger the retry and correctly
remain empty — preserving the anti-hallucination behavior (that test was the
one thing passing locally, and it still passes).

## Why these were invisible until now

Both are small-model output-quality issues, not logic issues:
  - Groq's Llama-8B wrote clean JSON and followed extraction instructions.
  - Qwen-3B writes LaTeX and under-extracts.
The architecture is provider-agnostic; the prompt-following and output-format
robustness it relies on is not. v7.1.9 hardens the seams so the pipeline
tolerates a weaker model's habits.

## Expected impact

These two fixes should recover most of the local failures:
  - Coulomb's law: the pick was correct, only the LaTeX parse killed it -> Fix 1
  - F=ma direct, KE chain: empty givens starved the chain -> Fix 2
  - "Find the velocity": still correctly UNVERIFIED (unchanged)

Re-run `python tests/test_live.py` after applying. If chains still wander
AFTER givens are populated, that points to the 3B model's reasoning ceiling on
multi-item selection — at which point a slightly larger local model (e.g.
qwen2.5:7b if it fits, or back to Groq's 8B/70B for Stage 2) is the lever, not
more prompt work.

## Files changed in v7.1.9

  - `solver/llm_interface.py`:
    - `_sanitize_json_escapes` (new) — doubles invalid JSON backslash escapes
    - `_extract_json` — retries parse with sanitized escapes on failure
    - `parse_question` — empty-givens retry for numeric questions
    - Stage 2 prompt — plain-ASCII-math (no LaTeX) directive in reason field
  - `tests/test_deterministic.py` — 3 new tests (LaTeX tolerance, sanitizer
    correctness, retry trigger logic)
  - `CHANGELOG_v7_1_9.md` — this file

## Deployment

```bash
tar -xzf physics_solver_v7_1_9_tar.gz
cd physics_solver_v7
pip install -r requirements.txt
python tests/test_deterministic.py        # expect: 54 passed
python tests/test_live.py                  # re-run the 5 locally
```

No re-ingest needed (no rag_text changes).
