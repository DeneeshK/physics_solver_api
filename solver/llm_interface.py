"""
solver/llm_interface.py
All Groq API calls for the 5-stage pipeline.

Stage 1  parse_question()       — extended: gives, target, dimensions, implicit constants
Stage 2  call_round_selector()  — batched conceptual equation selection per round
Stage 4  narrate_from_trace()   — trace-based student-facing narration
Stage 5  generate_distractors() — 3 wrong MCQ options (unchanged)

v7.1.2: every LLM call is logged through solver.solver_log. Look in
logs/solver.log for the trace.
"""
from __future__ import annotations
import json
import re
import time
from config import (
    GROQ_API_KEY, MODEL_FAST, MODEL_SMART, GROQ_TEMPERATURE,
    IMPLICIT_CONSTANTS_CATALOG,
    LLM_BASE_URL, LLM_API_KEY,
)
from solver.frontier_resolver import FrontierItem
from solver.solver_log import log, log_error
from config import STAGE2_MODEL, STAGE2_BATCH_MODE

# ── LLM client: local (Ollama/LM Studio/vLLM) or Groq cloud ───────────────────
# v7.1.7: provider is chosen by config. If LLM_BASE_URL is set we use the
# OpenAI-compatible client pointed at that local server; otherwise we use Groq
# exactly as before. Both expose the identical `.chat.completions.create(...)`
# surface, so NOTHING below this block changes between providers.
if LLM_BASE_URL:
    # Local / OpenAI-compatible server (e.g. Ollama at http://localhost:11434/v1)
    from openai import OpenAI
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    _LLM_PROVIDER = f"local:{LLM_BASE_URL}"
else:
    # Groq cloud (default original behavior)
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    _LLM_PROVIDER = "groq"

# One-time provider log so the trace shows which backend served a run.
try:
    log("llm_provider_selected", provider=_LLM_PROVIDER,
        model_fast=MODEL_FAST, model_smart=MODEL_SMART, stage2_model=STAGE2_MODEL)
except Exception:
    pass


def _call(model: str, system: str, user: str, temperature: float = GROQ_TEMPERATURE,
          stage: str = "?", _attempt: int = 1) -> str:
    """
    Wrapped Groq call. Logs request and response (with latency) through
    solver.solver_log. Rate-limit errors and other Groq exceptions get
    log_error'd before re-raising.

    v7.1.5: rate-limit retry with backoff.
    Groq's 429 response includes "Please try again in Xms" or "Xs". We parse
    that hint, sleep the indicated duration (capped at 30s), and retry up to
    3 times. Without this, ONE rate limit anywhere in a test run killed
    every subsequent test. Now a hit on the free-tier TPM cap just slows us
    down — it doesn't crash the run.
    """
    MAX_RETRIES = 3
    log("llm_request",
        stage=stage, model=model, temperature=temperature,
        system_len=len(system), user_len=len(user),
        attempt=_attempt,
        user_preview=user[:500])
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        emsg = str(e)
        is_rate_limit = "429" in emsg or "rate_limit" in emsg.lower() or "tpm" in emsg.lower()

        # v7.1.5: parse Groq's retry-after hint from the error message.
        # Format: "Please try again in 410ms" or "Please try again in 2.5s".
        # Default 5s if not present.
        retry_after_s = 5.0
        if is_rate_limit:
            m_ms = re.search(r"try again in (\d+(?:\.\d+)?)ms", emsg)
            m_s  = re.search(r"try again in (\d+(?:\.\d+)?)s\b", emsg)
            if m_ms:
                retry_after_s = float(m_ms.group(1)) / 1000.0
            elif m_s:
                retry_after_s = float(m_s.group(1))
            retry_after_s = min(retry_after_s + 0.5, 30.0)  # +0.5s buffer, 30s cap

        log_error("llm_error",
                  exc=e, stage=stage, model=model, elapsed_ms=elapsed_ms,
                  is_rate_limit=is_rate_limit,
                  attempt=_attempt,
                  retry_after_s=retry_after_s if is_rate_limit else None,
                  will_retry=is_rate_limit and _attempt < MAX_RETRIES,
                  error_msg=emsg[:1000])

        if is_rate_limit and _attempt < MAX_RETRIES:
            time.sleep(retry_after_s)
            return _call(model, system, user, temperature, stage, _attempt + 1)
        raise
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    content = resp.choices[0].message.content.strip()
    usage = getattr(resp, "usage", None)
    log("llm_response",
        stage=stage, model=model, elapsed_ms=elapsed_ms,
        attempt=_attempt,
        response_chars=len(content),
        prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        total_tokens=getattr(usage, "total_tokens", None) if usage else None,
        response_preview=content[:1000])
    return content


def _sanitize_json_escapes(s: str) -> str:
    """
    v7.1.9: Fix invalid backslash escapes that small local models (Qwen,
    etc.) emit when they write LaTeX math inside JSON string values — e.g.
    "W = Fs\\cos(\\theta)" or "\\( v^2 = u^2 + 2a\\Delta s \\)". In JSON the
    ONLY legal escapes are  \\"  \\\\  \\/  \\b  \\f  \\n  \\r  \\t  \\uXXXX.
    A backslash before anything else (\\c, \\D, \\(, \\t-as-in-theta) is an
    "Invalid \\escape" parse error.

    Groq's Llama didn't write LaTeX, so this never bit on cloud. The local
    model picks the RIGHT equation but its reasoning prose kills the parse,
    so the whole selection is discarded and recorded as a false "no-fit".

    We double any backslash that isn't the start of a valid JSON escape,
    turning \\cos into \\\\cos (a literal backslash in the string value).
    The content is preserved; the JSON becomes valid. \\uXXXX is left intact
    only when followed by 4 hex digits.
    """
    # Matches a backslash NOT followed by a valid JSON escape character.
    # Valid next-chars: " \ / b f n r t  OR  u followed by 4 hex digits.
    invalid_escape = re.compile(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})')
    return invalid_escape.sub(r'\\\\', s)


def _extract_json(text: str) -> dict | list:
    """Strip markdown fences, find the first complete JSON object/array.

    v7.1.9: each json.loads attempt now falls back to a LaTeX-escape-
    sanitized retry before giving up, so local models that write math in
    their reason strings still parse.
    """
    clean = re.sub(r"```(?:json)?|```", "", text).strip()

    def _loads(candidate: str):
        # Try as-is; on invalid-escape, retry with sanitized backslashes.
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return json.loads(_sanitize_json_escapes(candidate))

    # Sometimes the model adds prose before/after the JSON
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = clean.find(start_char)
        if start != -1:
            depth = 0
            for i, ch in enumerate(clean[start:], start):
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        try:
                            return _loads(clean[start:i+1])
                        except json.JSONDecodeError:
                            break
    return _loads(clean)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Parse question
# ═══════════════════════════════════════════════════════════════════════════════

_CONSTANTS_CATALOG_TEXT = "\n".join(
    f'  "{sym}": cue="{meta["cue"]}"'
    for sym, meta in IMPLICIT_CONSTANTS_CATALOG.items()
)

def _build_parse_system(valid_domains: set[str]) -> str:
    domains_text = ", ".join(sorted(valid_domains)) if valid_domains else "(none provided)"
    return f"""
You are a JEE/NEET physics question parser. Extract structured information.

Respond ONLY with valid JSON in this EXACT format:
{{
  "given": {{
    "<symbol>": {{
      "value": <number>,
      "unit": "<SI unit>",
      "name": "<English name>",
      "dimension": "<dimensional formula e.g. M, L, T, MLT-2, LT-1>"
    }}
  }},
  "unknown": {{
    "symbol": "<single symbol>",
    "name": "<English name>",
    "unit": "<SI unit>",
    "dimension": "<dimensional formula>"
  }},
  "implicit_constants": ["<symbol1>", "<symbol2>"],
  "likely_domains": ["<domain1>", "<domain2>"],
  "search_query": "<one sentence describing the unknown and the physical scenario, for equation retrieval>"
}}

Rules:

NEVER INVENT VALUES — this is critical
- Extract ONLY numeric values that are EXPLICITLY present in the question
  text, or that follow from an unambiguous physics phrase ("starts from
  rest" → u=0).
- If the question contains NO numeric data at all (e.g. "Find the velocity
  of the object" with nothing else), the "given" object MUST be empty: {{}}.
  Do NOT invent a displacement, time, mass, or any other value. Inventing
  numbers produces a confident answer to a question that was never asked —
  the worst possible failure.
- A question with no givens and no derivable values is UNDERSPECIFIED. Return
  the empty given set and let the downstream solver report that it cannot
  proceed. That is the correct, honest outcome — not a fabricated scenario.
- Do not carry over numbers from physics you have seen before. The only
  numbers that may appear in "given" are ones a reader can point to in THIS
  question's text (or a phrase-implied zero).

UNIT CONVERSION
- Convert all given values to SI units before outputting
  (km/h→m/s, g/cm³→kg/m³, minutes→seconds, etc.)

SYMBOLS
- Use natural physics symbols (F, m, a, v, u, s, t, rho, V, g, T, P, E, q,
  I, r, n, lambda, etc.). The downstream system will match against a graph
  of equations, with dimension and meaning as the primary disambiguators.
  Don't agonize over symbol choice — pick the most conventional one for the
  scenario; the dimension filter handles ambiguity.

- IMPORTANT: in a motion question, a "height" through which an object
  moves IS a displacement. If the question says "raised to a height of 5m
  and released", and asks for velocity, the 5m is the displacement of the
  motion — represent it as s=5 (displacement), not h=5. Reserve h for
  questions about static stored height (gravitational potential energy in
  isolation, hydrostatic pressure at depth, a column of fluid) where the
  height is a configuration, not a motion variable.

- A "distance traveled" or "path length" in motion is also displacement s,
  not d or x.

IMPLICIT GIVENS — extract values from PHRASES, not just numbers
- "starts from rest" → u = 0 (initial velocity is zero). This is ALWAYS
  an explicit given, not a missing variable. Stage 2 cannot reason about
  rest as a kinematic state; it needs the numeric u=0.
- "comes to rest" / "comes to a stop" → v = 0 (final velocity is zero).
- "released from rest" → u = 0.
- "dropped" / "released" / "let fall" → u = 0 (the object had zero velocity
  when released).
- "from a height of H" used in motion context → s = H AND u = 0 (the
  object is released, so initial velocity is zero; the height is the
  displacement of fall).
- "horizontally" combined with a launch → vertical component of u is 0.
- "uniform motion" / "constant velocity" → a = 0.
- "rests on" / "at rest on a surface" (static scenario, no motion asked)
  → does not imply v=0 unless motion is involved in the question.

Always parse phrase-implied numerics into the given dict. The downstream
solver treats missing-symbol and known-zero-value DIFFERENTLY: known-zero
unblocks chains, missing forces an extra round that often fails.

DIMENSIONAL FORMULAS
- M=mass, L=length, T=time, A=current, K=temperature, N=amount of substance.
  Examples: force=MLT-2, velocity=LT-1, acceleration=LT-2, mass=M,
  density=ML-3, energy=ML2T-2, power=ML2T-3, charge=AT, current=A,
  momentum=MLT-1, resistance=ML2T-3A-2.
- Use compact form (MLT-2), no spaces, no carets, no asterisks.

IMPLICIT CONSTANTS
- implicit_constants: list ONLY symbols from this catalog that the scenario
  implies WITHOUT the problem stating a numeric value:
{_CONSTANTS_CATALOG_TEXT}
  Examples: "free fall" → ["g"], "in vacuum" with Coulomb → ["epsilon_0"],
  "universal gravitation" → ["G"]. Do NOT list constants whose value is
  already given explicitly in the problem.

DOMAINS
- likely_domains: list 1-3 domains, using EXACT spelling, from this set that
  the problem's physics involves: {domains_text}
  This is used only to reduce noise in a later step, never to exclude
  anything outright — but if you're unsure between two domains, include
  both rather than guessing narrowly.

THE UNKNOWN
- unknown: this is whatever the question is ULTIMATELY asking to find,
  calculate, or determine — usually signaled by explicit phrasing
  ("find X", "calculate X", "what is X", "determine X"), and that phrasing
  is often at the very END of the question, after several sentences of
  setup describing OTHER quantities. The unknown is NOT necessarily the
  quantity that appears most often or is most elaborately described —
  a question can spend most of its words on intermediate details while
  asking for something else entirely at the end.
  Example: "A body of density 8000 kg/m³ and volume 0.5 m³ accelerates
  from 10 m/s to 30 m/s over 40 m. Find the net force." — even though
  velocities and displacement dominate the sentence, the explicit ask is
  for force (F), NOT acceleration. Acceleration is an intermediate
  quantity needed to get there, not the unknown.

SEARCH QUERY (CONCEPT-LEVEL — important)
- search_query: the conceptual identity of the equation that solves this
  problem, named at the physics-concept level, NOT a description of the
  question's keywords or symbols.

  The search_query is matched against equation descriptions that are
  themselves concept-level identifiers (e.g. "Newton's Second Law of
  Motion", "Archimedes' Principle for the buoyant force",
  "Time-Free Kinematic Relation", "Coulomb's Law", "Photon Energy
  (Planck-Einstein Relation)"). To make matching work, write the
  search_query in the same register.

  Read the question's STORY: what's moving, in what context (in a fluid?
  in a circuit? in vacuum? near Earth? in a magnetic field?), and what's
  being asked. Then NAME the physics concept that would solve it.

  GOOD examples:
    Question: "A body of density 8000 kg/m^3 and volume 0.5 m^3 accelerates
              from 10 m/s to 30 m/s over 40 m. Find the net force."
      → "Newton's second law applied to find net force on a body that is
         undergoing acceleration"
    (NOT "find force given density volume velocity displacement" —
     that's keyword-level, will mis-rank against equations that happen to
     contain density and volume.)

    Question: "A block is raised to a height of 5 m and released, falling
              freely under gravity. Find the velocity just before impact."
      → "time-free kinematic relation for a body falling freely from a
         height under constant gravitational acceleration"
    (NOT "find velocity given height and free fall" — too keyword-y.)

    Question: "A 50 ohm resistor has a potential difference of 10 V across
              it. Determine the power dissipated."
      → "electrical power dissipated by a resistor in a circuit with known
         voltage and current"

    Question: "Two point charges of 2 microC and 3 microC are separated by
              10 cm in vacuum. Find the force between them."
      → "Coulomb's law for the electrostatic force between two point
         charges separated by a distance"

  BAD pattern to avoid: listing the given variables or symbols. This
  makes the search query keyword-shaped, which causes the retrieval to
  match by surface variable-overlap rather than by physics concept. A
  question that gives density and volume to compute mass for use in F=ma
  must not produce a search query mentioning "density and volume" —
  that would mis-match against buoyancy equations.

  Style: one sentence, no numbers, name the physics concept by its
  standard name when there is one, otherwise describe the conceptual
  mechanism. Length: 10-25 words.

OUTPUT
- Output ONLY the JSON object, nothing else.
"""


def _normalize_given_to_si(given: dict) -> dict:
    """
    v7.1.11: deterministically convert given values to SI in CODE, not in the
    LLM's head.

    Stage 1's prompt asks the model to output SI units, but the model (esp.
    the 7B) is unreliable about it — it left "q1: value 2, unit 'μC'" instead
    of "2e-6 C", making SymPy compute with 2 coulombs and producing an answer
    off by 10^12. Unit conversion is deterministic arithmetic, so it belongs
    in code, and doing it here also honors the design principle that the LLM
    should not be the one computing.

    This scans each given's unit string for:
      1. An SI prefix on the leading symbol (μ, m, k, c, n, p, G, M, etc.)
      2. A known non-SI unit with a fixed factor (g→kg, cm→m, km→m, etc.)
    and multiplies the value by the corresponding factor, rewriting the unit
    to its SI base. It is GENERIC — it works for every prefixed/compound unit
    and every question, not any specific case.

    Conservative by design: if a unit isn't recognized, the value is left
    UNTOUCHED (no guessing). Constants already injected in SI (g, epsilon_0,
    etc.) have plain SI units and are unaffected. Only the leading prefix of
    simple units is handled; complex compound units (e.g. 'g/cm^3') are
    handled by an explicit table where they're common in JEE/NEET.
    """
    # SI prefix → factor. Note 'm' is ambiguous (milli vs metre); handled
    # specially below so 'm' alone (metre) is NOT scaled.
    SI_PREFIX = {
        "Y": 1e24, "Z": 1e21, "E": 1e18, "P": 1e15, "T": 1e12, "G": 1e9,
        "M": 1e6,  "k": 1e3,  "h": 1e2,  "da": 1e1,
        "d": 1e-1, "c": 1e-2, "m": 1e-3, "u": 1e-6, "μ": 1e-6, "µ": 1e-6,
        "n": 1e-9, "p": 1e-12, "f": 1e-15, "a": 1e-18,
    }
    # Base units we recognize as the tail after a prefix.
    BASE_UNITS = {
        "m", "g", "s", "A", "C", "V", "N", "J", "W", "Pa", "Hz", "T",
        "F", "H", "ohm", "Ω", "mol", "eV", "Wb", "rad",
    }
    # Explicit whole-unit conversions (non-prefix or compound). value*factor,
    # and the SI unit the value becomes.
    EXPLICIT = {
        "g":      (1e-3, "kg"),       # gram → kilogram
        "g/cm^3": (1e3,  "kg/m^3"),   # common density unit
        "g/cm³":  (1e3,  "kg/m^3"),
        "km":     (1e3,  "m"),
        "cm":     (1e-2, "m"),
        "mm":     (1e-3, "m"),
        "μm":     (1e-6, "m"), "um": (1e-6, "m"),
        "nm":     (1e-9, "m"),
        "km/h":   (1/3.6, "m/s"),
        "kmph":   (1/3.6, "m/s"),
        "minute": (60.0, "s"), "min": (60.0, "s"),
        "hour":   (3600.0, "s"), "h": (3600.0, "s"),
        "day":    (86400.0, "s"),
        "L":      (1e-3, "m^3"), "litre": (1e-3, "m^3"), "liter": (1e-3, "m^3"),
        "mL":     (1e-6, "m^3"), "ml": (1e-6, "m^3"),
        "eV":     (1.602176634e-19, "J"),
        "kWh":    (3.6e6, "J"),
        "atm":    (101325.0, "Pa"),
        "bar":    (1e5, "Pa"),
        "kPa":    (1e3, "Pa"),
        "MPa":    (1e6, "Pa"),
        "GPa":    (1e9, "Pa"),
        "tonne":  (1e3, "kg"), "t": (1e3, "kg"),
    }

    def _convert_one(value, unit):
        if value is None or not isinstance(value, (int, float)):
            return value, unit
        if not unit or not isinstance(unit, str):
            return value, unit
        u = unit.strip()
        # 0) Units that are ALREADY SI base/derived and must never be
        #    decomposed by the prefix matcher. 'kg' is the classic trap: it
        #    looks like kilo+gram but is the SI base unit of mass. Likewise
        #    'cd', 'mol', 'rad' etc. shouldn't be split.
        SI_PROTECTED = {"kg", "mol", "cd", "rad", "sr", "Pa", "Wb", "Hz"}
        if u in SI_PROTECTED:
            return value, unit
        # 1) Exact explicit-table match first (handles compound + ambiguous).
        if u in EXPLICIT:
            f, si = EXPLICIT[u]
            return value * f, si
        # 2) Prefix + base unit (e.g. μC, kPa-not-here, nm-handled-above, mA).
        #    Try the longest matching prefix so 'da' wins over 'd'.
        for plen in (2, 1):
            if len(u) > plen:
                pre, tail = u[:plen], u[plen:]
                if pre in SI_PREFIX and tail in BASE_UNITS:
                    # Guard the milli/metre ambiguity: 'm' as a *prefix* is
                    # only valid when there's a real base unit after it, which
                    # tail in BASE_UNITS already guarantees (e.g. 'mA', 'mS').
                    # Bare 'm' (metre) has no tail, so it never reaches here.
                    return value * SI_PREFIX[pre], tail
        # 3) Unknown unit: leave untouched (no guessing).
        return value, unit

    out = {}
    for sym, meta in given.items():
        if not isinstance(meta, dict):
            out[sym] = meta
            continue
        new_meta = dict(meta)
        nv, nu = _convert_one(meta.get("value"), meta.get("unit"))
        if nv != meta.get("value"):
            new_meta["value"] = nv
            new_meta["unit"] = nu
            new_meta["_si_converted_from"] = meta.get("unit")
        out[sym] = new_meta
    return out


def parse_question(question: str, valid_domains: set[str] | None = None) -> dict:
    """
    Returns dict: {given, unknown, implicit_constants, likely_domains}
    given: {symbol: {value, unit, name, dimension}}
    unknown: {symbol, name, unit, dimension}
    implicit_constants: [symbol, ...]
    likely_domains: [domain, ...]   -- used downstream only as an optional
                                        narrowing hint with a guaranteed
                                        fallback; never a hard exclusion.
    """
    system = _build_parse_system(valid_domains or set())
    log("stage1_entry", question=question, n_valid_domains=len(valid_domains or set()))
    raw = _call(MODEL_FAST, system, question, stage="stage1_parse")
    try:
        parsed = _extract_json(raw)
    except (json.JSONDecodeError, ValueError) as e:
        log_error("stage1_parse_failed", exc=e, raw_preview=raw[:600])
        raise ValueError(f"Stage 1 parse failed. Raw output:\n{raw}\nError: {e}")

    # v7.1.9: empty-givens retry for weaker local models.
    # Small instruct models (Qwen-3B etc.) sometimes return "given": {} even
    # when the question plainly contains numeric values — they describe the
    # unknown correctly but skip extracting the inputs. With no givens, no
    # chain can resolve. If the question text contains digits but the model
    # extracted zero givens, retry ONCE with a sharpened instruction. This
    # never fires when extraction worked, so it costs nothing on the happy
    # path. (Genuinely underspecified questions like "Find the velocity of
    # the object." have no digits, so they correctly do NOT trigger a retry
    # and remain empty — preserving the anti-hallucination behavior.)
    question_has_numbers = bool(re.search(r"\d", question))
    if question_has_numbers and not parsed.get("given"):
        log("stage1_empty_givens_retry", question_preview=question[:120])
        retry_suffix = (
            "\n\nCRITICAL: The question above contains explicit numeric values. "
            "You MUST extract every numeric quantity into the \"given\" object — "
            "do NOT return an empty \"given\". For each number in the question, "
            "create an entry with its symbol, value, unit, name, and dimension. "
            "If the question says 'starts from rest' or 'dropped', also add the "
            "implied u=0. Return the SAME JSON schema, now with \"given\" populated."
        )
        raw2 = _call(MODEL_FAST, system, question + retry_suffix,
                     stage="stage1_parse_retry")
        try:
            parsed2 = _extract_json(raw2)
            # Only accept the retry if it actually produced givens; otherwise
            # keep the original (don't let a worse retry overwrite).
            if parsed2.get("given"):
                # Preserve the original unknown if the retry mangled it.
                parsed2.setdefault("unknown", parsed.get("unknown", {}))
                parsed = parsed2
        except (json.JSONDecodeError, ValueError) as e:
            log_error("stage1_parse_retry_failed", exc=e, raw_preview=raw2[:600])
            # Fall through with the original parse.

    # v7.1.11: deterministically normalize given values to SI in code. The
    # model is unreliable about converting prefixed units (it left μC as 2
    # instead of 2e-6). This runs BEFORE constant injection so the injected
    # constants (already in SI) are untouched. Generic across all units.
    if parsed.get("given"):
        before = {s: (m.get("value"), m.get("unit")) for s, m in parsed["given"].items()
                  if isinstance(m, dict)}
        parsed["given"] = _normalize_given_to_si(parsed["given"])
        converted = {
            s: {"from": f"{before[s][0]} {before[s][1]}",
                "to": f"{m.get('value')} {m.get('unit')}"}
            for s, m in parsed["given"].items()
            if isinstance(m, dict) and m.get("_si_converted_from")
        }
        if converted:
            log("stage1_si_normalized", conversions=converted)

    # Inject implicit constants the LLM flagged based on scenario cues
    # (e.g. "free fall" -> g, "in vacuum" -> epsilon_0). This is the ONLY
    # path by which a context-dependent constant like g enters `given` —
    # there is no blanket default. True universal constants (c, G, epsilon_0,
    # etc.) don't need this step at all: frontier_resolver excludes every
    # symbol in PHYSICAL_CONSTANTS from ever being treated as "needed",
    # so they're safe regardless of whether they appear here.
    for sym in parsed.get("implicit_constants", []):
        if sym not in parsed.get("given", {}) and sym in IMPLICIT_CONSTANTS_CATALOG:
            cat = IMPLICIT_CONSTANTS_CATALOG[sym]
            parsed["given"][sym] = {
                "value":     cat["value"],
                "unit":      cat["unit"],
                "name":      cat["name"],
                "dimension": cat["dimension"],
            }

    parsed.setdefault("likely_domains", [])
    parsed.setdefault("search_query", "")
    log("stage1_parsed",
        n_given=len(parsed.get("given", {})),
        given_symbols=list(parsed.get("given", {}).keys()),
        unknown_symbol=parsed.get("unknown", {}).get("symbol"),
        implicit_constants=parsed.get("implicit_constants", []),
        likely_domains=parsed.get("likely_domains", []),
        search_query=parsed.get("search_query", ""))
    return parsed


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Batched round selector
# ═══════════════════════════════════════════════════════════════════════════════

ROUND_SELECT_SYSTEM = """
You are a physics reasoning engine for JEE/NEET competitive exam problems.

Your ONLY job: for each "needed quantity", select the CONCEPTUALLY CORRECT
equation from its candidate list for THIS specific problem.

You are NOT computing anything. SymPy handles all arithmetic.
Your decisions are purely about which equation describes what is PHYSICALLY
HAPPENING in this problem.

## What you see per candidate

  - id          : the equation's stable identifier (use this in chosen_eq_id)
  - equation    : the symbolic form, e.g. "F = m*a"
  - concept     : a short identifier of WHAT THIS EQUATION IS (e.g. "Newton's
                  Second Law of Motion", "Archimedes' Principle (Buoyant
                  Force)", "Time-Free Kinematic Relation"). This is the
                  equation's physics identity — use it to tell candidates
                  apart even when their symbols look similar.
  - variables   : the equation's variables (other than what's already known)
                  with their physical names and units.

You have physics knowledge. The concept name and the equation form,
combined with what each variable means, are enough for you to reason about
which equation fits.

## Decision principle 1: pick by CONCEPT, not by surface-symbol match

The concept name tells you what the equation IS. The question's story tells
you what physics is happening. Pick the equation whose concept matches the
story.

  - A body accelerating in air or on a surface → "Newton's Second Law of
    Motion" (F = m*a). The body's mass and acceleration may not appear
    directly as numbers in the question — that's fine, see Decision Principle 2.
  - A body submerged in a fluid, with apparent weight or floating → "Archimedes'
    Principle" (F = rho*V*g).
  - Two point charges attracting/repelling → "Coulomb's Law".
  - A photon's energy from frequency or wavelength → "Photon Energy
    (Planck-Einstein Relation)".

Equations from different concepts can share variables. F = m*a and
F = rho*V*g both have F. Their physics is completely different. The concept
name disambiguates.

## Decision principle 2: derivability, not direct match

DO NOT reject an equation because its variables don't appear in the question's
given values.

The pipeline solves problems in CHAINS. An equation is the right pick if its
needed variables can be derived from what's known — directly or through other
equations in subsequent rounds.

Examples:

  Question gives density (rho) and volume (V); asks for force.
  Candidate equations for F include "Newton's Second Law" (F = m*a).
  F = m*a needs m and a — neither is in the question's given values.
  But m can be derived from rho and V via Density = m/V.
  And a can be derived from kinematics (e.g., v^2 = u^2 + 2*a*s) if the
  question also gives initial velocity, final velocity, displacement.
  So F = m*a IS the right pick. The downstream rounds will resolve m and a.

  Question gives initial velocity, acceleration, and time; asks for kinetic energy.
  Candidate equations for K include K = (1/2)*m*v^2.
  v is not directly in the question, but it's derivable from v = u + a*t.
  K = (1/2)*m*v^2 IS the right pick. Round 1 will resolve v.

When in doubt, ask: "Is there a plausible physics chain from what the
question gives to what this equation needs?" If yes, this is a valid pick.

## Decision principle 3: don't pick by "most variables known"

An equation that happens to have many of its variables already known is NOT
automatically the right pick. The CONCEPT must match the physics of the
question. Example: in a kinematics question, F = rho*V*g (buoyancy) might
have rho and V already known — but the question isn't about a submerged body,
so it's the wrong concept, full stop.

## Decision principle 4: conditions

If a candidate's conditions are clearly violated (e.g., "small-angle
approximation" for a wide-swing pendulum), flag it via conditions_concern.
Pick the next-best concept-matching candidate when conditions are violated.

## Decision principle 5: defer or none

If a needed quantity will appear as a BYPRODUCT of an equation you're
choosing for ANOTHER frontier item this round, say "defer".

If NO candidate's concept fits the question's physics, say "decision": "none".
Do not force a wrong pick. Picking a clearly-wrong equation is worse than
admitting no candidate fits.

## Decision principle 6: respect the SOLUTION SO FAR (the working memory)

The prompt may show a "SOLVING FOR (ultimate goal)" line, a "SOLUTION SO FAR"
list of equations already chosen, and a "SYMBOLS ALREADY BEING SOLVED" line.
This is your working memory for the whole problem — USE IT.

  - You are part of a multi-step solve building toward the ULTIMATE GOAL. Keep
    that goal in mind: an equation you pick for a sub-variable must not work
    AGAINST reaching the goal.
  - NEVER pick a candidate that RE-INTRODUCES a symbol already in "SYMBOLS
    ALREADY BEING SOLVED" if doing so would create a conflicting/circular
    constraint. Classic trap: you are ultimately solving for F via F = m*a
    (already chosen), and you now need m. A candidate F = m*g (weight) DOES
    contain m — but picking it re-introduces F (your goal) AND forces a = g,
    contradicting the a you are solving separately. REJECT it. Choose an
    equation that gives the needed symbol WITHOUT dragging the goal or an
    already-solved symbol back in — e.g. Density = m/V to get m from rho and V.
  - In short: the right equation for a sub-variable resolves THAT variable
    from known/derivable quantities, and does NOT re-derive the goal or any
    symbol already being computed elsewhere. If a candidate would, treat it as
    a wrong concept for THIS step and pick another (or say "none").

## Decision principle 7: answer ONLY about the symbol you are asked

You will be asked about ONE specific "needed_symbol" at a time (shown in
NEEDED QUANTITIES THIS ROUND). Your "needed_symbol" in the response MUST be
exactly that symbol. Do NOT switch to talking about a different symbol because
it feels more central to the problem.

WRONG behavior to avoid: you are asked to find the equation for "v" (velocity),
but you respond about "a" (acceleration) because a = F/m feels like the "real"
answer. That is an error — you were asked about v, so answer about v. If the
equation you picked for an earlier symbol needs v as one of its inputs, then v
genuinely IS needed, and your job right now is to pick the equation that
produces v (e.g. v = u + a*t), NOT to re-derive the earlier symbol.

You are also NOT deriving or computing values in your reasoning. Do not write
things like "a = F/m = 20/5 = 4". That is SymPy's job. Your reason should name
the CONCEPT and state that the equation's inputs are obtainable — nothing more.
Stay a concept-matcher, not a calculator.

## How candidates were surfaced

Each candidate has a "landing_source" field — informational, not a ranking:
  - "symbol": contains the needed symbol (most common path).
  - "semantic": rag_text matched the question scenario via vector search.
  - "both": surfaced by both routes. Strong signal, still pick by physics.


## Response format — ONLY valid JSON, no prose outside it

CRITICAL JSON RULE: In the "reason" field, write PLAIN TEXT only. Do NOT use
LaTeX or backslash math notation. Write "v^2 = u^2 + 2*a*s" not "\\( v^2 =
u^2 + 2a\\Delta s \\)". Write "cos(theta)" not "\\cos(\\theta)". Backslashes
break JSON parsing. Use plain ASCII: ^ for powers, * for multiply, spell out
Greek letters (theta, Delta, omega).

{
  "selections": [
    {
      "needed_symbol": "<symbol>",
      "decision": "pick",
      "chosen_eq_id": "<equation id from the candidates>",
      "reason": "<one paragraph, PLAIN TEXT no LaTeX, explaining why THIS concept fits the physics scenario, AND why the equation's other variables are derivable from what's given>",
      "conditions_concern": "<condition text if a stated condition may not hold, else null>"
    }
  ]
}

For deferred items: {"needed_symbol": "...", "decision": "defer", "reason": "..."}
For no-fit items:   {"needed_symbol": "...", "decision": "none",  "reason": "..."}
"""


def _extract_concept(eq: dict) -> str:
    """
    Get the equation's concept name. Order:
      1. node['concept_name'] if apply_exemplars_only.py wrote it
      2. first sentence of rag_text, split on ':' or '.'
      3. empty string

    The concept name is the equation's unique physics identifier — the
    short label like "Newton's Second Law of Motion" that lets the LLM
    disambiguate candidates without needing the full rag_text.
    """
    concept = eq.get("concept_name", "")
    if concept:
        return concept
    rag = eq.get("rag_text", "")
    if not rag:
        return ""
    # Our hand-authored rag_texts open with "<Concept Name>: ..." or
    # "<Concept Name>. ...". Take everything up to the first ':' or first
    # '. ' (period + space) as the concept name.
    for sep in (":", ". "):
        idx = rag.find(sep)
        if idx > 0 and idx < 120:  # don't take whole-paragraph as concept
            return rag[:idx].strip()
    return rag[:80]  # last-resort cap


def _format_candidate(eq: dict, known_symbols: set[str] | None = None) -> dict:
    """
    Compact representation of an equation for the Stage 2 LLM prompt.

    v7.1.4 change: the full rag_text is no longer included. With v7.1.1's
    concept-level rag_texts (~600 chars each), surfacing all of them across
    5 candidates per item × 2 items per round was a ~6000-char token budget
    just for descriptions — which the 8B fast model couldn't reliably handle.
    The user's observation: the LLM already has physics knowledge; we don't
    need to teach it Newton's Second Law in every prompt. We just need to
    name the concept clearly enough to disambiguate from neighbors.

    What we ship instead is the concept_name — a short label like "Newton's
    Second Law of Motion" — plus the equation form, plus variable
    name+unit map. The model uses its physics training to fill in the rest.

    `known_symbols`: variables already in "ALREADY KNOWN". Omitted from the
    candidate's variables map (no point re-listing what the LLM has just
    seen in the available state).
    """
    known_symbols = known_symbols or set()
    out = {
        "id":         eq["id"],
        "equation":   eq["equation_str"],
        "concept":    _extract_concept(eq),
        "variables":  {
            sym: {"name": meta["name"], "unit": meta["unit"]}
            for sym, meta in eq["variables"].items()
            if sym not in known_symbols
        },
    }
    # Conditions matter when they restrict applicability (small-angle, etc.)
    # We still surface them but only when non-empty.
    conds = eq.get("conditions", [])
    if conds:
        out["conditions"] = conds[:2]
    if eq.get("landing_source"):
        out["landing_source"] = eq["landing_source"]
    return out


def estimate_round_tokens(round_data: list[dict]) -> int:
    """
    Rough token estimate (chars // 4) for one round's candidate payload,
    built the same way call_round_selector formats it. Used by
    frontier_resolver as a safety valve: if a batched round's estimate
    exceeds config.MAX_CANDIDATES_TOKENS_PER_ROUND, it splits the round
    into sequential single-symbol calls instead of risking an oversized
    request (this is what crashed with a real 413 on llama-3.1-8b-instant).
    """
    sections = []
    for rd in round_data:
        fi = rd["frontier_item"]
        sections.append({
            "symbol": fi.symbol, "name": fi.name,
            "unit": fi.unit, "dimension": fi.dimension,
            "candidates": [_format_candidate(eq) for eq in rd["candidates"]],
        })
    return len(json.dumps(sections, separators=(",", ":"))) // 4


def call_round_selector(
    question:    str,
    available:   dict[str, dict],   # {symbol: {value,unit,name,dimension}}
    round_data:  list[dict],        # [{frontier_item, candidates}]
    round_num:   int = 0,
    solve_context: dict | None = None,
) -> list[dict]:
    """
    Dispatches to either a single batched LLM call or a sequence of per-item
    calls, depending on `STAGE2_BATCH_MODE` (env var, defaults to "auto").

    v7.1.3 change: previously this function always made one LLM call covering
    every frontier item in the round. That worked when the rag_texts were
    templated 120-char boilerplate (v6), but with v7.1's ~600-char concept-
    level rag_texts the prompt becomes large for multi-item rounds and the
    fast 8B model starts dropping items off the back of its response —
    producing `llm_omitted_item` events specifically on chained problems
    (F=ma where m comes from rho*V and a comes from kinematics).

    Splitting per-item keeps each LLM call focused on one symbol; the model
    can't drop an item because there's only one to address. Round 0 is
    unaffected (it almost always has one item).

    Returns one selection dict per frontier item, same shape as before:
      {frontier_item, chosen_eq, reason, conditions_concern, deferred,
       _candidates, fallback_used}
    """
    # Decide mode
    mode = STAGE2_BATCH_MODE
    if mode not in ("auto", "all", "single"):
        # Fall back to safe default on bad config
        mode = "auto"
    if mode == "auto":
        mode = "single" if len(round_data) > 1 else "all"

    log("stage2_dispatch",
        round_num=round_num,
        n_items=len(round_data),
        mode=mode,
        items=[rd["frontier_item"].symbol for rd in round_data])

    if mode == "single" and len(round_data) > 1:
        # One LLM call per frontier item.
        results = []
        for i, rd in enumerate(round_data):
            results.extend(
                _round_select_call(question, available, [rd], round_num,
                                   sub_index=i, sub_total=len(round_data),
                                   solve_context=solve_context)
            )
        return results

    # Batched: one LLM call covering everything in round_data.
    return _round_select_call(question, available, round_data, round_num,
                              solve_context=solve_context)


def _round_select_call(
    question:    str,
    available:   dict[str, dict],
    round_data:  list[dict],
    round_num:   int = 0,
    sub_index:   int | None = None,
    sub_total:   int | None = None,
    solve_context: dict | None = None,
) -> list[dict]:
    """
    The actual LLM-call helper. Builds the prompt, calls the model, parses
    the response, and produces selection dicts for every item in
    `round_data` (which may be one item or many).

    When `sub_index`/`sub_total` are set, the log lines carry them so the
    user can correlate per-item calls back to the round they belong to.

    v7.2.3: `solve_context` carries a SMALL agentic working memory so each
    sub-decision is aware of the whole solve, not amnesiac. It holds the
    ultimate goal symbol, the equations already committed (and what each
    solves), and which symbols are therefore already being handled. This lets
    the model REJECT a candidate by meaning when it conflicts with the plan —
    e.g. choosing F=m*g to get mass while F is the ultimate goal and a is
    already being solved separately would re-introduce F and pin a=g, a
    contradiction the model can now SEE and avoid. Only committed (valid)
    steps go in — not dead-ends or rejected attempts — to keep it tiny.
    """
    # v7.1.5: Filter constants in the ALREADY KNOWN display.
    # Without this filter, every Stage 2 prompt listed all 10 universal
    # constants (pi, c, G, h_planck, k_B, R_g, NA, epsilon_0, mu_0, e_charge)
    # even when the question was pure kinematics. That's ~300 bytes of
    # noise per prompt and contributes to TPM exhaustion on the free tier.
    # We still keep constants in `available` for SymPy substitution
    # downstream — we just don't show irrelevant ones to the LLM.
    from config import UNIVERSAL_CONSTANTS
    # The set of variables any candidate in this round actually uses
    candidate_var_symbols: set = set()
    for rd in round_data:
        for eq in rd["candidates"]:
            candidate_var_symbols.update(eq.get("variables", {}).keys())

    avail_lines = []
    known_symbols = set()
    for sym, meta in available.items():
        val = meta.get("value")
        if val is None:
            continue
        # If this is a universal constant AND no candidate equation uses it,
        # skip it. Question-given values are always shown.
        if sym in UNIVERSAL_CONSTANTS and sym not in candidate_var_symbols:
            known_symbols.add(sym)  # still mark as known for the candidate var filter
            continue
        avail_lines.append(
            f"  {sym} ({meta.get('name', sym)}): {val} {meta.get('unit', '')}"
        )
        known_symbols.add(sym)

    # Build needed quantities section
    needed_sections = []
    for rd in round_data:
        fi = rd["frontier_item"]
        cands = rd["candidates"]
        section = {
            "symbol":     fi.symbol,
            "name":       fi.name,
            "unit":       fi.unit,
            "dimension":  fi.dimension,
            "candidates": [_format_candidate(eq, known_symbols) for eq in cands],
        }
        needed_sections.append(section)

    # v7.2.3: build the compact working-memory block from solve_context.
    # Small by design — goal, the committed steps, and what's already handled.
    context_block = ""
    if solve_context:
        goal_sym  = solve_context.get("goal_symbol", "")
        goal_name = solve_context.get("goal_name", "")
        steps     = solve_context.get("chosen_steps", [])  # [{symbol,eq_str,concept}]
        being_solved = solve_context.get("being_solved", [])  # [symbols]
        lines = []
        if goal_sym:
            lines.append(f"SOLVING FOR (ultimate goal): {goal_sym}"
                         + (f" ({goal_name})" if goal_name else ""))
        if steps:
            lines.append("SOLUTION SO FAR (already chosen — do not re-derive or contradict these):")
            for s in steps:
                lines.append(f"  • {s['symbol']} ← {s['eq_str']}")
        if being_solved:
            lines.append("SYMBOLS ALREADY BEING SOLVED (do NOT pick an equation that "
                         "re-introduces or conflicts with these): "
                         + ", ".join(being_solved))
        if lines:
            context_block = "\n".join(lines) + "\n\n"

    user_prompt = (
        f"ORIGINAL QUESTION:\n{question}\n\n"
        + context_block
        + f"ALREADY KNOWN:\n" + ("\n".join(avail_lines) or "  (none yet)") + "\n\n"
        f"NEEDED QUANTITIES THIS ROUND (round {round_num}):\n"
        + json.dumps(needed_sections, separators=(",", ":"))
    )

    # v7.1.2/v7.1.3: log the round entry — what items are being asked for,
    # what candidates were given for each, and what landing route surfaced
    # them. With v7.1.3's per-item batching mode, this may fire multiple
    # times per logical round, with sub_index/sub_total set so you can
    # correlate them.
    log("stage2_round_entry",
        round_num=round_num,
        sub_index=sub_index,
        sub_total=sub_total,
        items=[{
            "symbol":           rd["frontier_item"].symbol,
            "name":             rd["frontier_item"].name,
            "n_candidates":     len(rd["candidates"]),
            "candidate_ids":    [c["id"] for c in rd["candidates"]],
            "landing_sources":  [c.get("landing_source", "?") for c in rd["candidates"]],
        } for rd in round_data],
        n_known=len(known_symbols))

    raw = _call(STAGE2_MODEL, ROUND_SELECT_SYSTEM, user_prompt, stage="stage2_round_select")

    try:
        parsed = _extract_json(raw)
        # parsed may be a dict (normal), a bare selection dict, or a bare list.
        if isinstance(parsed, list):
            selections_raw = parsed
        else:
            selections_raw = parsed.get("selections", [])
            # v7.2.2: lenient fallback for the 7B's inconsistent output shape.
            # The model sometimes returns a BARE selection object instead of
            # wrapping it in {"selections": [...]}, e.g. {"needed_symbol": "a",
            # "decision": "pick", "chosen_eq_id": "..."}. Without this,
            # parsed.get("selections") is empty and a CORRECT pick is discarded
            # as "omitted". If selections is missing/empty but the object itself
            # carries selection fields, treat the whole object as a single
            # selection. Generic — recovers any single correct pick that lost
            # its array wrapper, not a specific case.
            if not selections_raw and (
                "needed_symbol" in parsed or "chosen_eq_id" in parsed or "decision" in parsed
            ):
                selections_raw = [parsed]
                log("stage2_bare_selection_recovered", round_num=round_num,
                    needed_symbol=parsed.get("needed_symbol"))
    except (json.JSONDecodeError, ValueError) as parse_err:
        # Fallback: return empty selections (pipeline will treat as unresolvable)
        log_error("stage2_json_parse_failed",
                  exc=parse_err, round_num=round_num,
                  raw_preview=raw[:600])
        return [
            {"frontier_item": rd["frontier_item"], "chosen_eq": None,
             "reason": f"LLM parse error: {raw[:200]}", "deferred": False,
             "conditions_concern": None, "_candidates": rd["candidates"]}
            for rd in round_data
        ]

    # Map selections back to frontier items + candidate dicts
    # Build lookup by symbol
    rd_by_symbol = {rd["frontier_item"].symbol: rd for rd in round_data}
    result = []

    # v7.1.2: log what symbols the LLM actually addressed vs. what we asked
    # about. This catches the llm_omitted_item case at the source: if the LLM
    # only returned selections for some items, we see exactly which ones.
    addressed = [s.get("needed_symbol") for s in selections_raw]
    asked = [rd["frontier_item"].symbol for rd in round_data]
    # v7.1.5: detect HALLUCINATED symbols — the LLM answered about a symbol
    # we didn't ask about. We saw this in the live log: Round 2 asked for 'u',
    # the LLM returned a selection with needed_symbol="a". The previous code
    # silently treated 'u' as omitted, hiding the actual problem. Now we
    # explicitly log it as a hallucination so the diagnostic is clear.
    asked_set = set(asked)
    hallucinated = [s for s in addressed if s and s not in asked_set]
    if hallucinated:
        log("stage2_hallucinated_symbol",
            round_num=round_num,
            asked_for=asked,
            llm_responded_about=addressed,
            hallucinated=hallucinated)
    log("stage2_llm_selections_received",
        round_num=round_num,
        asked_for=asked,
        addressed=addressed,
        omitted=[s for s in asked if s not in addressed],
        hallucinated=hallucinated,
        n_selections=len(selections_raw))

    for sel in selections_raw:
        sym = sel.get("needed_symbol")
        rd  = rd_by_symbol.get(sym)
        if rd is None:
            continue
        fi        = rd["frontier_item"]
        cands     = rd["candidates"]
        decision  = sel.get("decision")
        deferred  = decision == "defer"
        chosen_eq = None
        # v7: surface this through the result dict so frontier_resolver and
        # decision_log can distinguish "LLM said none fit" from "LLM gave a
        # bad ID we had to fall back from".
        fallback_used = None

        if deferred:
            pass
        elif decision == "none":
            # LLM explicitly says no candidate fits. Surface as no-pick so
            # frontier_resolver fails this round cleanly (and backtracking
            # can fire if applicable) — far better than picking a wrong
            # equation that produces a confident wrong answer.
            chosen_eq = None
            fallback_used = "llm_decision_none"
        else:
            chosen_id = sel.get("chosen_eq_id")
            chosen_eq = next((eq for eq in cands if eq["id"] == chosen_id), None)
            if chosen_eq is None:
                # v7 change: do NOT silently substitute the first candidate.
                # That hid every "LLM hallucinated an ID" event behind what
                # looked like an intentional pick. Surface it as no-pick;
                # frontier_resolver will treat as unresolvable and downstream
                # backtracking can fire.
                fallback_used = (
                    f"llm_invalid_id: got {chosen_id!r}, not in candidates "
                    f"{[c['id'] for c in cands]}"
                )

        result.append({
            "frontier_item":      fi,
            "chosen_eq":          chosen_eq,
            "reason":             sel.get("reason", ""),
            "conditions_concern": sel.get("conditions_concern"),
            "deferred":           deferred,
            "_candidates":        cands,
            "fallback_used":      fallback_used,
        })
        # v7.1.2: log the per-item outcome. Filter logs for
        # event=stage2_item_decision to see what the LLM picked or why it
        # didn't, with the reason text it provided.
        log("stage2_item_decision",
            round_num=round_num,
            symbol=fi.symbol,
            decision=decision,
            chosen_eq_id=(chosen_eq["id"] if chosen_eq else None),
            fallback_used=fallback_used,
            llm_reason=(sel.get("reason", "") or "")[:500],
            candidate_ids=[c["id"] for c in cands])

    # Make sure every frontier item has a result entry. v6 silently picked
    # the first candidate here; v7 marks the item as no-pick + records why,
    # so the decision_log shows it and backtracking can react.
    answered_syms = {r["frontier_item"].symbol for r in result}
    for rd in round_data:
        fi = rd["frontier_item"]
        if fi.symbol not in answered_syms:
            cands = rd["candidates"]
            # v7.1.2: this is the llm_omitted_item path — the LLM returned
            # JSON but didn't include a selection for this symbol. Most
            # common cause: model capacity (8B model giving up) or prompt
            # confusion. Log explicitly so the user sees it.
            log("stage2_item_omitted",
                round_num=round_num,
                symbol=fi.symbol,
                n_candidates=len(cands),
                candidate_ids=[c["id"] for c in cands])
            result.append({
                "frontier_item":      fi,
                "chosen_eq":          None,
                "reason":             "LLM did not address this item — surfaced as no-pick.",
                "conditions_concern": None,
                "deferred":           False,
                "_candidates":        cands,
                "fallback_used":      "llm_omitted_item",
            })

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Trace-based narration
# ═══════════════════════════════════════════════════════════════════════════════

NARRATE_SYSTEM = """
You are an expert JEE/NEET physics teacher explaining a solution to a student.

The algorithm has already determined the correct equations, computed exact
numerical values, and produced a step-by-step substitution trace.
Your job: turn that trace into clear, educational prose.

STRICT RULES:
1. ONLY use numbers, equations, and quantities that appear in the trace
   or final_answer below. NEVER alter a number from the trace, and just as
   importantly, NEVER introduce a number, equation, or computed quantity
   that isn't already there — even if you can work out what it "should"
   be, even if the original question asks for something the trace doesn't
   cover. You are narrating a finished computation, not completing one.
2. Quote each step's "substituted" string and "result" value exactly as
   given, including sign. Do NOT independently re-derive a step's algebra
   to "show the work" — the substitution is already given to you; restate
   it, don't recompute it. If a value is negative (e.g. deceleration,
   opposing direction), keep it negative in your prose — never silently
   restate it as positive.
3. Only add a one-line note about final_answer's symbol if it genuinely
   does NOT seem to match what the question's final sentence asks for.
   If it does match (the normal case), say nothing about this at all —
   do not add a routine "this confirms/matches what was asked" sentence;
   that's noise when there's nothing to flag.
4. For each equation used, explain WHY it is the right physical choice.
5. Briefly mention any rejected alternatives (from decision_log) and why.
6. Add the relevant physics law or principle justifying each step.
7. Write for a Class 11/12 JEE/NEET student — clear English, no jargon overload.
8. End with one sentence summarising the overall strategy.
9. Write flowing numbered steps — no bullet points.
"""


def narrate_from_trace(
    question:     str,
    trace_steps:  list[dict],  # [{equation_str, solving_for, symbolic, substituted, result, unit, reason}]
    decision_log: list[dict],
    final_answer: dict,        # {value_exact, value_float, unit, symbol}
) -> str:
    prompt = f"""Student question:
{question}

Step-by-step substitution trace (DO NOT alter any numbers):
{json.dumps(trace_steps, indent=2)}

Decision log (what was chosen and why, what was available but rejected):
{json.dumps(decision_log, indent=2)}

Final answer: {final_answer.get('value_exact')} {final_answer.get('unit')} ({final_answer.get('value_float')} {final_answer.get('unit')})

Write the explanation now.
"""
    return _call(MODEL_SMART, NARRATE_SYSTEM, prompt, temperature=0.2, stage="stage4_narrate")


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 5 — Distractors (unchanged in spirit)
# ═══════════════════════════════════════════════════════════════════════════════

DISTRACT_SYSTEM = """
You are generating wrong answer options for a JEE/NEET MCQ physics question.
Generate exactly 3 wrong numeric options representing specific student mistakes.

Each wrong option:
- Is a specific number (not a description)
- Comes from applying a concrete mistake to the same problem
- Differs from the correct answer and from each other
- Is physically plausible (correct units, same order of magnitude)

Respond ONLY with valid JSON:
[
  {"value": <number>, "unit": "<unit>", "mistake": "<brief error description>"},
  {"value": <number>, "unit": "<unit>", "mistake": "<brief error description>"},
  {"value": <number>, "unit": "<unit>", "mistake": "<brief error description>"}
]
"""


def generate_distractors(
    question:      str,
    correct_value: float,
    correct_unit:  str,
    chain_nodes:   list[dict],
) -> list[dict]:
    all_mistakes = []
    for node in chain_nodes:
        all_mistakes.extend(node.get("common_mistakes", []))
    mistakes_text = "\n".join(f"- {m}" for m in all_mistakes[:9])

    prompt = (
        f"Problem: {question}\n\n"
        f"Correct answer: {correct_value} {correct_unit}\n\n"
        f"Common student mistakes for equations used:\n{mistakes_text}\n\n"
        f"Generate 3 wrong MCQ options now."
    )
    raw = _call(MODEL_FAST, DISTRACT_SYSTEM, prompt, stage="stage5_distractors")
    try:
        d = _extract_json(raw)
        if isinstance(d, list):
            return d[:3]
    except Exception:
        pass
    return [
        {"value": round(correct_value * 2,  4), "unit": correct_unit, "mistake": "doubled result"},
        {"value": round(correct_value / 2,  4), "unit": correct_unit, "mistake": "halved result"},
        {"value": round(correct_value * 10, 4), "unit": correct_unit, "mistake": "unit conversion error"},
    ]
