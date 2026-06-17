"""
solver/llm_interface.py
All Groq API calls for the 5-stage pipeline.

Stage 1  parse_question()       — extended: gives, target, dimensions, implicit constants
Stage 2  call_round_selector()  — batched conceptual equation selection per round
Stage 4  narrate_from_trace()   — trace-based student-facing narration
Stage 5  generate_distractors() — 3 wrong MCQ options (unchanged)
"""
from __future__ import annotations
import json
import re
from groq import Groq
from config import (
    GROQ_API_KEY, MODEL_FAST, MODEL_SMART, GROQ_TEMPERATURE,
    IMPLICIT_CONSTANTS_CATALOG,
)
from solver.frontier_resolver import FrontierItem

client = Groq(api_key=GROQ_API_KEY)


def _call(model: str, system: str, user: str, temperature: float = GROQ_TEMPERATURE) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()


def _extract_json(text: str) -> dict | list:
    """Strip markdown fences, find the first complete JSON object/array."""
    clean = re.sub(r"```(?:json)?|```", "", text).strip()
    # Sometimes the model adds prose before/after the JSON
    # Try to extract just the JSON portion
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = clean.find(start_char)
        if start != -1:
            # Find the matching closing bracket
            depth = 0
            for i, ch in enumerate(clean[start:], start):
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(clean[start:i+1])
                        except json.JSONDecodeError:
                            break
    return json.loads(clean)


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
  "likely_domains": ["<domain1>", "<domain2>"]
}}

Rules:
- Convert all given values to SI units before outputting (km/h→m/s, g/cm³→kg/m³, etc.)
- Use standard symbols: F=force, m=mass, a=acceleration, v=final velocity,
  u=initial velocity, s=displacement, t=time, rho=density, V=volume,
  g=gravitational acceleration, T=temperature/period, P=pressure/power,
  E=energy/electric field, q=charge, I=current/moment of inertia/impulse,
  r=radius/distance, h=height, n=refractive index, lambda=wavelength
- Dimensional formulas: M=mass, L=length, T=time, A=current, K=temperature.
  Examples: force=MLT-2, velocity=LT-1, acceleration=LT-2, mass=M,
  density=ML-3, energy=ML2T-2, charge=AT, current=A, momentum=MLT-1
- implicit_constants: list ONLY symbols from this catalog that the scenario
  implies WITHOUT the problem stating a numeric value:
{_CONSTANTS_CATALOG_TEXT}
  Examples: "free fall" → ["g"], "in vacuum" with Coulomb → ["epsilon_0"],
  "universal gravitation" → ["G"]. Do NOT list constants whose value is
  already given explicitly in the problem.
- likely_domains: list 1-3 domains, using EXACT spelling, from this set that
  the problem's physics involves: {domains_text}
  This is used only to reduce noise in a later step, never to exclude
  anything outright — but if you're unsure between two domains, include
  both rather than guessing narrowly.
- Output ONLY the JSON object, nothing else.
"""


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
    raw = _call(MODEL_FAST, system, question)
    try:
        parsed = _extract_json(raw)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Stage 1 parse failed. Raw output:\n{raw}\nError: {e}")

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

## Decision rules

1. READ the original question carefully — understand the physical scenario.
2. For each needed quantity, ask:
   - Which candidate equation describes the physics of THIS problem?
   - Does the equation's rag_text match what the scenario is about?
   - Do the equation's conditions hold for this scenario?
3. DO NOT rank candidates by "how many variables are already known."
   An equation that is immediately solvable may still be the WRONG physics.
   Example: if a body accelerates (kinematics), F=m*a is correct even if
   F=rho*V*g (buoyant force) happens to have more known variables — the
   problem is not about fluid submersion.
4. If a candidate's conditions are clearly violated, flag it AND pick the
   next-best candidate instead. Only mark conditions_concern if you are
   still choosing that equation despite the concern.
5. If you believe a needed quantity will appear as a BYPRODUCT of an
   equation you're choosing for ANOTHER frontier item this round, say
   "defer" for it. Use this only when genuinely redundant.

## Response format — ONLY valid JSON, no prose outside it

{
  "selections": [
    {
      "needed_symbol": "<symbol>",
      "decision": "pick",
      "chosen_eq_id": "<equation id from the candidates>",
      "reason": "<one paragraph explaining why THIS equation fits the physical scenario, NOT just 'most variables known'>",
      "conditions_concern": "<condition text if a stated condition may not hold, else null>"
    }
  ]
}

For deferred items: {"needed_symbol": "...", "decision": "defer", "reason": "..."}
"""


def _format_candidate(eq: dict, known_symbols: set[str] | None = None) -> dict:
    """
    Compact but complete representation of an equation for the LLM prompt.
    `known_symbols`: variables already shown in the "ALREADY KNOWN" section —
    excluded here since repeating them in every single candidate is pure
    redundancy, not signal. The symbol being solved for is never in
    known_symbols by construction (it wouldn't be a frontier item otherwise),
    so it always survives this filter.
    """
    rag = eq.get("rag_text", "")
    if len(rag) > 120:
        rag = rag[:117] + "..."
    known_symbols = known_symbols or set()
    return {
        "id":           eq["id"],
        "equation":     eq["equation_str"],
        "description":  rag,
        "conditions":   eq.get("conditions", [])[:2],
        "variables":    {
            sym: {"name": meta["name"], "unit": meta["unit"]}
            for sym, meta in eq["variables"].items()
            if sym not in known_symbols
        },
    }


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
) -> list[dict]:
    """
    One batched LLM call for a full frontier resolution round.
    Returns list of selection dicts, one per frontier item.

    Each selection dict:
      {frontier_item, chosen_eq, reason, conditions_concern, deferred,
       _candidates}
    """
    # Build available summary (omit constants — they're always implicit)
    avail_lines = []
    known_symbols = set()
    for sym, meta in available.items():
        val = meta.get("value")
        if val is not None:
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

    user_prompt = (
        f"ORIGINAL QUESTION:\n{question}\n\n"
        f"ALREADY KNOWN:\n" + ("\n".join(avail_lines) or "  (none yet)") + "\n\n"
        f"NEEDED QUANTITIES THIS ROUND (round {round_num}):\n"
        + json.dumps(needed_sections, separators=(",", ":"))
    )

    raw = _call(MODEL_FAST, ROUND_SELECT_SYSTEM, user_prompt)

    try:
        parsed = _extract_json(raw)
        selections_raw = parsed.get("selections", [])
    except (json.JSONDecodeError, ValueError):
        # Fallback: return empty selections (pipeline will treat as unresolvable)
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

    for sel in selections_raw:
        sym = sel.get("needed_symbol")
        rd  = rd_by_symbol.get(sym)
        if rd is None:
            continue
        fi        = rd["frontier_item"]
        cands     = rd["candidates"]
        deferred  = sel.get("decision") == "defer"
        chosen_eq = None

        if not deferred:
            chosen_id = sel.get("chosen_eq_id")
            chosen_eq = next((eq for eq in cands if eq["id"] == chosen_id), None)
            if chosen_eq is None and cands:
                # LLM gave wrong ID — fall back to first candidate and flag it
                chosen_eq = cands[0]

        result.append({
            "frontier_item":     fi,
            "chosen_eq":         chosen_eq,
            "reason":            sel.get("reason", ""),
            "conditions_concern": sel.get("conditions_concern"),
            "deferred":          deferred,
            "_candidates":       cands,
        })

    # Make sure every frontier item has a result entry (LLM may have skipped some)
    answered_syms = {r["frontier_item"].symbol for r in result}
    for rd in round_data:
        fi = rd["frontier_item"]
        if fi.symbol not in answered_syms:
            cands = rd["candidates"]
            result.append({
                "frontier_item":     fi,
                "chosen_eq":         cands[0] if cands else None,
                "reason":            "LLM did not address this item — using first candidate.",
                "conditions_concern": None,
                "deferred":          False,
                "_candidates":       cands,
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
1. NEVER change any number from the trace — they are exact and correct.
2. For each equation used, explain WHY it is the right physical choice.
3. Briefly mention any rejected alternatives (from decision_log) and why.
4. Add the relevant physics law or principle justifying each step.
5. Write for a Class 11/12 JEE/NEET student — clear English, no jargon overload.
6. End with one sentence summarising the overall strategy.
7. Write flowing numbered steps — no bullet points.
8. Show the substitution exactly as given (exact fractions if present).
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
    return _call(MODEL_SMART, NARRATE_SYSTEM, prompt, temperature=0.2)


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
    raw = _call(MODEL_FAST, DISTRACT_SYSTEM, prompt)
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
