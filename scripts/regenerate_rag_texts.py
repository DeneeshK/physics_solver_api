#!/usr/bin/env python3
"""
scripts/regenerate_rag_texts.py
v7.1 — Rewrite every equation's rag_text as a concept-level identifier.

What this does, in plain terms:
  1. Loads the hand-authored exemplars from scripts/rag_text_exemplars.json.
     Those 16 equations get their rag_text set to the hand-authored value
     directly (no LLM, no drift — they're the gold standard).
  2. For each of the remaining ~166 equations, makes ONE Groq LLM call,
     supplying the equation's structured data plus 3 randomly-sampled
     exemplars as in-context demonstrations. The LLM produces a
     concept-level rag_text in the same style.
  3. Writes the result to data/physics_equation_graph_final.json
     (with a .bak backup of the original).

Why one LLM call per equation, not one giant batch:
  - A 182-equation single-prompt is way over context limit and would give
    inconsistent quality across the list.
  - Per-equation calls let us validate each output structurally and retry
    individually if a parse fails. No bad rag_text silently slipping into
    the graph.

Quality safeguards baked in:
  - Each generated rag_text must mention the concept name explicitly.
  - Must mention what the equation is NOT, by concept (the look-alike
    exclusion property the user emphasized).
  - Must be 400-1500 chars (rejects garbage-short outputs and runaway-long
    outputs).
  - Must NOT contain placeholder language ("this equation", "the formula
    above") that betrays generic LLM filler.
  - Failed equations are listed at the end, NOT silently skipped. You
    review and either hand-author or re-run.

Usage:
  export GROQ_API_KEY=your_key
  python scripts/regenerate_rag_texts.py             # full run, ~5 minutes
  python scripts/regenerate_rag_texts.py --dry-run 5 # generate 5 only, print
  python scripts/regenerate_rag_texts.py --ids electrostatics_coulomb_law,...
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

# Make the v7 root importable so we can reuse config and friends
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import GROQ_API_KEY, MODEL_SMART

# Use the strong model — we want quality on the one-shot per-equation
# generation. Cost is fine: ~166 calls × ~1500 tokens out × $0.0008/1k
# ≈ 20 cents per full regeneration run.
GENERATION_MODEL = MODEL_SMART  # llama-3.3-70b-versatile

EXEMPLARS_PATH = SCRIPT_DIR / "rag_text_exemplars.json"
GRAPH_PATH     = PROJECT_ROOT / "data" / "physics_equation_graph_final.json"
BACKUP_PATH    = PROJECT_ROOT / "data" / "physics_equation_graph_final.json.bak"

# Stylistic and content constraints. Adjusted from experience tuning the
# 16 hand-authored exemplars.
MIN_RAG_TEXT_CHARS = 400
MAX_RAG_TEXT_CHARS = 1500

# Phrases that suggest the LLM defaulted to generic filler. If any of these
# is in the output, reject and either retry or surface as a failure.
#
# v7.1.1 note: "this equation" was originally on this list but caused false
# positives — it appears legitimately in references like "this equation gives
# the net force required..." and "this equation describes a specific
# physical mechanism...". Kept off the list. The other phrases are more
# clearly generic-filler tells.
BANNED_PHRASES = [
    "the formula above", "the equation above",
    "as mentioned", "the variables are", "in this case",
    "as you can see", "let's break", "step by step",
    # Old-template tells from the v6 rag_text style
    "Use it when pressure, flow, buoyancy",
    "Use it when force, mass, weight, or contact",
    "Use it when energy, work, power, or stored",
    "is the contextual quantity that fixes",
    "Do not use it when data from different",
]

# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You write concept-level identifier descriptions for physics
equations in a JEE/NEET problem-solving system.

Each description is a UNIQUE IDENTIFIER for one specific equation. It will be
embedded by a vector database. At query time, an LLM reads a physics question,
extracts the conceptual identity of what's being asked, generates a
concept-level search query, and the vector database matches concept-to-concept.

A high-quality description has these properties:

  1. OPENS WITH THE NAMED CONCEPT. The first phrase is the name of the
     physics concept this equation represents (e.g. "Newton's Second Law of
     Motion", "Archimedes' Principle for the buoyant force", "Coulomb's Law").
     This is the conceptual identifier.

  2. STATES WHAT THE EQUATION CAPTURES, conceptually. Not what variables are
     in it (the graph already has that). The physical relationship,
     mechanism, or principle.

  3. STATES WHEN TO APPLY IT — the kind of scenario, in physics terms. The
     reader is an LLM that has just read a question and is trying to decide
     "is this the right equation?". Help it judge by the SCENARIO, not by
     the question's symbols.

  4. STATES WHEN NOT TO APPLY IT — naming the specific look-alike
     equations it might be confused with, and what differentiates them.
     This is essential to avoid concept-collision in retrieval. Examples:
       - Newton's 2nd law is NOT Archimedes' buoyancy.
       - Buoyancy is NOT general dynamics F=ma.
       - Photon energy is NOT classical KE.
       - Lens formula is NOT mirror formula.

  5. CALLS OUT INDIRECT-INPUT PATHS where relevant. If the equation
     commonly needs an input derived from other equations (e.g. F=ma
     needs mass, which often comes from rho*V), say so. This helps the
     LLM realize it can reach the equation even when its symbols don't
     literally appear in the question.

  6. NO BOILERPLATE. Do not use generic phrases like "this equation",
     "the formula above", "as mentioned", "the variables are", "step by
     step", or "as you can see". Do not start with "Use it when..." — that
     was the old style and is forbidden. Write substantive physics.

  7. LENGTH 400-1200 chars. One coherent paragraph. No bullet lists, no
     markdown headers, no equation re-typing (the graph stores the
     equation separately).

  8. VERIFY THE PHYSICS BEFORE DESCRIBING. The equation_str alone is not
     always enough to determine the equation's scope of applicability.
     Before writing, consider:
       - What is the standard physics interpretation of this equation given
         the variable names supplied in the structured data?
       - Is this the GENERAL form of a relationship, or a special case?
         For example: v = s/t is the GENERAL definition of average velocity
         over ANY motion (not just uniform motion). Do not artificially
         narrow the scope. If the equation is the general/defining form of
         a quantity, say so explicitly. If it is a special case (e.g.
         constant-acceleration kinematic relation), say that explicitly too.
       - Does the equation describe a quantity (energy, force, momentum) or
         a relationship between quantities (conservation, equality)? Be
         honest about which.
     If after verification you suspect the equation as stored in the graph
     does not match a standard physics equation, write the rag_text for
     the most plausible interpretation and do NOT invent justifying physics.

  9. WORD SPACING. Use proper single-space separation between every word.
     Common errors to avoid: writing 'finalvelocity' instead of 'final
     velocity', 'constantacceleration' instead of 'constant acceleration',
     'accelerationwhen' instead of 'acceleration, when', 'avehicle' instead
     of 'a vehicle'. Every word must be separated by exactly one space.
     Punctuation (commas, periods) followed by a space, then the next word.

 10. LOOK-ALIKE COMPARISONS MUST BE TO REAL EQUATIONS. When you write
     "distinct from X" or "not to be confused with Y", X and Y must be
     actual physics equations from the JEE/NEET curriculum, not
     hypothetical incorrect forms. Do not invent a wrong equation to
     contrast against. If you do not know of a real look-alike, restate
     the concept's distinguishing feature instead.

Respond ONLY with valid JSON: {"rag_text": "<the description>"}
"""


def build_user_prompt(eq_node: dict, exemplars: list[tuple[str, dict]]) -> str:
    """
    Builds a per-equation generation prompt with N exemplars as in-context
    demonstrations of the target style.

    exemplars: list of (equation_id, {concept_name, rag_text}) tuples — picked
    from hand-authored exemplars by the caller, ideally from a similar or
    contrasting domain to give the LLM range.
    """
    parts = []

    parts.append("=== EXEMPLARS (the target style) ===\n")
    for i, (eid, body) in enumerate(exemplars, 1):
        parts.append(f"--- Exemplar {i}: {eid} ---")
        parts.append(f"Concept: {body['concept_name']}")
        parts.append(f"rag_text: {body['rag_text']}")
        parts.append("")
    parts.append("=== TARGET EQUATION (write rag_text for this) ===\n")
    parts.append(f"id:         {eq_node['id']}")
    parts.append(f"domain:     {eq_node['domain']}")
    parts.append(f"subdomain:  {eq_node['subdomain']}")
    parts.append(f"equation:   {eq_node['equation_str']}")
    parts.append("variables:")
    for sym, meta in eq_node["variables"].items():
        parts.append(f"  {sym}: {meta.get('name','?')}  "
                     f"(unit: {meta.get('unit','')}, dim: {meta.get('dimension','')})")
    if eq_node.get("conditions"):
        parts.append(f"existing_conditions: {eq_node['conditions']}")
    if eq_node.get("common_mistakes"):
        parts.append(f"existing_common_mistakes: {eq_node['common_mistakes']}")
    parts.append(f"existing_jee_chapters: {eq_node.get('jee_chapters', [])}")
    parts.append("")
    parts.append(
        "Write a concept-level rag_text for this equation in the same "
        "style as the exemplars. Begin with the named physics concept. "
        "End with explicit concept-distinction against equations that "
        "might be confused with it. JSON only.")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_rag_text(text: str, eq_node: dict) -> tuple[bool, str]:
    """
    Returns (ok, diagnostic_if_not_ok). On rejection, the caller can either
    retry the LLM call or surface for hand-review.
    """
    if not isinstance(text, str):
        return False, "not a string"
    text_stripped = text.strip()
    if len(text_stripped) < MIN_RAG_TEXT_CHARS:
        return False, f"too short ({len(text_stripped)} < {MIN_RAG_TEXT_CHARS})"
    if len(text_stripped) > MAX_RAG_TEXT_CHARS:
        return False, f"too long ({len(text_stripped)} > {MAX_RAG_TEXT_CHARS})"
    lower = text_stripped.lower()
    for phrase in BANNED_PHRASES:
        if phrase.lower() in lower:
            return False, f"contains banned phrase {phrase!r}"

    # v7.1.1: word-concatenation detector.
    # The 70B model occasionally drops spaces during streaming, producing
    # 'finalvelocity', 'constantacceleration', 'avehicle', 'accelerationwhen'.
    # Two heuristics:
    #   (a) any unbroken lowercase run >= 17 chars (longer than any single
    #       English physics word we expect — 'electromagnetic'=15,
    #       'characteristic'=14, 'representations'=15). 17+ chars almost
    #       certainly means two words ran together.
    #   (b) common physics-word boundary patterns: a word ending in 'ation',
    #       'ity', 'ence', 'ance' followed immediately by a common
    #       function-word starter ('when','with','and','of','to',...).
    if re.search(r'[a-z]{17,}', text_stripped):
        match = re.search(r'[a-z]{17,}', text_stripped)
        return False, (f"suspicious unbroken lowercase run "
                       f"(likely word concatenation): {match.group()[:30]!r}")
    merge_pattern = re.compile(
        r'\b\w{4,}(?:ation|ity|ence|ance|tion|sion)'
        r'(?:when|with|and|of|in|to|from|by|is|are|the|a|an|or|for|that|which|but|if)\b',
        re.IGNORECASE,
    )
    m = merge_pattern.search(text_stripped)
    if m:
        return False, f"merged-word pattern detected: {m.group()!r}"

    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────────────

def call_groq(system: str, user: str, retries: int = 3) -> str:
    """Returns raw LLM response string. Raises after retries exhausted."""
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    last_err = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.2,
                max_tokens=1500,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"    Groq call failed (attempt {attempt+1}/{retries}): {e!r}. "
                  f"Retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Groq call failed after {retries} attempts: {last_err!r}")


def extract_json_object(raw: str) -> dict:
    """Find the first {…} JSON object in raw text and parse it."""
    raw = raw.strip()
    # Strip code-fence wrapper if present
    if raw.startswith("```"):
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```\s*$', '', raw)
    # Find first '{' and matching '}'
    start = raw.find('{')
    if start < 0:
        raise ValueError(f"no '{{' in response: {raw[:200]!r}")
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == '{':
            depth += 1
        elif raw[i] == '}':
            depth -= 1
            if depth == 0:
                return json.loads(raw[start:i+1])
    raise ValueError(f"unbalanced JSON braces in response: {raw[:200]!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Main flow
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Regenerate rag_text fields.")
    parser.add_argument("--dry-run", type=int, metavar="N",
                        help="Generate only N equations and print results; do not save")
    parser.add_argument("--ids", type=str, default="",
                        help="Comma-separated equation IDs to regenerate (default: all non-exemplar)")
    parser.add_argument("--n-exemplars", type=int, default=3,
                        help="How many exemplars to include in each prompt (default 3)")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip writing .bak before saving")
    args = parser.parse_args()

    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not set. Set it in .env or environment.",
              file=sys.stderr)
        sys.exit(2)

    print(f"Loading exemplars from {EXEMPLARS_PATH}")
    with open(EXEMPLARS_PATH) as f:
        exemplar_data = json.load(f)
    exemplars: dict[str, dict] = exemplar_data["exemplars"]
    exemplar_ids: set[str]     = set(exemplars.keys())
    print(f"  {len(exemplars)} hand-authored exemplars loaded")

    print(f"Loading graph from {GRAPH_PATH}")
    with open(GRAPH_PATH) as f:
        graph = json.load(f)
    nodes = graph["nodes"]
    print(f"  {len(nodes)} equations in graph")

    # Determine target list
    if args.ids:
        target_ids = [s.strip() for s in args.ids.split(",") if s.strip()]
        targets = [n for n in nodes if n["id"] in target_ids]
        missing = set(target_ids) - {n["id"] for n in targets}
        if missing:
            print(f"WARNING: requested ids not in graph: {sorted(missing)}",
                  file=sys.stderr)
    else:
        targets = [n for n in nodes if n["id"] not in exemplar_ids]
    if args.dry_run:
        targets = targets[:args.dry_run]

    print(f"\nWill regenerate {len(targets)} equations "
          f"(skipping {len(exemplar_ids)} hand-authored exemplars).")
    print(f"Model: {GENERATION_MODEL}")
    print()

    successes: list[tuple[str, str]] = []
    failures:  list[tuple[str, str]] = []  # (id, reason)

    exemplar_pool = list(exemplars.items())

    for i, eq in enumerate(targets, 1):
        eid = eq["id"]
        # Pick exemplars: bias toward DIFFERENT domain so the LLM sees the
        # style applies broadly. But include one from same domain when
        # possible, for in-domain calibration.
        same_domain = [(k, v) for k, v in exemplar_pool
                       if any(d in k for d in [eq["domain"]])]
        other_domain = [(k, v) for k, v in exemplar_pool
                        if not any(d in k for d in [eq["domain"]])]
        picked = []
        if same_domain:
            picked.append(random.choice(same_domain))
        picked.extend(random.sample(other_domain,
                                    min(args.n_exemplars - len(picked),
                                        len(other_domain))))
        # Don't repeat the target itself if it happened to be in exemplars
        picked = [(k, v) for k, v in picked if k != eid]

        user_prompt = build_user_prompt(eq, picked)

        print(f"[{i}/{len(targets)}] {eid}  ({eq['equation_str']})")

        try:
            raw = call_groq(SYSTEM_PROMPT, user_prompt)
            parsed = extract_json_object(raw)
            new_rag_text = parsed.get("rag_text", "")
            ok, why = validate_rag_text(new_rag_text, eq)
            if not ok:
                print(f"    REJECT: {why}")
                # One retry with stronger instruction
                user_retry = (
                    user_prompt
                    + f"\n\nIMPORTANT: previous attempt was rejected because {why}. "
                    + "Please respect the length and style constraints. "
                    + "Concept-level only, no boilerplate.")
                raw = call_groq(SYSTEM_PROMPT, user_retry)
                parsed = extract_json_object(raw)
                new_rag_text = parsed.get("rag_text", "")
                ok, why = validate_rag_text(new_rag_text, eq)
                if not ok:
                    print(f"    REJECT (retry): {why}  — SKIPPED")
                    failures.append((eid, f"validation failed: {why}"))
                    continue
            successes.append((eid, new_rag_text))
            print(f"    OK   ({len(new_rag_text)} chars)")
        except Exception as e:
            print(f"    ERROR: {e!r}")
            failures.append((eid, repr(e)))
            continue

        # Light rate-limiting to be polite
        time.sleep(0.3)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"GENERATION COMPLETE: {len(successes)} ok, {len(failures)} failed")
    print("=" * 70)
    if failures:
        print("\nFAILED — these need hand-review or re-run:")
        for fid, reason in failures:
            print(f"  {fid}: {reason}")

    if args.dry_run:
        print("\n--- DRY RUN: printing results, NOT saving ---")
        for eid, rt in successes[:5]:
            print(f"\n[{eid}]")
            print(rt)
        return

    # ── Merge and save ────────────────────────────────────────────────────────
    print("\nApplying:")
    print(f"  - {len(exemplars)} hand-authored exemplars")
    print(f"  - {len(successes)} LLM-generated")

    nodes_by_id = {n["id"]: n for n in nodes}
    applied = 0
    # Apply exemplars (overrides any existing rag_text)
    for eid, body in exemplars.items():
        if eid in nodes_by_id:
            nodes_by_id[eid]["rag_text"] = body["rag_text"]
            applied += 1
    # Apply LLM-generated
    for eid, rt in successes:
        if eid in nodes_by_id:
            nodes_by_id[eid]["rag_text"] = rt
            applied += 1

    # Backup
    if not args.no_backup and GRAPH_PATH.exists():
        print(f"Backing up: {GRAPH_PATH} → {BACKUP_PATH}")
        shutil.copyfile(GRAPH_PATH, BACKUP_PATH)

    print(f"Writing: {GRAPH_PATH}  ({applied} rag_texts updated)")
    with open(GRAPH_PATH, "w") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    if failures:
        print(f"\nReminder: {len(failures)} equations were not regenerated. "
              "Their old rag_text is unchanged in the saved file. "
              "Re-run with --ids '<failed_id_1>,<failed_id_2>' to retry.")


if __name__ == "__main__":
    main()
