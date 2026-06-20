#!/usr/bin/env python3
"""
scripts/diagnose_question.py
v7.1.2 — Run a single question with full logging and print a human-readable
summary of what happened at every stage. Use this when a test fails and
you want to know why, without grepping logs.

The script:
  1. Truncates logs/solver.log to a fresh file (so this run's logs are
     isolated). Comment out that line if you want to keep history.
  2. Runs the question through the full pipeline with SOLVER_LOG_VERBOSE=1.
  3. Reads the log and prints a per-stage breakdown:
       Stage 1: parse output (given, unknown, search_query)
       ChromaDB retrieval: query, top-k results
       Stage 2: per round, what was asked, what LLM picked or rejected
       Final: success/error, chain summary
  4. Exits 0 on success, 1 on failure — so you can use it in shell loops.

Usage:
  python scripts/diagnose_question.py "A body of density 8000 kg/m^3..."
  python scripts/diagnose_question.py --question-file q.txt
  python scripts/diagnose_question.py --keep-log "A 5 kg object..."
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_FILE = PROJECT_ROOT / "logs" / "solver.log"


def _read_log_events() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    out = []
    for line in LOG_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def _print_section(title: str):
    print()
    print("═" * 72)
    print(f"  {title}")
    print("═" * 72)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="?", default=None)
    p.add_argument("--question-file", type=str, default=None)
    p.add_argument("--keep-log", action="store_true",
                   help="Don't truncate solver.log before running.")
    args = p.parse_args()

    if args.question_file:
        question = Path(args.question_file).read_text().strip()
    elif args.question:
        question = args.question
    else:
        p.print_help()
        sys.exit(1)

    if not args.keep_log:
        LOG_FILE.parent.mkdir(exist_ok=True)
        LOG_FILE.write_text("")  # truncate

    os.environ["SOLVER_LOG_VERBOSE"] = "1"  # full prompts/responses captured

    # Run the pipeline
    from solver.pipeline import PhysicsSolver
    solver = PhysicsSolver()
    print(f"Question: {question}\n")
    print("Running through pipeline (cold start may take ~15s for BGE)...")
    try:
        resp = solver.solve(question)
    except Exception as e:
        print(f"\n💥 Pipeline raised: {type(e).__name__}: {e}")
        resp = None

    events = _read_log_events()

    # ── Print structured breakdown ────────────────────────────────────────────

    _print_section("STAGE 1: question parsing")
    stage1 = [e for e in events if e.get("event") == "stage1_parsed"]
    if stage1:
        e = stage1[0]
        print(f"  given symbols:      {e.get('given_symbols')}")
        print(f"  unknown:            {e.get('unknown_symbol')}")
        print(f"  implicit constants: {e.get('implicit_constants')}")
        print(f"  likely domains:     {e.get('likely_domains')}")
        print(f"  search_query:       {e.get('search_query')!r}")
    else:
        print("  (no stage1_parsed event in log — Stage 1 likely failed)")

    _print_section("RETRIEVAL: ChromaDB hybrid search")
    retrievals = [e for e in events if e.get("event") == "retrieval_search"]
    for i, e in enumerate(retrievals, 1):
        print(f"  Query #{i}: {e.get('query', '')[:100]!r}")
        for r in e.get("results", [])[:6]:
            print(f"    {r['score']:.4f} (sem={r['sem']:.2f} bm25={r['bm25']:.2f})  {r['id']}")
        if len(e.get("results", [])) > 6:
            print(f"    ...and {len(e['results']) - 6} more")
        print()
    if not retrievals:
        print("  (no retrieval_search events — Chroma may be disabled or down)")

    _print_section("STAGE 2: per-round equation selection")
    by_round = {}
    for e in events:
        ev = e.get("event")
        if ev in ("stage2_round_entry", "stage2_llm_selections_received",
                  "stage2_item_decision", "stage2_item_omitted"):
            by_round.setdefault(e.get("round_num", -1), []).append(e)

    for rnum in sorted(by_round):
        print(f"\n  ── Round {rnum} ──")
        for ev in by_round[rnum]:
            t = ev.get("event")
            if t == "stage2_round_entry":
                for item in ev.get("items", []):
                    print(f"    asking for: {item['symbol']} ({item.get('name','')})")
                    print(f"      candidates: {item['n_candidates']}")
                    for cid, ls in zip(item['candidate_ids'], item.get('landing_sources', [])):
                        print(f"        [{ls}] {cid}")
            elif t == "stage2_llm_selections_received":
                if ev.get("omitted"):
                    print(f"    ⚠️  LLM addressed {ev['addressed']}, OMITTED {ev['omitted']}")
                else:
                    print(f"    LLM addressed all requested symbols.")
            elif t == "stage2_item_decision":
                sym = ev["symbol"]
                if ev.get("fallback_used"):
                    print(f"    ⚠️  {sym}: fallback_used={ev['fallback_used']!r}")
                    print(f"        LLM reason: {ev.get('llm_reason', '')[:200]}")
                else:
                    print(f"    ✓ {sym}: chose {ev.get('chosen_eq_id')}")
                    print(f"        reason: {ev.get('llm_reason', '')[:200]}")
            elif t == "stage2_item_omitted":
                print(f"    ❌ {ev['symbol']}: LLM did not address — surfaced as llm_omitted_item")
                print(f"        candidates were: {ev.get('candidate_ids')}")

    _print_section("ERRORS / RATE LIMITS")
    errors = [e for e in events if e.get("level") == "ERROR"]
    if errors:
        for e in errors:
            print(f"  {e.get('event')}: {e.get('error_msg') or e.get('traceback', ['?'])[-1].strip()}")
            if e.get("is_rate_limit"):
                print(f"    🚨 RATE LIMIT — Groq returned 429 / TPM cap.")
    else:
        print("  (none — no rate limits or exceptions during this run)")

    _print_section("FINAL")
    if resp is None:
        print("  Pipeline crashed (see ERRORS section above).")
        sys.exit(1)
    print(f"  confidence:   {resp.confidence}")
    print(f"  final value:  {resp.final_value} {resp.final_unit}")
    print(f"  exact:        {resp.final_value_exact}")
    print(f"  chain:        {resp.chain_summary}")
    if getattr(resp, "error", None):
        print(f"  error:        {resp.error}")
    print(f"  elapsed:      {resp.time_taken_s}s")

    print(f"\nFull log: {LOG_FILE}")
    sys.exit(0 if resp.confidence == "HIGH" else 1)


if __name__ == "__main__":
    main()
