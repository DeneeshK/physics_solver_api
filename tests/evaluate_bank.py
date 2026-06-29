"""
tests/evaluate_bank.py
Evaluation harness for a JSON question bank.

Runs every question in a bank file through the real PhysicsSolver, compares
the answer against the expected value (within per-question tolerance), and
produces a report grouped THREE ways:

  1. by domain            — which physics areas pass/fail
  2. by reasoning_type    — single_step / multi_step_chain / implicit_given /
                            symbol_mismatch_or_trap / underspecified
  3. by failure_stage     — WHERE each failure happened: stage1_parse,
                            retrieval, stage2_selection, execution, or
                            answer_mismatch

The third grouping is the point. It turns "62% pass" into "multi_step_chain
fails 40%, almost all in stage1_parse" — so you fix failure CATEGORIES, not
individual questions.

USAGE
  # default bank location: questions/question_bank.json
  python tests/evaluate_bank.py

  # explicit bank file
  python tests/evaluate_bank.py questions/batch_kinematics.json

  # multiple bank files (concatenated)
  python tests/evaluate_bank.py questions/batch_*.json

  # write a detailed per-question JSON report alongside the console summary
  python tests/evaluate_bank.py questions/question_bank.json --report out.json

  # limit to first N questions (smoke test)
  python tests/evaluate_bank.py questions/question_bank.json --limit 10

WORKS WITH LOCAL OR GROQ
  The harness just calls solver.solve(); whichever provider your .env selects
  (local Ollama or Groq) is what runs. No token-per-minute limit locally, so a
  100-question bank runs without rate-limit babysitting.

QUESTION BANK SCHEMA (per element) — matches QUESTION_BANK_PROMPT.md
  {
    "id": "kinematics_multi_step_chain_01",
    "domain": "kinematics",
    "reasoning_type": "multi_step_chain",
    "difficulty": "medium",
    "question": "...",
    "expected_answer": 45000.0,          // null for underspecified
    "expected_unit": "N",                // null for underspecified
    "answer_tolerance_pct": 1.0,
    "solution_outline": "...",
    "implicit_values": {"u": 0},
    "expected_equations": ["F = m*a"],
    "notes": "..."
  }
"""
import sys
import os
import json
import glob
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "logs" / "solver.log"


# ── Bank loading ──────────────────────────────────────────────────────────────
def load_bank(patterns: list[str]) -> list[dict]:
    """Load and concatenate one or more bank files (glob patterns allowed)."""
    files: list[str] = []
    for pat in patterns:
        matched = glob.glob(pat)
        if not matched:
            # Treat as a literal path so we can error clearly below
            matched = [pat]
        files.extend(sorted(matched))

    bank: list[dict] = []
    seen_ids: set[str] = set()
    for fp in files:
        p = Path(fp)
        if not p.exists():
            print(f"  WARNING: bank file not found: {fp}")
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "questions" in data:
            data = data["questions"]   # tolerate {"questions": [...]} wrapper
        if not isinstance(data, list):
            print(f"  WARNING: {fp} is not a JSON array; skipping")
            continue
        for q in data:
            qid = q.get("id", f"<no-id-{len(bank)}>")
            if qid in seen_ids:
                print(f"  WARNING: duplicate id {qid!r} (keeping first)")
                continue
            seen_ids.add(qid)
            q["_source_file"] = str(p.name)
            bank.append(q)
    return bank


# ── Per-question log slicing for failure-stage detection ──────────────────────
def read_log_events_since(marker_offset: int) -> list[dict]:
    """Read JSON log events written after the given byte offset."""
    if not LOG_PATH.exists():
        return []
    events = []
    with open(LOG_PATH, encoding="utf-8") as f:
        f.seek(marker_offset)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def current_log_offset() -> int:
    """Byte offset at the end of the current log (so we can read only what a
    single solve appends)."""
    if not LOG_PATH.exists():
        return 0
    return LOG_PATH.stat().st_size


def classify_failure_stage(events: list[dict], resp, expected_answer) -> str:
    """
    Given the log events for ONE solve and the response, decide WHERE it
    failed. Order matters — we attribute to the earliest stage that broke.

    Returns one of:
      "stage1_parse"      — Stage 1 produced bad/empty givens or wrong unknown
      "retrieval"         — the search didn't surface a usable candidate
      "stage2_selection"  — the LLM omitted/none'd/invalid'd the pick
      "execution"         — SymPy couldn't compute despite a chain
      "answer_mismatch"   — produced a number, but it's outside tolerance
      "rate_limit"        — a provider rate limit interrupted the run
      "unknown"           — couldn't attribute (shouldn't happen often)
    """
    # Rate limit anywhere is its own bucket — it's an infra problem, not a
    # solver-logic problem.
    if any(e.get("is_rate_limit") for e in events):
        return "rate_limit"

    # Did the solver error out? Inspect the error string + the events leading
    # up to it.
    err = (getattr(resp, "error", None) or "").lower()

    # Stage 2 failure signatures (the most common real failure)
    stage2_omitted = any(e.get("event") == "stage2_item_omitted" for e in events)
    stage2_none = any(
        e.get("event") == "stage2_item_decision" and e.get("fallback_used") == "llm_decision_none"
        for e in events
    )
    stage2_invalid = any(
        e.get("event") == "stage2_item_decision" and e.get("fallback_used") == "llm_invalid_id"
        for e in events
    )
    stage2_hallucinated = any(e.get("event") == "stage2_hallucinated_symbol" for e in events)

    # Stage 1 signal: parse produced no givens at all (and the question wasn't
    # meant to be underspecified — caller checks that separately).
    parsed = [e for e in events if e.get("event") == "stage1_parsed"]
    stage1_empty = bool(parsed) and parsed[-1].get("n_given", 0) == 0

    # Retrieval signal: a search returned but the top results were weak / the
    # round entry shows no usable candidate. We approximate: if Stage 2 never
    # even ran (no round_entry) but Stage 1 succeeded, retrieval likely failed.
    had_retrieval = any(e.get("event") == "retrieval_search" for e in events)
    had_round_entry = any(e.get("event") == "stage2_round_entry" for e in events)

    # Attribute in priority order.
    if stage1_empty:
        return "stage1_parse"

    if "stage 2 failed" in err or stage2_omitted or stage2_none or stage2_invalid:
        # But if the root is a Stage 1 mis-parse that cascaded (e.g. wrong
        # symbol leading to an impossible chain), we still call it stage2 here
        # because that's where it surfaced. The by-reasoning_type grouping
        # will show whether these cluster with parse-sensitive types.
        return "stage2_selection"

    if stage2_hallucinated:
        return "stage2_selection"

    if had_retrieval and not had_round_entry and not err:
        return "retrieval"

    # If we got a value but it's wrong, that's a mismatch (not a crash).
    val = getattr(resp, "final_value", None)
    if val is not None and expected_answer is not None:
        return "answer_mismatch"

    # Execution: a chain was built but SymPy produced nothing.
    if err and ("sympy" in err or "could not" in err or "no value" in err or "execution" in err):
        return "execution"

    if err:
        # Generic solver error we couldn't bucket precisely.
        return "unknown"

    return "unknown"


# ── Single-question evaluation ────────────────────────────────────────────────
def evaluate_question(solver, q: dict) -> dict:
    """Run one question, return a result record."""
    qid = q.get("id", "<no-id>")
    question = q.get("question", "")
    domain = q.get("domain", "<no-domain>")
    rtype = q.get("reasoning_type", "<no-type>")
    difficulty = q.get("difficulty", "?")
    expected = q.get("expected_answer", None)
    expected_unit = q.get("expected_unit", None)
    tol_pct = q.get("answer_tolerance_pct", 2.0)
    is_underspecified = (rtype == "underspecified") or (expected is None)

    record = {
        "id": qid, "domain": domain, "reasoning_type": rtype,
        "difficulty": difficulty, "question": question,
        "expected_answer": expected, "expected_unit": expected_unit,
        "passed": False, "failure_stage": None, "detail": "",
        "got_value": None, "got_unit": None, "got_confidence": None,
        "got_chain": None, "got_error": None,
    }

    offset = current_log_offset()
    try:
        resp = solver.solve(question)
    except Exception as e:
        # A hard crash (rare — the pipeline usually returns a response object
        # with .error set). Capture and classify from the log.
        events = read_log_events_since(offset)
        record["failure_stage"] = classify_failure_stage(events, _Empty(), expected)
        record["detail"] = f"solver raised: {e}"
        record["got_error"] = str(e)
        return record

    events = read_log_events_since(offset)

    record["got_value"] = getattr(resp, "final_value", None)
    record["got_unit"] = getattr(resp, "final_unit", None)
    record["got_confidence"] = getattr(resp, "confidence", None)
    record["got_chain"] = getattr(resp, "chain_summary", None)
    record["got_error"] = getattr(resp, "error", None)

    # ── Underspecified questions: pass iff the solver did NOT fabricate an
    #    answer. We accept either UNVERIFIED confidence or no final value.
    if is_underspecified:
        conf = (getattr(resp, "confidence", "") or "").upper()
        val = getattr(resp, "final_value", None)
        # Treat 0.0-with-no-chain as "no answer" (the pipeline's UNVERIFIED shape)
        no_real_answer = (val is None) or (val == 0.0 and not getattr(resp, "chain_summary", None))
        if conf == "UNVERIFIED" or no_real_answer:
            record["passed"] = True
            record["detail"] = "correctly refused / UNVERIFIED"
        else:
            record["passed"] = False
            record["failure_stage"] = "hallucinated_answer"
            record["detail"] = (
                f"fabricated answer {val} {getattr(resp,'final_unit','')} "
                f"with confidence {conf} for an underspecified question"
            )
        return record

    # ── Normal questions: compare numeric answer within tolerance.
    val = getattr(resp, "final_value", None)
    if val is None:
        record["passed"] = False
        record["failure_stage"] = classify_failure_stage(events, resp, expected)
        record["detail"] = f"no answer produced; error={record['got_error']!r}"
        return record

    try:
        diff = abs(float(val) - float(expected)) / max(abs(float(expected)), 1e-12)
    except (TypeError, ValueError):
        record["passed"] = False
        record["failure_stage"] = "answer_mismatch"
        record["detail"] = f"non-numeric answer {val!r}"
        return record

    if diff <= tol_pct / 100.0:
        record["passed"] = True
        record["detail"] = f"answer {val} within {tol_pct}% of {expected}"
    else:
        record["passed"] = False
        record["failure_stage"] = classify_failure_stage(events, resp, expected)
        # If classify said mismatch (value present, wrong), keep it; otherwise
        # the value-present case is by definition a mismatch.
        if record["failure_stage"] in (None, "unknown"):
            record["failure_stage"] = "answer_mismatch"
        record["detail"] = (
            f"answer {val} differs from {expected} by {diff*100:.1f}% "
            f"(tol {tol_pct}%)"
        )
    return record


class _Empty:
    """Stand-in response for hard-crash classification."""
    error = "solver raised"
    final_value = None
    confidence = None
    chain_summary = None


# ── Reporting ─────────────────────────────────────────────────────────────────
def _bar(passed: int, total: int, width: int = 24) -> str:
    if total == 0:
        return " " * width
    filled = int(round(width * passed / total))
    return "█" * filled + "░" * (width - filled)


def print_group(title: str, groups: dict[str, list[dict]]):
    print(f"\n{title}")
    print("─" * 72)
    # Sort by pass-rate ascending so the worst categories are at the top.
    def rate(recs):
        return sum(r["passed"] for r in recs) / max(len(recs), 1)
    for key in sorted(groups, key=lambda k: rate(groups[k])):
        recs = groups[key]
        p = sum(r["passed"] for r in recs)
        t = len(recs)
        pct = 100.0 * p / max(t, 1)
        print(f"  {key:<28} {_bar(p,t)}  {p:>2}/{t:<2}  {pct:5.1f}%")


def print_failure_breakdown(records: list[dict]):
    fails = [r for r in records if not r["passed"]]
    if not fails:
        print("\nNo failures. 🎉")
        return
    by_stage = defaultdict(list)
    for r in fails:
        by_stage[r["failure_stage"] or "unknown"].append(r)

    print(f"\nFAILURES BY STAGE  ({len(fails)} total)")
    print("─" * 72)
    for stage in sorted(by_stage, key=lambda s: -len(by_stage[s])):
        recs = by_stage[stage]
        print(f"\n  ▸ {stage}  ({len(recs)})")
        # Show which reasoning_types dominate this failure stage
        rtypes = defaultdict(int)
        for r in recs:
            rtypes[r["reasoning_type"]] += 1
        rtype_summary = ", ".join(
            f"{rt}×{n}" for rt, n in sorted(rtypes.items(), key=lambda x: -x[1])
        )
        print(f"     reasoning types: {rtype_summary}")
        # List up to 6 example questions
        for r in recs[:6]:
            print(f"       - [{r['id']}] {r['detail']}")
        if len(recs) > 6:
            print(f"       ... and {len(recs)-6} more")


def cross_tab(records: list[dict]):
    """reasoning_type × failure_stage cross-tabulation — the key diagnostic."""
    fails = [r for r in records if not r["passed"]]
    if not fails:
        return
    rtypes = sorted({r["reasoning_type"] for r in records})
    stages = sorted({r["failure_stage"] or "unknown" for r in fails})
    if not stages:
        return
    print(f"\nCROSS-TAB: reasoning_type × failure_stage (failure counts)")
    print("─" * 72)
    # Header
    colw = max(14, max((len(s) for s in stages), default=14)) + 2
    header = f"  {'reasoning_type':<26}" + "".join(f"{s:>{colw}}" for s in stages)
    print(header)
    for rt in rtypes:
        row_recs = [r for r in fails if r["reasoning_type"] == rt]
        if not row_recs:
            continue
        cells = []
        for s in stages:
            n = sum(1 for r in row_recs if (r["failure_stage"] or "unknown") == s)
            cells.append(f"{n:>{colw}}" if n else f"{'·':>{colw}}")
        print(f"  {rt:<26}" + "".join(cells))


def print_report(records: list[dict]):
    total = len(records)
    passed = sum(r["passed"] for r in records)
    print("\n" + "=" * 72)
    print(f" EVALUATION REPORT — {passed}/{total} passed "
          f"({100.0*passed/max(total,1):.1f}%)")
    print("=" * 72)

    by_domain = defaultdict(list)
    by_type = defaultdict(list)
    by_difficulty = defaultdict(list)
    for r in records:
        by_domain[r["domain"]].append(r)
        by_type[r["reasoning_type"]].append(r)
        by_difficulty[r["difficulty"]].append(r)

    print_group("BY REASONING TYPE  (worst first)", by_type)
    print_group("BY DOMAIN  (worst first)", by_domain)
    print_group("BY DIFFICULTY", by_difficulty)
    print_failure_breakdown(records)
    cross_tab(records)

    print("\n" + "=" * 72)
    print(" INTERPRETATION HINTS")
    print("=" * 72)
    print("""  - If failures concentrate in ONE failure_stage across many domains,
    that's a SYSTEMATIC bug to fix once (not per-question).
  - If 'stage1_parse' dominates: the parser is dropping givens or mis-typing
    the unknown. Prompt-level fix.
  - If 'stage2_selection' dominates on multi_step_chain: the concept/derivability
    reasoning or the candidate set is the issue (this is the concept-aware
    mapping work).
  - If 'retrieval' dominates: rag_texts for those equations need tuning, or
    ChromaDB is stale (run scripts/verify_rag.py --check-sync).
  - If 'answer_mismatch' dominates: the chain is right but execution/units are
    off — check the graph-content quirks list (v7.3).
  - If 'hallucinated_answer' appears: the anti-hallucination guard isn't
    catching that phrasing — Stage 1 prompt fix.""")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Evaluate a physics question bank.")
    ap.add_argument("bank", nargs="*", default=["questions/question_bank.json"],
                    help="Bank JSON file(s) or glob(s). Default: questions/question_bank.json")
    ap.add_argument("--report", default=None,
                    help="Write a detailed per-question JSON report to this path.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only evaluate the first N questions (smoke test).")
    ap.add_argument("--only", default=None,
                    help="Run only matching questions (comma-separated). Each "
                         "term matches by 1-based index (e.g. 7), exact id, or "
                         "case-insensitive substring of the id. "
                         "Example: --only 7,19  or  --only capacitors,emi")
    args = ap.parse_args()

    # Load bank
    bank = load_bank(args.bank)
    if not bank:
        print("No questions loaded. Check your bank path(s).")
        print(f"  Looked for: {args.bank}")
        sys.exit(1)

    if args.only:
        terms = [t.strip() for t in args.only.split(",") if t.strip()]
        selected = []
        for idx, q in enumerate(bank, 1):
            qid = str(q.get("id", ""))
            for t in terms:
                if (t == str(idx)) or (t == qid) or (t.lower() in qid.lower()):
                    selected.append(q)
                    break
        if not selected:
            print(f"No questions matched --only {args.only!r}.")
            print("  Available ids:")
            for idx, q in enumerate(bank, 1):
                print(f"    [{idx}] {q.get('id', '<no-id>')}")
            sys.exit(1)
        bank = selected
    elif args.limit:
        bank = bank[:args.limit]

    print(f"Loaded {len(bank)} questions from: {', '.join(args.bank)}")

    # Build solver (same path as test_live.py)
    from dotenv import load_dotenv
    load_dotenv()
    from solver.graph_loader import load_graphs
    from solver.pipeline import PhysicsSolver
    graph = load_graphs()
    solver = PhysicsSolver(graph)

    # Run
    records = []
    for i, q in enumerate(bank, 1):
        qid = q.get("id", f"<{i}>")
        print(f"[{i:>3}/{len(bank)}] {qid:<40}", end=" ", flush=True)
        rec = evaluate_question(solver, q)
        records.append(rec)
        if rec["passed"]:
            print("✓")
        else:
            print(f"✗  ({rec['failure_stage']})")

    # Report
    print_report(records)

    # Optional detailed JSON dump
    if args.report:
        out = {
            "summary": {
                "total": len(records),
                "passed": sum(r["passed"] for r in records),
            },
            "records": records,
        }
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nDetailed report written to {args.report}")

    # Exit code: 0 if all passed, else 1 (handy for CI / scripting)
    sys.exit(0 if all(r["passed"] for r in records) else 1)


if __name__ == "__main__":
    main()
