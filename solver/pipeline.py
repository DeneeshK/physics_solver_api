"""
solver/pipeline.py
5-stage end-to-end pipeline (redesigned).

Stage 1  parse_question()       — LLM extracts given/target/implicit constants
Stage 2  resolve_frontier()     — LLM-driven conceptual equation selection
Stage 3  execute_plan()         — SymPy exact-arithmetic execution with traces
Stage 4  narrate_from_trace()   — LLM narrates the already-correct trace
Stage 5  generate_distractors() — LLM produces 3 wrong MCQ options
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field

from solver.graph_loader    import GraphIndex
from solver.frontier_resolver import (
    FrontierItem, ResolutionResult, resolve_frontier,
)
from solver.sympy_executor  import execute_plan, ExecutionTrace
from solver.llm_interface   import (
    parse_question, call_round_selector,
    narrate_from_trace, generate_distractors,
)
from config import PHYSICAL_CONSTANTS, UNIVERSAL_CONSTANTS, IMPLICIT_CONSTANTS_CATALOG

MAX_BACKTRACK_ATTEMPTS = 3  # Stage 3 failure -> exclude offending eq(s) -> retry Stage 2


@dataclass
class SolverResponse:
    question:        str
    final_value:     float
    final_value_exact: str
    final_unit:      str
    final_symbol:    str
    explanation:     str
    decision_log:    list[dict]
    mcq:             dict
    confidence:      str          # "HIGH" | "UNVERIFIED"
    chain_summary:   list[str]
    time_taken_s:    float
    error:           str = ""


class PhysicsSolver:
    """Instantiate once per process; call .solve(question) per request."""

    def __init__(self, graph_index: GraphIndex):
        self.graph = graph_index

    def solve(self, question: str) -> SolverResponse:
        t0 = time.time()

        # ═══════════════════════════════════════════════════════════════════════
        # Stage 1 — Parse
        # ═══════════════════════════════════════════════════════════════════════
        try:
            parsed = parse_question(question)
        except Exception as e:
            return self._error(question, f"Stage 1 parse failed: {e}", t0)

        given_meta  = parsed.get("given", {})   # {sym: {value,unit,name,dimension}}
        unknown     = parsed.get("unknown", {})  # {symbol,name,unit,dimension}
        target_sym  = unknown.get("symbol", "")

        if not target_sym:
            return self._error(question, "Could not identify the unknown variable.", t0)

        # Build {symbol: float} for Stage 3
        given_values: dict[str, float] = {}
        for sym, meta in given_meta.items():
            try:
                given_values[sym] = float(meta["value"])
            except (TypeError, ValueError, KeyError):
                pass

        # Build {symbol: {value,unit,name,dimension}} for Stage 2
        given_full: dict[str, dict] = {
            sym: {
                "value":     meta.get("value"),
                "unit":      meta.get("unit", ""),
                "name":      meta.get("name", sym),
                "dimension": meta.get("dimension", ""),
            }
            for sym, meta in given_meta.items()
            if meta.get("value") is not None
        }
        # Inject TRUE universal constants only (pi, c, G, epsilon_0, etc.) —
        # these are context-independent so no per-question judgment is needed.
        # 'g' is deliberately excluded: it only enters given_full if Stage 1
        # actually flagged it as implied by the scenario (see llm_interface.py).
        for sym in UNIVERSAL_CONSTANTS:
            if sym in IMPLICIT_CONSTANTS_CATALOG and sym not in given_full:
                cat = IMPLICIT_CONSTANTS_CATALOG[sym]
                given_full[sym] = {
                    "value":     cat["value"],
                    "unit":      cat["unit"],
                    "name":      cat["name"],
                    "dimension": cat["dimension"],
                }

        # Build target FrontierItem
        target = FrontierItem(
            symbol    = target_sym,
            name      = unknown.get("name", target_sym),
            unit      = unknown.get("unit", ""),
            dimension = unknown.get("dimension", ""),
        )

        # ═══════════════════════════════════════════════════════════════════════
        # Stage 2 + 3 — Frontier resolution with bounded backtracking
        # ═══════════════════════════════════════════════════════════════════════
        # Backtracking triggers on ANY Stage 3 failure — not only when the LLM
        # self-flagged a conditions_concern. An LLM can be confidently wrong
        # with no flag raised at all; that failure mode needs to retry too.
        confidence       = "HIGH"
        excluded_eqs: set = set()
        full_decision_log: list[dict] = []
        resolution  = None
        exec_trace  = None

        for attempt in range(1, MAX_BACKTRACK_ATTEMPTS + 1):
            resolution = resolve_frontier(
                target        = target,
                given         = given_full,
                graph_index   = self.graph,
                question      = question,
                llm_round_fn  = call_round_selector,
                excluded_eqs  = excluded_eqs,
            )
            for entry in resolution.decision_log:
                full_decision_log.append({**entry, "attempt": attempt})

            if not resolution.success:
                # Stage 2 itself couldn't find a chain (no candidates left
                # after exclusions, or LLM declared it unanswerable).
                break

            exec_trace = execute_plan(
                plan          = resolution.plan,
                given_values  = given_values,
                target_symbol = target_sym,
                target_unit   = unknown.get("unit", ""),
                target_dim    = unknown.get("dimension", ""),
            )

            if exec_trace.success:
                break

            # Stage 3 failed. Exclude every equation that could be at fault:
            # the one(s) SymPy actually choked on, plus any the LLM itself
            # flagged as a conditions risk this round. Retry Stage 2 with
            # those banned, so it's a fresh LLM choice — not a silent
            # deterministic fallback.
            provisional_ids = {
                s.equation["id"] for s in resolution.plan
                if hasattr(s, "is_provisional") and s.is_provisional
            }
            newly_excluded = set(exec_trace.failed_eq_ids) | provisional_ids
            if not newly_excluded or newly_excluded <= excluded_eqs:
                # Nothing new to exclude — retrying would just repeat the
                # same failure, so stop instead of looping pointlessly.
                break
            excluded_eqs |= newly_excluded

        if resolution is None or not resolution.success:
            return self._error(
                question,
                f"Stage 2 failed: {resolution.failure_reason if resolution else 'no attempts ran'}",
                t0,
                decision_log=full_decision_log,
            )

        if exec_trace is None or not exec_trace.success:
            # All backtrack attempts exhausted and Stage 3 still couldn't
            # produce a valid result. Stop here — narrating a final_value
            # of 0.0 or generating distractors around it would be nonsense,
            # not a real (if low-confidence) answer.
            last_error = exec_trace.error if exec_trace else "no execution attempted"
            return self._error(
                question,
                f"Stage 3 failed after {len(excluded_eqs)} excluded equation(s): {last_error}",
                t0,
                decision_log=full_decision_log,
            )

        # ═══════════════════════════════════════════════════════════════════════
        # Stage 4 — Narrate
        # ═══════════════════════════════════════════════════════════════════════
        trace_steps = [
            {
                "equation_str":   t.equation_str,
                "solving_for":    t.solving_for,
                "unit":           t.unit,
                "symbolic":       t.trace.symbolic,
                "substituted":    t.trace.substituted,
                "result_exact":   t.trace.result_exact,
                "result_float":   t.trace.result_float,
                "conditions_concern": t.conditions_concern,
            }
            for t in exec_trace.step_traces
        ]
        final_answer = {
            "symbol":      target_sym,
            "value_exact": exec_trace.final_exact_str,
            "value_float": _format_float(exec_trace.final_float),
            "unit":        exec_trace.final_unit,
        }

        explanation = narrate_from_trace(
            question     = question,
            trace_steps  = trace_steps,
            decision_log = full_decision_log,
            final_answer = final_answer,
        )

        # ═══════════════════════════════════════════════════════════════════════
        # Stage 5 — Distractors
        # ═══════════════════════════════════════════════════════════════════════
        chain_nodes = [
            s.equation for s in resolution.plan
            if hasattr(s, "equation")
        ]
        distractors = generate_distractors(
            question      = question,
            correct_value = exec_trace.final_float,
            correct_unit  = exec_trace.final_unit,
            chain_nodes   = chain_nodes,
        )

        mcq = _build_mcq(exec_trace.final_float, exec_trace.final_unit, distractors)

        chain_summary = [
            (f"{s.equation['equation_str']} → {s.solves_for.symbol}"
             if hasattr(s, "equation")
             else f"simultaneous({', '.join(e['equation_str'] for e in s.equations)}) → {[u.symbol for u in s.unknowns]}")
            for s in resolution.plan
        ]

        return SolverResponse(
            question        = question,
            final_value     = exec_trace.final_float,
            final_value_exact = exec_trace.final_exact_str,
            final_unit      = exec_trace.final_unit,
            final_symbol    = target_sym,
            explanation     = explanation,
            decision_log    = full_decision_log,
            mcq             = mcq,
            confidence      = confidence,
            chain_summary   = chain_summary,
            time_taken_s    = round(time.time() - t0, 2),
        )

    def _error(
        self, question: str, error: str, t0: float,
        decision_log: list = None,
    ) -> SolverResponse:
        return SolverResponse(
            question=question, final_value=0.0,
            final_value_exact="", final_unit="", final_symbol="",
            explanation=f"Could not solve: {error}",
            decision_log=decision_log or [],
            mcq={}, confidence="UNVERIFIED",
            chain_summary=[], time_taken_s=round(time.time() - t0, 2),
            error=error,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_float(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 1e-3 and abs(v) < 1e7:
        return f"{v:.6g}"
    return f"{v:.4e}"


def _build_mcq(correct_val: float, correct_unit: str, distractors: list) -> dict:
    import random
    options = [
        {
            "label":      chr(65 + i),
            "value":      d.get("value", 0),
            "unit":       d.get("unit", correct_unit),
            "is_correct": False,
            "mistake":    d.get("mistake", ""),
        }
        for i, d in enumerate(distractors)
    ]
    options.append({
        "label":      chr(65 + len(options)),
        "value":      correct_val,
        "unit":       correct_unit,
        "is_correct": True,
        "mistake":    "",
    })
    random.shuffle(options)
    for i, opt in enumerate(options):
        opt["label"] = chr(65 + i)
    return {
        "correct": {"value": correct_val, "unit": correct_unit},
        "options": options,
    }
