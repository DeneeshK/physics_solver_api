"""
solver/frontier_resolver.py
Stage 2: LLM-driven frontier resolution loop.

Replaces backward_chain.py.  The key design difference:
  OLD: deterministic scoring picks the equation with most inputs available.
  NEW: LLM picks based on what the question is *physically describing*.
       Deterministic code only generates + filters candidates; never chooses.

Entry point: resolve_frontier(target, given, graph_index, question, llm_round_fn)
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from config import PHYSICAL_CONSTANTS, MAX_CHAIN_DEPTH


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FrontierItem:
    """One quantity that still needs to be resolved."""
    symbol:       str
    name:         str
    unit:         str
    dimension:    str
    introduced_by: str = ""   # equation ID that needed this (for trace)


@dataclass
class ResolvedStep:
    """
    One chosen equation in the solution plan.
    solves_for: the FrontierItem this equation resolves.
    inputs_used: symbols that will be substituted (excluding constants).
    """
    equation:          dict
    solves_for:        FrontierItem
    inputs_used:       list[str]
    round_num:         int
    llm_reason:        str
    conditions_concern: str | None = None
    is_provisional:    bool = False   # flagged condition, may be swapped on backtrack


@dataclass
class SimultaneousGroup:
    """
    Two or more equations that must be solved as a system (circular dependency).
    Example: Coulomb's law and F=ma both equated via F when both F and a unknown.
    """
    equations:  list[dict]
    unknowns:   list[FrontierItem]
    round_num:  int
    llm_reason: str = "Mutually dependent equations — solved as simultaneous system."


@dataclass
class ResolutionResult:
    """
    Output of resolve_frontier().
    plan: ordered list of ResolvedStep | SimultaneousGroup, ready for Stage 3.
    decision_log: everything shown to the LLM + what it chose + why.
    """
    plan:           list                # list[ResolvedStep | SimultaneousGroup]
    final_symbol:   str
    success:        bool
    failure_reason: str = ""
    decision_log:   list[dict] = field(default_factory=list)
    status:         str = "SUCCESS"     # "SUCCESS" | "UNVERIFIED"


# ── Topological sort + cycle detection ───────────────────────────────────────

def _topological_sort(
    steps: list[ResolvedStep],
    given_symbols: set[str],
) -> tuple[list, list[ResolvedStep]]:
    """
    Sort steps so each step's required inputs are available before it runs.
    Returns (sorted_plan, cyclic_steps).
    cyclic_steps is non-empty if a mutual dependency was detected.

    Uses index-based keys internally so dataclass objects need not be hashable.
    """
    if not steps:
        return [], []

    n = len(steps)
    # symbol → index of step that produces it
    produces: dict[str, int] = {steps[i].solves_for.symbol: i for i in range(n)}

    # deps[i] = set of step indices that step i depends on
    deps: list[set[int]] = [set() for _ in range(n)]
    for i, step in enumerate(steps):
        eq_vars = set(step.equation["variables"].keys())
        inputs_needed = eq_vars - {step.solves_for.symbol} - PHYSICAL_CONSTANTS
        for inp in inputs_needed:
            if inp in produces and produces[inp] != i:
                deps[i].add(produces[inp])

    # Kahn's algorithm (index-based)
    in_degree  = [len(deps[i]) for i in range(n)]
    ready      = deque(i for i in range(n) if in_degree[i] == 0)
    sorted_idx = []
    processed  = set()

    while ready:
        i = ready.popleft()
        sorted_idx.append(i)
        processed.add(i)
        for j in range(n):
            if i in deps[j] and j not in processed:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    ready.append(j)

    sorted_out = [steps[i] for i in sorted_idx]
    cyclic     = [steps[i] for i in range(n) if i not in processed]
    return sorted_out, cyclic


def _merge_cycles(
    sorted_steps: list[ResolvedStep],
    cyclic_steps: list[ResolvedStep],
) -> list:
    """
    Produce a final plan that replaces cyclic steps with SimultaneousGroup(s).
    Simple case: all cyclic steps become one group, inserted at the end
    of the sorted non-cyclic steps (they'll be solved together after their
    non-cyclic prerequisites are resolved).
    """
    if not cyclic_steps:
        return sorted_steps

    group = SimultaneousGroup(
        equations=[s.equation for s in cyclic_steps],
        unknowns=[s.solves_for for s in cyclic_steps],
        round_num=max(s.round_num for s in cyclic_steps),
    )
    return sorted_steps + [group]


# ── Main frontier resolution loop ─────────────────────────────────────────────

def resolve_frontier(
    target:         FrontierItem,
    given:          dict[str, dict],  # {symbol: {value, unit, name, dimension}}
    graph_index,
    question:       str,
    llm_round_fn,   # callable(question, available, round_data) → list[dict]
    max_rounds:     int = MAX_CHAIN_DEPTH,
    excluded_eqs:   set[str] | None = None,  # for backtracking: ban specific equations
) -> ResolutionResult:
    """
    Iteratively resolves the target quantity by:
      1. Building a frontier of "needed" quantities.
      2. Asking the LLM (in one batched call per round) to pick equations.
      3. Adding the picked equations' missing inputs to the next frontier.
      4. Topological-sorting the chosen steps for execution order.

    llm_round_fn signature:
        fn(question: str,
           available: dict[str, dict],
           round_data: list[{frontier_item, candidates}]) -> list[dict]
        Each returned dict:
          {frontier_item: FrontierItem,
           chosen_eq: dict | None,
           reason: str,
           conditions_concern: str | None,
           deferred: bool}
    """
    if excluded_eqs is None:
        excluded_eqs = set()

    # State
    available: dict[str, dict] = dict(given)   # symbol → {value,unit,name,dimension}
    frontier:  list[FrontierItem] = [target]
    visited_eqs: set[str] = set(excluded_eqs)
    chosen_steps: list[ResolvedStep] = []
    decision_log: list[dict] = []

    for round_num in range(max_rounds):
        if not frontier:
            break

        # ── Step 1: Generate candidates for each frontier item ────────────────
        round_data = []
        for fi in frontier:
            candidates = graph_index.candidates_for_quantity(
                needed_symbol=fi.symbol,
                needed_name=fi.name,
                needed_dimension=fi.dimension,
                visited_eqs=visited_eqs,
            )
            round_data.append({"frontier_item": fi, "candidates": candidates})

        # ── Step 2: Batched LLM call ──────────────────────────────────────────
        selections = llm_round_fn(
            question=question,
            available=available,
            round_data=round_data,
            round_num=round_num,
        )

        # ── Step 3: Apply picks, build next frontier ──────────────────────────
        new_frontier: list[FrontierItem] = []
        seen_in_frontier: set[str] = set()   # avoid duplicate frontier items

        for sel in selections:
            fi: FrontierItem = sel["frontier_item"]

            if sel.get("deferred", False):
                # LLM says this quantity will be supplied as a byproduct of
                # another equation chosen this round — keep in next frontier
                if fi.symbol not in seen_in_frontier:
                    new_frontier.append(fi)
                    seen_in_frontier.add(fi.symbol)
                continue

            eq: dict = sel.get("chosen_eq")
            if eq is None:
                # No candidate was available; mark unresolvable
                return ResolutionResult(
                    plan=[], final_symbol=target.symbol, success=False,
                    failure_reason=(
                        f"No candidate equations found for '{fi.symbol}' "
                        f"({fi.name}) in round {round_num}."
                    ),
                    decision_log=decision_log,
                    status="UNVERIFIED",
                )

            # Record the step
            eq_vars = set(eq["variables"].keys())
            inputs = [
                s for s in eq_vars
                if s != fi.symbol and s not in PHYSICAL_CONSTANTS
            ]
            step = ResolvedStep(
                equation=eq,
                solves_for=fi,
                inputs_used=inputs,
                round_num=round_num,
                llm_reason=sel.get("reason", ""),
                conditions_concern=sel.get("conditions_concern"),
                is_provisional=bool(sel.get("conditions_concern")),
            )
            chosen_steps.append(step)
            visited_eqs.add(eq["id"])

            # What new quantities does this equation introduce?
            for sym, meta in eq["variables"].items():
                if sym == fi.symbol:
                    continue
                if sym in PHYSICAL_CONSTANTS:
                    continue
                if sym in available:
                    continue
                if sym in seen_in_frontier:
                    continue
                new_fi = FrontierItem(
                    symbol=sym,
                    name=meta.get("name", sym),
                    unit=meta.get("unit", ""),
                    dimension=meta.get("dimension", ""),
                    introduced_by=eq["id"],
                )
                new_frontier.append(new_fi)
                seen_in_frontier.add(sym)

            # Decision log entry
            decision_log.append({
                "round":             round_num,
                "solving_for":       fi.symbol,
                "solving_for_name":  fi.name,
                "chosen_eq_id":      eq["id"],
                "equation_str":      eq["equation_str"],
                "reason":            sel.get("reason", ""),
                "conditions_concern": sel.get("conditions_concern"),
                "n_candidates":      len(sel.get("_candidates", [])),
            })

        frontier = new_frontier

    # If frontier still has items, we couldn't resolve everything
    if frontier:
        unresolved = [f"{fi.symbol} ({fi.name})" for fi in frontier]
        return ResolutionResult(
            plan=[], final_symbol=target.symbol, success=False,
            failure_reason=f"Could not resolve quantities: {unresolved}",
            decision_log=decision_log,
            status="UNVERIFIED",
        )

    # ── Step 4: Topological sort ──────────────────────────────────────────────
    given_syms = set(given.keys())
    sorted_steps, cyclic = _topological_sort(chosen_steps, given_syms)
    plan = _merge_cycles(sorted_steps, cyclic)

    return ResolutionResult(
        plan=plan,
        final_symbol=target.symbol,
        success=True,
        decision_log=decision_log,
        status="SUCCESS",
    )
