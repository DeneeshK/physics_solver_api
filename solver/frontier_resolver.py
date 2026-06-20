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
    allowed_domains: set[str] | None = None,  # narrow candidates by domain (with fallback)
    search_query:   str = "",                 # v7: Stage 1 scenario sentence for ChromaDB
    retriever       = None,                   # v7: optional Retriever; None → symbol-only landing
) -> ResolutionResult:
    """
    Iteratively resolves the target quantity by:
      1. ROUND 0 — landing. Combine symbol-table lookup (v6 behavior) with
         optional ChromaDB semantic lookup (new in v7) to surface candidates
         for the target unknown. The LLM picks one based on physical fit.
      2. ROUND 1+ — expansion. For each chosen equation's missing inputs,
         look them up by symbol from the graph (deterministic, no ChromaDB),
         dimension-filtered. The LLM picks one per missing input.
      3. Topologically sort the chosen steps for execution order.

    llm_round_fn signature:
        fn(question: str,
           available: dict[str, dict],
           round_data: list[{frontier_item, candidates}]) -> list[dict]
        Each returned dict:
          {frontier_item: FrontierItem,
           chosen_eq: dict | None,
           reason: str,
           conditions_concern: str | None,
           deferred: bool,
           fallback_used: str | None}

    allowed_domains: optional hint (e.g. {'laws_of_motion', 'kinematics'})
    used to shrink the candidate set shown to the LLM. Safe by construction:
    graph_index.candidates_for_quantity() falls back to the full candidate
    set whenever the domain filter would otherwise leave zero options.

    search_query / retriever: v7 ChromaDB landing. If retriever is None or
    search_query is empty, round 0 falls back to symbol-only (v6 behavior).
    """
    # Lazy imports — avoids circular import (llm_interface imports
    # FrontierItem from this module), and lets the module be imported
    # cheaply in test contexts that don't exercise these paths.
    from solver.llm_interface import estimate_round_tokens
    from solver.landing       import get_landing_candidates
    from config               import MAX_CANDIDATES_TOKENS_PER_ROUND

    if excluded_eqs is None:
        excluded_eqs = set()

    # State
    available: dict[str, dict] = dict(given)
    frontier:  list[FrontierItem] = [target]
    visited_eqs: set[str] = set(excluded_eqs)
    chosen_steps: list[ResolvedStep] = []
    decision_log: list[dict] = []
    already_targeted: set[str] = set()

    for round_num in range(max_rounds):
        if not frontier:
            break

        # ── Step 1: Generate candidates for each frontier item ────────────────
        # Round 0 (initial landing) uses the unified landing layer: symbol
        # candidates UNIONED with semantic candidates from ChromaDB (if
        # available). Round 1+ uses the same landing layer but with no
        # semantic query (symbol-only), AND with knowns-overlap re-ranking
        # enabled — equations that share more variables with what we
        # already have bubble to the top, so the bridging equation
        # (e.g. rho=m/V when we need m and have rho, V) ranks first.
        # v7.1.4: pass known_symbols + round_num for the re-ranking step.
        round_data = []
        # Compute the set of currently-known symbols (those with a resolved
        # value in `available`). This is what we treat as "what we already
        # have" for the overlap-ranking logic.
        known_symbols = {sym for sym, meta in available.items()
                         if meta.get("value") is not None}
        for fi in frontier:
            if round_num == 0:
                candidates = get_landing_candidates(
                    graph_index      = graph_index,
                    target_symbol    = fi.symbol,
                    target_name      = fi.name,
                    target_dimension = fi.dimension,
                    search_query     = search_query,
                    visited_eqs      = visited_eqs,
                    allowed_domains  = allowed_domains,
                    retriever        = retriever,
                    known_symbols    = known_symbols,
                    round_num        = round_num,
                )
            else:
                # Round 1+: symbol-only landing (no fresh semantic query —
                # the question's Stage 1 search_query was about the original
                # unknown, not the current frontier item). The overlap
                # re-ranking promotes bridging equations.
                candidates = get_landing_candidates(
                    graph_index      = graph_index,
                    target_symbol    = fi.symbol,
                    target_name      = fi.name,
                    target_dimension = fi.dimension,
                    search_query     = "",  # disables semantic step
                    visited_eqs      = visited_eqs,
                    allowed_domains  = allowed_domains,
                    retriever        = None,  # disables semantic step
                    known_symbols    = known_symbols,
                    round_num        = round_num,
                )
            round_data.append({"frontier_item": fi, "candidates": candidates})

        # ── Step 2: Batched LLM call — split if this round risks oversize ─────
        if len(round_data) > 1 and estimate_round_tokens(round_data) > MAX_CANDIDATES_TOKENS_PER_ROUND:
            selections = []
            for rd in round_data:
                selections.extend(llm_round_fn(
                    question=question,
                    available=available,
                    round_data=[rd],
                    round_num=round_num,
                ))
        else:
            selections = llm_round_fn(
                question=question,
                available=available,
                round_data=round_data,
                round_num=round_num,
            )

        # ── Step 3: Apply picks, build next frontier ──────────────────────────
        new_frontier: list[FrontierItem] = []
        seen_in_frontier: set[str] = set()

        already_targeted |= {
            sel["frontier_item"].symbol for sel in selections
            if not sel.get("deferred", False) and sel.get("chosen_eq") is not None
        }

        for sel in selections:
            fi: FrontierItem = sel["frontier_item"]
            # v7: pull the candidate list this LLM call actually saw, for
            # decision_log. Lets Stage 4 honestly narrate rejected alternatives
            # and lets debugging trace what was visible vs. what was picked.
            candidates_shown = sel.get("_candidates", [])

            if sel.get("deferred", False):
                if fi.symbol not in seen_in_frontier:
                    new_frontier.append(fi)
                    seen_in_frontier.add(fi.symbol)
                # Still log the deferral so Stage 4 can see why this round
                # didn't directly resolve this symbol.
                decision_log.append({
                    "round":              round_num,
                    "solving_for":        fi.symbol,
                    "solving_for_name":   fi.name,
                    "chosen_eq_id":       None,
                    "equation_str":       None,
                    "reason":             sel.get("reason", "deferred"),
                    "conditions_concern": None,
                    "fallback_used":      sel.get("fallback_used"),
                    "n_candidates":       len(candidates_shown),
                    "candidates_shown": [
                        {"id": c["id"], "equation_str": c["equation_str"],
                         "landing_source": c.get("landing_source")}
                        for c in candidates_shown
                    ],
                    "decision":           "defer",
                })
                continue

            eq: dict = sel.get("chosen_eq")
            if eq is None:
                # No equation was picked — either no candidates at all, or
                # the LLM said "none fits". Log what the LLM saw, then bail.
                fb = sel.get("fallback_used") or ""
                reason_for_failure = (
                    f"No candidate equations found for '{fi.symbol}' "
                    f"({fi.name}) in round {round_num}."
                    if not candidates_shown else
                    f"LLM rejected all {len(candidates_shown)} candidate(s) "
                    f"for '{fi.symbol}' in round {round_num} ({fb or 'no-fit'})."
                )
                decision_log.append({
                    "round":              round_num,
                    "solving_for":        fi.symbol,
                    "solving_for_name":   fi.name,
                    "chosen_eq_id":       None,
                    "equation_str":       None,
                    "reason":             sel.get("reason", reason_for_failure),
                    "conditions_concern": None,
                    "fallback_used":      sel.get("fallback_used"),
                    "n_candidates":       len(candidates_shown),
                    "candidates_shown": [
                        {"id": c["id"], "equation_str": c["equation_str"],
                         "landing_source": c.get("landing_source")}
                        for c in candidates_shown
                    ],
                    "decision":           "none",
                })
                return ResolutionResult(
                    plan=[], final_symbol=target.symbol, success=False,
                    failure_reason=reason_for_failure,
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
                if sym in already_targeted:
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

            # Decision log entry — now includes everything the LLM saw,
            # not just what it picked.
            decision_log.append({
                "round":              round_num,
                "solving_for":        fi.symbol,
                "solving_for_name":   fi.name,
                "chosen_eq_id":       eq["id"],
                "equation_str":       eq["equation_str"],
                "reason":             sel.get("reason", ""),
                "conditions_concern": sel.get("conditions_concern"),
                "fallback_used":      sel.get("fallback_used"),
                "n_candidates":       len(candidates_shown),
                "candidates_shown": [
                    {"id": c["id"], "equation_str": c["equation_str"],
                     "landing_source": c.get("landing_source")}
                    for c in candidates_shown
                ],
                "decision":           "pick",
            })

        frontier = new_frontier

    if frontier:
        unresolved = [f"{fi.symbol} ({fi.name})" for fi in frontier]
        return ResolutionResult(
            plan=[], final_symbol=target.symbol, success=False,
            failure_reason=f"Could not resolve quantities: {unresolved}",
            decision_log=decision_log,
            status="UNVERIFIED",
        )

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
