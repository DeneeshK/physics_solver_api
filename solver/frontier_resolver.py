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
from solver.solver_log import log


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
    dead_end_root_eq: str = ""          # v7.2.2: the Round-0 equation whose
    #                                     chain dead-ended, so the caller can
    #                                     exclude it and try the next Round-0
    #                                     node (the "drop node, go to next of
    #                                     top-5" rollback).


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

def _deepest_culprit_on_branch(failed_fi, chosen_steps: list) -> str:
    """
    v7.2.5 — pick the equation to exclude when chasing `failed_fi` dead-ends.

    The bad pick is the DEEPEST equation on the branch that led to this
    dead-end — the one whose own required inputs we ultimately couldn't
    satisfy — NOT the shallow equation that first asked for `failed_fi`, and
    never the round-0 landing if there's anything deeper to blame. This keeps a
    correct landing (e.g. K=½mv²) committed while the retry re-picks the
    genuinely faulty deeper step.

    We reconstruct the branch by following `introduced_by` links: the failed
    item was introduced by some equation; that equation was chosen while
    solving some variable, which was itself introduced by an earlier equation,
    and so on back toward the landing. Among the equations actually CHOSEN on
    that branch (i.e. present in chosen_steps), return the one with the highest
    round_num that is not the sole round-0 landing. Returns "" if the only
    candidate is the landing itself (caller then falls back).
    """
    if not chosen_steps:
        return ""
    by_eqid = {s.equation["id"]: s for s in chosen_steps}

    # Walk the branch: start from the equation that introduced the failed item,
    # then hop to the step that introduced THAT equation's solved symbol, etc.
    branch_steps = []
    seen = set()
    eqid = getattr(failed_fi, "introduced_by", None)
    while eqid and eqid in by_eqid and eqid not in seen:
        seen.add(eqid)
        step = by_eqid[eqid]
        branch_steps.append(step)
        # Hop up: which step introduced the variable THIS step solves for?
        intro = None
        for s in chosen_steps:
            if s.solves_for.symbol == step.solves_for.symbol:
                intro = getattr(s.solves_for, "introduced_by", None)
                break
        eqid = intro if intro and intro != eqid else None

    if not branch_steps:
        return ""
    # Prefer the deepest (highest round) non-landing equation on the branch.
    non_landing = [s for s in branch_steps if s.round_num > 0]
    pick_from = non_landing or branch_steps
    deepest = max(pick_from, key=lambda s: s.round_num)
    # Never return a round-0 landing if a deeper step exists on the branch.
    if deepest.round_num == 0 and non_landing:
        deepest = max(non_landing, key=lambda s: s.round_num)
    return deepest.equation["id"]


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
    from solver.landing       import (get_landing_candidates,
                                      get_landing_candidates_semantic,
                                      get_neighbor_candidates)
    from config               import MAX_CANDIDATES_TOKENS_PER_ROUND, RAG_TOP_K

    # v7.2.4: how many top-similarity nodes the Round-0 sequential scan walks.
    # The scan asks the LLM "does this concept fit?" per node, top-down, and
    # commits to the first that fits. Kept small (the right starting concept
    # for a well-formed question is reliably in the top handful).
    LANDING_TOP_K = max(RAG_TOP_K, 8)

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

        round_data = []
        # Currently-known symbols — "what we already have" for the
        # knowns-overlap ranking (Round 1+) and the state document.
        known_symbols = {sym for sym, meta in available.items()
                         if meta.get("value") is not None}
        # ── Steps 1+2 differ by round ────────────────────────────────────────
        # v7.2.4: Round 0 (initial landing) is now a SEQUENTIAL SCAN down the
        # pure-semantic ranked list — the user's design. We retrieve the top_k
        # most similar equations (vector search only; no symbol-table source,
        # no domain filter) and walk them ONE AT A TIME in similarity order,
        # asking the LLM "does this equation's concept match the question?" per
        # node. The first node the LLM accepts becomes the landing equation;
        # rejected nodes are skipped; if the committed chain later dead-ends,
        # the node is excluded and the next attempt resumes the scan further
        # down the (stable) list. Round 1+ keeps the graph neighbor-walk with
        # the batched selector call.
        # v7.2.5: the STATE DOCUMENT carried into every decision (the user's
        # design). It always reflects the live solve: the ultimate goal, the
        # variables we ALREADY HAVE (givens + everything solved so far, with
        # names so the model knows rho=density, V=volume, ...), the committed
        # steps, and the unknowns STILL LEFT to solve. The available list only
        # grows — a known is never removed just because an equation used it
        # (it stays reusable, which handles a variable feeding several
        # sub-problems). `remaining` shrinks as each unknown is resolved.
        known_named = [
            {"symbol": sym, "name": meta.get("name", sym)}
            for sym, meta in available.items()
            if meta.get("value") is not None
            and sym not in PHYSICAL_CONSTANTS
        ]
        remaining_unknowns = [
            {"symbol": fi.symbol, "name": fi.name} for fi in frontier
        ]
        solve_context = {
            "goal_symbol":  target.symbol,
            "goal_name":    target.name,
            "chosen_steps": [
                {"symbol": s.solves_for.symbol, "eq_str": s.equation["equation_str"]}
                for s in chosen_steps
            ],
            "being_solved": sorted({s.solves_for.symbol for s in chosen_steps}),
            "available":    known_named,
            "remaining":    remaining_unknowns,
        }

        if round_num == 0:
            fi = frontier[0]  # Round 0 always has exactly one item: the target
            ranked = get_landing_candidates_semantic(
                retriever    = retriever,
                search_query = search_query,
                visited_eqs  = visited_eqs,
                top_k        = LANDING_TOP_K,
            )
            log("landing_scan_start", target=fi.symbol,
                n_ranked=len(ranked),
                ranked_ids=[c["id"] for c in ranked])
            selections = []
            scan_pos = 0
            for scan_pos, cand in enumerate(ranked):
                # Ask the LLM about THIS ONE node only: does its concept fit?
                one = [{"frontier_item": fi, "candidates": [cand]}]
                sel = llm_round_fn(
                    question=question,
                    available=available,
                    round_data=one,
                    round_num=round_num,
                    solve_context=solve_context,
                )
                picked = sel and sel[0].get("chosen_eq") is not None
                log("landing_scan_node",
                    rank=scan_pos,
                    eq_id=cand["id"],
                    score=cand.get("_retrieval_score"),
                    decision="pick" if picked else "reject",
                    reason=(sel[0].get("reason", "") if sel else ""))
                if picked:
                    selections = sel
                    break
                # rejected → continue down the ranked list to the next node
            if not selections:
                # Walked the whole ranked list; the LLM accepted none. For a
                # well-formed question the right concept is in the top_k, so
                # this means genuinely no equation fits (e.g. underspecified
                # question) → fail UNVERIFIED. No node to exclude.
                reason_for_failure = (
                    f"Landing scan rejected all {len(ranked)} retrieved "
                    f"candidate(s) for '{fi.symbol}' — no equation's concept "
                    f"matched the question."
                    if ranked else
                    f"No equations retrieved for '{fi.symbol}' "
                    f"(empty semantic result)."
                )
                log("landing_scan_exhausted",
                    target=fi.symbol, n_ranked=len(ranked))
                return ResolutionResult(
                    plan=[], final_symbol=target.symbol, success=False,
                    failure_reason=reason_for_failure,
                    decision_log=decision_log,
                    status="UNVERIFIED",
                    dead_end_root_eq="",
                )
        else:
            # ── Round 1+ : graph neighbor-walk + batched selector call ─────────
            round_data = []
            for fi in frontier:
                # GRAPH NEIGHBOR WALK — chasing `fi.symbol`, a variable some
                # already-chosen equation introduced. Candidates = graph
                # neighbors of the chosen equations that SHARE this variable.
                walk_from = {s.equation["id"] for s in chosen_steps}
                if fi.introduced_by:
                    walk_from.add(fi.introduced_by)
                candidates = get_neighbor_candidates(
                    graph_index   = graph_index,
                    target_symbol = fi.symbol,
                    from_eq_ids   = walk_from,
                    visited_eqs   = visited_eqs,
                    search_query  = search_query,
                    retriever     = retriever,
                    known_symbols = known_symbols,
                )
                round_data.append({"frontier_item": fi, "candidates": candidates})

            if len(round_data) > 1 and estimate_round_tokens(round_data) > MAX_CANDIDATES_TOKENS_PER_ROUND:
                selections = []
                for rd in round_data:
                    selections.extend(llm_round_fn(
                        question=question,
                        available=available,
                        round_data=[rd],
                        round_num=round_num,
                        solve_context=solve_context,
                    ))
            else:
                selections = llm_round_fn(
                    question=question,
                    available=available,
                    round_data=round_data,
                    round_num=round_num,
                    solve_context=solve_context,
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
                # v7.2.5: report the SPECIFIC equation to exclude so the
                # pipeline re-picks the actual culprit, NEVER the good landing.
                # When chasing `fi` dead-ends, the bad pick is the DEEPEST
                # equation on the failed branch — the one whose own inputs we
                # couldn't meet — not the shallow equation that merely asked
                # for `fi`. The previous logic blamed `fi.introduced_by`, which
                # for a sub-variable directly under the landing node IS the
                # landing node, so a deep dead-end (e.g. K=½mv² → v → u → s, and
                # `s` can't close) wrongly banned K=½mv² and it vanished from the
                # next retrieval. We instead walk the chosen steps along the
                # branch that introduced `fi` and exclude the deepest one, so
                # the retry re-picks THAT step and keeps the correct landing +
                # all good earlier steps. `fi.introduced_by` is the fallback for
                # a direct first-level miss; the landing node is excluded only
                # when it is itself the sole step (nothing deeper to blame).
                culprit_eq = _deepest_culprit_on_branch(fi, chosen_steps)
                if not culprit_eq:
                    culprit_eq = fi.introduced_by or ""
                if not culprit_eq:
                    for s in chosen_steps:
                        if s.round_num == 0:
                            culprit_eq = s.equation["id"]
                            break
                return ResolutionResult(
                    plan=[], final_symbol=target.symbol, success=False,
                    failure_reason=reason_for_failure,
                    decision_log=decision_log,
                    status="UNVERIFIED",
                    dead_end_root_eq=culprit_eq,
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
