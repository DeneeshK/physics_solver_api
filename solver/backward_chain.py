"""
solver/backward_chain.py
Core backward-chaining algorithm.

Given:
  - target_sym:     the variable to solve for (e.g. "F")
  - available_syms: set of symbols already known (e.g. {"rho","V","u","v","s"})
  - graph_index:    GraphIndex (pre-loaded)

Returns an ordered ChainResult — a list of ChainStep, each one an equation
to execute and the variable it computes.  Execute steps in order; the last
step produces target_sym.

Design principles:
  - Symbol-level matching only (never var_id level)
  - Physical constants are never treated as unknowns
  - Output-role equations are preferred over both-role
  - Fewer-variable equations rank higher (simpler)
  - Full decision log is built during traversal for student narration
"""
from __future__ import annotations
from dataclasses import dataclass, field
from config import PHYSICAL_CONSTANTS, MAX_CHAIN_DEPTH

# Conservation-law equations (e.g. P*V^gamma = constant) cannot give a
# direct numerical answer — skip them during backward chaining.
NON_SOLVABLE_SYMBOLS = {'constant'}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ChainStep:
    equation:         dict          # full equation node
    solving_for:      str           # symbol this step computes
    inputs_used:      list[str]     # symbols substituted in
    depth:            int
    decision_note:    str           # why this equation was chosen
    rejected:         list[str] = field(default_factory=list)  # rejected paths


@dataclass
class ChainResult:
    steps:          list[ChainStep]
    final_symbol:   str
    success:        bool
    failure_reason: str = ""

    @property
    def equation_chain(self) -> list[dict]:
        return [s.equation for s in self.steps]

    @property
    def solve_order(self) -> list[tuple[str, str]]:
        """[(equation_str, solving_for), ...]"""
        return [(s.equation["equation_str"], s.solving_for) for s in self.steps]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _can_supply(eq_node: dict, target_sym: str, available: set[str]) -> bool:
    """
    True if this equation contains target_sym AND all other variables
    in the equation are in available_syms.
    Equations containing 'constant' are conservation-law forms that
    cannot be solved for a specific value — skip them.
    """
    eq_vars = set(eq_node["variables"].keys())
    if eq_vars & NON_SOLVABLE_SYMBOLS:       # skip conservation-law equations
        return False
    if target_sym not in eq_vars:
        return False
    inputs_needed = eq_vars - {target_sym} - PHYSICAL_CONSTANTS
    return inputs_needed.issubset(available)


def _score_equation(
    eq_node:       dict,
    graph_index,
    target_sym:    str,
    available:     set[str] | None = None,
) -> tuple[int, int, int]:
    """
    Priority tuple for sorting candidates (lower = better):
      [0] natural-output match (0 = this equation is already solved FOR target_sym,
                                1 = target_sym is on the RHS or implicit)
      [1] inputs used (negated) — MORE available inputs used = more specific match.
          Example: v²=u²+2as uses v,u,s (3 inputs) vs v=u-a uses v,u (2 inputs)
          when solving for a with {v,u,s} available → v²=u²+2as wins.
      [2] total variables — fewer = simpler equation (tiebreaker)

    The natural_output check is TARGET-SPECIFIC:
      Eq(v, u + a*t)  → natural output is 'v'.
        When solving for 'a': role_score=1 (v ≠ a)
        When solving for 'v': role_score=0 (v == v) ← preferred
      This prevents v=u-a from getting priority when solving for a.
    """
    natural_out = graph_index.natural_output.get(eq_node["id"])
    role_score  = 0 if natural_out == target_sym else 1
    complexity  = len(eq_node["variables"])
    if available:
        eq_inputs   = set(eq_node["variables"].keys()) - {target_sym} - PHYSICAL_CONSTANTS
        inputs_used = len(eq_inputs & available)
    else:
        inputs_used = 0
    return (role_score, -inputs_used, complexity)


def _find_direct_suppliers(
    target_sym:   str,
    available:    set[str],
    graph_index,
    visited:      set[str],
) -> list[dict]:
    """Equations that can immediately solve for target_sym from available vars."""
    candidates = [
        eq for eq in graph_index.equations_containing_symbol(target_sym)
        if eq["id"] not in visited
        and _can_supply(eq, target_sym, available)
    ]
    candidates.sort(key=lambda e: _score_equation(e, graph_index, target_sym, available))
    return candidates


def _find_partial_suppliers(
    target_sym:   str,
    available:    set[str],
    graph_index,
    visited:      set[str],
) -> list[tuple[dict, set[str]]]:
    """
    Equations containing target_sym that are not yet fully satisfiable —
    some inputs are still missing. Returned sorted by number of missing
    inputs (fewest missing = try first).
    """
    candidates = []
    for eq in graph_index.equations_containing_symbol(target_sym):
        if eq["id"] in visited:
            continue
        eq_vars = set(eq["variables"].keys())
        inputs  = eq_vars - {target_sym} - PHYSICAL_CONSTANTS
        missing = inputs - available
        if missing:                          # some but not all inputs available
            candidates.append((eq, missing))

    candidates.sort(key=lambda x: (
        _score_equation(x[0], graph_index, target_sym, available)[0],
        len(x[1]),                           # fewer missing inputs first
        len(x[0]["variables"]),              # simpler equation first
    ))
    return candidates


# ── Main algorithm ────────────────────────────────────────────────────────────

def backward_chain(
    target_sym:   str,
    available:    set[str],
    graph_index,
    anchor_id:    str | None = None,
    visited:      set[str]  | None = None,
    depth:        int = 0,
) -> ChainResult:
    """
    Recursively builds a solution chain for target_sym.

    If anchor_id is provided the algorithm starts from that equation
    (the RAG-retrieved anchor) and resolves its missing inputs recursively.
    Otherwise it searches the whole graph.
    """
    if visited is None:
        visited = set()
    if depth > MAX_CHAIN_DEPTH:
        return ChainResult([], target_sym, False, f"Max depth reached at depth {depth}")

    available = set(available)  # local copy

    # ── Case 1: anchor equation provided (RAG-driven entry point) ────────────
    if anchor_id and depth == 0:
        anchor = graph_index.get_equation(anchor_id)
        if anchor is None:
            return ChainResult([], target_sym, False, f"Anchor {anchor_id} not found")

        eq_vars      = set(anchor["variables"].keys())
        inputs_needed = eq_vars - {target_sym} - PHYSICAL_CONSTANTS
        missing       = inputs_needed - available

        if not missing:
            step = ChainStep(
                equation=anchor, solving_for=target_sym,
                inputs_used=list(inputs_needed & available),
                depth=0,
                decision_note=f"Anchor {anchor['equation_str']} — all inputs available directly.",
            )
            return ChainResult([step], target_sym, True)

        # Resolve missing inputs recursively
        return _resolve_anchor(anchor, target_sym, available, missing, graph_index, visited, depth)

    # ── Case 2: search for direct supplier ───────────────────────────────────
    direct = _find_direct_suppliers(target_sym, available, graph_index, visited)
    if direct:
        best = direct[0]
        inputs = list(set(best["variables"].keys()) - {target_sym} - PHYSICAL_CONSTANTS)
        step = ChainStep(
            equation=best, solving_for=target_sym,
            inputs_used=inputs, depth=depth,
            decision_note=(
                f"All inputs ({', '.join(inputs)}) for {best['equation_str']} "
                f"are available — used directly."
            ),
        )
        return ChainResult([step], target_sym, True)

    # ── Case 3: try partial suppliers recursively ─────────────────────────────
    partial = _find_partial_suppliers(target_sym, available, graph_index, visited)
    rejected_paths = []

    for candidate, missing_inputs in partial:
        result = _resolve_anchor(
            candidate, target_sym, available, missing_inputs,
            graph_index, visited | {candidate["id"]}, depth,
        )
        if result.success:
            return result
        rejected_paths.append(f"{candidate['id']} (missing: {missing_inputs})")

    return ChainResult(
        [], target_sym, False,
        f"No valid chain found for '{target_sym}'. Tried: {rejected_paths[:5]}",
    )


def _resolve_anchor(
    anchor:        dict,
    target_sym:    str,
    available:     set[str],
    missing:       set[str],
    graph_index,
    visited:       set[str],
    depth:         int,
) -> ChainResult:
    """
    Given an anchor equation and its missing input symbols,
    recursively build sub-chains for each missing symbol.
    Returns the full ordered chain if successful.
    """
    visited = visited | {anchor["id"]}
    temp_available = set(available)
    all_steps:   list[ChainStep] = []
    rejected:    list[str]       = []

    for missing_sym in sorted(missing):   # sorted for determinism
        sub = backward_chain(
            target_sym=missing_sym,
            available=temp_available,
            graph_index=graph_index,
            visited=visited,
            depth=depth + 1,
        )
        if not sub.success:
            rejected.append(f"{missing_sym}: {sub.failure_reason}")
            return ChainResult(
                [], target_sym, False,
                f"Could not resolve '{missing_sym}' needed by {anchor['id']}. "
                f"Reason: {sub.failure_reason}",
            )
        all_steps.extend(sub.steps)
        temp_available.add(missing_sym)
        visited = visited | {s.equation["id"] for s in sub.steps}

    # All missing inputs resolved — add the anchor step
    inputs_used = list(
        (set(anchor["variables"].keys()) - {target_sym} - PHYSICAL_CONSTANTS)
        & temp_available
    )
    anchor_step = ChainStep(
        equation=anchor, solving_for=target_sym,
        inputs_used=inputs_used, depth=depth,
        decision_note=(
            f"After computing {[s.solving_for for s in all_steps]}, "
            f"all inputs for {anchor['equation_str']} are satisfied."
        ),
        rejected=rejected,
    )
    all_steps.append(anchor_step)
    return ChainResult(all_steps, target_sym, True)


# ── Multi-anchor search ───────────────────────────────────────────────────────

def find_solution_chain(
    target_sym:     str,
    available:      set[str],
    graph_index,
    anchor_candidates: list[dict],
) -> ChainResult:
    """
    Try each RAG-retrieved anchor equation in order.
    Returns the first successful chain, or the last failure.

    anchor_candidates: equation nodes from RAG retrieval + graph expansion,
                       ordered by relevance score (best first).
    """
    # Filter anchors to only those containing the target symbol
    valid_anchors = [
        eq for eq in anchor_candidates
        if target_sym in eq["variables"]
    ]

    if not valid_anchors:
        # Fall back to searching the whole graph
        return backward_chain(target_sym, available, graph_index)

    last_failure = ChainResult([], target_sym, False, "No anchors tried")

    for anchor in valid_anchors:
        result = backward_chain(
            target_sym=target_sym,
            available=available,
            graph_index=graph_index,
            anchor_id=anchor["id"],
        )
        if result.success:
            return result
        last_failure = result

    return last_failure
