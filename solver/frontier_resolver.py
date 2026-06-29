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
import re as _re
from collections import deque
from dataclasses import dataclass, field
from config import PHYSICAL_CONSTANTS, MAX_CHAIN_DEPTH
from solver.solver_log import log


# ── Binding helpers: OBEY the LLM, never infer physics ────────────────────────
# The LLM maps an equation onto the problem and tells us, per variable, where its
# value comes from. These helpers resolve the source the LLM NAMED against the
# state we actually showed it. They do NOT compare dimensions, units, or guess a
# physical relationship — they only connect the model's own reference back to an
# available quantity (by symbol OR by the quantity-name shown in the state doc),
# and recognize a literal value the model bound (e.g. a term that is zero here).

def _resolve_source(stated, available: dict) -> str | None:
    """Map the LLM's STATED source for an equation variable onto a real key in
    `available`. The small model is sloppy about HOW it names the source — it may
    write the canonical symbol ('k'), the quantity name ('resistivity'), a phrase
    that leads with the symbol ('k (force constant of spring)'), or even the VALUE
    ('200 N/m'). All of these are references to things WE showed it, so we resolve
    every form. This is NOT concept matching — it only connects the model's own
    reference back to an available quantity. Returns the available symbol or None.
    """
    if stated is None:
        return None
    s = str(stated).strip()
    if not s:
        return None

    def _by_symbol_or_name(text: str) -> str | None:
        if text in available:
            return text
        t = text.lower()
        for sym, meta in available.items():
            if sym.lower() == t or str(meta.get("name", "")).strip().lower() == t:
                return sym
        return None

    # 1) exact: symbol or quantity-name
    hit = _by_symbol_or_name(s)
    if hit:
        return hit
    # 2) leading identifier token — handles "k (force constant of spring)",
    #    "k_spring symbol", "R, the resistance", etc.
    m = _re.match(r"[A-Za-z_]\w*", s)
    if m and m.group(0) != s:
        hit = _by_symbol_or_name(m.group(0))
        if hit:
            return hit
    # 3) the model wrote the VALUE ('200 N/m', '0.5 kg') — connect to the given
    #    whose numeric value matches. Symbol/name routes are tried first, so this
    #    only fires when the model gave no usable symbol. Last-resort; first match.
    #    GUARD: only when the string is a plain number+unit, NOT an inline formula
    #    like '2*pi*f' or 'sqrt(R^2+XL^2)' (those carry a stray digit but mean
    #    "derive this" — they must fall through to becoming a sub-goal).
    is_value = bool(_re.match(r"^\s*[+-]?\d", s)) and not any(
        tok in s for tok in ("*", "(", ")", "sqrt", "/(",)) and "+" not in s.lstrip("+-")
    vm = _re.search(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", s) if is_value else None
    if vm:
        try:
            val = float(vm.group(0))
        except ValueError:
            val = None
        # val == 0 is NOT a reference to a zero-valued given — it means the term
        # is literally zero (handled by _as_literal). Don't value-match zeros.
        if val is not None and val != 0:
            for sym, meta in available.items():
                av = meta.get("value")
                if isinstance(av, (int, float)) and not isinstance(av, bool) \
                        and av != 0 and abs(av - val) <= 1e-9 * max(abs(av), abs(val), 1.0):
                    return sym
    return None


def _as_literal(stated) -> float | None:
    """If the LLM bound a variable to a bare number (e.g. "X_C": 0, meaning the
    term vanishes in THIS scenario), return it as a float. Else None. This lets
    the model say 'this equation term is zero here' as part of its binding."""
    if stated is None or isinstance(stated, bool):
        return None
    try:
        return float(str(stated).strip())
    except (TypeError, ValueError):
        return None


def _pick_output_var(eq: dict, llm_solves_for, fi_symbol: str) -> str:
    """Decide which equation variable to actually solve for — robustly, with NO
    matching. Priority:
      1. the LLM's `solves_for`, if it is a real variable of this equation;
      2. the frontier item's own symbol, if it is a variable of this equation
         (a sub-goal is always a variable of the equation chosen to produce it);
      3. the equation's DECLARED natural output (authored in the graph) — this is
         what fixes a malformed `solves_for` like '1/Req': we fall back to the
         equation's own stated output 'R', never to an arbitrary variable;
      4. last resort: the LHS symbol of the equation string.
    The old code's `next(iter(eq_vars))` could pick ANY variable, which silently
    solved the wrong quantity (e.g. -6.0 Ω). We never do that anymore."""
    eq_vars = set(eq.get("variables", {}).keys())
    sf = (llm_solves_for or "").strip() if isinstance(llm_solves_for, str) else ""
    if sf in eq_vars:
        return sf
    if fi_symbol in eq_vars:
        return fi_symbol
    nat = (eq.get("natural_output") or eq.get("output") or "").strip()
    if nat in eq_vars:
        return nat
    lhs = eq.get("equation_str", "").split("=")[0].strip()
    if lhs in eq_vars:
        return lhs
    return next(iter(eq_vars), fi_symbol)


def _pick_produces_target(sel: dict, fi_symbol: str) -> bool:
    """Does the LLM say the chosen equation actually PRODUCES the quantity we
    asked it to solve this round? Structural only — we read the navigator's own
    binding and compare symbols; NO dimension/name/semantic matching.

    The navigator names the variable it produces in `solves_for` and lists the
    quantities still missing in `needed`. The tell of a STEPPING-STONE pick is
    that it lists the very quantity we asked for (`fi_symbol`) as still-needed:
    it is saying "this equation gives some OTHER variable, the thing you want is
    still open." Observed verbatim from the 7B:
      - dielectric cap:  solves_for="C",   needed=["Q"]   (target Q = charge)
      - flux loop:       solves_for="Phi", needed=["ε"]   (target ε = emf)
    In both the code used to rename C→Q / Phi→ε and TERMINATE, returning the
    wrong physical quantity (farads as charge, webers as emf) with HIGH
    confidence. Honouring the navigator's `needed` flag here is the fix; we obey
    it instead of overriding it.

    A pick that explicitly sets solves_for to fi_symbol is always treated as
    producing it (covers the rare case where the model redundantly echoes the
    target into `needed`).

    Two stepping-stone tells, both structural:
      (a) the TARGET we asked for is itself still listed in `needed`
          (solves_for="C", needed=["Q"]  — "gives C, still need charge Q");
      (b) the equation's OWN output is listed in `needed`
          (solves_for="C", needed=["C"]  — the model is signalling a downstream
          step, e.g. reason "...needed to find Q via Q=C*V"). A var cannot be
          both produced by and missing from the same equation, so this only ever
          means "this is a prerequisite, not the final equation for fi"."""
    solves_for = (sel.get("solves_for") or "").strip()
    if solves_for and solves_for == fi_symbol:
        return True
    needed = sel.get("needed") or []
    if isinstance(needed, dict):
        needed = list(needed.keys())
    needed_syms = {str(n).strip() for n in needed}
    if fi_symbol in needed_syms:                       # (a)
        return False
    if solves_for and solves_for in needed_syms:       # (b)
        return False
    return True


def _solvable_closure(known_syms: set[str], graph_index) -> set[str]:
    """Forward solvability closure — the LOOKAHEAD the LLM structurally lacks.

    Starting from the known quantities (givens + everything solved so far) plus
    the physical constants, repeatedly add any variable that SOME equation can
    OUTPUT because all of that equation's OTHER variables are already in the set.
    The fixpoint is every quantity the graph can, in principle, produce from what
    we have.

    This is a STRUCTURAL graph fact — pure reachability, no dimensions, no
    concept matching, no equation-to-equation comparison. It is used ONLY to
    ADVISE the LLM (annotate candidates as obtainable / dead-end); the LLM still
    chooses. A variable left OUT of the closure cannot be reached from the
    current knowns by any chain, so an equation that needs it is a dead branch.
    """
    reachable = set(known_syms) | set(PHYSICAL_CONSTANTS)
    nodes = getattr(graph_index, "nodes", None) or []
    changed = True
    while changed:
        changed = False
        for node in nodes:
            vs = set(node.get("variables", {}).keys())
            if not vs:
                continue
            for v in vs:
                if v not in reachable and (vs - {v}) <= reachable:
                    reachable.add(v)
                    changed = True
    return reachable


def _annotate_reachability(candidates: list, output_sym: str,
                           known_syms: set, closure: set) -> None:
    """Tag each candidate equation in place with `_dead_end_inputs`: the inputs
    that CANNOT be obtained from the current knowns (not in the closure, not the
    quantity this equation would solve, not a known, not a constant). Choosing
    such a candidate strands the chain on an unreachable quantity. The LLM is
    shown this so it can avoid the dead branch — it is advice, not a filter."""
    for cand in candidates:
        dead = [
            v for v in cand.get("variables", {})
            if v not in closure and v not in known_syms
            and v != output_sym and v not in PHYSICAL_CONSTANTS
        ]
        cand["_dead_end_inputs"] = dead


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
    One chosen equation in the solution plan, as bound by the LLM.

    v8 — the LLM tells us how the equation maps onto the problem; the code does
    NO symbol/dimension matching:
      output_var       : the equation variable to solve for (LLM "solves_for").
      resolves_symbol  : the quantity-key this step's result is stored under
                         (the frontier item it resolves). May differ from
                         output_var when the equation names it differently.
      value_binding    : {equation_variable: source_key} — for each input
                         variable, the key (a given symbol, a constant, or
                         another step's resolves_symbol) whose value to
                         substitute. This is the LLM's `known`/`needed` mapping.
    """
    equation:          dict
    solves_for:        FrontierItem
    inputs_used:       list[str]
    round_num:         int
    llm_reason:        str
    output_var:        str = ""
    resolves_symbol:   str = ""
    value_binding:     dict = field(default_factory=dict)
    literal_values:    dict = field(default_factory=dict)  # {eq_var: number} — terms
    #                          the LLM bound to a literal (e.g. a vanishing term = 0)
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
    dead_end_root_eq: str = ""          # the DEEPEST (round>0) equation to ban so
    #                                     the retry re-picks an alternative sub-eq
    #                                     for the same sub-goal. "" means there is
    #                                     no deeper culprit — drop the whole node.
    landing_eq_id:   str = ""           # the Round-0 landing equation. The caller
    #                                     drops THIS (clean slate) only when there
    #                                     is no deeper culprit left — the "go to
    #                                     the next top-5 node" recovery.


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
    # resolves_symbol → index of the step that produces that quantity's value
    produces: dict[str, int] = {steps[i].resolves_symbol: i for i in range(n)}

    # deps[i] = set of step indices that step i depends on. A step needs every
    # value_binding SOURCE that another step produces (the LLM's binding tells us
    # the dependency graph directly — no symbol scanning).
    deps: list[set[int]] = [set() for _ in range(n)]
    for i, step in enumerate(steps):
        for src in set(step.value_binding.values()) - PHYSICAL_CONSTANTS:
            if src in produces and produces[src] != i:
                deps[i].add(produces[src])

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
    # Only ever blame a DEEPER (round > 0) equation. The round-0 landing is the
    # LLM's concept choice for the whole problem; a deep sub-goal that can't
    # close is NOT a reason to ban it (doing so cascades: ban a deep eq, the
    # intermediate it produced loses its only source, then the landing gets
    # blamed and re-landing rejects everything). If nothing deeper is to blame,
    # return "" — the caller then drops the WHOLE node (the landing) with a
    # clean slate, the user's "try the next top-5" recovery, instead of
    # accumulating bans that doom every subsequent attempt.
    non_landing = [s for s in branch_steps if s.round_num > 0]
    if not non_landing:
        return ""
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
    # The symbol the final answer is stored under. The round-0 step resolves the
    # target and stores its value under target.symbol (resolves_symbol), even
    # when the equation solved a differently-named variable (output_var), so the
    # final answer is always keyed by what Stage 1 called the unknown.
    resolved_target_symbol: str = target.symbol
    # The Round-0 landing equation (the LLM's concept choice for the whole
    # problem). Never banned by a deep dead-end; only dropped as a whole node.
    landing_eq_id: str = ""

    for round_num in range(max_rounds):
        if not frontier:
            break

        round_data = []
        # Currently-known symbols — "what we already have" for the
        # knowns-overlap ranking (Round 1+) and the state document.
        known_symbols = {sym for sym, meta in available.items()
                         if meta.get("value") is not None}
        # Reachability lookahead for THIS round: every quantity the graph can
        # produce from what we currently know. Candidates needing anything
        # outside this set are flagged dead-end branches for the LLM.
        reach_closure = _solvable_closure(known_symbols, graph_index)
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
                _annotate_reachability([cand], fi.symbol, known_symbols, reach_closure)
                one = [{"frontier_item": fi, "candidates": [cand]}]
                sel = llm_round_fn(
                    question=question,
                    available=available,
                    round_data=one,
                    round_num=round_num,
                    solve_context=solve_context,
                )
                picked = sel and sel[0].get("chosen_eq") is not None
                # Stepping-stone guard: the LLM may "pick" an equation that, by
                # its OWN binding, produces a different quantity and still lists
                # our target in `needed` (e.g. dielectric cap gives C, target is
                # Q). Such a node does not land the target — keep scanning so a
                # later candidate that actually produces it (e.g. q=C*V) is found.
                stepping_stone = picked and not _pick_produces_target(sel[0], fi.symbol)
                log("landing_scan_node",
                    rank=scan_pos,
                    eq_id=cand["id"],
                    score=cand.get("_retrieval_score"),
                    decision=("pick" if picked and not stepping_stone
                              else "reject_stepping_stone" if stepping_stone
                              else "reject"),
                    reason=(sel[0].get("reason", "") if sel else ""))
                if picked and not stepping_stone:
                    selections = sel
                    break
                # rejected (or stepping-stone) → continue down the ranked list
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
                _annotate_reachability(candidates, fi.symbol, known_symbols, reach_closure)
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

            # ── Resolve WITHOUT an equation: the LLM says this variable is
            # already known, or is zero/absent in this scenario ───────────────
            # The pick-time binding can under-bind a variable (the 7B sometimes
            # omits it from `known`), so it surfaces here as a sub-goal. The LLM
            # then correctly tells us it needs no equation: it's already in the
            # state, or it vanishes here. We OBEY that — back-patch the equation
            # that introduced the variable and move on. This is NOT a dead-end;
            # banning the (correct) introducing equation over it was the bug.
            resolve_known = sel.get("resolve_known_source")
            resolve_zero  = sel.get("resolve_zero")
            if (resolve_known or resolve_zero) and fi.introduced_by:
                intro_step = next(
                    (s for s in chosen_steps
                     if s.equation["id"] == fi.introduced_by), None)
                how = None
                if intro_step is not None:
                    if resolve_zero:
                        intro_step.literal_values[fi.symbol] = 0.0
                        intro_step.value_binding[fi.symbol]  = fi.symbol
                        how = "zero"
                    else:
                        resolved = (_resolve_source(resolve_known, available)
                                    or _resolve_source(fi.symbol, available))
                        if resolved is not None:
                            intro_step.value_binding[fi.symbol] = resolved
                            how = f"known={resolved}"
                if how is not None:
                    log("subgoal_resolved_without_equation",
                        round_num=round_num, symbol=fi.symbol,
                        introduced_by=fi.introduced_by, how=how,
                        reason=(sel.get("reason", "") or "")[:200])
                    decision_log.append({
                        "round": round_num, "solving_for": fi.symbol,
                        "solving_for_name": fi.name, "chosen_eq_id": None,
                        "equation_str": None, "reason": sel.get("reason", ""),
                        "conditions_concern": None, "fallback_used": None,
                        "n_candidates": len(candidates_shown),
                        "candidates_shown": [],
                        "decision": "resolved_zero" if resolve_zero else "resolved_known",
                    })
                    continue  # variable accounted for — not a sub-goal, not a dead-end
                # else: couldn't back-patch (no intro step, or unresolved source)
                # → fall through to the normal no-pick / dead-end handling below.

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
                # Report a DEEP culprit to ban so the retry re-picks an
                # alternative sub-equation for the SAME sub-goal — keeping the
                # correct landing committed. `fi.introduced_by` (a round>0 step)
                # is the fallback for a direct miss. We NEVER ban the round-0
                # landing here: if there is no deeper culprit, dead_end_root_eq
                # stays "" and we hand the caller the landing id separately, so
                # it can drop the WHOLE node with a clean slate (the user's
                # "next top-5" recovery) instead of cascading bans.
                culprit_eq = _deepest_culprit_on_branch(fi, chosen_steps)
                if not culprit_eq:
                    intro = fi.introduced_by or ""
                    # only accept the fallback if it is a deeper (non-landing) step
                    if intro and intro != landing_eq_id:
                        culprit_eq = intro
                return ResolutionResult(
                    plan=[], final_symbol=target.symbol, success=False,
                    failure_reason=reason_for_failure,
                    decision_log=decision_log,
                    status="UNVERIFIED",
                    dead_end_root_eq=culprit_eq,
                    landing_eq_id=landing_eq_id,
                )

            eq_vars = set(eq["variables"].keys())

            # ── Enforce the working-memory rule (NO matching — structural) ─────
            # An equation chosen to solve a SUB-goal must not pull back in a
            # quantity that is still being solved upstream (the ultimate goal, or
            # a committed step's not-yet-computed output). That creates a circular
            # dependency the solver collapses to a degenerate 0 (the bogus
            # "simultaneous → 0.0" failures). This is exactly the user's rule:
            # "the equations not to use are the ones that contain the unknown."
            # We reject the pick and hand this equation back as the culprit, so
            # the pipeline bans it and the LLM re-picks a different neighbour.
            if fi.symbol != target.symbol:  # never applies to the round-0 target itself
                being_solved = ({target.symbol}
                                | {s.resolves_symbol for s in chosen_steps}
                                | {s.output_var for s in chosen_steps})
                reintroduced = (eq_vars & being_solved) - {fi.symbol}
                if reintroduced:
                    log("subgoal_pick_rejected_cycle",
                        round_num=round_num, symbol=fi.symbol,
                        eq_id=eq["id"], reintroduces=sorted(reintroduced))
                    decision_log.append({
                        "round": round_num, "solving_for": fi.symbol,
                        "solving_for_name": fi.name, "chosen_eq_id": eq["id"],
                        "equation_str": eq["equation_str"],
                        "reason": (f"Rejected: this equation re-introduces "
                                   f"{sorted(reintroduced)}, which is still being "
                                   f"solved — that would create a circular chain."),
                        "conditions_concern": None, "fallback_used": "cycle_guard",
                        "n_candidates": len(candidates_shown),
                        "candidates_shown": [], "decision": "reject_cycle",
                    })
                    return ResolutionResult(
                        plan=[], final_symbol=target.symbol, success=False,
                        failure_reason=(f"Circular pick for '{fi.symbol}': "
                                        f"{eq['id']} re-introduces {sorted(reintroduced)}."),
                        decision_log=decision_log, status="UNVERIFIED",
                        dead_end_root_eq=eq["id"], landing_eq_id=landing_eq_id,
                    )

            # ── Stepping-stone guard (structural — honour the LLM's binding) ──
            # The LLM picked an equation but, by its OWN binding, it produces a
            # different quantity and still lists fi.symbol in `needed` (e.g. it
            # picked flux Phi=B*A*cos(theta) for the emf sub-goal and said emf is
            # still needed). Committing it would rename the wrong quantity to fi
            # and report it (webers as emf). Reject + hand the equation back as
            # the culprit so the neighbour walk re-picks one that produces fi.
            if not _pick_produces_target(sel, fi.symbol):
                log("subgoal_pick_rejected_stepping_stone",
                    round_num=round_num, symbol=fi.symbol, eq_id=eq["id"],
                    solves_for=sel.get("solves_for"), needed=sel.get("needed"))
                decision_log.append({
                    "round": round_num, "solving_for": fi.symbol,
                    "solving_for_name": fi.name, "chosen_eq_id": eq["id"],
                    "equation_str": eq["equation_str"],
                    "reason": (f"Rejected: by its own binding this equation "
                               f"produces '{sel.get('solves_for')}', not "
                               f"'{fi.symbol}' (still listed as needed) — it is a "
                               f"stepping-stone, not the equation for this quantity."),
                    "conditions_concern": None,
                    "fallback_used": "stepping_stone_guard",
                    "n_candidates": len(candidates_shown),
                    "candidates_shown": [], "decision": "reject_stepping_stone",
                })
                culprit_eq = eq["id"] if eq["id"] != landing_eq_id else \
                    _deepest_culprit_on_branch(fi, chosen_steps)
                return ResolutionResult(
                    plan=[], final_symbol=target.symbol, success=False,
                    failure_reason=(f"Stepping-stone pick for '{fi.symbol}': "
                                    f"{eq['id']} produces "
                                    f"'{sel.get('solves_for')}', not the target."),
                    decision_log=decision_log, status="UNVERIFIED",
                    dead_end_root_eq=culprit_eq, landing_eq_id=landing_eq_id,
                )

            # ── Follow the LLM's variable binding (NO matching in code) ───────
            # The LLM mapped the equation onto the problem by MEANING and told us:
            #   solves_for : which equation variable IS this unknown,
            #   known      : {equation_var -> ALREADY-KNOWN symbol giving its value},
            #   needed     : equation variables still to solve.
            # The code obeys it — it never compares symbols or dimensions itself.
            output_var = _pick_output_var(eq, sel.get("solves_for"), fi.symbol)
            llm_known = sel.get("known") or {}

            # Where each input variable's value comes from. We OBEY the LLM's
            # binding: resolve the source it named against the state we showed it
            # (by symbol OR quantity-name), and honor a literal it bound (e.g. a
            # term that is zero in this scenario). Constants map to themselves.
            # Anything the LLM left genuinely unbound becomes a new sub-goal.
            value_binding:  dict[str, str]   = {}
            literal_values: dict[str, float] = {}
            for v in eq_vars:
                if v == output_var:
                    continue
                if v in PHYSICAL_CONSTANTS:
                    value_binding[v] = v
                    continue
                stated   = llm_known.get(v)
                resolved = _resolve_source(stated, available)
                if resolved is not None:
                    value_binding[v] = resolved          # bound to a known quantity
                    continue
                lit = _as_literal(stated)
                if lit is not None:
                    literal_values[v] = lit              # bound to a literal (e.g. 0)
                    value_binding[v]  = v
                    continue
                value_binding[v] = v                     # genuine sub-goal

            step = ResolvedStep(
                equation=eq,
                solves_for=fi,
                inputs_used=[v for v, src in value_binding.items()
                             if src == v and v not in PHYSICAL_CONSTANTS
                             and v not in literal_values],
                round_num=round_num,
                llm_reason=sel.get("reason", ""),
                output_var=output_var,
                resolves_symbol=fi.symbol,
                value_binding=value_binding,
                literal_values=literal_values,
                conditions_concern=sel.get("conditions_concern"),
                is_provisional=bool(sel.get("conditions_concern")),
            )
            chosen_steps.append(step)
            if round_num == 0:
                landing_eq_id = eq["id"]
            visited_eqs.add(eq["id"])
            # This step makes fi.symbol computable — register it so later rounds
            # and the state document can treat it as an available quantity.
            available.setdefault(fi.symbol, {
                "value": None, "name": fi.name,
                "unit": fi.unit, "dimension": fi.dimension})

            # New sub-goals: every input the LLM did NOT bind to a known quantity
            # or to a literal. (A term resolved to a literal value — e.g. 0 — is
            # accounted for, so it is NOT chased as a sub-goal.)
            for v, src in value_binding.items():
                if v in PHYSICAL_CONSTANTS or src != v or v in literal_values:
                    continue  # constant, known from an existing quantity, or literal
                if v in available or v in seen_in_frontier or v in already_targeted:
                    continue
                meta = eq["variables"][v]
                new_frontier.append(FrontierItem(
                    symbol=v, name=meta.get("name", v),
                    unit=meta.get("unit", ""), dimension=meta.get("dimension", ""),
                    introduced_by=eq["id"]))
                seen_in_frontier.add(v)

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
            landing_eq_id=landing_eq_id,
        )

    given_syms = set(given.keys())
    sorted_steps, cyclic = _topological_sort(chosen_steps, given_syms)
    plan = _merge_cycles(sorted_steps, cyclic)

    return ResolutionResult(
        plan=plan,
        final_symbol=resolved_target_symbol,
        success=True,
        decision_log=decision_log,
        status="SUCCESS",
    )
