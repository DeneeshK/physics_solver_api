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
from config import (
    PHYSICAL_CONSTANTS, UNIVERSAL_CONSTANTS, IMPLICIT_CONSTANTS_CATALOG,
    ENABLE_CHROMA_LANDING, SYMBOL_ALIASES, SYMBOL_ALIASES_BY_DIMENSION,
)

MAX_BACKTRACK_ATTEMPTS = 6  # v7.2.2: covers both Stage-3 exclusions AND
#                             Stage-2 node-rollback (drop a dead-ended Round-0
#                             node, try the next of the top-5). Needs to be a
#                             few more than the retrieval top_k so several
#                             candidate nodes can be tried before giving up.


def _free_inter_question_memory() -> None:
    """
    v7.1.10: free per-question working memory between questions WITHOUT
    unloading the LLM.

    On an 8GB GPU running one resident 7B model, memory pressure across a
    long run (e.g. the 89-question bank) comes from two places: Python
    objects accumulating, and the local embedding framework's CUDA cache (if
    EMBED_DEVICE=cuda). This clears both. It deliberately does NOT call
    Ollama's unload — the model is meant to stay resident the whole run, so
    we keep the weights in VRAM and only release transient allocations.

    Note: the Ollama server holds the LLM weights in its OWN process, so this
    never touches them — it only frees what the solver's own process holds.
    Safe to call even when torch isn't installed (the embedder may be on CPU
    or use a non-torch backend); failures are swallowed.
    """
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        # torch not present, or CUDA not in use (e.g. embedder on CPU) —
        # gc.collect() above is sufficient.
        pass


def _canonicalize_symbol(sym: str, dimension: str = "") -> str:
    """
    v7.1.6: Map an LLM-chosen symbol to the graph's canonical symbol for the
    same physical quantity.

    Two-tier resolution:
      1. Direct alias (E_k → K, PE → U): unambiguous, dimension-independent.
      2. Dimension-aware alias (E + dimension ML2T-2 → K): for symbols whose
         meaning depends on context, the dimension Stage 1 reported picks the
         canonical target.

    Returns the canonical symbol, or the original if no mapping applies.
    """
    if sym in SYMBOL_ALIASES:
        return SYMBOL_ALIASES[sym]
    if sym in SYMBOL_ALIASES_BY_DIMENSION and dimension:
        # Normalize dimension formatting for the lookup: strip spaces and
        # carets, and fold unicode superscripts (²³) to ASCII digits, since
        # Stage 1 emits both 'ML2T-2' and 'ML²T-2' for the same dimension.
        dim_norm = (dimension.replace(" ", "").replace("^", "")
                    .replace("²", "2").replace("³", "3")
                    .replace("⁴", "4").replace("⁻", "-").replace("¹", "1"))
        dim_map = SYMBOL_ALIASES_BY_DIMENSION[sym]
        if dim_norm in dim_map:
            return dim_map[dim_norm]
    return sym


def _canonicalize_given_symbols(
    given_meta: dict, target_sym: str, canon=None
) -> tuple[dict, dict]:
    """
    v7.2.6: Canonicalize Stage-1 GIVEN symbols to the graph's naming.

    Previously _canonicalize_symbol() ran on the unknown target ONLY, so when
    Stage 1 named a *given* quantity differently from the graph (e.g. it gave
    `l=1.0` for a pendulum whose equation uses `L`), the resolver saw `L` as a
    fresh unknown, launched a neighbor walk, and the model — correctly — found
    nothing fitting. The failure audit traced ~66% of failures to exactly this
    Stage-1 ↔ graph symbol gap. This closes it on the given side.

    Collision guard — the reason this isn't a blind dict-remap:

      Some aliases collapse a family of Stage-1 symbols onto one graph symbol
      (m1, m2 → m; rho_block, rho_water → rho). When a single problem supplies
      TWO members of such a family, remapping both would silently overwrite one
      given's value with the other's. So a remap is skipped whenever its
      canonical form would:
        - be the canonical form of another given this round (m1 AND m2 → m), or
        - already exist as a literal given key (problem gives both `l` and `L`),
          or
        - collide with the (already-canonicalized) unknown target symbol.

      Such collapsing cases (e.g. two-body collisions, buoyancy with two
      densities) need per-equation instantiation, not aliasing; leaving them
      un-mapped preserves both values rather than corrupting the chain.

    Returns (new_given_meta, applied) where `applied` is {original: canonical}
    for logging.
    """
    if not given_meta:
        return given_meta, {}
    if canon is None:
        canon = lambda s, d="", u="", nm="": _canonicalize_symbol(s, d)

    # Pass 1 — compute each given's candidate canonical form (by concept: alias,
    # then name + dimension match against the registry).
    candidate: dict[str, str] = {}
    for sym, meta in given_meta.items():
        dim = meta.get("dimension", "") if isinstance(meta, dict) else ""
        unit = meta.get("unit", "") if isinstance(meta, dict) else ""
        name = meta.get("name", "") if isinstance(meta, dict) else ""
        candidate[sym] = canon(sym, dim, unit, name)

    # How many givens want each canonical name? (>1 ⇒ family collision.)
    from collections import Counter
    canon_demand = Counter(candidate.values())

    new_meta: dict = {}
    applied: dict = {}
    for sym, meta in given_meta.items():
        canon = candidate[sym]
        if canon == sym:
            new_meta[sym] = meta
            continue
        collides = (
            canon_demand[canon] > 1     # two givens fight for the same name
            or canon in given_meta      # canonical name is itself a literal given
            or canon == target_sym      # would shadow the unknown
        )
        if collides:
            new_meta[sym] = meta        # keep original — never destroy a value
            continue
        new_meta[canon] = meta
        applied[sym] = canon
    return new_meta, applied


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
        # v7: lazy-load Retriever. Modes:
        #   "false"/"0"/"no"  → never load Chroma, force v6 behavior
        #   "true"/"1"/"yes"  → require Chroma; raise if it can't load
        #   "auto" (default)  → load if the index exists, otherwise None
        # In every "None" case, frontier_resolver falls through to symbol-only
        # candidate generation — exactly v6 behavior. This is the safety
        # property: enabling Chroma can only add candidates, never remove.
        self.retriever = self._init_retriever()

    def _init_retriever(self):
        mode = (ENABLE_CHROMA_LANDING or "auto").lower()
        if mode in ("false", "0", "no", "off"):
            print("[PhysicsSolver] ChromaDB landing disabled by config. "
                  "Using symbol-only landing (v6 behavior).")
            return None
        # Defer heavy import to here so test environments that don't have
        # sentence-transformers / chromadb installed can still import
        # PhysicsSolver as long as they don't enable Chroma.
        from solver.retrieval import Retriever
        retriever = Retriever.try_load(self.graph)
        if retriever is None and mode in ("true", "1", "yes", "on"):
            raise RuntimeError(
                "ENABLE_CHROMA_LANDING=true but the ChromaDB and/or BM25 "
                "index could not be loaded. Run `python -m solver.ingest` "
                "to build them, or set ENABLE_CHROMA_LANDING=false to "
                "force v6 behavior."
            )
        if retriever is None:
            print("[PhysicsSolver] ChromaDB index not found. "
                  "Using symbol-only landing (v6 behavior). "
                  "Run `python -m solver.ingest` to enable ChromaDB landing.")
        return retriever

    def solve(self, question: str) -> SolverResponse:
        t0 = time.time()
        from solver.solver_log import log, log_error
        # v7.1.10: free transient per-question memory before starting the
        # next question (clears CUDA cache / Python garbage from the prior
        # solve) WITHOUT unloading the resident LLM. Runs at solve entry so
        # it executes between questions regardless of how the previous solve
        # ended.
        _free_inter_question_memory()
        log("solve_entry", question=question)

        # ═══════════════════════════════════════════════════════════════════════
        # Stage 1 — Parse
        # ═══════════════════════════════════════════════════════════════════════
        try:
            parsed = parse_question(question, valid_domains=self.graph.all_domains)
        except Exception as e:
            log_error("solve_stage1_failed", exc=e)
            return self._error(question, f"Stage 1 parse failed: {e}", t0)

        given_meta  = parsed.get("given", {})   # {sym: {value,unit,name,dimension}}
        unknown     = parsed.get("unknown", {})  # {symbol,name,unit,dimension}
        target_sym  = unknown.get("symbol", "")
        target_dim  = unknown.get("dimension", "")

        # v8: NO symbol canonicalization or matching in code. Stage 1's symbols
        # are kept exactly as the model wrote them; the Stage-2 LLM is what maps
        # the chosen equation's variables onto these quantities, by meaning (see
        # frontier_resolver's use of the LLM `solves_for`/`known`/`needed`
        # binding). The graph's variable names and the question's names never
        # need to agree — that is the whole point of having the LLM traverse.

        search_query = parsed.get("search_query", "")  # v7: for Chroma landing
        # Hint only — candidates_for_quantity() falls back to the full set
        # whenever this would otherwise leave zero candidates for a quantity,
        # so a wrong/incomplete guess here costs prompt size, not correctness.
        allowed_domains = set(parsed.get("likely_domains", [])) or None

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
        # Two exclusion scopes so a deep dead-end can't cascade into banning the
        # correct landing (the bug that turned one mis-bound sub-goal into
        # "rejected all candidates"):
        #   dropped_landings — Round-0 nodes whose WHOLE chain couldn't close;
        #                      persist, so re-landing moves to the next top-5 node.
        #   deep_excluded    — deeper (round>0) culprit equations; scoped to the
        #                      CURRENT landing and cleared when we drop the node,
        #                      so a banned deep eq never starves a different node.
        dropped_landings: set = set()
        deep_excluded:    set = set()
        full_decision_log: list[dict] = []
        resolution  = None
        exec_trace  = None

        for attempt in range(1, MAX_BACKTRACK_ATTEMPTS + 1):
            excluded_eqs = dropped_landings | deep_excluded
            log("solve_attempt", attempt=attempt,
                target=target_sym, n_excluded=len(excluded_eqs))
            resolution = resolve_frontier(
                target          = target,
                given           = given_full,
                graph_index     = self.graph,
                question        = question,
                llm_round_fn    = call_round_selector,
                excluded_eqs    = excluded_eqs,
                allowed_domains = allowed_domains,
                search_query    = search_query,
                retriever       = self.retriever,
            )
            for entry in resolution.decision_log:
                full_decision_log.append({**entry, "attempt": attempt})

            if not resolution.success:
                # Stage 2 couldn't complete a chain. First try to re-pick the
                # DEEPER equation that failed (keeping the correct landing); only
                # when no deeper alternative is left do we drop the whole node and
                # move to the next top-5 landing — with a CLEAN deep slate.
                deep = resolution.dead_end_root_eq        # round>0 culprit, or ""
                if deep and deep not in deep_excluded:
                    log("stage2_node_rollback", kind="deep_reselect",
                        excluded_eq=deep, attempt=attempt,
                        reason="sub-goal dead-ended; banning this deeper equation "
                               "and re-picking an alternative for the same sub-goal")
                    deep_excluded.add(deep)
                    continue
                landing = resolution.landing_eq_id
                if landing and landing not in dropped_landings:
                    log("stage2_node_rollback", kind="drop_node",
                        excluded_root=landing, attempt=attempt,
                        reason="deep alternatives exhausted; dropping this Round-0 "
                               "node and retrying the next top-5 node (clean slate)")
                    dropped_landings.add(landing)
                    deep_excluded.clear()    # fresh slate for the new landing
                    continue
                # No deeper culprit and no landing left to drop — genuinely stuck.
                break

            # v8: execute under the symbol the resolver actually solved for.
            # Concept-binding may have rebound the target (e.g. 'T' → 'T_p') to
            # the landing equation's variable for the same quantity, so the plan
            # produces `resolution.final_symbol`, which can differ from the
            # Stage-1 target string. Fall back to target_sym for older paths.
            solve_symbol = getattr(resolution, "final_symbol", "") or target_sym
            exec_trace = execute_plan(
                plan          = resolution.plan,
                given_values  = given_values,
                target_symbol = solve_symbol,
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
            if not newly_excluded or newly_excluded <= (dropped_landings | deep_excluded):
                # Nothing new to exclude — retrying would just repeat the
                # same failure, so stop instead of looping pointlessly.
                break
            # A SymPy failure bans the offending equation(s) as deep culprits,
            # so the retry re-picks alternatives while keeping the landing.
            deep_excluded |= newly_excluded

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

        log("solve_success",
            final_value=exec_trace.final_float,
            final_unit=exec_trace.final_unit,
            confidence=confidence,
            chain_summary=chain_summary,
            elapsed_s=round(time.time() - t0, 2))
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
        from solver.solver_log import log
        log("solve_error", error=error, elapsed_s=round(time.time() - t0, 2))
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
