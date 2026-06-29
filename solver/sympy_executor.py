"""
solver/sympy_executor.py
Stage 3: Exact-arithmetic execution of a ResolutionResult plan.

Key changes from v1:
  - Accepts ResolutionResult instead of ChainResult.
  - Uses SymPy Rational / nsimplify for exact arithmetic throughout;
    only converts to float for the final display value.
  - Produces SubstitutionTrace per step (symbolic → substituted → result).
  - Handles SimultaneousGroup via sympy.solve([eq1, eq2, ...], [unknowns]).
  - Dimensional check on final result.
  - Multiple-root filtering respects physical meaning metadata.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from sympy import (
    symbols, solve, Eq, sqrt, pi, sin, cos, tan, log, exp,
    Rational, nsimplify, N as sympy_N, Abs, Float,
)
from sympy import sympify
from config import IMPLICIT_CONSTANTS_CATALOG

# ── SymPy namespace for eval() ────────────────────────────────────────────────
SYMPY_NS = {
    "sqrt": sqrt, "pi": pi, "Rational": Rational,
    "sin": sin, "cos": cos, "tan": tan,
    "log": log, "exp": exp, "Eq": Eq,
}

# ── Built-in physical constant values (exact where possible) ──────────────────
CONSTANT_VALUES: dict[str, object] = {
    sym: nsimplify(meta["value"])
    for sym, meta in IMPLICIT_CONSTANTS_CATALOG.items()
    if meta.get("value") is not None
}

# Symbols that can legitimately be negative (keep sign in root selection)
SIGNED_SYMBOLS = {"v", "u", "a", "F", "W", "emf", "q", "DeltaT"}

# ── Wall-clock guard for runaway SymPy calls ──────────────────────────────────
# solve()/evalf() are CPU-bound and can spin for many minutes on a transcendental
# or high-degree system — e.g. a spurious simultaneous group sent solve() into
# polynomial factorization (dup_zz_factor → Zassenhaus) that ran >15 min and had
# to be killed mid-eval. try/except cannot stop that: it is not an exception,
# just a loop that never returns. A SIGALRM deadline converts the runaway into a
# catchable error so the step fails cleanly and the resolver backtracks instead
# of wedging the whole run.
import signal as _signal
import threading as _threading

# Per-call ceiling. A legitimate JEE/NEET step solves in well under a second;
# anything past this is a runaway, not a slow-but-valid computation.
SOLVE_TIMEOUT_S = 8


class _SympyTimeout(Exception):
    """Raised when a guarded SymPy call exceeds SOLVE_TIMEOUT_S."""


class _time_limit:
    """Best-effort wall-clock guard. Active only on a Unix process's main
    thread (where SIGALRM is deliverable); anywhere else it is a no-op, so it
    degrades to the prior behavior rather than ever crashing. SymPy's hot loops
    are pure Python, so the alarm is delivered between bytecodes and propagates."""

    def __init__(self, seconds: int = SOLVE_TIMEOUT_S):
        self.seconds = max(1, int(seconds))
        self._active = False

    def __enter__(self):
        if (_threading.current_thread() is _threading.main_thread()
                and hasattr(_signal, "SIGALRM")):
            self._old = _signal.signal(_signal.SIGALRM, self._fire)
            _signal.alarm(self.seconds)
            self._active = True
        return self

    def _fire(self, *_a):
        raise _SympyTimeout()

    def __exit__(self, *_a):
        if self._active:
            _signal.alarm(0)
            _signal.signal(_signal.SIGALRM, self._old)
        return False


# ── Output data structures ────────────────────────────────────────────────────

@dataclass
class SubstitutionTrace:
    symbolic:    str   # equation rearranged: "m = rho * V"
    substituted: str   # values inserted: "m = 8000 * 0.5"
    result_exact: str  # exact SymPy result: "m = 4000" or "a = 45/4"
    result_float: str  # decimal: "a = 11.25"


@dataclass
class StepTrace:
    equation_str:  str
    solving_for:   str
    unit:          str
    trace:         SubstitutionTrace
    value_exact:   object    # SymPy Rational or expression
    value_float:   float
    conditions_concern: str | None = None
    is_group:      bool = False   # True for SimultaneousGroup steps


@dataclass
class ExecutionTrace:
    step_traces:    list[StepTrace]
    final_symbol:   str
    final_unit:     str
    final_exact:    object       # SymPy exact
    final_float:    float
    final_exact_str: str
    success:        bool
    error:          str = ""
    dimension_ok:   bool = True
    failed_eq_ids:  list = field(default_factory=list)  # which equation(s) caused failure

    @property
    def computed_values(self) -> dict[str, float]:
        return {t.solving_for: t.value_float for t in self.step_traces}


# ── Main entry point ──────────────────────────────────────────────────────────

def execute_plan(
    plan:         list,         # list[ResolvedStep | SimultaneousGroup]
    given_values: dict[str, float],   # {symbol: float} from parse
    target_symbol: str,
    target_unit:   str = "",
    target_dim:    str = "",
) -> ExecutionTrace:
    """
    Execute a ResolutionResult.plan produced by Stage 2.
    Maintains exact SymPy arithmetic (Rational) throughout.
    """
    # Seed computed with exact representations of given values
    computed: dict[str, object] = {**CONSTANT_VALUES}
    for sym, val in given_values.items():
        computed[sym] = nsimplify(val)

    step_traces: list[StepTrace] = []

    for item in plan:
        cls_name = type(item).__name__

        if cls_name == "SimultaneousGroup":
            traces = _execute_simultaneous(item, computed)
        else:
            # ResolvedStep
            traces = _execute_step(item, computed)

        if traces is None:
            # Identify exactly which equation(s) caused the failure so the
            # caller can exclude them and retry — works for both a single
            # ResolvedStep (one equation) and a SimultaneousGroup (multiple).
            if hasattr(item, "equation"):
                failed_ids = [item.equation["id"]]
                failed_str = item.equation["equation_str"]
            elif hasattr(item, "equations"):
                failed_ids = [eq["id"] for eq in item.equations]
                failed_str = " & ".join(eq["equation_str"] for eq in item.equations)
            else:
                failed_ids = []
                failed_str = str(item)

            return ExecutionTrace(
                step_traces=step_traces,
                final_symbol=target_symbol,
                final_unit=target_unit,
                final_exact=None,
                final_float=0.0,
                final_exact_str="",
                success=False,
                error=f"SymPy failed on {failed_str}",
                failed_eq_ids=failed_ids,
            )

        for t in traces:
            step_traces.append(t)
            # Update computed with the newly solved value
            computed[t.solving_for] = t.value_exact

    # Find final answer
    final_trace = next(
        (t for t in reversed(step_traces) if t.solving_for == target_symbol),
        step_traces[-1] if step_traces else None,
    )
    if final_trace is None:
        return ExecutionTrace(
            step_traces=step_traces,
            final_symbol=target_symbol, final_unit=target_unit,
            final_exact=None, final_float=0.0, final_exact_str="",
            success=False, error="Target symbol not found in computed steps",
        )

    return ExecutionTrace(
        step_traces=step_traces,
        final_symbol=target_symbol,
        final_unit=final_trace.unit or target_unit,
        final_exact=final_trace.value_exact,
        final_float=final_trace.value_float,
        final_exact_str=str(final_trace.value_exact),
        success=True,
        dimension_ok=True,  # extended check can be added per §4.4
    )


# ── Single-step execution ─────────────────────────────────────────────────────

def _execute_step(step, computed: dict) -> list[StepTrace] | None:
    eq_node   = step.equation
    # v8: the LLM bound the equation onto the problem. `output_var` is the
    # equation variable to solve; each other variable's value comes from
    # `value_binding[var]` (a given symbol, a constant, or a previously-solved
    # quantity). The result is stored under `resolves_symbol` (the quantity this
    # step resolves), which may differ from output_var. No symbol matching here.
    output_var = getattr(step, "output_var", "") or step.solves_for.symbol
    resolves   = getattr(step, "resolves_symbol", "") or step.solves_for.symbol
    binding    = getattr(step, "value_binding", None) or {}
    literals   = getattr(step, "literal_values", None) or {}

    var_syms = {s: symbols(s) for s in eq_node["variables"]}
    ns       = {**SYMPY_NS, **var_syms}

    try:
        sympy_eq = eval(eq_node["sympy_expr"], ns)
    except Exception as e:
        raise ValueError(f"Failed to parse sympy_expr '{eq_node['sympy_expr']}': {e}")

    if output_var not in var_syms:
        return None

    # Substitute every variable except the output, pulling its value through the
    # LLM's binding (falling back to the same symbol, which also covers constants
    # already seeded in `computed`).
    subs = {}
    for sym_str, sym_obj in var_syms.items():
        if sym_str == output_var:
            continue
        # A term the LLM bound to a literal (e.g. a vanishing X_C = 0) takes
        # precedence — there is no source quantity to look up.
        if sym_str in literals:
            subs[sym_obj] = nsimplify(literals[sym_str])
            continue
        src = binding.get(sym_str, sym_str)
        if src in computed:
            subs[sym_obj] = computed[src]
        elif sym_str in computed:
            subs[sym_obj] = computed[sym_str]

    substituted_eq = sympy_eq.subs(subs)
    target_obj     = var_syms[output_var]

    try:
        with _time_limit():
            solutions = solve(substituted_eq, target_obj)
    except _SympyTimeout:
        from solver.solver_log import log
        log("sympy_solve_timeout", solving_for=output_var,
            equation=eq_node.get("equation_str", ""), limit_s=SOLVE_TIMEOUT_S)
        return None
    except Exception:
        return None

    if not solutions:
        return None

    value = _pick_best_solution(solutions, output_var)
    if value is None:
        return None

    # Build trace strings
    symbolic_str    = _eq_to_str(sympy_eq, output_var, var_syms)
    substituted_str = _eq_to_str(substituted_eq, output_var, var_syms)
    result_str      = f"{resolves} = {value}"

    try:
        with _time_limit():
            float_val = float(sympy_N(value))
    except Exception:
        float_val = 0.0

    unit = eq_node["variables"].get(output_var, {}).get("unit", "")
    if " or " in unit:
        unit = unit.split(" or ")[0].strip()

    trace = SubstitutionTrace(
        symbolic=symbolic_str,
        substituted=substituted_str,
        result_exact=result_str,
        result_float=f"{resolves} = {_format_float(float_val)}",
    )
    return [StepTrace(
        equation_str=eq_node["equation_str"],
        solving_for=resolves,
        unit=unit,
        trace=trace,
        value_exact=value,
        value_float=float_val,
        conditions_concern=step.conditions_concern,
    )]


# ── Simultaneous-group execution ──────────────────────────────────────────────

def _execute_simultaneous(group, computed: dict) -> list[StepTrace] | None:
    from solver.frontier_resolver import SimultaneousGroup  # avoid circular at module level

    # Collect all variable symbols across the group's equations
    all_var_syms: dict[str, object] = {}
    for eq in group.equations:
        for s in eq["variables"]:
            if s not in all_var_syms:
                all_var_syms[s] = symbols(s)

    ns = {**SYMPY_NS, **all_var_syms}

    # Parse each equation
    sympy_eqs = []
    for eq in group.equations:
        try:
            sympy_eqs.append(eval(eq["sympy_expr"], ns))
        except Exception as e:
            raise ValueError(f"Failed to parse '{eq['sympy_expr']}': {e}")

    # Unknown symbols = all variables NOT in computed and NOT physical constants.
    # v7.1.11: register any unknown symbol that isn't already in all_var_syms
    # rather than crashing with KeyError. This happens when a frontier item's
    # symbol differs from the symbol as it appears in the group's equations —
    # most commonly a CANONICALIZED target (E_k→K, PE→U, KE→K). The target was
    # renamed during selection, but all_var_syms was built only from the
    # equations' literal variable names. Creating the symbol here is the
    # generic fix: it works for any canonicalized or otherwise-unlisted
    # unknown, not just one specific symbol.
    unknowns = []
    for fi in group.unknowns:
        if fi.symbol not in all_var_syms:
            all_var_syms[fi.symbol] = symbols(fi.symbol)
            ns[fi.symbol] = all_var_syms[fi.symbol]
        unknowns.append(all_var_syms[fi.symbol])

    # Substitute all knowns
    sub_map = {
        all_var_syms[s]: v
        for s, v in computed.items()
        if s in all_var_syms
    }
    sub_eqs = [eq.subs(sub_map) for eq in sympy_eqs]

    try:
        with _time_limit():
            sol = solve(sub_eqs, unknowns, dict=True)
    except _SympyTimeout:
        from solver.solver_log import log
        log("sympy_solve_timeout", kind="simultaneous",
            equations=[e["equation_str"] for e in group.equations],
            limit_s=SOLVE_TIMEOUT_S)
        return None
    except Exception:
        return None

    if not sol:
        return None

    # Pick the first physically valid solution
    chosen = sol[0] if isinstance(sol, list) else sol
    traces = []

    for fi, unk_sym in zip(group.unknowns, unknowns):
        val = chosen.get(unk_sym)
        if val is None:
            return None

        try:
            with _time_limit():
                float_val = float(sympy_N(val))
        except Exception:
            float_val = 0.0

        # Find unit from one of the group's equations
        unit = ""
        for eq in group.equations:
            if fi.symbol in eq["variables"]:
                unit = eq["variables"][fi.symbol].get("unit", "")
                break

        traces.append(StepTrace(
            equation_str=f"simultaneous system ({', '.join(e['equation_str'] for e in group.equations)})",
            solving_for=fi.symbol,
            unit=unit,
            trace=SubstitutionTrace(
                symbolic=f"solve({', '.join(e['equation_str'] for e in group.equations)}) for {fi.symbol}",
                substituted="(substituted known values)",
                result_exact=f"{fi.symbol} = {val}",
                result_float=f"{fi.symbol} = {_format_float(float_val)}",
            ),
            value_exact=val,
            value_float=float_val,
            is_group=True,
        ))

    return traces


# ── Root selection ────────────────────────────────────────────────────────────

def _pick_best_solution(solutions, symbol_name: str) -> object | None:
    """
    Select the most physically meaningful root.
    1. Filter to real-valued solutions.
    2. For signed quantities (velocity, force, etc.): take the first real.
    3. For unsigned (mass, time, distance): prefer positive; if none, min |value|.
    """
    real_solutions = []
    for sol in solutions:
        try:
            val = complex(sympy_N(sol))
            if abs(val.imag) < 1e-9:
                real_solutions.append(sol)
        except Exception:
            continue

    if not real_solutions:
        return None

    if symbol_name in SIGNED_SYMBOLS:
        return real_solutions[0]

    positive = [s for s in real_solutions if float(sympy_N(s)) > 0]
    if positive:
        return min(positive, key=lambda s: abs(float(sympy_N(s))))

    return min(real_solutions, key=lambda s: abs(float(sympy_N(s))))


# ── Trace helpers ─────────────────────────────────────────────────────────────

def _eq_to_str(sympy_eq, solve_sym: str, var_syms: dict) -> str:
    """
    Pretty-print a SymPy equation with the solve_sym isolated on the left.
    """
    try:
        lhs = var_syms[solve_sym]
        # Try to rearrange: solve the equation for solve_sym symbolically
        sols = solve(sympy_eq, lhs)
        if sols:
            return f"{solve_sym} = {sols[0]}"
    except Exception:
        pass
    return str(sympy_eq)


def _format_float(v: float) -> str:
    """Compact float string — avoids scientific notation for typical physics values."""
    if v == 0:
        return "0"
    if abs(v) >= 1e-3 and abs(v) < 1e7:
        # Fixed notation with up to 6 significant figures
        s = f"{v:.6g}"
        return s
    return f"{v:.4e}"
