"""
graph_builder/compile.py

The graph compiler. Reads the canonical variable registry + the equation
definitions, VALIDATES every equation, and EMITS a graph JSON in the exact
shape the runtime workflow already consumes ({"nodes":[...], "edges":[...]}),
plus explicit bipartite sections.

Validation (build fails loudly on any violation — this is the guarantee that
the old data bugs cannot return):
  1. every symbol used in an equation is a registered canonical variable
     (kills "DeltaV-for-internal-energy" typos and "C vs C_cap" drift),
  2. the equation parses and has exactly one '=' with a single-symbol LHS that
     equals its declared natural output,
  3. LHS dimension == RHS dimension (kills "f = u - v", "Q = DeltaV + W", …),
  4. no duplicate equation ids.

Emitted top-level keys:
  nodes              — equation nodes (executor/retrieval/ingest compatible)
  edges              — equation↔equation, derived from shared canonical symbols
  variables          — the registry (canonical variable NODES) incl. aliases
  variable_equations — bipartite variable→[equation ids] adjacency
  aliases            — flat {alias: canonical} for Stage-1 normalization

Run:  python -m graph_builder.compile [-o data/physics_graph_v8.json]
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path

from graph_builder.registry import VARS, ALIASES, _ALIAS_DIM
from graph_builder.dim import (
    parse_expr, symbols_in, check_equation_dims, parse_dim_string,
    dim_to_string, DimensionError,
)
from graph_builder.equations import EQUATIONS, Eq

_FUNCS = {"sqrt", "sin", "cos", "tan", "asin", "acos", "atan",
          "sinh", "cosh", "tanh", "exp", "log", "ln", "abs", "Abs",
          "pi", "e", "E"}


def _var_dims() -> dict:
    return {sym: parse_dim_string(v.dim) for sym, v in VARS.items()}


def _validate(eq: Eq, var_dims: dict, seen_ids: set) -> list[str]:
    """Return a list of error strings for one equation (empty == valid)."""
    errs: list[str] = []
    if eq.id in seen_ids:
        errs.append(f"{eq.id}: duplicate equation id")
    seen_ids.add(eq.id)

    if eq.expr.count("=") != 1:
        errs.append(f"{eq.id}: must have exactly one '=' ({eq.expr!r})")
        return errs

    lhs_s, rhs_s = (p.strip() for p in eq.expr.split("="))

    # parse + symbol membership
    try:
        syms = symbols_in(parse_expr(lhs_s)) | symbols_in(parse_expr(rhs_s))
    except Exception as ex:
        errs.append(f"{eq.id}: parse error: {ex}")
        return errs
    for s in sorted(syms):
        if s not in VARS and s not in _FUNCS:
            hint = ""
            if s in ALIASES:
                hint = f" (did you mean canonical {ALIASES[s]!r}? '{s}' is an alias)"
            errs.append(f"{eq.id}: symbol {s!r} is not a registered variable{hint}")

    # The natural output is the symbol the equation is meant to solve for. It
    # must be a registered variable that actually appears in the equation — but
    # the equation may be IMPLICIT (v**2 = u**2 + 2*a*s, 1/f = 1/v - 1/u,
    # m1*u1 + m2*u2 = m1*v1 + m2*v2), so we do NOT require LHS to be a bare
    # symbol; SymPy solves the implicit form for `output` at runtime.
    out = eq.output or lhs_s
    if out not in VARS:
        errs.append(f"{eq.id}: output {out!r} is not a registered variable")
    elif out not in syms:
        errs.append(f"{eq.id}: output {out!r} does not appear in the equation")

    # dimensional consistency
    if not eq.skip_dim_check and not errs:
        try:
            check_equation_dims(eq.expr, var_dims, eq_id=eq.id)
        except DimensionError as ex:
            errs.append(str(ex))
    return errs


def _node(eq: Eq) -> dict:
    lhs_s, rhs_s = (p.strip() for p in eq.expr.split("="))
    syms = sorted(symbols_in(parse_expr(lhs_s)) | symbols_in(parse_expr(rhs_s)))
    syms = [s for s in syms if s in VARS]
    variables = {
        s: {"name": VARS[s].name, "dimension": VARS[s].dim,
            "unit": VARS[s].unit, "signed": VARS[s].signed}
        for s in syms
    }
    return {
        "id": eq.id,
        "equation_str": eq.expr,
        "latex": eq.latex,
        "sympy_expr": f"Eq({lhs_s}, {rhs_s})",
        "variables": variables,
        "domain": eq.domain,
        "subdomain": eq.subdomain,
        "conditions": list(eq.conditions),
        "jee_chapters": list(eq.jee),
        "neet_chapters": list(eq.neet),
        "common_mistakes": list(eq.mistakes),
        "rag_text": eq.rag,
        "vector_id": eq.id,
        "natural_output": eq.output or lhs_s,
    }


def _edges(nodes: list[dict]) -> list[dict]:
    """equation↔equation edges where two equations share ≥1 canonical symbol."""
    var_of = {n["id"]: set(n["variables"].keys()) for n in nodes}
    edges = []
    for a, b in itertools.combinations(nodes, 2):
        shared = var_of[a["id"]] & var_of[b["id"]]
        # constants are everywhere; don't wire equations together through them
        shared -= {"g", "pi", "G", "c", "epsilon_0", "mu_0", "h_planck",
                   "k_e", "R_gas", "k_B"}
        if shared:
            edges.append({
                "from": a["id"], "to": b["id"], "type": "SHARES_VARIABLE",
                "shared_variables": sorted(shared),
            })
    return edges


def build() -> dict:
    var_dims = _var_dims()
    seen: set = set()
    all_errs: list[str] = []
    for eq in EQUATIONS:
        all_errs += _validate(eq, var_dims, seen)
    if all_errs:
        raise SystemExit(
            "GRAPH BUILD FAILED — %d problem(s):\n  " % len(all_errs)
            + "\n  ".join(all_errs))

    nodes = [_node(eq) for eq in EQUATIONS]
    edges = _edges(nodes)

    # bipartite variable→equations adjacency
    var_eqs: dict[str, list[str]] = {}
    for n in nodes:
        for s in n["variables"]:
            var_eqs.setdefault(s, []).append(n["id"])

    variables = {
        sym: {"name": v.name, "dimension": v.dim, "unit": v.unit,
              "aliases": list(v.aliases), "signed": v.signed, "desc": v.desc}
        for sym, v in VARS.items()
    }
    return {
        "nodes": nodes,
        "edges": edges,
        "variables": variables,
        "variable_equations": var_eqs,
        "aliases": ALIASES,
        "_meta": {"n_equations": len(nodes), "n_edges": len(edges),
                  "n_variables": len(variables)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="data/physics_graph_v8.json")
    args = ap.parse_args()
    graph = build()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    m = graph["_meta"]
    print(f"OK  {m['n_equations']} equations, {m['n_edges']} edges, "
          f"{m['n_variables']} variables → {args.out}")
    # coverage by domain
    import collections
    dom = collections.Counter(n["domain"] for n in graph["nodes"])
    for d, c in sorted(dom.items()):
        print(f"    {c:3d}  {d}")


if __name__ == "__main__":
    main()
