# graph_builder — the physics graph compiler

Rebuilds the equation graph from a **canonical variable registry** + **validated
equation definitions**, so the data-quality bugs that sank the old graph
(`f = u - v`, `Q = DeltaV + W`, `C` vs `C_cap`, overloaded `T`/`R`/`Q`/`a`)
**cannot be shipped** — they fail the build.

## Files
- `registry.py` — every physical quantity as ONE canonical variable (globally
  unique symbol), with dimension, SI unit, and the Stage-1 **aliases** that map
  to it. This is the single source of truth for symbol naming + canonicalization.
- `equations.py` — equations written using only canonical symbols, with a
  concept `rag` string for retrieval.
- `dim.py` — dependency-free expression parser + dimensional algebra. Verifies
  every equation's LHS dimension == RHS dimension.
- `compile.py` — validates everything and emits the graph JSON.

## Build + cutover
```bash
# 1. compile (validates: symbol membership, single output, dimensional consistency)
python -m graph_builder.compile -o data/physics_graph_v8.json

# 2. point the runtime at it — already done in config.MAIN_GRAPH_PATH

# 3. MANDATORY: rebuild the retrieval indexes from the new nodes
python -m solver.ingest --reingest

# 4. run the eval
python tests/evaluate_bank.py questions/question_bank.json --report after_v8.json
```

## Invariants enforced at build time
1. Every symbol in every equation is a registered canonical variable.
2. Each equation declares an `output` variable that appears in it.
3. `dim(LHS) == dim(RHS)` for every equation.
4. No duplicate equation ids.

## Emitted graph shape (drop-in for the existing workflow)
`nodes` (equation nodes, executor/retrieval/ingest compatible) · `edges`
(equation↔equation, shared canonical symbol) · `variables` (canonical registry +
aliases) · `variable_equations` (bipartite variable→equations) · `aliases`
(flat Stage-1→canonical). `GraphIndex` loads `variables`/`aliases` and exposes
`.canonical(sym, dim)`, which the pipeline uses for symbol normalization.

## Adding/fixing an equation
Edit `equations.py` (and add any new quantity to `registry.py`), then re-run
steps 1–4. If you misname a symbol or write a dimensionally-inconsistent
formula, the compile step fails with the exact reason. **Never hand-edit the
emitted JSON** — the compiler is the source of truth.
