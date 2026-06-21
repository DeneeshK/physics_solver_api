# Changelog v7.2 — Graph neighbor-walk traversal (removes the symbol/dimension gate)

This is the architectural change the user pushed for across several sessions:
Round 1+ candidate generation no longer uses a symbol-presence lookup or a
dimension filter that could REJECT the right equation. Instead, chasing a
variable mid-chain walks the graph to the equations connected through that
variable, and the LLM judges fit by meaning over that small set.

## The problem this fixes

The old `candidates_for_quantity` started from `sym_to_eqs[symbol]` — "every
equation literally containing this symbol" — then rejected candidates on a
dimension-string match. Two failures resulted:

  1. The dimension reject silently dropped correct, concept-matched equations
     when the LLM's dimension string format differed from the graph's
     (the F=ma-chain failure: Newton's second law retrieved at score 1.0 but
     filtered out before the LLM saw it). v7.1.11 patched the string
     comparison, but the GATE still existed.
  2. The symbol-presence requirement is the wrong selection criterion for a
     chaining solver. A question gives density and volume, not mass; the
     equation that PRODUCES mass (rho=m/V) must be reachable by meaning, not
     rejected because the question didn't mention mass.

The user's repeated point: the only legitimate question at a node is "does
this equation give me a way to get the variable I'm chasing — directly or by
chaining?" That is an LLM meaning-judgment, never a symbol/dimension filter.

## The design (as the user specified it)

  - Round 0 (first hop): full ChromaDB concept retrieval over the question.
    Top-5 nodes. The LLM walks them, judging meaning. This is the one
    legitimate semantic use — we have only the question text to start from.
  - Round 1+ (chasing a variable the chosen equation introduced): GRAPH
    NEIGHBOR WALK. Candidates = the equations connected, through that
    variable, to what we've already chosen. The LLM judges the small set
    (typically 5-12 equations) by meaning. No symbol gate, no dimension
    reject, no semantic re-search, no ranking — the set is already small, so
    it's shown raw with each equation's concept + ID.
  - Fallback tiers: if the neighbor walk yields nothing, widen to all
    equations containing the variable (still LOCAL, not a global semantic
    search); if still nothing, the resolver rolls back to the Round-0
    candidates and tries a different starting node. Never a global DB search
    mid-chain — a question's quantities are connected, so the needed equation
    is reachable through the graph.

## Key implementation insight: compute neighbors from variable membership

The graph's precomputed `edges` list is INCOMPLETE — e.g. F=ma and v²=u²+2as
both contain `a` but have no edge between them (so an edge-based walk missed
v²=u²+2as for the F=ma chain). The fix: compute neighbors directly from
`sym_to_eqs[variable]` — every equation touching that variable. This is the
TRUE bipartite traversal the user described ("reach the variable's node, look
at its neighbors"): sym_to_eqs[variable] IS the variable-node's adjacency
list. It can never miss a real connection and is robust to a stale/sparse edge
list. Verified: the F=ma chain now resolves m via rho=m/V (landing_source=
neighbor) and a via v²=u²+2as (landing_source=neighbor).

## Code changes

  - `solver/graph_loader.py`:
    - `neighbors_sharing_variable(from_eq_ids, variable, visited_eqs)` (new) —
      equations sharing `variable`, computed from sym_to_eqs (bipartite
      traversal), excluding sources/visited and non-solvable conservation
      forms. No dimension/symbol rejection.
    - `all_equations_with_variable(variable, visited_eqs)` (new) — the local
      widest fallback (variable-membership), still not a global search.
  - `solver/landing.py`:
    - `get_neighbor_candidates(...)` (new) — tier 1/2 neighbor walk → tier 3
      local fallback, returns the same shape as get_landing_candidates,
      tagged landing_source = "neighbor" / "variable_fallback".
  - `solver/frontier_resolver.py`:
    - Round 1+ now calls get_neighbor_candidates, walking from the equations
      chosen so far (chosen_steps) + the equation that introduced the current
      item (introduced_by). Round 0 unchanged (ChromaDB retrieval).
  - `tests/test_deterministic.py`:
    - `_StubGraphIndex` upgraded to expose the neighbor-walk interface
      (edges auto-built from shared variables + the two new methods), so the
      frontier tests exercise the v7.2 path.

## What did NOT change

  - Round 0 behavior (concept retrieval over the question).
  - The LLM's per-node judgment (concept-match, derivability, stay-on-symbol).
  - SymPy doing all math. Stage 1 parsing. The v7.1.x unit/dimension/crash
    fixes (those remain and still apply).
  - The architecture's principle: the LLM is the only thing that decides fit;
    the graph just hands it a small, relevant, connected set to judge.

## Tests

57 deterministic tests pass. The F=ma chain (the canonical
density→mass→force case) resolves end-to-end through the neighbor-walk:
  m: general_density_definition  (landing_source=neighbor)
  a: kinematics_v2_u2_2as         (landing_source=neighbor)
  F: laws_of_motion_newton_second_law  (landing_source=symbol, Round 0)

## Deployment

```bash
tar -xzf physics_solver_v7_2_tar.gz
cd physics_solver_v7
python tests/test_deterministic.py        # expect: 57 passed
python tests/test_live.py                  # re-run the 5 on the 7B
```

No re-ingest needed. Embedder stays CPU, 7B stays resident (v7.1.10 settings).

## What to watch in live testing

The neighbor sets are small but can include the known v7.3 graph-content
quirks (e.g. the malformed momentum_collisions_conservation_two_body shows up
when chasing velocity). The LLM's meaning-judgment should reject those. If a
chain still fails, the evaluate_bank cross-tab will show whether it's the
model's judgment over the neighbor set (a model question) or a missing graph
connection (a graph-content question) — cleanly separated now that the
symbol/dimension gate is gone.
