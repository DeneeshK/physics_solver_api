# Changelog v7.2.2 — Round-0 node rollback + lenient selection parsing

Two fixes. The first is the one that was MISSING from the architecture all
along: when a chosen node dead-ends, drop it and try the next of the top-5
retrieval nodes, instead of giving up.

## Fix 1 (the important one): rollback to the next Round-0 node on dead-end

The design always called for this: Round 0 retrieves the top-5 nodes; the LLM
walks them; if a chosen node leads to a chain that DEAD-ENDS (an introduced
variable can't be resolved, or the LLM says "none fits" mid-chain), the solver
must DROP that node and try the NEXT node from the top-5 — only failing after
all candidates are exhausted.

This was never implemented. `resolve_frontier` returned success=False on the
first dead-end, and the pipeline's backtrack loop only retried on Stage-3
(SymPy) failures, not Stage-2 dead-ends — so it gave up on the first node.

This is why the F=ma-CHAIN case failed: Round 0's retrieval for "find net
force" surfaced work/power equations, the LLM picked W=F*s*cos(theta) (a valid
but awkward path), that introduced `theta` (an angle the problem doesn't have),
`theta` dead-ended, and the solver QUIT — instead of dropping
work_constant_force and trying Newton's second law, which was sitting in the
top-5 at score 1.0.

The fix:
  - `solver/frontier_resolver.py`: ResolutionResult gains `dead_end_root_eq`
    — the Round-0 equation at the root of the dead-ended chain. Populated on
    the failure return (the first round-0 chosen step).
  - `solver/pipeline.py`: the backtrack loop now handles Stage-2 dead-ends.
    When resolution fails with a dead_end_root_eq, it EXCLUDES that root and
    retries. Because the resolver seeds visited_eqs from excluded_eqs, Round 0
    re-runs with that node banned, so get_landing_candidates surfaces the
    next-best retrieval node and the LLM picks from those. Logs
    `stage2_node_rollback`.
  - MAX_BACKTRACK_ATTEMPTS raised 3 → 6, so several top-5 nodes can be tried
    (plus Stage-3 exclusions) before giving up.

This is the "drop the node, go to the next of the top-5" rollback, built on
the retrieval metadata exactly as specified. Nothing is hardcoded; the LLM
still chooses among the (now node-excluded) candidates each retry.

## Fix 2: lenient selection parsing (recovers correct picks)

The 7B sometimes returns a BARE selection object instead of wrapping it in
{"selections": [...]}, e.g.:
    {"needed_symbol": "a", "decision": "pick", "chosen_eq_id": "..."}
The parser only read parsed["selections"], so a bare object yielded zero
selections and a CORRECT pick was discarded as "omitted". This is why the
F=ma-DIRECT case failed even though the model picked F=m*a perfectly.

Fix (`solver/llm_interface.py`): if `selections` is missing/empty but the
parsed object itself carries selection fields (needed_symbol / chosen_eq_id /
decision), treat the whole object as a single selection. Also handle a bare
top-level list. Generic — recovers any correct pick that lost its array
wrapper.

## Tests

59 deterministic tests pass (2 new):
  - test_dead_end_reports_root_for_rollback — a dead-ended chain reports its
    Round-0 root in dead_end_root_eq so the pipeline can exclude+retry.
  - test_bare_selection_object_parsed — a bare selection object is recovered;
    the wrapped shape still works.

## Expected effect on the 5 live tests

Last run: 3/5 (Coulomb, KE, underspecified). The two failures should now
recover:
  - F=ma-direct: bare-object pick now parsed → F=m*a resolves.
  - F=ma-chain: work_constant_force dead-ends on theta → that node excluded →
    Round 0 retries → Newton's second law (top-5, score 1.0) picked → m via
    density, a via v²=u²+2as → solves.
Whether the 7B's exact outputs cooperate is what the run will show, but the
mechanisms that were dropping correct work are now fixed.

## Deployment

```bash
tar -xzf physics_solver_v7_2_2_tar.gz
cd physics_solver_v7
python tests/test_deterministic.py        # expect: 59 passed
python tests/test_live.py                  # the 5; watch stage2_node_rollback
```

No re-ingest. Embedder CPU, 7B resident.

## Still open (noted honestly)

  - Round-0 retrieval surfacing work/power before Newton's second law for a
    "find force from motion" query is a retrieval-ranking quirk. The rollback
    now recovers from it, but the ideal would be retrieval ranking the most
    direct equation higher. Separate, lower-priority.
  - Constant bloat (R_g etc. injected into unrelated questions) — still
    present, still just prompt noise.
