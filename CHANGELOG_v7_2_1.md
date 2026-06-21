# Changelog v7.2.1 — Rank-and-cap large neighbor sets (fixes model overload)

The v7.2 neighbor-walk worked, but the first live run on the 7B exposed a
problem the small-set testing had hidden: neighbor sets are tiny for rare
variables (mass ~5, acceleration ~12) but LARGE for common ones (velocity ~37,
force ~20). Dumping 37 equations into the Stage 2 prompt (an 18,000-character
request) overloaded the 7B — it stopped choosing and started chatting ("If you
need help with any specific equation, please let me know!"), which scored as a
no-fit. Every failure in that run was a round with 20-37 candidates; every
success had 4-6.

## The fix: order, don't reject

When chasing a variable mid-chain, the neighbor set is now RANKED by relevance
to the question and capped to the top ~8. This is ordering, NOT a rejection
filter — every neighbor remains reachable; those below the cut simply aren't
shown in this view, and the resolver's fallback tiers still apply if the top-8
dead-end. Small sets (<= 8) are shown whole, unchanged.

The ranking reuses the same hybrid (semantic + BM25) signal as Round-0
retrieval, but computed ONLY over the supplied neighbor candidates — it is NOT
a global corpus search. So the design constraints hold: no global semantic
search mid-chain (we rank a set the graph already produced), and no
symbol/dimension rejection gate (that stays gone from v7.2).

This is the approach the user chose: "use the question's context to rank
neighbors, show top ~8 (light semantic, only to ORDER not reject)."

## Code changes

  - `solver/retrieval.py`:
    - `rank_candidates(query, candidates, top_k=8)` (new) — orders a FIXED
      candidate set by hybrid relevance to the query and returns top_k. Not a
      corpus search; scores only the supplied candidates. Falls back to
      given order if no query. Logs `neighbor_rank`.
  - `solver/landing.py`:
    - `get_neighbor_candidates` gains `search_query`, `retriever`, `top_k`
      params. After the neighbor walk, if the set exceeds top_k AND a
      retriever+query are available, it ranks-and-caps. Tier-3 fallback also
      ranked-and-capped. Small sets returned whole.
  - `solver/frontier_resolver.py`:
    - Round 1+ passes `search_query` and `retriever` into
      get_neighbor_candidates so large neighbor sets get ranked.

## What this fixes (from the live log)

  - KE chain Round 1 chasing `v`: was 37 candidates / 18k chars → model gave
    up. Now top-8 ranked; v=u+at and v²=u²+2as should rank at the top.
  - F=ma direct Round 1 chasing `F`: was 20 candidates → drift. Now top-8.
  - F=ma chain Round 0 chasing `F`: was 21 candidates → malformed output.
    (Round 0 uses get_landing_candidates, already top-k from retrieval; the
    21 there came from the symbol+semantic union — see note below.)

## VERIFIED vs. NEEDS-YOUR-CONFIRMATION

VERIFIED in the sandbox:
  - 57 deterministic tests pass.
  - rank-and-cap logic: without a retriever returns the full set (no silent
    gating); with one, caps to top_k ordered by relevance (tested with a
    controlled stub).

NEEDS YOUR MACHINE TO CONFIRM (the sandbox has no ChromaDB index or embedding
model, so I could not run the REAL ranking):
  - That BGE+BM25 actually ranks the physically-correct velocity equations
    (v=u+at, v²=u²+2as) into the top 8 out of velocity's 37 neighbors. The
    logic is right; whether the embedding puts the right physics on top is
    what your run will show. Check the new `neighbor_rank` log line — it lists
    the top_ids shown for each hop.

## Known remaining issues (NOT fixed here, noted for next)

  1. Round 0 still uses the symbol+semantic UNION (get_landing_candidates),
     which produced 21 candidates for `F` in the F=ma-chain case. If Round 0
     also overloads, the same rank-and-cap idea should apply there — but
     Round 0's union is a different code path and the user hasn't approved
     changing it yet.
  2. Constants bloat: `R_g` (gas constant), `G`, `epsilon_0` are being
     injected into unrelated questions (e.g. R_g into a pure kinematics
     "find velocity"). Padding, not fatal, but adds prompt noise. Separate
     fix.
  3. The 7B still occasionally drifts off the asked symbol under load and
     occasionally emits malformed JSON (the "concept" field instead of
     decision/chosen_eq_id). Smaller candidate sets should reduce both;
     whether they fully resolve is a model-capacity question the bank will
     quantify.

## Deployment

```bash
tar -xzf physics_solver_v7_2_1_tar.gz
cd physics_solver_v7
python tests/test_deterministic.py        # expect: 57 passed
python tests/test_live.py                  # re-run the 5; watch neighbor_rank logs
```

No re-ingest needed. Embedder stays CPU, 7B resident.
