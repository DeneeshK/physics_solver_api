"""
solver/landing.py
v7: The Round-0 landing layer.

Per the original architecture brief, the initial equation-finding step
("landing") is supposed to combine:

  - Symbol-table lookup: every equation containing the target symbol, then
    filtered by dimension compatibility. This is what v6 did exclusively.
  - ChromaDB semantic lookup: top-K equations whose rag_text matches the
    question's scenario, by hybrid BGE + BM25. v6 had this code but never
    called it; v7 restores it.

The two sets are unioned, deduped, and passed together to the Stage 2 LLM
selector. The LLM picks one based on which equation describes the physics —
not on which list it came from.

Why a union (not a replacement):

  Safety property — any equation v6 would have shown is still shown in v7.
  ChromaDB landing can ONLY add candidates the LLM gets to consider; it
  never removes one. This guarantees no regression from enabling Chroma
  on a question v6 already handled.

Why the LLM, not a re-ranker, makes the final pick:

  The brief's principle — text-similarity scoring is what causes F=ρgh to
  beat F=ma in a kinematics question — applies to the ranking step too.
  The hybrid score is used only to surface candidates, never to choose
  among them.
"""
from __future__ import annotations
from typing import Optional


def get_landing_candidates(
    *,
    graph_index,
    target_symbol:     str,
    target_name:       str,
    target_dimension:  str,
    search_query:      str,
    visited_eqs:       set,
    allowed_domains:   Optional[set] = None,
    retriever          = None,  # Retriever | None
    rag_top_k:         int   = 5,
) -> list[dict]:
    """
    Round-0 candidates for the initial unknown. Combines two sources:

      1. SYMBOL: graph_index.candidates_for_quantity(target_symbol, ...)
         — every equation containing target_symbol, dimension-filtered,
         optionally domain-filtered (with the same fallback-to-full rule
         as before).
      2. SEMANTIC: retriever.search(search_query, top_k=rag_top_k)
         — every equation whose rag_text hybrid-matches the scenario.
         Skipped silently if retriever is None.

    Each candidate has a 'landing_source' field tagged 'symbol', 'semantic',
    or 'both' so Stage 2's prompt can tell the LLM what flagged it. This is
    informational for the model only — it does not affect ranking.

    Returns: list of equation node dicts (each augmented with landing_source).
    Order: symbol candidates first (preserves v6 ordering for that subset),
    then semantic-only candidates ordered by hybrid score.
    """
    # --- Source 1: symbol-table candidates (v6 behavior, exactly) ---
    symbol_candidates = graph_index.candidates_for_quantity(
        needed_symbol=target_symbol,
        needed_name=target_name,
        needed_dimension=target_dimension,
        visited_eqs=visited_eqs,
        allowed_domains=allowed_domains,
    )
    symbol_ids = {c["id"] for c in symbol_candidates}

    # --- Source 2: semantic candidates (new in v7, optional) ---
    semantic_results = []
    if retriever is not None and search_query:
        try:
            semantic_results = retriever.search(search_query, top_k=rag_top_k)
        except Exception as e:
            # Retrieval failure must not break the solver. Log and continue
            # with symbol-only candidates — this is the same "safe fallback"
            # philosophy as Retriever.try_load.
            print(f"[landing] Retriever.search failed: {e!r}. "
                  f"Continuing with symbol-only candidates.")
            semantic_results = []

    # --- Tag each candidate with where it came from ---
    out = []
    for c in symbol_candidates:
        # Stamp a copy, not the original — same node may appear in multiple
        # questions per process; we don't want stale 'landing_source' on it.
        node = dict(c)
        node["landing_source"] = "symbol"
        out.append(node)

    semantic_only_added = 0
    for r in semantic_results:
        node = r["node"]
        if node["id"] in visited_eqs:
            continue
        if node["id"] in symbol_ids:
            # Promote shared candidates' source to 'both'. This lets Stage 2's
            # prompt note that an equation was confirmed by BOTH lookups —
            # a stronger signal than either alone.
            for existing in out:
                if existing["id"] == node["id"]:
                    existing["landing_source"] = "both"
                    existing["_retrieval_score"] = r["score"]
                    break
            continue
        # Apply the same dimension filter we'd apply to a symbol candidate,
        # so the LLM doesn't see e.g. an optics fringe-order m as a
        # candidate for "mass" just because the rag_text happens to mention
        # the word 'mass' incidentally.
        # NOTE: this filter is on the *target's* dimension, not on the
        # equation's natural output. An equation can be a legitimate
        # candidate even if its natural output isn't target_symbol —
        # e.g. F=ma can produce m by rearrangement. So we only filter
        # equations that contain target_symbol AT ALL with a dimensionally-
        # incompatible meaning. If target_symbol isn't even in the
        # equation, the equation is still a candidate — Stage 2 may
        # choose it for cross-equation chaining.
        if target_symbol in node["variables"]:
            stored_dim = node["variables"][target_symbol].get("dimension", "")
            from solver.graph_loader import _dimensions_compatible
            if not _dimensions_compatible(stored_dim, target_dimension):
                continue
        # Apply domain filter ONLY if it would still leave at least one
        # candidate overall. We've already passed the safety threshold for
        # symbol candidates above; if domain narrowing would now exclude
        # a semantic-only result, that's fine as long as we still have
        # symbol candidates to show.
        if allowed_domains and node.get("domain") not in allowed_domains:
            # Skip silently — same fallback semantics: if symbol set is
            # non-empty, narrowing is safe. If symbol set is empty AND
            # this is our only hope, we keep it.
            if symbol_candidates:
                continue
        tagged = dict(node)
        tagged["landing_source"] = "semantic"
        tagged["_retrieval_score"] = r["score"]
        out.append(tagged)
        semantic_only_added += 1

    return out


def is_chroma_landing_enabled(retriever_or_none) -> bool:
    """Just a readability helper. Used in logging."""
    return retriever_or_none is not None
