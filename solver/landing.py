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


def get_landing_candidates_semantic(
    *,
    retriever,
    search_query:  str,
    visited_eqs:   set,
    top_k:         int = 8,
) -> list[dict]:
    """
    v7.2.4 — Round-0 landing by PURE semantic vector search, ranked.

    This is the user's intended landing design: the LLM reads the question,
    distills the concept, and a search_query is built describing the physics.
    We search the WHOLE vector DB with that query (cosine similarity over the
    182 concept-level rag_texts) and return the top_k most similar equation
    nodes, IN SIMILARITY ORDER. Nothing else — no symbol-table source (which
    dumped every equation containing the target symbol, e.g. all 21 equations
    with 'F'), and no domain filter (a mechanical string-match reject that
    silently dropped good equations when Stage 1 mis-spelled a domain name).

    The resolver walks this ranked list ONE NODE AT A TIME, asking the LLM
    "does this equation's concept match the question?" for each, committing to
    the first that fits — exactly like a student scanning their formula sheet
    from most-relevant down. Rejection is the LLM's concept judgment, never a
    symbol/dimension/domain rule.

    Returns equation-node dicts tagged landing_source="semantic", ranked.
    Excluded/visited nodes are filtered out so a dead-ended node isn't
    re-offered; the scan then naturally continues down the list.
    """
    if retriever is None or not search_query:
        return []
    try:
        results = retriever.search(search_query, top_k=top_k)
    except Exception as e:
        print(f"[landing] semantic search failed: {e!r}.")
        return []
    out = []
    for r in results:
        node = r["node"]
        if node["id"] in visited_eqs:
            continue
        n = dict(node)
        n["landing_source"] = "semantic"
        n["_retrieval_score"] = r.get("score")
        out.append(n)
    return out


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
    known_symbols:     Optional[set] = None,  # v7.1.4
    round_num:         int   = 0,             # v7.1.4
    round0_cap:        int   = 8,             # v7.2.4
) -> list[dict]:
    """
    Candidates for one frontier item. Combines two sources:

      1. SYMBOL: every equation containing target_symbol, dimension-filtered,
         optionally domain-filtered.
      2. SEMANTIC: top-K equations whose rag_text hybrid-matches the
         search_query. Skipped silently if retriever is None.

    v7.1.4 — knowns-overlap re-ranking for Round 1+:

      When round_num > 0 and known_symbols is non-empty, candidates are
      re-ranked so that equations sharing more variables with the known set
      bubble to the top. The rationale: in Round 1+, we're looking for an
      equation that BRIDGES from known values to the target symbol. The
      equation with the most overlap between its variables and our known
      values is the most likely bridge.

      Example: looking for 'm' with knowns = {rho, V, u, v, s}:
        - rho = m/V         shares rho, V  → overlap 2 (winner)
        - p = m*v           shares v       → overlap 1
        - K = (1/2)*m*v^2   shares v       → overlap 1
        - F = m*a           shares nothing → overlap 0

      Round 0 keeps the original ranking — there, the concept (not overlap)
      determines fit, and the LLM is told explicitly to ignore overlap as
      a primary signal. Applying overlap ranking in Round 0 would push
      e.g. F = rho*V*g above F = m*a in a kinematics problem just because
      the question gave rho and V (intended for mass derivation).

    Returns: list of equation node dicts, each augmented with landing_source.
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

    # v7.1.4: knowns-overlap re-ranking for Round 1+
    # In Round 1+ we're looking for an equation that BRIDGES known values to
    # the target. Equations whose variables overlap more with what's already
    # known are more likely to be that bridge. We re-rank candidates by
    # overlap count (descending) while preserving original order as the
    # tie-breaker via stable sort.
    if round_num > 0 and known_symbols:
        def overlap(node):
            return len(set(node.get("variables", {}).keys()) & known_symbols)
        # Stable sort: equations with more knowns-overlap float up; ties keep
        # the prior order (symbol-source first, then semantic-source).
        out.sort(key=overlap, reverse=True)

    # v7.2.4: Round-0 RANK-AND-CAP. The symbol source returns EVERY equation
    # containing target_symbol. Normally the domain filter in
    # candidates_for_quantity trims this to a handful, but that filter relies
    # on Stage 1 emitting a domain string that matches the graph's domain
    # names. When Stage 1 emits a near-miss (e.g. "newton_laws_of_motion"
    # instead of "laws_of_motion"), the domain filter matches nothing, falls
    # back to the FULL set, and Round 0 can balloon to 20+ candidates — which
    # overloads the 7B (it stops choosing and starts chatting, producing
    # llm_omitted_item). Mirror the neighbor-walk's rank-and-cap: when the set
    # is large and we can rank it, ORDER by relevance to the question and show
    # the top round0_cap. This is ordering, NOT rejection — the true concept
    # match (e.g. Newton's second law, the ChromaDB #1 hit) floats to the top
    # and survives the cut; node-rollback still reaches the rest across
    # attempts. Round 1+ is unaffected (it returns earlier via its own path).
    if round_num == 0 and retriever is not None and search_query and len(out) > round0_cap:
        out = retriever.rank_candidates(search_query, out, top_k=round0_cap)

    return out


def get_neighbor_candidates(
    *,
    graph_index,
    target_symbol:  str,
    from_eq_ids:    set,
    visited_eqs:    set,
    search_query:   str = "",
    retriever=None,
    top_k:          int = 8,
    known_symbols:  set = None,
) -> list[dict]:
    """
    v7.2 — Round 1+ candidate generation by GRAPH NEIGHBOR WALK, not symbol
    lookup or semantic search.

    When chasing a variable introduced mid-chain (e.g. `m` after choosing
    F=ma), the candidates are the graph neighbors of the equations already
    chosen that SHARE that variable. The graph's edges encode "these two
    equations share variable X", so this walks to exactly the equations
    connected, through `target_symbol`, to what we've already committed to.

    v7.2.1 — RANK-AND-CAP for large neighbor sets. Neighbor sets are tiny for
    rare variables (mass ~5) but large for common ones (velocity ~37), and
    dumping 37 equations overloads the model (it stops choosing and starts
    chatting). When a retriever + search_query are supplied AND the set
    exceeds top_k, we ORDER the neighbors by relevance to the question and
    show the top_k. This is ordering, NOT rejection — every neighbor stays
    reachable; those below the cut just aren't shown this view, and the
    fallback tiers still apply. Small sets (<= top_k) are shown whole.

    Tiers (the caller — resolver — handles roll back as the final tier):
      Tier 1/2: neighbors_sharing_variable(from chosen eqs, via target_symbol),
                then rank-and-cap to top_k if a retriever is available.
      Tier 3 (local fallback): if the neighbor walk yields NOTHING, widen to
                all_equations_with_variable (still LOCAL, not a global
                semantic search), also rank-and-capped.

    No symbol-presence gate (the walk IS the reachability), no dimension
    rejection. Returns equation-node dicts tagged with landing_source.
    """
    # Tier 1/2: neighbor walk from the equations already chosen.
    neighbors = graph_index.neighbors_sharing_variable(
        from_eq_ids = from_eq_ids,
        variable    = target_symbol,
        visited_eqs = visited_eqs,
    )
    out = []
    for node in neighbors:
        n = dict(node)
        n["landing_source"] = "neighbor"
        out.append(n)

    if out:
        # v7.2.5: rank by KNOWNS-OVERLAP first, THEN cap. The bug this fixes:
        # chasing `m` after F=m*a, the neighbor set (correctly) contains the
        # bridging equation density m=rho*V — but ranking by the ORIGINAL force
        # query ("net force / Newton") buried it (density isn't about "force"),
        # so the cap-at-top_k chopped it off and the model never saw it, picking
        # weight F=m*g instead and poisoning the chain. The fix is to prefer
        # equations whose OTHER variables we ALREADY HAVE: to find mass, density
        # uses rho and V (both known) — overlap 2; weight uses F (the goal,
        # unknown) — overlap 1. So density now ranks above weight and survives
        # the cap. This is ORDERING, not rejection (nothing is dropped from the
        # candidate set itself), and it mirrors how a student reasons: "to get
        # mass, use the relation whose other quantities I already know." The
        # semantic query is the tie-breaker within equal overlap.
        if known_symbols:
            def _overlap(eq):
                v = set(eq.get("variables", {}).keys()) - {target_symbol}
                return len(v & known_symbols)
            # Stable sort by descending overlap; ties keep prior (semantic/graph)
            # order, so the semantic relevance still orders equally-grounded eqs.
            if retriever is not None and search_query and len(out) > top_k:
                # Rank the FULL set semantically first (gives a sensible
                # within-overlap order), then stable-sort by overlap, then cap.
                out = retriever.rank_candidates(search_query, out, top_k=len(out))
            out.sort(key=_overlap, reverse=True)
            return out[:top_k]
        # No knowns provided → fall back to the prior semantic rank-and-cap.
        if retriever is not None and search_query and len(out) > top_k:
            out = retriever.rank_candidates(search_query, out, top_k=top_k)
        return out

    # Tier 3 (local fallback): equations containing the variable at all.
    # Still local (variable-membership), never a global semantic search.
    fallback = graph_index.all_equations_with_variable(
        variable    = target_symbol,
        visited_eqs = visited_eqs,
    )
    for node in fallback:
        n = dict(node)
        n["landing_source"] = "variable_fallback"
        out.append(n)
    if retriever is not None and search_query and len(out) > top_k:
        out = retriever.rank_candidates(search_query, out, top_k=top_k)
    return out


def is_chroma_landing_enabled(retriever_or_none) -> bool:
    """Just a readability helper. Used in logging."""
    return retriever_or_none is not None
