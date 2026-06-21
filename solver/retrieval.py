"""
solver/retrieval.py
Hybrid retrieval: BAAI/bge-large semantic search + BM25 keyword search.

v7 changes:
  - Restored to the live pipeline (v6 had this file present but unused; even
    the import was broken — BGE_QUERY_PREFIX was referenced but undefined in
    config.py).
  - Added try_load() classmethod for graceful absence of the index. If the
    user hasn't run `python -m solver.ingest` yet, this returns None and the
    pipeline silently falls back to symbol-only landing.
  - search() now returns (node, scores) tuples instead of bare nodes so the
    landing layer can show the LLM why each candidate was surfaced.
"""
from __future__ import annotations
import os
import pickle
from pathlib import Path

import numpy as np

from config import (
    CHROMA_DIR, BM25_INDEX_PATH, COLLECTION_NAME,
    EMBED_MODEL, BGE_QUERY_PREFIX,
    HYBRID_ALPHA, RAG_TOP_K, GRAPH_HOPS,
)


class Retriever:
    """
    Wraps ChromaDB + BM25 over the equation graph's rag_text field.
    Instantiate once per process; .search() per query.

    Heavy: loading the BGE embedding model is ~1.3 GB and takes ~10s on cold
    start. PhysicsSolver lazily instantiates a single Retriever instance at
    server startup.
    """

    def __init__(self, graph_index):
        # Heavy imports kept inside __init__ so the module can be imported
        # cheaply even when no Retriever is going to be constructed (e.g.
        # in deterministic tests that mock out retrieval).
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.graph   = graph_index
        self.n_nodes = len(graph_index.nodes)
        self._id_to_idx = {nid: i for i, nid in enumerate(
            n["id"] for n in graph_index.nodes
        )}

        print(f"[Retriever] Loading embedding model: {EMBED_MODEL}")
        # v7.1.10: pin the embedding model to a chosen device. On an 8GB GPU
        # shared with the LLM and the display, putting BGE on the GPU
        # (~1.3GB) steals headroom the 7B model needs to stay resident.
        # The embedder runs only ONCE per question (embedding the search
        # query), so CPU is nearly free in wall-clock terms and frees that
        # VRAM for the model. Override with EMBED_DEVICE=cuda if you ever
        # want it on GPU.
        _embed_device = os.getenv("EMBED_DEVICE", "cpu")
        print(f"[Retriever] Embedding model device: {_embed_device} "
              f"(set EMBED_DEVICE=cuda to override)")
        self.model = SentenceTransformer(EMBED_MODEL, device=_embed_device)

        print(f"[Retriever] Connecting to ChromaDB at {CHROMA_DIR}")
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = client.get_collection(COLLECTION_NAME)

        print(f"[Retriever] Loading BM25 index from {BM25_INDEX_PATH}")
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25     = bm25_data["bm25"]
        self.node_ids = bm25_data["node_ids"]

        print("[Retriever] Ready.")

    # ── Graceful loader ───────────────────────────────────────────────────────
    @classmethod
    def try_load(cls, graph_index) -> "Retriever | None":
        """
        Returns a Retriever if the ChromaDB + BM25 indexes both exist on disk.
        Returns None if either is missing — caller falls back to symbol-only
        landing. Never raises; the absence of an index is an expected state
        during initial deployment or in test environments.
        """
        chroma_path = Path(CHROMA_DIR)
        bm25_path   = Path(BM25_INDEX_PATH)
        if not chroma_path.exists() or not bm25_path.exists():
            return None
        try:
            return cls(graph_index)
        except Exception as e:
            # Lower than a print — but visible. If Chroma was started but the
            # collection doesn't exist, fall back rather than crashing the
            # whole solver.
            print(f"[Retriever] Could not load: {e!r}. Falling back to "
                  f"symbol-only landing.")
            return None

    # ── Core search ───────────────────────────────────────────────────────────
    def rank_candidates(
        self,
        query:      str,
        candidates: list[dict],
        top_k:      int = 8,
    ) -> list[dict]:
        """
        v7.2.1 — ORDER a fixed candidate set by relevance to the query and
        return the top_k. This is NOT a search over the corpus and NOT a
        rejection filter: the candidates are supplied by the graph
        neighbor-walk, and we only re-order them so the most contextually
        relevant appear first, then take the top_k for display.

        Why this exists: neighbor sets are tiny for rare variables (mass ~5)
        but large for common ones (velocity ~37). Dumping 37 equations into
        the Stage 2 prompt overloads the model (it stops choosing and starts
        chatting). Ranking by the question's own search query and showing the
        top_k keeps the set small WITHOUT gating any equation out on a
        symbol/dimension criterion — every candidate remains reachable; those
        below the cut are simply not shown in this view, and the resolver's
        fallback tiers still apply if the top_k dead-ends.

        Scoring reuses the same hybrid (semantic + BM25) signal as search(),
        but computed ONLY over the supplied candidates — no global corpus
        query. If there are <= top_k candidates, they're returned ranked but
        not cut.
        """
        if not candidates:
            return []
        if not query:
            # No query to rank by — preserve given order, just cap.
            return candidates[:top_k] if top_k else candidates

        # Semantic: embed the query once, score each candidate's stored
        # embedding via the collection (by id), falling back to 0 if missing.
        q_emb = self.model.encode(
            BGE_QUERY_PREFIX + query,
            normalize_embeddings=True,
        ).tolist()
        cand_ids = [c["id"] for c in candidates]
        sem_by_id = {cid: 0.0 for cid in cand_ids}
        # Query the collection restricted to these ids (Chroma supports a where
        # filter on ids); fall back to a full query intersected with our set.
        try:
            sem_res = self.collection.query(
                query_embeddings=[q_emb],
                n_results=max(len(cand_ids), 1),
                ids=cand_ids,
                include=["distances"],
            )
            for res_id, dist in zip(sem_res["ids"][0], sem_res["distances"][0]):
                if res_id in sem_by_id:
                    sem_by_id[res_id] = 1.0 - dist
        except Exception:
            # Older Chroma without `ids` arg: score over all, then intersect.
            sem_res = self.collection.query(
                query_embeddings=[q_emb],
                n_results=self.n_nodes,
                include=["distances"],
            )
            for res_id, dist in zip(sem_res["ids"][0], sem_res["distances"][0]):
                if res_id in sem_by_id:
                    sem_by_id[res_id] = 1.0 - dist

        # BM25 over the candidates only.
        import numpy as _np
        bm25_all = _np.array(self.bm25.get_scores(query.lower().split()))
        bm25_by_id = {}
        for cid in cand_ids:
            i = self._id_to_idx.get(cid)
            bm25_by_id[cid] = float(bm25_all[i]) if i is not None else 0.0

        # Normalize each signal within the candidate set, then combine.
        sem_vals = _np.array([sem_by_id[c] for c in cand_ids])
        bm_vals  = _np.array([bm25_by_id[c] for c in cand_ids])
        if sem_vals.max() > 0:
            sem_vals = sem_vals / sem_vals.max()
        if bm_vals.max() > 0:
            bm_vals = bm_vals / bm_vals.max()
        combined = HYBRID_ALPHA * sem_vals + (1 - HYBRID_ALPHA) * bm_vals

        order = _np.argsort(combined)[::-1]
        ranked = [candidates[i] for i in order]
        cut = ranked[:top_k] if top_k else ranked

        from solver.solver_log import log
        log("neighbor_rank",
            query=query, n_candidates=len(candidates), shown=len(cut),
            top_ids=[c["id"] for c in cut])
        return cut

    def search(
        self,
        query:     str,
        top_k:     int   = RAG_TOP_K,
        alpha:     float = HYBRID_ALPHA,
    ) -> list[dict]:
        """
        Returns ordered list of dicts: {node, score, semantic_score, bm25_score}.
        Highest relevance first. No graph expansion — this is the landing layer
        only; frontier expansion is the graph_index's job.
        """
        n   = self.n_nodes
        idx = self._id_to_idx

        # ── Semantic ──────────────────────────────────────────────────────────
        q_emb = self.model.encode(
            BGE_QUERY_PREFIX + query,
            normalize_embeddings=True,
        ).tolist()
        sem_res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=n,
            include=["distances"],
        )
        sem_scores = np.zeros(n)
        for res_id, dist in zip(sem_res["ids"][0], sem_res["distances"][0]):
            i = idx.get(res_id)
            if i is not None:
                sem_scores[i] = 1.0 - dist
        if sem_scores.max() > 0:
            sem_scores = sem_scores / sem_scores.max()

        # ── BM25 ──────────────────────────────────────────────────────────────
        bm25_raw = np.array(self.bm25.get_scores(query.lower().split()))
        bm25_scores = bm25_raw / bm25_raw.max() if bm25_raw.max() > 0 else bm25_raw

        # ── Combine ───────────────────────────────────────────────────────────
        combined    = alpha * sem_scores + (1 - alpha) * bm25_scores
        top_indices = np.argsort(combined)[::-1][:top_k]

        results = []
        for i in top_indices:
            nid = self.node_ids[i]
            if nid not in self.graph.nodes_by_id:
                continue
            results.append({
                "node":           self.graph.nodes_by_id[nid],
                "score":          round(float(combined[i]),    4),
                "semantic_score": round(float(sem_scores[i]),  4),
                "bm25_score":     round(float(bm25_scores[i]), 4),
            })

        # v7.1.2: log the retrieval result. This is the line to inspect when
        # answering "did my RAG find the right equation?". Shows the query,
        # top-k IDs with hybrid scores, and the semantic/BM25 breakdown so you
        # can tell whether ChromaDB or BM25 surfaced each hit.
        from solver.solver_log import log
        log("retrieval_search",
            query=query, top_k=top_k, alpha=alpha,
            n_corpus=n,
            results=[{
                "id":      r["node"]["id"],
                "score":   r["score"],
                "sem":     r["semantic_score"],
                "bm25":    r["bm25_score"],
            } for r in results])
        return results
