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
        self.model = SentenceTransformer(EMBED_MODEL)

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
        return results
