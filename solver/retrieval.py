"""
solver/retrieval.py
Hybrid retrieval: BAAI/bge-large semantic search + BM25 keyword search.
Returns top-k equation nodes + their graph neighbors as the candidate pool.
"""
import pickle
from pathlib import Path

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from config import (
    CHROMA_DIR, BM25_INDEX_PATH, COLLECTION_NAME,
    EMBED_MODEL, BGE_QUERY_PREFIX,
    HYBRID_ALPHA, RAG_TOP_K, GRAPH_HOPS,
)


class Retriever:
    """
    Wraps ChromaDB + BM25 + graph expansion.
    Instantiate once, call .search() per query.
    """

    def __init__(self, graph_index):
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
        self.node_ids = bm25_data["node_ids"]   # ordered list for BM25 score array

        print("[Retriever] Ready.")

    # ── Core search ───────────────────────────────────────────────────────────
    def search(
        self,
        query:     str,
        top_k:     int   = RAG_TOP_K,
        alpha:     float = HYBRID_ALPHA,
        expand:    bool  = True,
        hops:      int   = GRAPH_HOPS,
    ) -> list[dict]:
        """
        Returns ordered list of equation node dicts.
        Seeds (highest relevance) come first.
        If expand=True, one-hop graph neighbors are appended.
        """
        seed_nodes = self._hybrid_search(query, top_k, alpha)
        if not expand:
            return seed_nodes

        seed_ids  = [n["id"] for n in seed_nodes]
        all_nodes = self.graph.expand_neighbors(seed_ids, hops=hops)
        return all_nodes

    def search_scores(
        self,
        query: str,
        top_k: int   = RAG_TOP_K,
        alpha: float = HYBRID_ALPHA,
    ) -> list[dict]:
        """
        Same as search() but returns dicts with score breakdown.
        Useful for debugging retrieval quality.
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
            sem_scores /= sem_scores.max()

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

    # ── Private ───────────────────────────────────────────────────────────────
    def _hybrid_search(self, query: str, top_k: int, alpha: float) -> list[dict]:
        scored = self.search_scores(query, top_k, alpha)
        return [r["node"] for r in scored]
