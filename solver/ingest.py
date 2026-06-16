"""
solver/ingest.py
Builds ChromaDB (semantic) and BM25 (keyword) indexes from the equation graph.
Run once: python -m solver.ingest
Re-run with --reingest to rebuild from scratch.
"""
import json
import pickle
import argparse
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MAIN_GRAPH_PATH, CHROMA_DIR, BM25_INDEX_PATH,
    COLLECTION_NAME, EMBED_MODEL,
)


def build_bm25_document(node: dict) -> str:
    """
    Rich tokenizable string for BM25.
    Combines rag_text + equation symbols + variable plain-English names
    + domain + chapter names for best keyword coverage.
    """
    parts = [node["rag_text"], node["equation_str"]]

    for sym, meta in node["variables"].items():
        parts.append(sym)
        parts.append(meta.get("name", ""))

    parts.append(node["domain"].replace("_", " "))
    parts.append(node["subdomain"].replace("_", " "))
    parts.extend(node.get("jee_chapters", []))
    parts.extend(node.get("conditions", []))

    return " ".join(p for p in parts if p).lower()


def build_chroma_metadata(node: dict) -> dict:
    """Flat string metadata for ChromaDB storage."""
    return {
        "node_id":       node["id"],
        "domain":        node["domain"],
        "subdomain":     node["subdomain"],
        "equation_str":  node["equation_str"],
        "sympy_expr":    node["sympy_expr"],
        "variables":     ",".join(node["variables"].keys()),
        "conditions":    "|".join(node.get("conditions", [])),
        "jee_chapters":  "|".join(node.get("jee_chapters", [])),
        "neet_chapters": "|".join(node.get("neet_chapters", [])),
    }


def ingest(force: bool = False):
    print(f"Loading graph from {MAIN_GRAPH_PATH}")
    with open(MAIN_GRAPH_PATH, encoding="utf-8") as f:
        graph = json.load(f)
    nodes = graph["nodes"]
    print(f"  {len(nodes)} equation nodes")

    chroma_done = Path(CHROMA_DIR).exists()
    bm25_done   = Path(BM25_INDEX_PATH).exists()

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if force and chroma_done:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("  Cleared existing ChromaDB collection")
        except Exception:
            pass
        chroma_done = False

    if not chroma_done:
        print(f"\nLoading embedding model: {EMBED_MODEL}")
        print("  (First run downloads ~1.3 GB to HuggingFace cache)")
        model = SentenceTransformer(EMBED_MODEL)

        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        rag_texts = [n["rag_text"] for n in nodes]
        ids       = [n["vector_id"] for n in nodes]
        metas     = [build_chroma_metadata(n) for n in nodes]

        print(f"Embedding {len(nodes)} nodes (batch_size=32)...")
        embeddings = model.encode(
            rag_texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).tolist()

        collection.add(
            ids=ids,
            documents=rag_texts,
            embeddings=embeddings,
            metadatas=metas,
        )
        print(f"ChromaDB stored → {CHROMA_DIR}/")
    else:
        print(f"ChromaDB already exists at {CHROMA_DIR}/ (use --reingest to rebuild)")
        model = SentenceTransformer(EMBED_MODEL)
        collection = client.get_collection(COLLECTION_NAME)

    # ── BM25 ──────────────────────────────────────────────────────────────────
    if force and bm25_done:
        Path(BM25_INDEX_PATH).unlink()
        bm25_done = False

    if not bm25_done:
        print("\nBuilding BM25 index...")
        node_ids  = [n["id"] for n in nodes]
        docs      = [build_bm25_document(n) for n in nodes]
        tokenized = [doc.split() for doc in docs]
        bm25      = BM25Okapi(tokenized)

        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump({"bm25": bm25, "node_ids": node_ids}, f)
        print(f"BM25 saved → {BM25_INDEX_PATH}")
    else:
        print(f"BM25 index already exists at {BM25_INDEX_PATH} (use --reingest to rebuild)")
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        bm25     = bm25_data["bm25"]
        node_ids = bm25_data["node_ids"]

    print("\nIngestion complete.")
    return model, collection, bm25, node_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reingest", action="store_true",
                        help="Force rebuild of all indexes")
    args = parser.parse_args()
    ingest(force=args.reingest)
