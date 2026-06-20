#!/usr/bin/env python3
"""
scripts/verify_rag.py
v7.1.2 — Verify the RAG pipeline is actually working.

This script answers three questions you should be able to answer at any
time without guesswork:

  1. Is ChromaDB indexed against the SAME rag_texts that are in the graph
     file? (If not, you ran the batch generator/exemplar apply but didn't
     re-ingest. ChromaDB will retrieve based on OLD embeddings while the
     LLM sees NEW rag_texts. Disaster mode, silent.)

  2. For a given test question (e.g. "Find the net force on a body of
     density 8000 kg/m^3 and volume 0.5 m^3..."), what does retrieval
     actually return? Are the right equations in the top-k? Are
     unrelated equations ranked above them?

  3. For a given equation ID, what's the closest rag_text in the corpus?
     If two equations have rag_texts that vector-embed too similarly,
     retrieval will have trouble distinguishing them.

Usage:
  python scripts/verify_rag.py --check-sync
      Compare every rag_text in the graph with what's in ChromaDB.
      Reports any mismatches (stale-embedding bug).

  python scripts/verify_rag.py --query "Find the net force on a body of density 8000 kg/m^3 and volume 0.5 m^3 that starts from rest..."
      Run a retrieval query and show the top-10 results with scores.

  python scripts/verify_rag.py --neighbors laws_of_motion_newton_second_law
      Show the 5 nearest neighbors in vector space for this equation —
      these are the equations the LLM will be most easily confused with.

  python scripts/verify_rag.py --all
      Run all three checks with sensible defaults.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MAIN_GRAPH_PATH, CHROMA_DIR, COLLECTION_NAME
from solver.graph_index import GraphIndex


def _load_chroma_collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(name=COLLECTION_NAME)


def check_sync():
    """
    Compare rag_texts in graph file vs what's stored in ChromaDB.
    A mismatch means: you regenerated rag_texts but didn't re-ingest.
    """
    print("Loading graph...")
    with open(MAIN_GRAPH_PATH) as f:
        graph = json.load(f)
    graph_rag = {n["id"]: n.get("rag_text", "") for n in graph["nodes"]}

    print("Loading ChromaDB collection...")
    try:
        coll = _load_chroma_collection()
    except Exception as e:
        print(f"\n❌ Could not open ChromaDB at {CHROMA_DIR}: {e}")
        print(f"   Run: python -m solver.ingest --reingest")
        sys.exit(2)

    print(f"  Graph nodes:       {len(graph_rag)}")
    print(f"  ChromaDB count:    {coll.count()}")

    # Pull all stored documents
    stored = coll.get(include=["documents"])
    stored_map = dict(zip(stored["ids"], stored["documents"]))

    only_in_graph    = set(graph_rag) - set(stored_map)
    only_in_chroma   = set(stored_map) - set(graph_rag)
    mismatched_text  = [
        nid for nid in graph_rag
        if nid in stored_map and stored_map[nid] != graph_rag[nid]
    ]

    print()
    print("=== SYNC CHECK ===")
    print(f"  In graph, missing from ChromaDB:  {len(only_in_graph)}")
    if only_in_graph:
        for n in list(only_in_graph)[:5]: print(f"    - {n}")
        if len(only_in_graph) > 5: print(f"    ...and {len(only_in_graph)-5} more")
    print(f"  In ChromaDB, missing from graph:  {len(only_in_chroma)}")
    if only_in_chroma:
        for n in list(only_in_chroma)[:5]: print(f"    - {n}")
    print(f"  IDs present in both but text differs: {len(mismatched_text)}")
    if mismatched_text:
        print("    ⚠️  STALE INDEX — these IDs have a different rag_text in ChromaDB")
        print("       than the current graph file. Run:")
        print("       python -m solver.ingest --reingest")
        for n in mismatched_text[:5]:
            g_len = len(graph_rag[n]); c_len = len(stored_map[n])
            print(f"    - {n}: graph={g_len} chars, chroma={c_len} chars")
        if len(mismatched_text) > 5:
            print(f"    ...and {len(mismatched_text)-5} more")
        return False

    if not only_in_graph and not only_in_chroma:
        print("\n✅ Sync OK — ChromaDB is consistent with the graph file.")
        return True
    else:
        print("\n⚠️  Sync mismatch (counts differ). Run: python -m solver.ingest --reingest")
        return False


def run_query(query: str, top_k: int = 10):
    """
    Run a retrieval query and show what comes back.
    Use this when you suspect retrieval is missing the right equation.
    """
    print(f"Loading graph + retriever (cold start can take ~10s)...")
    graph = GraphIndex(MAIN_GRAPH_PATH)
    from solver.retrieval import Retriever
    retriever = Retriever.try_load(graph)
    if retriever is None:
        print(f"\n❌ Could not load retriever. Run: python -m solver.ingest --reingest")
        sys.exit(2)

    print(f"\nQuery: {query!r}")
    print(f"Top-{top_k} hybrid retrieval results:\n")
    results = retriever.search(query, top_k=top_k)
    for i, r in enumerate(results, 1):
        node = r["node"]
        rag_preview = (node.get("rag_text", "")[:140] + "...") if len(node.get("rag_text", "")) > 140 else node.get("rag_text", "")
        print(f"  #{i}  score={r['score']:.4f}  sem={r['semantic_score']:.4f}  bm25={r['bm25_score']:.4f}")
        print(f"      {node['id']}")
        print(f"      eq: {node['equation_str']}")
        print(f"      rag: {rag_preview}")
        print()


def show_neighbors(eq_id: str, top_k: int = 5):
    """
    For a given equation, show the closest semantic neighbors in the corpus.
    These are the equations the LLM is most likely to confuse it with.
    """
    print("Loading graph + retriever (cold start can take ~10s)...")
    graph = GraphIndex(MAIN_GRAPH_PATH)
    from solver.retrieval import Retriever
    retriever = Retriever.try_load(graph)
    if retriever is None:
        print(f"\n❌ Could not load retriever. Run: python -m solver.ingest --reingest")
        sys.exit(2)

    if eq_id not in graph.nodes_by_id:
        print(f"\n❌ {eq_id} not in graph.")
        all_ids = sorted(graph.nodes_by_id.keys())
        matches = [i for i in all_ids if eq_id in i][:5]
        if matches:
            print(f"   Did you mean: {matches}")
        sys.exit(2)

    target = graph.nodes_by_id[eq_id]
    print(f"\nTarget:  {eq_id}")
    print(f"         eq: {target['equation_str']}")
    print(f"         concept: {target.get('rag_text', '')[:100]}...\n")
    # Use the target's rag_text as the query — semantic neighbors are
    # whatever else clusters near it
    results = retriever.search(target["rag_text"], top_k=top_k + 1)
    # Filter out the target itself
    results = [r for r in results if r["node"]["id"] != eq_id][:top_k]
    print(f"Closest semantic neighbors (concept-collision risks):\n")
    for i, r in enumerate(results, 1):
        node = r["node"]
        rag_preview = (node.get("rag_text", "")[:140] + "...") if len(node.get("rag_text", "")) > 140 else node.get("rag_text", "")
        print(f"  #{i}  sem={r['semantic_score']:.4f}  {node['id']}")
        print(f"      eq: {node['equation_str']}")
        print(f"      {rag_preview}")
        print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check-sync", action="store_true",
                   help="Verify ChromaDB matches the graph file.")
    p.add_argument("--query", type=str, default=None,
                   help="Run a retrieval query and show top-k results.")
    p.add_argument("--neighbors", type=str, default=None,
                   help="For an equation ID, show its semantic neighbors.")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--all", action="store_true",
                   help="Run sync check + a few canned queries.")
    args = p.parse_args()

    if not (args.check_sync or args.query or args.neighbors or args.all):
        p.print_help()
        sys.exit(1)

    if args.check_sync or args.all:
        ok = check_sync()
        if not ok and not args.all:
            sys.exit(1)
        print()

    if args.query:
        run_query(args.query, args.top_k)
        print()

    if args.neighbors:
        show_neighbors(args.neighbors, args.top_k)

    if args.all:
        print("=== CANNED QUERIES (the stress-test cases) ===\n")
        for q in [
            "Newton's second law applied to find net force on a body undergoing acceleration",
            "Archimedes' principle for the buoyant force on a body submerged in fluid",
            "Coulomb's law for the electrostatic force between two point charges",
            "Density as the bridge from given density and volume to mass needed in Newton's second law",
        ]:
            run_query(q, top_k=5)
            print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
