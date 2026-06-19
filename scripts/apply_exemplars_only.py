#!/usr/bin/env python3
"""
scripts/apply_exemplars_only.py
v7.1 — Apply the 16 hand-authored exemplar rag_texts to the graph.

This is a no-LLM-required step that gets you the gold-standard exemplars
into the graph immediately. After this runs, ChromaDB landing for those 16
equations is at full v7.1 quality even before the batch generator runs the
remaining ~166 equations.

Why this exists as a separate script:
  - The full batch (regenerate_rag_texts.py) takes a Groq key and a few
    minutes. The exemplars don't need either. You can ship the exemplar-
    only version, validate it on the stress tests, then run the full batch.
  - If the full batch later fails on some equations, those equations keep
    their old rag_text (it's not destructive). The exemplars are
    independently safe to apply.

After running this:
  - Run `python -m solver.ingest` to rebuild the ChromaDB index over
    the new rag_texts (the embeddings need to be regenerated; they're
    text-derived).
  - Run the deterministic tests to make sure nothing else broke.

Usage:
  python scripts/apply_exemplars_only.py             # apply, write graph
  python scripts/apply_exemplars_only.py --dry-run   # show diff, don't save
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
EXEMPLARS_PATH = SCRIPT_DIR / "rag_text_exemplars.json"
GRAPH_PATH     = PROJECT_ROOT / "data" / "physics_equation_graph_final.json"
BACKUP_PATH    = PROJECT_ROOT / "data" / "physics_equation_graph_final.json.bak"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change; don't save")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip writing .bak before saving")
    args = parser.parse_args()

    print(f"Loading exemplars from {EXEMPLARS_PATH}")
    with open(EXEMPLARS_PATH) as f:
        exemplar_data = json.load(f)
    exemplars: dict[str, dict] = exemplar_data["exemplars"]
    print(f"  {len(exemplars)} hand-authored exemplars loaded")

    print(f"Loading graph from {GRAPH_PATH}")
    with open(GRAPH_PATH) as f:
        graph = json.load(f)
    nodes = graph["nodes"]
    nodes_by_id = {n["id"]: n for n in nodes}

    missing_in_graph = [eid for eid in exemplars if eid not in nodes_by_id]
    if missing_in_graph:
        print(f"ERROR: these exemplar IDs are not in the graph: "
              f"{missing_in_graph}", file=sys.stderr)
        sys.exit(2)

    print(f"\nReplacing rag_text in {len(exemplars)} nodes:")
    diffs = []
    for eid, body in exemplars.items():
        node = nodes_by_id[eid]
        old_len = len(node.get("rag_text", ""))
        new_text = body["rag_text"]
        new_len  = len(new_text)
        diffs.append((eid, body["concept_name"], old_len, new_len))
        if not args.dry_run:
            node["rag_text"] = new_text

    for eid, concept, old_len, new_len in diffs:
        print(f"  {eid:55s}  {old_len:>4d} → {new_len:>4d} chars  ({concept})")

    if args.dry_run:
        print("\n--- DRY RUN: not saving ---")
        return

    if not args.no_backup:
        print(f"\nBacking up: {GRAPH_PATH} → {BACKUP_PATH}")
        shutil.copyfile(GRAPH_PATH, BACKUP_PATH)

    print(f"Writing: {GRAPH_PATH}")
    with open(GRAPH_PATH, "w") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    print("\nDone. Reminder: rebuild the ChromaDB index now:")
    print("  python -m solver.ingest")


if __name__ == "__main__":
    main()
