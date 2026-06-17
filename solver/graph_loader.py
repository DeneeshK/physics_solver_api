"""
solver/graph_loader.py
Loads the physics equation graph and builds fast in-memory lookup structures.
Called once at startup — all other modules import from here.

Changes from v1:
  - Added candidates_for_quantity() with dimension compatibility filtering.
    This is the §5 symbol-collision guardrail: keeps only equations whose
    variable matching `symbol` has a compatible dimension to what is needed.
"""
import json
from collections import defaultdict
from config import MAIN_GRAPH_PATH, NON_SOLVABLE_SYMBOLS


def load_graphs():
    """Returns a GraphIndex built from the main graph file."""
    with open(MAIN_GRAPH_PATH, encoding="utf-8") as f:
        main = json.load(f)
    return GraphIndex(main)


def _dimensions_compatible(stored_dim: str, needed_dim: str) -> bool:
    """
    True if needed_dim is compatible with stored_dim.
    Handles ambiguous stored dimensions like 'MLT-1 or ML2 or A' by
    checking if needed_dim matches ANY of the alternatives.
    If either is empty/unknown, returns True (can't filter).
    """
    if not stored_dim or not needed_dim:
        return True
    stored_parts = {p.strip() for p in stored_dim.split(" or ")}
    needed_parts = {p.strip() for p in needed_dim.split(" or ")}
    return bool(stored_parts & needed_parts)  # non-empty intersection


class GraphIndex:
    def __init__(self, main_graph: dict):
        # ── Equation nodes ────────────────────────────────────────────────────
        self.nodes: list[dict] = main_graph["nodes"]
        self.nodes_by_id: dict[str, dict] = {n["id"]: n for n in self.nodes}

        # ── Graph edges (kept for legacy neighbor-expansion if needed) ────────
        self.edges: list[dict] = main_graph["edges"]

        # ── Symbol → [equation_ids] index ────────────────────────────────────
        # sym_to_eqs["F"] → all equation IDs that contain symbol "F"
        self.sym_to_eqs: dict[str, list[str]] = defaultdict(list)
        for node in self.nodes:
            for sym in node["variables"]:
                self.sym_to_eqs[sym].append(node["id"])

        # ── Natural output map ────────────────────────────────────────────────
        # For each equation, which symbol does it most naturally solve for.
        self.natural_output: dict[str, str] = self._build_natural_output()

        # ── Adjacency (kept for legacy expand_neighbors) ──────────────────────
        self.adjacency: dict[str, set[str]] = defaultdict(set)
        for e in self.edges:
            self.adjacency[e["from"]].add(e["to"])
            self.adjacency[e["to"]].add(e["from"])

        # ── Domain taxonomy ────────────────────────────────────────────────────
        # All distinct `domain` values present in the graph, e.g. 'kinematics',
        # 'fluid_mechanics', 'electrostatics'. Used to (a) give Stage 1 a fixed
        # taxonomy to pick from rather than freeform text, and (b) filter
        # candidates_for_quantity() down to the domains relevant to a question.
        self.all_domains: set[str] = {
            n.get("domain") for n in self.nodes if n.get("domain")
        }

        print(
            f"[GraphIndex] Loaded {len(self.nodes)} equations, "
            f"{len(self.edges)} edges"
        )

    def _build_natural_output(self) -> dict[str, str]:
        """
        Parse each equation's sympy_expr to find its natural output symbol.
        Eq(v, u + a*t) → natural output of that eq is 'v'.
        """
        import re
        natural = {}
        pattern = re.compile(r"^Eq\(([A-Za-z_][A-Za-z0-9_]*)\s*,")
        for node in self.nodes:
            m = pattern.match(node.get("sympy_expr", ""))
            if m:
                natural[node["id"]] = m.group(1)
        return natural

    # ── Core lookup ───────────────────────────────────────────────────────────

    def get_equation(self, eq_id: str) -> dict | None:
        return self.nodes_by_id.get(eq_id)

    def equations_containing_symbol(self, sym: str) -> list[dict]:
        """Return all equation nodes that contain the given symbol."""
        return [
            self.nodes_by_id[eq_id]
            for eq_id in self.sym_to_eqs.get(sym, [])
            if eq_id in self.nodes_by_id
        ]

    # ── §5 Symbol-collision guardrail ─────────────────────────────────────────

    def candidates_for_quantity(
        self,
        needed_symbol: str,
        needed_name: str,
        needed_dimension: str,
        visited_eqs: set[str],
        allowed_domains: set[str] | None = None,
    ) -> list[dict]:
        """
        Generate candidate equations for a needed quantity.

        Filters out:
        1. Equations already in visited_eqs (already chosen or tried).
        2. Equations that contain NON_SOLVABLE_SYMBOLS (conservation-law forms
           like 'P*V^gamma = constant' that cannot be rearranged for a value).
        3. Equations where the variable matching needed_symbol has an
           INCOMPATIBLE dimension to needed_dimension. This is the deterministic
           symbol-collision guardrail: it prevents e.g. an optics 'm' (fringe
           order, dimensionless) from appearing as a candidate when we need
           mass 'm' (dimension M).

        Note: same-dimension-different-name cases (e.g. 'radius' vs 'separation
        distance', both dimension L) are NOT filtered here — they reach the LLM
        as legitimate candidates for conceptual disambiguation.

        `allowed_domains`, if given, narrows the result to equations whose
        `domain` field is in that set (e.g. {'laws_of_motion', 'kinematics'}
        for a dynamics problem) — purely to reduce how many candidates get
        sent to the LLM, not to change correctness. CRITICAL SAFETY PROPERTY:
        if narrowing by domain would leave ZERO candidates (the domain guess
        was wrong or incomplete), this returns the full dimension-filtered
        set instead, never an empty one. Nothing the LLM would conceptually
        need ever becomes permanently unreachable — domain filtering only
        changes what's shown by default, never what's reachable.

        Returns: list of equation nodes, unordered (LLM chooses among them).
        """
        candidates = []
        for eq_id in self.sym_to_eqs.get(needed_symbol, []):
            if eq_id in visited_eqs:
                continue
            node = self.nodes_by_id.get(eq_id)
            if node is None:
                continue
            # Skip conservation-law / non-solvable forms
            if set(node["variables"].keys()) & NON_SOLVABLE_SYMBOLS:
                continue
            # Dimension compatibility check
            var_meta = node["variables"].get(needed_symbol, {})
            stored_dim = var_meta.get("dimension", "")
            if not _dimensions_compatible(stored_dim, needed_dimension):
                continue
            candidates.append(node)

        if allowed_domains:
            narrowed = [c for c in candidates if c.get("domain") in allowed_domains]
            if narrowed:
                return narrowed
            # Fallback: domain guess didn't match anything for this quantity —
            # return the full set rather than silently excluding everything.
        return candidates

    # ── Legacy helpers ────────────────────────────────────────────────────────

    def expand_neighbors(self, seed_ids: list[str], hops: int = 1) -> list[dict]:
        """Return all equations reachable from seed_ids within `hops` edges."""
        visited  = set(seed_ids)
        frontier = set(seed_ids)
        for _ in range(hops):
            next_f = set()
            for node_id in frontier:
                for neighbor in self.adjacency.get(node_id, set()):
                    if neighbor not in visited:
                        next_f.add(neighbor)
            visited  |= next_f
            frontier  = next_f
        ordered = list(seed_ids) + [nid for nid in visited if nid not in seed_ids]
        return [self.nodes_by_id[nid] for nid in ordered if nid in self.nodes_by_id]
