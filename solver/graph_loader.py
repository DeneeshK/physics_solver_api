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
import re
from collections import defaultdict
from config import MAIN_GRAPH_PATH, NON_SOLVABLE_SYMBOLS


def load_graphs():
    """Returns a GraphIndex built from the main graph file."""
    with open(MAIN_GRAPH_PATH, encoding="utf-8") as f:
        main = json.load(f)
    return GraphIndex(main)


# ── Dimensional tokenizer ─────────────────────────────────────────────────────
# v7 fix: v6's regex was `[MLTAK](-?\d*)` — it only handled single-letter
# dimensions M, L, T, A, K. The graph actually uses multi-character dimension
# tokens too:
#   - 'Theta' for temperature (alternative to K, used in some equations)
#   - 'N' for amount of substance / moles (e.g. gas constant ML2T-2N-1Theta-1)
#   - the literal token 'varies' as a sentinel on conservation-law equations
# Under the old regex 'Theta' parsed as T*T*A (silently), 'N' was dropped, and
# 'varies' decayed to ('A', 1) — meaning a temperature dimension could
# accidentally match a current dimension. Both sides usually broke the same
# way so cross-comparison happened to work most of the time, but the bug was
# latent and would surface the moment one side adopted a different notation.
#
# The fix tokenizes by name first (longest-match wins), then exponent. The
# regex is anchored on word boundaries within the cleaned string.
DIMENSION_TOKEN_ORDER = ('Theta', 'mol', 'M', 'L', 'T', 'A', 'K', 'N')
# 'mol' is accepted as an alias for N (some equations write 'mol-1' directly).
# 'Theta' takes precedence over 'T' (longest-match) so 'Theta' doesn't get
# eaten as 'T'+'heta'.
DIMENSION_PATTERN = re.compile(
    r'(' + '|'.join(DIMENSION_TOKEN_ORDER) + r')(-?\d*)',
    re.IGNORECASE,
)


def _normalize_dimension(dim: str) -> tuple:
    """
    Parses a dimensional formula into a canonical, order-independent,
    format-independent representation: a sorted tuple of (token, exponent)
    pairs with nonzero exponents only.

    Recognized tokens (case-insensitive):
        M=mass, L=length, T=time, A=current, K=temperature,
        Theta=temperature (alternative to K), N or mol=amount of substance.

    The 'varies' sentinel returns an explicit marker that never matches any
    real physical dimension — used on conservation-law equations whose
    'constant' variable has no fixed dimension.

    Format-independent: "ML2T-3", "M L^2 T^-3", "ml2t-3", "M*L2*T-3" all
    normalize to (('L', 2), ('M', 1), ('T', -3)).
    """
    if not dim:
        return ()
    # v7.1.11: fold Unicode superscript digits and the Unicode minus sign to
    # ASCII before parsing. The LLM (especially the 7B) emits dimensions
    # inconsistently — sometimes "MLT-2" (ASCII), sometimes "MLT⁻²" (Unicode
    # superscripts). The graph stores ASCII. Without this fold, "MLT⁻²" fails
    # to match "MLT-2": the regex doesn't recognize ⁻² as an exponent, so it
    # parses as MLT (all exponent 1) and the dimension-compatibility check
    # wrongly rejects the equation. This silently dropped correct candidates
    # (e.g. Newton's second law for a force target) from Stage 2. The fold is
    # generic — it fixes dimension matching for every symbol and equation,
    # not any specific case.
    _SUPERSCRIPT_MAP = str.maketrans({
        "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
        "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
        "⁺": "+", "⁻": "-",          # Unicode superscript plus/minus
        "−": "-",                     # Unicode MINUS SIGN (U+2212) → ASCII hyphen
    })
    dim = dim.translate(_SUPERSCRIPT_MAP)
    cleaned = dim.replace("^", "").replace("*", "").replace(" ", "")
    if cleaned.lower() == 'varies':
        # Sentinel: never compatible with any real dimension. Use a marker
        # exponent on a synthetic token so set-intersection comparison
        # cleanly fails.
        return (('__VARIES__', 1),)
    # Normalize Theta and K both to the K bucket so they're interchangeable;
    # normalize mol and N both to the N bucket. The user side of the graph
    # uses either spelling; we want them to match. Canonical keys are
    # title-cased multi-char or single-letter uppercase.
    canonical = {'Theta': 'K', 'Mol': 'N'}
    exponents: dict[str, int] = {}
    pos = 0
    for m in DIMENSION_PATTERN.finditer(cleaned):
        # Refuse overlapping or out-of-order matches — if the regex skipped
        # over unrecognized characters, that's a malformed dimension; surface
        # as empty rather than silently mis-parse.
        if m.start() != pos:
            # Unknown chars between matches (e.g. an unrecognized token).
            # Skip them but record we did — strict parsing would refuse.
            pass
        token = m.group(1)
        # Canonicalize case for single-letter, preserve multi-char
        if len(token) == 1:
            token = token.upper()
        else:
            token = token[0].upper() + token[1:].lower()  # 'theta' → 'Theta', 'mol' → 'Mol'
        token = canonical.get(token, token)
        exp_str = m.group(2)
        if exp_str in ("", "-"):
            exp = 1
        else:
            exp = int(exp_str)
        exponents[token] = exponents.get(token, 0) + exp
        pos = m.end()
    return tuple(sorted((k, v) for k, v in exponents.items() if v != 0))


_UNIT_SUPERSCRIPT = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5",
    "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-", "−": "-", "·": "*",
})


_GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "Δ": "Delta",
    "ε": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta", "ϑ": "theta",
    "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi",
    "π": "pi", "ρ": "rho", "σ": "sigma", "ς": "sigma", "τ": "tau",
    "φ": "phi", "ϕ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
    "Ω": "Omega", "Φ": "Phi", "Σ": "Sigma", "Λ": "Lambda", "Γ": "Gamma",
    "Θ": "Theta", "Π": "Pi", "Ψ": "Psi",
}


def _greek_to_ascii(s: str) -> str:
    """Replace unicode Greek letters with their spelled-out ASCII names so a
    Stage-1 'σ'/'λ'/'ω' matches the registry's 'sigma'/'lambda'/'omega'."""
    if not s:
        return s
    if any(ch in _GREEK for ch in s):
        return "".join(_GREEK.get(ch, ch) for ch in s)
    return s


# Generic words that carry no disambiguating power in a quantity name. Two
# quantities sharing only these (e.g. "kinetic energy" vs "potential energy",
# both have "energy") are NOT the same concept — distinctive tokens must match.
_CONCEPT_STOP = {
    "of", "the", "a", "an", "in", "on", "at", "to", "and", "or", "per", "due",
    "from", "for", "with", "by", "its", "out", "into",
    "energy", "force", "velocity", "speed", "time", "number", "total", "net",
    "constant", "value", "quantity", "amount", "rate", "change",
}


def _concept_tokens(name: str) -> set:
    """Distinctive lowercase word tokens of a quantity name (length>2, minus
    generic stopwords). Used to decide whether two names denote the same
    physical concept."""
    if not name:
        return set()
    return {t for t in re.findall(r"[a-z]+", name.lower())
            if len(t) > 2 and t not in _CONCEPT_STOP}


def _normalize_unit(u: str) -> str:
    """Loosely normalize a unit string so Stage-1 spellings compare equal to the
    registry's. Folds unicode superscripts/middots, drops '^' and spaces, so
    'm/s²' == 'm/s^2', 'kg·m^2' == 'kg*m^2'. Case is preserved ('C' coulomb is
    NOT 'c'). Returns "" for falsy input."""
    if not u:
        return ""
    u = u.translate(_UNIT_SUPERSCRIPT)
    return u.replace("^", "").replace(" ", "")


def _dimensions_compatible(stored_dim: str, needed_dim: str) -> bool:
    """
    True if needed_dim is compatible with stored_dim.
    Handles ambiguous stored dimensions like 'MLT-1 or ML2 or A' by
    checking if needed_dim matches ANY of the alternatives.
    If either is empty/unknown, returns True (can't filter).
    Comparison is format-independent (see _normalize_dimension) — "ML2T-3",
    "M L^2 T^-3", and "ml2t-3" are all treated as the same dimension.
    """
    if not stored_dim or not needed_dim:
        return True
    stored_parts  = {_normalize_dimension(p) for p in stored_dim.split(" or ")}
    needed_parts  = {_normalize_dimension(p) for p in needed_dim.split(" or ")}
    return bool(stored_parts & needed_parts)  # non-empty intersection


class GraphIndex:
    def __init__(self, main_graph: dict):
        # ── Equation nodes ────────────────────────────────────────────────────
        self.nodes: list[dict] = main_graph["nodes"]
        self.nodes_by_id: dict[str, dict] = {n["id"]: n for n in self.nodes}

        # ── Graph edges (kept for legacy neighbor-expansion if needed) ────────
        self.edges: list[dict] = main_graph["edges"]

        # ── Canonical variable registry (v8 graphs) ───────────────────────────
        # The compiled v8 graph ships a `variables` registry and an `aliases`
        # map (Stage-1 notation → canonical symbol). When present, this is the
        # single source of truth for symbol canonicalization, replacing the
        # hand-maintained SYMBOL_ALIASES in config. Older graphs omit these keys
        # → the maps are empty and the pipeline falls back to config aliases, so
        # this is fully backward-compatible.
        self.variables: dict[str, dict] = main_graph.get("variables", {})
        self.aliases: dict[str, str] = main_graph.get("aliases", {})
        # Build dimension-aware alias resolution: an alias may map to different
        # canonicals by dimension (e.g. 'lambda' → wavelength [L] vs decay
        # constant [T-1]). Index alias → [(dimension, canonical)].
        self._alias_dim: dict[str, list] = {}
        for sym, meta in self.variables.items():
            for al in meta.get("aliases", []):
                self._alias_dim.setdefault(al, []).append((meta.get("dimension", ""), sym))

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

    # ── Symbol canonicalization (registry-driven) ─────────────────────────────

    def canonical(self, sym: str, dim: str = "", unit: str = "") -> str:
        """Map a Stage-1 symbol to the graph's canonical symbol for the SAME
        physical quantity.

        Some conventional symbols name two different quantities — 'T' is BOTH
        temperature and period, 'Q' is heat and charge, 'P' is pressure and
        power, 'V' is voltage and volume, 'I' is current and moment of inertia,
        'E' is energy and electric field. These are registered with the second
        meaning as an alias of the appropriate canonical, and disambiguated here
        by the quantity Stage 1 reported. UNIT is tried first because Stage 1's
        unit ('s', 'C', 'W') is far more reliable than its dimension field
        (which is often a unit in disguise, e.g. 'A*s' or 'IT' for charge);
        dimension is the fallback.

        Returns `sym` unchanged if already canonical with no conflicting
        candidate, or if unknown. Backward-compatible: graphs without a registry
        have empty maps, so the pipeline's config fallback stays in charge.
        """
        if not sym:
            return sym
        # Stage 1 often emits unicode Greek (σ, λ, μ, ω, ρ, θ, Δ…) where the
        # registry uses the spelled-out ASCII name (sigma, lambda, mu…). Fold to
        # ASCII first so the alias lookup can match.
        sym = _greek_to_ascii(sym)
        # Gather candidate canonicals: the symbol itself (if canonical) plus any
        # canonical it is an alias of. The canonical-self is listed first so a
        # no-hint call keeps the symbol's primary meaning.
        cands: list[str] = []
        if sym in self.variables:
            cands.append(sym)
        for _d, canon in self._alias_dim.get(sym, []):
            if canon not in cands:
                cands.append(canon)
        if not cands:
            return self.aliases.get(sym, sym)
        if len(cands) == 1:
            return cands[0]
        # Disambiguate by unit, then dimension.
        if unit:
            want_u = _normalize_unit(unit)
            for c in cands:
                if _normalize_unit(self.variables[c].get("unit", "")) == want_u:
                    return c
        if dim:
            want_d = _normalize_dimension(dim)
            for c in cands:
                if _normalize_dimension(self.variables[c].get("dimension", "")) == want_d:
                    return c
        return cands[0]

    def has_registry(self) -> bool:
        """True when this graph ships a canonical variable registry (v8+)."""
        return bool(self.variables)

    def concept_symbol(self, sym: str, name: str = "", unit: str = "",
                       dim: str = "") -> str:
        """Resolve a Stage-1 quantity to its canonical symbol by CONCEPT, not
        just by alias. Two tiers:

          1. alias/unit canonicalization (canonical()) — the fast path that
             handles known notations (mu_k→mu, T+s→T_p, Q+C→q).
          2. NAME matching against the registry — the general fallback. Stage 1
             labels every quantity with a `name` ("amplitude", "specific heat
             capacity of water", "slit separation"); we match that name to the
             registry variable whose name shares the most DISTINCTIVE tokens and
             whose dimension/unit is compatible. This is what lets a given the
             model called `A` (amplitude, unit m) bind to `A_amp` even though
             `A` is canonically area, or `c_water` bind to `c_sp`.

        Returns the canonical symbol, or the alias result if no confident name
        match exists. No registry → returns `sym` (config fallback elsewhere)."""
        base = self.canonical(sym, dim, unit)
        if not self.variables:
            return base
        want = _concept_tokens(name)
        if not want:
            return base
        wu = _normalize_unit(unit)
        wd = _normalize_dimension(dim)
        # Score every registry variable by distinctive-name overlap, restricted
        # to dimensionally/unit-compatible ones. The current `base` is included
        # as a candidate so a correct alias result still wins ties.
        best_sym, best_score = base, 0
        if base in self.variables:
            best_score = len(want & _concept_tokens(self.variables[base].get("name", "")))
        for csym, meta in self.variables.items():
            mu = _normalize_unit(meta.get("unit", ""))
            md = _normalize_dimension(meta.get("dimension", ""))
            # require compatible unit OR dimension (Stage-1 unit is reliable;
            # dimension is the looser backstop).
            unit_ok = bool(wu) and bool(mu) and wu == mu
            dim_ok = bool(wd) and bool(md) and wd == md
            if not (unit_ok or dim_ok):
                continue
            score = len(want & _concept_tokens(meta.get("name", "")))
            # strict improvement only, so `base` keeps ties (prevents drift
            # between equivalent-scoring quantities like m vs m1).
            if score > best_score:
                best_sym, best_score = csym, score
        return best_sym if best_score >= 1 else base

    def equations_containing_symbol(self, sym: str) -> list[dict]:
        """Return all equation nodes that contain the given symbol."""
        return [
            self.nodes_by_id[eq_id]
            for eq_id in self.sym_to_eqs.get(sym, [])
            if eq_id in self.nodes_by_id
        ]

    # ── Neighbor-walk (v7.2 traversal core) ────────────────────────────────────
    # The graph stores equation→equation edges with `shared_variables`. To chase
    # a variable mid-chain (e.g. need `m` after choosing F=ma), we walk to the
    # neighbors of the equations already chosen that SHARE that variable. This
    # replaces the flat sym_to_eqs symbol-gate for later hops: instead of "every
    # equation literally containing m", we get "the equations connected to what
    # we've already chosen, through m" — a small, relevant neighborhood the LLM
    # judges by meaning. No dimension rejection; no symbol-presence gate.

    def _edges_for(self, eq_id: str) -> list[dict]:
        """All edges incident to eq_id, normalized so 'other' is the neighbor
        and 'shared_variables' is present."""
        out = []
        for e in self.edges:
            if e.get("from") == eq_id:
                out.append({"other": e.get("to"),
                            "shared_variables": e.get("shared_variables", [])})
            elif e.get("to") == eq_id:
                out.append({"other": e.get("from"),
                            "shared_variables": e.get("shared_variables", [])})
        return out

    def neighbors_sharing_variable(
        self,
        *,
        from_eq_ids: set[str],
        variable: str,
        visited_eqs: set[str],
    ) -> list[dict]:
        """
        The equations connected to what we've already chosen THROUGH `variable`
        — i.e. the equations that also contain `variable`, excluding the
        sources and anything visited. Returns equation node dicts (unordered;
        the LLM chooses). Dimension/symbol are NOT used to reject.

        v7.2 design note — why this computes from variable membership, not the
        precomputed `edges`: the graph's edge list is an incomplete snapshot
        (e.g. F=ma and v²=u²+2as both contain `a` but have no edge between
        them). Computing neighbors directly from sym_to_eqs[variable] is the
        TRUE bipartite traversal — sym_to_eqs[variable] IS the variable-node's
        adjacency list (every equation touching that variable), so it can never
        miss a real connection. This realizes the "reach the variable's node
        and look at its neighbors" design exactly, and is robust to a stale or
        sparse edge list.

        The `from_eq_ids` argument is kept for interface symmetry and possible
        future edge-weighting, but reachability is by shared variable: any
        equation containing `variable` is reachable from any chosen equation
        that also contains it. Since the chosen equations are what introduced
        `variable` into the frontier, they contain it by construction.

        The design assumption (user-stated): a question's quantities are
        connected, so the equation that resolves `variable` is reachable this
        way. If nothing comes back, the caller rolls back to Round-0
        candidates — we never fall back to a global semantic search, which
        would only surface unconnected equations.
        """
        out = []
        seen: set[str] = set()
        for eq_id in self.sym_to_eqs.get(variable, []):
            if eq_id in from_eq_ids or eq_id in visited_eqs or eq_id in seen:
                continue
            node = self.nodes_by_id.get(eq_id)
            if node is None:
                continue
            # Skip pure conservation/non-solvable forms (same exclusion the
            # symbol route used) — these aren't usable to SOLVE for a value.
            if set(node["variables"].keys()) & NON_SOLVABLE_SYMBOLS:
                continue
            seen.add(eq_id)
            out.append(node)
        return out

    def all_equations_with_variable(
        self,
        *,
        variable: str,
        visited_eqs: set[str],
    ) -> list[dict]:
        """
        The widest LOCAL fallback when neighbor-walk yields nothing usable but
        before rolling back: every equation in the graph that contains
        `variable` (still not a semantic/global concept search — just the
        variable-membership set). Kept available for the resolver's fallback
        tier. No dimension rejection.
        """
        out = []
        for eq_id in self.sym_to_eqs.get(variable, []):
            if eq_id in visited_eqs:
                continue
            node = self.nodes_by_id.get(eq_id)
            if node is None:
                continue
            if set(node["variables"].keys()) & NON_SOLVABLE_SYMBOLS:
                continue
            out.append(node)
        return out

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
