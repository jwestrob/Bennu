from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from pathlib import Path
import yaml


@dataclass
class SignatureSpec:
    name: str
    window_default: int
    motifs: Dict[str, Any]
    clauses: List[Dict[str, Any]]
    exclusions: Dict[str, List[str]]
    # Dynamic anchors for domain set resolution
    domain_sets: Dict[str, Dict[str, Dict[str, List[str]]]]


class SignatureRegistry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._signatures: Dict[str, SignatureSpec] = {}
        self._load_defaults()

    def _load_yaml(self, path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"Signature registry file not found: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def _load_defaults(self) -> None:
        # Load built-in PROPHAGE spec
        spec = self._load_yaml(self.root / "config" / "signatures" / "prophage.yml")
        try:
            p = SignatureSpec(**spec)
        except Exception as e:
            raise RuntimeError(f"Invalid signature spec for PROPHAGE: {e}")
        self._signatures[p.name.upper()] = p

    def get(self, name: str) -> SignatureSpec:
        key = (name or "").upper()
        if key not in self._signatures:
            raise ValueError(f"Unknown signature: {name}")
        return self._signatures[key]

    def resolve_domain_set(self, db_runner, group: str, name: str, *, limit: int = 250) -> Tuple[List[str], List[str]]:
        """Resolve anchors for a given (group, name) into concrete PFAM/KO id lists using DB templates.

        Uses resources/cypher/pfam_ids_by_query.cypher and ko_ids_by_query.cypher.
        Returns (pfam_ids, ko_ids) as lowercase, deduplicated lists.
        """
        # Select signature spec that contains domain_sets (expect single active spec for now)
        # Use the first signature loaded (e.g., PROPHAGE)
        if not self._signatures:
            raise RuntimeError("No signatures loaded in registry")
        spec = next(iter(self._signatures.values()))
        group_map = (spec.domain_sets or {}).get(group, {})
        entry = group_map.get(name)
        if not entry:
            return [], []
        pfam_ids: List[str] = []
        ko_ids: List[str] = []
        # PFAM anchors
        for q in (entry.get("pfam_query") or []):
            rows = db_runner.run_template("pfam_ids_by_query.cypher", {"q": q, "limit": int(limit)})
            for r in rows or []:
                rid = (r.get("pfam_id") or r.get("id") or "").lower()
                if rid:
                    pfam_ids.append(rid)
        # KO anchors
        for q in (entry.get("kegg_query") or []):
            rows = db_runner.run_template("ko_ids_by_query.cypher", {"q": q, "limit": int(limit)})
            for r in rows or []:
                kid = (r.get("ko_id") or "").lower()
                if kid:
                    ko_ids.append(kid)
        # Dedup while preserving order
        pfam_ids = list(dict.fromkeys(pfam_ids))
        ko_ids = list(dict.fromkeys(ko_ids))
        return pfam_ids, ko_ids

    def resolve_all(self, db_runner, *, limit: int = 250) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
        if not self._signatures:
            raise RuntimeError("No signatures loaded in registry")
        spec = next(iter(self._signatures.values()))
        out: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
        for group, names in (spec.domain_sets or {}).items():
            out[group] = {}
            for nm in names.keys():
                pf, kk = self.resolve_domain_set(db_runner, group, nm, limit=limit)
                out[group][nm] = {"pfam": pf, "kegg": kk}
        return out
