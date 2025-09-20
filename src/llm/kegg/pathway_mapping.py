from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Dict, Set, Tuple, Iterable, Optional
import os


def _normalize_path_id(pid: str) -> str:
    if not isinstance(pid, str):
        return ""
    s = pid.strip()
    if ":" in s:
        s = s.split(":", 1)[-1]
    # keep only KEGG map IDs like map00010
    return s


@lru_cache(maxsize=1)
def load_ko_pathway_maps(ko_list_path: Optional[str] = None) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Load ko_pathway.list and build maps.

    Returns (pathway_to_kos, ko_to_pathways).
    Accepts optional KO_LIST_PATH override via arg or env; otherwise defaults to data/reference/ko_pathway.list.
    """
    # Resolve path
    candidate = (
        ko_list_path
        or os.getenv("KO_LIST_PATH")
        or str(Path(__file__).resolve().parents[3] / "data" / "reference" / "ko_pathway.list")
    )
    p = Path(candidate)
    if not p.exists():
        raise FileNotFoundError(f"ko_pathway.list not found at {p}")

    pathway_to_kos: Dict[str, Set[str]] = {}
    ko_to_pathways: Dict[str, Set[str]] = {}
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            try:
                ko_part, path_part = line.split("\t", 1)
                ko = ko_part.replace("ko:", "").strip()
                path_id = _normalize_path_id(path_part.strip())
                if not path_id.startswith("map"):
                    continue
            except Exception:
                continue
            pathway_to_kos.setdefault(path_id, set()).add(ko)
            ko_to_pathways.setdefault(ko, set()).add(path_id)
    return pathway_to_kos, ko_to_pathways


def filter_pathway_ids(all_path_ids: Iterable[str], requested: Optional[Iterable[str]]) -> Set[str]:
    if not requested:
        return set(all_path_ids)
    req_norm = {_normalize_path_id(x) for x in requested if isinstance(x, str) and x.strip()}
    return {pid for pid in all_path_ids if pid in req_norm}

