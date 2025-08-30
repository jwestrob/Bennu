from __future__ import annotations
from typing import Any, Dict, List, Tuple
from pathlib import Path
import os
import time

from .base import OperatorContext, OperatorSpec, register_operator
from ...options.template_runner import FileCypherRunner


# Simple file-backed catalog loaders with mtime-aware caching
_pfam_cache: Dict[str, Any] = {"mtime": 0.0, "rows": []}
_ko_cache: Dict[str, Any] = {"mtime": 0.0, "rows": []}


def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    for ch in "[]()/:,;._-\t\n\r":
        s = s.replace(ch, " ")
    return " ".join(s.split())


def _tokenize(s: str) -> List[str]:
    s = _normalize_text(s)
    toks = [t for t in s.split() if t]
    return toks


def _score(query: str, target: str) -> float:
    # Token overlap + substring boost + ID boost
    q = _normalize_text(query)
    t = _normalize_text(target)
    qt = set(q.split())
    tt = set(t.split())
    if not qt or not tt:
        return 0.0
    overlap = len(qt & tt) / max(1, len(qt))
    substr = 0.2 if q and (q in t) else 0.0
    idboost = 0.0
    if any(tok.startswith("k") and len(tok) == 6 and tok[1:].isdigit() for tok in qt):
        idboost += 0.2
    if any(tok.startswith("pf") and len(tok) == 7 and tok[2:].isdigit() for tok in qt):
        idboost += 0.2
    return overlap + substr + idboost


def _load_pfam_catalog(project_root: str | None = None) -> List[Tuple[str, str, str]]:
    """Return list of (pfam_id, short, desc)."""
    base = Path(project_root) if project_root else Path(os.getcwd())
    path = base / "data/reference/pfam_id_desc.tsv"
    try:
        mtime = path.stat().st_mtime
    except Exception:
        return []
    if _pfam_cache["mtime"] == mtime and _pfam_cache["rows"]:
        return _pfam_cache["rows"]
    rows: List[Tuple[str, str, str]] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if not parts:
                    continue
                pfid = (parts[0] if len(parts) > 0 else "").strip()
                short = (parts[1] if len(parts) > 1 else "").strip()
                desc = (parts[2] if len(parts) > 2 else "").strip()
                if pfid:
                    rows.append((pfid, short, desc))
    except Exception:
        rows = []
    _pfam_cache["mtime"] = mtime
    _pfam_cache["rows"] = rows
    return rows


def _load_ko_catalog(project_root: str | None = None) -> List[Tuple[str, str]]:
    """Return list of (ko_id, label) using simplified_definition when available."""
    base = Path(project_root) if project_root else Path(os.getcwd())
    path = base / "data/reference/ko_list"
    try:
        mtime = path.stat().st_mtime
    except Exception:
        return []
    if _ko_cache["mtime"] == mtime and _ko_cache["rows"]:
        return _ko_cache["rows"]
    rows: List[Tuple[str, str]] = []
    try:
        import csv
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except Exception:
                dialect = None
            if dialect:
                reader = csv.DictReader(f, dialect=dialect)
                cols = {k.lower(): k for k in (reader.fieldnames or [])}
                for row in reader:
                    ko = (row.get(cols.get("knum", ""), "") or row.get(cols.get("ko", ""), "")).strip()
                    sdef = (row.get(cols.get("simplified_definition", ""), "")).strip()
                    defin = (row.get(cols.get("definition", ""), "")).strip()
                    label = sdef or defin
                    if ko and label:
                        rows.append((ko, label))
            else:
                # Fallback: tab/space-delimited lines
                for i, raw in enumerate(f):
                    if i == 0 and ("knum" in raw.lower() and "\t" in raw):
                        continue
                    raw = raw.strip()
                    if not raw:
                        continue
                    parts = raw.split("\t") if "\t" in raw else raw.split(None, 1)
                    ko = parts[0].strip() if parts else ""
                    label = parts[1].strip() if len(parts) > 1 else ""
                    if ko and label and (ko.startswith("K") or ko.lower().startswith("ko:")):
                        rows.append((ko, label))
    except Exception:
        rows = []
    _ko_cache["mtime"] = mtime
    _ko_cache["rows"] = rows
    return rows


def _search_pfam(q: str, project_root: str | None, top_n: int) -> List[Dict[str, Any]]:
    rows = _load_pfam_catalog(project_root)
    if not q or not rows:
        return []
    scored = []
    for pfid, short, desc in rows:
        text = f"{pfid} {short} {desc}"
        s = _score(q, text)
        if s > 0:
            label = (short or desc or pfid).strip()
            scored.append({"pfam_id": pfid, "short": short, "desc": desc, "label": label, "score": round(s, 4)})
    scored.sort(key=lambda x: (-x["score"], x["pfam_id"]))
    return scored[: max(1, int(top_n))]


def _search_ko(q: str, project_root: str | None, top_n: int) -> List[Dict[str, Any]]:
    rows = _load_ko_catalog(project_root)
    if not q or not rows:
        return []
    scored = []
    for ko, label in rows:
        s = _score(q, f"{ko} {label}")
        if s > 0:
            scored.append({"ko_id": ko, "label": label, "score": round(s, 4)})
    scored.sort(key=lambda x: (-x["score"], x["ko_id"]))
    return scored[: max(1, int(top_n))]


def _search_pfam_catalog(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    q = str(params.get("q") or params.get("keyword") or "").strip()
    top_n = int(params.get("top_n", 25) or 25)
    hits = _search_pfam(q, ctx.project_root, top_n)
    # Provide flat lists for both accession IDs and short names to maximize match
    pfam_ids: List[str] = []
    pfam_terms: List[str] = []
    seen = set()
    for h in hits:
        acc = (h.get("pfam_id") or "").strip()
        short = (h.get("short") or "").strip()
        if acc and acc not in seen:
            seen.add(acc)
            pfam_ids.append(acc)
            pfam_terms.append(acc)
        if short and short not in seen:
            seen.add(short)
            pfam_terms.append(short)
    return {"pfam_catalog_hits": hits, "pfam_ids": pfam_ids, "pfam_terms": pfam_terms}


def _search_ko_catalog(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    q = str(params.get("q") or params.get("keyword") or "").strip()
    top_n = int(params.get("top_n", 25) or 25)
    hits = _search_ko(q, ctx.project_root, top_n)
    ko_ids = [h.get("ko_id") for h in hits if h.get("ko_id")]
    return {"ko_catalog_hits": hits, "ko_ids": ko_ids}


def _query_proteins_by_ids(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    runner = FileCypherRunner(ctx.neo4j_driver)
    pfam_in = params.get("pfam_ids") or inputs.get("pfam_ids") or inputs.get("pfam_terms") or params.get("pfam_terms")
    ko_in = params.get("ko_ids") or inputs.get("ko_ids")
    # Support passing a dict containing both lists (from ExtractIdsFromCatalogHits)
    if isinstance(pfam_in, dict):
        pfam_ids = []
        pfam_ids.extend(pfam_in.get("pfam_ids") or [])
        pfam_ids.extend(pfam_in.get("pfam_terms") or [])
    else:
        pfam_ids = pfam_in or []
    if isinstance(ko_in, dict) and "ko_ids" in ko_in:
        ko_ids = ko_in.get("ko_ids") or []
    else:
        ko_ids = ko_in or []
    genome_ids = params.get("genome_ids") or []
    try:
        limit = int(params.get("limit", 1000))
    except Exception:
        limit = 1000

    # Normalize PFAM tokens while preserving meaningful short names.
    # - Extract canonical accessions (PFxxxxx) when present
    # - Also keep lowercased short-name tokens like 'rubisco_large' to match d.id
    def _norm_tokens(tokens: List[Any]) -> List[str]:
        import re
        out: List[str] = []
        seen: set[str] = set()
        for tok in tokens:
            s = str(tok or '').strip()
            if not s:
                continue
            m = re.search(r"(PF\d{5})", s, re.IGNORECASE)
            if m:
                acc = m.group(1).upper()
                if acc not in seen:
                    seen.add(acc)
                    out.append(acc)
            else:
                # keep concise alphabetic/underscore names (avoid long descriptions)
                name = s.lower()
                if 2 <= len(name) <= 64 and all(ch.isalnum() or ch in {'_','-'} for ch in name):
                    if name not in seen:
                        seen.add(name)
                        out.append(name)
        return out
    if pfam_ids:
        pfam_ids = _norm_tokens(pfam_ids)

    pf_rows = []
    ko_rows = []
    if pfam_ids:
        pf_rows = runner.run_template(
            "proteins_by_pfam_ids.cypher",
            {"pfam_ids": pfam_ids, "genome_ids": genome_ids, "limit": limit},
        ) or []
    if ko_ids:
        ko_rows = runner.run_template(
            "proteins_by_ko_ids.cypher",
            {"ko_ids": ko_ids, "genome_ids": genome_ids, "limit": limit},
        ) or []

    # Merge to discovered_proteins shape: one row per (genome, protein) with pfams/kos arrays
    merged: Dict[str, Dict[str, Any]] = {}
    for r in pf_rows:
        gid = str(r.get("genome_id"))
        pid = str(r.get("protein_id"))
        key = f"{gid}\t{pid}"
        e = merged.setdefault(key, {"genome_id": gid, "protein_id": pid, "pfams": [], "kos": []})
        pf = r.get("pfam_id") or r.get("domain_id")
        if pf and pf not in e["pfams"]:
            e["pfams"].append(pf)
    for r in ko_rows:
        gid = str(r.get("genome_id"))
        pid = str(r.get("protein_id"))
        key = f"{gid}\t{pid}"
        e = merged.setdefault(key, {"genome_id": gid, "protein_id": pid, "pfams": [], "kos": []})
        ko = r.get("ko_id")
        if ko and ko not in e["kos"]:
            e["kos"].append(ko)

    out = list(merged.values())
    out.sort(key=lambda x: (x.get("genome_id", ""), x.get("protein_id", "")))
    return {"discovered_proteins": out}


# Register operators
register_operator(OperatorSpec(
    name="SearchPfamCatalogFuzzy",
    inputs=[],
    outputs=["pfam_catalog_hits", "pfam_ids", "pfam_terms"],
    params={"q": "string", "top_n": "int (default 25)"},
    run=_search_pfam_catalog,
    description="Fuzzy search PFAM catalog (local TSV) and return top matches with pfam_ids and pfam_terms (accessions + short names)",
))

register_operator(OperatorSpec(
    name="SearchKoCatalogFuzzy",
    inputs=[],
    outputs=["ko_catalog_hits", "ko_ids"],
    params={"q": "string", "top_n": "int (default 25)"},
    run=_search_ko_catalog,
    description="Fuzzy search KO catalog (ko_list; uses simplified definition when available). Returns ko_ids.",
))

register_operator(OperatorSpec(
    name="QueryProteinsByIds",
    inputs=["pfam_ids", "ko_ids"],
    outputs=["discovered_proteins"],
    params={"pfam_ids": "List[str] | null", "ko_ids": "List[str] | null", "genome_ids": "List[str] | null", "limit": "int (default 1000)"},
    run=_query_proteins_by_ids,
    description="Query proteins by exact PFAM/KO IDs (IN filters)",
))

# Utility: extract id lists from catalog hits
def _extract_ids(ctx: OperatorContext, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    pf_hits = inputs.get("pfam_catalog_hits") or []
    ko_hits = inputs.get("ko_catalog_hits") or []
    pf_ids: List[str] = []
    pf_terms: List[str] = []
    ko_ids: List[str] = []
    try:
        for h in pf_hits:
            pf = (h.get("pfam_id") or h.get("id") or "").strip()
            if pf and pf not in pf_ids:
                pf_ids.append(pf)
            short = (h.get("short") or "").strip()
            if short and short not in pf_terms:
                pf_terms.append(short)
            if pf and pf not in pf_terms:
                pf_terms.append(pf)
    except Exception:
        pass
    try:
        for h in ko_hits:
            kid = (h.get("ko_id") or h.get("id") or "").strip()
            if kid and kid not in ko_ids:
                ko_ids.append(kid)
    except Exception:
        pass
    return {"pfam_ids": pf_ids, "pfam_terms": pf_terms, "ko_ids": ko_ids}

register_operator(OperatorSpec(
    name="ExtractIdsFromCatalogHits",
    inputs=[],  # inputs optional; will read pfam_catalog_hits/ko_catalog_hits if present
    outputs=["pfam_ids", "ko_ids"],
    params={},
    run=_extract_ids,
    description="Extract PFAM/KO id lists from catalog search hits (optional inputs)",
))
