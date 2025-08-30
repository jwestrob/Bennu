from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Claim(BaseModel):
    claim_id: str
    type: str  # observation|inference|recommendation
    text: str
    provenance: Dict[str, Any]
    metrics: Dict[str, float] = {}
    status: str = "tentative"  # tentative|confirmed|retracted


class Node(BaseModel):
    type: str
    data: Dict[str, Any] = {}
    children: List["Node"] = []


try:
    # Pydantic v1
    Node.update_forward_refs()
except Exception:
    # Pydantic v2 fallback name
    try:
        Node.model_rebuild()
    except Exception:
        pass


class Document(Node):
    type: str = "document"


class Section(Node):
    type: str = "section"  # data.id required


class Paragraph(Node):
    type: str = "paragraph"  # data.claims: List[Claim]


def new_document(meta: Dict[str, Any] | None = None) -> Document:
    meta = meta or {}
    return Document(type="document", data={"meta": meta}, children=[])


def find_section(doc: Document, anchor_id: str) -> Optional[Section]:
    for ch in doc.children:
        if isinstance(ch, Section) or (isinstance(ch, Node) and getattr(ch, "type", "") == "section"):
            sid = (ch.data or {}).get("id")
            if sid == anchor_id:
                # Re-validate as Section
                try:
                    # Prefer pydantic v2 path
                    try:
                        return Section.model_validate(ch.model_dump())  # type: ignore
                    except Exception:
                        return Section.model_validate(ch.dict())  # type: ignore
                except Exception:
                    # v1 fallback
                    try:
                        return Section.parse_obj(getattr(ch, 'dict', lambda: {})())  # type: ignore
                    except Exception:
                        # attempt to coerce
                        return Section(type="section", data=ch.data, children=ch.children)  # type: ignore
    return None


def _ensure_section(doc: Document, anchor_id: str, title: Optional[str] = None) -> Section:
    sec = find_section(doc, anchor_id)
    if sec is not None:
        return sec
    sec = Section(type="section", data={"id": anchor_id, "title": title or anchor_id}, children=[])
    # Append to document
    doc.children.append(sec)  # type: ignore
    return sec


def serialize_section(doc: Document, anchor_id: str) -> str:
    """Return a compact JSON string of the target section for prompting."""
    import json
    sec = find_section(doc, anchor_id)
    if sec is None:
        # Provide an empty skeleton with anchor id
        payload = {
            "type": "section",
            "data": {"id": anchor_id, "title": anchor_id},
            "children": [],
        }
        return json.dumps(payload, separators=(",", ":"))
    try:
        try:
            return json.dumps(sec.model_dump(), default=str, separators=(",", ":"))
        except Exception:
            return json.dumps(sec.dict(), default=str, separators=(",", ":"))
    except Exception:
        return "{}"


def append_claim_paragraph(doc: Document, anchor_id: str, claims: List[Claim]) -> None:
    sec = _ensure_section(doc, anchor_id)
    para = Paragraph(type="paragraph", data={"claims": [c.dict() for c in claims]}, children=[])
    sec.children.append(para)  # type: ignore


def to_markdown(doc: Document) -> str:
    lines: List[str] = []
    lines.append("# Report")
    # Title
    meta = (doc.data or {}).get("meta", {}) if isinstance(doc.data, dict) else {}
    if meta.get("title"):
        lines[0] = f"# {meta.get('title')}"
    for ch in doc.children:
        if getattr(ch, "type", "") == "section":
            raw_title = (ch.data or {}).get("title") or (ch.data or {}).get("id") or "Section"
            # Make anchor IDs human-friendly: sec:<ns>:<name> → Ns: Name
            title = raw_title
            try:
                if isinstance(raw_title, str) and raw_title.startswith("sec:"):
                    parts = raw_title.split(":", 2)
                    if len(parts) == 3:
                        ns = parts[1].strip()
                        name = parts[2].strip()
                        ns_disp = ns.capitalize()
                        name_disp = name.replace("_", " ")
                        title = f"{ns_disp}: {name_disp}"
                    elif len(parts) == 2:
                        title = parts[1].strip()
            except Exception:
                title = raw_title
            lines.append(f"\n## {title}")
            # Paragraphs
            for ph in ch.children or []:
                if getattr(ph, "type", "") == "paragraph":
                    pdata = ph.data or {}
                    claims = pdata.get("claims") or []
                    def _format_examples(ex):
                        try:
                            if isinstance(ex, list):
                                return ", ".join([str(x) for x in ex[:5]])
                            return str(ex)
                        except Exception:
                            return str(ex)
                    if claims and isinstance(claims, list):
                        for c in claims:
                            # Try strict Claim, then permissive dict rendering
                            try:
                                try:
                                    cl = Claim.model_validate(c)
                                except Exception:
                                    cl = Claim.parse_obj(c)
                                msg = f"- {cl.type.capitalize()}: {cl.text}"
                                # Optional metrics/provenance pretty-print
                                try:
                                    cnt = None
                                    if isinstance(cl.metrics, dict):
                                        cnt = cl.metrics.get('count') or cl.metrics.get('rows')
                                    if cnt is not None:
                                        msg += f" (count: {cnt})"
                                except Exception:
                                    pass
                                try:
                                    prov = cl.provenance or {}
                                    # Flatten a few IDs for readability (show up to 5)
                                    ids = []
                                    for k in ("neo4j", "sql", "lancedb"):
                                        v = prov.get(k)
                                        if isinstance(v, list):
                                            ids.extend([str(x) for x in v[:5]])
                                    if ids:
                                        msg += f" [examples: {', '.join(ids)}]"
                                except Exception:
                                    pass
                                lines.append(msg)
                                continue
                            except Exception:
                                # Fallback formatting for dict-like claims
                                try:
                                    if isinstance(c, dict):
                                        txt = c.get('text') or c.get('message') or str(c)
                                        cnt = c.get('count') or c.get('rows')
                                        prov = c.get('provenance') or c.get('examples') or []
                                        msg = f"- {txt}"
                                        if cnt is not None:
                                            msg += f" (count: {cnt})"
                                        if prov:
                                            msg += f" [examples: {_format_examples(prov)}]"
                                        lines.append(msg)
                                    else:
                                        lines.append(f"- {str(c)}")
                                except Exception:
                                    lines.append(f"- {str(c)}")
                    # If no claims, render paragraph text if present
                    elif isinstance(pdata, dict) and pdata.get('text'):
                        lines.append(f"- {pdata.get('text')}")
    return "\n".join(lines) + "\n"
