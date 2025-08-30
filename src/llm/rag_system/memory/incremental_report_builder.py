from __future__ import annotations
from typing import List, Dict, Any, Tuple
import uuid
import os
from .doc_ast import Document, new_document, serialize_section, append_claim_paragraph, to_markdown, Claim
import json
from .patch_apply import apply_patch
from .patch_types import PatchEnvelope, JsonPatchOp
from .patch_validator import validate_patch
from .anchors import group_by_anchor
from .evidence_ledger import append_claim as ledger_append


class IncrementalReportBuilder:
    def __init__(self, note_keeper, model_allocator, tool_cache, config):
        self.nk = note_keeper
        self.alloc = model_allocator
        self.cache = tool_cache
        self.cfg = config
        self.failed = False
        self.fail_reason = ""
        self.session_dir = getattr(self.nk, "session_path", None)
        self.state_path = os.path.join(str(self.session_dir), "irb_state.json") if self.session_dir else None
        self.ledger_path = os.path.join(str(self.session_dir), "evidence_ledger.jsonl") if self.session_dir else None
        self.doc = self._load_or_init_ast()
        self.editor_calls = 0
        # Code-level defaults (no env required)
        self.max_editor_calls = int(getattr(self.cfg, 'IRB_MAX_EDITOR_CALLS', 30) or 30)

    # Public API
    def run(self, raw_items: List[Dict[str, Any]], obligations: List[str] | None = None) -> Document:
        obligations = obligations or []
        macro_results = [it for it in raw_items if isinstance(it, dict) and it.get("type") == "macro_result"]
        followups = [it for it in raw_items if isinstance(it, dict) and it.get("type") == "followup_request"]
        # Flatten rows and track anchor by row
        batches = self._schedule(macro_results)
        # Build token-estimated items and pack into near-30k batches
        items: List[Dict[str, Any]] = []
        for anchor, rows in batches:
            snippet = serialize_section(self.doc, anchor)
            summary = self._compact_batch(anchor, rows)
            tokens = self._estimate_tokens(snippet, summary)
            items.append({
                'anchor': anchor,
                'rows': rows,
                'snippet': snippet,
                'summary': summary,
                'tokens': tokens,
            })
        packs = self._pack_items(items, limit=25000)
        self._log_info(f"IRB_PLAN: anchors={len(items)} packs={len(packs)} max_calls={self.max_editor_calls}")
        if not batches and followups:
            # Write a Next Steps section so the report isn't empty
            anchor = "sec:community:next_steps"
            claim_text = "Insufficient direct evidence in this pass. Proposed follow-up tasks are available to broaden the search (catalog → IDs → exact retrieval)."
            claim = Claim(claim_id=str(uuid.uuid4()), type="recommendation", text=claim_text, provenance={"neo4j": [], "sql": [], "lancedb": []})
            append_claim_paragraph(self.doc, anchor, [claim])
            return self.doc
        # Execute per pack to minimize API calls
        for pi, pack in enumerate(packs):
            if self._budget_exhausted():
                return self._bug_out("Call/token budget exhausted")
            tot_tokens = sum(it['tokens'] for it in pack)
            self._log_info(f"IRB_PROGRESS: pack {pi+1}/{len(packs)} anchors={len(pack)} tokens~{tot_tokens}")
            # Try multi-anchor editor call first
            envs = self._call_editor_multi(pack, obligations)
            if isinstance(envs, list) and envs:
                for env_dict in envs:
                    try:
                        env = PatchEnvelope(**env_dict)
                        res = validate_patch(env, self.doc, neo4j=getattr(self.cfg, 'database', None), sql=None, lancedb=None, allow_nli=getattr(self.cfg, 'IRB_ALLOW_NLI', False))
                        if res.ok:
                            self.doc = self._apply_anchor_envelope(self.doc, env)
                    except Exception:
                        # Ignore malformed envelopes from multi-call
                        pass
                self.editor_calls += 1
                continue
            # Fallback: single-anchor calls for each item in pack
            for it in pack:
                env = self._call_editor(it['anchor'], it['snippet'], it['summary'], obligations)
                res = validate_patch(env, self.doc, neo4j=getattr(self.cfg, 'database', None), sql=None, lancedb=None, allow_nli=getattr(self.cfg, 'IRB_ALLOW_NLI', False))
                if res.ok:
                    try:
                        self.doc = self._apply_anchor_envelope(self.doc, env)
                    except AssertionError as e:
                        return self._bug_out(f"Patch application failed: {e}")
                    self._record(env, it['rows'])
                else:
                    self._fallback_append(it['anchor'], it['summary'])
            # Count fallback path as one editor call for budgeting purposes
            self.editor_calls += 1
        return self.doc

    # Internal helpers
    def _load_or_init_ast(self) -> Document:
        # Minimal new document; title can be filled later
        return new_document({"title": "Genomic Analysis Report"})

    def _schedule(self, macro_results: List[Dict[str, Any]]) -> List[Tuple[str, List[dict]]]:
        # Union all rows across macro_results and group by anchor
        all_rows: List[dict] = []
        for mr in macro_results:
            rows = mr.get('rows') or []
            # annotate each row with capability for anchoring fallback
            cap = mr.get('name') or 'capability'
            for r in rows:
                if isinstance(r, dict):
                    r = dict(r)
                    r.setdefault('_capability', cap)
                all_rows.append(r)
        groups = group_by_anchor(all_rows)
        # Do not drop small anchors; retain all anchors to preserve rare but high-signal markers
        # Sort anchors by descending size
        items = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
        return items

    def _estimate_tokens(self, anchor_snippet: str, batch_summary: Dict[str, Any]) -> int:
        try:
            import tiktoken
            s = anchor_snippet + "\n" + json.dumps(batch_summary, separators=(",", ":"))
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(s))
        except Exception:
            return int((len(anchor_snippet) + len(str(batch_summary))) / 4)

    def _pack_items(self, items: List[Dict[str, Any]], limit: int = 28000) -> List[List[Dict[str, Any]]]:
        # Best-Fit Decreasing packing by estimated tokens
        items_sorted = sorted(items, key=lambda x: x['tokens'], reverse=True)
        packs: List[List[Dict[str, Any]]] = []
        loads: List[int] = []
        for it in items_sorted:
            best_idx = -1
            best_space = None
            for i, load in enumerate(loads):
                space = limit - load
                if it['tokens'] <= space:
                    rem = space - it['tokens']
                    if best_space is None or rem < best_space:
                        best_space = rem
                        best_idx = i
            if best_idx == -1:
                packs.append([it])
                loads.append(it['tokens'])
            else:
                packs[best_idx].append(it)
                loads[best_idx] += it['tokens']
        return packs

    def _compact_batch(self, anchor: str, rows: List[dict]) -> Dict[str, Any]:
        # True totals, representative example IDs, and light context (unique PFAMs/KOs)
        total = len(rows)
        examples = []
        cap = int(getattr(self.cfg, 'SUMMARY_EXAMPLE_CAP', 10) or 10)
        uniq_pfams, uniq_kos = set(), set()
        for r in rows:
            # Collect example IDs
            gid = r.get('genome_id')
            pid = r.get('protein_id')
            if gid and pid and (cap == 0 or len(examples) < cap):
                examples.append({"genome_id": gid, "protein_id": pid})
            # Collect light context
            pf = r.get('pfams') or []
            ko = r.get('kos') or []
            if isinstance(pf, list):
                for x in pf:
                    if isinstance(x, str) and x:
                        uniq_pfams.add(x)
            if isinstance(ko, list):
                for x in ko:
                    if isinstance(x, str) and x:
                        uniq_kos.add(x)
        return {
            "anchor": anchor,
            "total_rows": total,
            "examples": examples,
            "unique_pfams": sorted(list(uniq_pfams))[:25],
            "unique_kos": sorted(list(uniq_kos))[:25],
            "cache_refs": [],
        }

    def _call_editor(self, anchor: str, anchor_snippet: str, batch_summary: Dict[str, Any], obligations: List[str]) -> PatchEnvelope:
        # Try GPT-5 (low effort) via model allocator with PatchProposalSignature
        schema = {
            "type": "object",
            "required": ["anchor", "patch", "evidence", "rationale"],
            "properties": {
                "anchor": {"type": "string"},
                "obligations": {"type": "array", "items": {"type": "string"}},
                "patch": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["op", "path"],
                        "properties": {
                            "op": {"type": "string", "enum": ["add", "remove", "replace", "test"]},
                            "path": {"type": "string"},
                            "value": {}
                        }
                    }
                },
                "evidence": {"type": "object"},
                "rationale": {"type": "string"},
                "risk": {"type": "string", "enum": ["low", "medium", "high"]}
            }
        }
        editor_instructions = (
            "You are editing a single section only. Output a strict RFC 6902 PatchEnvelope.\n"
            "Constraints:\n"
            "- Edit ONLY the provided anchor section; do not modify other sections.\n"
            "- Include at least two test ops for the target section: test '/type'=='section' and test '/data/id'==anchor.\n"
            "- Add exactly TWO paragraphs (two add ops to '/children/-') in this order:\n"
            "  1) Evidence paragraph with CLAIMS: value={type:'paragraph', data:{claims:[Claim,...]}, children:[]}\n"
            "     * Claim schema: {claim_id, type:[observation|inference|recommendation], text, provenance:{neo4j:[], sql:[], lancedb:[]}, metrics:{}}\n"
            "     * Keep each claim to ONE sentence; mention batch_summary.total_rows and include representative example IDs (IDs only).\n"
            "     * Prefer high-signal aggregation (no raw lists).\n"
            "  2) Light analysis paragraph (no claims): value={type:'paragraph', data:{text:'...'}, children:[]}\n"
            "     * 1–2 sentences on possible functional role, notable co-occurrences/absences, or pathway cues, strictly based on batch_summary and anchor_snippet.\n"
            "     * If batch_summary.unique_pfams or unique_kos are present, reference a few IDs by name (not the full list).\n"
            "- Do NOT add bare headings or empty sections. No raw JSON dumps.\n"
            "- Return JSON only (the PatchEnvelope)."
        )
        try:
            # Direct minimal call for IRB editor only (use 4.1-mini for speed/stability)
            import dspy
            from ..dspy_signatures import PatchProposalSignature
            # Use LiteLLM-backed chat with param dropping; keep reasoning minimal
            lm = dspy.LM(
                model="openai/gpt-4.1-mini",
                model_type="chat",
                temperature=0.0,
                # Use LiteLLM param dropping to strip any token caps for GPT-5 responses API
                drop_params=True,
                additional_drop_params=[
                    "max_tokens",
                    "max_completion_tokens",
                    "max_output_tokens",
                ],
            )
            # Surgical removal in case DSPy remaps internally
            try:
                for k in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
                    lm.kwargs.pop(k, None)
            except Exception:
                pass
            module = dspy.Predict(PatchProposalSignature)
            with dspy.context(lm=lm):
                res = module(
                    anchor_snippet=anchor_snippet,
                    batch_summary=json.dumps(batch_summary),
                    open_obligations=", ".join(obligations) if obligations else "",
                    schema_reminder=json.dumps(schema),
                    editor_instructions=editor_instructions
                )
            env_raw = getattr(res, 'patch_envelope', None) if res else None
            if isinstance(env_raw, (str, bytes)):
                env_dict = json.loads(env_raw)
            elif isinstance(env_raw, dict):
                env_dict = env_raw
            else:
                env_dict = None
            if isinstance(env_dict, dict):
                return PatchEnvelope(**env_dict)
        except Exception:
            pass

        # Deterministic minimal patch fallback
        patch_ops = [
            # Anchor-relative test: ensure the target section node has correct type
            JsonPatchOp(op="test", path="/type", value="section"),
            JsonPatchOp(op="add", path="/children/-", value={
                "type": "section",
                "data": {"id": anchor, "title": anchor},
                "children": []
            }),
        ]
        return PatchEnvelope(
            anchor=anchor,
            obligations=obligations,
            patch=patch_ops,
            evidence={"neo4j": [], "sql": [], "lancedb": []},
            rationale="Append empty section for anchor (fallback)",
            risk="low",
        )

    # --- Anchor-aware patch application ---
    def _apply_anchor_envelope(self, doc: Document, env) -> Document:
        """Ensure anchor section exists, rewrite relative paths to absolute, then apply.

        - Creates the anchor section if missing (id=env.anchor)
        - Rewrites op.path like '/data/id' to '/children/{idx}/data/id'
        - Leaves root-level '/type' and any '/children/..' absolute paths unchanged
        """
        # Serialize to mutable dict
        try:
            base = json.loads(doc.json())
        except Exception:
            base = json.loads(doc.model_dump_json())

        # 1) Find or create anchor section and its index
        children = base.get('children') or []
        sec_idx = None
        for i, ch in enumerate(children):
            if isinstance(ch, dict) and ch.get('type') == 'section' and isinstance(ch.get('data'), dict) and ch['data'].get('id') == env.anchor:
                sec_idx = i
                break
        if sec_idx is None:
            # Create a new section
            section = {
                'type': 'section',
                'data': {'id': env.anchor, 'title': env.anchor},
                'children': []
            }
            children.append(section)
            base['children'] = children
            sec_idx = len(children) - 1

        # 2) Rewrite op paths relative to anchor
        rewritten = []
        def _coerce_value_to_node(val):
            # Ensure any child insertion is a proper Node with 'type'
            if isinstance(val, dict) and val.get('type'):
                return val
            return {
                'type': 'paragraph',
                'data': {'text': str(val)},
                'children': []
            }

        for op in env.patch:
            p = op.path or '/'
            # Always anchor-scope: ensure paths target /children/{sec_idx}/...
            if not p.startswith('/'):
                p = '/' + p
            prefix = f"/children/{sec_idx}/"
            if p.startswith(prefix):
                new_path = p
            else:
                new_path = f"/children/{sec_idx}{p}"
            payload = {'op': op.op, 'path': new_path}
            if op.value is not None:
                # If adding into children, coerce to valid node
                if op.op == 'add' and '/children' in new_path:
                    payload['value'] = _coerce_value_to_node(op.value)
                else:
                    payload['value'] = op.value
            rewritten.append(payload)

        # 3) Apply patch on rewritten JSON (enforces test ops)
        return apply_patch(Document.parse_obj(base), {'patch': rewritten})

    def _call_editor_multi(self, pack: List[Dict[str, Any]], obligations: List[str]):
        try:
            import dspy, json
            from ..dspy_signatures import MultiPatchProposalSignature
            anchors_payload = []
            for it in pack:
                anchors_payload.append({
                    "anchor": it["anchor"],
                    "anchor_snippet": it["snippet"],
                    "batch_summary": it["summary"],
                })
            schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["anchor", "patch", "evidence", "rationale"],
                    "properties": {
                        "anchor": {"type": "string"},
                        "obligations": {"type": "array", "items": {"type": "string"}},
                        "patch": {"type": "array", "minItems": 1, "items": {"type": "object"}},
                        "evidence": {"type": "object"},
                        "rationale": {"type": "string"},
                        "risk": {"type": "string"}
                    }
                }
            }
            editor_instructions = (
                "You are editing multiple sections. For EACH item, output ONE PatchEnvelope. Constraints:\n"
                "- Edit ONLY the provided anchor section for that item.\n"
                "- Include test ops for '/type'=='section' and '/data/id'==anchor per envelope.\n"
                "- Add exactly one paragraph with claims per section: path '/children/-', value {type:'paragraph', data:{claims:[...]} }.\n"
                "- Claims should include counts (batch_summary.total_rows) and representative example IDs; include provenance arrays (ids only).\n"
                "- JSON only. No commentary."
            )
            # Direct minimal call for multi-anchor proposal (switchable; default GPT-5 minimal for quality)
            lm = dspy.LM(
                model="openai/gpt-5-2025-08-07",
                model_type="responses",
                temperature=1.0,
                drop_params=True,
                additional_drop_params=[
                    "max_tokens",
                    "max_completion_tokens",
                    "max_output_tokens",
                ],
            )
            try:
                for k in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
                    lm.kwargs.pop(k, None)
            except Exception:
                pass
            module = dspy.Predict(MultiPatchProposalSignature)
            with dspy.context(lm=lm):
                res = module(
                    anchors_json=json.dumps(anchors_payload),
                    schema_reminder=json.dumps(schema),
                    editor_instructions=(
                        "You are editing multiple sections. For EACH item, output ONE PatchEnvelope.\n"
                        "Constraints:\n"
                        "- Edit ONLY that item's anchor section; include test ops for '/type'=='section' and '/data/id'==anchor per envelope.\n"
                        "- Add exactly TWO paragraphs per envelope (claims paragraph first, then a light analysis paragraph).\n"
                        "  * Claims paragraph uses Claim objects (one sentence each), mentions total_rows, and includes representative example IDs; no raw lists.\n"
                        "  * Analysis paragraph is concise (1–2 sentences), referencing unique_pfams/unique_kos when present for context.\n"
                        "- JSON only. Return an array of PatchEnvelopes."
                    )
                )
            envs_raw = getattr(res, 'patch_envelopes', None)
            if isinstance(envs_raw, (str, bytes)):
                return json.loads(envs_raw)
            if isinstance(envs_raw, list):
                return envs_raw
            return None
        except Exception:
            return None

    def _fallback_append(self, anchor: str, summary: Dict[str, Any]) -> None:
        # Non-LLM, deterministic append for resilience
        total = int(summary.get('total_rows', 0) or 0)
        ex = summary.get('examples') or []
        # Format up to 5 representative genome:protein pairs for readability
        rep_ids: list[str] = []
        try:
            for e in ex[:5]:
                if isinstance(e, dict):
                    gid = e.get('genome_id')
                    pid = e.get('protein_id')
                    if gid and pid:
                        rep_ids.append(f"{gid}:{pid}")
        except Exception:
            rep_ids = []

        # Message explicitly inlines a few example IDs when available
        if rep_ids:
            shown = "; ".join(rep_ids[:3])
            claim_text = (
                f"Detected {total} proteins contributing to this marker; "
                f"examples: {shown}."
            )
        else:
            claim_text = f"Detected {total} proteins contributing to this marker."

        claim = Claim(
            claim_id=str(uuid.uuid4()),
            type="observation",
            text=claim_text,
            provenance={"neo4j": rep_ids, "sql": [], "lancedb": []},
        )
        append_claim_paragraph(self.doc, anchor, [claim])
        # Optional lightweight context paragraph (non-claim) for readability
        try:
            uniq_pf = summary.get('unique_pfams') or []
            uniq_ko = summary.get('unique_kos') or []
            ctx_bits = []
            if uniq_ko:
                ctx_bits.append(f"KOs: {', '.join(uniq_ko[:5])}")
            if uniq_pf:
                ctx_bits.append(f"PFAMs: {', '.join(uniq_pf[:5])}")
            if ctx_bits:
                append_claim_paragraph(
                    self.doc,
                    anchor,
                    [Claim(claim_id=str(uuid.uuid4()), type="inference", text=f"Context markers — {'; '.join(ctx_bits)}.", provenance={"neo4j": [], "sql": [], "lancedb": []})]
                )
        except Exception:
            pass

    def _record(self, env: PatchEnvelope, rows: List[dict]) -> None:
        if self.ledger_path:
            ledger_append(self.ledger_path, {
                "anchor": env.anchor,
                "rationale": env.rationale,
                "rows": len(rows),
            })

    def _budget_exhausted(self) -> bool:
        return self.editor_calls >= getattr(self, 'max_editor_calls', 100)

    def _bug_out(self, reason: str) -> Document:
        self.failed = True
        self.fail_reason = reason
        return self.doc

    def to_markdown(self) -> str:
        return to_markdown(self.doc)

    def _log_info(self, msg: str) -> None:
        try:
            import logging
            logging.getLogger(__name__).info(msg)
        except Exception:
            pass
