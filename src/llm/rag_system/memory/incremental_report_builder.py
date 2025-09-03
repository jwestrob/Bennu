from __future__ import annotations
from typing import List, Dict, Any, Tuple
import uuid
import os
from .doc_ast import Document, new_document, serialize_section, append_claim_paragraph, to_markdown, Claim
from ...lm_factory import make_lm
import json
from .patch_apply import apply_patch
from .patch_types import PatchEnvelope, JsonPatchOp
from .patch_validator import validate_patch
from .anchors import group_by_anchor
from .evidence_ledger import append_claim as ledger_append


class IncrementalReportBuilder:
    def __init__(self, note_keeper, model_allocator, tool_cache, config):
        self.nk = note_keeper
        self.alloc = model_allocator  # preserved for compatibility; unused
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

        # Phase 0 instrumentation: persist per-anchor debug summaries
        try:
            if self.nk and hasattr(self.nk, 'synthesis_notes_path'):
                sdir = self.nk.synthesis_notes_path
                os.makedirs(sdir, exist_ok=True)
                anchors_debug: Dict[str, Any] = {}
                for it in items:
                    a = it.get('anchor')
                    s = it.get('summary') or {}
                    if not a:
                        continue
                    anchors_debug[a] = {
                        'total_rows': s.get('total_rows', 0),
                        'unique_pfams': (s.get('unique_pfams') or [])[:10],
                        'unique_pfam_accessions': (s.get('unique_pfam_accessions') or [])[:10],
                        'unique_kos': (s.get('unique_kos') or [])[:10],
                        'examples': (s.get('examples') or [])[:5],
                    }
                with open(os.path.join(sdir, 'anchors_debug.json'), 'w', encoding='utf-8') as f_dbg:
                    json.dump(anchors_debug, f_dbg, indent=2, default=str)
        except Exception as _dbg_err:
            self._log_info(f"anchors_debug save skipped: {_dbg_err}")
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
                applied = 0
                for env_dict in envs:
                    try:
                        env = PatchEnvelope(**env_dict)
                        res = validate_patch(env, self.doc, neo4j=getattr(self.cfg, 'database', None), sql=None, lancedb=None, allow_nli=getattr(self.cfg, 'IRB_ALLOW_NLI', False))
                        if res.ok:
                            self.doc = self._apply_anchor_envelope(self.doc, env)
                            applied += 1
                    except Exception:
                        # Ignore malformed envelopes from multi-call
                        pass
                # Count one editor call for the multi-anchor attempt
                self.editor_calls += 1
                # If nothing applied from multi-call, fall back to single-anchor editor per item in this pack
                if applied == 0:
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
                # Done with this pack either way
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
        uniq_pf_acc = set()
        for r in rows:
            # Collect example IDs
            gid = r.get('genome_id')
            pid = r.get('protein_id')
            if gid and pid and (cap == 0 or len(examples) < cap):
                examples.append({"genome_id": gid, "protein_id": pid})
            # Collect light context
            pf = r.get('pfams') or []
            pf_ids = r.get('pfam_ids') or []
            ko = r.get('kos') or []
            if isinstance(pf, list):
                for x in pf:
                    if isinstance(x, str) and x:
                        uniq_pfams.add(x)
            if isinstance(pf_ids, list):
                for x in pf_ids:
                    if isinstance(x, str) and x:
                        xl = x.lower()
                        if xl.startswith('pf') and len(x) >= 7:
                            uniq_pf_acc.add(x[:7].upper())
            if isinstance(ko, list):
                for x in ko:
                    if isinstance(x, str) and x:
                        uniq_kos.add(x)
        return {
            "anchor": anchor,
            "total_rows": total,
            "example_count": len(examples),
            "examples": examples,
            "unique_pfams": sorted(list(uniq_pfams))[:25],
            "unique_kos": sorted(list(uniq_kos))[:25],
            # internal accessions for editor/hallmark logic (not printed)
            "unique_pfam_accessions": sorted(list(uniq_pf_acc))[:25],
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
            "     * Keep each claim to ONE sentence; mention batch_summary.total_rows.\n"
            "     * STRICTLY FORBIDDEN: Do NOT include any real identifiers (no protein IDs, contig names, locus strings).\n"
            "     * If helpful, refer ONLY to counts: use batch_summary.example_count to say 'N representative proteins'.\n"
            "     * Prefer high-signal aggregation (no raw lists).\n"
            "  2) Light analysis paragraph (no claims): value={type:'paragraph', data:{text:'...'}, children:[]}\n"
            "     * 1–2 sentences on possible functional role, notable co-occurrences/absences, or pathway cues, strictly based on batch_summary and anchor_snippet.\n"
            "     * If batch_summary.unique_pfams or unique_kos are present, reference a few IDs by name (not the full list).\n"
            "- Do NOT add bare headings or empty sections. No raw JSON dumps.\n"
            "- Return JSON only (the PatchEnvelope)."
        )
        try:
            # Direct minimal call for IRB editor; allow CLI override via config.irb_model
            import dspy
            from ..dspy_signatures import PatchProposalSignature
            override = getattr(self.cfg, 'irb_model', None)
            if override:
                lm = make_lm(override, step="irb")
                self._log_info(f"IRB_EDITOR(single): using override model='{override}'")
            else:
                # Default to cost-effective, non-reasoning model
                lm = make_lm("openai/gpt-4.1-mini", step="irb")
                self._log_info("IRB_EDITOR(single): using default model='openai/gpt-4.1-mini'")
            module = dspy.Predict(PatchProposalSignature)
            import time as _t
            _t0 = _t.time()
            with dspy.context(lm=lm):
                res = module(
                    anchor_snippet=anchor_snippet,
                    batch_summary=json.dumps(batch_summary),
                    open_obligations=", ".join(obligations) if obligations else "",
                    schema_reminder=json.dumps(schema),
                    editor_instructions=editor_instructions
                )
            self._log_info(f"IRB_EDITOR(single): call_ms={( _t.time()-_t0 )*1000:.0f}")
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
        """Multi-anchor editor: emit minimal JSON per anchor; build patches locally.

        Expected model output (JSON array):
        [
          {"anchor": "sec:module:rubisco", "claims": ["one-sentence claim", ...], "analysis": "1–2 sentences"},
          ...
        ]
        """
        try:
            import dspy, json, time as _t
            from ..dspy_signatures import MultiPatchProposalSignature

            # Prepare compact payload with a pre-filled skeleton per anchor
            # The model must fill ONLY the empty fields (claims/analysis), without reordering or dropping items.
            anchors_payload = []
            summary_by_anchor = {}
            for it in pack:
                # Sanitize batch_summary to avoid exposing real IDs to the model
                s = dict(it["summary"] or {})
                if "examples" in s:
                    # Remove explicit example IDs from the payload; keep count only
                    s["examples"] = []
                payload = {
                    "anchor": it["anchor"],
                    "batch_summary": s,
                    # Provide example_count explicitly (counts-only referencing allowed)
                    "example_count": int((it["summary"] or {}).get("example_count") or 0),
                    # Pre-filled skeleton fields to be completed by the model
                    "claims": [],
                    "analysis": "",
                }
                anchors_payload.append(payload)
                summary_by_anchor[it["anchor"]] = it["summary"]

            # Editor prompt: minimal JSON per anchor; no RFC6902 in the model output
            expected_count = len(anchors_payload)
            # Tight, skeleton-based instructions to reduce omissions and reordering
            editor_instructions = (
                "You are given a JSON ARRAY of skeleton objects. For EACH object, FILL ONLY: 'claims' and 'analysis'.\n"
                "Do NOT add/remove/reorder items. Do NOT modify 'anchor', 'batch_summary', or 'example_count'.\n"
                f"Return EXACTLY {expected_count} items, IN THE SAME ORDER. JSON ONLY (no prose, no code fences).\n"
                "Per anchor limits: at most ONE claim (<=20 words) OR a short analysis (<=20 words).\n"
                "STRICTLY FORBIDDEN: Do NOT include ANY real identifiers (no protein IDs, contig names, locus strings).\n"
                "If helpful, refer ONLY to counts: use 'example_count' to say e.g., 'N representative proteins'. If unsure, leave claims empty and set analysis='insufficient evidence'.\n"
                "Use ONLY 'batch_summary' for counts; do not fabricate or list raw rows."
            )

            # Model selection
            override = getattr(self.cfg, 'irb_model', None)
            if override:
                lm = make_lm(override, step="irb")
                self._log_info(f"IRB_EDITOR(multi): using override model='{override}' for {len(pack)} anchors")
            else:
                lm = make_lm("openai/gpt-4.1-mini", step="irb")
                self._log_info(f"IRB_EDITOR(multi): using default model='openai/gpt-4.1-mini' for {len(pack)} anchors")

            module = dspy.Predict(MultiPatchProposalSignature)

            _t0 = _t.time()
            with dspy.context(lm=lm):
                res = module(
                    anchors_json=json.dumps(anchors_payload),
                    schema_reminder="[]",  # unused in this mode
                    editor_instructions=editor_instructions,
                )
            self._log_info(f"IRB_EDITOR(multi): call_ms={( _t.time()-_t0 )*1000:.0f}")

            raw = getattr(res, 'patch_envelopes', None)
            # Parse model output to Python list of {anchor, claims, analysis}
            if isinstance(raw, (str, bytes)):
                try:
                    items = json.loads(raw)
                except Exception:
                    return None
            elif isinstance(raw, list):
                items = raw
            else:
                return None

            # Convert minimal JSON to RFC6902 PatchEnvelopes locally
            envelopes: List[Dict[str, Any]] = []
            for item in items:
                try:
                    anchor = str(item.get('anchor') or '').strip()
                    if not anchor:
                        continue
                    # Claims text lines
                    claims_txt = item.get('claims') or item.get('claim_lines') or []
                    if not isinstance(claims_txt, list):
                        claims_txt = [str(claims_txt)] if claims_txt else []
                    claims_objs = []
                    for ct in claims_txt[:3]:
                        try:
                            claims_objs.append(Claim(claim_id=str(uuid.uuid4()), type="observation", text=str(ct), provenance={"neo4j": [], "sql": [], "lancedb": []}, metrics={}))
                        except Exception:
                            continue
                    analysis_text = item.get('analysis') or item.get('text') or ''

                    # Build patch ops for this anchor
                    patch_ops = [
                        JsonPatchOp(op="test", path="/type", value="section"),
                        JsonPatchOp(op="test", path="/data/id", value=anchor),
                    ]
                    if claims_objs:
                        patch_ops.append(
                            JsonPatchOp(op="add", path="/children/-", value={
                                "type": "paragraph",
                                "data": {"claims": [c.dict() for c in claims_objs]},
                                "children": []
                            })
                        )
                    if analysis_text:
                        patch_ops.append(
                            JsonPatchOp(op="add", path="/children/-", value={
                                "type": "paragraph",
                                "data": {"text": str(analysis_text)},
                                "children": []
                            })
                        )
                    env = {
                        "anchor": anchor,
                        "obligations": obligations,
                        "patch": [op.__dict__ for op in patch_ops],
                        "evidence": {"neo4j": [], "sql": [], "lancedb": []},
                        "rationale": "local_patch_from_minimal_json",
                        "risk": "low",
                    }
                    envelopes.append(env)
                except Exception:
                    continue

            return envelopes if envelopes else None
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

        # Counts-only phrasing; do not expose real IDs in the report
        example_count = int(summary.get('example_count', 0) or 0)
        if example_count > 0:
            claim_text = (
                f"Detected {total} proteins contributing to this marker; "
                f"examples available ({example_count})."
            )
        else:
            claim_text = f"Detected {total} proteins contributing to this marker."

        claim = Claim(
            claim_id=str(uuid.uuid4()),
            type="observation",
            text=claim_text,
            provenance={"neo4j": [], "sql": [], "lancedb": []},
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
