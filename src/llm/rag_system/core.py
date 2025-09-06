#!/usr/bin/env python3
"""
Core GenomicRAG class with working implementation.
Restored from backup with modular organization.
"""

import logging
import re
from typing import List, Dict, Any, Optional
import os
import json
import asyncio

try:
    import dspy
    from rich.console import Console
    DSPY_AVAILABLE = True
    console = Console()
except ImportError:
    DSPY_AVAILABLE = False
    # Create a fallback console that prints to stdout
    class FallbackConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = FallbackConsole()
    logging.warning("DSPy not available - install dsp-ml package")

from ..config import LLMConfig
from ..lm_factory import make_lm
from ..query_processor import Neo4jQueryProcessor, LanceDBQueryProcessor, HybridQueryProcessor
from .dspy_signatures import NEO4J_SCHEMA
from .utils import setup_debug_logging, GenomicContext
from .log_formatter import setup_enhanced_logging
from .dspy_signatures import PlannerAgent, QueryClassifier, ContextRetriever, GenomicAnswerer
# Legacy TaskGraph types gated behind flag to enable quarantine without breakage
if os.getenv("AGENT_ENABLE_LEGACY_TASKGRAPH", "0") == "1":
    try:
        from .task_management import TaskGraph, Task, TaskType, TaskStatus  # type: ignore
    except Exception:  # pragma: no cover
        TaskGraph = Task = TaskType = TaskStatus = None  # type: ignore
else:
    TaskGraph = Task = TaskType = TaskStatus = None  # type: ignore
from .external_tools import AVAILABLE_TOOLS
from .intelligent_routing import IntelligentRouter
from .genome_selection import UnifiedGenomeSelector
from ..context_compression import ContextCompressor
from .memory import NoteKeeper
from .policy_engine import get_policy_engine
from .genome_context_extractor import GenomeContextExtractor
from ..mfp.operators import builtin as _mfp_builtin  # noqa: F401  # register builtins
from ..mfp.operators import catalog_search as _mfp_catalog_search  # noqa: F401  # register catalog search ops
from ..mfp.operators import planning_utils as _mfp_planning_utils  # noqa: F401  # register planning utility ops
from ..mfp.planning.composites import COMPOSITE_EXPANDERS, planner_catalog_overlay
from ..mfp.operators.base import operator_catalog, OperatorContext
from ..mfp.executor import execute_plan
from .query_validator import QueryValidator
# Old genome_selector.py replaced by unified genome_selection.py
from .router import get_router
from .agent.tools.validate import validate_toolcall
from .tracing import get_tracer
from .context.scope import GenomeScope

logger = logging.getLogger(__name__)

class GenomicRAG(dspy.Module if DSPY_AVAILABLE else object):
    """
    Main genomic RAG system with working implementation.
    
    Combines structured queries (Neo4j) with semantic search (LanceDB)
    and intelligent code interpreter enhancement.
    """
    
    def __init__(self, config: LLMConfig, chunk_context_size: int = 4096, enable_memory: bool = True, enhanced_logging: bool = False):
        """Initialize the genomic RAG system."""
        if DSPY_AVAILABLE:
            super().__init__()
        
        self.config = config
        self.chunk_context_size = chunk_context_size
        self.enable_memory = enable_memory
        
        # Initialize processors
        self.neo4j_processor = Neo4jQueryProcessor(config)
        self.lancedb_processor = LanceDBQueryProcessor(config)
        self.hybrid_processor = HybridQueryProcessor(config)
        
        # Initialize new intelligent components
        self.intelligent_router = IntelligentRouter()
        self.router = get_router()
        self.tracer = get_tracer()
        self.genome_selector = UnifiedGenomeSelector(self.neo4j_processor)
        self.context_compressor = ContextCompressor()
        self.genome_context_extractor = GenomeContextExtractor()
        self.query_validator = QueryValidator()
# Unified genome selector initialized above
        
        # Initialize memory system
        self.note_keeper = NoteKeeper() if enable_memory else None
        self.progressive_synthesizer = None  # Will be initialized when needed
        
        # Manual per-step model selection; legacy allocator removed
        
        # Initialize policy engine
        self.policy_engine = get_policy_engine()
        
        # Configure DSPy with model allocation
        self._configure_dspy()
        
        # DSPy components are now instantiated on-demand via _run() method
        # No need for persistent Predict attributes
        
        # Store DSPy availability for task executor
        self.dspy_available = DSPY_AVAILABLE
        
        # Set up enhanced logging if requested, otherwise use debug logging
        if enhanced_logging:
            try:
                setup_enhanced_logging(
                    log_level="INFO",
                    filter_noise=True,
                    show_timestamps=True,
                    export_to_file=False
                )
                logger.info("🎯 GenomicRAG initialized with enhanced logging")
            except Exception as e:
                logger.warning(f"Enhanced logging setup failed: {e}, using default logging")
                setup_debug_logging()
        else:
            setup_debug_logging()
        
        logger.info("✅ GenomicRAG system initialized successfully")
        logger.info(f"🔧 Configuration: Neo4j={bool(config.database.neo4j_uri)}, LanceDB={bool(config.database.lancedb_path)}")
        logger.info(f"🧠 Memory: {'Enabled' if enable_memory else 'Disabled'}, Chunk Size: {chunk_context_size}")
        logger.info(f"🔥 Model: {config.llm_model} ({config.model_mode} mode)")
    
    def _configure_dspy(self):
        """Minimal DSPy env setup. No global LM, no allocation, no max_tokens."""
        if not DSPY_AVAILABLE:
            return
            
        try:
            # Quiet noisy external loggers around API calls unless explicitly enabled
            try:
                import logging as _lg, os as _os
                # Quieten noisy deps
                _lg.getLogger("LiteLLM").setLevel(_lg.CRITICAL)
                _lg.getLogger("litellm").setLevel(_lg.CRITICAL)
                _lg.getLogger("litellm.proxy").setLevel(_lg.CRITICAL)
                _lg.getLogger("httpx").setLevel(_lg.WARNING)
                _lg.getLogger("dspy.adapters.json_adapter").setLevel(_lg.ERROR)
                # Disable LiteLLM standard/cold storage logging to avoid atexit errors
                _os.environ.setdefault('LITELLM_LOG', 'CRITICAL')
                _os.environ.setdefault('LITELLM_LOGGING', 'False')
                _os.environ.setdefault('LITELLM_PROXY_LOGGING', 'False')
                _os.environ.setdefault('LITELLM_DISABLE_COLD_STORAGE', '1')
                _os.environ.setdefault('LITELLM_DISABLE_STANDARD_LOGGING', '1')
            except Exception:
                pass
            # Configure API keys only (no LM setup)
            api_key = self.config.get_api_key()
            import os as _os
            if self.config.llm_provider == "openai" and api_key:
                _os.environ['OPENAI_API_KEY'] = api_key
            elif self.config.llm_provider == "anthropic" and api_key:
                _os.environ['ANTHROPIC_API_KEY'] = api_key
            else:
                logger.warning("No LLM API key configured for DSPy")
                
        except Exception as e:
            logger.error(f"Failed to configure DSPy: {e}")
            # No global fallback LM
            pass
    
    def _run(self, task_name: str, signature_cls, **kwargs):
        """
        Centralized Predict wrapper. Allocates the appropriate model via ModelAllocator,
        instantiates the module, executes it, and returns the result.
        """
        def _call(module):
            return module(**kwargs)
        
        try:
            import dspy
            lm = dspy.LM(model="openai/gpt-4.1-mini")
            module = dspy.Predict(signature_cls)
            with dspy.context(lm=lm):
                return _call(module)
        except Exception as e:
            logger.error(f"_run failed: {e}")
            return None
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all system components."""
        try:
            health_status = {}
            
            # Check processors
            health_status['neo4j'] = self.neo4j_processor.health_check() if hasattr(self.neo4j_processor, 'health_check') else False
            health_status['lancedb'] = self.lancedb_processor.health_check() if hasattr(self.lancedb_processor, 'health_check') else False
            health_status['hybrid'] = self.hybrid_processor.health_check() if hasattr(self.hybrid_processor, 'health_check') else False
            
            # Check DSPy
            health_status['dspy'] = DSPY_AVAILABLE
            
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {
                'neo4j': False,
                'lancedb': False, 
                'hybrid': False,
                'dspy': False
            }
    
    async def ask(self, question: str) -> Dict[str, Any]:
        """
        Main method to answer genomic questions with agentic planning.
        
        Args:
            question: Natural language question about genomic data
            
        Returns:
            Dict containing answer, confidence, sources, and metadata
        """
        try:
            console.print(f"🧬 [bold blue]Processing question:[/bold blue] {question}")
            try:
                self.tracer.emit("pipeline.start", {"question": question})
            except Exception:
                pass
            
            if not DSPY_AVAILABLE:
                return {
                    "question": question,
                    "answer": "DSPy not available - install dsp-ml package for full functionality",
                    "confidence": "low",
                    "citations": "",
                    "error": "Missing dependencies"
                }
            
            # STEP 0A: Agent-designed Macro Plan (operator catalog → plan → MFP executor)
            try:
                if getattr(self.config, 'FAST_PATH_ENABLED', True) and getattr(self.config, 'USE_MFP_PLANNER', True):
                    from .dspy_signatures import MacroPlannerSignature
                    logger.info("🧭 MacroPlanner: proposing operator plan")
                    # Light instrumentation: expose operator catalog size and key operators
                    try:
                        # Show only composites to the planner and in logs
                        oc = planner_catalog_overlay()
                        op_names = [o.get("name") for o in oc.get("operators", [])]
                        logger.info("🧭 Planner-visible operators: %d (examples: %s)", len(op_names), ", ".join(op_names[:5]))
                    except Exception:
                        pass
                    # Do NOT include large KO/PFAM reference blobs in planner context by default.
                    # We rely on local catalog search operators instead of stuffing catalogs into prompts.
                    ko_ref = ""
                    pf_ref = ""
                    if os.getenv('INCLUDE_REFERENCE_IN_PLANNER') in ("1", "true", "True"):
                        # Optional, user-forced inclusion with optional caps
                        try:
                            import csv  # noqa: F401
                            _ko_max = os.getenv('KO_REFERENCE_MAX_LINES')
                            ko_max_lines = int(_ko_max) if _ko_max not in (None, '', 'none', 'all') else None
                        except Exception:
                            ko_max_lines = None
                        try:
                            _pf_max = os.getenv('PFAM_REFERENCE_MAX_LINES')
                            pf_max_lines = int(_pf_max) if _pf_max not in (None, '', 'none', 'all') else None
                        except Exception:
                            pf_max_lines = None
                        try:
                            # Local small loaders to avoid imports when unused
                            def _ld_ko(max_lines=None):
                                import csv
                                path = os.path.join(os.getcwd(), 'data', 'reference', 'ko_list')
                                if not os.path.exists(path):
                                    return ""
                                lines = []
                                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                                    sample = f.read(4096)
                                    f.seek(0)
                                    try:
                                        dialect = csv.Sniffer().sniff(sample)
                                    except Exception:
                                        dialect = None
                                    if dialect:
                                        reader = csv.DictReader(f, dialect=dialect)
                                        cols = {k.lower(): k for k in (reader.fieldnames or [])}
                                        for i, row in enumerate(reader):
                                            if max_lines is not None and i >= max_lines:
                                                break
                                            knum = (row.get(cols.get('knum', ''), '') or row.get(cols.get('ko', ''), '')).strip()
                                            sdef = (row.get(cols.get('simplified_definition', ''), '') or row.get(cols.get('definition', ''), '')).strip()
                                            if knum and sdef:
                                                lines.append(f"{knum}: {sdef}")
                                    else:
                                        for i, raw in enumerate(f):
                                            if max_lines is not None and i >= max_lines:
                                                break
                                            raw = raw.strip()
                                            if not raw:
                                                continue
                                            parts = raw.split('\t') if '\t' in raw else raw.split(None, 1)
                                            knum = parts[0].strip() if parts else ''
                                            defin = parts[1].strip() if len(parts) > 1 else ''
                                            if knum and defin and (knum.startswith('K') or knum.lower().startswith('ko:')):
                                                lines.append(f"{knum}: {defin}")
                                return "\n".join(lines)
                            def _ld_pfam(max_lines=None):
                                import csv
                                path = os.path.join(os.getcwd(), 'data', 'reference', 'pfam_id_desc.tsv')
                                if not os.path.exists(path):
                                    return ""
                                lines = []
                                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                                    reader = csv.reader(f, delimiter='\t')
                                    for i, row in enumerate(reader):
                                        if max_lines is not None and i >= max_lines:
                                            break
                                        if not row:
                                            continue
                                        pfid = (row[0] if len(row) > 0 else '').strip()
                                        short = (row[1] if len(row) > 1 else '').strip()
                                        desc = (row[2] if len(row) > 2 else '').strip()
                                        if pfid and (short or desc):
                                            if short and desc and short not in desc:
                                                lines.append(f"{pfid}: {short}; {desc}")
                                            elif short and not desc:
                                                lines.append(f"{pfid}: {short}")
                                            else:
                                                lines.append(f"{pfid}: {desc}")
                                return "\n".join(lines)
                            ko_ref = _ld_ko(ko_max_lines)
                            pf_ref = _ld_pfam(pf_max_lines)
                        except Exception:
                            ko_ref = pf_ref = ""
                    def planner_call_inputs():
                        hard_constraints = (
                            "HARD CONSTRAINTS:\n"
                            "- If you include NeighborhoodContext, you MUST provide explicit seeds.\n"
                            "  Provide EITHER inputs.discovered_proteins referencing a bound rowset,\n"
                            "  OR params.protein_ids, OR params.seed_pfam_ids/seed_ko_ids.\n"
                            "- Prefer chaining identifiers via operator inputs, not guessing.\n"
                            "  Example: SearchPfamCatalogFuzzy → AnnotationDiscovery (rowset) with inputs.pfam_ids=pfam_ids → bind → NeighborhoodContext with inputs.discovered_proteins=<binding>.\n"
                            "- Use AnnotationDiscovery.inputs.pfam_ids/ko_ids to pass IDs from prior steps; do not hard-code IDs if you have them from catalog search.\n"
                            "- AnnotationDiscovery MUST set params.keyword (or params.q).\n"
                            "  You may ALSO bind ID lists via inputs (e.g., inputs:{'pfam_ids':'pfam_ids'} or inputs:{'ko_ids':'ko_ids'}) to focus rowsets, but do not omit keyword.\n"
                            "  Do NOT call AnnotationDiscovery with only formatting params (output_profile/group_by/fields) — that yields empty results.\n"
                            "- If a catalog search step is present, bind its ID outputs and, if relevant, pass them via inputs into AnnotationDiscovery; do not assume implicit availability.\n"
                            "- Keyword hygiene: When the user asks for context around specific genes/subunits/components, prefer direct subunit terms and concise synonyms; avoid broad class tokens and hyphenated 'like' analog terms unless the user explicitly requests exploratory breadth. Limit synonyms to a small number (e.g., ≤2) per theme.\n"
                            "- Quantity discipline: For gene/subunit context, keep PFAM catalog probes small — set FeatureDiscovery.limits.top_k ≈ 3–8 (default PFAM top_n=5). Use larger values only for broad capability surveys.\n"
                            "\nINTENT HINTS (operator bias):\n"
                            "- neighborhood|context|flanking|operon|adjacent → prefer GeneContext\n"
                            "- pathway|completeness|KEGG → prefer PathwayProfile\n"
                            "- CAZy|cazyme|BGC|biosynthetic → prefer ModuleProfile\n"
                            "- evidence|follow-up|sufficient → prefer EvidenceAndNext\n"
                            "- PFAM|KO|search|discover|find → prefer FeatureDiscovery\n"
                        )
                        # Restrict planner-visible catalog to composites only
                        planner_catalog = planner_catalog_overlay()
                        return dict(
                            question=question,
                            operator_catalog=json.dumps(planner_catalog),
                            constraints=hard_constraints,
                            ko_reference=ko_ref,
                            pfam_reference=pf_ref,
                        )

                    # Backward-compat + macro expansion helpers (planner output → executor input)
                    PRIMITIVE_TO_COMPOSITE = {
                        "SearchPfamCatalogFuzzy": "FeatureDiscovery",
                        "SearchKoCatalogFuzzy": "FeatureDiscovery",
                        "ExtractIdsFromCatalogHits": "FeatureDiscovery",
                        "QueryProteinsByIds": "FeatureDiscovery",
                        "AnnotationDiscovery": "FeatureDiscovery",
                        "NeighborhoodContext": "GeneContext",
                        "FetchPresentKOs": "PathwayProfile",
                        "LoadKoPathwayTotals": "PathwayProfile",
                        "ComputePathwayCompleteness": "PathwayProfile",
                        "QueryCazymesByGenome": "ModuleProfile",
                        "CountCazymeFamilies": "ModuleProfile",
                        "QueryBGCsByGenome": "ModuleProfile",
                        "AssessEvidence": "EvidenceAndNext",
                        "ProposeFollowup": "EvidenceAndNext",
                    }

                    def _rewrite_to_composites(plan_dict: dict) -> dict:
                        try:
                            steps_in = list(plan_dict.get('steps', []) or [])
                        except Exception:
                            return plan_dict
                        rewritten: List[dict] = []
                        for st in steps_in:
                            name = (st or {}).get('op')
                            comp = PRIMITIVE_TO_COMPOSITE.get(name)
                            if comp:
                                rewritten.append({"op": comp, "params": st.get("params", {})})
                            else:
                                rewritten.append(st)
                        plan_dict['steps'] = rewritten
                        return plan_dict

                    def _expand_composites(plan_dict: dict, question_text: str) -> dict:
                        try:
                            steps_in = list(plan_dict.get('steps', []) or [])
                        except Exception:
                            return plan_dict
                        expanded: List[dict] = []
                        for st in steps_in:
                            name = (st or {}).get('op')
                            params = (st or {}).get('params') or {}
                            if name in COMPOSITE_EXPANDERS:
                                substeps = COMPOSITE_EXPANDERS[name](params, {"question": question_text})
                                expanded.extend(substeps)
                            else:
                                expanded.append(st)
                        plan_dict['steps'] = expanded
                        return plan_dict

                    # Planner model (manual allocation): default to gpt-5-high
                    plan_res = None
                    try:
                        import dspy
                        model_id = getattr(self.config, 'planner_model', None) or 'gpt-5-high'
                        lm = make_lm(model_id, step="planner")
                        module = dspy.Predict(MacroPlannerSignature)
                        # Timing + effort visibility
                        import time as _t
                        _t0 = _t.time()
                        with dspy.context(lm=lm):
                            _pci = planner_call_inputs()
                            try:
                                # Persist full planner call inputs for debug
                                if self.note_keeper and hasattr(self.note_keeper, 'synthesis_notes_path'):
                                    sdir = self.note_keeper.synthesis_notes_path
                                    os.makedirs(sdir, exist_ok=True)
                                    # Decode operator_catalog if it's a JSON string for readability
                                    _pci_dump = dict(_pci)
                                    try:
                                        if isinstance(_pci_dump.get('operator_catalog'), str) and _pci_dump['operator_catalog'].strip().startswith('{'):
                                            _pci_dump['operator_catalog'] = json.loads(_pci_dump['operator_catalog'])
                                    except Exception:
                                        pass
                                    # Persist the signature docstring/context used by the planner
                                    try:
                                        from .dspy_signatures import MacroPlannerSignature as _MPS
                                        sig_text = (_MPS.__doc__ or '').strip()
                                        with open(os.path.join(sdir, 'planner_signature.txt'), 'w', encoding='utf-8') as f_sig:
                                            f_sig.write(sig_text)
                                    except Exception as _sig_err:
                                        logger.info(f"Planner signature save skipped: {_sig_err}")
                                    with open(os.path.join(sdir, 'planner_call_inputs.json'), 'w', encoding='utf-8') as f_in:
                                        json.dump(_pci_dump, f_in, indent=2)
                            except Exception as _pci_err:
                                logger.info(f"Planner call inputs save skipped: {_pci_err}")
                            plan_res = module(**_pci)
                        _ms = int((_t.time() - _t0) * 1000)
                        try:
                            # Best-effort effort extraction from alias
                            from ..lm_factory import _extract_gpt5_effort as _eff
                            eff = _eff(model_id) or 'n/a'
                        except Exception:
                            eff = 'n/a'
                        logger.info(f"🕒 Planner latency: {_ms} ms (model={model_id}, effort={eff})")
                    except Exception as _e:
                        logger.warning(f"Planner call failed: {_e}")
                    plan_text = getattr(plan_res, 'plan_json', '') if plan_res else ''
                    # Persist raw planner output for debugging/inspection
                    try:
                        if self.note_keeper and hasattr(self.note_keeper, 'synthesis_notes_path'):
                            sdir = self.note_keeper.synthesis_notes_path
                            os.makedirs(sdir, exist_ok=True)
                            with open(os.path.join(sdir, 'planner_raw.txt'), 'w', encoding='utf-8') as f_pr:
                                f_pr.write(str(plan_text) if plan_text is not None else '')
                            # Also persist the planner-visible catalog overlay for full context
                            try:
                                cat = planner_catalog_overlay()
                                with open(os.path.join(sdir, 'planner_catalog_overlay.json'), 'w', encoding='utf-8') as f_cat:
                                    json.dump(cat, f_cat, indent=2)
                            except Exception as _cat_err:
                                logger.info(f"Planner catalog overlay save skipped: {_cat_err}")
                    except Exception as _pr_err:
                        logger.info(f"Planner raw save skipped: {_pr_err}")
                    # Helper: extract first balanced JSON object from text
                    def _extract_first_json_object(txt: str) -> str | None:
                        if not isinstance(txt, str) or not txt:
                            return None
                        # Find first '{'
                        try:
                            start = txt.find('{')
                            if start == -1:
                                return None
                            depth = 0
                            in_str = False
                            esc = False
                            for i in range(start, len(txt)):
                                ch = txt[i]
                                if in_str:
                                    if esc:
                                        esc = False
                                    elif ch == '\\':
                                        esc = True
                                    elif ch == '"':
                                        in_str = False
                                else:
                                    if ch == '"':
                                        in_str = True
                                    elif ch == '{':
                                        depth += 1
                                    elif ch == '}':
                                        depth -= 1
                                        if depth == 0:
                                            return txt[start:i+1]
                            return None
                        except Exception:
                            return None

                    plan = None
                    if isinstance(plan_text, str) and plan_text.strip():
                        candidate = plan_text.strip()
                        # If it doesn't end cleanly, try to repair by extracting first JSON object
                        if not candidate.startswith('{') or not candidate.endswith('}'):
                            repaired = _extract_first_json_object(candidate)
                        else:
                            repaired = candidate
                        try:
                            if repaired:
                                p = json.loads(repaired)
                                try:
                                    raw_ops = [st.get('op') for st in (p.get('steps') or [])]
                                    logger.info("🧭 MacroPlanner proposed composites/primitives: %s", ", ".join([str(x) for x in raw_ops]))
                                except Exception:
                                    pass
                                # Rewrite legacy primitives → composites, then expand composites → primitives
                                p = _rewrite_to_composites(p)
                                plan = _expand_composites(p, question)
                                if repaired != candidate:
                                    logger.info("🔧 Planner JSON repaired (trailing/preamble text ignored)")
                        except Exception as _perr:
                            logger.info(f"Planner JSON parse failed: {_perr}")
                    if plan is not None:
                        def _env_has_results(env_dict: Dict[str, Any]) -> bool:
                            try:
                                for k, v in (env_dict or {}).items():
                                    if isinstance(v, list) and len(v) > 0:
                                        return True
                                    # Also consider bound dicts that wrap rows
                                    if isinstance(v, dict):
                                        for vv in v.values():
                                            if isinstance(vv, list) and len(vv) > 0:
                                                return True
                                return False
                            except Exception:
                                return False

                        def _collect_macro_raw_items(env_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
                            items: List[Dict[str, Any]] = []
                            seen_proteins: set = set()
                            try:
                                # Only include whitelisted list bindings to avoid massive context
                                allowed_list_keys = {
                                    'discovered_proteins',
                                    'pathway_completeness',
                                    'bgcs',
                                    'cazymes',
                                    'cazyme_family_counts',
                                }
                                for k, v in (env_dict or {}).items():
                                    if isinstance(v, list):
                                        if k in allowed_list_keys:
                                            rows = v
                                            # Deduplicate discovered_proteins globally across items
                                            if k == 'discovered_proteins':
                                                filtered = []
                                                dropped = 0
                                                for r in rows:
                                                    gid = str(r.get('genome_id',''))
                                                    pid = str(r.get('protein_id',''))
                                                    sig = (gid, pid)
                                                    if sig in seen_proteins:
                                                        dropped += 1
                                                        continue
                                                    seen_proteins.add(sig)
                                                    filtered.append(r)
                                                rows = filtered
                                                if dropped:
                                                    logger.info(f"Context trim: dropped {dropped} duplicate discovered_proteins rows (binding='{k}')")
                                            items.append({'type': 'macro_result', 'name': k, 'rows': rows})
                                    elif isinstance(v, dict):
                                        # Allow planner-produced structured items (e.g., followup_request)
                                        if isinstance(v.get('type'), str):
                                            items.append(v)
                                        # Also extract common list payloads from bound dicts
                                        for key in ('discovered_proteins',):
                                            rows2 = v.get(key)
                                            if isinstance(rows2, list) and rows2:
                                                # Deduplicate globally
                                                filtered2 = []
                                                dropped2 = 0
                                                for r in rows2:
                                                    gid = str(r.get('genome_id',''))
                                                    pid = str(r.get('protein_id',''))
                                                    sig = (gid, pid)
                                                    if sig in seen_proteins:
                                                        dropped2 += 1
                                                        continue
                                                    seen_proteins.add(sig)
                                                    filtered2.append(r)
                                                if dropped2:
                                                    logger.info(f"Context trim: dropped {dropped2} duplicate discovered_proteins rows (binding='{k}.{key}')")
                                                items.append({'type': 'macro_result', 'name': f"{k}.{key}", 'rows': filtered2, 'format': v.get('_format')})
                                        # Pass facet_summary through as macro_result rows for KO/PFAM facets
                                        try:
                                            fs = v.get('facet_summary')
                                            if isinstance(fs, dict):
                                                kos = fs.get('kos')
                                                if isinstance(kos, list) and kos:
                                                    rows_k = []
                                                    for it in kos:
                                                        if isinstance(it, dict) and it.get('id'):
                                                            rows_k.append({'kos': [str(it['id'])], 'count': int(it.get('count', 0) or 0)})
                                                    if rows_k:
                                                        items.append({'type': 'macro_result', 'name': f"{k}.facet_kos", 'rows': rows_k})
                                                pfs = fs.get('pfams')
                                                if isinstance(pfs, list) and pfs:
                                                    rows_p = []
                                                    for it in pfs:
                                                        if isinstance(it, dict) and it.get('id'):
                                                            rows_p.append({'pfams': [str(it['id'])], 'count': int(it.get('count', 0) or 0)})
                                                    if rows_p:
                                                        items.append({'type': 'macro_result', 'name': f"{k}.facet_pfams", 'rows': rows_p})
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                            return items

                        all_raw_items: List[Dict[str, Any]] = []
                        combined_env: Dict[str, Any] = {}
                        attempts = 0
                        max_attempts = 2  # first pass + one retry

                        while attempts < max_attempts:
                            try:
                                ops_list = [step.get("op") for step in plan.get("steps", [])]
                                logger.info("🧭 MacroPlanner plan ops: %s", ", ".join([str(x) for x in ops_list]))
                            except Exception:
                                pass
                            logger.info("🧭 MacroPlanner: executing %d steps (attempt %d/%d)", len(plan.get('steps', [])), attempts + 1, max_attempts)
                            ctx = OperatorContext(neo4j_driver=self.neo4j_processor.driver, project_root=str(getattr(self, 'project_root', '')))
                            env = execute_plan(plan, ctx)
                            # Merge environments (last write wins on same keys)
                            try:
                                combined_env.update(env)
                            except Exception:
                                combined_env = env
                            # Collect raw items for this attempt
                            attempt_items = _collect_macro_raw_items(env)
                            all_raw_items.extend(attempt_items)
                            # Add plan note for this attempt
                            plan_note = {"type": "task_note", "task_id": f"mfp_plan_attempt_{attempts+1}", "description": "Macro plan executed (operators + params)", "observations": [], "key_findings": [], "quantitative_data": {"operators": [step.get("op") for step in plan.get("steps", [])], "plan": plan}, "cross_task_connections": []}
                            all_raw_items.append(plan_note)
                            # Sufficiency check
                            if _env_has_results(env) or attempts == max_attempts - 1:
                                break
                            # Retry: re-plan and re-execute
                            logger.info("🔁 MacroPlanner retry: insufficient evidence, attempting broader pass")
                            def planner_call_retry(module):
                                retry_constraints = (
                                    "ALLOW_KEYWORD_DISCOVERY=1\n" \
                                    "HARD CONSTRAINTS:\n"
                                    "- If you include NeighborhoodContext, you MUST provide explicit seeds (inputs.discovered_proteins OR params.protein_ids OR params.seed_pfam_ids/seed_ko_ids).\n"
                                    "- Chain IDs via inputs (e.g., inputs.pfam_ids=pfam_ids) for rowset steps, then bind and pass discovered_proteins to NeighborhoodContext.\n"
                                )
                                return module(
                                    question=question,
                                    operator_catalog=json.dumps(operator_catalog()),
                                    constraints=retry_constraints,
                                    ko_reference=ko_ref,
                                    pfam_reference=pf_ref,
                                )
                            try:
                                import dspy
                                model_id = getattr(self.config, 'planner_model', None) or 'gpt-5-high'
                                lm = make_lm(model_id, step="planner")
                                module = dspy.Predict(MacroPlannerSignature)
                                import time as _t
                                _t0 = _t.time()
                                with dspy.context(lm=lm):
                                    plan_res2 = module(
                                        question=question,
                                        operator_catalog=json.dumps(planner_catalog_overlay()),
                                        constraints="allow_keyword_discovery=1",
                                        ko_reference=ko_ref,
                                        pfam_reference=pf_ref,
                                    )
                                _ms = int((_t.time() - _t0) * 1000)
                                try:
                                    from ..lm_factory import _extract_gpt5_effort as _eff
                                    eff = _eff(model_id) or 'n/a'
                                except Exception:
                                    eff = 'n/a'
                                logger.info(f"🕒 Planner retry latency: {_ms} ms (model={model_id}, effort={eff})")
                            except Exception as _e3:
                                logger.warning(f"Planner retry failed: {_e3}")
                                plan_res2 = None
                            plan2_text = getattr(plan_res2, 'plan_json', '') if plan_res2 else ''
                            # Persist raw planner RETRY output for debugging
                            try:
                                if self.note_keeper and hasattr(self.note_keeper, 'synthesis_notes_path'):
                                    sdir = self.note_keeper.synthesis_notes_path
                                    os.makedirs(sdir, exist_ok=True)
                                    with open(os.path.join(sdir, 'planner_raw_retry.txt'), 'w', encoding='utf-8') as f_pr2:
                                        f_pr2.write(str(plan2_text) if plan2_text is not None else '')
                            except Exception as _pr2_err:
                                logger.info(f"Planner raw retry save skipped: {_pr2_err}")
                            # Attempt JSON repair for retry output as well
                            if isinstance(plan2_text, str) and plan2_text.strip():
                                candidate2 = plan2_text.strip()
                                if not candidate2.startswith('{') or not candidate2.endswith('}'):
                                    repaired2 = _extract_first_json_object(candidate2)
                                else:
                                    repaired2 = candidate2
                                try:
                                    if repaired2:
                                        p2 = json.loads(repaired2)
                                        p2 = _rewrite_to_composites(p2)
                                        plan = _expand_composites(p2, question)
                                        if repaired2 != candidate2:
                                            logger.info("🔧 Planner JSON (retry) repaired (trailing/preamble text ignored)")
                                except Exception:
                                    pass
                            attempts += 1
                            if attempts >= max_attempts:
                                break
                        # Follow-up proposals (if any) should come from planned operators,
                        # not decided here. The synthesizer is invoked only at the end.

                        # Always synthesize at the end (with whatever we gathered)
                        # ProgressiveSynthesizer is deprecated here; final synthesis handled later
                        # Debug: summarize raw items sizes to pinpoint context inflation sources
                        try:
                            summary = []
                            for it in all_raw_items:
                                if isinstance(it, dict) and it.get('type') == 'macro_result':
                                    name = it.get('name','')
                                    rows = it.get('rows') or []
                                    summary.append((name, len(rows)))
                            if summary:
                                summary.sort(key=lambda x: x[1], reverse=True)
                                top = ", ".join([f"{n}:{c}" for n,c in summary[:10]])
                                total_rows = sum(c for _, c in summary)
                                logger.debug(f"Context debug: {len(summary)} result lists, total_rows={total_rows}. Top lists: {top}")
                        except Exception:
                            pass

                        # Phase 0 instrumentation: persist full MacroPlanner environment
                        try:
                            if self.note_keeper and hasattr(self.note_keeper, 'synthesis_notes_path'):
                                sdir = self.note_keeper.synthesis_notes_path
                                os.makedirs(sdir, exist_ok=True)
                                with open(os.path.join(sdir, 'all_env.json'), 'w', encoding='utf-8') as f_env:
                                    json.dump(combined_env, f_env, indent=2, default=str)
                                try:
                                    tool_calls = combined_env.get('__tool_calls')
                                    if tool_calls:
                                        with open(os.path.join(sdir, 'tool_calls.json'), 'w', encoding='utf-8') as f_tc:
                                            json.dump(tool_calls, f_tc, indent=2, default=str)
                                except Exception as _tc_err:
                                    logger.info(f"Tool calls save skipped: {_tc_err}")
                        except Exception as _env_save_err:
                            logger.info(f"MacroPlanner env save skipped: {_env_save_err}")

                        # Decide whether to bypass IRB based on token budget (small contexts go straight to final synthesis)
                        def _estimate_tokens(s: str) -> int:
                            try:
                                import tiktoken
                                enc = tiktoken.encoding_for_model(getattr(self.config, 'llm_model', 'gpt-4.1-mini'))
                                return len(enc.encode(s or ''))
                            except Exception:
                                try:
                                    # Rough heuristic
                                    return max(1, int(len(s or '') / 4))
                                except Exception:
                                    return 1000000

                        # Pretty-print a simple task graph for final synthesis context
                        def _render_task_graph(plan_dict: dict) -> str:
                            try:
                                steps = plan_dict.get('steps', []) if isinstance(plan_dict, dict) else []
                                lines = ["TASK GRAPH:"]
                                for i, st in enumerate(steps, 1):
                                    op = st.get('op')
                                    params = st.get('params') or {}
                                    # Keep params compact
                                    import json as _json
                                    p = _json.dumps(params, separators=(',',':'))[:200]
                                    lines.append(f"{i}. {op} params={p}")
                                return "\n".join(lines)
                            except Exception:
                                return ""

                        # Compute raw context size once
                        import json as _json
                        raw_context_json = _json.dumps(all_raw_items, default=str, separators=(',',':'))
                        # Bypass threshold defaults to 30k tokens; ensure it also fits target reporter context
                        bypass_cap = 0
                        try:
                            import os as _os
                            bypass_cap = int(_os.getenv('IRB_BYPASS_TOKENS', '30000'))
                        except Exception:
                            bypass_cap = 30000

                        # Rough reporter context capacity
                        def _reporter_cap() -> int:
                            try:
                                rep = getattr(self.config, 'reporter_model', None)
                                if not rep:
                                    # Use allocator's premium model default (gpt-5 ~30k)
                                    return 30000
                                low = rep.lower()
                                if 'gpt-5' in low or '/o1' in low:
                                    return 30000
                                if 'gpt-4.1' in low:
                                    return 1_000_000
                                if 'claude-sonnet-4' in low:
                                    return 200_000
                                return 100_000
                            except Exception:
                                return 30000

                        raw_tokens = _estimate_tokens(raw_context_json)
                        fits_window = raw_tokens <= max(1, _reporter_cap() - 1000)
                        should_bypass_irb = (not getattr(self.config, 'IRB_ENABLED', True)) or (raw_tokens <= bypass_cap and fits_window)

                        report_context = None
                        task_graph_text = _render_task_graph(plan)

                        if should_bypass_irb:
                            # Bypass IRB: synthesize directly from compact JSON context
                            report_context = (task_graph_text + "\n\nCONTEXT (JSON):\n" + raw_context_json)
                        else:
                            # IRB (Incremental Report Builder) path
                            try:
                                from .memory.incremental_report_builder import IncrementalReportBuilder
                                from .memory.doc_ast import to_markdown as _to_md
                            except Exception as e:
                                logger.error(f"IRB import failed: {e}")
                                raise
                            # Build tool cache compatible with NoteKeeper
                            trc = None
                            try:
                                if self.note_keeper and hasattr(self.note_keeper, 'session_path'):
                                    from .memory.tool_result_cache import ToolResultCache
                                    trc = ToolResultCache(str(self.note_keeper.session_path))
                            except Exception:
                                trc = None
                            irb = IncrementalReportBuilder(self.note_keeper, None, trc, self.config)
                            doc = irb.run(all_raw_items, obligations=[])
                            if getattr(irb, 'failed', False):
                                raise RuntimeError(f"IRB bug-out: {getattr(irb, 'fail_reason', 'unknown')}")
                            irb_markdown = _to_md(doc)
                            report_context = (task_graph_text + "\n\n" + irb_markdown)
                            # Persist IRB outputs for inspection under session notes
                            try:
                                if self.note_keeper and hasattr(self.note_keeper, 'synthesis_notes_path'):
                                    sdir = self.note_keeper.synthesis_notes_path
                                    os.makedirs(sdir, exist_ok=True)
                                    # Save the raw IRB markdown
                                    with open(os.path.join(sdir, 'irb_report.md'), 'w', encoding='utf-8') as f_md:
                                        f_md.write(irb_markdown)
                                    # Save the exact report context handed to the reporter
                                    with open(os.path.join(sdir, 'report_context.md'), 'w', encoding='utf-8') as f_ctx:
                                        f_ctx.write(report_context)
                                    # Save the IRB document AST as JSON for deeper inspection
                                    try:
                                        import json as _json
                                        payload = None
                                        try:
                                            payload = doc.model_dump()
                                        except Exception:
                                            payload = getattr(doc, 'dict', lambda: {})()
                                        with open(os.path.join(sdir, 'irb_report.json'), 'w', encoding='utf-8') as f_js:
                                            _json.dump(payload, f_js, indent=2, default=str)
                                    except Exception as _serr:
                                        logger.info(f"IRB AST serialization skipped: {_serr}")
                            except Exception as _save_err:
                                logger.info(f"IRB report save skipped: {_save_err}")

                        # Final report generation (GPT-5 via DSPy) using GenomicSynthesizer
                        final_answer = None
                        try:
                            from .dspy_signatures import GenomicSynthesizer
                            import dspy
                            model_id = getattr(self.config, 'reporter_model', None) or 'gpt-5-high'
                            lm = make_lm(model_id, step="reporter")
                            module = dspy.Predict(GenomicSynthesizer)
                            # Optional neighborhoods payload from MacroPlanner env
                            neighborhoods_payload = None
                            try:
                                if isinstance(combined_env, dict):
                                    nb = combined_env.get('neighborhood_macro_result')
                                    if nb:
                                        import json as _json
                                        neighborhoods_payload = _json.dumps(nb, default=str)
                            except Exception:
                                neighborhoods_payload = None

                            with dspy.context(lm=lm):
                                synth_res = module(
                                    question=question,
                                    context=report_context,
                                    task_graph=task_graph_text,
                                    synthesis_mode="comprehensive_report",
                                    neighborhoods_json=neighborhoods_payload or "",
                                )
                            if synth_res and hasattr(synth_res, 'summary'):
                                final_answer = synth_res.summary
                        except Exception as e:
                            logger.warning(f"Final report synthesis call failed or unavailable: {e}")

                        # Fallbacks if synthesis not available
                        if not final_answer:
                            if not should_bypass_irb:
                                # IRB markdown is a reasonable fallback
                                final_answer = irb_markdown
                            else:
                                # As a last resort, emit a compact JSON summary
                                final_answer = (
                                    "Report synthesis unavailable; returning compact context.\n\n" + report_context[:20000]
                                )
                        return {
                            "question": question,
                            "answer": final_answer,
                            "confidence": "high",
                            "citations": "",
                            "query_metadata": {
                                "execution_mode": "mfp_planned",
                                "note_taking_enabled": self.note_keeper is not None,
                                "steps": len(plan.get('steps', [])),
                                "attempts": attempts + 1,
                            },
                        }
            except Exception as e:
                logger.info(f"MacroPlanner path skipped: {e}")

            # STEP 0B: Macro Fast Path (canonicalizer-first, deterministic execution)
            try:
                if getattr(self.config, "FAST_PATH_ENABLED", True):
                    from .agent_executor import UnifiedAgentExecutor
                    fp_agent = UnifiedAgentExecutor(self, note_keeper=self.note_keeper)
                    fp_result = await fp_agent._try_fast_path_locus_discovery(question)
                    if fp_result is not None:
                        console.print("🚄 [bold green]Macro Fast Path executed[/bold green]")
                        return {
                            "question": question,
                            "answer": fp_result.final_answer,
                            "confidence": fp_result.confidence,
                            "citations": fp_result.citations,
                            "query_metadata": {
                                "execution_mode": "macro_fast_path",
                                "total_steps": fp_result.total_steps,
                                "tools_used": fp_result.tools_used,
                                "execution_time": fp_result.total_execution_time,
                                "note_taking_enabled": self.note_keeper is not None,
                            },
                        }
                    # If fast path seeded steps for FSM, continue using the same executor
                    if getattr(fp_agent, "_seed_steps", None):
                        if getattr(self.config, 'DISABLE_FSM', False):
                            logger.info("FSM disabled by configuration; ignoring fast-path seed steps and continuing without FSM")
                        else:
                            console.print("🚦 [bold cyan]Fast Path seed ready → continuing with FSM[/bold cyan]")
                            agent_res = await fp_agent._execute_agent_workflow_fsm(question, selected_genome=None)
                            return {
                                "question": question,
                                "answer": agent_res.final_answer,
                                "confidence": agent_res.confidence,
                                "citations": agent_res.citations,
                                "query_metadata": {
                                    "execution_mode": "macro_fast_path_seeded_fsm",
                                    "total_steps": agent_res.total_steps,
                                    "tools_used": agent_res.tools_used,
                                    "execution_time": agent_res.total_execution_time,
                                    "note_taking_enabled": self.note_keeper is not None,
                                },
                            }
            except Exception as e:
                # Respect fail-fast on grammar/tool compile errors
                if getattr(self.config, 'FAIL_FAST_ON_GRAMMAR_ERROR', False):
                    raise
                logger.info(f"Macro Fast Path not taken: {e}")

            # STEP 1: Let the LLM decide execution strategy directly
            if getattr(self.config, 'DISABLE_FSM', False):
                console.print("🛑 [bold red]FSM disabled by configuration[/bold red]")
                return {
                    "question": question,
                    "answer": "Macro Fast Path did not yield sufficient results and FSM is disabled. Please refine the marker/signature or enable FSM for agentic exploration.",
                    "confidence": "low",
                    "citations": "",
                    "query_metadata": {
                        "execution_mode": "disabled_fsm",
                        "note_taking_enabled": self.note_keeper is not None,
                    },
                }
            console.print("🤖 [bold]Using LLM-based execution planning[/bold]")
            
            # Use model allocation for planning (gpt-5 for complex planning tasks)
            logger.info("🧠 Using model allocation for intelligent planning")
            
            def planning_call(module):
                return module(user_query=question)
            
            planning_result = None
            try:
                import dspy
                model_id = getattr(self.config, 'planner_model', None) or 'gpt-5-high'
                lm = make_lm(model_id, step="planner")
                module = dspy.Predict(PlannerAgent)
                with dspy.context(lm=lm):
                    planning_result = module(user_query=question)
            except Exception as _e:
                logger.warning(f"Planner call failed: {_e}")
            
            if planning_result is None:
                logger.warning("Model allocation failed for planning, falling back to default")
                planning_result = self._run("agentic_planning", PlannerAgent, user_query=question)
            
            console.print(f"🎯 Planning decision: {'agentic' if planning_result.requires_planning else 'traditional'}")
            try:
                self.tracer.emit("pipeline.plan_decision", {
                    "requires_planning": bool(planning_result.requires_planning),
                    "reasoning": getattr(planning_result, 'reasoning', None),
                })
            except Exception:
                pass
            console.print(f"💭 Reasoning: {planning_result.reasoning}")
            
            # Execute based on LLM's decision
            if planning_result.requires_planning:
                # AGENTIC PATH: Multi-step task execution with upfront genome selection
                task_plan = planning_result.task_plan
                if task_plan == "N/A" or not task_plan or task_plan.strip() == "":
                    console.print("⚠️ [yellow]Agentic mode chosen but no task plan provided, falling back to traditional mode[/yellow]")
                    return await self._execute_traditional_query(question, None)
                
                # INTELLIGENT UPFRONT GENOME SELECTION - One LLM call for the entire agentic workflow
                console.print("🧠 [bold blue]Analyzing genome selection intent for agentic workflow[/bold blue]")
                
                try:
                    # Use unified genome selector for LLM-based analysis
                    llm_selector = self.genome_selector
                    
                    selection_result = await llm_selector.analyze_genome_intent(question)
                    
                    if selection_result.success:
                        console.print(f"🧬 [bold green]LLM genome analysis:[/bold green] intent={selection_result.intent}, confidence={selection_result.confidence:.2f}")
                        console.print(f"💭 [dim]Reasoning: {selection_result.reasoning}[/dim]")
                        
                        if selection_result.intent == "specific" and selection_result.target_genomes:
                            selected_genome = selection_result.target_genomes[0]  # Use first genome for now
                            console.print(f"🎯 [bold cyan]All agentic tasks will target genome:[/bold cyan] {selected_genome}")
                        else:
                            selected_genome = None
                            console.print(f"🌐 [bold cyan]All agentic tasks will analyze across all genomes[/bold cyan] (intent: {selection_result.intent})")
                    else:
                        logger.warning(f"LLM genome analysis failed: {selection_result.error_message}")
                        selected_genome = None
                        console.print("🌐 [bold cyan]Falling back to global analysis across all genomes[/bold cyan]")
                        
                except Exception as e:
                    logger.error(f"LLM genome selection failed: {e}")
                    selected_genome = None
                    console.print("⚠️ [yellow]Genome selection error, using global analysis[/yellow]")
                
                return await self._execute_agentic_plan(question, planning_result, selected_genome)
            else:
                # TRADITIONAL PATH: Direct query execution
                return await self._execute_traditional_query(question, None)
                
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            
            # Check if this is a repairable error from query processor
            repair_message = None
            if hasattr(self.hybrid_processor, 'neo4j_processor') and hasattr(self.hybrid_processor.neo4j_processor, 'get_last_repair_result'):
                repair_result = self.hybrid_processor.neo4j_processor.get_last_repair_result()
                if repair_result and repair_result.success and repair_result.user_message:
                    repair_message = repair_result.user_message
                    logger.info(f"Using TaskRepairAgent message: {repair_message[:100]}...")
            
            if repair_message:
                return {
                    "question": question,
                    "answer": repair_message,
                    "confidence": "medium - error handled gracefully",
                    "citations": "",
                    "repair_info": "TaskRepairAgent provided helpful guidance"
                }
            else:
                return {
                    "question": question,
                    "answer": f"I encountered an error while processing your question: {str(e)}",
                    "confidence": "low",
                    "citations": "",
                    "error": str(e)
                }
    
    async def _execute_traditional_query(self, question: str, routing_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute traditional single-step query with enhanced genome scoping and compression."""
        console.print("📋 [dim]Using traditional query path[/dim]")

        # Stage A deterministic guardrail via unified router
        try:
            router_decision = self.router.route(question)
            if router_decision.tool == "whole_genome_reader":
                toolcall = {"tool": router_decision.tool, "params": router_decision.params}
                ok, errs = validate_toolcall(toolcall)
                if not ok:
                    raise ValueError(f"StageA whole_genome_reader toolcall invalid: {'; '.join(errs)}")

                console.print("🧬 [bold cyan]Stage A routed to whole_genome_reader[/bold cyan]")
                # Use curated spatial reader (global by default if no specific genome provided)
                from .whole_genome_reader import read_all_genomes_spatial
                spatial_results = await read_all_genomes_spatial(self.neo4j_processor)

                if spatial_results and spatial_results.get('success') and spatial_results.get('genome_contexts'):
                    scope = GenomeScope(genome_id="*", contig_ids=tuple(), coordinate_window=(0, 0))
                    context = GenomicContext(
                        structured_data=spatial_results['genome_contexts'],
                        semantic_data=[],
                        metadata={'analysis_type': 'SPATIAL_GENOMIC', 'tool_used': 'whole_genome_reader', 'genome_scope': scope.__dict__},
                        query_time=0.0,
                        compressed_context=""
                    )
                    formatted_context = self._format_spatial_context(context)
                    return await self._synthesize_answer(
                        question,
                        formatted_context,
                        query_type="SPATIAL_GENOMIC",
                        analysis_type="spatial_genomic",
                    )
                else:
                    return {
                        "question": question,
                        "answer": spatial_results.get('tool_output') if isinstance(spatial_results, dict) else "No spatial genomic data retrieved.",
                        "confidence": "low",
                        "citations": "",
                        "query_metadata": {"analysis_type": "SPATIAL_GENOMIC", "tool_used": "whole_genome_reader"}
                    }
            # Stage B: database_query via templates
            if router_decision.tool == "database_query":
                params = router_decision.params or {}
                template = params.get("template")
                slots = params.get("slots", {})
                # Inject default limit if not provided
                if isinstance(slots, dict) and "limit" not in slots:
                    try:
                        slots["limit"] = int(self.policy_engine.get_max_results("database_query"))
                    except Exception:
                        slots["limit"] = 100
                if template:
                    try:
                        self.tracer.emit("router.db_template.start", {"template": template})
                    except Exception:
                        pass
                    # Execute template safely via processor
                    db_result = await self.neo4j_processor.execute_named_template(template, slots)
                    # Derive scope (non-overridable) from slots when available
                    scope = self._derive_scope_from_slots(slots)
                    # Convert to GenomicContext and synthesize
                    context = GenomicContext(
                        structured_data=db_result.results,
                        semantic_data=[],
                        metadata={
                            'analysis_type': 'FUNCTIONAL_ANNOTATION',
                            'tool_used': 'database_query',
                            'template': template,
                            'genome_scope': scope.__dict__ if scope else None,
                            'result_count': db_result.metadata.get('result_count', 0)
                        },
                        query_time=db_result.execution_time,
                        compressed_context=""
                    )
                    formatted_context = self._format_context(context)
                    return await self._synthesize_answer(
                        question,
                        formatted_context,
                        query_type=f"template:{template}",
                        analysis_type="functional_annotation",
                    )
            # Stage B: literature_search
            if router_decision.tool == "literature_search":
                try:
                    self.tracer.emit("router.literature.start", {})
                except Exception:
                    pass
                lit = await self._execute_literature_search(question)
                return {
                    "question": question,
                    "answer": lit or "",
                    "confidence": "medium" if lit else "low",
                    "citations": "",
                    "query_metadata": {"tool": "literature_search", "genome_scope": None}
                }
            # Stage B: code_interpreter (no context yet; pass empty context)
            if router_decision.tool == "code_interpreter":
                try:
                    self.tracer.emit("router.code_interpreter.start", {})
                except Exception:
                    pass
                ci = await self._execute_code_interpreter(question, GenomicContext([], [], {}, 0.0))
                return {
                    "question": question,
                    "answer": ci or "",
                    "confidence": "medium" if ci else "low",
                    "citations": "",
                    "query_metadata": {"tool": "code_interpreter", "genome_scope": None}
                }
            # Stage B: similarity_search via LanceDB (by_id only)
            if router_decision.tool == "similarity_search":
                params = router_decision.params or {}
                mode = params.get("mode", "by_id")
                k = int(params.get("k", 10))
                filters = params.get("filters", {})
                if mode == "by_id":
                    protein_id = params.get("id")
                    try:
                        self.tracer.emit("router.similarity.start", {"mode": mode, "k": k})
                    except Exception:
                        pass
                    sim = await self.lancedb_processor.execute_similarity(mode, k, protein_id=protein_id, filters=filters)
                    context = GenomicContext(
                        structured_data=sim.results,
                        semantic_data=[],
                        metadata={
                            'analysis_type': 'SIMILARITY_SEARCH',
                            'tool_used': 'similarity_search',
                            'mode': mode,
                            'k': k,
                            'genome_scope': None,
                        },
                        query_time=sim.execution_time,
                        compressed_context=""
                    )
                    formatted_context = self._format_context(context)
                    return await self._synthesize_answer(
                        question,
                        formatted_context,
                        query_type=f"similarity:{mode}",
                        analysis_type="functional_annotation",
                    )
                else:
                    # by_sequence supported via runtime embedder (if deps available)
                    sequence = params.get("sequence")
                    try:
                        sim = await self.lancedb_processor.execute_similarity(mode, k, sequence=sequence, filters=filters)
                        context = GenomicContext(
                            structured_data=sim.results,
                            semantic_data=[],
                            metadata={
                                'analysis_type': 'SIMILARITY_SEARCH',
                                'tool_used': 'similarity_search',
                                'mode': mode,
                                'k': k,
                            },
                            query_time=sim.execution_time,
                            compressed_context=""
                        )
                        formatted_context = self._format_context(context)
                        return await self._synthesize_answer(
                            question,
                            formatted_context,
                            query_type=f"similarity:{mode}",
                            analysis_type="functional_annotation",
                        )
                    except Exception as e:
                        return {
                            "question": question,
                            "answer": f"Similarity by sequence unavailable: {e}",
                            "confidence": "low",
                            "citations": "",
                            "error": "similarity_by_sequence_unavailable"
                        }
        except Exception as e:
            logger.error(f"Stage A/B routing failed or not applicable: {e}")

        # If router suggested something else, log suggestion for tracing
        try:
            if 'router_decision' in locals() and router_decision and router_decision.tool != "whole_genome_reader":
                console.print(f"🧭 [dim]Router suggests: {router_decision.tool}[/dim]")
        except Exception:
            pass
        
        # Optional strict DB template mode: avoid free-form LLM Cypher by mapping question → template
        import os as _os
        if _os.getenv("AGENT_DB_TEMPLATES_ONLY", "1") == "1":
            mapped = await self._execute_from_templates_only(question)
            if mapped is not None:
                return mapped

        # Step 1: Classify the query type using model allocation (gpt-5 for biological reasoning)
        def classification_call(module):
            return module(question=question)
        
        from .dspy_signatures import QueryClassifier
        try:
            import dspy
            lm = make_lm(getattr(self.config, 'planner_model', None) or 'gpt-5-high', step="planner")
            module = dspy.Predict(QueryClassifier)
            with dspy.context(lm=lm):
                classification = classification_call(module)
        except Exception:
            # Fallback to a small model
            import dspy
            lm = dspy.LM(model="openai/gpt-4.1-mini")
            module = dspy.Predict(QueryClassifier)
            with dspy.context(lm=lm):
                classification = classification_call(module)
        
        # Step 1.5: Determine analysis type for biological context
        analysis_type = self._determine_analysis_type(question)
        
        # Step 1.6: Stage A handled spatial routing already; proceed with standard flow
        
        # classification should be set by now; if still missing, try minimal default
        if classification is None:
            classification = self._run("query_classification", QueryClassifier, question=question)
        
        console.print(f"📊 Query type: {classification.query_type}")
        console.print(f"💭 Reasoning: {classification.reasoning}")
        
        # Step 2: INTELLIGENT GENOME SELECTION - Use LLM to analyze genome selection intent
        genome_filter_required = False
        target_genome = ""
        task_context = "Global query across all genomes"
        
        try:
            # Use unified genome selector for LLM-based analysis  
            llm_selector = self.genome_selector
            
            # Check if this query needs genome selection analysis  
            if llm_selector.should_use_genome_selection(question):
                console.print("🔍 [bold yellow]Analyzing genome selection intent[/bold yellow]")
                
                selection_result = await llm_selector.analyze_genome_intent(question)
                
                if selection_result.success:
                    console.print(f"🧬 [bold green]LLM analysis:[/bold green] intent={selection_result.intent}, confidence={selection_result.confidence:.2f}")
                    console.print(f"💭 [dim]Reasoning: {selection_result.reasoning}[/dim]")
                    
                    if selection_result.intent == "specific" and selection_result.target_genomes:
                        genome_filter_required = True
                        target_genome = selection_result.target_genomes[0]  # Use first genome
                        task_context = f"Target genome: {target_genome}. LLM confidence: {selection_result.confidence:.2f}"
                        console.print(f"🎯 [bold cyan]Query will target genome:[/bold cyan] {target_genome}")
                    else:
                        console.print(f"🌐 [bold cyan]Query will analyze across all genomes[/bold cyan] (intent: {selection_result.intent})")
                else:
                    console.print(f"❌ [red]LLM genome analysis failed:[/red] {selection_result.error_message}")
                    console.print("🌐 [dim]Continuing with global analysis[/dim]")
            else:
                console.print("🌐 [dim]Using global analysis across all genomes[/dim]")
                
        except Exception as e:
            logger.error(f"LLM genome selection failed: {e}")
            console.print("⚠️ [yellow]Genome selection error, using global analysis[/yellow]")
        
        def retrieval_call(module):
            return module(
                db_schema=NEO4J_SCHEMA,
                question=question,
                query_type=classification.query_type,
                task_context=task_context,
                genome_filter_required=str(genome_filter_required),
                target_genome=target_genome,
                analysis_type=analysis_type
            )
        
        from .dspy_signatures import ContextRetriever
        try:
            import dspy
            # Use planner model for retrieval planning (or default GPT-5)
            lm = make_lm(getattr(self.config, 'planner_model', None) or 'gpt-5-high', step="planner")
            module = dspy.Predict(ContextRetriever)
            with dspy.context(lm=lm):
                retrieval_plan = retrieval_call(module)
        except Exception:
            retrieval_plan = None
        
        if retrieval_plan is None:
            logger.warning("Retrieval planning fell back to minimal default")
            retrieval_plan = self._run("context_preparation", ContextRetriever,
                db_schema=NEO4J_SCHEMA,
                question=question,
                query_type=classification.query_type,
                task_context=task_context,
                genome_filter_required=str(genome_filter_required),
                target_genome=target_genome,
                analysis_type=analysis_type
            )
        
        console.print(f"🔍 Search strategy: {retrieval_plan.search_strategy}")
        
        # Step 2.5: Validate query for comparative questions
        cypher_query = retrieval_plan.cypher_query
        validated_query = self._validate_comparative_query(question, cypher_query)
        if validated_query != cypher_query:
            logger.info("Fixed comparative query - removed inappropriate LIMIT")
            retrieval_plan.cypher_query = validated_query
        
        # Step 2.6: Validate genome filtering if required
        if genome_filter_required and self.query_validator.should_validate_for_genome(validated_query):
            validation_result = self.query_validator.validate_genome_filtering(
                validated_query, 
                genome_filter_required, 
                target_genome
            )
            
            if not validation_result.is_valid:
                console.print(f"⚠️ [yellow]Query validation failed:[/yellow] {validation_result.error_message}")
                
                if validation_result.modified_query:
                    console.print(f"🔧 [cyan]Auto-fixing query with genome filtering[/cyan]")
                    retrieval_plan.cypher_query = validation_result.modified_query
                    logger.info(f"Applied genome filtering fix: {validation_result.suggested_fix}")
                else:
                    console.print(f"💡 [blue]Suggestion:[/blue] {validation_result.suggested_fix}")
                    logger.warning(f"Could not auto-fix query: {validation_result.suggested_fix}")
            else:
                console.print(f"✅ [green]Query validation passed - genome filtering present[/green]")
        
        # Step 3: Enforce genome scoping in generated query
        scoped_query, scope_metadata = self.genome_selector.enforce_genome_scope(question, validated_query)
        
        if scope_metadata['scope_applied']:
            console.print(f"🎯 Applied genome scoping: {scope_metadata['scope_reasoning']}")
            retrieval_plan.cypher_query = scoped_query
        
        # Step 4: Execute database queries with fallback logic
        context = await self._retrieve_context_with_fallback(question, classification.query_type, retrieval_plan, scoped_query, cypher_query)
        
        # Check for TaskRepairAgent messages first
        if 'repair_message' in context.metadata:
            logger.info("TaskRepairAgent provided helpful guidance - returning repair message")
            return {
                "question": question,
                "answer": context.metadata['repair_message'],
                "confidence": "medium - error handled gracefully by TaskRepairAgent",
                "citations": "",
                "repair_info": "TaskRepairAgent provided helpful error guidance"
            }
        
        # Check for retrieval errors
        if 'retrieval_error' in context.metadata:
            error_msg = context.metadata['retrieval_error']
            logger.error(f"Context retrieval failed: {error_msg}")
            return {
                "question": question,
                "answer": f"I couldn't retrieve information to answer your question: {error_msg}",
                "confidence": "low",
                "citations": "",
                "error": error_msg
            }
        
        # Step 4: Format context and apply compression if needed
        formatted_context = self._format_context(context)
        compression_stats = None
        
        # Check if context is too large and apply compression
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(self.config.llm_model if hasattr(self.config, 'llm_model') else 'gpt-3.5-turbo')
            token_count = len(encoding.encode(formatted_context))
            
            if self.policy_engine.should_compress_context(token_count):
                logger.info(f"🗜️ Context too large ({token_count} tokens), applying compression")
                # Initialize context compressor only when needed
                compressor = ContextCompressor()
                
                # Get raw results for compression
                all_results = context.structured_data + context.semantic_data
                compressed_context, compression_stats = compressor.compress_context(all_results, target_size=25000)
                
                logger.info(f"Context compression: {compression_stats.original_count} -> {compression_stats.compressed_count} results")
                formatted_context = compressed_context
                console.print(f"🗜️ Applied compression: {compression_stats.original_count} → {compression_stats.compressed_count} results")
            else:
                logger.info(f"✅ Context size acceptable ({token_count} tokens), using full context")
                
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using full context")
        
        # Step 5: Check if external tools would be helpful and execute if so
        tool_results = await self._check_and_execute_tools(question, context, classification.query_type)
        
        # Step 6: Generate answer using model allocation (integrate tool results if available)
        final_context = formatted_context
        if tool_results:
            final_context = self._integrate_tool_results(formatted_context, tool_results)
        
        def answer_call(module):
            return module(
                question=question,
                context=final_context
            )
        
        from .dspy_signatures import GenomicAnswerer
        try:
            import dspy
            # Default to cost-effective model for intermediate answers
            lm = dspy.LM(model="openai/gpt-4.1-mini")
            module = dspy.Predict(GenomicAnswerer)
            with dspy.context(lm=lm):
                answer_result = answer_call(module)
        except Exception:
            answer_result = None
        
        if answer_result is None:
            logger.warning("Model allocation failed for answer generation, falling back to default")
            answer_result = self._run("biological_interpretation", GenomicAnswerer,
                question=question,
                context=formatted_context
            )
        
        # Return structured response
        metadata = {
            "query_type": classification.query_type,
            "search_strategy": retrieval_plan.search_strategy,
            "context_size": len(formatted_context),
            "retrieval_time": context.query_time,
            "total_results": len(context.structured_data) + len(context.semantic_data)
        }
        
        if compression_stats:
            metadata["compression_stats"] = compression_stats
        
        return {
            "question": question,
            "answer": answer_result.answer,
            "confidence": answer_result.confidence,
            "citations": answer_result.citations,
            "query_metadata": metadata
        }
    
    async def _check_and_execute_tools(self, question: str, context, query_type: str) -> Optional[Dict[str, Any]]:
        """Check if external tools would be helpful and execute them if so."""
        tool_results = {}
        
        # Check if literature search would be helpful and is available
        if (self._should_use_literature_search(question, query_type) and 
            self.policy_engine.should_use_tool("literature_search")):
            if await self._check_literature_search_availability():
                console.print("🔍 [dim]Literature search would be helpful, executing...[/dim]")
                literature_result = await self._execute_literature_search(question)
                if literature_result:
                    tool_results["literature_search"] = literature_result
            else:
                console.print("⚠️ [dim]Literature search unavailable (missing dependencies)[/dim]")
        
        # Check if code interpreter would be helpful and is available
        if (self._should_use_code_interpreter(question, context, query_type) and 
            self.policy_engine.should_use_tool("code_interpreter")):
            if await self._check_code_interpreter_availability():
                console.print("🧮 [dim]Code interpreter would be helpful, executing...[/dim]")
                code_result = await self._execute_code_interpreter(question, context)
                if code_result:
                    tool_results["code_interpreter"] = code_result
            else:
                console.print("⚠️ [dim]Code interpreter unavailable (service not running)[/dim]")
        
        return tool_results if tool_results else None
    
    async def _check_literature_search_availability(self) -> bool:
        """Check if literature search dependencies are available."""
        try:
            from Bio import Entrez
            return True
        except ImportError:
            logger.warning("Biopython not available for literature search")
            return False
    
    async def _check_code_interpreter_availability(self) -> bool:
        """Check if code interpreter service is available."""
        try:
            from .external_tools import check_code_interpreter_health
            return await check_code_interpreter_health()
        except Exception as e:
            logger.warning(f"Code interpreter health check failed: {e}")
            return False
    
    def _should_use_literature_search(self, question: str, query_type: str) -> bool:
        """Determine if literature search would be helpful."""
        question_lower = question.lower()
        
        # Look for explicit literature requests
        literature_keywords = ["recent", "literature", "research", "papers", "pubmed", "studies", "publications"]
        if any(keyword in question_lower for keyword in literature_keywords):
            return True
        
        # Look for functional questions that might benefit from literature
        functional_keywords = ["function", "role", "mechanism", "pathway", "regulation"]
        if any(keyword in question_lower for keyword in functional_keywords):
            return True
        
        return False
    
    def _should_use_code_interpreter(self, question: str, context, query_type: str) -> bool:
        """Determine if code interpreter would be helpful."""
        question_lower = question.lower()
        
        # Look for analysis/computation keywords
        analysis_keywords = ["analyze", "analysis", "distribution", "statistics", "statistical", 
                           "compare", "comparison", "pattern", "trend", "visualization", "plot", "chart"]
        if any(keyword in question_lower for keyword in analysis_keywords):
            return True
        
        # Check if we have large datasets that could benefit from analysis
        total_results = len(context.structured_data) + len(context.semantic_data)
        if total_results > 50:  # Arbitrary threshold for "large" datasets
            return True
        
        return False

    async def _execute_from_templates_only(self, question: str) -> Optional[Dict[str, Any]]:
        """Map question heuristically to a named template (strict mode) and execute.

        Returns a response dict if a template mapping is found, else None to continue legacy flow.
        """
        import re
        # Protein by id: look for protein:ID pattern
        m = re.search(r"\bprotein:([A-Za-z0-9:_\-\.]+)\b", question)
        if m:
            template = "protein_by_id"
            pid = m.group(0) if m.group(0).startswith("protein:") else f"protein:{m.group(1)}"
            slots = {"id": pid}
            db_result = await self.neo4j_processor.execute_named_template(template, slots)
            scope = self._derive_scope_from_slots(slots)
            ctx = GenomicContext(
                structured_data=db_result.results,
                semantic_data=[],
                metadata={"analysis_type": "FUNCTIONAL_ANNOTATION", "tool_used": "database_query", "template": template},
                query_time=db_result.execution_time,
                compressed_context="",
            )
            formatted = self._format_context(ctx)
            return await self._synthesize_answer(question, formatted, query_type=f"template:{template}", analysis_type="functional_annotation")

        # KO id: Kxxxxx
        m = re.search(r"\bK(\d{5})\b", question)
        if m:
            template = "proteins_with_ko"
            slots = {"ko": f"K{m.group(1)}"}
            db_result = await self.neo4j_processor.execute_named_template(template, slots)
            scope = self._derive_scope_from_slots(slots)
            ctx = GenomicContext(
                structured_data=db_result.results,
                semantic_data=[],
                metadata={"analysis_type": "FUNCTIONAL_ANNOTATION", "tool_used": "database_query", "template": template},
                query_time=db_result.execution_time,
                compressed_context="",
            )
            formatted = self._format_context(ctx)
            return await self._synthesize_answer(question, formatted, query_type=f"template:{template}", analysis_type="functional_annotation")

        # CAZy family: GH/PL/CE digits
        m = re.search(r"\b(GH|PL|CE)(\d+)\b", question, re.IGNORECASE)
        if m:
            template = "cazy_family"
            slots = {"family": f"{m.group(1).upper()}{m.group(2)}"}
            db_result = await self.neo4j_processor.execute_named_template(template, slots)
            scope = self._derive_scope_from_slots(slots)
            ctx = GenomicContext(
                structured_data=db_result.results,
                semantic_data=[],
                metadata={"analysis_type": "FUNCTIONAL_ANNOTATION", "tool_used": "database_query", "template": template},
                query_time=db_result.execution_time,
                compressed_context="",
            )
            formatted = self._format_context(ctx)
            return await self._synthesize_answer(question, formatted, query_type=f"template:{template}", analysis_type="functional_annotation")

        # Pathway map id: mapxxxxx
        m = re.search(r"\bmap(\d{5})\b", question, re.IGNORECASE)
        if m:
            template = "pathway_membership"
            slots = {"pathway": f"map{m.group(1)}"}
            db_result = await self.neo4j_processor.execute_named_template(template, slots)
            scope = self._derive_scope_from_slots(slots)
            ctx = GenomicContext(
                structured_data=db_result.results,
                semantic_data=[],
                metadata={"analysis_type": "FUNCTIONAL_ANNOTATION", "tool_used": "database_query", "template": template},
                query_time=db_result.execution_time,
                compressed_context="",
            )
            formatted = self._format_context(ctx)
            return await self._synthesize_answer(question, formatted, query_type=f"template:{template}", analysis_type="functional_annotation")

        return None
    
    async def _execute_literature_search(self, question: str) -> Optional[str]:
        """Execute literature search tool."""
        try:
            from .external_tools import literature_search
            
            # Configure search parameters from policy engine
            email = self.config.get("email", "user@example.com")  # Should be configured
            max_results = self.policy_engine.get_max_results("literature_search")
            
            # Execute search
            result = literature_search(question, email, max_results=max_results)
            # Handle envelope format
            if isinstance(result, dict):
                display = result.get("display_text") or ""
            else:
                display = str(result)
            logger.info(f"Literature search completed: {len(display)} characters")
            return display
            
        except Exception as e:
            logger.error(f"Literature search failed: {e}")
            return None
    
    async def _execute_code_interpreter(self, question: str, context) -> Optional[str]:
        """Execute code interpreter tool."""
        try:
            from .external_tools import code_interpreter_tool
            
            # Prepare data for analysis
            data_summary = self._prepare_data_for_analysis(context)
            
            # Generate analysis code based on question
            analysis_code = self._generate_analysis_code(question, data_summary)
            
            # Execute code
            result = await code_interpreter_tool(analysis_code)
            if isinstance(result, dict):
                success = bool(result.get("success"))
                display = result.get("display_text") or result.get("output") or ""
                if success:
                    logger.info("Code interpreter execution completed successfully")
                    return display
                else:
                    logger.warning(f"Code interpreter execution failed: {result.get('message', result.get('error', 'Unknown error'))}")
                    return None
            # Fallback
            return None
                
        except Exception as e:
            logger.error(f"Code interpreter execution failed: {e}")
            return None
    
    def _prepare_data_for_analysis(self, context) -> str:
        """Prepare a summary of available data for code analysis."""
        summary = []
        
        if context.structured_data:
            summary.append(f"Structured data: {len(context.structured_data)} records")
            # Add sample of data structure
            if context.structured_data:
                sample = context.structured_data[0]
                if isinstance(sample, dict):
                    summary.append(f"Sample keys: {list(sample.keys())[:5]}")
        
        if context.semantic_data:
            summary.append(f"Semantic data: {len(context.semantic_data)} records")
        
        return "; ".join(summary)
    
    def _generate_analysis_code(self, question: str, data_summary: str) -> str:
        """Generate Python code for analysis based on question."""
        # This is a simple heuristic approach - in practice, this could be more sophisticated
        question_lower = question.lower()
        
        if "distribution" in question_lower:
            return """
import pandas as pd
import matplotlib.pyplot as plt

# Create sample analysis for distribution
print("Distribution analysis would go here")
print("Data summary:", data_summary)
"""
        elif "compare" in question_lower or "comparison" in question_lower:
            return """
import pandas as pd
import numpy as np

# Create sample comparison analysis
print("Comparison analysis would go here")
print("Data summary:", data_summary)
"""
        else:
            return f"""
# General analysis
print("General analysis for question: {question[:50]}...")
print("Data available: {data_summary}")
"""

    # Fast-path finalization helper: render structure-first summary or optionally call heavy model
    def _finalize_from_locus_cards(self, cards, meta, use_heavy: bool = False) -> str:
        try:
            if not use_heavy:
                lines = []
                lines.append(f"LocusDiscovery (deterministic): {len(cards)} seeds contextualized.")
                for i, c in enumerate(cards or [], 1):
                    contig = getattr(c, 'contig_id', '')
                    genome = getattr(c, 'genome_id', '')
                    neigh = len(getattr(c, 'neighbors', []) or [])
                    seed_id = getattr(c, 'seed_protein_id', None) or '?'
                    lines.append(f"{i}. seed={seed_id} contig={contig} genome={genome} neighbors={neigh}")
                # Append concise kNN neighbor info if available
                if meta and meta.get('knn'):
                    lines.append("kNN stage executed.")
                    kr = meta.get('knn_results')
                    try:
                        # kr may be a list with one dict mapping or a dict
                        if isinstance(kr, list) and kr and isinstance(kr[0], dict):
                            kr = kr[0]
                        if isinstance(kr, dict):
                            lines.append("Nearest neighbors:")
                            shown = 0
                            for sid, items in kr.items():
                                if shown >= 5:
                                    break
                                top = items[:2] if isinstance(items, list) else []
                                nbrs = ", ".join([f"{r.get('protein_id')} (d={r.get('distance')})" for r in top])
                                lines.append(f"  - {sid}: {nbrs}")
                                shown += 1
                    except Exception:
                        pass
                # Append signature witness (boolean motifs) if present
                try:
                    if isinstance(meta, dict) and meta.get('witness'):
                        lines.append("")
                        if meta.get('signature'):
                            lines.append(f"Signature: {meta.get('signature')}")
                        lines.append("Signature Witness (boolean motifs):")
                        wit = meta.get('witness') or {}
                        shown = 0
                        for sid, w in wit.items():
                            if shown >= 5:
                                break
                            clauses = ", ".join(w.get('clauses', [])[:3]) if isinstance(w, dict) else ''
                            motifs = ", ".join(w.get('motifs_true', [])[:4]) if isinstance(w, dict) else ''
                            lines.append(f"  - {sid}: clauses=[{clauses}] motifs=[{motifs}]")
                            shown += 1
                except Exception:
                    pass
                return "\n".join(lines)
            # Optional: single heavy synthesis over structured cards
            # Deprecated heavy synthesis path removed; produce a simple structured summary
            payload = {
                "cards": [c.__dict__ if hasattr(c, '__dict__') else c for c in (cards or [])],
                "meta": meta or {},
            }
            # Promote kNN neighbors to top-level keys for summary
            try:
                if isinstance(meta, dict):
                    if 'neighbors_full' in meta:
                        payload['neighbors_full'] = meta.get('neighbors_full')
                    if 'knn_results' in meta:
                        payload['knn_picked'] = meta.get('knn_results')
                    if 'knn_stats' in meta:
                        payload['knn_stats'] = meta.get('knn_stats')
            except Exception:
                pass
            # Minimal textual summary
            import json as _json
            result_text = "LocusDiscovery summary (light):\n" + _json.dumps(payload, default=str, indent=2)[:8000]
            # Deterministic postscript: always report LanceDB stage outcome if present
            try:
                ps_lines = []
                if isinstance(payload, dict):
                    has_knn = bool(payload.get('meta', {}).get('knn')) if isinstance(payload.get('meta'), dict) else False
                    stats = payload.get('knn_stats') if 'knn_stats' in payload else payload.get('meta', {}).get('knn_stats')
                    if has_knn or isinstance(stats, dict):
                        ps_lines.append("\n\nLanceDB kNN (deterministic summary):")
                        if isinstance(stats, dict):
                            counts = stats.get('neighbors_counts') or {}
                            topk = stats.get('topk')
                            total_seeds = len(counts) if isinstance(counts, dict) else 0
                            total_neighbors = 0
                            if isinstance(counts, dict):
                                try:
                                    total_neighbors = sum(int(v or 0) for v in counts.values())
                                except Exception:
                                    total_neighbors = 0
                            ps_lines.append(f"- Queried seeds: {total_seeds}; topk: {topk}")
                            ps_lines.append(f"- Neighbors after filtering: {total_neighbors}")
                            # List per-seed neighbor counts without artificial cap
                            if isinstance(counts, dict) and counts:
                                # Stable order: sort by count desc, then seed id
                                try:
                                    for sid, cnt in sorted(counts.items(), key=lambda kv: (-int(kv[1] or 0), str(kv[0]))):
                                        ps_lines.append(f"  • {sid}: {cnt}")
                                except Exception:
                                    for sid, cnt in counts.items():
                                        ps_lines.append(f"  • {sid}: {cnt}")
                        else:
                            ps_lines.append("- kNN stage executed; no statistics available")
                    # Signature witness section
                    wit = payload.get('meta', {}).get('witness') if isinstance(payload.get('meta'), dict) else None
                    if isinstance(wit, dict) and wit:
                        ps_lines.append("\nSignature Witness (boolean motifs):")
                        if isinstance(payload.get('meta'), dict) and payload.get('meta', {}).get('signature'):
                            ps_lines.append(f"- Signature: {payload.get('meta', {}).get('signature')}")
                        shown = 0
                        for sid, w in wit.items():
                            if shown >= 5:
                                break
                            clauses = ", ".join(w.get('clauses', [])[:3]) if isinstance(w, dict) else ''
                            motifs = ", ".join(w.get('motifs_true', [])[:4]) if isinstance(w, dict) else ''
                            ps_lines.append(f"  • {sid}: clauses=[{clauses}] motifs=[{motifs}]")
                            shown += 1
                if ps_lines:
                    try:
                        ps_block = ''.join((line + '\n') for line in ps_lines)
                        result_text = (result_text + "\n" + ps_block).rstrip()
                    except Exception:
                        pass
            except Exception:
                pass
            return result_text
        except Exception as e:
            logger.warning(f"_finalize_from_locus_cards error: {e}")
            return "No loci passed deterministic gating; planner escalation not needed."
    
    def _integrate_tool_results(self, original_context: str, tool_results: Dict[str, Any]) -> str:
        """Integrate tool results into the context."""
        integrated_context = original_context
        
        # Add tool results section
        if tool_results:
            integrated_context += "\n\n=== EXTERNAL TOOL RESULTS ===\n"
            
            if "literature_search" in tool_results:
                integrated_context += f"\n--- Literature Search Results ---\n{tool_results['literature_search']}\n"
            
            if "code_interpreter" in tool_results:
                integrated_context += f"\n--- Code Analysis Results ---\n{tool_results['code_interpreter']}\n"
        
        return integrated_context
    
    async def _execute_agentic_plan(self, question: str, planning_result, selected_genome: Optional[str] = None) -> Dict[str, Any]:
        """Execute unified agent workflow with dynamic tool chaining."""
        console.print("🤖 [bold]Using unified agent execution path[/bold]")
        console.print("🔗 [dim]Agent will dynamically chain tools based on discoveries[/dim]")
        
        try:
            # Import unified agent executor
            from .agent_executor import UnifiedAgentExecutor
            
            # Create and execute unified agent
            agent = UnifiedAgentExecutor(self, note_keeper=self.note_keeper)
            
            if selected_genome:
                console.print(f"🧬 [cyan]Agent will target genome:[/cyan] {selected_genome}")
            
            # Execute agent workflow
            agent_result = await agent.execute_agent_workflow(question, selected_genome)
            
            if not agent_result.success:
                console.print("⚠️ [yellow]Agent execution failed[/yellow]")
                console.print("🔄 [dim]Falling back to traditional mode[/dim]")
                return await self._execute_traditional_query(question)
            
            console.print(f"✅ [green]Agent completed {agent_result.total_steps} steps[/green]")
            console.print(f"🛠️ Tools used: {', '.join(agent_result.tools_used)}")
            console.print(f"⏱️ Total time: {agent_result.total_execution_time:.1f}s")
            
            # Convert agent result to expected format
            return {
                "question": question,
                "answer": agent_result.final_answer,
                "confidence": agent_result.confidence,
                "citations": agent_result.citations,
                "query_metadata": {
                    "execution_mode": "unified_agent",
                    "total_steps": agent_result.total_steps,
                    "tools_used": agent_result.tools_used,
                    "execution_time": agent_result.total_execution_time,
                    "note_taking_enabled": self.note_keeper is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Agent execution failed: {str(e)}")
            console.print(f"⚠️ [yellow]Agent execution error: {str(e)}[/yellow]")
            console.print("🔄 [dim]Falling back to traditional mode[/dim]")
            return await self._execute_traditional_query(question)
    
    
    async def _retrieve_context_with_fallback(self, question: str, query_type: str, retrieval_plan, 
                                            scoped_query: str, original_query: str) -> GenomicContext:
        """
        Retrieve context with fallback logic - try scoped query first, fallback to original if no results.
        """
        # CRITICAL: Validate comparative queries BEFORE execution
        retrieval_plan.cypher_query = self._validate_comparative_query(question, retrieval_plan.cypher_query)
        
        # First try the scoped query
        logger.info("🎯 Trying scoped query first")
        context = await self._retrieve_context(query_type, retrieval_plan, question)
        
        # If we got results, return them
        if context.structured_data or context.semantic_data:
            logger.info(f"✅ Scoped query successful: {len(context.structured_data)} results")
            return context
        
        # If scoped query returned no results and we applied scoping, try original query
        if scoped_query != original_query:
            logger.info("⚠️ Scoped query returned no results, trying original unscoped query")
            
            # Restore original query and retry (also validate it)
            retrieval_plan.cypher_query = self._validate_comparative_query(question, original_query)
            fallback_context = await self._retrieve_context(query_type, retrieval_plan, question)
            
            if fallback_context.structured_data or fallback_context.semantic_data:
                logger.info(f"✅ Fallback unscoped query successful: {len(fallback_context.structured_data)} results")
                # Add metadata about fallback
                fallback_context.metadata['used_fallback'] = True
                fallback_context.metadata['fallback_reason'] = "Scoped query returned no results"
                return fallback_context
        
        logger.warning("❌ Both scoped and unscoped queries returned no results")
        return context
    
    async def _retrieve_context(self, query_type: str, retrieval_plan, question: str = "") -> GenomicContext:
        """
        Retrieve context based on query type and plan.
        This is a simplified version - the full implementation is complex.
        """
        import time
        start_time = time.time()
        
        try:
            if query_type in ["structural", "general"]:
                # Use Neo4j for structured queries
                cypher_query = retrieval_plan.cypher_query
                result = await self.neo4j_processor.process_query(cypher_query, query_type="cypher")
                
                # Check for repair messages
                repair_message = None
                if hasattr(self.neo4j_processor, 'last_repair_result') and self.neo4j_processor.last_repair_result:
                    repair_result = self.neo4j_processor.last_repair_result
                    if repair_result.success and repair_result.user_message:
                        repair_message = repair_result.user_message
                
                if result.results:
                    # Check if compression is needed based on context size
                    formatted_context = str(result.results)
                    
                    import tiktoken
                    try:
                        encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')  # Default for token counting
                        token_count = len(encoding.encode(formatted_context))
                        
                        if token_count > 30000:
                            logger.info(f"🗜️ Context too large ({token_count} tokens), applying compression")
                            # Apply context compression with smart target sizing
                            target_size = self._get_compression_target_size(retrieval_plan, result.results, question)
                            compressed_context, compression_stats = self.context_compressor.compress_context(
                                result.results, 
                                target_size=target_size, 
                                preserve_diversity=True
                            )
                            
                            metadata = result.metadata.copy()
                            metadata['compression_stats'] = compression_stats
                            
                            return GenomicContext(
                                structured_data=result.results,
                                semantic_data=[],
                                metadata=metadata,
                                query_time=time.time() - start_time,
                                compressed_context=compressed_context
                            )
                        else:
                            logger.info(f"✅ Context size acceptable ({token_count} tokens), using full results")
                            return GenomicContext(
                                structured_data=result.results,
                                semantic_data=[],
                                metadata=result.metadata,
                                query_time=time.time() - start_time
                            )
                    except Exception as e:
                        logger.warning(f"Token counting failed: {e}, using full results")
                        return GenomicContext(
                            structured_data=result.results,
                            semantic_data=[],
                            metadata=result.metadata,
                            query_time=time.time() - start_time
                        )
                elif repair_message:
                    return GenomicContext(
                        structured_data=[],
                        semantic_data=[],
                        metadata={"repair_message": repair_message},
                        query_time=time.time() - start_time
                    )
                else:
                    return GenomicContext(
                        structured_data=[],
                        semantic_data=[],
                        metadata={"retrieval_error": "No results found"},
                        query_time=time.time() - start_time
                    )
            
            else:
                # For semantic/hybrid queries, use hybrid processor
                result = await self.hybrid_processor.process_query(retrieval_plan.cypher_query)
                
                if result.results:
                    combined_data = result.results[0] if result.results else {}
                    return GenomicContext(
                        structured_data=combined_data.get("structured_data", []),
                        semantic_data=combined_data.get("semantic_data", []),
                        metadata=result.metadata,
                        query_time=time.time() - start_time
                    )
                else:
                    return GenomicContext(
                        structured_data=[],
                        semantic_data=[],
                        metadata={"retrieval_error": "No results found"},
                        query_time=time.time() - start_time
                    )
                    
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return GenomicContext(
                structured_data=[],
                semantic_data=[],
                metadata={"retrieval_error": str(e)},
                query_time=time.time() - start_time
            )
    
    def _validate_comparative_query(self, question: str, cypher_query: str) -> str:
        """
        Validate and fix comparative queries that incorrectly use LIMIT 1.
        
        Args:
            question: Original user question
            cypher_query: Generated Cypher query to validate
            
        Returns:
            Validated (and potentially fixed) Cypher query
        """
        import re
        
        # Define patterns that indicate comparative questions requiring ALL results
        comparative_patterns = [
            r"which\s+(?:of\s+the\s+)?genomes?\s+(?:have|has|contain)",  # "which (of the) genomes have"
            r"for\s+each\s+genome",  # "for each genome"
            r"compare\s+.*?\s+(?:across\s+)?(?:all\s+)?genomes?",  # "compare X across genomes"
            r"(?:most|least|highest|lowest|best|worst)\s+(?:among|across|between)\s+genomes?",  # "most among genomes"
            r"which\s+.*?genomes?\s+.*?(?:has|have)\s+.*?(?:most|least|highest|lowest)",  # "which ... genomes ... has ... most"
            r"how\s+(?:many|much).+(?:across|between|among)\s+genomes?",  # "how many across genomes"
            r"distribution\s+(?:across|among|between)\s+genomes?",  # "distribution across genomes"
            r"all\s+genomes?.+(?:count|number|amount)",  # "all genomes count"
            r"rank\s+genomes?\s+by",  # "rank genomes by"
            r"(?:count|number|total).+per\s+genome"  # "count per genome"
        ]
        
        # Check if question contains comparative patterns
        question_lower = question.lower()
        is_comparative = any(re.search(pattern, question_lower) for pattern in comparative_patterns)
        
        if not is_comparative:
            return cypher_query
        
        # Check if query has LIMIT 1 (problematic for comparative queries)
        if re.search(r'\bLIMIT\s+1\b', cypher_query, re.IGNORECASE):
            logger.warning(f"Detected LIMIT 1 in comparative query: {question}")
            
            # Remove LIMIT 1 but keep other LIMIT values
            fixed_query = re.sub(r'\bLIMIT\s+1\b', '', cypher_query, flags=re.IGNORECASE)
            
            # Clean up any trailing whitespace or newlines
            fixed_query = fixed_query.strip()
            
            logger.info(f"Fixed comparative query by removing LIMIT 1")
            return fixed_query
        
        return cypher_query

    def _format_context(self, context: GenomicContext) -> str:
        """Format genomic context for LLM processing."""
        formatted_parts = []
        
        # Add structured data
        if context.structured_data:
            formatted_parts.append(f"=== STRUCTURED DATA ({len(context.structured_data)} results) ===")
            for i, item in enumerate(context.structured_data[:50]):  # Limit for context size
                formatted_parts.append(f"Result {i+1}: {item}")
            
            if len(context.structured_data) > 50:
                formatted_parts.append(f"... and {len(context.structured_data) - 50} more results")
        
        # Add semantic data
        if context.semantic_data:
            formatted_parts.append(f"\\n=== SEMANTIC DATA ({len(context.semantic_data)} results) ===")
            for i, item in enumerate(context.semantic_data[:20]):  # Limit for context size
                formatted_parts.append(f"Similar {i+1}: {item}")
            
            if len(context.semantic_data) > 20:
                formatted_parts.append(f"... and {len(context.semantic_data) - 20} more results")
        
        # Add metadata
        if context.metadata:
            formatted_parts.append(f"\\n=== METADATA ===")
            for key, value in context.metadata.items():
                if key not in ['retrieval_error', 'repair_message']:  # Skip error fields
                    formatted_parts.append(f"{key}: {value}")
        
        return "\\n".join(formatted_parts)
    
    def _format_spatial_context(self, context: GenomicContext) -> str:
        """Format spatial genomic context for prophage/operon analysis."""
        formatted_parts = []
        
        # Add header for spatial genomic data
        if context.structured_data:
            formatted_parts.append(f"=== SPATIAL GENOMIC DATA ({len(context.structured_data)} genomic regions) ===")
            formatted_parts.append("Full genomic context with gene coordinates, annotations, and spatial organization:")
            formatted_parts.append("")
            
            # Format each genomic region with spatial information
            for i, region in enumerate(context.structured_data):
                formatted_parts.append(f"GENOMIC REGION {i+1}:")
                formatted_parts.append(str(region))
                formatted_parts.append("")
            
        # Add analysis metadata
        if context.metadata:
            formatted_parts.append("=== ANALYSIS METADATA ===")
            for key, value in context.metadata.items():
                formatted_parts.append(f"{key}: {value}")
        
        return "\\n".join(formatted_parts)

    def _derive_scope_from_slots(self, slots: Dict[str, Any]) -> Optional[GenomeScope]:
        """Derive an immutable GenomeScope from DB template slots when possible."""
        try:
            if not isinstance(slots, dict):
                return None
            gid = slots.get("genome_id")
            if isinstance(gid, str) and gid:
                return GenomeScope(genome_id=str(gid), contig_ids=tuple(), coordinate_window=(0, 0))
            contig = slots.get("contig")
            if isinstance(contig, str) and contig:
                return GenomeScope(genome_id="*", contig_ids=(str(contig),), coordinate_window=(0, 0))
        except Exception:
            pass
        return None
    
    async def _synthesize_answer(self, question: str, formatted_context: str, query_type: str, analysis_type: str) -> Dict[str, Any]:
        """Synthesize answer from formatted context using manual model selection (no allocator)."""
        try:
            from .dspy_signatures import GenomicAnswerer
            import dspy
            lm = dspy.LM(model="openai/gpt-4.1-mini")
            module = dspy.Predict(GenomicAnswerer)
            with dspy.context(lm=lm):
                answer_result = module(
                    question=question,
                    context=formatted_context,
                    analysis_type=analysis_type,
                )
            
            # Fallback through _run if needed
            if answer_result is None:
                answer_result = self._run("biological_interpretation", GenomicAnswerer,
                    question=question,
                    context=formatted_context,
                    analysis_type=analysis_type
                )
            
            # Return structured response
            return {
                "question": question,
                "answer": answer_result.answer,
                "confidence": answer_result.confidence,
                "citations": answer_result.citations,
                "query_metadata": {
                    "query_type": query_type,
                    "analysis_type": analysis_type,
                    "search_strategy": "spatial_genomic_tool",
                    "context_size": len(formatted_context),
                    "tool_used": "whole_genome_reader"
                }
            }
            
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            return {
                "question": question,
                "answer": f"I encountered an error while analyzing the spatial genomic data: {str(e)}",
                "confidence": "low",
                "citations": "",
                "error": str(e),
                "query_metadata": {
                    "query_type": query_type,
                    "analysis_type": analysis_type,
                    "error": "synthesis_failed"
                }
            }
    
    
    def close(self):
        """Close all processor connections."""
        try:
            if hasattr(self.neo4j_processor, 'close'):
                self.neo4j_processor.close()
            if hasattr(self.lancedb_processor, 'close'):
                self.lancedb_processor.close()
            if hasattr(self.hybrid_processor, 'close'):
                self.hybrid_processor.close()
            logger.info("🔌 GenomicRAG connections closed")
        except Exception as e:
            logger.error(f"❌ Error closing connections: {e}")

    # Legacy methods for backward compatibility
    async def ask_agentic(self, question: str, **kwargs) -> str:
        """Legacy method that returns string instead of dict."""
        result = await self.ask(question)
        return result.get('answer', 'No answer generated')
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all system components."""
        health = self.health_check()
        return {
            f"{component}_processor": "available" if status else "unavailable" 
            for component, status in health.items()
        }
    
    def _determine_analysis_type(self, question: str) -> str:
        """
        Determine the analysis type based on question content for biological context.
        
        Args:
            question: User's question
            
        Returns:
            Analysis type: spatial_genomic, functional_annotation, or comprehensive_discovery
        """
        question_lower = question.lower()
        
        # Spatial/genomic organization patterns
        spatial_patterns = [
            "operon", "operons", "gene cluster", "genomic region", "prophage", 
            "phage", "spatial", "neighborhood", "proximity", "adjacent",
            "genomic context", "gene organization", "cluster", "loci"
        ]
        
        # Functional annotation patterns  
        functional_patterns = [
            "function", "functional", "activity", "pathway", "metabolic",
            "enzyme", "protein family", "domain", "kegg", "pfam", "annotation"
        ]
        
        # Discovery/exploration patterns
        discovery_patterns = [
            "find", "discover", "explore", "look through", "see what", 
            "interesting", "novel", "unusual", "stands out", "browse"
        ]
        
        if any(pattern in question_lower for pattern in spatial_patterns):
            logger.info(f"🧬 Analysis type: SPATIAL_GENOMIC (detected patterns for spatial organization)")
            return "spatial_genomic"
        elif any(pattern in question_lower for pattern in functional_patterns):
            logger.info(f"🔬 Analysis type: FUNCTIONAL_ANNOTATION (detected patterns for functional analysis)")
            return "functional_annotation"
        elif any(pattern in question_lower for pattern in discovery_patterns):
            logger.info(f"🌐 Analysis type: COMPREHENSIVE_DISCOVERY (detected patterns for exploration)")
            return "comprehensive_discovery"
        else:
            # Default to functional annotation for general queries
            logger.info(f"📊 Analysis type: FUNCTIONAL_ANNOTATION (default for general queries)")
            return "functional_annotation"
