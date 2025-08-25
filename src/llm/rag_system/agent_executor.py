"""
Unified Agent Executor for Dynamic Tool Chaining.

Replaces the rigid task-based system (TaskPlanParser + TaskExecutor + TaskGraph)
with a flexible agent that dynamically chooses tools based on intermediate results.

This enables natural biological exploration where the LLM can:
1. Start with any tool (database query, spatial analysis, etc.)
2. Examine results and dynamically choose the next tool  
3. Chain tools naturally based on what it discovers
4. Synthesize when exploration is complete
"""

import asyncio
import json
import logging
import time
import ast
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import os

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

from .external_tools import AVAILABLE_TOOLS, TOOL_CAPABILITIES
from .memory import NoteKeeper, get_model_allocator
from .memory.tool_result_cache import ToolResultCache
from .utils import safe_log_data
from .fsm.action_graph import FSM, State
from ..kg.cypher_templates.registry import SPECS  # type: ignore
import json as _json

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    """Represents a single step in the agent's execution."""
    step_number: int
    tool_name: Optional[str]  # None for database queries
    tool_parameters: Dict[str, Any]
    reasoning: str
    result: Any
    execution_time: float
    success: bool
    error: Optional[str] = None


@dataclass  
class AgentExecutionResult:
    """Result from agent execution with all steps and final synthesis."""
    question: str
    success: bool
    steps: List[AgentStep]
    final_answer: str
    confidence: str
    citations: str
    total_execution_time: float
    total_steps: int
    tools_used: List[str]
    error: Optional[str] = None


class AgentDecisionMaker(dspy.Signature if DSPY_AVAILABLE else object):
    """
    Intelligent agent that decides the next action based on current context.

    The agent examines previous results and decides whether to:
    1. Use another tool to gather more information
    2. Synthesize the current results into a final answer

    TOOL SELECTION CRITERIA:

    Use 'database_query' for:
    - Initial exploration or specific lookups
    - Finding proteins, genes, pathways, or annotations
    - Counting or quantifying biological features
    - Answering questions with direct database matches

    Use 'whole_genome_reader' for:
    - Spatial genomic analysis (prophage, operons, gene clusters)
    - Global discovery across all genomes
    - Questions requiring gene coordinate order
    - Neighborhood context and genomic organization

    Use 'code_interpreter' for:
    - Statistical analysis of collected data
    - Pattern detection in large datasets
    - Visualization and plotting
    - Quantitative assessments after data collection

    Use 'literature_search' for:
    - Recent research validation
    - Novel finding verification
    - Background information on discoveries

    Use 'synthesize' action when:
    - Statistical analysis with comprehensive findings is complete (e.g., detailed protein/gene analysis with quantitative results)
    - Questions asking for comparison/distribution have been answered with descriptive statistics
    - Code interpreter has provided complete statistical breakdown with means, std deviations, and patterns
    - Database queries have retrieved necessary data AND code interpreter has analyzed it thoroughly
    - Ready to provide comprehensive answer based on completed analysis

    Consider dataset scale (data_profile) and available budgets (budget_state). Prefer lower-cost actions that still make progress. If an action may be expensive, you may mark it as requiring approval.
    """

    user_question = dspy.InputField(desc="Original user question with biological context")
    previous_steps = dspy.InputField(desc="Summary of previous tool executions and their results")
    available_tools = dspy.InputField(desc="Available tools with their capabilities and decision criteria")
    current_findings = dspy.InputField(desc="Current biological findings and data collected so far")
    # Optional, non-binding hints (kept generic; safe defaults provided by caller)
    data_profile = dspy.InputField(desc="Summarized data scale/complexity (e.g., contigs/genes/estimated chunks); may be empty")
    policy_hints = dspy.InputField(desc="Generic hints such as 'cheap-first', 'templates-only'; may be empty")
    budget_state = dspy.InputField(desc="Token/time budget context; may be empty")
    db_templates_catalog = dspy.InputField(desc="JSON catalog of available DB templates and slots; may be empty")
    tool_costs = dspy.InputField(desc="JSON map of tool cost tags; may be empty")
    functional_signatures_catalog = dspy.InputField(desc="JSON of optional functional signatures (e.g., PFAM/KOFAM); may be empty")
    progress_state = dspy.InputField(desc="JSON progress indicators: candidates collected, loci built, last_row_count, zero_result_streak")

    next_action = dspy.OutputField(desc="Next action: tool name from available_tools or 'synthesize' to finish")
    tool_parameters = dspy.OutputField(desc="JSON parameters for the selected tool (empty object {} for synthesize)")
    biological_reasoning = dspy.OutputField(desc="Detailed biological reasoning for this decision based on current findings")
    confidence = dspy.OutputField(desc="Confidence level 0.0-1.0 in this decision")
    exploration_complete = dspy.OutputField(desc="true if comprehensive analysis is complete (statistical analysis done, patterns identified, question fully answered), false if more tools needed")
    # Optional, non-binding outputs
    requires_approval = dspy.OutputField(desc="true if the chosen action may require user approval; optional and advisory")
    alternatives_json = dspy.OutputField(desc="Optional JSON array of alternative actions with brief justifications and cost/benefit notes")


class DecisionParamRepair(dspy.Signature if DSPY_AVAILABLE else object):
    """Repair invalid or missing tool parameters for a chosen action.

    Provide ONLY a JSON object that matches the provided schema. Use the
    db_templates_catalog to select valid template names when repairing
    database_query parameters.
    """

    instruction = dspy.InputField(desc="Short instruction about the repair task")
    tool_name = dspy.InputField(desc="Chosen tool name")
    user_question = dspy.InputField(desc="Original user question")
    bad = dspy.InputField(desc="Bad or missing tool parameters (JSON or text)")
    db_templates_catalog = dspy.InputField(desc="JSON catalog of available DB templates and slots")
    param_schema_json = dspy.InputField(desc="JSON schema for the expected parameters")
    json = dspy.OutputField(desc="Return ONLY a JSON object matching the schema")


class AnalysisCodeGenerator(dspy.Signature if DSPY_AVAILABLE else object):
    """Generate Python analysis code with clean data interface.
    
    The generated code will have access to:
    - step_data: Dict[str, Any] - Raw results from each previous step
    - dataframes: Dict[str, pd.DataFrame] - Ready-to-use DataFrames indexed by step number
    - metadata: Dict with step count, tools used, etc.
    
    Access patterns:
    - dataframes[1] or dataframes['step_1'] - DataFrame from step 1
    - step_data['step_1']['summary'] - Step metadata
    - len(dataframes) - Number of available datasets
    
    Expected output:
    - Store results in 'analysis_results' variable as dict
    - Include 'summary', 'key_findings', 'statistics' keys
    - Print important findings for user visibility
    """
    
    user_question = dspy.InputField(desc="User's original biological analysis question")
    available_data_summary = dspy.InputField(desc="Summary of available datasets: step numbers, data types, row counts, and key columns") 
    analysis_objective = dspy.InputField(desc="What specific analysis or computation is needed to answer the question")
    
    analysis_code = dspy.OutputField(desc="Python code using dataframes[step_num] to access data, performing the analysis, and storing results in analysis_results dict")


class UnifiedAgentExecutor:
    """
    Unified agent executor that dynamically chains tools based on intermediate results.
    
    Replaces the complex TaskGraph + TaskExecutor system with a simple agent loop:
    1. Agent examines current state and chooses next tool
    2. Tool executes and returns results  
    3. Agent examines results and decides next action
    4. Repeat until agent decides to synthesize final answer
    """
    
    def __init__(self, rag_system, note_keeper: Optional[NoteKeeper] = None):
        """
        Initialize unified agent executor.
        
        Args:
            rag_system: GenomicRAG instance for access to processors and tools
            note_keeper: Optional NoteKeeper for session persistence
        """
        self.rag_system = rag_system
        self.note_keeper = note_keeper
        
        # Initialize model allocator for agent decisions
        self.model_allocator = get_model_allocator()
        
        # Initialize tool result cache for reference-based storage
        if note_keeper and hasattr(note_keeper, 'session_path'):
            self.tool_cache = ToolResultCache(str(note_keeper.session_path))
            logger.info("🗂️ Tool result caching enabled for reference-based storage")
        else:
            self.tool_cache = None
            logger.info("⚠️ Tool result caching disabled - no session directory available")
        
        # Tool registry - maps tool names to execution methods
        self.tools = {
            "database_query": self._execute_database_query,
            "whole_genome_reader": self._execute_whole_genome_reader,
            "neighborhood_extractor": self._execute_neighborhood_extractor,
            "annotation_discovery": self._execute_annotation_discovery,
            "code_interpreter": self._execute_code_interpreter,
            "literature_search": self._execute_literature_search,
            "lancedb_knn": self._execute_lancedb_knn,
            "report_synthesis": self._execute_report_synthesis,
        }
        # Dedup cache for database queries (template+slots signature → envelope)
        self._db_dedup_cache: Dict[str, Any] = {}
        
        # Execution state
        self.max_steps = 8  # Prevent infinite loops
        self.step_timeout = 300  # 5 minutes per step
        self.guidance_frequency = 3  # Run guidance synthesis every N steps
        
        # Initialize data collection for code interpreter
        self._previous_step_data = {}
        # Progress tracking for decisions (generic, non-hardcoded)
        self._progress = {
            "distinct_protein_ids": set(),
            "loci_built": 0,
            "last_row_count": 0,
            "zero_result_streak": 0,
            "last_query_signature": None,
        }
        
        # Code generator will use model allocation system
        # (no need to initialize here, will use model_allocator.create_context_managed_call)
        
        logger.info("🤖 UnifiedAgentExecutor initialized - dynamic tool chaining enabled")
        # Initialize FSM for typed transitions
        self._fsm = FSM()
        # Obligation ledger for deterministic scheduling (optional)
        self.obligation_ledger = None
        # Router config debug
        try:
            from ..options.router import USE_GRAMMAR_ROUTER as _UGR
            logger.info(f"ROUTER_CONFIG: USE_GRAMMAR_ROUTER={_UGR}")
        except Exception:
            pass
    
    async def execute_agent_workflow(self, question: str, selected_genome: Optional[str] = None) -> AgentExecutionResult:
        """Entrypoint for user questions.
        
        Guarded Macro Fast Path (MFP) runs first if enabled and intent matches;
        otherwise falls back to the FSM-governed reactive planner.
        """
        try:
            # Stash current question for downstream guards/fallbacks
            try:
                self.current_question = question
            except Exception:
                pass
            # Fast Path preflight (deterministic, zero LLM calls)
            if getattr(self.rag_system.config, "FAST_PATH_ENABLED", True):
                fp_result = await self._try_fast_path_locus_discovery(question)
                if fp_result is not None:
                    return fp_result
        except Exception as e:
            logger.warning(f"Fast Path preflight failed or skipped: {e}")

        # Enforce FSM-governed workflow to avoid oscillations
        return await self._execute_agent_workflow_fsm(question, selected_genome)

    async def _try_fast_path_locus_discovery(self, question: str) -> Optional[AgentExecutionResult]:
        """Attempt Macro Fast Path locus discovery if the intent matches.

        Returns AgentExecutionResult on success, or None to fall back.
        """
        try:
            from ..options.router import parse_macro_intent
            intent = parse_macro_intent(question)
            # Two-stage fallback: if grammar fails to parse, try Canonicalizer → DSL → Lark
            if not intent:
                try:
                    from ..intent.canonicalizer import canonicalize
                    from ..intent.dsl_renderer import render_to_dsl
                    from ..options.intent_grammar import parse_intent as _parse_dsl
                    canon, raw = canonicalize(question, self.note_keeper, self.model_allocator)
                    dsl = render_to_dsl(canon)
                    # Persist DSL string
                    try:
                        if self.note_keeper and hasattr(self.note_keeper, 'session_path'):
                            dbg = self.note_keeper.session_path / "debug_data_flow"
                            dbg.mkdir(exist_ok=True)
                            with open(dbg / "canonicalizer.dsl.txt", 'w') as f:
                                f.write(dsl)
                    except Exception:
                        pass
                    intent = _parse_dsl(dsl)
                    if not intent:
                        logger.info("CANONICAL_FALLBACK_PARSE_FAIL: DSL did not parse; escalating")
                        return None
                except Exception as e2:
                    logger.info(f"CANONICAL_FALLBACK_FAIL: {e2}")
                    return None
            if not intent or intent.option_name != "LocusDiscovery":
                logger.info("MFP_PRECHECK: intent_unparsed_or_not_locus")
                return None
        except Exception:
            logger.info("MFP_PRECHECK: intent_exception")
            return None

        # Build deterministic option with file-based template runner
        try:
            from ..options.template_runner import FileCypherRunner
            from ..options.locus_discovery import LocusDiscoveryOption, LocusCard
        except Exception as e:
            logger.warning(f"Fast Path modules unavailable: {e}")
            return None

        from ..options.obligations import ObligationLedger
        db_runner = FileCypherRunner(self.rag_system.neo4j_processor.driver)
        # Provide lancedb only if present and configured; otherwise keep None to avoid imports
        ldb = None
        option = LocusDiscoveryOption(
            db=db_runner,
            lancedb=ldb,
            config={
                "SKEPTIC_ENABLED": getattr(self.rag_system.config, "SKEPTIC_ENABLED", True),
                "min_contig_len": 1500,
            }
        )
        ledger = ObligationLedger.from_intent(intent)
        logger.info(
            "MFP_INTENT: marker=%s N=%s k=%s ldb_required=%s nn=%s",
            intent.marker,
            getattr(intent.N, "value", None),
            getattr(intent.flank, "value", None),
            ledger.state.get("lancedb_knn", {}).get("required"),
            ledger.state.get("lancedb_knn", {}).get("nn"),
        )

        start = time.time()
        k = int(intent.flank.value or 4)
        required_knn = getattr(intent.obligations, "lancedb_knn", None)
        nn = int(required_knn.nn) if (required_knn and required_knn.required and required_knn.nn) else 0
        cards, meta = option.run(
            marker=intent.marker,
            N=int(intent.N.value or 5),
            k=k,
            nn=nn,
        )
        elapsed = time.time() - start

        # Mark obligations
        ledger.mark_done("seed_selection")
        ledger.mark_done("neighborhoods")
        # Add marker context to meta for synthesis
        try:
            if isinstance(meta, dict):
                meta.setdefault('marker', intent.marker)
        except Exception:
            pass
        # If LanceDB required, satisfy via first-class tool wrapper for cache/synth compatibility
        if nn > 0 and getattr(self.rag_system, 'lancedb_processor', None):
            try:
                ids = [c.seed_protein_id for c in cards if getattr(c, 'seed_protein_id', None)]
                if ids:
                    logger.info("TOOL_INVOCATION: lancedb_knn (fast_path)")
                    logger.info(f"LDB_KNN_PARAMS: seeds={len(ids)} nn={nn}")
                    ex_ns = (required_knn.exclude_namespace or 'pfam') if required_knn else 'pfam'
                    ex_markers = (required_knn.exclude_markers or []) if required_knn else []
                    env = await self._execute_lancedb_knn({
                        'seed_ids': ids,
                        'nn': nn,
                        'topk': max(10, 10 * nn),
                        'distance': getattr(required_knn, 'distance', 'cosine'),
                        'exclude_namespace': ex_ns,
                        'exclude_markers': ex_markers,
                    })
                    # Populate meta in a shape the synthesizer recognizes
                    sd = env.get('structured_data') if isinstance(env, dict) else None
                    neighbors_full = None
                    picked = None
                    stats = None
                    try:
                        if isinstance(sd, list) and len(sd) >= 2 and isinstance(sd[1], dict):
                            neighbors_full = sd[1].get('neighbors_full')
                        if isinstance(sd, list) and len(sd) >= 1 and isinstance(sd[0], dict):
                            picked = (sd[0].get('picked') or {})
                            stats = sd[0].get('stats')
                    except Exception:
                        neighbors_full = None
                    meta['knn'] = True
                    # Persist results/stats even if empty so synthesis can report zero matches
                    if picked is not None:
                        meta['knn_results'] = picked
                    if neighbors_full is not None:
                        meta['neighbors_full'] = neighbors_full
                    if stats is not None:
                        meta['knn_stats'] = stats
                    ledger.mark_done('lancedb_knn')
            except Exception as e:
                logger.info(f"Fast Path: LanceDB stage skipped due to error: {e}")

        if meta.get("escalate") or ledger.unmet():
            logger.info(
                f"Fast Path: escalating to planner; meta={meta}, unmet={list(ledger.unmet().keys())}"
            )
            return None

        # Synthesize final answer (FSM-parity heavy summary)
        try:
            final_answer = self.rag_system._finalize_from_locus_cards(cards, meta, use_heavy=True)
        except Exception:
            final_answer = self._render_locus_cards_summary(cards)

        # Emit final MFP_RESULT after kNN/meta updates
        try:
            logger.info(
                "MFP_RESULT: seeds=%d escalate=%s knn_present=%s",
                len(cards or []),
                meta.get("escalate"),
                bool(meta.get("knn")),
            )
        except Exception:
            pass

        step = AgentStep(
            step_number=1,
            tool_name="fast_path_locus_discovery",
            tool_parameters={
                "marker": intent.marker,
                "N": int(intent.N.value or 5),
                "k": k,
                "nn": nn,
            },
            reasoning="Deterministic MFP: seeds → EVI gate → batched neighborhoods → persist loci",
            result={
                "count": len(cards),
                "meta": meta,
                "seeds": [getattr(c, 'seed_protein_id', None) for c in cards],
                "contigs": list({c.contig_id for c in cards}),
            },
            execution_time=elapsed,
            success=True,
            error=None,
        )

        return AgentExecutionResult(
            question=question,
            success=True,
            steps=[step],
            final_answer=final_answer,
            confidence="high",
            citations="",
            total_execution_time=elapsed,
            total_steps=1,
            tools_used=["fast_path_locus_discovery"],
            error=None,
        )

    def _render_locus_cards_summary(self, cards: List[Any]) -> str:
        if not cards:
            return "No loci passed deterministic gating; planner escalation not needed."
        lines = []
        lines.append(f"LocusDiscovery (deterministic): {len(cards)} seeds contextualized.")
        for i, c in enumerate(cards, 1):
            contig = getattr(c, "contig_id", "")
            genome = getattr(c, "genome_id", "")
            neigh = len(getattr(c, "neighbors", []) or [])
            seed = getattr(c, 'seed_protein_id', None)
            lines.append(f"{i}. seed={seed} contig={contig} genome={genome} neighbors={neigh}")
        return "\n".join(lines)

    async def _execute_agent_workflow_fsm(self, question: str, selected_genome: Optional[str] = None) -> AgentExecutionResult:
        """FSM-governed agent workflow with typed transitions and strict cycle control."""
        self.current_user_question = question
        # Stash selected genome for decision context (scale-aware hints)
        self.selected_genome = selected_genome
        logger.info(f"🚀 Starting FSM agent workflow for: {question[:100]}...")
        start_time = time.time()
        steps: List[AgentStep] = []
        current_findings = f"Analyzing question: {question}"
        tools_used: List[str] = []
        # Reset FSM state
        self._fsm.state = State.PLAN
        if self.note_keeper:
            self.note_keeper.set_session_context(question, "unified_agent_fsm")
        # Initialize obligation ledger early for scheduling if applicable
        try:
            from ..options.router import parse_macro_intent
            from ..options.obligations import ObligationLedger
            intent0 = parse_macro_intent(question)
            if intent0:
                self.obligation_ledger = ObligationLedger.from_intent(intent0)
                # Fail fast if tool is required but missing
                if self.obligation_ledger.state.get("lancedb_knn", {}).get("required"):
                    from .external_tools import AVAILABLE_TOOLS
                    if "lancedb_knn" not in AVAILABLE_TOOLS or AVAILABLE_TOOLS.get("lancedb_knn") is None:
                        raise RuntimeError("CONFIG_ERROR: obligation 'lancedb_knn' requires tool 'lancedb_knn' but it is not registered.")
        except Exception as e:
            logger.info(f"Obligation ledger init skipped: {e}")
        try:
            for step_number in range(1, self.max_steps + 1):
                logger.info(f"🔄 FSM Agent step {step_number}/{self.max_steps}")
                # Decision must occur at DECIDE or PLAN
                if self._fsm.state in (State.DB, State.SIM, State.GENOME):
                    # Should not happen at start of loop; force ACCUM->DECIDE
                    try:
                        self._fsm.transition(State.ACCUM)
                        self._fsm.transition(State.DECIDE)
                    except Exception as fsm_err:
                        logger.error(f"FSM correction before decision: {fsm_err}")

                decision = await self._make_agent_decision(
                    question=question,
                    steps=steps,
                    current_findings=current_findings
                )

                if decision.exploration_complete and decision.next_action == "synthesize":
                    try:
                        if self._fsm.state == State.ACCUM:
                            self._fsm.transition(State.DECIDE)
                        if self._fsm.state in (State.PLAN, State.DECIDE):
                            self._fsm.transition(State.SYN)
                    except Exception as fsm_err:
                        logger.error(f"FSM synthesize transition warning: {fsm_err}")
                    break

                # DECIDE -> PLAN
                try:
                    if self._fsm.state == State.DECIDE:
                        self._fsm.transition(State.PLAN)
                except Exception as fsm_err:
                    logger.error(f"FSM DECIDE->PLAN warning: {fsm_err}")

                # PLAN -> tool state
                try:
                    if decision.next_action == "database_query" and self._fsm.state == State.PLAN:
                        self._fsm.transition(State.DB)
                    elif decision.next_action == "similarity_search" and self._fsm.state == State.PLAN:
                        self._fsm.transition(State.SIM)
                    elif decision.next_action == "whole_genome_reader" and self._fsm.state == State.PLAN:
                        self._fsm.transition(State.GENOME)
                except Exception as fsm_err:
                    logger.error(f"FSM PLAN->tool warning: {fsm_err}")

                # Execute step
                step_result = await self._execute_agent_step(
                    step_number=step_number,
                    tool_name=decision.next_action,
                    tool_parameters=decision.tool_parameters,
                    reasoning=decision.biological_reasoning,
                    selected_genome=selected_genome
                )
                steps.append(step_result)
                self._update_previous_step_data(steps)
                self._update_progress(step_result)
                self._save_task_debug_data(step_result, step_number)
                # Optional fail-fast: abort agent workflow on first tool error (e.g., template compile failure)
                if (not step_result.success) and getattr(self.rag_system.config, 'FAIL_FAST_ON_TOOL_ERROR', False):
                    total_time = time.time() - start_time
                    logger.error(f"FAIL_FAST: Aborting on step {step_number} error: {step_result.error}")
                    return AgentExecutionResult(
                        question=question,
                        success=False,
                        steps=steps,
                        final_answer=f"Aborted due to tool error at step {step_number}: {step_result.error}",
                        confidence="low",
                        citations="",
                        total_execution_time=total_time,
                        total_steps=len(steps),
                        tools_used=tools_used,
                        error=step_result.error,
                    )
                if self.note_keeper and step_result.success:
                    self._save_agent_step_as_note(step_result, question)
                if step_result.tool_name and step_result.tool_name not in tools_used:
                    tools_used.append(step_result.tool_name)
                elif step_result.tool_name is None and "database_query" not in tools_used:
                    tools_used.append("database_query")
                if step_result.success and step_result.result:
                    result_summary = self._summarize_step_result(step_result)
                    current_findings += f"\n\nStep {step_number} findings: {result_summary}"
                else:
                    current_findings += f"\n\nStep {step_number} failed: {step_result.error or 'Unknown error'}"
                logger.info(f"✅ FSM Step {step_number} completed: {step_result.tool_name or 'database_query'}")

                # Post-exec transitions
                try:
                    if self._fsm.state in (State.DB, State.SIM, State.GENOME):
                        self._fsm.transition(State.ACCUM)
                    if self._fsm.state == State.ACCUM:
                        self._fsm.transition(State.DECIDE)
                except Exception as fsm_err:
                    logger.error(f"FSM post-exec warning: {fsm_err}")

            # Final synthesis (pre-synthesis obligation gate)
            # Build ledger opportunistically once before finalization
            if self.obligation_ledger is None:
                try:
                    from ..options.router import parse_macro_intent
                    from ..options.obligations import ObligationLedger
                    intent = parse_macro_intent(question)
                    if intent:
                        self.obligation_ledger = ObligationLedger.from_intent(intent)
                except Exception:
                    self.obligation_ledger = None
            if self.obligation_ledger is None:
                logger.info("FINALIZATION_GATE: no_obligation_ledger_present")
            if self.obligation_ledger is not None and self.obligation_ledger.unmet():
                unmet = list(self.obligation_ledger.unmet().keys())
                logger.error(f"FINALIZATION_BLOCKED: unmet obligations: {unmet}")
                total_time = time.time() - start_time
                return AgentExecutionResult(
                    question=question,
                    success=False,
                    steps=steps,
                    final_answer=f"Cannot finalize; unmet obligations: {unmet}",
                    confidence="low",
                    citations="",
                    total_execution_time=total_time,
                    total_steps=len(steps),
                    tools_used=tools_used,
                    error="unmet_obligations",
                )

            logger.info("📊 Running final reporting synthesis with all notes (FSM mode)")
            final_answer, confidence, citations = await self._run_reporting_synthesis(
                question=question,
                steps=steps,
                current_findings=current_findings
            )
            total_time = time.time() - start_time
            return AgentExecutionResult(
                question=question,
                success=True,
                steps=steps,
                final_answer=final_answer,
                confidence=confidence,
                citations=citations,
                total_execution_time=total_time,
                total_steps=len(steps),
                tools_used=tools_used
            )
        except Exception as e:
            logger.error(f"❌ FSM agent execution failed: {str(e)}")
            total_time = time.time() - start_time
            return AgentExecutionResult(
                question=question,
                success=False,
                steps=steps,
                final_answer=f"Agent execution failed: {str(e)}",
                confidence="low",
                citations="",
                total_execution_time=total_time,
                total_steps=len(steps),
                tools_used=tools_used,
                error=str(e)
            )
    
    async def _make_agent_decision(self, question: str, steps: List[AgentStep], current_findings: str) -> Any:
        """
        Agent decides the next action based on current state.
        
        Args:
            question: Original user question
            steps: Previous execution steps
            current_findings: Summary of current findings
            
        Returns:
            Decision object with next action and reasoning
        """
        # Prepare previous steps summary
        steps_summary = ""
        if steps:
            steps_summary = "\n".join([
                f"Step {step.step_number}: {step.tool_name or 'database_query'} - {step.reasoning[:100]}... "
                f"{'Success' if step.success else 'Failed'}"
                for step in steps[-3:]  # Last 3 steps to avoid context bloat
            ])
        else:
            steps_summary = "No previous steps - starting exploration"
        
        # Prepare available tools information (structured JSON)
        try:
            caps = TOOL_CAPABILITIES
            # Obligation-aware restriction: prioritize neighborhoods, then LanceDB
            if getattr(self, "obligation_ledger", None):
                unmet = self.obligation_ledger.unmet()
                if "neighborhoods" in unmet:
                    caps = {k: v for k, v in TOOL_CAPABILITIES.items() if k == "neighborhood_extractor"}
                    logger.info("FSM_ALLOWED_TOOLS: ['neighborhood_extractor']")
                elif "lancedb_knn" in unmet:
                    caps = {k: v for k, v in TOOL_CAPABILITIES.items() if k == "lancedb_knn"}
                    logger.info("FSM_ALLOWED_TOOLS: ['lancedb_knn']")
            available_tools_json = _json.dumps(caps)
        except Exception:
            available_tools_json = str(TOOL_CAPABILITIES)

        # Build a concise DB templates catalog (name, required, optional)
        def _catalog_from_specs():
            try:
                items = []
                for name, spec in SPECS.items():
                    items.append({
                        "name": name,
                        "required": list(spec.required.keys()),
                        "optional": list(spec.optional.keys()),
                        "category": getattr(spec, 'category', 'general'),
                        "returns": getattr(spec, 'returns', 'table'),
                        "cost": getattr(spec, 'cost', 'cheap'),
                        "slot_hints": getattr(spec, 'slot_hints', {}) or {},
                    })
                return _json.dumps({"templates": items})
            except Exception:
                return _json.dumps({"templates": []})

        db_templates_catalog = _catalog_from_specs()

        # Optional functional signatures catalog (external config; advisory only)
        def _load_functional_signatures() -> str:
            import os
            from pathlib import Path
            path = os.getenv("FUNCTIONAL_SIGNATURES_PATH", "config/functional_signatures.json")
            p = Path(path)
            if p.exists():
                try:
                    return p.read_text(encoding="utf-8")
                except Exception:
                    return _json.dumps({})
            return _json.dumps({})

        functional_signatures_catalog = _load_functional_signatures()

        # Compute lightweight data profile (scale hints) if a genome was selected earlier in the workflow
        data_profile = ""
        try:
            selected_genome = getattr(self, 'selected_genome', None)
            if selected_genome and hasattr(self.rag_system, 'neo4j_processor') and self.rag_system.neo4j_processor.driver:
                with self.rag_system.neo4j_processor.driver.session() as session:
                    # Count genes for the selected genome
                    gene_count = session.run(
                        "MATCH (g:Gene)-[:BELONGSTOGENOME]->(gen:Genome {genomeId: $gid}) RETURN count(g) as c",
                        gid=selected_genome,
                    ).single().get("c", 0)
                    # Approximate contig count via distinct gene.contig
                    contig_count = session.run(
                        "MATCH (g:Gene)-[:BELONGSTOGENOME]->(gen:Genome {genomeId: $gid}) RETURN count(DISTINCT g.contig) as c",
                        gid=selected_genome,
                    ).single().get("c", 0)
                # Rough chunk estimate assuming ~100 genes per analysis chunk (heuristic only)
                try:
                    import math
                    est_chunks = int(math.ceil((gene_count or 0) / 100.0))
                except Exception:
                    est_chunks = 0
                data_profile = f"genome={selected_genome}; contigs={contig_count}; genes={gene_count}; est_chunks≈{est_chunks}"
                # Stash in progress for later advisory checks
                try:
                    self._progress["est_chunks"] = est_chunks
                except Exception:
                    pass
        except Exception as _e:
            # Keep empty if any failure; hints are optional
            data_profile = ""

        # Generic policy/budget/tool-cost hints (non-binding)
        policy_hints = "templates-only-db; prefer-cheap-first; actions-may-require-approval"
        budget_state = "tokens_left=unknown; time_left=unknown; tool_budget=moderate"
        tool_costs = _json.dumps({
            "database_query": "cheap",
            "whole_genome_reader": "expensive",
            "similarity_search": "moderate",
            "code_interpreter": "moderate",
            "literature_search": "moderate",
            "synthesize": "cheap"
        })
        
        # Use model allocation for agent decisions (o3 for complex reasoning)
        # Build progress_state JSON (advisory)
        try:
            progress_state = {
                "candidates_collected": len(self._progress.get("distinct_protein_ids", set())),
                "loci_built": int(self._progress.get("loci_built", 0)),
                "last_row_count": int(self._progress.get("last_row_count", 0)),
                "zero_result_streak": int(self._progress.get("zero_result_streak", 0)),
            }
            progress_state_json = _json.dumps(progress_state)
        except Exception:
            progress_state_json = _json.dumps({})

        # Add dynamic hint for repeated no-op results
        if self._progress.get("zero_result_streak", 0) >= 2:
            policy_hints = policy_hints + "; no-op-repeat"

        def decision_call(module):
            # Provide enriched, but non-binding, context to improve tool selection
            return module(
                user_question=question,
                previous_steps=steps_summary,
                available_tools=available_tools_json,
                current_findings=current_findings,
                data_profile=data_profile,
                policy_hints=policy_hints,
                budget_state=budget_state,
                db_templates_catalog=db_templates_catalog,
                tool_costs=tool_costs,
                functional_signatures_catalog=functional_signatures_catalog,
                progress_state=progress_state_json,
            )
        
        logger.debug(f"🧠 Agent making decision for step {len(steps) + 1}")
        
        result = self.model_allocator.create_context_managed_call(
            task_name="agent_decision",  # Maps to COMPLEX = o3
            signature_class=AgentDecisionMaker,
            module_call_func=decision_call,
            query=question,
            task_context=f"Agent decision making for step {len(steps) + 1}"
        )

        if result is None:
            raise Exception("Model allocation failed for agent decision")

        logger.info(f"🧠 Agent decision: {result.next_action} (confidence: {result.confidence})")
        logger.debug(f"🎯 Reasoning: {result.biological_reasoning[:200]}...")

        # Enforce obligation-aware tool choice (neighborhoods first, then lancedb_knn)
        try:
            if getattr(self, "obligation_ledger", None):
                unmet = self.obligation_ledger.unmet()
                if "neighborhoods" in unmet:
                    # Force neighborhoods using recent seeds from cache if present
                    seed_ids: List[str] = []
                    try:
                        from pathlib import Path
                        import json as _json2
                        session_path = getattr(getattr(self, 'note_keeper', None), 'session_path', None)
                        if session_path:
                            tool_dir = Path(session_path) / 'tool_results'
                            if tool_dir.exists():
                                db_files = sorted(tool_dir.glob('db_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
                                for f in db_files[:5]:
                                    data = _json2.loads(f.read_text())
                                    rows = (data.get('tool_result') or {}).get('structured_data') or []
                                    for row in rows:
                                        pid = row.get('protein_id') or row.get('id')
                                        if isinstance(pid, str) and pid not in seed_ids:
                                            seed_ids.append(pid)
                                    if len(seed_ids) >= 10:
                                        break
                    except Exception:
                        seed_ids = []
                    if seed_ids:
                        setattr(result, 'next_action', 'neighborhood_extractor')
                        setattr(result, 'tool_parameters', _json.dumps({
                            'protein_ids': seed_ids,
                            'seeds_limit': 5,
                        }))
                        logger.info(f"🔒 Obligation gate: forcing neighborhood_extractor with seeds={len(seed_ids)}")
                elif "lancedb_knn" in unmet:
                    # Hard enforce: redirect to lancedb_knn with best-effort seeds
                    try:
                        from ..options.router import parse_macro_intent
                        intent0 = parse_macro_intent(question)
                        nn_val = unmet["lancedb_knn"].get("nn") or (getattr(getattr(intent0, 'nn', None), 'value', None) or 2)
                    except Exception:
                        nn_val = 2
                    # Try to extract recent seed_ids from cached DB results
                    seed_ids: List[str] = []
                    try:
                        from pathlib import Path
                        import json as _json2
                        session_path = getattr(getattr(self, 'note_keeper', None), 'session_path', None)
                        if session_path:
                            tool_dir = Path(session_path) / 'tool_results'
                            if tool_dir.exists():
                                db_files = sorted(tool_dir.glob('db_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
                                for f in db_files[:5]:
                                    data = _json2.loads(f.read_text())
                                    rows = (data.get('tool_result') or {}).get('structured_data') or []
                                    for row in rows:
                                        # Accept sanitized rows ({"protein_id": ...}) or nested node maps
                                        pid = row.get('protein_id') or row.get('id')
                                        if not pid:
                                            p = row.get('p') or row.get('protein')
                                            if isinstance(p, dict):
                                                pid = p.get('id')
                                        if isinstance(pid, str) and pid not in seed_ids:
                                            seed_ids.append(pid)
                                    if len(seed_ids) >= 10:
                                        break
                    except Exception:
                        seed_ids = []
                    # If no seeds yet, nudge the planner to fetch PFAM seeds deterministically
                    if not seed_ids:
                        # Overwrite decision toward a deterministic DB fetch using registry template
                        from ..kg.cypher_templates.registry import SPECS  # type: ignore
                        if 'proteins_with_pfam' in SPECS:
                            try:
                                from ..options.router import parse_macro_intent
                                it = parse_macro_intent(question)
                                marker = getattr(it, 'marker', None) or 'integrase'
                            except Exception:
                                marker = 'integrase'
                            setattr(result, 'next_action', 'database_query')
                            setattr(result, 'tool_parameters', _json.dumps({
                                'template': 'proteins_with_pfam',
                                'slots': {'pfam': marker, 'limit': 50, 'exact': False}
                            }))
                            logger.info("🔒 Obligation gate: forcing database_query(proteins_with_pfam) to collect seeds")
                        else:
                            # As last resort, preserve decision and let repair step try again
                            pass
                    else:
                        setattr(result, 'next_action', 'lancedb_knn')
                        # Use default PFAM exclusion if present in ledger
                        ex = unmet.get('lancedb_knn', {})
                        tool_params = {
                            'seed_ids': seed_ids,
                            'nn': int(nn_val),
                            'topk': max(10, 10 * int(nn_val)),
                            'distance': 'cosine',
                            'exclude_namespace': ex.get('exclude_namespace') or 'pfam',
                            'exclude_markers': ex.get('exclude_markers') or [],
                        }
                        setattr(result, 'tool_parameters', _json.dumps(tool_params))
                        logger.info(f"🔒 Obligation gate: forcing lancedb_knn with seeds={len(seed_ids)} nn={nn_val}")
        except Exception:
            pass

        # Advisory: mark WGR as requiring approval on large datasets based on est_chunks threshold (env-driven)
        try:
            import os as _os
            threshold = int(_os.getenv("AGENT_WGR_APPROVAL_CHUNKS", "0"))  # 0 means disabled
            est = int(self._progress.get("est_chunks", 0)) if isinstance(self._progress, dict) else 0
            if result.next_action == "whole_genome_reader" and threshold > 0 and est >= threshold:
                try:
                    setattr(result, 'requires_approval', True)
                    logger.info(f"⚠️ WGR on large dataset (est_chunks≈{est}) marked requires_approval (threshold={threshold})")
                except Exception:
                    pass
        except Exception:
            pass

        # Attempt to repair or enrich tool parameters for strict tool schemas (non-binding, best-effort)
        try:
            next_action = getattr(result, 'next_action', '') or ''
            params_text = getattr(result, 'tool_parameters', '') or ''
            # Normalize to dict if JSON
            try:
                params_obj = _json.loads(params_text) if isinstance(params_text, str) and params_text.strip().startswith('{') else params_text
            except Exception:
                params_obj = {}

            # If the agent mistakenly returned a DB template name as a tool, map it to database_query
            try:
                from ..kg.cypher_templates.registry import SPECS  # type: ignore
                if next_action and next_action not in self.tools and next_action in SPECS:
                    orig = next_action
                    setattr(result, 'next_action', 'database_query')
                    # Seed minimal parameters; downstream repair/validation will fill slots
                    setattr(result, 'tool_parameters', _json.dumps({"template": orig, "slots": params_obj if isinstance(params_obj, dict) else {}}))
                    next_action = 'database_query'
                    params_text = getattr(result, 'tool_parameters', '') or ''
                    try:
                        params_obj = _json.loads(params_text)
                    except Exception:
                        params_obj = {}
                    logger.info(f"🔁 Mapped template-name '{orig}' to database_query tool")
            except Exception:
                pass

            if next_action == 'database_query':
                # Expect {'template': <str>, 'slots': {}}
                need_repair = not isinstance(params_obj, dict) or 'template' not in params_obj or 'slots' not in params_obj or not isinstance(params_obj.get('slots'), dict)
                if need_repair:
                    schema = {
                        "type": "object",
                        "required": ["template", "slots"],
                        "additionalProperties": False,
                        "properties": {
                            "template": {"type": "string"},
                            "slots": {"type": "object"}
                        }
                    }
                    def repair_call(module):
                        return module(
                            instruction="Repair tool_parameters to match param_schema_json using db_templates_catalog.",
                            tool_name=next_action,
                            user_question=question,
                            bad=params_text if isinstance(params_text, str) else _json.dumps(params_obj),
                            db_templates_catalog=db_templates_catalog,
                            param_schema_json=_json.dumps(schema)
                        )
                    fixed = self.model_allocator.create_context_managed_call(
                        task_name="agent_decision_repair",
                        signature_class=DecisionParamRepair,
                        module_call_func=repair_call,
                        query="agent_decision_repair",
                        task_context="Agent decision parameter repair"
                    )
                    if fixed is not None:
                        fixed_text = getattr(fixed, 'json', '') or ''
                        try:
                            fixed_obj = _json.loads(fixed_text)
                            if isinstance(fixed_obj, dict) and 'template' in fixed_obj and 'slots' in fixed_obj:
                                # Overwrite decision parameters with repaired JSON and update locals
                                setattr(result, 'tool_parameters', _json.dumps(fixed_obj))
                                params_text = getattr(result, 'tool_parameters', '') or ''
                                params_obj = fixed_obj
                                logger.info("🔧 Repaired agent decision tool_parameters for database_query")
                        except Exception:
                            pass
                # Secondary validation: try compiling the template; if it fails, perform one more repair with error context
                try:
                    from ..kg.cypher_templates.registry import compile_query  # type: ignore
                    name = params_obj.get('template') if isinstance(params_obj, dict) else None
                    slots = params_obj.get('slots') if isinstance(params_obj, dict) else None
                    if isinstance(name, str) and isinstance(slots, dict):
                        try:
                            _cypher, _p = compile_query(name, slots)
                        except Exception as comp_err:
                            logger.warning(f"⚠️ Compile failed for template '{name}': {comp_err}. Attempting param repair with constraints.")
                            def repair2_call(module):
                                instruction = (
                                    "Fix database_query parameters so that the template compiles.\n"
                                    f"Current template: {name}\n"
                                    f"Compile error: {str(comp_err)}\n"
                                    "Use only templates/slots from db_templates_catalog; adjust slots accordingly."
                                )
                                return module(
                                    instruction=instruction,
                                    tool_name=next_action,
                                    user_question=question,
                                    bad=_json.dumps(params_obj),
                                    db_templates_catalog=db_templates_catalog,
                                    param_schema_json=_json.dumps({
                                        "type": "object",
                                        "required": ["template", "slots"],
                                        "additionalProperties": False,
                                        "properties": {"template": {"type": "string"}, "slots": {"type": "object"}},
                                    }),
                                )
                            fixed2 = self.model_allocator.create_context_managed_call(
                                task_name="agent_decision_repair",
                                signature_class=DecisionParamRepair,
                                module_call_func=repair2_call,
                                query="agent_decision_repair2",
                                task_context="Agent decision parameter repair (compile failure)"
                            )
                            if fixed2 is not None:
                                fixed_text2 = getattr(fixed2, 'json', '') or ''
                                try:
                                    fixed_obj2 = _json.loads(fixed_text2)
                                    if isinstance(fixed_obj2, dict) and 'template' in fixed_obj2 and 'slots' in fixed_obj2:
                                        # Validate compile again
                                        _cy2, _p2 = compile_query(fixed_obj2['template'], fixed_obj2['slots'])
                                        setattr(result, 'tool_parameters', _json.dumps(fixed_obj2))
                                        params_text = getattr(result, 'tool_parameters', '') or ''
                                        params_obj = fixed_obj2
                                        logger.info("🔧 Repaired agent decision tool_parameters after compile failure")
                                except Exception:
                                    pass
                except Exception:
                    # Ignore validation errors in preflight; downstream execution will still validate
                    pass
        except Exception as _e:
            # Non-fatal; proceed with original decision
            pass

        return result
    
    async def _execute_agent_step(self, step_number: int, tool_name: str, tool_parameters: str, 
                                reasoning: str, selected_genome: Optional[str] = None) -> AgentStep:
        """
        Execute a single agent step with the chosen tool.
        
        Args:
            step_number: Current step number
            tool_name: Name of tool to execute  
            tool_parameters: JSON parameters for the tool
            reasoning: Agent's reasoning for this step
            selected_genome: Optional genome scoping
            
        Returns:
            AgentStep with execution results
        """
        step_start = time.time()
        
        try:
            # Parse tool parameters
            import json
            try:
                params = json.loads(tool_parameters) if tool_parameters else {}
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool parameters: {tool_parameters}")
                params = {}
            
            # Add genome scoping if available
            if selected_genome and tool_name != "literature_search":
                params["target_genome"] = selected_genome
            
            # Execute the chosen tool
            if tool_name in self.tools:
                result = await self.tools[tool_name](params)
                success = True
                error = None
            elif tool_name == "synthesize":
                # Synthesis is handled separately
                result = "Ready to synthesize"
                success = True  
                error = None
            else:
                # Unknown tool - default to database query
                logger.warning(f"Unknown tool '{tool_name}', defaulting to database_query")
                result = await self.tools["database_query"](params)
                tool_name = None  # Indicates database_query
                success = True
                error = None
                
        except Exception as e:
            logger.error(f"❌ Step {step_number} execution failed: {str(e)}")
            result = None
            success = False
            error = str(e)
        
        execution_time = time.time() - step_start
        
        return AgentStep(
            step_number=step_number,
            tool_name=tool_name if tool_name != "synthesize" else None,
            tool_parameters=params,
            reasoning=reasoning,
            result=result,
            execution_time=execution_time,
            success=success,
            error=error
        )

    async def _execute_lancedb_knn(self, params: Dict[str, Any]) -> Any:
        """Execute LanceDB kNN tool via first-class tool wrapper."""
        from .external_tools import AVAILABLE_TOOLS
        tool = AVAILABLE_TOOLS.get("lancedb_knn")
        if tool is None:
            raise RuntimeError("Tool 'lancedb_knn' not registered")
        # Required params
        seed_ids = params.get("seed_ids") or []
        nn = int(params.get("nn") or 0)
        if not seed_ids or nn <= 0:
            raise ValueError("lancedb_knn requires 'seed_ids' (list) and 'nn' (>0)")
        # Optional params
        topk = int(params.get("topk") or max(10, 10 * nn))
        distance = params.get("distance", "cosine")
        exclude_namespace = params.get("exclude_namespace", "pfam")
        exclude_markers = params.get("exclude_markers") or []
        # Invoke tool
        result = await tool(
            rag_system=self.rag_system,
            seed_ids=seed_ids,
            nn=nn,
            topk=topk,
            distance=distance,
            exclude_namespace=exclude_namespace,
            exclude_markers=exclude_markers,
        )
        # Mark obligation done on success
        try:
            if getattr(self, 'obligation_ledger', None) and result and result.get('success'):
                self.obligation_ledger.mark_done('lancedb_knn')
        except Exception:
            pass
        return result
    
    async def _execute_database_query(self, params: Dict[str, Any]) -> Any:
        """Execute database query via STRICT template path only (envelope result)."""
        from .tool_schemas import ToolResultEnvelope  # For shape reference only
        from ..kg.cypher_templates.registry import SPECS  # type: ignore
        from ..options.router import parse_macro_intent

        template = params.get("template")
        slots = params.get("slots", {})
        if not template:
            raise ValueError("database_query requires 'template' and 'slots' (strict mode)")
        # Hard guard: if template not in registry, attempt one deterministic fallback using parsed intent
        if template not in SPECS:
            try:
                it = parse_macro_intent(self.current_question if hasattr(self, 'current_question') else '')
            except Exception:
                it = None
            marker = getattr(it, 'marker', None) if it else None
            if marker and 'proteins_with_pfam' in SPECS:
                logger.warning(f"FORCED_TEMPLATE_SELECTION: unknown template '{template}' → proteins_with_pfam")
                template = 'proteins_with_pfam'
                slots = {'pfam': marker, 'limit': 50, 'exact': False}
            elif 'proteins_with_pfam' in SPECS:
                logger.warning(f"FORCED_TEMPLATE_SELECTION: unknown template '{template}' → proteins_with_pfam (default integrase)")
                template = 'proteins_with_pfam'
                slots = {'pfam': 'integrase', 'limit': 50, 'exact': False}
            else:
                raise ValueError(f"Unknown template: {template}")
        # Inject default limit for list-style templates when missing
        try:
            spec = SPECS.get(template)
            if spec is not None:
                returns = getattr(spec, 'returns', 'table')
                # Heuristic: list-like templates that return gene/protein rows can be bounded by limit
                if returns in ("protein", "gene") and isinstance(slots, dict) and "limit" not in slots:
                    try:
                        # Prefer policy engine if available
                        default_limit = int(self.rag_system.policy_engine.get_max_results("database_query"))
                    except Exception:
                        import os as _os
                        default_limit = int(_os.getenv("AGENT_DEFAULT_DB_LIMIT", "100"))
                    slots["limit"] = max(1, min(default_limit, 5000))
        except Exception:
            pass
        # Deduplicate identical template+slots within this executor instance
        import json as _json
        try:
            sig = _json.dumps({"template": template, "slots": slots}, sort_keys=True)
        except Exception:
            sig = f"{template}|{str(sorted(slots.items()))}"
        if sig in self._db_dedup_cache:
            cached_env = self._db_dedup_cache[sig]
            env = dict(cached_env)
            env.setdefault("summary", {}).update({"deduplicated": True})
            return env

        # Execute template safely through Neo4j processor
        qres = await self.rag_system.neo4j_processor.execute_named_template(template, slots)
        rows = qres.results or []
        row_count = len(rows)

        # Build concise display text and structured envelope
        display = f"template={template} rows={row_count}"
        try:
            logger.info(f"🔎 DB template executed: {template} slots={slots} rows={row_count}")
            # Seed summary for common discovery templates
            if template in ("proteins_with_pfam", "proteins_with_pfams") and row_count:
                # Try to extract a few stable identifiers for debugging
                seeds = []
                for r in rows[:5]:
                    pid = r.get("protein_id") or r.get("id") or r.get("pid")
                    if pid:
                        seeds.append(str(pid))
                logger.info(f"DB_SEED_SUMMARY: n={row_count} sample={seeds}")
        except Exception:
            pass
        # Sanitize rows for common discovery templates so downstream tools can consume stable IDs
        try:
            if template in ("proteins_with_pfam", "proteins_with_pfams", "proteins_with_kos", "proteins_by_genome"):
                sanitized = []
                for r in rows:
                    # Attempt to extract protein ID from various shapes
                    pid = None
                    # First, accept already-sanitized rows
                    if isinstance(r, dict) and 'protein_id' in r and isinstance(r['protein_id'], str):
                        sanitized.append({"protein_id": r['protein_id']})
                        continue
                    pval = r.get('p') or r.get('protein') or r
                    try:
                        if isinstance(pval, dict) and 'id' in pval:
                            pid = pval['id']
                        elif hasattr(pval, '__getitem__'):
                            pid = pval['id']
                    except Exception:
                        pid = None
                    if pid:
                        sanitized.append({"protein_id": pid})
                if sanitized:
                    rows = sanitized
        except Exception:
            pass

        envelope = {
            "tool_name": "database_query",
            "success": True,
            "version": "1.0",
            "display_text": display,
            "structured_data": rows,
            "summary": {
                "template": template,
                "slots": slots,
                "row_count": row_count,
                "debug": {
                    "cypher": qres.metadata.get("cypher"),
                    "execution_time_sec": qres.execution_time,
                },
            },
            "references": [],
        }
        # Store in dedup cache and return
        self._db_dedup_cache[sig] = envelope
        # Mark seed selection obligation done for seed-fetching templates
        try:
            if getattr(self, 'obligation_ledger', None) and envelope.get('success'):
                if template in ("proteins_with_pfam", "proteins_with_pfams", "proteins_with_kos") and row_count > 0:
                    self.obligation_ledger.mark_done('seed_selection')
        except Exception:
            pass
        return envelope
    
    async def _execute_whole_genome_reader(self, params: Dict[str, Any]) -> Any:
        """Execute hierarchical genomic analysis instead of raw data dumping."""
        from .whole_genome_reader import WholeGenomeReader
        from .hierarchical_analysis import HierarchicalGenomeAnalyzer
        
        # Get the user's original question for context-driven analysis
        user_question = getattr(self, 'current_user_question', '') or params.get('user_question', '')
        
        # Initialize hierarchical analyzer
        hierarchical_analyzer = HierarchicalGenomeAnalyzer()
        
        # Get raw genomic data using existing reader
        reader = WholeGenomeReader(self.rag_system.neo4j_processor)
        
        # Extract genome ID and parameters
        genome_id = params.get("target_genome") or params.get("genome_id")
        max_genes = params.get("max_genes_per_contig", 2000)  # Increased for chunking
        
        if not genome_id:
            # If no specific genome, use global spatial reading
            from .whole_genome_reader import read_all_genomes_spatial
            raw_result = await read_all_genomes_spatial(self.rag_system.neo4j_processor)
            
            if raw_result.get("success") and raw_result.get("genome_contexts"):
                # Use hierarchical analysis instead of raw data dump
                hierarchical_result = hierarchical_analyzer.analyze_genome_hierarchically(
                    genome_contexts=raw_result["genome_contexts"],
                    user_question=user_question
                )
                
                return {
                    "success": True,
                    "analysis_type": "hierarchical_multi_genome",
                    "tool_output": self._format_hierarchical_output(hierarchical_result),
                    "summary": {
                        "loci_count": len(hierarchical_result.interesting_loci) if hierarchical_result.interesting_loci else 0,
                        "analysis_type": "curated_hierarchical_analysis"
                    }
                }
            else:
                return raw_result
        else:
            # Read specific genome
            result = await reader.read_complete_genome(genome_id, max_genes)
            
            if result["success"] and result.get("genome_context"):
                # Use hierarchical analysis for single genome
                hierarchical_result = hierarchical_analyzer.analyze_genome_hierarchically(
                    genome_contexts=[result["genome_context"]],
                    user_question=user_question
                )
                
                return {
                    "success": True,
                    "analysis_type": "hierarchical_single_genome",
                    "genome_id": genome_id,
                    "tool_output": self._format_hierarchical_output(hierarchical_result),
                    "summary": {
                        "loci_count": len(hierarchical_result.interesting_loci) if hierarchical_result.interesting_loci else 0,
                        "analysis_type": "curated_hierarchical_analysis"
                    }
                }
            else:
                return result
    
    def _format_hierarchical_output(self, hierarchical_result) -> str:
        """Format hierarchical analysis result for LLM consumption."""
        try:
            output_parts = []
            
            # Analysis summary
            if hierarchical_result.analysis_summary:
                output_parts.append("=== HIERARCHICAL GENOMIC ANALYSIS SUMMARY ===")
                summary = hierarchical_result.analysis_summary
                
                if "interesting_loci_count" in summary:
                    output_parts.append(f"Identified {summary['interesting_loci_count']} interesting loci from {summary.get('total_candidates_screened', 0)} candidates")
                
                if "loci_type_distribution" in summary:
                    type_dist = summary["loci_type_distribution"]
                    output_parts.append(f"Loci types: {', '.join([f'{k}: {v}' for k, v in type_dist.items()])}")
                
                if "genomic_coverage" in summary:
                    coverage = summary["genomic_coverage"]
                    output_parts.append(f"Total genes analyzed: {coverage.get('total_genes_in_loci', 0)}")
            
            # Interesting loci details
            if hierarchical_result.interesting_loci:
                output_parts.append("\\n=== INTERESTING LOCI ===")
                
                for i, locus in enumerate(hierarchical_result.interesting_loci, 1):
                    hyp_pct = (locus.hypothetical_count / locus.gene_count * 100) if locus.gene_count > 0 else 0
                    output_parts.append(f"\\nLocus #{i}: {locus.genomic_coordinates}")
                    output_parts.append(f"  Type: {locus.locus_type}")
                    output_parts.append(f"  Size: {locus.gene_count} genes ({locus.hypothetical_count} hypothetical, {hyp_pct:.1f}%)")
                    
                    if locus.biological_features:
                        output_parts.append(f"  Features: {'; '.join(locus.biological_features[:3])}")
                    
                    # Display detailed gene annotations
                    if hasattr(locus, 'detailed_genes') and locus.detailed_genes:
                        output_parts.append(f"  Detailed Gene Annotations:")
                        for j, gene in enumerate(locus.detailed_genes, 1):
                            gene_line = f"    {j}. {gene.gene_id} ({gene.start}-{gene.end}, {gene.strand})"
                            
                            # Add functional annotation
                            if gene.ko_description:
                                gene_line += f" - {gene.ko_description}"
                            elif gene.annotation:
                                gene_line += f" - {gene.annotation}"
                            else:
                                gene_line += " - hypothetical protein"
                            
                            output_parts.append(gene_line)
                            
                            # Add PFAM domains if available
                            if gene.pfam_domains:
                                domains_str = ", ".join(gene.pfam_domains[:3])  # Show first 3 domains
                                if len(gene.pfam_domains) > 3:
                                    domains_str += f" (+{len(gene.pfam_domains)-3} more)"
                                output_parts.append(f"       Domains: {domains_str}")
                            
                            # Add KEGG ID if available  
                            if gene.ko_id:
                                output_parts.append(f"       KEGG: {gene.ko_id}")
            
            # Detailed analyses
            if hierarchical_result.detailed_analyses:
                output_parts.append("\\n=== DETAILED LOCUS ANALYSES ===")
                
                for i, analysis in enumerate(hierarchical_result.detailed_analyses, 1):
                    locus = analysis.locus
                    output_parts.append(f"\\nDetailed Analysis #{i}: {locus.genomic_coordinates}")
                    
                    if analysis.functional_predictions:
                        output_parts.append(f"  Functional Predictions:")
                        for pred in analysis.functional_predictions[:3]:
                            output_parts.append(f"    - {pred}")
                    
                    if analysis.novelty_assessment:
                        output_parts.append(f"  Novelty: {analysis.novelty_assessment}")
                    
                    if analysis.detailed_genes:
                        output_parts.append(f"  Gene Count: {len(analysis.detailed_genes)} genes in locus")
            
            # Processing statistics
            if hierarchical_result.processing_stats:
                stats = hierarchical_result.processing_stats
                output_parts.append("\\n=== PROCESSING STATISTICS ===")
                output_parts.append(f"Chunks analyzed: {stats.get('successful_chunks', 0)}/{stats.get('total_chunks', 0)}")
                output_parts.append(f"Candidates identified: {stats.get('total_candidate_loci', 0)}")
                output_parts.append(f"Final interesting loci: {stats.get('interesting_loci', 0)}")
            
            return "\\n".join(output_parts)
            
        except Exception as e:
            return f"Error formatting hierarchical output: {e}"
    
    async def _execute_code_interpreter(self, params: Dict[str, Any]) -> Any:
        """Execute code interpreter for analysis."""
        from .external_tools import code_interpreter_tool
        
        # Enhance params with previous step data and original question
        enhanced_params = params.copy()
        enhanced_params["previous_step_data"] = getattr(self, '_previous_step_data', {})
        enhanced_params["original_question"] = getattr(self, 'current_user_question', '')
        
        # Generate analysis code based on enhanced parameters
        analysis_code = self._generate_analysis_code(enhanced_params)
        
        # Validate generated code syntax
        if not self._validate_code_syntax(analysis_code):
            raise Exception("Generated code has syntax errors")
        
        # Execute code with extended timeout
        result = await code_interpreter_tool(analysis_code, timeout=120)
        
        if result and result.get("success"):
            output = result.get("output", "")
            if len(output.strip()) == 0:
                raise Exception("Code interpreter produced no output - analysis may have failed")
            return output
        else:
            raise Exception(f"Code interpreter failed: {result.get('error', 'Unknown error')}")
    
    def _validate_code_syntax(self, code: str) -> bool:
        """Validate that generated code has correct syntax."""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            logger.error(f"Generated code has syntax error: {e}")
            return False
    
    def _update_previous_step_data(self, steps: List[AgentStep]) -> None:
        """Update the collection of previous step data for code interpreter access."""
        import json
        
        self._previous_step_data = {
            "step_count": len(steps),
            "tools_used": [],
            "step_results": {}
        }
        
        for step in steps:
            if step.success and step.result:
                # Track tools used
                tool_name = step.tool_name or "database_query"
                if tool_name not in self._previous_step_data["tools_used"]:
                    self._previous_step_data["tools_used"].append(tool_name)
                
                # Store step results (with size limits for code injection)
                step_key = f"step_{step.step_number}_{tool_name}"
                
                # For database_query results, extract key data
                if tool_name == "database_query" and isinstance(step.result, (list, dict)):
                    if isinstance(step.result, list):
                        self._previous_step_data["step_results"][step_key] = {
                            "tool": tool_name,
                            "data_type": "list",
                            "count": len(step.result),
                            "sample_data": step.result,  # Store full data, no 100-item limit
                            "full_data": step.result  # Include full data for analysis
                        }
                    else:
                        self._previous_step_data["step_results"][step_key] = {
                            "tool": tool_name,
                            "data_type": "dict", 
                            "data": step.result
                        }
                
                # For code interpreter results, extract structured data from JSON output
                elif tool_name == "code_interpreter" and isinstance(step.result, str):
                    # Try to extract structured analysis results from code interpreter output
                    try:
                        import re
                        # Look for the ANALYSIS RESULTS JSON block
                        json_match = re.search(r'ANALYSIS RESULTS:\s*(\{.*?\})\s*={50}', step.result, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                            analysis_results = json.loads(json_str)
                            
                            # Check if we have statistical data that can be converted to DataFrames
                            statistics = analysis_results.get('statistics', {})
                            extracted_data = []
                            
                            # Look for data tables in the statistics
                            for key, value in statistics.items():
                                if isinstance(value, str) and ('rows' in value or 'columns' in value):
                                    # This looks like DataFrame string representation
                                    # For now, store the analysis results as-is but mark as structured
                                    extracted_data.append({
                                        'type': 'analysis_summary',
                                        'content': analysis_results
                                    })
                                    break
                            
                            # Store the comprehensive analysis as structured data
                            self._previous_step_data["step_results"][step_key] = {
                                "tool": tool_name,
                                "data_type": "code_analysis",
                                "analysis_results": analysis_results,
                                "summary": analysis_results.get('summary', 'Code interpreter analysis completed'),
                                "key_findings": analysis_results.get('key_findings', []),
                                "statistics": statistics,
                                "comprehensive_analysis": True  # Flag for synthesis system
                            }
                        else:
                            # Fallback to string storage
                            result_str = str(step.result)
                            self._previous_step_data["step_results"][step_key] = {
                                "tool": tool_name,
                                "data_type": "string",
                                "summary": result_str[:500] + "..." if len(result_str) > 500 else result_str
                            }
                    except (json.JSONDecodeError, AttributeError) as e:
                        # Fallback to string storage if JSON parsing fails
                        result_str = str(step.result)
                        self._previous_step_data["step_results"][step_key] = {
                            "tool": tool_name,
                            "data_type": "string", 
                            "summary": result_str[:500] + "..." if len(result_str) > 500 else result_str
                        }
                
                # For other tools, store summary
                else:
                    result_str = str(step.result)
                    self._previous_step_data["step_results"][step_key] = {
                        "tool": tool_name,
                        "data_type": "string",
                        "summary": result_str[:500] + "..." if len(result_str) > 500 else result_str
                    }
    
    async def _execute_literature_search(self, params: Dict[str, Any]) -> Any:
        """Execute literature search."""
        from .external_tools import literature_search
        
        query = params.get("query", params.get("description", "biological research"))
        email = self.rag_system.config.get("email", "user@example.com")
        
        # Execute search
        result = literature_search(query, email, max_results=5)
        
        return result

    async def _execute_report_synthesis(self, params: Dict[str, Any]) -> Any:
        """Execute report synthesis tool wrapper."""
        from .external_tools import report_synthesis_tool
        description = params.get("description", "Generate final report")
        original_question = getattr(self, 'current_user_question', '')
        return report_synthesis_tool(
            description=description,
            original_question=original_question,
        )

    async def _execute_neighborhood_extractor(self, params: Dict[str, Any]) -> Any:
        """Execute DB-backed neighborhood extraction via curated templates."""
        from .external_tools import neighborhood_extractor_tool
        # Pass rag_system so the tool can execute DB templates
        result = await neighborhood_extractor_tool(
            rag_system=self.rag_system,
            protein_id=params.get("protein_id"),
            contig=params.get("contig"),
            start=params.get("start"),
            end=params.get("end"),
            k=params.get("k"),
            limit=params.get("limit"),
            protein_ids=params.get("protein_ids"),
            seeds_limit=params.get("seeds_limit", 5),
        )
        # Mark neighborhoods obligation done on success
        try:
            if getattr(self, 'obligation_ledger', None) and result and result.get('success'):
                self.obligation_ledger.mark_done('neighborhoods')
        except Exception:
            pass
        return result

    async def _execute_annotation_discovery(self, params: Dict[str, Any]) -> Any:
        """Execute integrated PFAM+KOFAM discovery for a keyword (default 'integrase')."""
        from .external_tools import annotation_discovery_tool
        return await annotation_discovery_tool(
            rag_system=self.rag_system,
            keyword=params.get("keyword") or params.get("q") or "integrase",
            limit=int(params.get("limit", 100)),
            protein_limit=int(params.get("protein_limit", 100)),
        )
    
    async def _execute_traditional_query_logic(self, question: str) -> Any:
        """
        Execute traditional query logic from core.py for database queries.
        
        This reuses the existing sophisticated query processing pipeline.
        """
        try:
            # Use existing query classification and retrieval logic
            from .dspy_signatures import QueryClassifier, ContextRetriever, NEO4J_SCHEMA
            
            # Classify query type
            def classification_call(module):
                return module(question=question)
            
            classification = self.model_allocator.create_context_managed_call(
                task_name="query_classification",
                signature_class=QueryClassifier,
                module_call_func=classification_call
            )
            
            if not classification:
                raise Exception("Query classification failed")
            
            # Generate retrieval plan
            def retrieval_call(module):
                return module(
                    db_schema=NEO4J_SCHEMA,
                    question=question,
                    query_type=classification.query_type,
                    task_context="Agent database query",
                    genome_filter_required="false",
                    target_genome="",
                    analysis_type="functional_annotation"
                )
            
            retrieval_plan = self.model_allocator.create_context_managed_call(
                task_name="context_preparation",
                signature_class=ContextRetriever,
                module_call_func=retrieval_call
            )
            
            if not retrieval_plan:
                raise Exception("Retrieval plan generation failed")
            
            # Execute the query
            if classification.query_type in ["structural", "general"]:
                result = await self.rag_system.neo4j_processor.process_query(
                    retrieval_plan.cypher_query, 
                    query_type="cypher"
                )
                return result.results if result.results else []
            else:
                result = await self.rag_system.hybrid_processor.process_query(retrieval_plan.cypher_query)
                combined_data = result.results[0] if result.results else {}
                return {
                    "structured_data": combined_data.get("structured_data", []),
                    "semantic_data": combined_data.get("semantic_data", [])
                }
                
        except Exception as e:
            logger.error(f"Traditional query execution failed: {e}")
            return []
    
    def _generate_analysis_code(self, params: Dict[str, Any]) -> str:
        """Generate Python analysis code using clean data interface."""
        
        description = params.get("description", "data analysis")
        previous_data = params.get("previous_step_data", {})
        original_question = params.get("original_question", "")
        
        # Create clean data interface template
        base_code = self._create_data_interface_template(previous_data, original_question, description)
        
        # Generate dynamic analysis code using LLM via model allocation
        if DSPY_AVAILABLE:
            try:
                # Create clear data summary for the LLM
                data_summary = self._create_data_summary(previous_data)
                
                # Use model allocation system for code generation
                def code_generation_call(module):
                    return module(
                        user_question=original_question,
                        available_data_summary=data_summary,
                        analysis_objective=description
                    )
                
                generated_result = self.model_allocator.create_context_managed_call(
                    task_name="code_generation",
                    signature_class=AnalysisCodeGenerator,
                    module_call_func=code_generation_call,
                    query=original_question,
                    task_context=f"Generate analysis code for: {description}"
                )
                
                if generated_result:
                    analysis_code = generated_result.analysis_code
                    logger.info(f"🐍 Generated analysis code ({len(analysis_code)} chars)")
                    logger.debug(f"Generated code preview: {analysis_code[:200]}...")
                else:
                    raise Exception("Code generation returned no result")
                
                # Add result validation template
                validation_code = """

# Validate analysis results format
if 'analysis_results' not in locals():
    analysis_results = {'summary': 'Analysis completed but no results stored'}

if not isinstance(analysis_results, dict):
    analysis_results = {'summary': str(analysis_results)}

# Ensure required fields exist
analysis_results.setdefault('summary', 'Analysis completed')
analysis_results.setdefault('key_findings', [])
analysis_results.setdefault('statistics', {})

print("\\n" + "="*50)
print("ANALYSIS RESULTS:")
print(json.dumps(analysis_results, indent=2, default=str))
print("="*50)
"""
                
                return base_code + "\n" + analysis_code + "\n" + validation_code
                
            except Exception as e:
                logger.error(f"LLM code generation failed: {e}")
                # Fallback to basic analysis
                return self._generate_fallback_analysis_code(base_code, original_question)
        else:
            # Fallback when DSPy not available
            return self._generate_fallback_analysis_code(base_code, original_question)
    
    def _create_data_interface_template(self, previous_data: Dict[str, Any], original_question: str, description: str) -> str:
        """Create clean data interface template with predictable access patterns."""
        
        # Process step results into clean format
        step_data = {}
        dataframes_data = {}
        
        for step_key, step_result in previous_data.get('step_results', {}).items():
            # Handle both list data and dict data with structured_data
            extracted_data = None
            data_count = 0
            
            if step_result.get('data_type') == 'list' and 'full_data' in step_result:
                # Direct list data (old format)
                extracted_data = step_result['full_data']
                data_count = step_result.get('count', len(extracted_data))
            elif step_result.get('data_type') == 'dict' and 'data' in step_result:
                # Dict data - check if it has structured_data field (database_query results)
                dict_data = step_result['data']
                if isinstance(dict_data, dict) and 'structured_data' in dict_data:
                    # Extract the structured_data list from database query results
                    extracted_data = dict_data['structured_data']
                    data_count = len(extracted_data) if isinstance(extracted_data, list) else 0
                    
            # If we found extractable data, process it
            if extracted_data and isinstance(extracted_data, list) and len(extracted_data) > 0:
                # Extract step number from key (e.g., "step_5_database_query" -> 5)
                step_num = None
                if step_key.startswith('step_'):
                    try:
                        step_num = int(step_key.split('_')[1])
                    except (IndexError, ValueError):
                        step_num = len(step_data) + 1
                else:
                    step_num = len(step_data) + 1
                
                # Store both by number and by original key for flexibility
                dataframes_data[step_num] = extracted_data
                dataframes_data[f'step_{step_num}'] = extracted_data
                
                step_data[step_num] = {
                    'tool': step_result.get('tool', 'unknown'),
                    'count': data_count,
                    'summary': f"Step {step_num} {step_result.get('tool', 'data')}"
                }
        
        # Create clean, safe template
        template = f'''
import pandas as pd
import numpy as np
import json
from builtins import *

# Analysis context
user_question = """{original_question}"""
analysis_objective = """{description}"""

print(f"User Question: {{user_question}}")
print(f"Analysis Objective: {{analysis_objective}}")

# Safe data loading with minimal complexity
dataframes = {{}}
step_data = {{}}

print("\\nLoading data from previous steps...")
'''

        # Add each dataset individually using the same enhanced extraction logic
        step_num = 1
        for step_key, step_result in previous_data.get('step_results', {}).items():
            # Use the same extraction logic as we used in dataframes_data
            extracted_data = None
            
            if step_result.get('data_type') == 'list' and 'full_data' in step_result:
                # Direct list data (old format)
                extracted_data = step_result['full_data']
            elif step_result.get('data_type') == 'dict' and 'data' in step_result:
                # Dict data - check if it has structured_data field (database_query results)
                dict_data = step_result['data']
                if isinstance(dict_data, dict) and 'structured_data' in dict_data:
                    # Extract the structured_data list from database query results
                    extracted_data = dict_data['structured_data']
            
            # Process extracted data
            if extracted_data and isinstance(extracted_data, list) and len(extracted_data) > 0:
                # Generate actual step number from key
                actual_step_num = step_num
                if step_key.startswith('step_'):
                    try:
                        actual_step_num = int(step_key.split('_')[1])
                    except (IndexError, ValueError):
                        actual_step_num = step_num
                
                template += f'''
try:
    # Load step {actual_step_num} data ({step_result.get('tool', 'unknown')})
    # FIXED: Use full dataset instead of artificial 5-record limit
    # User feedback: "We need to remove this threshold entirely"
    step_{actual_step_num}_data = {json.dumps(extracted_data, default=str)}  # Full data, no sampling
    if len(step_{actual_step_num}_data) > 0:
        df_{actual_step_num} = pd.DataFrame({json.dumps(extracted_data, default=str)})
        dataframes[{actual_step_num}] = df_{actual_step_num}
        dataframes['step_{actual_step_num}'] = df_{actual_step_num}
        step_data[{actual_step_num}] = {{'tool': '{step_result.get('tool', 'unknown')}', 'count': {len(extracted_data)}}}
        print(f"Loaded dataframes[{actual_step_num}]: {{df_{actual_step_num}.shape}} with columns {{list(df_{actual_step_num}.columns)}}")
except Exception as e:
    print(f"Could not load step {actual_step_num}: {{e}}")
'''
                step_num += 1

        template += f'''
print(f"\\nData Interface Ready:")
print(f"- {{len(dataframes)}} DataFrames available")
print(f"- Access via: dataframes[1], dataframes[2], etc.")
print("=" * 50)
'''
        return template
    
    def _create_data_summary(self, previous_data: Dict[str, Any]) -> str:
        """Create concise data summary for LLM code generation."""
        summaries = []
        
        for step_key, step_result in previous_data.get('step_results', {}).items():
            # Handle both list data and dict data with structured_data (same logic as _create_data_interface_template)
            extracted_data = None
            
            if step_result.get('data_type') == 'list' and 'full_data' in step_result:
                # Direct list data (old format)
                extracted_data = step_result['full_data']
            elif step_result.get('data_type') == 'dict' and 'data' in step_result:
                # Dict data - check if it has structured_data field (database_query results)
                dict_data = step_result['data']
                if isinstance(dict_data, dict) and 'structured_data' in dict_data:
                    # Extract the structured_data list from database query results
                    extracted_data = dict_data['structured_data']
            
            # Process extracted data
            if extracted_data and isinstance(extracted_data, list) and len(extracted_data) > 0:
                # Extract step number
                step_num = None
                if step_key.startswith('step_'):
                    try:
                        step_num = int(step_key.split('_')[1])
                    except (IndexError, ValueError):
                        step_num = len(summaries) + 1
                else:
                    step_num = len(summaries) + 1
                
                # Infer columns and types
                if isinstance(extracted_data[0], dict):
                    columns = list(extracted_data[0].keys())
                    sample_values = {k: str(extracted_data[0][k])[:50] for k in columns[:3]}
                    summaries.append(
                        f"dataframes[{step_num}]: {len(extracted_data)} rows, columns {columns}, "
                        f"tool: {step_result.get('tool', 'unknown')}, sample: {sample_values}"
                    )
        
        if not summaries:
            return "No structured data available for analysis"
        
        return "Available datasets: " + " | ".join(summaries)
    
    def _generate_fallback_analysis_code(self, base_code: str, question: str) -> str:
        """Generate basic analysis code when LLM generation fails."""
        fallback_analysis = f"""
# Fallback analysis - basic data exploration using clean interface
print("\\nPerforming basic data exploration...")

for step_num, df in dataframes.items():
    if isinstance(step_num, int):  # Only process numeric keys to avoid duplicates
        print(f"\\n--- Analysis of dataframes[{step_num}] ---")
        print(f"Shape: {{df.shape}}")
        print(f"Columns: {{list(df.columns)}}")
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"Numeric columns: {{list(numeric_cols)}}")
            print("Basic statistics:")
            print(df[numeric_cols].describe())
        
        # Value counts for categorical columns  
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols[:3]:  # First 3 categorical columns
            print(f"\\nValue counts for {{col}}:")
            print(df[col].value_counts().head())

# Store basic results using new format
analysis_results = {{
    'summary': "Basic data exploration completed",
    'key_findings': ["Data loaded and explored", "Question: {question}"],
    'statistics': {{'datasets_analyzed': len([k for k in dataframes.keys() if isinstance(k, int)])}},
    'analysis_type': 'fallback_exploration'
}}
"""
        return base_code + fallback_analysis
    
    def _summarize_step_result(self, step: AgentStep) -> str:
        """Create a concise summary of step results for next decision."""
        if not step.success:
            return f"Failed: {step.error or 'Unknown error'}"
        
        if not step.result:
            return "No results returned"
        
        # Summarize based on result type with biological context awareness
        if isinstance(step.result, list):
            return f"Found {len(step.result)} items"
        elif isinstance(step.result, dict):
            if "structured_data" in step.result and "semantic_data" in step.result:
                structured_count = len(step.result.get("structured_data", []))
                semantic_count = len(step.result.get("semantic_data", []))
                return f"Database results: {structured_count} structured, {semantic_count} semantic matches"
            
            # Extract meaningful biological information from dictionary results
            summary_parts = []
            
            # Check for common genomic data patterns
            if "tool_output" in step.result:
                tool_output = step.result["tool_output"]
                if isinstance(tool_output, str) and len(tool_output) > 100:
                    summary_parts.append(f"Genomic analysis: {len(tool_output)} characters of spatial data")
            
            if "genome_context" in step.result:
                genome_ctx = step.result["genome_context"]
                if isinstance(genome_ctx, dict):
                    if "total_genes" in genome_ctx:
                        summary_parts.append(f"{genome_ctx['total_genes']} genes analyzed")
                    if "genomes" in genome_ctx:
                        summary_parts.append(f"{len(genome_ctx['genomes'])} genomes")
                    if "contigs" in genome_ctx:
                        summary_parts.append(f"{genome_ctx['contigs']} contigs")
            
            if "success" in step.result and step.result.get("success"):
                if not summary_parts:
                    summary_parts.append("Analysis completed successfully")
            
            # For code interpreter results, extract comprehensive analysis details
            if step.tool_name == "code_interpreter":
                if isinstance(step.result, str):
                    result_text = step.result.lower()
                    
                    # Look for comprehensive biological analysis completion signals
                    if ("distribution" in result_text or "comparison" in result_text) and "protein" in result_text:
                        # Extract data size indicators from the results dynamically
                        import re
                        # Look for patterns like "X proteins", "X records", "X,XXX proteins" etc.
                        protein_matches = re.findall(r'(\d{1,3}(?:,\d{3})*|\d+)\s+(?:proteins?|records?)', step.result, re.IGNORECASE)
                        if protein_matches:
                            protein_count = protein_matches[0].replace(',', '')
                            summary_parts.append(f"COMPREHENSIVE biological analysis: {protein_count} proteins analyzed across genomes with complete statistical distribution")
                        else:
                            summary_parts.append("Comprehensive distribution analysis completed with statistical breakdown")
                    
                    # Look for statistical completion signals  
                    elif ("mean" in result_text and "std" in result_text) or "statistics" in result_text:
                        summary_parts.append("COMPREHENSIVE statistical analysis completed with descriptive statistics")
                    
                    # Look for comparative analysis completion
                    elif "compare" in result_text and ("genome" in result_text or "distribution" in result_text):
                        summary_parts.append("COMPARATIVE analysis completed across multiple datasets")
                    
                    # General completion patterns
                    elif "identified" in result_text or "found" in result_text:
                        summary_parts.append("Computational analysis with identified patterns")
                    elif "analysis" in result_text:
                        summary_parts.append("Statistical analysis completed")
                    else:
                        summary_parts.append("Code execution completed")
            
            if summary_parts:
                return "; ".join(summary_parts)
            else:
                return f"Dictionary result with {len(step.result)} keys"
                
        elif isinstance(step.result, str):
            return step.result[:200] + "..." if len(step.result) > 200 else step.result
        else:
            return f"Result type: {type(step.result).__name__}"
    
    async def _synthesize_agent_results(self, question: str, steps: List[AgentStep], 
                                      current_findings: str) -> Tuple[str, str, str]:
        """
        Synthesize final answer from all agent steps.
        
        Args:
            question: Original user question
            steps: All execution steps
            current_findings: Summary of findings
            
        Returns:
            Tuple of (answer, confidence, citations)
        """
        # Collect all successful results
        all_results = []
        tools_used = []
        
        for step in steps:
            if step.success and step.result:
                all_results.append({
                    "step": step.step_number,
                    "tool": step.tool_name or "database_query",
                    "reasoning": step.reasoning,
                    "result": step.result
                })
                tool_name = step.tool_name or "database_query"
                if tool_name not in tools_used:
                    tools_used.append(tool_name)
        
        # Use progressive synthesizer if available
        if self.note_keeper and hasattr(self.rag_system, 'progressive_synthesizer'):
            # Convert steps to note-like format for synthesis
            synthesis_data = []
            for result in all_results:
                synthesis_data.append({
                    "_source_task": f"agent_step_{result['step']}",
                    "_data_type": f"{result['tool']}_result",
                    "reasoning": result["reasoning"],
                    "data": result["result"]
                })
            
            # Use progressive synthesis
            synthesizer = self.rag_system.progressive_synthesizer
            if not synthesizer:
                from .memory import ProgressiveSynthesizer
                synthesizer = ProgressiveSynthesizer(self.note_keeper)
            
            answer = synthesizer.synthesize_progressive(
                task_notes=[],  # No traditional task notes
                question=question,
                raw_data=synthesis_data
            )
            
            return answer, "high", f"Agent exploration using: {', '.join(tools_used)}"
        
        # Fallback synthesis using GenomicAnswerer
        from .dspy_signatures import GenomicAnswerer
        
        # Format results for synthesis
        formatted_context = f"AGENT EXPLORATION RESULTS:\n\n"
        for result in all_results:
            formatted_context += f"Step {result['step']} ({result['tool']}):\n"
            formatted_context += f"Reasoning: {result['reasoning']}\n"
            formatted_context += f"Results: {str(result['result'])[:1000]}...\n\n"
        
        def answerer_call(module):
            return module(
                question=question,
                context=formatted_context
            )
        
        answer_result = self.model_allocator.create_context_managed_call(
            task_name="biological_interpretation",
            signature_class=GenomicAnswerer,
            module_call_func=answerer_call
        )
        
        if answer_result:
            return answer_result.answer, answer_result.confidence, answer_result.citations
        else:
            # Final fallback
            return (
                f"Completed {len(steps)} exploration steps using {', '.join(tools_used)} "
                f"to analyze: {question}. Results available but synthesis failed.",
                "medium",
                f"Agent exploration with {len(steps)} steps"
            )
    
    async def _run_guidance_synthesis(self, question: str, steps: List[AgentStep], current_findings: str) -> Optional[str]:
        """
        Run lightweight guidance synthesis for agent situational awareness.
        
        Args:
            question: Original user question
            steps: Completed agent steps so far
            current_findings: Current accumulated findings
            
        Returns:
            Brief guidance summary for next steps, or None if failed
        """
        if not self.note_keeper or not steps:
            return None
        
        try:
            # Convert recent steps to simplified notes (last 3 steps for guidance)
            recent_steps = steps[-3:] if len(steps) > 3 else steps
            recent_notes = self._steps_to_task_notes(recent_steps, question)
            
            # Use progressive synthesizer in guidance mode
            from .memory import ProgressiveSynthesizer
            synthesizer = ProgressiveSynthesizer(self.note_keeper)
            
            guidance = synthesizer.synthesize_progressive(
                task_notes=recent_notes,
                question=question,
                synthesis_mode="guidance"  # NEW: Lightweight mode
            )
            
            return guidance
            
        except Exception as e:
            logger.warning(f"⚠️ Guidance synthesis failed: {e}")
            return None
    
    async def _run_reporting_synthesis(self, question: str, steps: List[AgentStep], current_findings: str) -> Tuple[str, str, str]:
        """
        Run comprehensive reporting synthesis using all session notes.
        
        Args:
            question: Original user question
            steps: All completed agent steps
            current_findings: All accumulated findings
            
        Returns:
            Tuple of (answer, confidence, citations)
        """
        try:
            # First try using all task notes from the session
            if self.note_keeper:
                all_notes = self.note_keeper.get_all_task_notes()
                
                if all_notes:
                    logger.info(f"📊 Using {len(all_notes)} task notes for comprehensive reporting")
                    from .memory import ProgressiveSynthesizer
                    synthesizer = ProgressiveSynthesizer(self.note_keeper)
                    
                    final_answer = synthesizer.synthesize_progressive(
                        task_notes=all_notes,
                        question=question,
                        synthesis_mode="report"  # NEW: High-quality mode
                    )
                    
                    return final_answer, "high", f"Comprehensive analysis using {len(all_notes)} task notes"
            
            # Fallback to the original agent result synthesis
            logger.info("📊 No task notes available, using agent step results for reporting")
            return await self._synthesize_agent_results(question, steps, current_findings)
            
        except Exception as e:
            logger.error(f"❌ Reporting synthesis failed: {e}")
            # Final fallback
            return await self._synthesize_agent_results(question, steps, current_findings)
    
    def _save_agent_step_as_note(self, step: AgentStep, question: str) -> bool:
        """
        Save an agent step as a persistent task note with reference-based storage.
        
        Args:
            step: Agent step to save as note
            question: Original question for context
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from .memory.note_schemas import NotingDecisionResult, ConfidenceLevel
            
            # Create noting decision result 
            decision_result = NotingDecisionResult(
                should_record=True,
                importance_score=7.0,  # Medium-high priority for agent steps
                reasoning="Agent exploration step with biological findings"
            )
            
            # Extract observations and findings from step result
            observations = [f"Tool used: {step.tool_name or 'database_query'}"]
            key_findings = [step.reasoning]
            
            # Store execution metadata in quantitative_data
            quantitative_data = {
                "execution_time": step.execution_time,
                "step_number": step.step_number,
                "success": step.success
            }
            
            if step.result:
                result_summary = self._summarize_step_result(step)
                observations.append(f"Result: {result_summary}")
                
                # REVOLUTIONARY CHANGE: Use reference-based caching for large results
                if self.tool_cache:
                    # Cache the full tool result and store reference ID
                    tool_name = step.tool_name or "database_query"
                    step_context = f"Step {step.step_number}: {step.reasoning[:50]}..."
                    
                    result_id = self.tool_cache.cache_tool_result(
                        tool_name=tool_name,
                        tool_result=step.result,
                        step_context=step_context
                    )
                    
                    if result_id:
                        # Store reference instead of full result (99.5% size reduction!)
                        quantitative_data["tool_result_ref"] = result_id
                        quantitative_data["tool_result_summary"] = self.tool_cache.get_result_summary(result_id)
                        
                        # Extract key biological discoveries for immediate access
                        discoveries = self.tool_cache.extract_key_discoveries(tool_name, step.result)
                        if discoveries:
                            key_findings.extend(discoveries)
                            logger.info(f"🔬 Extracted {len(discoveries)} biological discoveries from {tool_name}")
                    else:
                        # Fallback: store result directly if caching fails
                        quantitative_data["full_tool_result"] = step.result
                        logger.warning(f"⚠️ Tool result caching failed for step {step.step_number}, storing directly")
                else:
                    # No cache available - store directly
                    quantitative_data["full_tool_result"] = step.result
                
                # Legacy biological finding extraction (backup)
                biological_findings = self._extract_biological_findings(step)
                if biological_findings:
                    key_findings.extend(biological_findings)
            
            # Record the task note
            success = self.note_keeper.record_task_notes(
                task_id=f"agent_step_{step.step_number}",
                task_type=f"agent_{step.tool_name or 'database_query'}",
                description=step.reasoning,
                decision_result=decision_result,
                observations=observations,
                key_findings=key_findings,
                quantitative_data=quantitative_data,
                cross_connections=[],
                confidence=ConfidenceLevel.HIGH if step.success else ConfidenceLevel.LOW,
                execution_time=step.execution_time,
                tokens_used=0  # We don't have token tracking for agent steps yet
            )
            
            if success:
                logger.info(f"💾 Saved agent step {step.step_number} as reference-based task note")
            else:
                logger.error(f"❌ Failed to save agent step {step.step_number} as task note")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Error saving agent step as note: {e}")
            return False

    def _extract_biological_findings(self, step: AgentStep) -> List[str]:
        """
        Extract specific biological findings from tool results.
        
        Args:
            step: Agent step with tool results
            
        Returns:
            List of biological findings as strings
        """
        findings = []
        
        if not step.result:
            return findings
            
        try:
            # Handle whole_genome_reader results
            if step.tool_name == "whole_genome_reader":
                if isinstance(step.result, dict):
                    if "tool_output" in step.result:
                        tool_output = step.result["tool_output"]
                        if isinstance(tool_output, str):
                            # Extract key biological patterns from genomic analysis
                            if "genes" in tool_output.lower():
                                findings.append("Genomic analysis completed with gene spatial mapping")
                            if "hypothetical" in tool_output.lower():
                                findings.append("Identified regions with hypothetical proteins")
                            if "operon" in tool_output.lower():
                                findings.append("Detected potential operon structures")
                    
                    if "genome_context" in step.result:
                        ctx = step.result["genome_context"]
                        if isinstance(ctx, dict) and "total_genes" in ctx:
                            findings.append(f"Analyzed {ctx['total_genes']} genes across genomic regions")
            
            # Handle code_interpreter results
            elif step.tool_name == "code_interpreter":
                if isinstance(step.result, str):
                    result_lower = step.result.lower()
                    # Look for analysis patterns indicating discoveries
                    if "loci" in result_lower or "locus" in result_lower:
                        findings.append("Computational analysis identified candidate loci")
                    if "prophage" in result_lower:
                        findings.append("Analysis detected potential prophage regions")
                    if "cluster" in result_lower and "protein" in result_lower:
                        findings.append("Identified protein clustering patterns")
                    if "score" in result_lower or "rank" in result_lower:
                        findings.append("Quantitative scoring and ranking performed")
            
            # Handle database query results
            elif step.tool_name is None:  # database_query
                # New envelope shape: dict with structured_data
                if isinstance(step.result, dict) and 'structured_data' in step.result:
                    rows = step.result.get('structured_data') or []
                    if isinstance(rows, list) and rows:
                        findings.append(f"Retrieved {len(rows)} database records")
                        sample_size = min(3, len(rows))
                        for record in rows[:sample_size]:
                            if isinstance(record, dict):
                                if "protein_id" in record or "gene_id" in record:
                                    findings.append("Database results include protein/gene identifiers")
                                    break
                                if "ko_description" in record:
                                    findings.append("Results include KEGG functional annotations")
                                    break
                # Legacy shape: direct list of rows
                elif isinstance(step.result, list) and len(step.result) > 0:
                    findings.append(f"Retrieved {len(step.result)} database records")
                    sample_size = min(3, len(step.result))
                    for record in step.result[:sample_size]:
                        if isinstance(record, dict):
                            if "protein_id" in record or "gene_id" in record:
                                findings.append("Database results include protein/gene identifiers")
                                break
                            if "ko_description" in record:
                                findings.append("Results include KEGG functional annotations")
                                break
            
            # Handle literature search results
            elif step.tool_name == "literature_search":
                if isinstance(step.result, dict) and "papers" in step.result:
                    paper_count = len(step.result["papers"])
                    findings.append(f"Retrieved {paper_count} relevant research papers")
        
        except Exception as e:
            logger.warning(f"Error extracting biological findings: {e}")
            
        return findings

    def _steps_to_task_notes(self, steps: List[AgentStep], question: str) -> List:
        """
        Convert agent steps to TaskNote-like format for synthesis.
        
        Args:
            steps: Agent steps to convert
            question: Original question for context
            
        Returns:
            List of task note dictionaries
        """
        from .memory.note_schemas import TaskNote, ConfidenceLevel, NotingDecisionResult
        from datetime import datetime
        
        task_notes = []
        for step in steps:
            if step.success and step.result:
                # Create a simplified task note
                decision_result = NotingDecisionResult(
                    should_record=True,
                    importance_score=7.0,
                    reasoning="Agent exploration step with biological findings"
                )
                
                task_note = TaskNote(
                    task_id=f"agent_step_{step.step_number}",
                    task_type=f"agent_{step.tool_name or 'database_query'}",
                    description=step.reasoning,
                    note_decision=decision_result,
                    observations=[f"Tool: {step.tool_name or 'database_query'}", 
                                f"Result: {self._summarize_step_result(step)}"],
                    key_findings=[step.reasoning],
                    confidence_level=ConfidenceLevel.HIGH if step.success else ConfidenceLevel.LOW,
                    quantitative_data={
                        "execution_time": step.execution_time,
                        "step_number": step.step_number,
                        "success": step.success
                    },
                    cross_task_connections=[],
                    execution_time=step.execution_time
                )
                task_notes.append(task_note)
        
        return task_notes
    
    def _save_task_debug_data(self, step: AgentStep, step_number: int) -> None:
        """
        Save individual task result for debugging data flow.
        
        Args:
            step: Agent step to debug
            step_number: Step number for identification
        """
        try:
            if not self.note_keeper or not hasattr(self.note_keeper, 'session_path'):
                return
            
            # Create debug directory
            debug_dir = self.note_keeper.session_path / "debug_data_flow"
            debug_dir.mkdir(exist_ok=True)
            
            # Create debug file for this task
            from datetime import datetime
            timestamp = datetime.now().strftime("%H%M%S")
            debug_file = debug_dir / f"task_step_{step_number}_{timestamp}.json"
            
            # Prepare debug payload with full task information
            debug_payload = {
                "step_number": step_number,
                "tool_name": step.tool_name or "database_query",
                "success": step.success,
                "execution_time": step.execution_time,
                "reasoning": step.reasoning,
                "tool_parameters": step.tool_parameters,
                "result_type": type(step.result).__name__,
                "result_size_chars": len(str(step.result)) if step.result else 0,
                "error": step.error,
                "timestamp": datetime.now().isoformat(),
                "full_result": step.result  # ⭐ The complete result data
            }
            
            # Write debug file
            import json
            with open(debug_file, 'w') as f:
                json.dump(debug_payload, f, indent=2, default=str)
            
            logger.debug(f"🐛 DEBUG: Saved task step {step_number} result to {debug_file.name} ({debug_payload['result_size_chars']} chars)")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save task debug data for step {step_number}: {e}")

    def _update_progress(self, step: AgentStep) -> None:
        """Update generic progress indicators from a completed step (non-hardcoded)."""
        try:
            prog = self._progress
            if not isinstance(prog, dict):
                return
            # Database query path (tool_name None)
            if step.tool_name is None:
                rows = []
                if isinstance(step.result, dict) and 'structured_data' in step.result:
                    rows = step.result.get('structured_data') or []
                elif isinstance(step.result, list):
                    rows = step.result
                row_count = len(rows) if isinstance(rows, list) else 0
                prog["last_row_count"] = row_count

                # Query signature from parameters
                sig = None
                try:
                    if isinstance(step.tool_parameters, dict):
                        if 'template' in step.tool_parameters and 'slots' in step.tool_parameters:
                            sig = f"{step.tool_parameters['template']}|{_json.dumps(step.tool_parameters['slots'], sort_keys=True)}"
                        else:
                            sig = _json.dumps(step.tool_parameters, sort_keys=True)
                    else:
                        sig = str(step.tool_parameters)
                except Exception:
                    sig = None

                last_sig = prog.get("last_query_signature")
                if row_count == 0 and sig and sig == last_sig:
                    prog["zero_result_streak"] = int(prog.get("zero_result_streak", 0)) + 1
                else:
                    prog["zero_result_streak"] = 0
                if sig:
                    prog["last_query_signature"] = sig

                # Track distinct protein ids (if present generically)
                if isinstance(rows, list):
                    ids = [r.get('protein_id') for r in rows if isinstance(r, dict) and r.get('protein_id')]
                    if ids and isinstance(prog.get("distinct_protein_ids"), set):
                        prog["distinct_protein_ids"].update(ids)

            elif step.tool_name == "neighborhood_extractor":
                rows = []
                if isinstance(step.result, dict) and 'structured_data' in step.result:
                    rows = step.result.get('structured_data') or []
                row_count = len(rows) if isinstance(rows, list) else 0
                if row_count > 0:
                    prog["loci_built"] = int(prog.get("loci_built", 0)) + 1
        except Exception as e:
            logger.debug(f"Progress update skipped: {e}")
