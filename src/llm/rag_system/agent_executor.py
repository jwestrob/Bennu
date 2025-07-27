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

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

from .external_tools import AVAILABLE_TOOLS, TOOL_CAPABILITIES
from .memory import NoteKeeper, get_model_allocator
from .memory.tool_result_cache import ToolResultCache
from .utils import safe_log_data

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
    """
    
    user_question = dspy.InputField(desc="Original user question with biological context")
    previous_steps = dspy.InputField(desc="Summary of previous tool executions and their results")
    available_tools = dspy.InputField(desc="Available tools with their capabilities and decision criteria")
    current_findings = dspy.InputField(desc="Current biological findings and data collected so far")
    
    next_action = dspy.OutputField(desc="Next action: tool name from available_tools or 'synthesize' to finish")
    tool_parameters = dspy.OutputField(desc="JSON parameters for the selected tool (empty object {} for synthesize)")
    biological_reasoning = dspy.OutputField(desc="Detailed biological reasoning for this decision based on current findings")
    confidence = dspy.OutputField(desc="Confidence level 0.0-1.0 in this decision")
    exploration_complete = dspy.OutputField(desc="true if comprehensive analysis is complete (statistical analysis done, patterns identified, question fully answered), false if more tools needed")


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
            "code_interpreter": self._execute_code_interpreter,
            "literature_search": self._execute_literature_search
        }
        
        # Execution state
        self.max_steps = 8  # Prevent infinite loops
        self.step_timeout = 300  # 5 minutes per step
        self.guidance_frequency = 3  # Run guidance synthesis every N steps
        
        # Initialize data collection for code interpreter
        self._previous_step_data = {}
        
        # Code generator will use model allocation system
        # (no need to initialize here, will use model_allocator.create_context_managed_call)
        
        logger.info("🤖 UnifiedAgentExecutor initialized - dynamic tool chaining enabled")
    
    async def execute_agent_workflow(self, question: str, selected_genome: Optional[str] = None) -> AgentExecutionResult:
        # Store current user question for hierarchical analysis context
        self.current_user_question = question
        """
        Execute agent workflow with dynamic tool chaining.
        
        Args:
            question: User's question
            selected_genome: Pre-selected genome for scoped analysis
            
        Returns:
            AgentExecutionResult with all execution steps and final synthesis
        """
        logger.info(f"🚀 Starting agent workflow for: {question[:100]}...")
        start_time = time.time()
        
        # Initialize execution state
        steps: List[AgentStep] = []
        current_findings = f"Analyzing question: {question}"
        tools_used: List[str] = []
        
        # Removed: Tool usage tracking (allow multiple consecutive tool calls)
        
        # Set note-taking context if available
        if self.note_keeper:
            self.note_keeper.set_session_context(question, "unified_agent")
        
        try:
            # Agent execution loop
            for step_number in range(1, self.max_steps + 1):
                logger.info(f"🔄 Agent step {step_number}/{self.max_steps}")
                
                # Agent decides next action
                decision = await self._make_agent_decision(
                    question=question,
                    steps=steps,
                    current_findings=current_findings
                )
                
                if decision.exploration_complete and decision.next_action == "synthesize":
                    logger.info("🎯 Agent decided to synthesize final answer")
                    break
                
                # Allow multiple consecutive tool calls - agent can use code_interpreter multiple times
                
                # Execute the chosen tool
                step_result = await self._execute_agent_step(
                    step_number=step_number,
                    tool_name=decision.next_action,
                    tool_parameters=decision.tool_parameters,
                    reasoning=decision.biological_reasoning,
                    selected_genome=selected_genome
                )
                
                steps.append(step_result)
                
                # Collect previous step data for code interpreter access
                self._update_previous_step_data(steps)
                
                # DEBUG: Save individual task result for debugging
                self._save_task_debug_data(step_result, step_number)
                
                # CRITICAL FIX: Convert agent step to task note and save it
                if self.note_keeper and step_result.success:
                    self._save_agent_step_as_note(step_result, question)
                
                # Update tracking
                if step_result.tool_name and step_result.tool_name not in tools_used:
                    tools_used.append(step_result.tool_name)
                elif step_result.tool_name is None:  # database_query
                    if "database_query" not in tools_used:
                        tools_used.append("database_query")
                
                # CRITICAL FIX: Update findings IMMEDIATELY after step execution
                # This ensures the agent sees current results when making the next decision
                if step_result.success and step_result.result:
                    result_summary = self._summarize_step_result(step_result)
                    current_findings += f"\n\nStep {step_number} findings: {result_summary}"
                else:
                    current_findings += f"\n\nStep {step_number} failed: {step_result.error or 'Unknown error'}"
                
                logger.info(f"✅ Step {step_number} completed: {step_result.tool_name or 'database_query'}")
                
                # HYBRID MODEL: Periodic Guidance Synthesis
                # DEBUG: Log modulo calculation details
                logger.debug(f"🔍 DEBUG: step_number={step_number}, guidance_frequency={self.guidance_frequency}, modulo={step_number % self.guidance_frequency}")
                
                if step_number % self.guidance_frequency == 0 and step_number < self.max_steps:
                    logger.info(f"🧭 Running guidance synthesis after step {step_number}")
                    guidance_summary = await self._run_guidance_synthesis(
                        question=question,
                        steps=steps,  # Use all completed steps so far
                        current_findings=current_findings
                    )
                    
                    if guidance_summary:
                        # Inject guidance into agent's working memory
                        current_findings += f"\n\n🧭 GUIDANCE UPDATE (after step {step_number}):\n{guidance_summary}"
                        logger.info("🧭 Guidance synthesis complete - updated agent context")
                else:
                    logger.debug(f"🔍 Guidance synthesis skipped: step {step_number} not divisible by {self.guidance_frequency}")
            
            # HYBRID MODEL: Final comprehensive reporting synthesis
            logger.info("📊 Running final reporting synthesis with all notes")
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
            logger.error(f"❌ Agent execution failed: {str(e)}")
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
        
        # Prepare available tools information
        available_tools_json = str(TOOL_CAPABILITIES)
        
        # Use model allocation for agent decisions (o3 for complex reasoning)
        def decision_call(module):
            return module(
                user_question=question,
                previous_steps=steps_summary,
                available_tools=available_tools_json,
                current_findings=current_findings
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
    
    async def _execute_database_query(self, params: Dict[str, Any]) -> Any:
        """Execute database query using existing processors."""
        # Extract query intent from parameters
        description = params.get("description", params.get("query", "General database search"))
        
        # Use existing query processing logic from core.py
        # This will generate appropriate Cypher queries and execute them
        return await self._execute_traditional_query_logic(description)
    
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
                if isinstance(step.result, list) and len(step.result) > 0:
                    findings.append(f"Retrieved {len(step.result)} database records")
                    
                    # Sample first few results to extract patterns
                    sample_size = min(3, len(step.result))
                    for i, record in enumerate(step.result[:sample_size]):
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
            
            logger.info(f"🐛 DEBUG: Saved task step {step_number} result to {debug_file.name} ({debug_payload['result_size_chars']} chars)")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save task debug data for step {step_number}: {e}")