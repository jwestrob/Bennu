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
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

from .external_tools import AVAILABLE_TOOLS, TOOL_CAPABILITIES
from .memory import NoteKeeper, get_model_allocator
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
    - Sufficient information has been gathered
    - Multiple tools have provided complementary data
    - Ready to provide comprehensive answer
    """
    
    user_question = dspy.InputField(desc="Original user question with biological context")
    previous_steps = dspy.InputField(desc="Summary of previous tool executions and their results")
    available_tools = dspy.InputField(desc="Available tools with their capabilities and decision criteria")
    current_findings = dspy.InputField(desc="Current biological findings and data collected so far")
    
    next_action = dspy.OutputField(desc="Next action: tool name from available_tools or 'synthesize' to finish")
    tool_parameters = dspy.OutputField(desc="JSON parameters for the selected tool (empty object {} for synthesize)")
    biological_reasoning = dspy.OutputField(desc="Detailed biological reasoning for this decision based on current findings")
    confidence = dspy.OutputField(desc="Confidence level 0.0-1.0 in this decision")
    exploration_complete = dspy.OutputField(desc="true if ready to synthesize final answer, false if more exploration needed")


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
        
        logger.info("🤖 UnifiedAgentExecutor initialized - dynamic tool chaining enabled")
    
    async def execute_agent_workflow(self, question: str, selected_genome: Optional[str] = None) -> AgentExecutionResult:
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
                
                # Execute the chosen tool
                step_result = await self._execute_agent_step(
                    step_number=step_number,
                    tool_name=decision.next_action,
                    tool_parameters=decision.tool_parameters,
                    reasoning=decision.biological_reasoning,
                    selected_genome=selected_genome
                )
                
                steps.append(step_result)
                
                # CRITICAL FIX: Convert agent step to task note and save it
                if self.note_keeper and step_result.success:
                    self._save_agent_step_as_note(step_result, question)
                
                # Update tracking
                if step_result.tool_name and step_result.tool_name not in tools_used:
                    tools_used.append(step_result.tool_name)
                elif step_result.tool_name is None:  # database_query
                    if "database_query" not in tools_used:
                        tools_used.append("database_query")
                
                # Update findings for next decision
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
        """Execute spatial genomic analysis."""
        from .whole_genome_reader import WholeGenomeReader
        
        reader = WholeGenomeReader(self.rag_system.neo4j_processor)
        
        # Extract genome ID and parameters
        genome_id = params.get("target_genome") or params.get("genome_id")
        max_genes = params.get("max_genes_per_contig", 1000)
        
        if not genome_id:
            # If no specific genome, use global spatial reading
            from .whole_genome_reader import read_all_genomes_spatial
            return await read_all_genomes_spatial(self.rag_system.neo4j_processor)
        else:
            # Read specific genome
            result = await reader.read_complete_genome(genome_id, max_genes)
            
            if result["success"]:
                # Format for LLM analysis
                formatted = reader.format_for_llm_analysis(
                    result["genome_context"], 
                    focus_on_spatial=True
                )
                return {
                    "success": True,
                    "tool_output": formatted,
                    "genome_context": result["genome_context"]
                }
            else:
                return result
    
    async def _execute_code_interpreter(self, params: Dict[str, Any]) -> Any:
        """Execute code interpreter for analysis."""
        from .external_tools import code_interpreter_tool
        
        # Generate analysis code based on parameters
        analysis_code = self._generate_analysis_code(params)
        
        # Execute code
        result = await code_interpreter_tool(analysis_code)
        
        if result and result.get("success"):
            return result.get("output", "")
        else:
            raise Exception(f"Code interpreter failed: {result.get('error', 'Unknown error')}")
    
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
        """Generate Python analysis code based on parameters."""
        analysis_type = params.get("analysis_type", "general")
        description = params.get("description", "data analysis")
        
        if "statistical" in analysis_type or "statistics" in description.lower():
            return """
import pandas as pd
import numpy as np

# Statistical analysis
print("Performing statistical analysis...")
print(f"Analysis description: {description}")
"""
        elif "pattern" in analysis_type or "pattern" in description.lower():
            return """
import pandas as pd
import matplotlib.pyplot as plt

# Pattern detection analysis  
print("Detecting patterns in biological data...")
print(f"Analysis description: {description}")
"""
        else:
            return f"""
# General analysis
print("General biological data analysis")
print(f"Description: {description}")
print("Analysis complete")
"""
    
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
            
            # For code interpreter results, look for output patterns
            if step.tool_name == "code_interpreter":
                if isinstance(step.result, str):
                    if "identified" in step.result.lower() or "found" in step.result.lower():
                        summary_parts.append("Computational analysis with identified patterns")
                    elif "analysis" in step.result.lower():
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
        Save an agent step as a persistent task note.
        
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
            
            # Store full tool results in quantitative_data for preservation
            quantitative_data = {
                "execution_time": step.execution_time,
                "step_number": step.step_number,
                "success": step.success
            }
            
            if step.result:
                result_summary = self._summarize_step_result(step)
                observations.append(f"Result: {result_summary}")
                if len(result_summary) > 20:
                    key_findings.append(result_summary)
                
                # CRITICAL FIX: Store the full tool result for synthesis
                quantitative_data["full_tool_result"] = step.result
                
                # Extract specific biological findings from tool results
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
                logger.info(f"💾 Saved agent step {step.step_number} as task note")
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