"""
LLM-First Tool Selection for Genomic Analysis Tasks.

Uses sophisticated biological reasoning (gpt-5) to select appropriate tools based on:
- Analysis type (spatial vs functional vs comparative)
- Query scope (global vs targeted vs lookup)
- Scientific intent (discovery vs annotation vs quantification)

No regex fallbacks - the LLM has complete authority over tool selection.
Fail-fast approach ensures biological appropriateness of tool choices.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import json
import os

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

from .external_tools import AVAILABLE_TOOLS, TOOL_CAPABILITIES
from .memory.model_allocation import get_model_allocator
from .task_management import Task, TaskType

logger = logging.getLogger(__name__)

# Quarantine gate: legacy selectors are DISABLED by default.
# Re-enable by setting AGENT_ENABLE_LEGACY_SELECTORS=1 for rollback.
if os.getenv("AGENT_ENABLE_LEGACY_SELECTORS", "0") != "1":
    raise ImportError(
        "Legacy tool selectors are quarantined by default. Set "
        "AGENT_ENABLE_LEGACY_SELECTORS=1 to re-enable, or migrate to "
        "src/llm/rag_system/router."
    )


@dataclass
class ToolSelectionResult:
    """Result from agent-based tool selection."""
    selected_tool: Optional[str]
    tool_arguments: Dict[str, Any]
    reasoning: str
    confidence: float
    fallback_used: bool = False


class BiologicalToolSelector(dspy.Signature if DSPY_AVAILABLE else object):
    """Intelligent tool selection for genomic analysis based on biological reasoning.
    
    Analyzes user intent and selects appropriate tools based on:
    - Analysis type (spatial vs functional vs comparative)
    - Data scope (single genome vs cross-genome vs global)
    - Biological intent (discovery vs lookup vs comparison)
    
    CRITICAL DECISION CRITERIA:
    
    Use 'whole_genome_reader' for:
    - Global prophage/phage discovery across ALL genomes
    - Spatial analysis requiring gene coordinate order
    - Discovery queries ("find", "explore", "discover")
    - Operon identification needing neighborhood context
    - Cross-genome spatial pattern analysis
    
    Use 'database_query' for:
    - Simple annotation lookups
    - Counting specific protein types
    - Direct searches for known functional categories
    - Questions with specific IDs already identified
    
    Use 'code_interpreter' for:
    - Statistical analysis of retrieved data
    - Visualization and plotting
    - Quantitative assessments after data retrieval
    
    Use 'genome_selector' for:
    - Targeting specific organisms/species mentioned by name
    - Resolving ambiguous organism references
    """
    
    user_query = dspy.InputField(desc="Original user question with full biological context")
    task_description = dspy.InputField(desc="Specific task to accomplish")
    available_tools = dspy.InputField(desc="Detailed tool capabilities with decision criteria")
    analysis_context = dspy.InputField(desc="Previous task context and workflow state")
    
    selected_tool = dspy.OutputField(desc="Exact tool name from available_tools, or 'database_query' for direct database queries")
    tool_parameters = dspy.OutputField(desc="Valid JSON object with parameters for the selected tool (no comments, proper JSON syntax)")
    biological_reasoning = dspy.OutputField(desc="Detailed biological rationale explaining WHY this tool was selected for this specific query")
    analysis_type = dspy.OutputField(desc="spatial_genomic|functional_annotation|comparative_analysis|database_lookup|statistical_analysis")
    confidence_score = dspy.OutputField(desc="Confidence level 0.0-1.0 in tool selection based on biological appropriateness")


# Removed BiologicalIntentClassifier - functionality integrated into BiologicalToolSelector


class IntelligentToolSelector:
    """
    LLM-first tool selection system using sophisticated biological reasoning.
    
    Features:
    - Pure LLM-based tool selection with no regex fallbacks
    - Rich biological context and decision criteria
    - Detailed reasoning capture for debugging and validation
    - Fail-fast approach - LLM is the authority
    
    The LLM has complete authority over tool selection based on:
    - Biological analysis type (spatial vs functional vs comparative)
    - Query scope (global vs targeted vs lookup)
    - Scientific intent (discovery vs annotation vs quantification)
    """
    
    def __init__(self):
        """Initialize the LLM-first tool selector."""
        self.model_allocator = get_model_allocator()
        
        # Use enhanced tool capabilities with decision criteria
        self.tool_capabilities = TOOL_CAPABILITIES
        
        if DSPY_AVAILABLE:
            logger.info("🧠 LLM-first tool selector initialized - gpt-5 has full authority over tool selection")
        else:
            logger.error("❌ DSPy not available - tool selection requires LLM capabilities")
            raise RuntimeError("LLM-first tool selector requires DSPy for biological reasoning")
    
    
    async def select_tool_for_task(self, 
                                 task_description: str,
                                 original_user_query: str,
                                 previous_task_context: str = "") -> ToolSelectionResult:
        """
        Select the appropriate tool using LLM biological reasoning.
        
        Args:
            task_description: Description of the task to perform
            original_user_query: Original user query for context preservation
            previous_task_context: Context from previous tasks in workflow
            
        Returns:
            ToolSelectionResult with selected tool and detailed biological reasoning
        """
        logger.info(f"🧠 LLM analyzing task for tool selection: {task_description[:100]}...")
        
        try:
            return await self._llm_based_selection(
                task_description, original_user_query, previous_task_context
            )
        except Exception as e:
            logger.error(f"💥 LLM tool selection failed: {type(e).__name__}: {e}")
            import traceback
            logger.debug(f"🐛 Full trace: {traceback.format_exc()}")
            
            # Fail fast - no fallback to regex patterns
            return ToolSelectionResult(
                selected_tool=None,
                tool_arguments={"description": task_description},
                reasoning=f"LLM tool selection failed: {str(e)}",
                confidence=0.0,
                fallback_used=True
            )
    
    async def _llm_based_selection(self, 
                                 task_description: str,
                                 original_user_query: str,
                                 previous_task_context: str) -> ToolSelectionResult:
        """Use LLM to select tool based on sophisticated biological reasoning."""
        
        # Prepare enhanced tool capabilities with decision criteria
        available_tools_json = json.dumps(self.tool_capabilities, indent=2)
        
        # Use model allocation for intelligent tool selection (complex biological reasoning = gpt-5)
        def selection_call(module):
            return module(
                user_query=original_user_query,
                task_description=task_description,
                available_tools=available_tools_json,
                analysis_context=previous_task_context
            )
        
        # Use gpt-5 for sophisticated biological tool selection
        logger.debug(f"🧠 o3_biological_reasoning: task='{task_description[:50]}...', query='{original_user_query[:50]}...'")
        result = self.model_allocator.create_context_managed_call(
            task_name="tool_selection",  # Maps to COMPLEX = gpt-5
            signature_class=BiologicalToolSelector,
            module_call_func=selection_call,
            query=original_user_query,
            task_context=task_description
        )
        
        logger.debug(f"🧠 o3_selection_result: type={type(result)}")
        
        if not result:
            logger.error("❌ o3_biological_reasoning: Model allocation returned None")
            raise Exception("Model allocation failed for biological tool selection")
        
        # Extract selection details
        selected_tool = getattr(result, 'selected_tool', None)
        tool_parameters = getattr(result, 'tool_parameters', '{}') 
        biological_reasoning = getattr(result, 'biological_reasoning', 'No reasoning provided')
        analysis_type = getattr(result, 'analysis_type', 'unknown')
        confidence_score = float(getattr(result, 'confidence_score', 0.8))
        
        logger.info(f"🧠 LLM selected tool: '{selected_tool}' (analysis: {analysis_type}, confidence: {confidence_score:.2f})")
        logger.debug(f"🎯 Biological reasoning: {biological_reasoning[:200]}...")
        
        # Parse tool parameters
        try:
            cleaned_json = self._clean_json_response(tool_parameters) if tool_parameters else "{}"
            tool_args = json.loads(cleaned_json)
            logger.debug(f"✅ Parsed {len(tool_args)} tool parameters")
        except json.JSONDecodeError as e:
            logger.warning(f"❌ JSON parse failed: {e}")
            logger.warning(f"📝 Raw parameters: {tool_parameters}")
            tool_args = {"description": task_description}
        
        # Validate selected tool
        if selected_tool and selected_tool not in AVAILABLE_TOOLS and selected_tool != "database_query":
            logger.warning(f"❌ Invalid tool '{selected_tool}' not in {list(AVAILABLE_TOOLS.keys())} - defaulting to database_query")
            selected_tool = "database_query"
        
        # Enhance tool arguments with intelligent defaults
        if selected_tool == "whole_genome_reader":
            # Check if this should be global analysis
            if self._should_use_global_analysis(original_user_query, task_description, biological_reasoning):
                tool_args["global_analysis"] = True
                logger.info(f"🌍 LLM-detected global analysis requirement - setting global_analysis=True")
            
            # Ensure spatial focus for discovery queries
            if analysis_type == "spatial_genomic" or "discovery" in biological_reasoning.lower():
                tool_args["focus_on_spatial"] = True
                tool_args["max_genes_per_contig"] = 10000  # Ensure complete genomic context
        
        # For database_query, return None as selected_tool to indicate ATOMIC_QUERY
        if selected_tool == "database_query":
            return ToolSelectionResult(
                selected_tool=None,  # None means ATOMIC_QUERY (database query)
                tool_arguments=tool_args,
                reasoning=f"[{analysis_type}] {biological_reasoning}",
                confidence=confidence_score,
                fallback_used=False
            )
        else:
            # Real tool selected
            return ToolSelectionResult(
                selected_tool=selected_tool,
                tool_arguments=tool_args,
                reasoning=f"[{analysis_type}] {biological_reasoning}",
                confidence=confidence_score,
                fallback_used=False
            )
    
    # Removed _regex_based_selection - LLM has full authority
    
    def _should_use_global_analysis(self, user_query: str, task_description: str, reasoning: str) -> bool:
        """
        Intelligent detection of global analysis requirements.
        
        Args:
            user_query: Original user question
            task_description: Current task description
            reasoning: LLM's biological reasoning
            
        Returns:
            True if this should be global analysis across all genomes
        """
        import re
        
        combined_text = f"{user_query} {task_description} {reasoning}".lower()
        
        # Strong indicators for global analysis
        global_indicators = [
            "across all genomes", "all genomes", "globally", "find prophage", 
            "discover prophage", "prophage discovery", "cross-genome",
            "comparative", "between genomes", "genome-wide", "global search",
            "explore all", "through all genomes", "entire dataset"
        ]
        
        # Check for global analysis indicators
        for indicator in global_indicators:
            if indicator in combined_text:
                logger.debug(f"🌍 Global analysis detected: '{indicator}' in query/reasoning")
                return True
        
        # Check if no specific genome mentioned and task is exploratory
        exploratory_words = ["find", "discover", "explore", "identify", "search"]
        specific_genome_pattern = r'genome[_\s]+[a-zA-Z0-9_]+'  # Pattern for specific genome IDs
        
        has_exploratory = any(word in combined_text for word in exploratory_words)
        has_specific_genome = bool(re.search(specific_genome_pattern, combined_text))
        
        if has_exploratory and not has_specific_genome:
            logger.debug(f"🌍 Global analysis inferred: exploratory query without specific genome")
            return True
            
        return False
    
    def _clean_json_response(self, json_str: str) -> str:
        """
        Clean LLM JSON responses by removing comments and fixing formatting.
        
        Args:
            json_str: Raw JSON string from LLM that may contain comments
            
        Returns:
            Cleaned JSON string that can be parsed
        """
        import re
        
        if not json_str or not json_str.strip():
            return "{}"
        
        # Remove line comments (// ...)
        json_str = re.sub(r'//.*?(?=\n|$)', '', json_str)
        
        # Remove block comments (/* ... */)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        return json_str.strip()


class CachedToolSelector:
    """
    Cached tool selection to reduce API calls while maintaining LLM intelligence.
    
    Rules:
    - Main tasks get full LLM tool selection (cached for sub-tasks)
    - Sub-tasks and chunks inherit tool selection from parent
    - Synthesis tasks get conditional tool selection only if needed
    """
    
    def __init__(self, base_selector: IntelligentToolSelector):
        self.base_selector = base_selector
        self.main_task_cache: Dict[str, ToolSelectionResult] = {}
        self.call_count = 0
        self.cache_hits = 0
    
    async def select_tool_for_task_with_caching(self, 
                                              task: Task,
                                              original_user_query: str,
                                              previous_task_context: str = "") -> ToolSelectionResult:
        """
        Smart tool selection with three-tier caching strategy.
        
        Args:
            task: Task object with hierarchy information
            original_user_query: Original user query
            previous_task_context: Context from previous tasks
            
        Returns:
            ToolSelectionResult with selection and caching metadata
        """
        logger.info(f"🧠 Tool selection request for task: {task.task_id} (main: {task.is_main_task})")
        
        # Rule 1: Main tasks get full LLM selection
        if task.is_main_task:
            logger.info(f"⚡ FULL LLM SELECTION for main task: {task.task_id}")
            self.call_count += 1
            
            result = await self.base_selector.select_tool_for_task(
                task_description=task.description,
                original_user_query=original_user_query,
                previous_task_context=previous_task_context
            )
            
            # Cache the decision for sub-tasks
            self.main_task_cache[task.task_id] = result
            task.tool_selection_result = result
            task.tool_selection_source = "planned"
            
            logger.info(f"✅ Cached main task selection: {result.selected_tool or 'database_query'}")
            return result
        
        # Rule 2: Sub-tasks inherit from parent
        if task.parent_task_id and task.parent_task_id in self.main_task_cache:
            logger.info(f"📋 INHERITING tool selection from parent: {task.parent_task_id}")
            self.cache_hits += 1
            
            inherited = self.main_task_cache[task.parent_task_id]
            
            # Create inherited result
            result = ToolSelectionResult(
                selected_tool=inherited.selected_tool,
                tool_arguments=inherited.tool_arguments.copy(),
                reasoning=f"Inherited from main task {task.parent_task_id}: {inherited.reasoning}",
                confidence=inherited.confidence,
                fallback_used=False
            )
            
            task.tool_selection_result = result
            task.tool_selection_source = "inherited"
            
            logger.info(f"✅ Inherited tool: {result.selected_tool or 'database_query'}")
            return result
        
        # Rule 3: Synthesis tasks get conditional selection
        if task.task_type == TaskType.SYNTHESIS:
            return await self._conditional_synthesis_selection(task, original_user_query, previous_task_context)
        
        # Fallback: Full selection (should be rare)
        logger.warning(f"⚠️ FALLBACK to full selection for task: {task.task_id}")
        self.call_count += 1
        
        result = await self.base_selector.select_tool_for_task(
            task_description=task.description,
            original_user_query=original_user_query,
            previous_task_context=previous_task_context
        )
        
        task.tool_selection_result = result
        task.tool_selection_source = "fallback"
        
        return result
    
    async def _conditional_synthesis_selection(self, 
                                             task: Task,
                                             original_user_query: str,
                                             previous_task_context: str) -> ToolSelectionResult:
        """
        Conditional tool selection for synthesis tasks.
        
        Only makes new LLM call if synthesis needs different tools than execution.
        """
        logger.info(f"🔄 CONDITIONAL synthesis selection for: {task.task_id}")
        
        # Check if synthesis task mentions different analysis needs
        synthesis_indicators = [
            "analyze", "statistical", "visualize", "plot", "compute", 
            "calculate", "aggregate", "summarize", "report", "interpret"
        ]
        
        task_text = task.description.lower()
        needs_analysis_tool = any(indicator in task_text for indicator in synthesis_indicators)
        
        if needs_analysis_tool:
            logger.info(f"📊 Synthesis needs analysis tool - making LLM call")
            self.call_count += 1
            
            result = await self.base_selector.select_tool_for_task(
                task_description=task.description,
                original_user_query=original_user_query,
                previous_task_context=previous_task_context
            )
            
            task.tool_selection_result = result
            task.tool_selection_source = "synthesized"
            
            return result
        else:
            # Synthesis doesn't need special tools - inherit or use database_query
            logger.info(f"📝 Simple synthesis - using database_query")
            self.cache_hits += 1
            
            result = ToolSelectionResult(
                selected_tool=None,  # database_query
                tool_arguments={"description": task.description},
                reasoning="Simple synthesis task - using database query for data retrieval",
                confidence=0.9,
                fallback_used=False
            )
            
            task.tool_selection_result = result
            task.tool_selection_source = "inherited"
            
            return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get caching statistics."""
        total_requests = self.call_count + self.cache_hits
        cache_hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "total_requests": total_requests,
            "llm_calls": self.call_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate_percent": cache_hit_rate,
            "api_call_reduction": f"{cache_hit_rate:.1f}% fewer API calls"
        }


# Global instances for easy access
_tool_selector = None
_cached_selector = None

def get_tool_selector() -> IntelligentToolSelector:
    """Get the global LLM-first tool selector instance."""
    global _tool_selector
    if _tool_selector is None:
        _tool_selector = IntelligentToolSelector()
    return _tool_selector

def get_cached_tool_selector() -> CachedToolSelector:
    """Get the global cached tool selector instance."""
    global _cached_selector
    if _cached_selector is None:
        base_selector = get_tool_selector()
        _cached_selector = CachedToolSelector(base_selector)
    return _cached_selector
