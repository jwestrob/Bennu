"""
Policy Engine for dynamic agent execution control.

Implements guard evaluation, stop conditions, and evidence assessment
without hard-coded biological constants. Uses generic rules based on
resolver targets and tool metrics.
"""

import logging
from typing import Dict, Any, List, Optional
from .models import Guard, StopCondition, Intent, ToolOutput

logger = logging.getLogger(__name__)


class PolicyEngine:
    """
    Policy engine for guard evaluation and evidence assessment.
    
    Provides generic rules for conclusiveness determination without
    hard-coded biological identifiers or domain-specific logic.
    """
    
    def __init__(self, settings=None):
        """
        Initialize policy engine.
        
        Args:
            settings: Settings instance for thresholds and configuration
        """
        self.settings = settings
        logger.info("🛡️ PolicyEngine initialized with generic rule evaluation")
    
    def evaluate_guard(self, guard: Guard, context: Dict[str, Any]) -> bool:
        """
        Evaluate guard predicate against execution context.
        
        Args:
            guard: Guard to evaluate
            context: Execution context with tool outputs, targets, etc.
            
        Returns:
            True if guard passes (tool should be eligible)
        """
        try:
            guard_name = guard.name
            guard_args = guard.args
            
            logger.debug(f"🛡️ Evaluating guard: {guard_name}")
            
            # Generic guard predicates
            if guard_name == "requires_anchor":
                return self._has_anchor_entities(context)
            elif guard_name == "requires_inconclusive":
                return self._is_evidence_inconclusive(context)
            elif guard_name == "cheap_first":
                return self._cheap_tools_attempted(context)
            elif guard_name == "requires_spatial_intent":
                return self._is_spatial_intent(context)
            else:
                logger.warning(f"Unknown guard: {guard_name}, defaulting to True")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error evaluating guard {guard.name}: {e}")
            return False
    
    def assess(self, intent: Intent, outputs: List[ToolOutput], targets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess evidence conclusiveness based on intent and tool outputs.
        
        Args:
            intent: Query intent classification
            outputs: Tool execution history
            targets: Resolved biological targets from SchemaResolver
            
        Returns:
            Dict with state ("conclusive_present"|"conclusive_absent"|"inconclusive"),
            confidence, and rationale
        """
        try:
            logger.debug(f"📊 Assessing evidence: {len(outputs)} outputs, intent={intent}")
            
            # Extract evidence metrics from tool outputs
            evidence_metrics = self._extract_evidence_metrics(outputs)
            
            # Apply intent-specific conclusiveness rules
            if intent == Intent.PRESENCE_ABSENCE:
                return self._assess_presence_absence(evidence_metrics, targets)
            elif intent == Intent.QUANTIFICATION:
                return self._assess_quantification(evidence_metrics, targets)
            elif intent == Intent.SPATIAL_NEIGHBORHOOD:
                return self._assess_spatial_analysis(evidence_metrics, targets)
            elif intent == Intent.NOVELTY_SCAN:
                return self._assess_novelty_scan(evidence_metrics, targets)
            else:  # GENERIC_QNA
                return self._assess_generic_query(evidence_metrics, targets)
                
        except Exception as e:
            logger.error(f"❌ Error assessing evidence: {e}")
            return {
                "state": "inconclusive",
                "confidence": 0.0,
                "rationale": f"Assessment error: {e}"
            }
    
    def _has_anchor_entities(self, context: Dict[str, Any]) -> bool:
        """Check if anchor entities exist for spatial analysis."""
        resolved_targets = context.get("resolved_targets", {})
        
        # Anchor types that enable spatial/neighborhood analysis
        anchor_types = ["proteins", "domains", "functions"]
        anchor_count = 0
        
        for t in anchor_types:
            target_value = resolved_targets.get(t, [])
            # Handle multiple value types safely
            try:
                if isinstance(target_value, (list, tuple)):
                    anchor_count += len(target_value)
                elif isinstance(target_value, int):
                    anchor_count += target_value
                elif isinstance(target_value, str) and target_value.strip():
                    anchor_count += 1  # Non-empty string counts as 1 anchor
                # Ignore None, empty strings, other types
            except Exception as e:
                logger.debug(f"Failed to process anchor type {t}: {target_value} - {e}")
                continue
        
        return anchor_count > 0
    
    def _is_evidence_inconclusive(self, context: Dict[str, Any]) -> bool:
        """Check if current evidence is inconclusive (allows expensive tools)."""
        outputs = context.get("tool_outputs", [])
        
        # If no tools executed yet, evidence is inconclusive
        if not outputs:
            return True
        
        # Check if any tool reported conclusive results
        for output in outputs:
            # Handle both dict and ToolOutput objects
            if hasattr(output, 'metrics'):
                metrics = output.metrics
            else:
                metrics = output.get("metrics", {})
            
            if metrics.get("conclusive", False):
                return False  # Evidence is conclusive, block expensive tools
        
        return True  # Evidence remains inconclusive
    
    def _cheap_tools_attempted(self, context: Dict[str, Any]) -> bool:
        """Check if cheap tools have been attempted before expensive ones."""
        outputs = context.get("tool_outputs", [])
        
        # Allow if no tools executed yet (first tool)
        if not outputs:
            return True
        
        # Check if at least one cheap tool was executed
        cheap_tools = {"database_query", "vector_search"}
        executed_tools = set()
        for output in outputs:
            if hasattr(output, 'tool'):
                executed_tools.add(output.tool)
            else:
                executed_tools.add(output.get("tool", ""))
        
        return bool(cheap_tools.intersection(executed_tools))
    
    def _is_spatial_intent(self, context: Dict[str, Any]) -> bool:
        """Check if query intent requires spatial analysis."""
        intent = context.get("intent", "")
        return intent == Intent.SPATIAL_NEIGHBORHOOD
    
    def _extract_evidence_metrics(self, outputs: List[ToolOutput]) -> Dict[str, Any]:
        """Extract evidence metrics from tool execution history."""
        metrics = {
            "total_tools": len(outputs),
            "successful_tools": sum(1 for out in outputs if out.success),
            "kg_hits": 0,
            "vector_hits": 0,
            "vector_max_similarity": 0.0,
            "spatial_regions_found": 0,
            "conclusive_tools": 0
        }
        
        for output in outputs:
            # Handle both dict and ToolOutput objects
            if hasattr(output, 'metrics'):
                tool_metrics = output.metrics
                tool_name = output.tool
            else:
                tool_metrics = output.get("metrics", {})
                tool_name = output.get("tool", "")
            
            # Aggregate evidence metrics across tools
            metrics["kg_hits"] += tool_metrics.get("kg_matches", 0)
            metrics["vector_hits"] += tool_metrics.get("vector_matches", 0) 
            
            # Track maximum similarity across vector searches
            similarity = tool_metrics.get("max_similarity", 0.0)
            if similarity > metrics["vector_max_similarity"]:
                metrics["vector_max_similarity"] = similarity
            
            # Count spatial regions from whole_genome_reader
            if tool_name == "whole_genome_reader":
                metrics["spatial_regions_found"] += tool_metrics.get("regions_found", 0)
            
            # Track tools that reported conclusive results
            if tool_metrics.get("conclusive", False):
                metrics["conclusive_tools"] += 1
        
        return metrics
    
    def _assess_presence_absence(self, metrics: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:
        """Assess presence/absence queries for conclusiveness."""
        kg_hits = metrics["kg_hits"]
        vector_max_sim = metrics["vector_max_similarity"]
        
        # Get similarity threshold from settings
        similarity_threshold = getattr(self.settings, "vector_hit_threshold", 0.7)
        
        # Conclusive present: Found matches in KG or high-similarity vector hits
        if kg_hits > 0 or vector_max_sim >= similarity_threshold:
            return {
                "state": "conclusive_present",
                "confidence": 0.9 if kg_hits > 0 else 0.8,
                "rationale": f"Found {kg_hits} KG matches, max vector similarity {vector_max_sim:.3f}"
            }
        
        # Conclusive absent: No KG matches AND low vector similarity AND cheap tools executed
        elif kg_hits == 0 and vector_max_sim < similarity_threshold and metrics["total_tools"] >= 2:
            return {
                "state": "conclusive_absent", 
                "confidence": 0.85,
                "rationale": f"No KG matches, max vector similarity {vector_max_sim:.3f} below threshold"
            }
        
        # Inconclusive: Need more evidence
        else:
            return {
                "state": "inconclusive",
                "confidence": 0.5,
                "rationale": "Insufficient evidence from initial searches"
            }
    
    def _assess_quantification(self, metrics: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quantification queries."""
        # Quantification typically needs multiple tools to provide counts
        if metrics["total_tools"] < 2:
            return {
                "state": "inconclusive",
                "confidence": 0.3,
                "rationale": "Quantification requires multiple data sources"
            }
        
        return {
            "state": "conclusive_present",
            "confidence": 0.8,
            "rationale": f"Quantification from {metrics['total_tools']} tools"
        }
    
    def _assess_spatial_analysis(self, metrics: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:
        """Assess spatial/neighborhood analysis queries."""
        regions_found = metrics["spatial_regions_found"]
        
        # Spatial analysis is conclusive if whole_genome_reader executed
        if regions_found > 0:
            return {
                "state": "conclusive_present",
                "confidence": 0.9,
                "rationale": f"Spatial analysis found {regions_found} genomic regions"
            }
        elif any(out.tool == "whole_genome_reader" for out in []):  # Spatial tool attempted
            return {
                "state": "conclusive_absent",
                "confidence": 0.8,
                "rationale": "Spatial analysis completed with no regions found"
            }
        else:
            return {
                "state": "inconclusive", 
                "confidence": 0.4,
                "rationale": "Spatial analysis not yet attempted"
            }
    
    def _assess_novelty_scan(self, metrics: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:
        """Assess novelty/discovery queries."""
        # Novelty scans typically need comprehensive analysis
        return {
            "state": "inconclusive",
            "confidence": 0.6,
            "rationale": "Novelty scans require comprehensive analysis"
        }
    
    def _assess_generic_query(self, metrics: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:
        """Assess generic Q&A queries."""
        total_evidence = metrics["kg_hits"] + metrics["vector_hits"]
        
        if total_evidence > 0:
            return {
                "state": "conclusive_present",
                "confidence": 0.75,
                "rationale": f"Found evidence from {total_evidence} sources"
            }
        else:
            return {
                "state": "inconclusive",
                "confidence": 0.5,
                "rationale": "Generic query needs additional evidence"
            }


# Global policy engine instance (following existing pattern)
_policy_engine = None


def get_policy_engine(settings=None) -> PolicyEngine:
    """Get global policy engine instance."""
    global _policy_engine
    if _policy_engine is None:
        _policy_engine = PolicyEngine(settings)
    return _policy_engine