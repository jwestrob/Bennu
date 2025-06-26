"""
TaskRepairAgent: Autonomous error detection and repair for genomic RAG queries.

This agent detects common error patterns in DSPy-generated queries and attempts
to repair them or provide helpful user messages.
"""

import logging
import re
from typing import Optional, List, Dict, Any

from .repair_types import RepairResult, RepairStrategy, SchemaInfo
from .error_patterns import ErrorPatternRegistry, RelationshipMapper, EntitySuggester

logger = logging.getLogger(__name__)


class TaskRepairAgent:
    """Autonomous error detection and repair for genomic RAG queries"""
    
    def __init__(self, schema_info: Optional[SchemaInfo] = None):
        """
        Initialize the TaskRepairAgent
        
        Args:
            schema_info: Information about the Neo4j database schema
        """
        self.schema_info = schema_info or SchemaInfo.default_genomic_schema()
        self.error_registry = ErrorPatternRegistry()
        self.relationship_mapper = RelationshipMapper()
        self.entity_suggester = EntitySuggester()
        
        logger.info("TaskRepairAgent initialized with schema info")
    
    def detect_and_repair(self, query: str, error: Exception) -> RepairResult:
        """
        Main entry point - detect error type and attempt repair
        
        Args:
            query: The original query that failed
            error: The exception that was raised
            
        Returns:
            RepairResult with repair attempt details
        """
        error_message = str(error)
        logger.info(f"TaskRepairAgent analyzing error: {error_message[:100]}...")
        
        # Find matching error patterns
        matching_patterns = self.error_registry.find_matching_patterns(query, error_message)
        
        if not matching_patterns:
            logger.warning("No matching error patterns found")
            return self._create_fallback_result(query, error_message)
        
        # Try repair strategies in order of confidence
        matching_patterns.sort(key=lambda p: p.confidence_threshold, reverse=True)
        
        for pattern in matching_patterns:
            logger.info(f"Attempting repair with strategy: {pattern.repair_strategy}")
            
            try:
                result = self._apply_repair_strategy(pattern, query, error_message)
                if result.success:
                    logger.info(f"Repair successful using strategy: {pattern.repair_strategy}")
                    return result
            except Exception as repair_error:
                logger.error(f"Repair strategy {pattern.repair_strategy} failed: {repair_error}")
                continue
        
        # All repair strategies failed
        logger.warning("All repair strategies failed")
        return self._create_fallback_result(query, error_message)
    
    def _apply_repair_strategy(self, pattern, query: str, error_message: str) -> RepairResult:
        """Apply a specific repair strategy"""
        
        if pattern.repair_strategy == RepairStrategy.COMMENT_QUERY_EXPLANATION:
            return self._repair_comment_query(query, error_message)
        
        elif pattern.repair_strategy == RepairStrategy.INVALID_ENTITY_SUGGESTION:
            return self._repair_invalid_entity(query, error_message)
        
        elif pattern.repair_strategy == RepairStrategy.RELATIONSHIP_MAPPING:
            return self._repair_invalid_relationship(query, error_message)
        
        elif pattern.repair_strategy == RepairStrategy.SCHEMA_VALIDATION:
            return self._repair_schema_validation(query, error_message)
        
        elif pattern.repair_strategy == RepairStrategy.PARAMETER_SUBSTITUTION:
            return self._repair_parameter_missing(query, error_message)
        
        else:
            return self._create_fallback_result(query, error_message)
    
    def _repair_comment_query(self, query: str, error_message: str) -> RepairResult:
        """
        Repair DSPy-generated comment queries
        
        Example:
        Input: /* No valid query can be constructed: label `FakeNode` is not part of the graph schema */
        Output: User-friendly message with suggestions
        """
        logger.debug("Repairing comment query")
        
        # Extract entity name from comment
        invalid_entity = self.entity_suggester.extract_entity_from_comment(query)
        
        # Get suggestions for valid alternatives
        suggestions = self.entity_suggester.suggest_alternatives(
            invalid_entity, 
            self.schema_info.node_labels,
            max_suggestions=3
        )
        
        # Create user-friendly message
        if invalid_entity != "Unknown":
            user_message = (
                f"The entity type '{invalid_entity}' doesn't exist in our genomic database. "
                f"Available entity types include: {', '.join(suggestions)}. "
                f"You might want to try searching for proteins, genes, or domains instead."
            )
        else:
            user_message = (
                f"The requested entity type is not available in our genomic database. "
                f"Available entity types include: {', '.join(self.schema_info.node_labels)}. "
                f"Try searching for proteins, genes, or functional domains."
            )
        
        return RepairResult(
            success=True,
            repaired_query=None,  # No query repair, just user message
            user_message=user_message,
            suggested_alternatives=suggestions,
            confidence=0.9,
            repair_strategy_used=RepairStrategy.COMMENT_QUERY_EXPLANATION,
            original_error=error_message
        )
    
    def _repair_invalid_entity(self, query: str, error_message: str) -> RepairResult:
        """Repair queries with invalid entity references"""
        logger.debug("Repairing invalid entity reference")
        
        # Extract entity name from error message or query
        entity_patterns = [
            r"Variable `(\w+)` not defined",
            r"MATCH \((\w+):",
            r":\s*(\w+)\s*\)"
        ]
        
        invalid_entity = None
        for pattern in entity_patterns:
            match = re.search(pattern, query + " " + error_message)
            if match:
                invalid_entity = match.group(1)
                break
        
        if not invalid_entity:
            return RepairResult(success=False, original_error=error_message)
        
        # Get suggestions
        suggestions = self.entity_suggester.suggest_alternatives(
            invalid_entity,
            self.schema_info.node_labels
        )
        
        user_message = (
            f"The entity '{invalid_entity}' is not recognized. "
            f"Did you mean one of these: {', '.join(suggestions)}?"
        )
        
        return RepairResult(
            success=True,
            user_message=user_message,
            suggested_alternatives=suggestions,
            confidence=0.8,
            repair_strategy_used=RepairStrategy.INVALID_ENTITY_SUGGESTION,
            original_error=error_message
        )
    
    def _repair_invalid_relationship(self, query: str, error_message: str) -> RepairResult:
        """Repair queries with invalid relationship types"""
        logger.debug("Repairing invalid relationship")
        
        # Find invalid relationships in query
        invalid_relationships = []
        for invalid_rel in self.relationship_mapper.RELATIONSHIP_MAPPINGS.keys():
            if invalid_rel in query:
                invalid_relationships.append(invalid_rel)
        
        if not invalid_relationships:
            return RepairResult(success=False, original_error=error_message)
        
        # Repair the query by mapping relationships
        repaired_query = query
        for invalid_rel in invalid_relationships:
            valid_rel = self.relationship_mapper.map_relationship(invalid_rel)
            repaired_query = repaired_query.replace(invalid_rel, valid_rel)
        
        user_message = (
            f"I've mapped the relationship(s) {', '.join(invalid_relationships)} "
            f"to valid genomic relationships. Here are the results:"
        )
        
        return RepairResult(
            success=True,
            repaired_query=repaired_query,
            user_message=user_message,
            suggested_alternatives=self.relationship_mapper.get_valid_relationships(),
            confidence=0.85,
            repair_strategy_used=RepairStrategy.RELATIONSHIP_MAPPING,
            original_error=error_message
        )
    
    def _repair_schema_validation(self, query: str, error_message: str) -> RepairResult:
        """Handle general schema validation errors"""
        logger.debug("Attempting schema validation repair")
        
        # Check if it's a comment query that slipped through
        if query.strip().startswith("/*") and query.strip().endswith("*/"):
            return self._repair_comment_query(query, error_message)
        
        # Generic schema validation message
        user_message = (
            f"There was an issue with the database query structure. "
            f"Our genomic database contains information about: "
            f"{', '.join(self.schema_info.node_labels)}. "
            f"Please try rephrasing your question to focus on proteins, genes, or functional domains."
        )
        
        return RepairResult(
            success=True,
            user_message=user_message,
            suggested_alternatives=self.schema_info.node_labels,
            confidence=0.6,
            repair_strategy_used=RepairStrategy.SCHEMA_VALIDATION,
            original_error=error_message
        )
    
    def _repair_parameter_missing(self, query: str, error_message: str) -> RepairResult:
        """Repair queries with missing parameters"""
        logger.debug("Repairing parameter missing error")
        
        # Extract parameter name from error message
        import re
        match = re.search(r"Expected parameter\(s\): (\w+)", error_message)
        if not match:
            return RepairResult(success=False, original_error=error_message)
        
        parameter_name = match.group(1)
        
        user_message = (
            f"The query generated by our system used a parameter (${parameter_name}) without providing values. "
            f"This is a technical issue with query generation. Please try rephrasing your question, "
            f"or ask for specific proteins, genes, or functional categories instead of complex multi-step queries."
        )
        
        return RepairResult(
            success=True,
            user_message=user_message,
            suggested_alternatives=["Try simpler queries", "Ask for specific protein names", "Search by functional category"],
            confidence=0.8,
            repair_strategy_used=RepairStrategy.PARAMETER_SUBSTITUTION,
            original_error=error_message
        )
    
    def _create_fallback_result(self, query: str, error_message: str) -> RepairResult:
        """Create a fallback result when no repair strategies work"""
        user_message = (
            f"I encountered a technical issue while processing your query. "
            f"Our genomic database contains information about proteins, genes, and functional domains. "
            f"Please try rephrasing your question or ask about specific genomic features."
        )
        
        return RepairResult(
            success=False,
            user_message=user_message,
            suggested_alternatives=self.schema_info.node_labels,
            confidence=0.3,
            repair_strategy_used=RepairStrategy.FALLBACK_MESSAGE,
            original_error=error_message
        )
    
    def get_schema_summary(self) -> Dict[str, Any]:
        """Get a summary of the current schema information"""
        return {
            "node_labels": self.schema_info.node_labels,
            "relationship_types": self.schema_info.relationship_types,
            "total_patterns": len(self.error_registry.patterns),
            "available_strategies": [strategy.value for strategy in RepairStrategy]
        }