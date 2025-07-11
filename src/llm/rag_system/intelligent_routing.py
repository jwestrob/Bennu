#!/usr/bin/env python3
"""
Intelligent routing system for traditional vs agentic query processing.
Provides better heuristics and context for query complexity assessment.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

class QueryScope(Enum):
    """Query scope patterns."""
    SINGLE_ENTITY = "single_entity"
    ENTITY_LIST = "entity_list"
    CROSS_GENOME = "cross_genome"
    ANALYTICAL = "analytical"

@dataclass
class QueryAnalysis:
    """Analysis of query characteristics."""
    complexity: QueryComplexity
    scope: QueryScope
    expected_result_size: str  # "small", "medium", "large"
    requires_computation: bool
    requires_comparison: bool
    requires_visualization: bool
    genome_context: Optional[str]
    confidence: float
    reasoning: str

class IntelligentRouter:
    """
    Intelligent routing system that makes better decisions about traditional vs agentic execution.
    
    Uses rule-based heuristics combined with pattern matching to determine:
    1. Query complexity and scope
    2. Expected result size
    3. Computational requirements
    4. Appropriate execution strategy
    """
    
    def __init__(self):
        self.simple_patterns = [
            r"how many\s+\w+",
            r"what is\s+the\s+function",
            r"show me\s+\w+",
            r"list\s+all\s+\w+",
            r"find\s+protein\s+\w+",
            r"what does\s+\w+\s+do"
        ]
        
        self.complex_patterns = [
            r"compare\s+\w+\s+between",
            r"analyze\s+the\s+distribution", 
            r"generate\s+a\s+report",
            r"provide\s+a\s+full\s+report",
            r"comprehensive\s+report",
            r"detailed\s+report",
            r"create\s+a\s+visualization",
            r"find\s+similar\s+proteins\s+and",
            r"what\s+are\s+the\s+differences",
            r"statistical\s+analysis", 
            r"clustering\s+analysis",
            r"look\s+through\s+all\s+available\s+data",
            r"all\s+available\s+data",
            r"comprehensive\s+analysis",
            r"full\s+analysis",
            r"generate.*comprehensive",
            r"conflicting\s+functions.*report",
            r"multiple\s+databases.*report"
        ]
        
        self.computation_indicators = [
            "distribution", "statistics", "analysis", "comparison", 
            "clustering", "correlation", "pattern", "trend",
            "composition", "similarity", "difference", "ratio"
        ]
        
        self.visualization_indicators = [
            "plot", "graph", "chart", "visualization", "figure",
            "diagram", "heatmap", "histogram", "scatter"
        ]
        
        self.genome_patterns = [
            r"in\s+genome\s+(\w+)",
            r"from\s+(\w+)\s+genome",
            r"(\w+)\s+genomes?",
            r"across\s+genomes?",
            r"between\s+genomes?"
        ]
    
    def analyze_query(self, question: str) -> QueryAnalysis:
        """
        Analyze query characteristics to determine execution strategy.
        
        Args:
            question: User's natural language question
            
        Returns:
            QueryAnalysis with routing recommendations
        """
        logger.info(f"🔍 Analyzing query: {question[:50]}...")
        
        question_lower = question.lower()
        
        # Analyze complexity
        complexity = self._assess_complexity(question_lower)
        
        # Analyze scope
        scope = self._assess_scope(question_lower)
        
        # Check computational requirements
        requires_computation = any(indicator in question_lower for indicator in self.computation_indicators)
        requires_comparison = any(word in question_lower for word in ["compare", "difference", "versus", "vs", "between"])
        requires_visualization = any(indicator in question_lower for indicator in self.visualization_indicators)
        
        # Extract genome context
        genome_context = self._extract_genome_context(question)
        
        # Estimate result size
        expected_result_size = self._estimate_result_size(question_lower, scope)
        
        # Calculate confidence and reasoning
        confidence, reasoning = self._calculate_confidence(
            complexity, scope, requires_computation, requires_comparison, requires_visualization
        )
        
        analysis = QueryAnalysis(
            complexity=complexity,
            scope=scope,
            expected_result_size=expected_result_size,
            requires_computation=requires_computation,
            requires_comparison=requires_comparison,
            requires_visualization=requires_visualization,
            genome_context=genome_context,
            confidence=confidence,
            reasoning=reasoning
        )
        
        # Store original question for routing decisions
        analysis._original_question = question
        
        logger.info(f"📊 Query analysis: {complexity.value} complexity, {scope.value} scope")
        logger.info(f"💭 Reasoning: {reasoning}")
        
        return analysis
    
    def should_use_agentic_mode(self, analysis: QueryAnalysis) -> bool:
        """
        Determine if agentic mode should be used based on analysis.
        
        Args:
            analysis: Query analysis results
            
        Returns:
            True if agentic mode should be used, False for traditional mode
        """
        # SPECIAL CASE: Force CAZyme queries to use traditional mode
        # The agentic mode has broken query generation for CAZyme-specific data
        question_lower = getattr(analysis, '_original_question', '').lower()
        cazyme_terms = ['cazyme', 'carbohydrate', 'glycoside', 'carbohydrate-active', 'dbcan']
        if any(term in question_lower for term in cazyme_terms):
            logger.info("🧬 CAZyme query detected - forcing traditional mode for proper query generation")
            return False
        
        # SPECIAL CASE: Force BGC queries to use traditional mode
        # The agentic mode loses context and generates unrelated KEGG queries
        bgc_terms = ['bgc', 'biosynthetic', 'gene cluster', 'secondary metabolite', 'natural product', 'polyketide', 'nrps', 'terpene']
        if any(term in question_lower for term in bgc_terms):
            logger.info("🧬 BGC query detected - forcing traditional mode for proper query generation")
            return False
        
        # SPECIAL CASE: Force transport system queries to use traditional mode
        # The agentic mode generates overly complex task plans that timeout
        transport_terms = ['transport', 'transporter', 'abc transport', 'metal transport', 'iron transport', 'permease', 'channel']
        if any(term in question_lower for term in transport_terms):
            logger.info("🚛 Transport system query detected - forcing traditional mode for efficiency")
            return False
        
        # SPECIAL CASE: Force metabolic pathway queries to use traditional mode
        # The agentic mode generates massive contexts (>1M tokens) for pathway reconstruction
        pathway_terms = ['glycolysis', 'pathway', 'tca cycle', 'metabolism', 'metabolic', 'carbon fixation', 'biosynthesis', 'degradation']
        if any(term in question_lower for term in pathway_terms):
            logger.info("🧪 Metabolic pathway query detected - forcing traditional mode for efficiency")
            return False
        
        # Always use agentic mode for complex queries
        if analysis.complexity == QueryComplexity.COMPLEX:
            return True
        
        # Use agentic mode for medium complexity with computation requirements
        if analysis.complexity == QueryComplexity.MEDIUM and (
            analysis.requires_computation or 
            analysis.requires_comparison or 
            analysis.requires_visualization
        ):
            return True
        
        # Use agentic mode for cross-genome analysis
        if analysis.scope == QueryScope.CROSS_GENOME:
            return True
        
        # Use agentic mode for analytical queries regardless of complexity
        if analysis.scope == QueryScope.ANALYTICAL:
            return True
        
        # Use agentic mode for large expected result sizes that need processing
        if analysis.expected_result_size == "large" and analysis.requires_computation:
            return True
        
        # Default to traditional mode
        return False
    
    def _assess_complexity(self, question_lower: str) -> QueryComplexity:
        """Assess query complexity based on patterns."""
        # Check for simple patterns
        for pattern in self.simple_patterns:
            if re.search(pattern, question_lower):
                return QueryComplexity.SIMPLE
        
        # Check for complex patterns
        for pattern in self.complex_patterns:
            if re.search(pattern, question_lower):
                return QueryComplexity.COMPLEX
        
        # Count complexity indicators
        complexity_indicators = [
            "and", "then", "also", "furthermore", "additionally",
            "relationship", "correlation", "pattern", "trend"
        ]
        
        indicator_count = sum(1 for indicator in complexity_indicators if indicator in question_lower)
        
        if indicator_count >= 3:
            return QueryComplexity.COMPLEX
        elif indicator_count >= 1:
            return QueryComplexity.MEDIUM
        else:
            return QueryComplexity.SIMPLE
    
    def _assess_scope(self, question_lower: str) -> QueryScope:
        """Assess query scope based on patterns."""
        # Check for analytical scope
        if any(word in question_lower for word in ["analyze", "analysis", "statistical", "distribution"]):
            return QueryScope.ANALYTICAL
        
        # Check for cross-genome scope
        if any(word in question_lower for word in ["across", "between", "compare", "genomes"]):
            return QueryScope.CROSS_GENOME
        
        # Check for entity list scope
        if any(word in question_lower for word in ["all", "list", "show", "find all"]):
            return QueryScope.ENTITY_LIST
        
        # Default to single entity
        return QueryScope.SINGLE_ENTITY
    
    def _extract_genome_context(self, question: str) -> Optional[str]:
        """Extract genome identifiers from question."""
        for pattern in self.genome_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return match.group(1) if match.groups() else None
        return None
    
    def _estimate_result_size(self, question_lower: str, scope: QueryScope) -> str:
        """Estimate expected result size."""
        # Large result indicators
        if any(word in question_lower for word in ["all", "every", "total", "complete"]):
            return "large"
        
        # Medium result indicators
        if scope in [QueryScope.ENTITY_LIST, QueryScope.CROSS_GENOME]:
            return "medium"
        
        # Small result indicators
        if any(word in question_lower for word in ["specific", "particular", "single", "one"]):
            return "small"
        
        # Default based on scope
        if scope == QueryScope.ANALYTICAL:
            return "large"
        elif scope == QueryScope.CROSS_GENOME:
            return "medium"
        else:
            return "small"
    
    def _calculate_confidence(self, complexity: QueryComplexity, scope: QueryScope, 
                            requires_computation: bool, requires_comparison: bool, 
                            requires_visualization: bool) -> Tuple[float, str]:
        """Calculate confidence in routing decision and provide reasoning."""
        
        confidence = 0.7  # Base confidence
        reasoning_parts = []
        
        # Complexity-based confidence
        if complexity == QueryComplexity.SIMPLE:
            confidence += 0.2
            reasoning_parts.append("Simple query pattern detected")
        elif complexity == QueryComplexity.COMPLEX:
            confidence += 0.2
            reasoning_parts.append("Complex query pattern detected")
        
        # Scope-based confidence
        if scope == QueryScope.SINGLE_ENTITY:
            confidence += 0.1
            reasoning_parts.append("Single entity scope")
        elif scope == QueryScope.ANALYTICAL:
            confidence += 0.1
            reasoning_parts.append("Analytical scope detected")
        
        # Requirement-based confidence
        if requires_computation:
            confidence += 0.1
            reasoning_parts.append("Computational requirements detected")
        
        if requires_comparison:
            confidence += 0.1
            reasoning_parts.append("Comparison requirements detected")
        
        if requires_visualization:
            confidence += 0.1
            reasoning_parts.append("Visualization requirements detected")
        
        # Cap confidence at 1.0
        confidence = min(confidence, 1.0)
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Default heuristics applied"
        
        return confidence, reasoning
    
    def get_routing_recommendation(self, question: str) -> Dict[str, Any]:
        """
        Get comprehensive routing recommendation for a query.
        
        Args:
            question: User's natural language question
            
        Returns:
            Dictionary with routing recommendation and analysis
        """
        analysis = self.analyze_query(question)
        use_agentic = self.should_use_agentic_mode(analysis)
        
        return {
            "use_agentic_mode": use_agentic,
            "analysis": analysis,
            "recommendation": "agentic" if use_agentic else "traditional",
            "reasoning": analysis.reasoning,
            "confidence": analysis.confidence,
            "expected_result_size": analysis.expected_result_size,
            "genome_context": analysis.genome_context
        }