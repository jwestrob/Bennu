#!/usr/bin/env python3
"""
DSPy signatures for genomic RAG system.
Defines structured prompting interfaces for LLM interactions.
"""

import logging
from typing import List, Optional

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    logging.warning("DSPy not available - install dsp-ml package")

logger = logging.getLogger(__name__)

if DSPY_AVAILABLE:
    
    class PlannerAgent(dspy.Signature):
        """
        Intelligent planning agent that determines if multi-step agentic execution is needed.
        
        Analyze the user's question to determine if it requires:
        1. Simple database lookup (traditional mode)
        2. Multi-step analysis with tools (agentic mode)
        
        Examples requiring agentic mode:
        - "Find proteins similar to X and analyze their functions" (similarity search + analysis)
        - "Compare metabolic capabilities between genomes" (multiple queries + comparison)
        - "Generate a report on CAZyme distribution" (query + code analysis + visualization)
        
        Examples for traditional mode:
        - "How many proteins are there?" (simple count)
        - "What is the function of protein X?" (direct lookup)
        - "Show me transport proteins" (single query)
        """
        
        user_query = dspy.InputField(desc="User's natural language question")
        requires_planning = dspy.OutputField(desc="Boolean: Does this query require multi-step agentic execution?")
        reasoning = dspy.OutputField(desc="Explanation of why agentic planning is or isn't needed")
        task_plan = dspy.OutputField(desc="If agentic: high-level task breakdown. If traditional: 'N/A'")
    
    class QueryClassifier(dspy.Signature):
        """
        Classify genomic queries into categories for appropriate retrieval strategy.
        
        Query types:
        - structural: Specific database lookups (protein IDs, gene locations, annotations)
        - semantic: Similarity-based searches (find proteins similar to X)
        - hybrid: Combination of both (find similar proteins and their locations)
        - functional: Function-based searches (transport proteins, metabolic enzymes)
        - comparative: Cross-genome comparisons
        """
        
        question = dspy.InputField(desc="User's question about genomic data")
        query_type = dspy.OutputField(desc="Query classification: structural, semantic, hybrid, functional, or comparative")
        reasoning = dspy.OutputField(desc="Explanation of classification")
        key_entities = dspy.OutputField(desc="Key biological entities mentioned (proteins, genes, functions, etc.)")
    
    class ContextRetriever(dspy.Signature):
        """
        Generate retrieval strategy for genomic queries based on classification and schema.
        
        Uses Neo4j schema to determine optimal query patterns and search strategies.
        Considers data scaling (small vs large result sets) and user intent.
        """
        
        db_schema = dspy.InputField(desc="Neo4j database schema with node types, relationships, and query patterns")
        question = dspy.InputField(desc="User's question")
        query_type = dspy.InputField(desc="Classified query type from QueryClassifier")
        
        search_strategy = dspy.OutputField(desc="Retrieval approach: direct_query, similarity_search, or hybrid_search")
        cypher_query = dspy.OutputField(desc="Generated Cypher query for Neo4j database")
        reasoning = dspy.OutputField(desc="Explanation of retrieval strategy choice")
        expected_result_size = dspy.OutputField(desc="Estimated result size: small, medium, or large")
    
    class GenomicAnswerer(dspy.Signature):
        """
        Generate comprehensive answers to genomic questions using retrieved context.
        
        Provides detailed biological interpretation with:
        - Scientific accuracy and proper terminology
        - Confidence assessment based on data quality
        - Relevant citations and data sources
        - Clear explanations for non-experts when appropriate
        """
        
        question = dspy.InputField(desc="Original user question")
        context = dspy.InputField(desc="Retrieved genomic data and annotations")
        answer = dspy.OutputField(desc="Comprehensive answer with biological insights")
        confidence = dspy.OutputField(desc="Confidence level: high, medium, or low")
        citations = dspy.OutputField(desc="Data sources and references used")
        
else:
    # Fallback classes when DSPy is not available
    
    class BaseMockSignature:
        """Base class for mock signatures when DSPy is unavailable."""
        
        def __init__(self, **kwargs):
            self.inputs = kwargs
            logger.warning("DSPy not available - using mock signature")
        
        def __call__(self, **kwargs):
            return type('MockResult', (), {
                'requires_planning': False,
                'reasoning': "DSPy not available",
                'task_plan': "N/A",
                'query_type': "structural", 
                'key_entities': [],
                'search_strategy': "direct_query",
                'cypher_query': "MATCH (n) RETURN count(n)",
                'expected_result_size': "small",
                'answer': "DSPy not available - install dsp-ml package",
                'confidence': "low",
                'citations': "Mock response"
            })()
    
    class PlannerAgent(BaseMockSignature):
        """Mock planner agent signature."""
        pass
    
    class QueryClassifier(BaseMockSignature):
        """Mock query classifier signature."""
        pass
    
    class ContextRetriever(BaseMockSignature):
        """Mock context retriever signature."""
        pass
    
    class GenomicAnswerer(BaseMockSignature):
        """Mock genomic answerer signature."""
        pass