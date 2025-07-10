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
        
        CRITICAL: Generate ONLY executable Cypher queries with NO COMMENTS.
        
        FORBIDDEN:
        - Comments like /* comment */ or // comment
        - Multiple MATCH statements
        - Section headers or explanatory text
        - Semicolons separating queries
        
        REQUIRED: Start directly with MATCH, WITH, or OPTIONAL MATCH.
        
        CAZYME QUERY DETECTION - MANDATORY:
        When user mentions CAZyme, carbohydrate, glycoside, carbohydrate-active, dbCAN:
        - ALWAYS use Cazymeannotation and Cazymefamily nodes (NOT Domain/PFAM)
        - Use Pattern 4 from schema: (p:Protein)-[:HASCAZYME]->(ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily)
        - Include cf.familyId, cf.cazymeType, ca.substrateSpecificity for family details
        - Connect to genome via: (p)-[:ENCODEDBY]->(g:Gene)-[:BELONGSTOGENOME]->(genome:Genome)
        
        TRANSPORT QUERY DETECTION:
        When user mentions transport, transporter, channel, permease:
        - Use Pattern 1 from schema: KEGG-based transport search
        - Filter by: WHERE toLower(ko.description) CONTAINS 'transport'
        
        COMPARATIVE QUERY RULES - NEVER USE LIMIT FOR THESE PATTERNS:
        - "Which genomes" → Show ALL genomes for comparison
        - "Compare across genomes" → Show ALL genomes
        - "Most/least among genomes" → Show ALL genomes ranked
        - "For each genome" → Show ALL genomes with counts
        - "All genomes" → Show ALL genomes
        - "Across all genomes" → Show ALL genomes
        - "Between genomes" → Show ALL genomes
        - "What genome has the most" → Show ALL genomes (let user see ranking)
        
        ONLY USE LIMIT 1 FOR:
        - "Show me ONE example"
        - "Just give me the top one"
        - "Only the best result"
        - "A single protein"
        
        EXAMPLES:
        ❌ BAD: "Which genome has most metal transporters?" → LIMIT 1
        ✅ GOOD: "Which genome has most metal transporters?" → ORDER BY count DESC (no LIMIT)
        
        ❌ BAD: "For each genome, how many transport proteins?" → LIMIT 1  
        ✅ GOOD: "For each genome, how many transport proteins?" → GROUP BY genome (no LIMIT)
        """
        
        db_schema = dspy.InputField(desc="Neo4j database schema with node types, relationships, and query patterns")
        question = dspy.InputField(desc="User's question")
        query_type = dspy.InputField(desc="Classified query type from QueryClassifier")
        
        search_strategy = dspy.OutputField(desc="Retrieval approach: direct_query, similarity_search, or hybrid_search")
        cypher_query = dspy.OutputField(desc="EXECUTABLE Cypher query with NO COMMENTS - must start with MATCH/WITH/OPTIONAL")
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
    
    class GenomicSummarizer(dspy.Signature):
        """
        Summarize large genomic datasets while preserving essential biological information.
        
        Designed for context compression in multi-stage RAG systems where large datasets
        need to be condensed to fit within token limits while maintaining scientific accuracy.
        
        Focus areas:
        - Functional annotations (KEGG, PFAM domains)
        - Unique biological insights and patterns
        - Quantitative summaries (counts, distributions)
        - Key examples representative of larger datasets
        """
        
        genomic_data = dspy.InputField(desc="Large genomic dataset to summarize")
        target_length = dspy.InputField(desc="Target length: brief, medium, or detailed")
        focus_areas = dspy.InputField(desc="Specific biological aspects to emphasize")
        summary = dspy.OutputField(desc="Concise summary preserving essential biological information")
        key_findings = dspy.OutputField(desc="Most important biological insights from the data")
        data_statistics = dspy.OutputField(desc="Quantitative summary of the dataset")
    
    class NotingDecision(dspy.Signature):
        """
        Decide whether task results warrant note-taking based on information value.
        
        Only record notes for:
        - Significant biological insights or patterns
        - Unexpected or contradictory findings
        - Cross-genome comparisons with clear differences
        - Quantitative results that inform broader analysis
        - Findings that connect to other tasks or validate/contradict previous results
        
        Skip notes for:
        - Routine lookups with expected results
        - Single data points without broader context
        - Redundant information already captured
        - Low-confidence or unclear results
        - Failed queries or error conditions
        
        Provide detailed observations and cross-task connections when recording notes.
        """
        
        task_description = dspy.InputField(desc="Description of the task that was executed")
        execution_result = dspy.InputField(desc="Results from task execution (structured data, context, etc.)")
        existing_notes = dspy.InputField(desc="Summary of notes from previous tasks in this session")
        
        should_record = dspy.OutputField(desc="Boolean: Should we record notes for this task?")
        importance_score = dspy.OutputField(desc="Importance score 1-10 for this information")
        reasoning = dspy.OutputField(desc="Explanation of note-taking decision")
        observations = dspy.OutputField(desc="If recording: key observations worth noting (list format)")
        key_findings = dspy.OutputField(desc="If recording: important biological findings or patterns")
        cross_connections = dspy.OutputField(desc="If recording: connections to other tasks (task_id:connection_type:description format)")
        quantitative_data = dspy.OutputField(desc="If recording: important numerical data or metrics")
    
    class ReportPartGenerator(dspy.Signature):
        """
        Generate a specific part of a multi-part genomic report.
        
        This signature handles individual sections of large reports that have been
        chunked to manage token limits while maintaining scientific rigor and
        comprehensive analysis.
        
        Each part should:
        - Be self-contained but reference overall context
        - Maintain consistent terminology and formatting
        - Include quantitative analysis and biological insights
        - Connect to broader themes when relevant
        """
        
        question = dspy.InputField(desc="Original user question for context")
        data_chunk = dspy.InputField(desc="Data subset for this specific report part")
        part_context = dspy.InputField(desc="Context about this part's role in the overall report")
        previous_parts_summary = dspy.InputField(desc="Summary of previous parts to maintain consistency")
        report_type = dspy.InputField(desc="Type of report (e.g., 'crispr_systems', 'functional_comparison')")
        
        part_content = dspy.OutputField(desc="Detailed scientific content for this report part")
        key_findings = dspy.OutputField(desc="Key biological findings from this part")
        quantitative_summary = dspy.OutputField(desc="Quantitative metrics and statistics")
        connections_to_whole = dspy.OutputField(desc="How this part connects to overall analysis")
    
    class ExecutiveSummaryGenerator(dspy.Signature):
        """
        Generate executive summary for multi-part genomic reports.
        
        Creates a concise overview that captures the essence of large-scale
        genomic analyses while highlighting key findings and providing roadmap
        for detailed sections.
        
        Should include:
        - Overall scope and methodology
        - Key quantitative findings
        - Major biological insights
        - Navigation guide for detailed sections
        """
        
        question = dspy.InputField(desc="Original user question")
        data_overview = dspy.InputField(desc="Statistical overview of the complete dataset")
        key_patterns = dspy.InputField(desc="Major patterns identified across all data")
        report_structure = dspy.InputField(desc="Structure of the multi-part report")
        
        executive_summary = dspy.OutputField(desc="Concise executive summary of findings")
        scope_and_methodology = dspy.OutputField(desc="Overview of analysis scope and approach")
        key_statistics = dspy.OutputField(desc="Essential quantitative findings")
        navigation_guide = dspy.OutputField(desc="Guide to navigating the detailed report parts")
    
    class ReportSynthesisGenerator(dspy.Signature):
        """
        Generate final synthesis section for multi-part reports.
        
        Integrates findings from all report parts into coherent conclusions,
        identifies cross-cutting themes, and provides biological interpretation
        and recommendations for future work.
        
        Should provide:
        - Integration of all parts
        - Cross-cutting biological insights
        - Evolutionary and ecological implications
        - Recommendations for follow-up studies
        """
        
        question = dspy.InputField(desc="Original user question")
        all_parts_summary = dspy.InputField(desc="Summary of findings from all report parts")
        cross_cutting_themes = dspy.InputField(desc="Themes that emerge across multiple parts")
        quantitative_integration = dspy.InputField(desc="Integrated quantitative analysis")
        
        synthesis_content = dspy.OutputField(desc="Comprehensive synthesis of all findings")
        biological_implications = dspy.OutputField(desc="Broader biological and evolutionary implications")
        recommendations = dspy.OutputField(desc="Recommendations for future research or applications")
        confidence_assessment = dspy.OutputField(desc="Assessment of confidence in conclusions")
        
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
                'citations': "Mock response",
                'summary': "DSPy not available - using fallback summary",
                'key_findings': "No findings available without DSPy",
                'data_statistics': "Statistics unavailable"
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
    
    class GenomicSummarizer(BaseMockSignature):
        """Mock genomic summarizer signature."""
        pass
    
    class NotingDecision(BaseMockSignature):
        """Mock noting decision signature."""
        
        def __call__(self, **kwargs):
            return type('MockResult', (), {
                'should_record': False,
                'importance_score': 5.0,
                'reasoning': "DSPy not available - note-taking disabled",
                'observations': [],
                'key_findings': [],
                'cross_connections': [],
                'quantitative_data': {}
            })()
    
    class ReportPartGenerator(BaseMockSignature):
        """Mock report part generator signature."""
        
        def __call__(self, **kwargs):
            return type('MockResult', (), {
                'part_content': "DSPy not available - multi-part reports disabled",
                'key_findings': [],
                'quantitative_summary': "No statistics available",
                'connections_to_whole': "Mock report part"
            })()
    
    class ExecutiveSummaryGenerator(BaseMockSignature):
        """Mock executive summary generator signature."""
        
        def __call__(self, **kwargs):
            return type('MockResult', (), {
                'executive_summary': "DSPy not available - summary generation disabled",
                'scope_and_methodology': "Mock analysis scope",
                'key_statistics': "No statistics available",
                'navigation_guide': "Mock navigation guide"
            })()
    
    class ReportSynthesisGenerator(BaseMockSignature):
        """Mock report synthesis generator signature."""
        
        def __call__(self, **kwargs):
            return type('MockResult', (), {
                'synthesis_content': "DSPy not available - synthesis disabled",
                'biological_implications': "Mock implications",
                'recommendations': "Mock recommendations",
                'confidence_assessment': "low"
            })()