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
        
        **STRONGLY PREFER AGENTIC MODE** for queries that would benefit from:
        - Cross-database integration (KEGG + PFAM + CAZyme + BGC data)
        - Statistical analysis or visualization
        - Comprehensive reports combining multiple data sources
        - Comparative analysis across genomes
        - Literature integration for functional annotation
        - Large dataset analysis requiring chunking and synthesis
        - Dataset exploration and discovery ("look through", "find interesting", "what stands out")
        - Multi-step reasoning with justification ("present with justifications", "explain why")
        - Novelty detection and pattern recognition
        
        Examples requiring agentic mode:
        - "Find proteins similar to X and analyze their functions" (similarity search + analysis)
        - "Compare metabolic capabilities between genomes" (multiple queries + comparison)
        - "Generate a report on CAZyme distribution" (query + code analysis + visualization)
        - "What are the transport proteins across all genomes?" (query + analysis + literature)
        - "Analyze functional distribution in genome X" (query + statistical analysis + visualization)
        - "Find all BGCs and their associated pathways" (BGC query + pathway analysis + literature)
        - "Create a comprehensive functional profile" (multi-database query + analysis + synthesis)
        - "Show me the most novel loci" (multiple novelty detection queries + characterization + ranking)
        - "Find interesting genomic features" (comprehensive novelty search + detailed analysis)
        - "What stands out in this genome" (multi-type novelty detection + comparative analysis)
        
        Examples for traditional mode:
        - "How many proteins are there?" (simple count)
        - "What is the function of protein X?" (direct lookup)
        - "Show me the sequence of gene Y" (single lookup)
        
        **TASK PLANNING GUIDELINES**:
        - Create 3-5 meaningful tasks that build upon each other
        - Include data retrieval, analysis, and synthesis steps
        - Consider literature search for functional questions
        - Include code analysis for large datasets or distributions
        - Plan for cross-database integration when relevant
        
        **AGENTIC BIOLOGICAL PLANNING**:
        
        Use your biological expertise to break down complex biological questions into logical steps.
        Consider what data, tools, and analyses would be needed to thoroughly answer the user's question.
        
        Available tools for the agent:
        - Database queries (Neo4j for structured data, LanceDB for similarity search)
        - Literature search (PubMed integration)
        - Code interpreter (statistical analysis, visualization)
        - Genome selector (when specific organism targeting is needed)
        
        Plan 3-5 meaningful steps that build upon each other to comprehensively address the biological question.
        """
        
        user_query = dspy.InputField(desc="User's natural language question")
        requires_planning = dspy.OutputField(desc="Boolean: Does this query require multi-step agentic execution?")
        reasoning = dspy.OutputField(desc="Explanation of why agentic planning is or isn't needed")
        task_plan = dspy.OutputField(desc="If agentic: detailed 3-5 step task breakdown with specific actions. If traditional: 'N/A'")
    
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
        🧬 🧬 🧬 EXPLICIT GENOME SELECTION SYSTEM 🧬 🧬 🧬
        
        The system now uses EXPLICIT GENOME SELECTION that queries the database for available genomes
        and automatically selects the correct match. When target_genome is provided, it has already
        been validated against the database and MUST be used exactly as specified.
        
        🚨 MANDATORY GENOME FILTERING WHEN target_genome IS PROVIDED:
        If target_genome input is not empty, queries MUST include exact genome filtering:
        WHERE genome.genomeId = 'EXACT_TARGET_GENOME_ID'
        
        ✅ CORRECT EXAMPLES:
        target_genome: "[SPECIFIC_GENOME_ID_PROVIDED_BY_SYSTEM]"
        Query: MATCH (p:Protein)-[:ENCODEDBY]->(g:Gene)-[:BELONGSTOGENOME]->(genome:Genome) WHERE genome.genomeId = '[EXACT_TARGET_GENOME_VALUE]' RETURN p
        
        ❌ FORBIDDEN PATTERNS:
        - WHERE toLower(genome.genomeId) CONTAINS 'nomurabacteria'  [Old pattern - no longer used]
        - MATCH (p:Protein) RETURN p  [Missing genome filter entirely]
        - WHERE genome.genomeId CONTAINS 'partial_name'  [Partial matching not allowed]
        
        🔴 VALIDATION CHECKLIST:
        1. Is target_genome input provided and not empty? → MUST use exact match: WHERE genome.genomeId = 'target_genome'
        2. Is genome_filter_required = "True"? → MUST include genome filtering node and WHERE clause
        3. Does query connect to genome via proper relationship path? → Required for filtering
        4. Are you using the EXACT target_genome value without modification? → Critical for accurate results
        
        ══════════════════════════════════════════════════════════════════
        
        TECHNICAL REQUIREMENTS:
        
        CRITICAL: Generate ONLY executable Cypher queries with NO COMMENTS.
        
        FORBIDDEN:
        - Comments like /* comment */ or // comment
        - Section headers or explanatory text
        - Semicolons separating queries
        - Multiple independent queries (use UNION instead)
        
        ALLOWED FOR COMPREHENSIVE QUERIES:
        - UNION queries that combine multiple novelty detection approaches in one query
        - Multiple MATCH statements within single query
        - OPTIONAL MATCH for additional data
        
        REQUIRED: Start directly with MATCH, WITH, or OPTIONAL MATCH.
        
        CAZYME QUERY DETECTION - MANDATORY:
        When user mentions CAZyme, carbohydrate, glycoside, carbohydrate-active, dbCAN:
        - ALWAYS use Cazymeannotation and Cazymefamily nodes (NOT Domain/PFAM)
        - Use Pattern 4 from schema: (p:Protein)-[:HASCAZYME]->(ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily)
        - Include cf.familyId, cf.cazymeType, ca.substrateSpecificity for family details
        - Connect to genome via: (p)-[:ENCODEDBY]->(g:Gene)-[:BELONGSTOGENOME]->(genome:Genome)
        
        🧬 BIOLOGICAL QUERY STRATEGY EXAMPLES:
        
        **For Spatial/Genomic Organization Questions** (operons, phage, gene clusters, prophage segments):
        ANALYSIS TYPE: spatial_genomic
        Priority: Genomic coordinates, gene neighborhoods, spatial relationships
        Query Strategy for PHAGE/PROPHAGE detection:
        - ALWAYS include startCoordinate, endCoordinate, strand in SELECT
        - ORDER BY startCoordinate, endCoordinate for spatial analysis
        - Focus on consecutive genes without functional annotations (hypothetical proteins)
        - Look for stretches of 4+ consecutive genes with missing/poor annotations
        - Search for unannotated regions between 5-50kb (typical prophage size)
        - AVOID BGC, transport, or metabolic gene analysis - these are NOT phage indicators
        - Look for genes lacking KEGG annotations (ko.id IS NULL)
        - Consider genes with generic descriptions like "hypothetical protein"
        - NO LIMIT clauses - need to see full genomic context
        
        **For Functional Annotation Questions** (protein families, domains, pathways, metabolic activities):
        ANALYSIS TYPE: functional_annotation
        Priority: PFAM domains, KEGG orthologs, functional classifications
        Query Strategy:
        - Use Domain, KEGGOrtholog nodes for functional searches
        - GROUP BY functional categories for systematic analysis
        - Include annotation confidence scores (evalue, coverage)
        - Focus on biochemical pathways and enzyme classifications
        - Can use chunking for large functional datasets
        
        **For Discovery/Exploratory Questions** (novel loci, unusual features, "what's interesting"):
        ANALYSIS TYPE: comprehensive_discovery  
        Priority: Systematic exploration without preconceptions
        Query Strategy:
        - Broad queries that capture diverse biological features
        - Include both annotated and poorly-annotated regions
        - Combine spatial + functional information for context
        - Avoid LIMIT restrictions that might miss interesting patterns
        - Look for unusual gene arrangements or novel combinations
        
        **CONTEXT-AWARE QUERY GENERATION**:
        - Spatial queries need ORDER BY genomic coordinates for meaningful analysis
        - Functional queries can use GROUP BY categories for systematic coverage  
        - Discovery queries should be comprehensive, avoiding arbitrary limits
        - Always consider whether the user wants spatial organization vs functional classification
        
        Generate queries that match the biological analysis type and provide appropriate data organization.
        
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
        task_context = dspy.InputField(desc="Task context including any mentioned genome/organism names")
        genome_filter_required = dspy.InputField(desc="Whether genome filtering is required for this query (True/False)")
        target_genome = dspy.InputField(desc="Specific genome name to filter by (if genome_filter_required=True)")
        analysis_type = dspy.InputField(desc="Analysis type: spatial_genomic (operons, phage, clusters), functional_annotation (proteins, domains), or comprehensive_discovery")
        
        search_strategy = dspy.OutputField(desc="Retrieval approach: direct_query, similarity_search, or hybrid_search")
        cypher_query = dspy.OutputField(desc="🚨 EXECUTABLE Cypher query with NO COMMENTS - must start with MATCH/WITH/OPTIONAL. 🚨 MANDATORY: If task mentions specific genome/organism, MUST include: WHERE toLower(genome.genomeId) CONTAINS 'organism_name' 🚨")
        reasoning = dspy.OutputField(desc="Explanation of retrieval strategy choice including genome filtering rationale and biological focus")
        expected_result_size = dspy.OutputField(desc="Estimated result size: small, medium, or large")
        biological_focus = dspy.OutputField(desc="Primary biological focus: spatial_organization, functional_classification, or discovery_exploration")
    
    class RelevanceValidator(dspy.Signature):
        """
        Validate that retrieved genomic data is actually relevant to the user's original question.
        
        This prevents the system from analyzing irrelevant data (e.g., BGCs when user asks about phage).
        
        CRITICAL: Only validate as relevant if the data directly addresses the user's question.
        For phage/prophage queries, data about transporters, BGCs, or general metabolism is NOT relevant.
        For functional annotation queries, spatial/genomic organization data may not be relevant.
        
        Be strict - it's better to reject irrelevant data than to provide wrong analysis.
        """
        
        original_question = dspy.InputField(desc="User's original question")
        retrieved_data_summary = dspy.InputField(desc="Summary of what data was retrieved from the database")
        analysis_type = dspy.InputField(desc="Expected analysis type: spatial_genomic, functional_annotation, or comprehensive_discovery")
        
        is_relevant = dspy.OutputField(desc="Boolean: Is the retrieved data relevant to answering the original question?")
        relevance_score = dspy.OutputField(desc="Relevance score 0.0-1.0")
        reasoning = dspy.OutputField(desc="Explanation of why the data is or isn't relevant")
        missing_data_types = dspy.OutputField(desc="What types of data would be more relevant to the question?")

    class GenomicAnswerer(dspy.Signature):
        """
        Generate comprehensive answers to genomic questions using retrieved context.
        
        Provides detailed biological interpretation with:
        - Scientific accuracy and proper terminology
        - Confidence assessment based on data quality
        - Relevant citations and data sources
        - Clear explanations for non-experts when appropriate
        
        CRITICAL: Only analyze data that is relevant to the original question.
        If the context doesn't contain relevant data, state this clearly rather than 
        providing analysis of unrelated genomic features.
        """
        
        question = dspy.InputField(desc="Original user question")
        context = dspy.InputField(desc="Retrieved genomic data and annotations")
        analysis_type = dspy.InputField(desc="Analysis type: spatial_genomic, functional_annotation, or comprehensive_discovery")
        answer = dspy.OutputField(desc="Comprehensive answer with biological insights, or statement that relevant data was not found")
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