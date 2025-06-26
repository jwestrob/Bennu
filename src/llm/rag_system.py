#!/usr/bin/env python3
"""
DSPy-based RAG system for genomic knowledge graph.
Combines structured queries (Neo4j) with semantic search (LanceDB).
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
import asyncio
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum

import dspy
from rich.console import Console

from .config import LLMConfig
from .query_processor import Neo4jQueryProcessor, LanceDBQueryProcessor, HybridQueryProcessor, QueryResult
from .dsp_sig import NEO4J_SCHEMA
from .sequence_tools import sequence_viewer, extract_protein_ids_from_analysis
from .annotation_tools import annotation_explorer, functional_classifier, annotation_selector

console = Console()
logger = logging.getLogger(__name__)

# ===== AGENTIC CAPABILITIES: TASK GRAPH SYSTEM =====

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # For tasks whose dependencies have failed

class TaskType(Enum):
    ATOMIC_QUERY = "atomic_query"  # Query against Neo4j or LanceDB
    TOOL_CALL = "tool_call"        # A call to an external tool
    AGGREGATE = "aggregate"        # A task to synthesize results

@dataclass
class Task:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: TaskType = TaskType.ATOMIC_QUERY
    query: Optional[str] = None
    dependencies: Set[str] = field(default_factory=set)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    
    # New fields for agentic capabilities
    retry_count: int = 0
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class TaskGraph:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        
    def add_task(self, task: Task) -> str:
        """Add a task to the graph and return its ID."""
        self.tasks[task.task_id] = task
        return task.task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    def get_ready_tasks(self) -> List[Task]:
        """Get all tasks that are ready to execute (dependencies satisfied)."""
        ready = []
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                deps_satisfied = all(
                    self.tasks.get(dep_id, Task(status=TaskStatus.FAILED)).status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                )
                if deps_satisfied:
                    ready.append(task)
        return ready
    
    def mark_task_status(self, task_id: str, status: TaskStatus, result: Optional[Any] = None, error: Optional[str] = None):
        """Update task status and optionally set result or error."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = status
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error
    
    def get_failed_dependencies(self, task_id: str) -> List[str]:
        """Get list of failed dependency task IDs for a given task."""
        task = self.tasks.get(task_id)
        if not task:
            return []
        
        failed_deps = []
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if dep_task and dep_task.status == TaskStatus.FAILED:
                failed_deps.append(dep_id)
        return failed_deps
    
    def mark_skipped_tasks(self):
        """Mark tasks as skipped if their dependencies have failed."""
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING and self.get_failed_dependencies(task.task_id):
                task.status = TaskStatus.SKIPPED
    
    def is_complete(self) -> bool:
        """Check if all tasks have completed (successfully, failed, or skipped)."""
        return all(
            task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED]
            for task in self.tasks.values()
        )
    
    def get_summary(self) -> Dict[str, int]:
        """Get a summary of task statuses."""
        summary = {status.value: 0 for status in TaskStatus}
        for task in self.tasks.values():
            summary[task.status.value] += 1
        return summary

# ===== AGENTIC CAPABILITIES: TOOL SUITE =====

# Try to import Bio.Entrez at module level for easier testing
try:
    from Bio import Entrez
    ENTREZ_AVAILABLE = True
except ImportError:
    ENTREZ_AVAILABLE = False
    Entrez = None

def literature_search(query: str, email: str, **kwargs) -> str:
    """
    Searches PubMed for biomedical literature using the Entrez API.
    
    Args:
        query: Search query for PubMed
        email: Email address for Entrez API (required by NCBI)
        **kwargs: Additional parameters (ignored for compatibility)
        
    Returns:
        Formatted string containing search results with abstracts and PMIDs
    """
    if not query:
        return "Error: Empty query provided"
    
    if not email:
        return "Error: Email address is required for PubMed API access"
    
    if not ENTREZ_AVAILABLE:
        return "Error: Biopython not available for PubMed search"
    
    try:
        # Configure Entrez with email
        Entrez.email = email
        
        # Search PubMed
        logger.info(f"Executing literature search for: {query}")
        search_handle = Entrez.esearch(db="pubmed", term=query, retmax=10)
        search_results = Entrez.read(search_handle)
        search_handle.close()
        
        # Get list of PMIDs
        pmids = search_results.get('IdList', [])
        
        if not pmids:
            return f"No recent literature found for enhanced query: '{query}' (searched 2020-2024 publications)"
        
        # Fetch detailed information for the articles
        fetch_handle = Entrez.efetch(db="pubmed", id=pmids, rettype="medline", retmode="xml")
        fetch_results = Entrez.read(fetch_handle)
        fetch_handle.close()
        
        # Format results
        formatted_results = []
        formatted_results.append(f"Literature search results for: '{query}'")
        formatted_results.append(f"Found {len(pmids)} articles:")
        formatted_results.append("")
        
        for i, article in enumerate(fetch_results.get('PubmedArticle', []), 1):
            try:
                citation = article['MedlineCitation']
                
                # Handle PMID - can be dict or StringElement
                pmid_obj = citation.get('PMID', {})
                if hasattr(pmid_obj, 'get'):
                    pmid = pmid_obj.get('content', str(pmid_obj))
                else:
                    pmid = str(pmid_obj)
                
                article_info = citation.get('Article', {})
                
                # Handle title - can be string or StringElement
                title_obj = article_info.get('ArticleTitle', 'No title available')
                title = str(title_obj)
                
                # Extract abstract - handle StringElement objects
                abstract_info = article_info.get('Abstract', {})
                abstract_texts = abstract_info.get('AbstractText', [])
                if abstract_texts:
                    if isinstance(abstract_texts, list):
                        abstract = ' '.join(str(text) for text in abstract_texts)
                    else:
                        abstract = str(abstract_texts)
                else:
                    abstract = "No abstract available"
                
                # Format entry
                formatted_results.append(f"{i}. PMID: {pmid}")
                formatted_results.append(f"   Title: {title}")
                formatted_results.append(f"   Abstract: {abstract[:500]}{'...' if len(abstract) > 500 else ''}")
                formatted_results.append("")
                
            except Exception as e:
                logger.warning(f"Error processing article {i}: {e}")
                formatted_results.append(f"{i}. Error processing article")
                formatted_results.append("")
        
        return '\n'.join(formatted_results)
    except Exception as e:
        logger.error(f"Literature search failed: {e}")
        return f"Error: Literature search failed - {str(e)}"

# Code interpreter tool (import at module level)
async def code_interpreter_tool(code: str, session_id: str = None, timeout: int = 30, **kwargs) -> Dict[str, Any]:
    """
    Execute Python code using the secure code interpreter service.
    
    Args:
        code: Python code to execute
        session_id: Optional session ID for stateful execution
        timeout: Execution timeout in seconds
        **kwargs: Additional parameters (for compatibility)
        
    Returns:
        Dictionary containing execution results
    """
    try:
        from ..code_interpreter.client import code_interpreter_tool as execute_code
        return await execute_code(code=code, session_id=session_id, timeout=timeout)
    except ImportError:
        return {
            "success": False,
            "error": "Code interpreter service not available",
            "stdout": "",
            "stderr": "",
            "execution_time": 0
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Code execution failed: {str(e)}",
            "stdout": "",
            "stderr": "",
            "execution_time": 0
        }

# Tool manifest for the agent
# Conditional import of code interpreter to avoid dependency issues
try:
    from ..code_interpreter.client import code_interpreter_tool
    CODE_INTERPRETER_AVAILABLE = True
except ImportError:
    logger.warning("Code interpreter not available - missing dependencies")
    async def code_interpreter_tool(*args, **kwargs):
        return {
            "success": False,
            "error": "Code interpreter service is not available",
            "stdout": "",
            "stderr": "",
            "execution_time": 0
        }
    CODE_INTERPRETER_AVAILABLE = False

AVAILABLE_TOOLS = {
    "literature_search": literature_search,
    "code_interpreter": code_interpreter_tool,
    "sequence_viewer": sequence_viewer,
    "annotation_explorer": annotation_explorer,
    "functional_classifier": functional_classifier,
    "annotation_selector": annotation_selector,
}

# ===== ENHANCED DSPy SIGNATURES FOR AGENTIC PLANNING =====

class PlannerAgent(dspy.Signature):
    """
    Advanced genomic research planner with agentic task decomposition.
    
    DECISION LOGIC:
    - requires_planning = false: Simple queries answerable with local data only (count, list, find specific items)
    - requires_planning = true: Complex queries needing external tools or multi-step analysis
    
    WHEN TO USE AGENTIC PLANNING (requires_planning = true):
    - Literature search needed: "What does recent research say about X?"
    - Code execution needed: "Plot the genomic neighborhood", "Calculate statistics", "Create a heatmap"
    - Data analysis workflows: "Analyze protein similarities and visualize results"
    - Cross-reference analysis: "Compare our data with published studies"
    - Multi-step workflows: "Find X, then search literature about Y, then combine"
    - Sequence analysis requests: "Find proteins and show me their sequences"
    - Transport protein queries: "Find transport proteins" (requires intelligent annotation curation)
    
    WHEN TO USE TRADITIONAL MODE (requires_planning = false):
    - Simple counts: "How many proteins?"
    - Direct lookups: "Find proteins with domain X"
    - Similarity searches: "Find proteins similar to Y"
    - Database statistics: "What genomes do we have?"
    
    AVAILABLE TOOLS:
    - literature_search: Search PubMed for scientific literature and papers
    - code_interpreter: Execute Python code for data analysis, visualization, and calculations
    - sequence_viewer: Display raw protein sequences for detailed LLM biological analysis
    
    TASK TYPES:
    - atomic_query: Query the local knowledge graph for known facts
    - tool_call: Execute external tools for additional information
    - aggregate: Combine and synthesize results from multiple tasks
    
    CRITICAL: If requires_planning is true, you MUST provide a valid JSON task plan.
    If requires_planning is false, set task_plan to "N/A".
    
    EXAMPLE TASK PLAN JSON (KEEP SIMPLE TO AVOID ESCAPING ISSUES):
    {
        "tasks": [
            {
                "id": "query_transport_proteins",
                "type": "atomic_query",
                "query": "Find transport proteins using KEGG annotations",
                "dependencies": []
            },
            {
                "id": "analyze_composition",
                "type": "tool_call",
                "tool_name": "code_interpreter",
                "tool_args": {"code": "# Access sequence database and analyze amino acid composition"},
                "dependencies": ["query_transport_proteins"]
            },
            {
                "id": "view_sequences", 
                "type": "tool_call",
                "tool_name": "sequence_viewer",
                "tool_args": {"protein_ids": [], "analysis_context": "amino acid composition analysis"},
                "dependencies": ["analyze_composition"]
            },
            {
                "id": "synthesize_results",
                "type": "aggregate", 
                "dependencies": ["query_transport_proteins", "analyze_composition", "view_sequences"]
            }
        ]
    }
    
    CRITICAL DEPENDENCY RULES:
    1. code_interpreter tasks MUST depend on atomic_query tasks that provide protein data
    2. aggregate tasks MUST depend on ALL tasks they need to synthesize
    3. NEVER run code_interpreter and atomic_query tasks in parallel - code needs data first
    4. Use sequential dependencies: query → analyze → synthesize
    
    JSON FORMATTING RULES:
    - Keep code snippets SHORT in tool_args to avoid escaping issues
    - Use simple descriptions for queries instead of full Cypher
    - DSPy will expand simple descriptions into proper queries
    - NEVER include complex code with quotes/backslashes in JSON
    """
    
    user_query = dspy.InputField(desc="User query about genomic data")
    requires_planning = dspy.OutputField(desc="Boolean: true if query needs multi-step agentic planning with external tools, false for simple direct queries using local data only")
    task_plan = dspy.OutputField(desc="JSON task graph with tasks, dependencies, and execution strategy. MUST include 'tool_name' field for tool_call tasks. ONLY provide JSON if requires_planning is true, otherwise return 'N/A'. FOR TRANSPORT PROTEIN QUERIES: Use 4-task intelligent annotation discovery: 1) annotation_explorer, 2) transport_classifier, 3) transport_selector, 4) sequence_viewer.")
    reasoning = dspy.OutputField(desc="Explanation of why planning is or isn't needed and the chosen strategy")

@dataclass
class GenomicContext:
    """Context extracted from database queries."""
    structured_data: List[Dict[str, Any]]
    semantic_data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    query_time: float


class QueryClassifier(dspy.Signature):
    """Classify genomic queries to determine optimal retrieval strategy.
    
    CLASSIFICATION GUIDELINES:
    
    🔍 SEMANTIC: Use when query involves:
    - Similarity searches with specific protein IDs ("proteins similar to PLM0_scaffold_123", "find related proteins")  
    - Functional searches that need similarity expansion ("alcohol dehydrogenase proteins", "iron transporters")
    - Comparative analysis starting from known proteins ("find orthologs", "functionally equivalent")
    → Strategy: Neo4j finds starting points → LanceDB similarity expansion
    
    🏗️ STRUCTURAL: Use when query targets:
    - Specific protein/gene IDs for context only (no similarity needed)
    - Domain family names and counts (GGDEF, TPR, etc.)
    - Genomic coordinates or neighborhoods
    - Pathway/function statistics and aggregations
    → Primary: Neo4j graph traversal only
    
    🔀 HYBRID: Use when query combines:
    - Similarity + genomic context ("similar proteins and their neighborhoods")
    - Functional search + structural analysis ("iron transporters and their operons")
    - Cross-genome comparisons with genomic context
    → Strategy: LanceDB similarity → Neo4j structural context for results
    
    📊 GENERAL: Use for:
    - Database overviews and statistics
    - Broad exploratory queries without specific targets
    → Primary: Neo4j aggregation queries
    """
    
    question = dspy.InputField(desc="User's question about genomic data")
    query_type = dspy.OutputField(desc="Query type: 'semantic', 'structural', 'hybrid', or 'general'")
    reasoning = dspy.OutputField(desc="Specific reasoning for classification based on guidelines above")
    primary_database = dspy.OutputField(desc="Primary database: 'lancedb', 'neo4j', or 'both'")


class ContextRetriever(dspy.Signature):
    """Generate database queries to retrieve relevant genomic context.

    MANDATORY QUERY TEMPLATE FOR TRANSPORT PROTEINS:
    
    MATCH (ko:KEGGOrtholog) 
    WHERE toLower(ko.description) CONTAINS 'transport'
    MATCH (p:Protein)-[:HASFUNCTION]->(ko)
    OPTIONAL MATCH (p)-[:ENCODEDBY]->(g:Gene)
    OPTIONAL MATCH (p)-[:HASDOMAIN]->(da:DomainAnnotation)-[:DOMAINFAMILY]->(dom:Domain)
    RETURN p.id AS protein_id, ko.id AS ko_id, ko.description AS ko_description,
           g.startCoordinate AS start_coordinate, g.endCoordinate AS end_coordinate, g.strand,
           collect(DISTINCT dom.id) AS pfam_accessions
    LIMIT 3
    
    CRITICAL RULES:
    - Node labels: Protein, KEGGOrtholog, Gene, Domain, DomainAnnotation (ONLY these exist)
    - Relationships: HASFUNCTION, ENCODEDBY, HASDOMAIN, DOMAINFAMILY (ALL UPPERCASE)
    - Properties: p.id, ko.description, g.startCoordinate, g.endCoordinate, g.strand
    - DO NOT use: Function, f.name, f.kegg_id, p.accession, g.start, g.end, exists()
    - USE: ko.description IS NOT NULL (not exists())
    
    For transport proteins, copy the template above EXACTLY with only the search term changed.
    """
    
    question = dspy.InputField(desc="User's question")
    query_type = dspy.InputField(desc="Classified query type")
    search_terms = dspy.OutputField(desc="Extract key search terms from the question (e.g., 'transport', 'ABC', 'permease')")
    neo4j_query = dspy.OutputField(desc="Complete Cypher query copying the mandatory template above exactly, only changing the search term in ko.description CONTAINS")
    requires_multi_stage = dspy.OutputField(desc="Boolean: true if query asks for 'proteins similar to X' where X is a functional description")
    search_strategy = dspy.OutputField(desc="Description of search approach")


class GenomicAnswerer(dspy.Signature):
    """Generate specific, data-driven genomic analysis with concrete biological insights."""
    
    question = dspy.InputField(desc="Original user question")
    context = dspy.InputField(desc="Retrieved genomic data including domain descriptions, KEGG functions, and quantitative metrics")
    answer = dspy.OutputField(desc="Structured biological analysis that MUST: 1) If NO relevant data was retrieved, clearly state 'We don't have that kind of information in our database' and explain what data IS available, 2) Ground all statements in specific data points (coordinates, counts, IDs) when data exists, 3) For sequence-based analyses, ANALYZE the provided amino acid sequences directly when available - examine length, composition, N/C termini, hydrophobic regions, and functional motifs, 4) When genomic neighborhood context is provided, analyze neighboring proteins and their functions to understand biological context, 5) Calculate and report specific distances between genes when coordinate data exists, 6) OPERON ANALYSIS (only when genomic neighborhood data is provided AND when it's biologically relevant): If the context includes neighboring genes with coordinates and the analysis would provide meaningful biological insights, perform operon structure analysis: a) Only analyze genomic context if proteins are functionally related or part of potential operons, b) Identify gene clusters on the same strand with <500bp intergenic spacing, c) Calculate precise intergenic distances and report exact spacing (e.g., 'hmuT-hmuU: 23bp, hmuU-hmuV: 31bp'), d) Assess functional coherence by analyzing whether neighboring genes encode related biological processes or pathway components, e) Predict operon boundaries and transcriptional units based on spacing and functional relationships, f) Use biological knowledge to identify canonical operon architectures (e.g., ABC transporter operons typically contain binding protein + permease + ATPase genes), g) Example output: 'hmuTUV genes form a 3-gene operon (total span: 2,847bp, intergenic spacers: 23bp and 31bp) encoding a complete heme ABC transport system with canonical architecture', 7) DOMAIN ARCHITECTURE ANALYSIS (only when multiple PFAM domains are present): For proteins containing TWO OR MORE PFAM domains, analyze architectural significance: a) Only perform this analysis if the context contains multiple PFAM accessions or domain information for the same protein, b) Identify canonical domain combinations and their biological meanings (e.g., PF01032+PF00950+PF00005 = ABC transporter system), c) Predict functional modules and their interactions within the protein, d) Analyze domain order and organization for functional insights, e) Identify unusual or novel domain arrangements that might indicate specialized functions, f) Compare architecture to known protein families and functional classes, g) Example output: 'N-terminal periplasmic binding domain (PF01032) + central transmembrane permease domains (PF00950) + C-terminal cytoplasmic ATPase domain (PF00005) = complete ABC import system architecture', 8) Use specific protein/domain names from the actual retrieved data, 9) Organize response logically: Data Availability → Genomic Context → Operon Analysis (if neighborhood data available and biologically relevant) → Domain Architecture (if multiple PFAM domains present) → Sequence Analysis → Functional Analysis → Biological Significance. CRITICAL: When protein sequences are provided in the context, ANALYZE them directly rather than referring to external databases.")
    confidence = dspy.OutputField(desc="Confidence level with reasoning based on data quality for the specific query: 'high - complete data available for all requested analyses', 'medium - partial data available, some gaps in requested information', 'low - minimal data available, significant limitations', 'none - no relevant data found in database'. Only mention literature if literature search was actually performed as part of the analysis.")
    citations = dspy.OutputField(desc="Specific data sources: PFAM accessions (PF#####), KEGG orthologs (K#####), domain names, genome IDs, code interpreter sessions, PubMed search terms used, or 'None - no data retrieved' if appropriate")


class GenomicRAG(dspy.Module):
    """Main RAG system for genomic question answering with agentic capabilities."""
    
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        
        # Initialize original DSPy components
        self.classifier = dspy.ChainOfThought(QueryClassifier)
        self.retriever = dspy.ChainOfThought(ContextRetriever)
        self.answerer = dspy.ChainOfThought(GenomicAnswerer)
        
        # Initialize NEW agentic components
        self.planner = dspy.ChainOfThought(PlannerAgent)
        
        # Initialize query processors
        self.neo4j_processor = Neo4jQueryProcessor(config)
        self.lancedb_processor = LanceDBQueryProcessor(config)
        self.hybrid_processor = HybridQueryProcessor(config)
        
        # Persistent session ID for code interpreter continuity
        self.code_interpreter_session_id = None
        
        # Configure DSPy LLM
        self._configure_dspy()
        
        logger.info("GenomicRAG system initialized with agentic capabilities")
    
    def _configure_dspy(self):
        """Configure DSPy with the specified LLM provider."""
        api_key = self.config.get_api_key()
        
        if not api_key:
            raise ValueError(f"No API key found for provider: {self.config.llm_provider}")
        
        if self.config.llm_provider == "openai":
            # Use LiteLLM wrapper for OpenAI
            import litellm
            lm = dspy.LM(
                model=f"openai/{self.config.llm_model}",
                api_key=api_key,
                #max_tokens=self.config.max_context_length,
                temperature=1.0,
                max_tokens=100_000
            )
        elif self.config.llm_provider == "anthropic":
            # Use LiteLLM wrapper for Anthropic
            import litellm
            lm = dspy.LM(
                model=f"anthropic/{self.config.llm_model}",
                api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
        
        dspy.settings.configure(lm=lm)
        logger.info(f"Configured DSPy with {self.config.llm_provider}: {self.config.llm_model}")
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all components."""
        return {
            "neo4j": self.neo4j_processor.health_check(),
            "lancedb": self.lancedb_processor.health_check(),
            "hybrid": self.hybrid_processor.health_check(),
            "dspy_configured": dspy.settings.lm is not None
        }
    
    async def ask(self, question: str) -> Dict[str, Any]:
        """
        Main method to answer genomic questions with agentic planning.
        
        Args:
            question: Natural language question about genomic data
            
        Returns:
            Dict containing answer, confidence, sources, and metadata
        """
        try:
            console.print(f"🧬 [bold blue]Processing question:[/bold blue] {question}")
            
            # STEP 1: Determine if agentic planning is needed
            planning_result = self.planner(user_query=question)
            console.print(f"🤖 Agentic planning: {planning_result.requires_planning}")
            console.print(f"💭 Planning reasoning: {planning_result.reasoning}")
            
            # Convert string boolean to actual boolean if needed
            requires_planning = planning_result.requires_planning
            if isinstance(requires_planning, str):
                requires_planning = requires_planning.lower() == 'true'
            
            if requires_planning:
                # AGENTIC PATH: Multi-step task execution
                # Check if we actually have a valid task plan
                task_plan = planning_result.task_plan
                if task_plan == "N/A" or not task_plan or task_plan.strip() == "":
                    console.print("⚠️ [yellow]Agentic planning requested but no task plan provided, falling back to traditional mode[/yellow]")
                    return await self._execute_traditional_query(question)
                return await self._execute_agentic_plan(question, planning_result)
            else:
                # TRADITIONAL PATH: Direct query execution
                return await self._execute_traditional_query(question)
                
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            
            # NEW: Check if this is a repairable error from query processor
            repair_message = None
            if hasattr(self.hybrid_processor, 'neo4j_processor') and hasattr(self.hybrid_processor.neo4j_processor, 'get_last_repair_result'):
                repair_result = self.hybrid_processor.neo4j_processor.get_last_repair_result()
                if repair_result and repair_result.success and repair_result.user_message:
                    repair_message = repair_result.user_message
                    logger.info(f"Using TaskRepairAgent message: {repair_message[:100]}...")
            
            if repair_message:
                return {
                    "question": question,
                    "answer": repair_message,
                    "confidence": "medium - error handled gracefully",
                    "citations": "",
                    "repair_info": "TaskRepairAgent provided helpful guidance"
                }
            else:
                return {
                    "question": question,
                    "answer": f"I encountered an error while processing your question: {str(e)}",
                    "confidence": "low",
                    "citations": "",
                    "error": str(e)
                }
    
    async def _execute_traditional_query(self, question: str) -> Dict[str, Any]:
        """Execute traditional single-step query (backward compatibility)."""
        console.print("📋 [dim]Using traditional query path[/dim]")
        
        # Step 1: Classify the query type
        classification = self.classifier(question=question)
        console.print(f"📊 Query type: {classification.query_type}")
        console.print(f"💭 Reasoning: {classification.reasoning}")
        
        # Step 2: Generate retrieval strategy
        retrieval_plan = self.retriever(
            db_schema=NEO4J_SCHEMA,
            question=question,
            query_type=classification.query_type
        )
        console.print(f"🔍 Search strategy: {retrieval_plan.search_strategy}")
        
        # Step 3: Execute database queries
        # Log template usage for debugging
        if hasattr(retrieval_plan, 'template_choice'):
            console.print(f"🔧 Using Template {retrieval_plan.template_choice} for query generation")
        
        context = await self._retrieve_context(classification.query_type, retrieval_plan)
        
        # NEW: Check for TaskRepairAgent messages first
        if 'repair_message' in context.metadata:
            logger.info("TaskRepairAgent provided helpful guidance - returning repair message")
            return {
                "question": question,
                "answer": context.metadata['repair_message'],
                "confidence": "medium - error handled gracefully by TaskRepairAgent",
                "citations": "",
                "repair_info": f"Repaired using strategy: {context.metadata.get('repair_strategy', 'unknown')}"
            }
        
        # Check for retrieval errors
        if 'retrieval_error' in context.metadata:
            raise Exception(f"Query execution failed: {context.metadata['retrieval_error']}")
        
        # Step 4: Generate answer using context
        answer_result = self.answerer(
            question=question,
            context=self._format_context(context)
        )
        
        # Compile final response
        response = {
            "question": question,
            "answer": answer_result.answer,
            "confidence": answer_result.confidence,
            "citations": answer_result.citations,
            "query_metadata": {
                "query_type": classification.query_type,
                "search_strategy": retrieval_plan.search_strategy,
                "context_items": len(context.structured_data) + len(context.semantic_data),
                "retrieval_time": context.query_time,
                "execution_mode": "traditional"
            }
        }
        
        console.print(f"✅ [green]Answer generated[/green] (confidence: {answer_result.confidence})")
        return response
    
    async def _execute_agentic_plan(self, question: str, planning_result) -> Dict[str, Any]:
        """Execute multi-step agentic plan with task orchestration."""
        console.print("🤖 [bold cyan]Using agentic planning path[/bold cyan]")
        
        try:
            # Parse the task plan with improved error handling
            task_plan_json = planning_result.task_plan
            if isinstance(task_plan_json, str):
                if not task_plan_json.strip():
                    raise ValueError("Empty task plan provided")
                
                # Clean up common JSON formatting issues from LLM output
                cleaned_json = task_plan_json.strip()
                
                # Try parsing the JSON with better error handling
                try:
                    task_plan = json.loads(cleaned_json)
                except json.JSONDecodeError as e:
                    # Log the problematic JSON for debugging
                    logger.error(f"JSON parse error at position {e.pos}: {e.msg}")
                    logger.error(f"Problematic JSON snippet: {cleaned_json[max(0, e.pos-50):e.pos+50]}")
                    raise e
            else:
                task_plan = task_plan_json
                
            # Validate task plan structure
            if not isinstance(task_plan, dict) or 'tasks' not in task_plan:
                raise ValueError(f"Invalid task plan structure: {task_plan}")
            
            if not task_plan.get('tasks'):
                raise ValueError("Task plan contains no tasks")
            
            console.print(f"📋 Executing {len(task_plan.get('tasks', []))} tasks")
            
            # Create task graph
            graph = TaskGraph()
            
            # Add tasks to graph
            for task_def in task_plan.get('tasks', []):
                # Validate task definition
                task_type = TaskType(task_def['type'])
                
                # For tool_call tasks, ensure tool_name is provided
                if task_type == TaskType.TOOL_CALL:
                    tool_name = task_def.get('tool_name')
                    if not tool_name or tool_name == 'None':
                        raise ValueError(f"Task {task_def['id']} is type 'tool_call' but missing 'tool_name' field")
                    if tool_name not in AVAILABLE_TOOLS:
                        raise ValueError(f"Task {task_def['id']} references unknown tool: {tool_name}")
                
                task = Task(
                    task_id=task_def['id'],
                    task_type=task_type,
                    query=task_def.get('query'),
                    dependencies=set(task_def.get('dependencies', [])),
                    tool_name=task_def.get('tool_name'),
                    tool_args=task_def.get('tool_args', {}),
                    metadata=task_def.get('metadata', {})
                )
                graph.add_task(task)
            
            # Execute task graph
            all_results = {}
            iteration = 0
            max_iterations = 10  # Prevent infinite loops
            
            while not graph.is_complete() and iteration < max_iterations:
                iteration += 1
                ready_tasks = graph.get_ready_tasks()
                
                if not ready_tasks:
                    # No ready tasks but graph not complete - check for failures
                    graph.mark_skipped_tasks()
                    break
                
                console.print(f"🔄 Iteration {iteration}: {len(ready_tasks)} ready tasks")
                
                # Debug: Log task dependencies
                for task in ready_tasks:
                    if task.dependencies:
                        logger.debug(f"Task {task.task_id} depends on: {list(task.dependencies)}")
                    else:
                        logger.debug(f"Task {task.task_id} has no dependencies")
                
                # Execute ready tasks
                for task in ready_tasks:
                    console.print(f"▶️  Executing {task.task_type.value}: {task.task_id}")
                    
                    try:
                        graph.mark_task_status(task.task_id, TaskStatus.RUNNING)
                        logger.debug(f"Passing previous_results with {len(all_results)} completed tasks to {task.task_id}")
                        result = await self._execute_task(task, all_results)
                        graph.mark_task_status(task.task_id, TaskStatus.COMPLETED, result=result)
                        all_results[task.task_id] = result
                        console.print(f"✅ Task {task.task_id} completed")
                        
                    except Exception as e:
                        logger.error(f"Task {task.task_id} failed: {e}")
                        graph.mark_task_status(task.task_id, TaskStatus.FAILED, error=str(e))
            
            # Generate final answer from all results
            combined_context = self._combine_task_results(all_results)
            
            answer_result = self.answerer(
                question=question,
                context=combined_context
            )
            
            # Compile response with agentic metadata
            summary = graph.get_summary()
            response = {
                "question": question,
                "answer": answer_result.answer,
                "confidence": answer_result.confidence,
                "citations": answer_result.citations,
                "query_metadata": {
                    "execution_mode": "agentic",
                    "tasks_completed": summary.get("completed", 0),
                    "tasks_failed": summary.get("failed", 0),
                    "tasks_skipped": summary.get("skipped", 0),
                    "total_iterations": iteration,
                    "task_plan": task_plan
                }
            }
            
            console.print(f"✅ [green]Agentic plan completed[/green] ({summary['completed']} tasks, confidence: {answer_result.confidence})")
            return response
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in agentic planning: {e}")
            console.print(f"❌ [red]Invalid task plan JSON, falling back to traditional mode[/red]")
            return await self._execute_traditional_query(question)
        except ValueError as e:
            logger.error(f"Task plan validation error: {e}")
            console.print(f"❌ [red]Invalid task plan structure, falling back to traditional mode[/red]")
            return await self._execute_traditional_query(question)
        except Exception as e:
            logger.error(f"Error in agentic planning: {e}")
            console.print(f"❌ [red]Agentic planning failed, falling back to traditional mode[/red]")
            return await self._execute_traditional_query(question)
    
    async def _execute_task(self, task: Task, previous_results: Dict[str, Any]) -> Any:
        """Execute a single task based on its type."""
        if task.task_type == TaskType.ATOMIC_QUERY:
            # Execute database query
            classification = self.classifier(question=task.query)
            retrieval_plan = self.retriever(
                db_schema=NEO4J_SCHEMA,
                question=task.query,
                query_type=classification.query_type
            )
            context = await self._retrieve_context(classification.query_type, retrieval_plan)
            
            if 'retrieval_error' in context.metadata:
                raise Exception(f"Query execution failed: {context.metadata['retrieval_error']}")
            
            return {
                "context": context,
                "query_type": classification.query_type,
                "search_strategy": retrieval_plan.search_strategy
            }
            
        elif task.task_type == TaskType.TOOL_CALL:
            # Execute external tool
            tool_function = AVAILABLE_TOOLS.get(task.tool_name)
            if not tool_function:
                raise ValueError(f"Unknown tool: {task.tool_name}")
            
            # Add default email for literature search if not provided
            if task.tool_name == "literature_search" and "email" not in task.tool_args:
                task.tool_args["email"] = "researcher@example.com"  # Default email
            
            # Enhance literature search queries with biological context from previous results
            if task.tool_name == "literature_search":
                enhanced_query = self._enhance_literature_query(task.tool_args.get("query", ""), previous_results)
                task.tool_args["query"] = enhanced_query
            
            # Enhance code interpreter with sequence database access and context
            elif task.tool_name == "code_interpreter":
                # Always enhance code interpreter with sequence database access
                enhanced_code = self._enhance_code_interpreter(task.tool_args.get("code", ""), previous_results)
                task.tool_args["code"] = enhanced_code
                
                # Use persistent session for continuity across queries
                if self.code_interpreter_session_id:
                    task.tool_args["session_id"] = self.code_interpreter_session_id
                    logger.debug(f"Using persistent session: {self.code_interpreter_session_id}")
            
            # Auto-populate sequence viewer with protein IDs from previous results
            elif task.tool_name == "sequence_viewer":
                current_protein_ids = task.tool_args.get("protein_ids")
                logger.debug(f"🔬 Sequence viewer task - current protein_ids: {current_protein_ids}")
                
                # Check if protein_ids contains unresolved template variables
                needs_auto_population = (
                    not current_protein_ids or 
                    (isinstance(current_protein_ids, str) and current_protein_ids.startswith("${")) or
                    (isinstance(current_protein_ids, list) and len(current_protein_ids) == 1 and 
                     isinstance(current_protein_ids[0], str) and current_protein_ids[0].startswith("${"))
                )
                
                if needs_auto_population:
                    logger.info("🔍 Auto-populating sequence viewer with protein IDs from previous results")
                    
                    # Extract protein IDs from code interpreter results
                    protein_ids = []
                    for task_id, result in previous_results.items():
                        if "tool_result" in result and result.get("tool_name") == "code_interpreter":
                            code_output = result["tool_result"].get("stdout", "")
                            extracted_ids = extract_protein_ids_from_analysis(code_output)
                            protein_ids.extend(extracted_ids)
                        
                        # Extract from atomic query results
                        elif "context" in result:
                            context = result["context"]
                            
                            # Handle GenomicContext with structured_data
                            if hasattr(context, 'structured_data'):
                                for item in context.structured_data:
                                    pid = item.get('protein_id', '')
                                    if pid:
                                        protein_ids.append(pid)
                            # Handle QueryResult object wrapped in context
                            elif hasattr(context, 'results'):
                                for item in context.results:
                                    pid = item.get('protein_id', '')
                                    if pid:
                                        protein_ids.append(pid)
                    
                    # Remove duplicates and limit
                    unique_ids = list(dict.fromkeys(protein_ids))[:5]
                    
                    # Log auto-population results for debugging
                    logger.info(f"🔍 Auto-population found {len(unique_ids)} protein IDs")
                    
                    task.tool_args["protein_ids"] = unique_ids
                    logger.info(f"✅ Auto-populated sequence viewer with {len(unique_ids)} protein IDs")
                    logger.debug(f"🧬 Final protein IDs for sequence viewer: {unique_ids}")
            
            # Handle both sync and async tool functions
            if asyncio.iscoroutinefunction(tool_function):
                result = await tool_function(**task.tool_args)
            else:
                result = tool_function(**task.tool_args)
            
            # Capture session ID from code interpreter for persistence
            if task.tool_name == "code_interpreter" and "session_id" in result:
                self.code_interpreter_session_id = result["session_id"]
                logger.info(f"Code interpreter session persisted: {self.code_interpreter_session_id}")
            
            return {"tool_result": result, "tool_name": task.tool_name}
            
        elif task.task_type == TaskType.AGGREGATE:
            # Combine results from dependencies
            combined_data = []
            for dep_id in task.dependencies:
                if dep_id in previous_results:
                    combined_data.append(previous_results[dep_id])
            
            return {"aggregated_results": combined_data}
        
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
    
    def _enhance_literature_query(self, original_query: str, previous_results: Dict[str, Any]) -> str:
        """
        Enhance literature search queries with biological context from previous database results.
        Prioritizes PFAM domains and biological terms that are publication-friendly.
        """
        # Extract biological terms from previous results
        pfam_domains = set()
        domain_descriptions = set()
        organism_names = set()
        
        for task_id, result in previous_results.items():
            if "context" in result:
                context = result["context"]
                structured_data = context.structured_data
                
                for item in structured_data:
                    # Extract PFAM accessions (highly citable)
                    pfam_accessions = item.get('pfam_accessions', []) or item.get('pfam_accession', [])
                    if pfam_accessions:
                        if isinstance(pfam_accessions, list):
                            pfam_domains.update(acc for acc in pfam_accessions if acc and acc != 'None')
                        else:
                            if pfam_accessions != 'None':
                                pfam_domains.add(pfam_accessions)
                    
                    # Extract domain descriptions (publication-friendly)
                    descriptions = item.get('domain_descriptions', []) or item.get('ko_description', '')
                    if descriptions:
                        if isinstance(descriptions, list):
                            for desc in descriptions:
                                if desc and desc != 'None' and len(desc) > 10:
                                    # Extract key biological terms
                                    desc_lower = desc.lower()
                                    if any(term in desc_lower for term in ['transport', 'receptor', 'permease', 'channel', 'pump']):
                                        domain_descriptions.add(desc.split('.')[0])  # Remove trailing details
                        elif descriptions != 'None' and len(descriptions) > 10:
                            descriptions_lower = descriptions.lower()
                            if any(term in descriptions_lower for term in ['transport', 'receptor', 'permease', 'channel', 'pump']):
                                domain_descriptions.add(descriptions.split('.')[0])
                    
                    # Extract organism context for specificity
                    protein_id = item.get('protein_id', '')
                    if protein_id and '_FULL_' in protein_id:
                        # Extract organism from protein ID structure
                        parts = protein_id.split('_FULL_')
                        if len(parts) > 1:
                            organism_part = parts[1].split('_')[0]
                            if organism_part in ['Acidovorax', 'Gammaproteobacteria', 'Burkholderiales']:
                                organism_names.add(organism_part)
        
        # Build enhanced query
        query_parts = []
        
        # Start with original query terms (cleaned)
        base_query = original_query.replace('recent[dp]', '').strip()
        if base_query:
            query_parts.append(base_query)
        
        # Add PFAM domains (most publication-relevant)
        if pfam_domains:
            pfam_list = list(pfam_domains)[:3]  # Limit to top 3 to avoid overly long queries
            pfam_query = ' OR '.join(f'"{pfam}"' for pfam in pfam_list)
            query_parts.append(f"({pfam_query})")
        
        # Add functional descriptions (biology-focused)
        if domain_descriptions:
            desc_list = list(domain_descriptions)[:2]  # Limit to avoid query length issues
            for desc in desc_list:
                # Clean up descriptions for PubMed
                clean_desc = desc.replace('_', ' ').replace('-', ' ')
                if len(clean_desc.split()) <= 4:  # Only short, focused terms
                    query_parts.append(f'"{clean_desc}"')
        
        # Add organism context if relevant
        if organism_names:
            org_name = list(organism_names)[0]  # Just one for specificity
            if org_name != 'Gammaproteobacteria':  # Too broad
                query_parts.append(org_name)
        
        # Add temporal constraint for recent papers
        query_parts.append('("2020"[Date - Publication] : "2024"[Date - Publication])')
        
        # Combine query parts with AND logic
        enhanced_query = ' AND '.join(query_parts)
        
        # Fallback to original if enhancement fails
        if len(enhanced_query) > 200 or not pfam_domains:  # Too long or no good terms
            return f"{base_query} AND heme AND transport AND bacteria"
        
        logger.info(f"Enhanced literature query: {enhanced_query}")
        return enhanced_query
    
    def _enhance_code_interpreter(self, original_code: str, previous_results: Dict[str, Any]) -> str:
        """
        Enhance code interpreter tasks with sequence database access and protein IDs from previous results.
        Usually perform analysis on retrieved sequence(s); can perform direct analysis of sequence data 
        if number of sequences is low and it feels appropriate and/or the user requests.
        """
        # Extract protein IDs from previous results
        protein_ids = set()
        
        for task_id, result in previous_results.items():
            if "context" in result:
                context = result["context"]
                structured_data = context.structured_data
                
                for item in structured_data:
                    # Extract protein IDs (already cleaned in context formatting)
                    protein_id = item.get('protein_id', '')
                    if protein_id:
                        protein_ids.add(protein_id)
        
        # Create enhanced code with sequence database setup
        if protein_ids:
            protein_list = list(protein_ids)[:10]  # Limit to first 10 for performance and stability
            
            protein_ids_code = '\n'.join([f'    "{pid}",' for pid in protein_list])
            enhanced_code = f"""
# Sequence Database Setup for Genomic Analysis
import sys
sys.path.append('/app')
sys.path.append('/app/build_kg')
from sequence_db import SequenceDatabase

# Common imports for analysis (available globally)
from collections import Counter, defaultdict, OrderedDict
import collections
import json
import re
import statistics
import itertools

# Check if biopython is available
if 'bio_available' in globals() and bio_available:
    print("✅ Biopython is available for sequence analysis")
else:
    print("⚠️  Biopython not available - basic sequence analysis only")

# Initialize sequence database
db = SequenceDatabase('/app/sequences.db', read_only=True)

# Protein IDs from previous query results
protein_ids = [
{protein_ids_code}
]

# Get protein sequences (strip 'protein:' prefix if present)
clean_protein_ids = [pid.replace('protein:', '') if pid.startswith('protein:') else pid for pid in protein_ids]
sequences = db.get_sequences(clean_protein_ids)
print(f"Retrieved {{len(sequences)}} sequences out of {{len(protein_ids)}} requested")

# Variable aliases for compatibility with different code patterns
proteins = sequences  # For code that expects 'proteins' variable
protein_sequences = sequences  # For code that expects 'protein_sequences' variable

print(f"✅ Enhanced code setup complete! Database has {{len(sequences)}} sequences ready for analysis.")

# Robust amino acid composition analysis template
if sequences:
    print("\\n=== AMINO ACID COMPOSITION ANALYSIS ===")
    analyzed_count = 0
    for protein_id, sequence in sequences.items():
        if analyzed_count >= 3:  # Limit to first 3 as requested
            break
        analyzed_count += 1
        
        print(f"\\nProtein {{analyzed_count}}: {{protein_id[:50]}}...")
        print(f"Length: {{len(sequence)}} amino acids")
        
        if len(sequence) > 0:
            # Calculate amino acid composition with safe scoping
            aa_counts = Counter(sequence)
            total_aa = len(sequence)
            
            # Top 5 most frequent amino acids
            top_aa = aa_counts.most_common(5)
            print("Top 5 amino acids:")
            for aa, count in top_aa:
                freq = count / total_aa
                print(f"  {{aa}}: {{freq:.3f}} ({{count}} residues)")
            
            # Transport protein properties - explicit calculation to avoid scoping issues
            hydrophobic_aa = ['A', 'V', 'L', 'I', 'M', 'F', 'W', 'Y', 'P']
            charged_aa = ['R', 'K', 'D', 'E']
            polar_aa = ['S', 'T', 'N', 'Q', 'H', 'C']
            
            hydrophobic_count = sum(aa_counts.get(aa, 0) for aa in hydrophobic_aa)
            charged_count = sum(aa_counts.get(aa, 0) for aa in charged_aa)
            polar_count = sum(aa_counts.get(aa, 0) for aa in polar_aa)
            
            print(f"Hydrophobic: {{hydrophobic_count/total_aa:.3f}} ({{hydrophobic_count}} residues)")
            print(f"Charged: {{charged_count/total_aa:.3f}} ({{charged_count}} residues)")
            print(f"Polar: {{polar_count/total_aa:.3f}} ({{polar_count}} residues)")
            
            # Sequence insights
            print(f"N-terminus (first 20): {{sequence[:20]}}")
            if len(sequence) > 20:
                print(f"C-terminus (last 20): {{sequence[-20:]}}")
            
            # Membrane protein prediction
            if hydrophobic_count/total_aa > 0.4:
                print("✓ High hydrophobic content - likely membrane protein")
            
            # Signal sequence check
            if sequence.startswith('M'):
                print("✓ Starts with methionine (typical start codon)")
        else:
            print("❌ Empty sequence")
else:
    print("❌ No sequences available for analysis")

# Original user analysis code (if any):
{original_code}
"""
        else:
            # No protein IDs available from previous results
            enhanced_code = f"""
# Sequence Database Setup for Genomic Analysis
import sys
sys.path.append('/app')
sys.path.append('/app/build_kg')
from sequence_db import SequenceDatabase

# Common imports for analysis (available globally)
from collections import Counter, defaultdict, OrderedDict
import collections
import json
import re
import statistics
import itertools

# Check if biopython is available
if 'bio_available' in globals() and bio_available:
    print("✅ Biopython is available for sequence analysis")
else:
    print("⚠️  Biopython not available - basic sequence analysis only")

# Initialize sequence database
db = SequenceDatabase('/app/sequences.db', read_only=True)

print("❌ No protein IDs available from previous query results")
print("This indicates a task coordination issue - the database query task may not have completed")

# Get database statistics for context
stats = db.get_statistics()
print(f"Database contains {{stats['total_sequences']}} total sequences across {{stats['unique_genomes']}} genomes")

# Original analysis code (will likely fail without protein IDs):
{original_code}
"""
        
        logger.info(f"Enhanced code interpreter with {len(protein_list) if protein_ids else 0} protein IDs (limited from {len(protein_ids)} total)")
        if protein_ids:
            logger.debug(f"Protein IDs found: {list(protein_ids)[:3]}...")  # Log first 3 IDs
        else:
            logger.warning("No protein IDs found in previous results - code interpreter may not have sequence access")
            logger.debug(f"Previous results structure: {list(previous_results.keys())}")
            # Log what's actually in the previous results for debugging
            for task_id, result in previous_results.items():
                if isinstance(result, dict):
                    logger.debug(f"Task {task_id}: {list(result.keys())}")
                    if 'context' in result:
                        context = result['context']
                        if hasattr(context, 'structured_data'):
                            logger.debug(f"  - structured_data: {len(context.structured_data)} items")
                        else:
                            logger.debug(f"  - context type: {type(context)}")
                else:
                    logger.debug(f"Task {task_id}: {type(result)}")
        return enhanced_code
    
    def _combine_task_results(self, all_results: Dict[str, Any]) -> str:
        """Combine results from all tasks into context for final answer generation."""
        context_parts = []
        
        # Separate different types of results for better organization
        database_results = []
        tool_results = []
        aggregation_results = []
        
        for task_id, result in all_results.items():
            if "context" in result:
                # Database query result
                context = result["context"]
                formatted_context = self._format_context(context)
                database_results.append(f"=== Local Database: {task_id} ===\n{formatted_context}")
                
            elif "tool_result" in result:
                # Tool execution result
                tool_result = result["tool_result"]
                tool_name = result.get("tool_name", "unknown")
                
                # Handle different tool types with appropriate formatting
                if tool_name == "sequence_viewer":
                    # For sequence viewer, use the formatted display directly without truncation
                    if isinstance(tool_result, dict) and "formatted_display" in tool_result:
                        tool_result_summary = tool_result["formatted_display"]
                        tool_results.append(f"=== Protein Sequences: {task_id} ===\n{tool_result_summary}")
                    else:
                        tool_result_summary = str(tool_result)
                        tool_results.append(f"=== Protein Sequences: {task_id} ===\n{tool_result_summary}")
                elif tool_name == "literature_search":
                    # For literature search, truncate long results
                    if len(str(tool_result)) > 2000:
                        tool_result_summary = str(tool_result)[:2000] + "\n\n[... Additional results truncated for context efficiency ...]"
                    else:
                        tool_result_summary = str(tool_result)
                    tool_results.append(f"=== External Literature: {tool_name} ({task_id}) ===\n{tool_result_summary}")
                else:
                    # For other tools, truncate if too long
                    if len(str(tool_result)) > 2000:
                        tool_result_summary = str(tool_result)[:2000] + "\n\n[... Additional results truncated for context efficiency ...]"
                    else:
                        tool_result_summary = str(tool_result)
                    tool_results.append(f"=== External Tool: {tool_name} ({task_id}) ===\n{tool_result_summary}")
                
            elif "aggregated_results" in result:
                # Aggregation result - these are less useful for final context
                aggregation_results.append(f"=== Synthesis: {task_id} ===\nMultiple source integration completed")
        
        # Organize results with database first, then external sources
        if database_results:
            context_parts.extend(database_results)
        if tool_results:
            context_parts.extend(tool_results)
        if aggregation_results:
            context_parts.extend(aggregation_results)
        
        combined_context = "\n\n".join(context_parts) if context_parts else "No context available from task execution."
        
        # Add a summary header for better LLM processing
        if len(all_results) > 1:
            summary_header = f"MULTI-SOURCE ANALYSIS RESULTS ({len(all_results)} tasks completed):\n{'='*60}\n"
            combined_context = summary_header + combined_context
        
        return combined_context
    
    async def _retrieve_context(self, query_type: str, retrieval_plan) -> GenomicContext:
        """Retrieve context based on query type and plan."""
        import time
        start_time = time.time()
        
        structured_data = []
        semantic_data = []
        metadata = {}
        
        try:
            # Execute Neo4j query if one was generated (for any query type)
            if hasattr(retrieval_plan, 'neo4j_query') and retrieval_plan.neo4j_query.strip():
                
                # Detect if this is a protein-specific query that should use enhanced protein_info
                protein_search = getattr(retrieval_plan, 'protein_search', '')
                is_protein_query = (
                    'Protein {id:' in retrieval_plan.neo4j_query or  # DSPy generated protein query
                    ('RIFCS' in protein_search or 'scaffold' in protein_search) or  # Protein ID patterns
                    (len(protein_search) > 15 and '_' in protein_search and not ' ' in protein_search)  # Long ID without spaces
                )
                
                if is_protein_query and protein_search.strip():
                    # Use enhanced protein_info query instead of basic cypher
                    neo4j_result = await self.neo4j_processor.process_query(
                        protein_search,
                        query_type="protein_info"
                    )
                else:
                    # Use the generated cypher query
                    print(f"🔍 DSPy Generated Neo4j Query:\n{retrieval_plan.neo4j_query}")
                    
                    # Use the original query for now - enhancement causing issues
                    enhanced_query = retrieval_plan.neo4j_query
                    
                    neo4j_result = await self.neo4j_processor.process_query(
                        enhanced_query,
                        query_type="cypher"
                    )
                    
                
                # Check for Neo4j query errors
                if 'error' in neo4j_result.metadata:
                    # NEW: Check if TaskRepairAgent provided a helpful message
                    if 'repair_message' in neo4j_result.metadata:
                        # Use the repair message instead of raising an exception
                        logger.info(f"Using TaskRepairAgent repair message instead of error")
                        return GenomicContext(
                            structured_data=[],
                            semantic_data=[],
                            metadata={
                                'repair_message': neo4j_result.metadata['repair_message'],
                                'repair_strategy': neo4j_result.metadata.get('repair_strategy', 'unknown'),
                                'query_time': time.time() - start_time
                            },
                            query_time=time.time() - start_time
                        )
                    else:
                        raise Exception(f"Neo4j query failed: {neo4j_result.metadata['error']}")
                
                structured_data = neo4j_result.results
                metadata['neo4j_execution_time'] = neo4j_result.execution_time
            
            if query_type in ["semantic", "hybrid"]:
                # Enhanced logic for semantic and hybrid queries with multi-stage support
                protein_search = getattr(retrieval_plan, 'protein_search', '')
                functional_search = getattr(retrieval_plan, 'functional_search', '')
                primary_database = getattr(retrieval_plan, 'primary_database', 'lancedb')
                requires_multi_stage = getattr(retrieval_plan, 'requires_multi_stage', False)
                seed_proteins_for_similarity = getattr(retrieval_plan, 'seed_proteins_for_similarity', False)
                
                # Debug output for development (can be removed in production)
                logger.debug(f"Query processing: {query_type}, multi_stage: {requires_multi_stage}, seed_proteins: {seed_proteins_for_similarity}")
                
                # Check if protein_search is actually a protein ID (not just a description)
                is_actual_protein_id = (
                    protein_search and (
                        "RIFCS" in protein_search or 
                        any(id_part in protein_search for id_part in ["scaffold", "contigs"]) or
                        (len(protein_search) > 15 and "_" in protein_search and not " " in protein_search)
                    )
                )
                
                if query_type == "semantic":
                    if requires_multi_stage and seed_proteins_for_similarity:
                        # MULTI-STAGE SEMANTIC: Stage 1 already done (Neo4j), now Stage 2 (LanceDB similarity)
                        if structured_data:
                            # Extract protein IDs from Neo4j results to use as similarity seeds
                            seed_protein_ids = []
                            for item in structured_data:
                                protein_id = item.get('protein_id', '')
                                if protein_id:
                                    # Remove "protein:" prefix for LanceDB
                                    clean_id = protein_id.replace("protein:", "") if protein_id.startswith("protein:") else protein_id
                                    seed_protein_ids.append(clean_id)
                            
                            # Run similarity search for each seed protein and aggregate results
                            all_similarity_results = []
                            for seed_id in seed_protein_ids[:3]:  # Limit to top 3 seeds to avoid overload
                                try:
                                    lancedb_result = await self.lancedb_processor.process_query(
                                        seed_id,
                                        query_type="similarity",
                                        limit=3  # Fewer results per seed
                                    )
                                    # Tag results with their seed protein
                                    for result in lancedb_result.results:
                                        result['seed_protein'] = seed_id
                                    all_similarity_results.extend(lancedb_result.results)
                                except Exception as e:
                                    logger.debug(f"Similarity search failed for {seed_id}: {e}")
                            
                            # Remove duplicates and sort by similarity
                            seen_proteins = set()
                            unique_results = []
                            for result in sorted(all_similarity_results, key=lambda x: x.get('similarity', -999), reverse=True):
                                protein_id = result.get('protein_id', '')
                                if protein_id not in seen_proteins:
                                    seen_proteins.add(protein_id)
                                    unique_results.append(result)
                            
                            semantic_data = unique_results[:5]  # Top 5 overall
                            metadata['lancedb_execution_time'] = sum([r.get('execution_time', 0) for r in all_similarity_results])
                            metadata['multi_stage_seeds_used'] = len(seed_protein_ids)
                    
                    elif is_actual_protein_id:
                        # SEMANTIC with protein ID: Neo4j for context → LanceDB for similarity
                        protein_info_result = await self.neo4j_processor.process_query(
                            protein_search,
                            query_type="protein_info"
                        )
                        structured_data.extend(protein_info_result.results)
                        metadata['protein_info_time'] = protein_info_result.execution_time
                        
                        # Execute similarity search (excluding self)
                        # Remove "protein:" prefix for LanceDB search
                        clean_protein_id = protein_search.replace("protein:", "") if protein_search.startswith("protein:") else protein_search
                        
                        lancedb_result = await self.lancedb_processor.process_query(
                            clean_protein_id,
                            query_type="similarity",
                            limit=max(5, self.config.max_results_per_query // 2)
                        )
                        semantic_data = lancedb_result.results
                        metadata['lancedb_execution_time'] = lancedb_result.execution_time
                    
                    elif functional_search or (protein_search and not is_actual_protein_id):
                        # SEMANTIC with functional description: LanceDB semantic search
                        search_term = functional_search if functional_search else protein_search
                        lancedb_result = await self.lancedb_processor.process_query(
                            search_term,
                            query_type="functional_search",
                            limit=self.config.max_results_per_query
                        )
                        semantic_data = lancedb_result.results
                        metadata['lancedb_execution_time'] = lancedb_result.execution_time
                
                elif query_type == "hybrid":
                    # HYBRID: Combine both approaches based on primary_database guidance
                    if primary_database == "lancedb" and functional_search:
                        # LanceDB similarity → Neo4j context for results
                        lancedb_result = await self.lancedb_processor.process_query(
                            functional_search,
                            query_type="functional_search",
                            limit=max(3, self.config.max_results_per_query // 2)
                        )
                        semantic_data = lancedb_result.results
                        metadata['lancedb_execution_time'] = lancedb_result.execution_time
                        
                        # Get additional context for top LanceDB results from Neo4j
                        if semantic_data:
                            top_protein_ids = [item.get('protein_id') for item in semantic_data[:3]]
                            for protein_id in top_protein_ids:
                                if protein_id:
                                    try:
                                        protein_context = await self.neo4j_processor.process_query(
                                            protein_id,
                                            query_type="protein_info"
                                        )
                                        structured_data.extend(protein_context.results)
                                    except Exception as e:
                                        logger.debug(f"Could not get context for {protein_id}: {e}")
                    
                    elif is_actual_protein_id:
                        # HYBRID starting from protein ID: detailed context + similarity
                        protein_info_result = await self.neo4j_processor.process_query(
                            protein_search,
                            query_type="protein_info"
                        )
                        structured_data.extend(protein_info_result.results)
                        metadata['protein_info_time'] = protein_info_result.execution_time
                        
                        # Add similarity search
                        lancedb_result = await self.lancedb_processor.process_query(
                            protein_search,
                            query_type="similarity",
                            limit=max(3, self.config.max_results_per_query // 3)
                        )
                        semantic_data = lancedb_result.results
                        metadata['lancedb_execution_time'] = lancedb_result.execution_time
            
            if query_type == "general":
                # General database overview
                neo4j_result = await self.neo4j_processor.process_query(
                    "database overview",
                    query_type="auto"
                )
                structured_data = neo4j_result.results
                metadata['neo4j_execution_time'] = neo4j_result.execution_time
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            metadata['retrieval_error'] = str(e)
        
        query_time = time.time() - start_time
        
        return GenomicContext(
            structured_data=structured_data,
            semantic_data=semantic_data,
            metadata=metadata,
            query_time=query_time
        )
    
    def _format_context(self, context: GenomicContext) -> str:
        """Format context for LLM consumption with enhanced genomic intelligence and quantitative insights."""
        formatted_parts = []
        
        # Safe formatting functions to handle string values from Neo4j
        def _safe_format_int(val, default="N/A"):
            """Safely format integer values (handles strings from Neo4j)."""
            try:
                if val is None or val == "N/A":
                    return default
                if isinstance(val, int):
                    return f"{val:,}"
                if isinstance(val, str):
                    return f"{int(val):,}"
                return default
            except (ValueError, TypeError):
                return default
                
        def _safe_format_float(val, precision=2, default="N/A"):
            """Safely format float values (handles strings from Neo4j)."""
            try:
                if val is None or val == "N/A":
                    return default
                if isinstance(val, (int, float)):
                    return f"{val:.{precision}f}"
                if isinstance(val, str):
                    return f"{float(val):.{precision}f}"
                return default
            except (ValueError, TypeError):
                return default
        
        def _analyze_neighbor_functions(neighbors: list) -> dict:
            """Analyze functional themes in genomic neighborhood."""
            if not neighbors:
                return {}
            
            # Define functional categories for small model compatibility
            transport_keywords = ['transport', 'ABC', 'permease', 'receptor', 'transporter']
            metabolism_keywords = ['synthase', 'reductase', 'transferase', 'kinase', 'dehydrogenase']
            regulation_keywords = ['sigma', 'regulator', 'activator', 'repressor', 'transcriptional']
            mobile_keywords = ['transpos', 'insertion', 'IS66', 'mobile', 'recombinase']
            
            themes = {
                'transport': 0,
                'metabolism': 0, 
                'regulation': 0,
                'mobile_elements': 0
            }
            
            for neighbor in neighbors:
                descriptions = neighbor.get('pfam_description', []) + neighbor.get('kegg_descriptions', [])
                for desc in descriptions:
                    if not desc or desc == 'None':
                        continue
                    desc_lower = desc.lower()
                    
                    if any(kw in desc_lower for kw in transport_keywords):
                        themes['transport'] += 1
                    if any(kw in desc_lower for kw in metabolism_keywords):
                        themes['metabolism'] += 1
                    if any(kw in desc_lower for kw in regulation_keywords):
                        themes['regulation'] += 1
                    if any(kw in desc_lower for kw in mobile_keywords):
                        themes['mobile_elements'] += 1
            
            # Determine dominant theme
            dominant_theme = max(themes, key=themes.get) if max(themes.values()) > 0 else None
            return {
                'themes': themes,
                'dominant_theme': dominant_theme,
                'total_annotated': sum(themes.values())
            }
        
        
        def _format_neighbor_context(neighbors: list, target_gene: dict) -> list:
            """Format rich neighbor context for LLM - scaffold neighbors only."""
            if not neighbors or not target_gene.get('gene_start'):
                return []
            
            target_pos = int(target_gene['gene_start'])
            
            # Extract target scaffold ID for filtering
            target_protein_id = target_gene.get('protein_id', '')
            target_scaffold = None
            if 'scaffold_' in target_protein_id:
                # Extract scaffold number from protein ID
                parts = target_protein_id.split('scaffold_')
                if len(parts) > 1:
                    scaffold_part = parts[1].split('_')[0]  # Get just the scaffold number
                    target_scaffold = f"scaffold_{scaffold_part}"
            
            # Group neighbors by protein_id, filtering by same scaffold only
            protein_groups = {}
            for neighbor in neighbors:
                if not neighbor.get('position') or not neighbor.get('protein_id'):
                    continue
                
                protein_id = neighbor['protein_id']
                
                # Only include neighbors from same scaffold
                if target_scaffold and 'scaffold_' in protein_id:
                    neighbor_parts = protein_id.split('scaffold_')
                    if len(neighbor_parts) > 1:
                        neighbor_scaffold_part = neighbor_parts[1].split('_')[0]
                        neighbor_scaffold = f"scaffold_{neighbor_scaffold_part}"
                        if neighbor_scaffold != target_scaffold:
                            continue  # Skip neighbors from different scaffolds
                
                if protein_id not in protein_groups:
                    distance = abs(neighbor['position'] - target_pos)
                    direction = 'upstream' if neighbor['position'] < target_pos else 'downstream'
                    
                    protein_groups[protein_id] = {
                        'protein_id': protein_id,
                        'distance': distance,
                        'direction': direction,
                        'strand': '+' if str(neighbor.get('strand', '0')) == '1' else '-' if str(neighbor.get('strand', '0')) == '-1' else '?',
                        'pfam_id': [],
                        'pfam_description': [],
                        'kegg_ko': [],
                        'kegg_descriptions': []
                    }
                
                # Add domains from this record
                if neighbor.get('pfam_ids') and neighbor['pfam_ids'] != 'None':
                    protein_groups[protein_id]['pfam_id'].append(neighbor['pfam_ids'])
                if neighbor.get('pfam_desc') and neighbor['pfam_desc'] != 'None':
                    protein_groups[protein_id]['pfam_description'].append(neighbor['pfam_desc'])
                if neighbor.get('kegg_id') and neighbor['kegg_id'] != 'None':
                    protein_groups[protein_id]['kegg_ko'].append(neighbor['kegg_id'])
                if neighbor.get('kegg_desc') and neighbor['kegg_desc'] != 'None':
                    protein_groups[protein_id]['kegg_descriptions'].append(neighbor['kegg_desc'])
            
            # Convert to list and sort by distance
            processed_neighbors = list(protein_groups.values())
            processed_neighbors.sort(key=lambda x: x['distance'])
            
            return processed_neighbors[:5]  # Show more neighbors since they're now truly local
        
        if context.structured_data:
            # Detect different query patterns
            is_domain_query = any(
                'p.id' in item and 'd.id' in item and 'd.bitscore' in item 
                for item in context.structured_data
            )
            
            is_count_query = any(
                'numberOfProteins' in item or 
                any(key.lower().startswith('count') or key.lower().endswith('_count') for key in item.keys()) or
                '_domain_total_count' in item  # Detect enhanced domain queries
                for item in context.structured_data
            )
            
            if is_count_query:
                formatted_parts.append("QUANTITATIVE ANALYSIS:")
                
                # Handle enhanced domain count metadata first (show total once)
                domain_total_shown = False
                for item in context.structured_data:
                    if '_domain_total_count' in item and not domain_total_shown:
                        domain_name = item.get('_domain_name', 'Unknown')
                        total_count = item.get('_domain_total_count', 0)
                        is_sample = item.get('_is_sample', False)
                        sample_size = item.get('_sample_size', 0)
                        
                        if is_sample and total_count > sample_size:
                            formatted_parts.append(f"  • Total {domain_name} domains in dataset: {_safe_format_int(total_count)} (showing {_safe_format_int(sample_size)} representative examples)")
                        else:
                            formatted_parts.append(f"  • Total {domain_name} domains found: {_safe_format_int(total_count)}")
                        domain_total_shown = True
                        break
                
                # Then handle regular count fields and genome quality metrics
                for item in context.structured_data:
                    for key, value in item.items():
                        if key.startswith('_'):  # Skip metadata fields
                            continue
                        
                        # Handle genome quality metrics specifically
                        if key == 'total_length_bp':
                            formatted_parts.append(f"  • Total genome length: {_safe_format_int(value)} bp")
                        elif key == 'contig_count':
                            formatted_parts.append(f"  • Contig count: {_safe_format_int(value)}")
                        elif key == 'largest_contig_bp':
                            formatted_parts.append(f"  • Largest contig: {_safe_format_int(value)} bp")
                        elif key == 'n50':
                            formatted_parts.append(f"  • N50: {_safe_format_int(value)} bp")
                        elif key == 'n75':
                            formatted_parts.append(f"  • N75: {_safe_format_int(value)} bp")
                        elif key == 'gc_content':
                            gc_val = _safe_format_float(value, precision=1)
                            if gc_val != "N/A":
                                try:
                                    gc_percent = float(str(value)) * 100 if float(str(value)) <= 1.0 else float(str(value))
                                    formatted_parts.append(f"  • GC content: {gc_percent:.1f}%")
                                except:
                                    formatted_parts.append(f"  • GC content: {gc_val}%")
                        elif key == 'contigs_gt_1kbp':
                            formatted_parts.append(f"  • Contigs ≥1kb: {_safe_format_int(value)}")
                        elif key == 'contigs_gt_5kbp':
                            formatted_parts.append(f"  • Contigs ≥5kb: {_safe_format_int(value)}")
                        elif key == 'contigs_gt_10kbp':
                            formatted_parts.append(f"  • Contigs ≥10kb: {_safe_format_int(value)}")
                        
                        # Handle regular count fields
                        elif 'protein' in key.lower() and ('count' in key.lower() or 'number' in key.lower()):
                            formatted_parts.append(f"  • {key.replace('numberOfProteins', 'Total proteins')}: {_safe_format_int(value)}")
                        elif 'domain' in key.lower() and 'count' in key.lower():
                            formatted_parts.append(f"  • {key}: {_safe_format_int(value)}")
                        elif key.endswith('_count') or key.startswith('count'):
                            formatted_parts.append(f"  • {key.replace('_', ' ').title()}: {_safe_format_int(value)}")
                    break  # Only process first item for counts to avoid repetition
            
            if is_domain_query:
                formatted_parts.append("DOMAIN ANALYSIS:")
                
                # Extract and organize domain information
                domain_data = []
                for item in context.structured_data:
                    if 'p.id' in item and 'd.id' in item:
                        protein_id = item['p.id']
                        domain_id = item['d.id']
                        bitscore = item.get('d.bitscore', 'N/A')
                        
                        # Extract domain type and position from domain_id
                        domain_type = "Unknown"
                        position_part = "unknown"
                        if '/domain/' in domain_id:
                            parts = domain_id.split('/domain/')[1]
                            if '/' in parts:
                                domain_type = parts.split('/')[0]
                                position_part = parts.split('/')[1]
                            else:
                                domain_type = parts
                        
                        domain_data.append({
                            'protein_id': protein_id,
                            'domain_type': domain_type,
                            'position': position_part,
                            'bitscore': bitscore
                        })
                
                if domain_data:
                    # Summary statistics with domain type and accurate counts
                    domain_type = domain_data[0]['domain_type']
                    
                    # Check if we have accurate count metadata
                    first_item = context.structured_data[0] if context.structured_data else {}
                    total_count = first_item.get('_domain_total_count', len(domain_data))
                    is_sample = first_item.get('_is_sample', False)
                    sample_size = first_item.get('_sample_size', len(domain_data))
                    
                    if is_sample and total_count > sample_size:
                        formatted_parts.append(f"  • Total {domain_type} domains in dataset: {_safe_format_int(total_count)} (showing {_safe_format_int(sample_size)} representative examples)")
                    else:
                        formatted_parts.append(f"  • Total {domain_type} domains found: {_safe_format_int(total_count)}")
                    
                    # Sort domains by position instead of score
                    sorted_domains = domain_data
                    
                    # Genomic distribution
                    genomes = {}
                    for d in domain_data:
                        parts = d['protein_id'].split('_')
                        if len(parts) >= 3:
                            genome = '_'.join(parts[:3])
                            genomes[genome] = genomes.get(genome, 0) + 1
                    
                    if len(genomes) > 1:
                        formatted_parts.append(f"  • Distribution across {len(genomes)} genomes:")
                        for genome, count in sorted(genomes.items(), key=lambda x: x[1], reverse=True)[:3]:
                            formatted_parts.append(f"    - {genome}: {count} domains")
                    
                    # Example domains
                    formatted_parts.append(f"\n  Example {domain_type} domains:")
                    for i, d in enumerate(sorted_domains[:3], 1):
                        protein_id = d['protein_id'].replace('protein:', '') if d['protein_id'].startswith('protein:') else d['protein_id']
                        formatted_parts.append(f"    {i}. Protein: {protein_id}")
                        formatted_parts.append(f"       Position: {d['position']} aa")
            
            # Handle protein-specific information with enhanced genomic context
            formatted_parts.append("\nPROTEIN ANALYSIS:")
            unique_proteins = {}
            
            # Deduplicate proteins by ID to avoid showing same protein multiple times
            for item in context.structured_data:
                # Handle both legacy field names (protein_id) and Neo4j field names (p.id)
                protein_id = item.get('protein_id') or item.get('p.id')
                if protein_id:
                    if protein_id not in unique_proteins:
                        unique_proteins[protein_id] = item
                    else:
                        # Merge data from multiple records for same protein
                        for key, value in item.items():
                            if key not in unique_proteins[protein_id] or not unique_proteins[protein_id][key]:
                                unique_proteins[protein_id][key] = value
            
            for i, (protein_id, item) in enumerate(list(unique_proteins.items())[:2]):  # Show max 2 proteins
                # Clean protein ID (remove 'protein:' prefix if present)
                clean_protein_id = protein_id.replace('protein:', '') if protein_id.startswith('protein:') else protein_id
                
                # Store clean protein ID for sequence database lookups
                item['protein_id'] = clean_protein_id
                
                # Use clean protein ID for display
                gene_id = clean_protein_id
                
                # Extract genome name from protein ID structure
                # Example: protein:RIFCSPHIGHO2_01_FULL_Gammaproteobacteria_61_200_...
                genome_id = 'N/A'
                contig_id = 'N/A'
                if '_FULL_' in protein_id:
                    parts = protein_id.split('_FULL_')
                    if len(parts) > 1:
                        genome_part = parts[1].split('_')
                        if len(genome_part) > 0:
                            genome_id = genome_part[0]  # Extract first part after _FULL_
                        
                        # Extract contig ID (full protein name with last '_'-delimited field removed)
                        # Example: RIFCSPHIGHO2_01_FULL_Gammaproteobacteria_61_200_rifcsphigho2_01_scaffold_513609_2 
                        #       -> RIFCSPHIGHO2_01_FULL_Gammaproteobacteria_61_200_rifcsphigho2_01_scaffold_513609
                        if '_' in gene_id:
                            parts = gene_id.split('_')
                            if len(parts) > 1:
                                contig_id = '_'.join(parts[:-1])  # Remove last field
                
                formatted_parts.append(f"\nProtein {i+1}:")
                formatted_parts.append(f"  • Protein ID: {clean_protein_id}")
                formatted_parts.append(f"  • Genome: {genome_id}")
                formatted_parts.append(f"  • Contig: {contig_id}")
                
                # Enhanced genomic coordinates with quantitative context
                # Handle both legacy field names and Neo4j field names
                start = item.get('gene_start') or item.get('start_coordinate') or item.get('g.startCoordinate', 'N/A')
                end = item.get('gene_end') or item.get('end_coordinate') or item.get('g.endCoordinate', 'N/A')
                strand = item.get('gene_strand') or item.get('g.strand', 'N/A')
                
                if start != 'N/A' and end != 'N/A':
                    strand_symbol = "+" if str(strand) == "1" else "-" if str(strand) == "-1" else strand
                    
                    # Calculate gene length in bp using safe formatting
                    try:
                        gene_length_bp = abs(int(str(end)) - int(str(start))) + 1
                        formatted_parts.append(f"  • Genomic Location: {_safe_format_int(start)}-{_safe_format_int(end)} bp (strand {strand_symbol})")
                        formatted_parts.append(f"  • Gene Length: {_safe_format_int(gene_length_bp)} bp")
                    except:
                        formatted_parts.append(f"  • Genomic Location: {start}-{end} bp (strand {strand_symbol})")
                    
                    if 'gene_length_aa' in item:
                        formatted_parts.append(f"  • Protein Length: {item.get('gene_length_aa', 'N/A')} amino acids")
                    
                    if 'gene_gc_content' in item:
                        try:
                            gc_content = float(item['gene_gc_content']) * 100
                            formatted_parts.append(f"  • GC Content: {gc_content:.1f}%")
                        except:
                            formatted_parts.append(f"  • GC Content: {item.get('gene_gc_content', 'N/A')}")
                
                # Enhanced domain annotations with quantitative emphasis
                domain_count = item.get('domain_count', 0)
                if domain_count > 0:
                    formatted_parts.append(f"  • Domain Annotations: {domain_count} detected")
                    
                    # Show all domain families
                    if item.get('protein_families') and any(item['protein_families']):
                        families = [f for f in item['protein_families'] if f and f != 'None']
                        if families:
                            formatted_parts.append(f"    - Families: {', '.join(families)}")
                    
                    # Show all domain descriptions  
                    if item.get('domain_descriptions') and any(item['domain_descriptions']):
                        descriptions = [d for d in item['domain_descriptions'] if d and d != 'None']
                        if descriptions:
                            formatted_parts.append(f"    - Functions: {', '.join(descriptions)}")
                    
                    
                    # Skip domain positions unless specifically relevant for analysis
                    # (Positions are technical details that clutter LLM context unless needed)
                
                # Enhanced PFAM domain information (CRITICAL: This was missing!)
                pfam_accessions = item.get('pfam_accessions', [])
                if pfam_accessions and any(pfam_accessions):
                    # Clean up PFAM accessions
                    clean_pfam = [p for p in pfam_accessions if p and p != 'None']
                    if clean_pfam:
                        formatted_parts.append(f"  • PFAM Domains: {', '.join(clean_pfam)}")
                
                # Enhanced KEGG functional information
                # Handle both legacy field names and Neo4j field names
                kegg_id = item.get('kegg_functions') or item.get('ko_id') or item.get('ko.id')
                kegg_desc = item.get('kegg_descriptions') or item.get('ko_description') or item.get('ko.description') or item.get('function_desc')
                
                # Convert single values to lists for consistent processing
                if kegg_id and not isinstance(kegg_id, list):
                    kegg_id = [kegg_id] if kegg_id else []
                if kegg_desc and not isinstance(kegg_desc, list):
                    kegg_desc = [kegg_desc] if kegg_desc else []
                
                if kegg_id and any(kegg_id):
                    functions = [f for f in kegg_id if f and f != 'None']
                    if functions:
                        formatted_parts.append(f"  • KEGG Functions: {', '.join(functions[:2])}")
                        
                        # Show all function descriptions
                        if kegg_desc and any(kegg_desc):
                            descriptions = [d for d in kegg_desc if d and d != 'None']
                            if descriptions:
                                formatted_parts.append(f"    - Details: {', '.join(descriptions)}")
                elif kegg_desc and any(kegg_desc):
                    # Handle case where we have function description but no KEGG ID
                    descriptions = [d for d in kegg_desc if d and d != 'None']
                    if descriptions:
                        formatted_parts.append(f"  • Function: {', '.join(descriptions)}")
                
                # Handle new neighbor analysis format from DSPy queries
                neighbors = item.get('neighbors', [])
                neighbor_functions = item.get('neighbor_functions', [])
                neighbor_domains = item.get('neighbor_domains', [])
                
                if neighbors and any(neighbors):
                    # Clean up neighbor data
                    clean_neighbors = [n for n in neighbors if n and n != 'None']
                    clean_functions = [f for f in neighbor_functions if f and f != 'None'] if neighbor_functions else []
                    clean_domains = [d for d in neighbor_domains if d and d != 'None'] if neighbor_domains else []
                    
                    if clean_neighbors:
                        formatted_parts.append(f"  • Genomic Neighborhood Analysis:")
                        formatted_parts.append(f"    - {len(clean_neighbors)} neighboring proteins within 10kb")
                        
                        if clean_functions:
                            # Show unique neighbor functions (avoid duplicates)
                            unique_functions = list(set(clean_functions))[:5]  # Show top 5 unique functions
                            formatted_parts.append(f"    - Neighbor Functions: {', '.join(unique_functions)}")
                        
                        if clean_domains:
                            # Show unique neighbor domains
                            unique_domains = list(set(clean_domains))[:5]  # Show top 5 unique domains
                            formatted_parts.append(f"    - Neighbor Domains: {', '.join(unique_domains)}")
                
                # Handle enhanced neighbor_details format with distances and directions
                if item.get('neighbor_details'):
                    try:
                        neighbor_details = [n for n in item['neighbor_details'] if n and n.get('neighbor_id')]
                        if neighbor_details:
                            formatted_parts.append(f"  • Detailed Genomic Neighborhood Analysis:")
                            formatted_parts.append(f"    - {len(neighbor_details)} neighboring proteins with precise positioning")
                        
                        # Sort neighbors by distance for proximity analysis
                        sorted_neighbors = sorted(neighbor_details, key=lambda x: x.get('distance', 0))
                        
                        # Analyze close neighbors (0-200bp)
                        close_neighbors = [n for n in sorted_neighbors if n.get('distance', float('inf')) < 200]
                        if close_neighbors:
                            formatted_parts.append(f"    - Close neighbors (0-200bp): {len(close_neighbors)}")
                            for neighbor in close_neighbors[:3]:  # Show top 3 closest
                                distance = neighbor.get('distance', 'unknown')
                                direction = neighbor.get('direction', 'unknown')
                                strand = neighbor.get('neighbor_strand', 'unknown')
                                function = neighbor.get('function', 'unknown function')
                                # Handle null function values
                                if function is None:
                                    function = 'unknown function'
                                target_strand = item.get('gene_strand') or item.get('g.strand')
                                same_strand = str(strand) == str(target_strand)
                                cotranscription = "likely co-transcribed" if same_strand and distance < 200 else "different regulation"
                                formatted_parts.append(f"      • {distance}bp {direction}, strand {strand} ({cotranscription}): {function[:50]}")
                        
                        # Analyze proximal neighbors (200-500bp)
                        proximal_neighbors = [n for n in sorted_neighbors if 200 <= n.get('distance', float('inf')) < 500]
                        if proximal_neighbors:
                            formatted_parts.append(f"    - Proximal neighbors (200-500bp): {len(proximal_neighbors)}")
                            for neighbor in proximal_neighbors[:2]:  # Show top 2
                                distance = neighbor.get('distance', 'unknown')
                                direction = neighbor.get('direction', 'unknown')
                                strand = neighbor.get('neighbor_strand', 'unknown')
                                function = neighbor.get('function', 'unknown function')
                                # Handle null function values
                                if function is None:
                                    function = 'unknown function'
                                target_strand = item.get('gene_strand') or item.get('g.strand')
                                same_strand = str(strand) == str(target_strand)
                                regulation = "same strand" if same_strand else "different strand"
                                formatted_parts.append(f"      • {distance}bp {direction}, strand {strand} ({regulation}): {function[:50]}")
                        
                        # Analyze distal neighbors (>500bp)
                        distal_neighbors = [n for n in sorted_neighbors if n.get('distance', float('inf')) > 500]
                        if distal_neighbors:
                            formatted_parts.append(f"    - Distal neighbors (>500bp): {len(distal_neighbors)}")
                            for neighbor in distal_neighbors[:2]:  # Show top 2
                                distance = neighbor.get('distance', 'unknown')
                                direction = neighbor.get('direction', 'unknown')
                                function = neighbor.get('function', 'unknown function')
                                # Handle null function values
                                if function is None:
                                    function = 'unknown function'
                                formatted_parts.append(f"      • {distance}bp {direction}: {function[:50]}")
                    except Exception as e:
                        print(f"Error processing neighbor_details: {e}")
                        # Fall back to basic neighbor info if available
                        if item.get('neighbors'):
                            formatted_parts.append(f"  • Basic Neighborhood Analysis:")
                            formatted_parts.append(f"    - {len(item['neighbors'])} neighboring proteins within 10kb")
                
                # Enhanced genomic neighborhood analysis using detailed_neighbors
                if item.get('detailed_neighbors'):
                    neighbors = [n for n in item['detailed_neighbors'] if n and n.get('neighbor_id')]
                    if neighbors:
                        # Analyze functional themes
                        target_gene = {
                            'gene_start': item.get('gene_start'),
                            'gene_strand': item.get('gene_strand'),
                            'protein_id': item.get('protein_id')
                        }
                        
                        # Filter neighbors to same scaffold first
                        target_protein_id = item.get('protein_id', '')
                        target_scaffold = None
                        if 'scaffold_' in target_protein_id:
                            parts = target_protein_id.split('scaffold_')
                            if len(parts) > 1:
                                scaffold_part = parts[1].split('_')[0]
                                target_scaffold = f"scaffold_{scaffold_part}"
                        
                        scaffold_neighbors = []
                        if target_scaffold:
                            for neighbor in neighbors:
                                if neighbor.get('protein_id') and 'scaffold_' in neighbor['protein_id']:
                                    neighbor_parts = neighbor['protein_id'].split('scaffold_')
                                    if len(neighbor_parts) > 1:
                                        neighbor_scaffold_part = neighbor_parts[1].split('_')[0]
                                        neighbor_scaffold = f"scaffold_{neighbor_scaffold_part}"
                                        if neighbor_scaffold == target_scaffold:
                                            scaffold_neighbors.append(neighbor)
                        
                        functional_analysis = _analyze_neighbor_functions(scaffold_neighbors)
                        formatted_neighbors = _format_neighbor_context(neighbors, target_gene)
                        
                        # Calculate intergenic distances and strand relationships
                        if item.get('gene_start'):
                            target_pos = int(item['gene_start'])
                            target_strand = item.get('gene_strand')
                            
                            # Process neighbors with distance calculations
                            neighbor_analysis = []
                            for neighbor in neighbors:
                                neighbor_start = neighbor.get('neighbor_start')
                                neighbor_strand = neighbor.get('neighbor_strand')
                                neighbor_id = neighbor.get('neighbor_id', '')
                                neighbor_function = neighbor.get('neighbor_function') or neighbor.get('neighbor_domain') or 'unknown function'
                                
                                if neighbor_start:
                                    try:
                                        distance = abs(int(neighbor_start) - target_pos)
                                        direction = 'upstream' if int(neighbor_start) < target_pos else 'downstream'
                                        
                                        # Determine strand compatibility
                                        same_strand = str(neighbor_strand) == str(target_strand)
                                        strand_info = f"same strand" if same_strand else f"opposite strand"
                                        
                                        # Classify proximity for biological interpretation
                                        if distance < 200:
                                            proximity = "close" if same_strand else "close_opposite"
                                            operon_status = "likely co-transcribed" if same_strand else "separate regulation"
                                        elif distance < 500:
                                            proximity = "proximal" if same_strand else "proximal_opposite"
                                            operon_status = "possible operon" if same_strand else "separate regulation"
                                        else:
                                            proximity = "distal"
                                            operon_status = "separate regulation"
                                        
                                        neighbor_analysis.append({
                                            'id': neighbor_id,
                                            'distance': distance,
                                            'direction': direction,
                                            'strand': neighbor_strand,
                                            'same_strand': same_strand,
                                            'strand_info': strand_info,
                                            'proximity': proximity,
                                            'operon_status': operon_status,
                                            'function': neighbor_function
                                        })
                                    except (ValueError, TypeError):
                                        continue
                            
                            # Sort by distance for reporting
                            neighbor_analysis.sort(key=lambda x: x['distance'])
                            
                            upstream_count = sum(1 for n in neighbor_analysis if n['direction'] == 'upstream')
                            downstream_count = len(neighbor_analysis) - upstream_count
                            
                            formatted_parts.append(f"  • Genomic Neighborhood Analysis:")
                            formatted_parts.append(f"    - {len(neighbor_analysis)} proteins within 5kb ({upstream_count} upstream, {downstream_count} downstream)")
                            
                            # Report close neighbors with operon analysis
                            close_neighbors = [n for n in neighbor_analysis if n['proximity'] in ['close', 'close_opposite']]
                            if close_neighbors:
                                formatted_parts.append(f"    \n    Close Neighbors (0-200bp):")
                                for neighbor in close_neighbors[:3]:
                                    short_id = neighbor['id'].split('_')[-1] if '_' in neighbor['id'] else neighbor['id']
                                    formatted_parts.append(f"    • {short_id}: {neighbor['distance']}bp {neighbor['direction']}, {neighbor['strand_info']}")
                                    formatted_parts.append(f"      - Status: {neighbor['operon_status']}")
                                    formatted_parts.append(f"      - Function: {neighbor['function'][:60]}")
                            
                            # Report proximal neighbors
                            proximal_neighbors = [n for n in neighbor_analysis if n['proximity'] in ['proximal', 'proximal_opposite']]
                            if proximal_neighbors:
                                formatted_parts.append(f"    \n    Proximal Neighbors (200-500bp):")
                                for neighbor in proximal_neighbors[:2]:
                                    short_id = neighbor['id'].split('_')[-1] if '_' in neighbor['id'] else neighbor['id']
                                    formatted_parts.append(f"    • {short_id}: {neighbor['distance']}bp {neighbor['direction']}, {neighbor['strand_info']}")
                                    formatted_parts.append(f"      - Function: {neighbor['function'][:60]}")
                            
                            # Summary of operon potential
                            likely_operonic = sum(1 for n in close_neighbors if n['operon_status'] == 'likely co-transcribed')
                            if likely_operonic > 0:
                                formatted_parts.append(f"    \n    Operon Analysis: {likely_operonic} gene(s) likely co-transcribed (same strand, <200bp)")
                            else:
                                formatted_parts.append(f"    \n    Operon Analysis: No evidence for co-transcription (no same-strand close neighbors)")
                        else:
                            formatted_parts.append(f"  • Genomic Neighborhood: {len(neighbors)} proteins (coordinates not available for distance analysis)")
                
        # Handle general structured data that doesn't match specific patterns
        if context.structured_data and not any(['p.id' in str(item) or 'protein_id' in str(item) for item in context.structured_data]):
            formatted_parts.append("\nSTRUCTURED DATA ANALYSIS:")
            
            # Handle contig/genomic queries
            if any('contig' in str(item) for item in context.structured_data):
                formatted_parts.append("  GENOMIC CONTIGS:")
                for i, item in enumerate(context.structured_data[:10]):  # Show up to 10 contigs
                    contig_id = item.get('contig_id', 'Unknown')
                    count = item.get('ribosomal_gene_count') or item.get('gene_count') or item.get('count', 0)
                    formatted_parts.append(f"    {i+1}. {contig_id}: {count} genes")
            
            # Handle pathway queries
            elif any('pathway' in str(item) for item in context.structured_data):
                formatted_parts.append("  PATHWAY ANALYSIS:")
                for i, item in enumerate(context.structured_data[:5]):
                    pathway_name = item.get('pathway_name') or item.get('name', 'Unknown pathway')
                    count = item.get('protein_count') or item.get('ko_count') or item.get('count', 0)
                    formatted_parts.append(f"    {i+1}. {pathway_name}: {count} proteins")
            
            # Handle general aggregation queries
            else:
                formatted_parts.append("  QUERY RESULTS:")
                for i, item in enumerate(context.structured_data[:10]):
                    # Show all non-metadata fields
                    item_parts = []
                    for key, value in item.items():
                        if not key.startswith('_'):  # Skip metadata
                            item_parts.append(f"{key}: {value}")
                    if item_parts:
                        formatted_parts.append(f"    {i+1}. {', '.join(item_parts)}")

        # Handle semantic similarity data with enhanced formatting
        if context.semantic_data:
            formatted_parts.append("\nFUNCTIONALLY SIMILAR PROTEINS:")
            for i, item in enumerate(context.semantic_data[:3]):  # Show top 3 similar proteins
                similarity = item.get('similarity', 0)
                protein_id = item.get('protein_id', 'Unknown')
                clean_protein_id = protein_id.replace('protein:', '') if protein_id.startswith('protein:') else protein_id
                
                formatted_parts.append(f"  {i+1}. {clean_protein_id} (ESM2 similarity: {similarity:.3f})")
                formatted_parts.append(f"     Genome: {item.get('genome_id', 'Unknown')}")
                
                # Add biological interpretation of similarity scores with enhanced context
                if similarity > 0.95:
                    formatted_parts.append(f"     Interpretation: IDENTICAL/ORTHOLOG (>95% similarity) - functionally equivalent")
                elif similarity > 0.8:
                    formatted_parts.append(f"     Interpretation: HIGHLY SIMILAR - likely functional ortholog with conserved structure")
                elif similarity > 0.6:
                    formatted_parts.append(f"     Interpretation: MODERATELY SIMILAR - possible functional analog or domain similarity")
                elif similarity > 0.4:
                    formatted_parts.append(f"     Interpretation: WEAKLY SIMILAR - distantly related or shared domain")
                else:
                    formatted_parts.append(f"     Interpretation: LOW SIMILARITY - possible distant evolutionary relationship")
                
                # Add functional annotation comparison if available
                if item.get('pfam_domains'):
                    domains = [d for d in item['pfam_domains'] if d and d != 'None']
                    if domains:
                        formatted_parts.append(f"     Domains: {', '.join(domains[:3])}")
                
                if item.get('kegg_functions'):
                    functions = [f for f in item['kegg_functions'] if f and f != 'None']
                    if functions:
                        formatted_parts.append(f"     KEGG Functions: {', '.join(functions[:2])}")
                
                # Add sequence length comparison if available
                if item.get('protein_length'):
                    length = item['protein_length']
                    formatted_parts.append(f"     Protein Length: {length} amino acids")
                
                # Add genomic context if available
                if item.get('start_coordinate') and item.get('end_coordinate'):
                    start = item['start_coordinate']
                    end = item['end_coordinate']
                    strand = item.get('strand', '?')
                    strand_symbol = "+" if str(strand) == "1" else "-" if str(strand) == "-1" else strand
                    formatted_parts.append(f"     Genomic Location: {_safe_format_int(start)}-{_safe_format_int(end)} bp (strand {strand_symbol})")
                
                # Note: Full protein ID is already displayed above
        
        return "\n".join(formatted_parts) if formatted_parts else "No relevant genomic context found."
    
    def close(self):
        """Clean up resources."""
        self.neo4j_processor.close()
        # LanceDB doesn't need explicit closing


# Example questions for testing
EXAMPLE_GENOMIC_QUESTIONS = [
    "How many genomes are in the database?",
    "What proteins are found in Burkholderiales_bacterium_RIFCSPHIGHO2_01_FULL_64_960_contigs?",
    "Find proteins similar to RIFCSPHIGHO2_01_FULL_Gammaproteobacteria_61_200_rifcsphigho2_01_scaffold_29964_1",
    "What KEGG functions are associated with ATP synthase?",
    "Which genome has the most protein families?",
    "What are the most common protein domains across all genomes?",
    "Find all proteins involved in energy metabolism",
    "Compare the functional profiles of different genomes",
    "What is the average protein length in each genome?",
    "Which proteins have the most functional annotations?"
]


async def demo_rag_system(config: LLMConfig):
    """Demonstration of the RAG system with example questions."""
    console.print("[bold green]🧬 Genomic RAG System Demo[/bold green]")
    console.print("="*60)
    
    rag = GenomicRAG(config)
    
    # Health check
    health = rag.health_check()
    console.print(f"System health: {health}")
    
    if not all(health.values()):
        console.print("[red]⚠️  Some components are not healthy. Check configuration.[/red]")
        return
    
    # Test with a few example questions
    for question in EXAMPLE_GENOMIC_QUESTIONS[:3]:  # Test first 3 questions
        console.print(f"\n[bold cyan]Question:[/bold cyan] {question}")
        
        response = await rag.ask(question)
        
        console.print(f"[bold green]Answer:[/bold green] {response['answer']}")
        console.print(f"[bold yellow]Confidence:[/bold yellow] {response['confidence']}")
        console.print(f"[dim]Query time: {response['query_metadata']['retrieval_time']:.2f}s[/dim]")
        console.print("-" * 40)
    
    rag.close()
    console.print("\n[green]✅ Demo completed successfully![/green]")


if __name__ == "__main__":
    import os
    
    # For testing - requires OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        console.print("[red]Please set OPENAI_API_KEY environment variable[/red]")
        exit(1)
    
    config = LLMConfig.from_env()
    asyncio.run(demo_rag_system(config))