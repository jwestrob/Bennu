#!/usr/bin/env python3
"""
External tools integration for agentic workflows.
Includes literature search, code interpreter, and tool registry.
"""

import logging
from typing import Dict, Any, Optional, List
import asyncio
import json
from .whole_genome_reader import read_complete_genome_spatial, read_all_genomes_spatial
from .tool_schemas import (
    ToolResultEnvelope,
    LiteratureArticleModel,
    CodeInterpreterResultModel,
    GenomeSelectorResultModel,
)
from dataclasses import is_dataclass, asdict

logger = logging.getLogger(__name__)

# Cache for genome reading results to reduce API calls
_genome_reading_cache = {}
_cache_hits = 0
_cache_misses = 0

def get_genome_reading_stats():
    """Get caching statistics for genome reading."""
    total_calls = _cache_hits + _cache_misses
    cache_hit_rate = (_cache_hits / total_calls * 100) if total_calls > 0 else 0
    return {
        "total_calls": total_calls,
        "cache_hits": _cache_hits,
        "cache_misses": _cache_misses,
        "cache_hit_rate_percent": cache_hit_rate,
        "api_call_reduction": f"{cache_hit_rate:.1f}% fewer API calls"
    }

async def whole_genome_reader_tool(genome_id: str = None, global_analysis: bool = False, rag_system=None, **kwargs) -> Dict[str, Any]:
    """
    Read genome(s) in spatial order for comprehensive operon and prophage analysis.
    
    This tool provides spatially-ordered genomic context for LLM analysis of operons,
    prophage segments, and other features that require reading genes in genomic order.
    
    Args:
        genome_id: Target genome identifier to read (required if global_analysis=False)
        global_analysis: If True, read ALL genomes spatially (default: False)
        rag_system: RAG system instance with Neo4j processor
        **kwargs: Additional parameters
        
    Returns:
        Formatted genome context for LLM analysis or error message
    """
    global _cache_hits, _cache_misses
    
    try:
        # Handle parameter variations from task parsing
        if kwargs.get('global', False) or global_analysis:
            global_analysis = True
            
        # Handle empty genome_id case (often means global analysis was intended)
        if not genome_id or genome_id.strip() == "":
            if not global_analysis:
                logger.info("🌐 Empty genome_id provided, defaulting to global analysis")
                global_analysis = True
        
        # Create cache key based on ONLY parameters that affect genome reading result
        import hashlib
        
        # Extract only core parameters that affect the actual genome data
        core_params = {
            'genome_id': genome_id or '',  # Normalize empty genome_id
            'global_analysis': global_analysis,
            'max_genes_per_contig': kwargs.get('max_genes_per_contig', 1000),  # Default from whole_genome_reader
            'focus_on_spatial': kwargs.get('focus_on_spatial', False)
        }
        
        # Create deterministic cache key
        cache_key_str = f"genome_id:{core_params['genome_id']}|global:{core_params['global_analysis']}|max_genes:{core_params['max_genes_per_contig']}|spatial:{core_params['focus_on_spatial']}"
        cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()
        
        # Debug cache key for verification
        logger.debug(f"🔑 Cache key components: {cache_key_str}")
        
        # Check cache first
        if cache_key in _genome_reading_cache:
            _cache_hits += 1
            logger.info(f"📚 Cache hit for genome reading (key: {cache_key[:8]}...) - saved genome read!")
            cached = _genome_reading_cache[cache_key]
            if isinstance(cached, dict):
                return cached
            return ToolResultEnvelope(
                tool_name="whole_genome_reader",
                success=True,
                display_text=str(cached),
            ).dict()
        
        _cache_misses += 1
        logger.info(f"💾 Cache miss - reading genome data (key: {cache_key[:8]}...) - performing full read")
        
        # Get Neo4j processor from RAG system
        neo4j_processor = rag_system.neo4j_processor
        
        if global_analysis:
            logger.info(f"🌐 Agent requesting GLOBAL spatial genome reading across all genomes")
            result = await read_all_genomes_spatial(neo4j_processor, **kwargs)
        else:
            logger.info(f"🧬 Agent requesting single genome spatial reading: {genome_id}")
            result = await read_complete_genome_spatial(genome_id, neo4j_processor, **kwargs)
        
        if result["success"]:
            scope = "all genomes" if global_analysis else f"genome {genome_id}"
            logger.info(f"✅ Successfully read {scope} in spatial order")
            
            # Note: Hard-coded discovery logic removed - LLM will analyze raw spatial data
            
            # Wrap in envelope and cache
            structured: List[Dict[str, Any]] = []
            raw_ctx = result.get("raw_context")
            if raw_ctx is not None:
                try:
                    if is_dataclass(raw_ctx):
                        structured.append(asdict(raw_ctx))
                    else:
                        structured.append(raw_ctx.__dict__)
                except Exception:
                    pass

            envelope = ToolResultEnvelope(
                tool_name="whole_genome_reader",
                success=True,
                summary=result.get("summary"),
                display_text=result.get("tool_output"),
                structured_data=structured or None,
            ).dict()

            _genome_reading_cache[cache_key] = envelope
            return envelope
        else:
            logger.error(f"❌ Failed to read genome(s): {result['error']}")
            error_msg = f"Genome reading failed: {result['error']}"
            # Don't cache errors
            return ToolResultEnvelope(
                tool_name="whole_genome_reader",
                success=False,
                message=error_msg,
            ).dict()
            
    except Exception as e:
        logger.error(f"Whole genome reader tool failed: {e}")
        return ToolResultEnvelope(
            tool_name="whole_genome_reader",
            success=False,
            message=str(e),
        ).dict()

async def genome_selector_tool(query: str, rag_system, **kwargs) -> Dict[str, Any]:
    """
    Agent tool for intelligent genome selection when needed.
    
    Args:
        query: The biological query that may require specific genome targeting
        rag_system: RAG system instance with genome selector
        **kwargs: Additional parameters
        
    Returns:
        Genome selection result or error message
    """
    try:
        logger.info(f"🧬 Agent requesting genome selection for: {query}")
        
        # Let the agent decide when to use this tool
        selection_result = await rag_system.genome_selector.select_genome(query)
        
        if selection_result.success:
            disp = f"Selected genome: {selection_result.selected_genome} (confidence: {selection_result.match_score:.2f}, reason: {selection_result.match_reason})"
            env = ToolResultEnvelope(
                tool_name="genome_selector",
                success=True,
                display_text=disp,
                structured_data=[{
                    "selected_genome": selection_result.selected_genome,
                    "match_score": selection_result.match_score,
                    "match_reason": selection_result.match_reason,
                    "available_genomes": selection_result.available_genomes,
                }],
            ).dict()
            return env
        else:
            available_info = f" Available genomes: {', '.join(selection_result.available_genomes[:5])}..." if selection_result.available_genomes else ""
            env = ToolResultEnvelope(
                tool_name="genome_selector",
                success=False,
                message=f"Genome selection failed: {selection_result.error_message}.{available_info}",
            ).dict()
            return env
            
    except Exception as e:
        logger.error(f"Genome selector tool failed: {e}")
        return ToolResultEnvelope(
            tool_name="genome_selector",
            success=False,
            message=str(e),
        ).dict()

def literature_search(query: str, email: str, **kwargs) -> Dict[str, Any]:
    """
    Search PubMed for relevant literature using Biopython.
    
    Args:
        query: Search query (enhanced with biological context)
        email: Email for NCBI API access
        **kwargs: Additional search parameters
        
    Returns:
        Formatted search results with abstracts and citations
    """
    try:
        from Bio import Entrez
        import time
        
        logger.info(f"🔍 Searching PubMed for: {query}")
        
        # Configure Entrez
        Entrez.email = email
        Entrez.api_key = kwargs.get('api_key')  # Optional API key for higher rate limits
        
        # Search parameters
        max_results = kwargs.get('max_results', 5)
        sort = kwargs.get('sort', 'relevance')
        
        # Search PubMed
        search_handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort=sort
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()
        
        if not search_results["IdList"]:
            return ToolResultEnvelope(
                tool_name="literature_search",
                success=True,
                display_text=f"No PubMed results found for query: {query}",
                structured_data=[],
            ).dict()
        
        # Fetch detailed information
        id_list = search_results["IdList"]
        fetch_handle = Entrez.efetch(
            db="pubmed",
            id=id_list,
            rettype="abstract",
            retmode="xml"
        )
        
        # Parse results
        try:
            fetch_results = Entrez.read(fetch_handle)
            fetch_handle.close()
        except Exception as e:
            fetch_handle.close()
            return f"Error parsing PubMed results: {e}"
        
        # Format results
        formatted_results: List[str] = []
        formatted_results.append(f"PubMed Search Results for: {query}")
        formatted_results.append(f"Found {len(id_list)} articles\n")
        articles: List[Dict[str, Any]] = []
        
        for i, article in enumerate(fetch_results['PubmedArticle'], 1):
            try:
                # Extract article information
                medline_citation = article['MedlineCitation']
                article_info = medline_citation['Article']
                
                # Title
                title = article_info.get('ArticleTitle', 'No title available')
                
                # Authors
                try:
                    authors = []
                    author_list = article_info.get('AuthorList', [])
                    for author in author_list[:3]:  # First 3 authors
                        if 'LastName' in author and 'Initials' in author:
                            authors.append(f"{author['LastName']} {author['Initials']}")
                    author_str = ", ".join(authors)
                    if len(author_list) > 3:
                        author_str += " et al."
                except:
                    author_str = "Authors not available"
                
                # Journal and year
                try:
                    journal = article_info['Journal']['Title']
                    pub_date = medline_citation['DateCompleted']
                    year = pub_date.get('Year', 'Unknown year')
                except:
                    journal = "Journal not available"
                    year = "Unknown year"
                
                # Abstract
                try:
                    abstract_list = article_info.get('Abstract', {}).get('AbstractText', [])
                    if abstract_list:
                        abstract = " ".join([str(abs_text) for abs_text in abstract_list])
                        # Truncate long abstracts
                        if len(abstract) > 500:
                            abstract = abstract[:497] + "..."
                    else:
                        abstract = "Abstract not available"
                except:
                    abstract = "Abstract not available"
                
                # PMID
                pmid = medline_citation['PMID']
                
                # Format article entry
                article_entry = [
                    f"[{i}] {title}",
                    f"Authors: {author_str}",
                    f"Journal: {journal} ({year})",
                    f"PMID: {pmid}",
                    f"Abstract: {abstract}",
                    ""
                ]
                
                formatted_results.extend(article_entry)
                articles.append(LiteratureArticleModel(
                    pmid=str(pmid),
                    title=str(title),
                    authors=str(author_str) if author_str else None,
                    journal=str(journal) if journal else None,
                    year=str(year) if year else None,
                    abstract=str(abstract) if abstract else None,
                ).dict())
                
            except Exception as e:
                logger.warning(f"Error formatting article {i}: {e}")
                formatted_results.append(f"[{i}] Error formatting article: {e}\\n")
        
        return ToolResultEnvelope(
            tool_name="literature_search",
            success=True,
            display_text="\n".join(formatted_results),
            structured_data=articles,
        ).dict()
        
    except ImportError:
        return ToolResultEnvelope(
            tool_name="literature_search",
            success=False,
            message="Literature search requires Biopython (pip install biopython)",
        ).dict()
    except Exception as e:
        logger.error(f"Literature search failed: {e}")
        return ToolResultEnvelope(
            tool_name="literature_search",
            success=False,
            message=f"Literature search failed: {e}",
        ).dict()

async def code_interpreter_tool(code: str, session_id: str = None, timeout: int = 30, **kwargs) -> Dict[str, Any]:
    """
    Execute Python code in secure code interpreter container.
    
    Args:
        code: Python code to execute
        session_id: Session ID for persistent sessions
        timeout: Execution timeout in seconds
        **kwargs: Additional parameters
        
    Returns:
        Dict with execution results, output, and error information
    """
    import httpx
    
    logger.info(f"🐍 Executing code in interpreter (session: {session_id})")
    
    try:
        # Code interpreter service endpoint
        base_url = kwargs.get('base_url', 'http://localhost:8000')
        
        # Prepare request
        request_data = {
            'code': code,
            'session_id': session_id or 'default',
            'timeout': timeout
        }
        
        # Execute code
        async with httpx.AsyncClient(timeout=timeout + 5) as client:
            response = await client.post(
                f"{base_url}/execute",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Check actual execution success, not just HTTP success
                if result.get('success', False):
                    logger.info(f"✅ Code execution completed successfully")
                    # Map stdout to output field for consistency
                    if 'stdout' in result and result.get('output') is None:
                        result['output'] = result['stdout']
                else:
                    logger.error(f"❌ Code execution failed: {result.get('error', 'Unknown error')}")
                    # Still map stdout to output in case there was partial output
                    if 'stdout' in result:
                        result['output'] = result['stdout']
                
                env = ToolResultEnvelope(
                    tool_name="code_interpreter",
                    success=bool(result.get('success', False)),
                    display_text=result.get('output') or result.get('stdout'),
                    structured_data=[CodeInterpreterResultModel(**{
                        'session_id': result.get('session_id'),
                        'success': bool(result.get('success', False)),
                        'stdout': result.get('stdout'),
                        'stderr': result.get('stderr'),
                        'output': result.get('output'),
                        'error': result.get('error'),
                        'execution_time': result.get('execution_time'),
                    }).dict()],
                ).dict()
                return env
            else:
                error_msg = f"Code interpreter service error: {response.status_code}"
                logger.error(error_msg)
                return ToolResultEnvelope(
                    tool_name="code_interpreter",
                    success=False,
                    message=error_msg,
                ).dict()
                
    except httpx.ConnectError:
        error_msg = "Code interpreter service not available - is the container running?"
        logger.error(error_msg)
        return ToolResultEnvelope(
            tool_name="code_interpreter",
            success=False,
            message=error_msg,
        ).dict()
    except httpx.TimeoutException:
        error_msg = f"Code execution timed out after {timeout} seconds"
        logger.error(error_msg)
        return ToolResultEnvelope(
            tool_name="code_interpreter",
            success=False,
            message=error_msg,
        ).dict()
    except Exception as e:
        error_msg = f"Code interpreter error: {e}"
        logger.error(error_msg)
        return ToolResultEnvelope(
            tool_name="code_interpreter",
            success=False,
            message=error_msg,
        ).dict()

def report_synthesis_tool(description: str, original_question: str = None, **kwargs) -> Dict[str, Any]:
    """
    Tool for generating reports and synthesizing results from session data.
    
    This tool signals that a report should be generated using the existing
    synthesis system rather than falling back to database queries.
    
    Args:
        description: Task description (e.g., "Generate report")
        original_question: Original user question for context
        **kwargs: Additional arguments
        
    Returns:
        Dict indicating this is a synthesis task
    """
    logger.info(f"🔍 Report synthesis task: {description}")
    
    return ToolResultEnvelope(
        tool_name="report_synthesis",
        success=True,
        message="Task requires synthesis of session results rather than database query",
        summary={
            "task_type": "synthesis",
            "description": description,
            "original_question": original_question,
            "status": "synthesis_required",
        },
    ).dict()

# Tool registry for agentic workflows
AVAILABLE_TOOLS = {
    "literature_search": literature_search,
    "code_interpreter": code_interpreter_tool,
    "genome_selector": genome_selector_tool,
    "whole_genome_reader": whole_genome_reader_tool,
    "report_synthesis": report_synthesis_tool,
}

# Enhanced tool capabilities for agent-based selection
TOOL_CAPABILITIES = {
    'whole_genome_reader': {
        'description': 'Read complete genome(s) in spatial coordinate order for discovery-based genomic analysis',
        'when_to_use': [
            'Global prophage/phage discovery across ALL genomes',
            'Operon identification requiring gene neighborhood context',
            'Spatial analysis of hypothetical protein clusters',
            'Cross-genome comparative spatial patterns',
            'Queries asking to "find", "discover", "explore", "look through" genomic regions',
            'Analysis requiring reading genes in genomic coordinate order',
            'Any query about spatial organization, gene neighborhoods, or genomic context'
        ],
        'when_NOT_to_use': [
            'Simple functional annotation lookups (use database_query)',
            'Counting specific protein types (use database_query)',
            'Direct database searches for known annotations',
            'Questions with specific protein/gene IDs already identified'
        ],
        'biological_scope': 'global_discovery|spatial_analysis|neighborhood_context|prophage_discovery',
        'query_indicators': ['find', 'discover', 'explore', 'prophage', 'phage', 'operon', 'spatial', 'across all genomes', 'through genomes'],
        'biological_functions': [
            'spatial_genomic_analysis',
            'prophage_discovery',
            'operon_detection',
            'hypothetical_protein_clustering',
            'genomic_coordinate_analysis',
            'gene_neighborhood_analysis',
            'spatial_pattern_recognition'
        ],
        'input_types': ['genome_sequences', 'annotation_data', 'spatial_coordinates'],
        'output_types': ['spatial_clusters', 'prophage_candidates', 'operon_predictions', 'genomic_context'],
        'analysis_types': ['discovery', 'exploration', 'spatial', 'contextual']
    },
    'code_interpreter': {
        'description': 'Execute Python code for statistical analysis and data visualization',
        'when_to_use': [
            'Statistical analysis of retrieved genomic data',
            'Creating plots, charts, or visualizations',
            'Computing metrics, scores, or quantitative assessments',
            'Data transformation and matrix operations',
            'Follow-up analysis after data retrieval'
        ],
        'when_NOT_to_use': [
            'Primary data retrieval (use database_query or whole_genome_reader)',
            'Initial genomic searches or discovery',
            'Reading raw genome sequences'
        ],
        'biological_scope': 'quantitative_analysis|visualization|statistical_processing',
        'query_indicators': ['analyze', 'calculate', 'compute', 'visualize', 'plot', 'statistics', 'metrics'],
        'biological_functions': [
            'statistical_analysis',
            'data_visualization',
            'computational_analysis',
            'quantitative_assessment',
            'matrix_operations',
            'novelty_scoring',
            'data_transformation'
        ],
        'input_types': ['structured_data', 'numeric_data', 'datasets', 'analysis_results'],
        'output_types': ['statistics', 'visualizations', 'analysis_reports', 'computed_metrics'],
        'analysis_types': ['statistical', 'computational', 'quantitative']
    },
    'genome_selector': {
        'description': 'Intelligent genome selection for targeted single-genome analysis',
        'when_to_use': [
            'User mentions specific organism names or species',
            'Queries targeting particular taxonomic groups',
            'Need to identify which genome to analyze from organism description',
            'Ambiguous organism references requiring clarification'
        ],
        'when_NOT_to_use': [
            'Global analysis across ALL genomes (use whole_genome_reader with global_analysis=True)',
            'User already specified genome IDs directly',
            'Comparative analysis across multiple genomes'
        ],
        'biological_scope': 'genome_targeting|organism_identification|taxonomic_selection',
        'query_indicators': ['organism', 'species', 'strain', 'specific genome', 'particular genome'],
        'biological_functions': [
            'genome_targeting',
            'genome_identification',
            'organism_selection',
            'taxonomic_filtering'
        ],
        'input_types': ['biological_queries', 'organism_names', 'taxonomic_terms'],
        'output_types': ['genome_selections', 'targeting_results', 'genome_matches'],
        'analysis_types': ['targeting', 'selection', 'filtering']
    },
    'database_query': {
        'description': 'Direct Neo4j database queries for specific annotation lookups',
        'when_to_use': [
            'Simple functional annotation lookups',
            'Counting specific protein types or families',
            'Direct searches for known functional categories',
            'Questions with specific protein/gene IDs already identified',
            'Straightforward database retrieval without spatial context'
        ],
        'when_NOT_to_use': [
            'Discovery queries requiring spatial genome reading',
            'Global prophage/operon identification',
            'Cross-genome comparative spatial analysis',
            'Questions requiring gene neighborhood context'
        ],
        'biological_scope': 'annotation_lookup|functional_search|direct_retrieval',
        'query_indicators': ['count', 'how many', 'show me', 'list', 'what proteins', 'specific annotation'],
        'note': 'This is selected automatically when no specialized tool is appropriate'
    },
    'literature_search': {
        'description': 'Search PubMed for relevant scientific literature',
        'biological_functions': [
            'literature_review',
            'research_background',
            'publication_search',
            'scientific_context'
        ],
        'input_types': ['research_queries', 'biological_terms', 'scientific_concepts'],
        'output_types': ['publications', 'abstracts', 'research_summaries', 'citations'],
        'analysis_types': ['research', 'background', 'literature_review'],
        'use_cases': [
            'finding relevant research papers',
            'gathering scientific background',
            'literature review for biological concepts'
        ]
    },
    'report_synthesis': {
        'description': 'Generate comprehensive reports from analysis results',
        'biological_functions': [
            'result_synthesis',
            'report_generation',
            'finding_compilation',
            'narrative_creation'
        ],
        'input_types': ['analysis_results', 'findings_data', 'discovered_patterns'],
        'output_types': ['comprehensive_reports', 'summaries', 'conclusions', 'recommendations'],
        'analysis_types': ['synthesis', 'reporting', 'compilation'],
        'use_cases': [
            'creating final analysis reports',
            'synthesizing findings from multiple steps',
            'generating narrative summaries'
        ]
    }
}

def register_tool(name: str, function):
    """Register a new tool for agentic workflows."""
    AVAILABLE_TOOLS[name] = function
    logger.info(f"Registered tool: {name}")

def get_tool(name: str):
    """Get a tool function by name."""
    return AVAILABLE_TOOLS.get(name)

def list_available_tools() -> Dict[str, str]:
    """Get list of available tools with descriptions."""
    tool_descriptions = {}
    
    for name, func in AVAILABLE_TOOLS.items():
        if hasattr(func, '__doc__') and func.__doc__:
            # Extract first line of docstring as description
            description = func.__doc__.strip().split('\\n')[0]
            tool_descriptions[name] = description
        else:
            tool_descriptions[name] = "No description available"
    
    return tool_descriptions

# Health check function for code interpreter
async def check_code_interpreter_health(base_url: str = 'http://localhost:8000') -> bool:
    """Check if code interpreter service is healthy."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url}/health")
            return response.status_code == 200
    except:
        return False

# Note: Hard-coded discovery functions removed - LLM will perform pattern analysis directly
