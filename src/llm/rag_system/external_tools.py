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

logger = logging.getLogger(__name__)

async def whole_genome_reader_tool(genome_id: str = None, global_analysis: bool = False, rag_system=None, **kwargs) -> str:
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
    try:
        # Handle parameter variations from task parsing
        if kwargs.get('global', False) or global_analysis:
            global_analysis = True
            
        # Handle empty genome_id case (often means global analysis was intended)
        if not genome_id or genome_id.strip() == "":
            if not global_analysis:
                logger.info("🌐 Empty genome_id provided, defaulting to global analysis")
                global_analysis = True
        
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
            
            return result["tool_output"]
        else:
            logger.error(f"❌ Failed to read genome(s): {result['error']}")
            return f"Genome reading failed: {result['error']}"
            
    except Exception as e:
        logger.error(f"Whole genome reader tool failed: {e}")
        return f"Genome reading tool error: {str(e)}"

async def genome_selector_tool(query: str, rag_system, **kwargs) -> str:
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
            return f"Selected genome: {selection_result.selected_genome} (confidence: {selection_result.match_score:.2f}, reason: {selection_result.match_reason})"
        else:
            available_info = f" Available genomes: {', '.join(selection_result.available_genomes[:5])}..." if selection_result.available_genomes else ""
            return f"Genome selection failed: {selection_result.error_message}.{available_info}"
            
    except Exception as e:
        logger.error(f"Genome selector tool failed: {e}")
        return f"Genome selection tool error: {str(e)}"

def literature_search(query: str, email: str, **kwargs) -> str:
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
            return f"No PubMed results found for query: {query}"
        
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
        formatted_results = []
        formatted_results.append(f"PubMed Search Results for: {query}")
        formatted_results.append(f"Found {len(id_list)} articles\\n")
        
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
                
            except Exception as e:
                logger.warning(f"Error formatting article {i}: {e}")
                formatted_results.append(f"[{i}] Error formatting article: {e}\\n")
        
        return "\\n".join(formatted_results)
        
    except ImportError:
        return "Literature search requires Biopython (pip install biopython)"
    except Exception as e:
        logger.error(f"Literature search failed: {e}")
        return f"Literature search failed: {e}"

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
                logger.info(f"✅ Code execution completed successfully")
                return result
            else:
                error_msg = f"Code interpreter service error: {response.status_code}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "output": "",
                    "error": error_msg,
                    "execution_time": 0.0
                }
                
    except httpx.ConnectError:
        error_msg = "Code interpreter service not available - is the container running?"
        logger.error(error_msg)
        return {
            "success": False,
            "output": "",
            "error": error_msg,
            "execution_time": 0.0
        }
    except httpx.TimeoutException:
        error_msg = f"Code execution timed out after {timeout} seconds"
        logger.error(error_msg)
        return {
            "success": False,
            "output": "",
            "error": error_msg,
            "execution_time": timeout
        }
    except Exception as e:
        error_msg = f"Code interpreter error: {e}"
        logger.error(error_msg)
        return {
            "success": False,
            "output": "",
            "error": error_msg,
            "execution_time": 0.0
        }

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
    
    return {
        "tool_name": "report_synthesis",
        "task_type": "synthesis",
        "description": description,
        "original_question": original_question,
        "status": "synthesis_required",
        "message": "Task requires synthesis of session results rather than database query"
    }

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
        'description': 'Read genome(s) in spatial order for comprehensive operon and prophage analysis',
        'biological_functions': [
            'spatial_genomic_analysis',
            'operon_detection', 
            'prophage_discovery',
            'hypothetical_protein_clustering',
            'genomic_coordinate_analysis',
            'gene_neighborhood_analysis',
            'spatial_pattern_recognition'
        ],
        'input_types': ['genome_sequences', 'annotation_data', 'spatial_coordinates'],
        'output_types': ['spatial_clusters', 'prophage_candidates', 'operon_predictions', 'genomic_context'],
        'analysis_types': ['discovery', 'exploration', 'spatial', 'contextual'],
        'use_cases': [
            'finding operons containing prophage segments',
            'reading through genomes directly',
            'identifying stretches of hypothetical proteins',
            'spatial genomic analysis',
            'gene neighborhood analysis'
        ]
    },
    'code_interpreter': {
        'description': 'Execute Python code for statistical analysis and data visualization',
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
        'analysis_types': ['statistical', 'computational', 'quantitative'],
        'use_cases': [
            'computing novelty scores',
            'statistical analysis of protein data',
            'creating visualizations',
            'data transformation and aggregation'
        ]
    },
    'genome_selector': {
        'description': 'Intelligent genome selection for targeted analysis',
        'biological_functions': [
            'genome_targeting',
            'genome_identification',
            'organism_selection',
            'taxonomic_filtering'
        ],
        'input_types': ['biological_queries', 'organism_names', 'taxonomic_terms'],
        'output_types': ['genome_selections', 'targeting_results', 'genome_matches'],
        'analysis_types': ['targeting', 'selection', 'filtering'],
        'use_cases': [
            'selecting specific genomes for analysis',
            'targeting particular organisms',
            'filtering by taxonomic criteria'
        ]
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
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url}/health")
            return response.status_code == 200
    except:
        return False

# Note: Hard-coded discovery functions removed - LLM will perform pattern analysis directly