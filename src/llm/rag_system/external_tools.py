#!/usr/bin/env python3
"""
External tools integration for agentic workflows.
Includes literature search, code interpreter, and tool registry.
"""

import logging
from typing import Dict, Any, Optional
import asyncio
import json

logger = logging.getLogger(__name__)

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

# Tool registry for agentic workflows
AVAILABLE_TOOLS = {
    "literature_search": literature_search,
    "code_interpreter": code_interpreter_tool,
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