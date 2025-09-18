#!/usr/bin/env python3
"""
External tools integration for agentic workflows.
Includes literature search, code interpreter, and tool registry.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
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
from ..tools.lancedb_knn import lancedb_knn_tool

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
    import httpx, os
    
    # Log API call and endpoint
    url_for_log = kwargs.get('base_url') or os.getenv('CODE_INTERPRETER_URL') or 'http://localhost:8000'
    logger.info(f"🐍 Executing code in interpreter (session: {session_id}, url: {url_for_log})")
    
    try:
        # Code interpreter service endpoint
        base_url = kwargs.get('base_url') or os.getenv('CODE_INTERPRETER_URL') or 'http://localhost:8000'
        download_dir = kwargs.get('download_dir')
        
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
                
                # Build artifacts list from files_created when available
                artifacts = []
                try:
                    fc = result.get('files_created') or []
                    if isinstance(fc, list):
                        for name in fc:
                            try:
                                artifacts.append({"name": str(name), "type": "file", "path": str(name)})
                            except Exception:
                                pass
                except Exception:
                    artifacts = []

                downloaded: list[str] = []
                # Optionally download artifacts to host
                if download_dir:
                    try:
                        outdir = Path(download_dir)
                        outdir.mkdir(parents=True, exist_ok=True)
                        files = (result.get('files_created') or [])
                        for name in files:
                            got = False
                            # Attempt direct download via file route
                            try:
                                url = f"{base_url}/sessions/{request_data['session_id']}/files/{name}"
                                r = await client.get(url)
                                if r.status_code == 200:
                                    dest = outdir / name
                                    dest.parent.mkdir(parents=True, exist_ok=True)
                                    dest.write_bytes(r.content)
                                    downloaded.append(str(dest))
                                    got = True
                                else:
                                    logger.info(f"CI download skipped (status {r.status_code}) for {url}")
                            except Exception:
                                pass

                            # Fallback via base64/stdout if route missing
                            if not got:
                                try:
                                    dl_code = (
                                        "import builtins as bi, base64\n"
                                        f"p=r'{name}'\n"
                                        "try:\n"
                                        "    b=bi.open(p,'rb').read()\n"
                                        "    print('B64:'+base64.b64encode(b).decode('ascii'))\n"
                                        "except Exception as e:\n"
                                        "    print('ERR:'+str(e))\n"
                                    )
                                    r2 = await client.post(f"{base_url}/execute", json={
                                        'code': dl_code,
                                        'session_id': request_data['session_id'],
                                        'timeout': 20
                                    })
                                    if r2.status_code == 200:
                                        j = r2.json()
                                        out = j.get('stdout') or ''
                                        import re, base64 as _b64
                                        m = re.search(r"B64:([A-Za-z0-9+/=]+)", out)
                                        if m:
                                            data = _b64.b64decode(m.group(1))
                                            dest = outdir / name
                                            dest.parent.mkdir(parents=True, exist_ok=True)
                                            dest.write_bytes(data)
                                            downloaded.append(str(dest))
                                except Exception:
                                    pass
                    except Exception:
                        downloaded = []

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
                        'artifacts': artifacts,
                    }).dict()],
                    summary={"files_created": result.get('files_created'), "downloaded_files": downloaded},
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

async def neighborhood_extractor_tool(
    rag_system,
    protein_id: Optional[str] = None,
    contig: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    k: Optional[int] = None,
    limit: Optional[int] = None,
    protein_ids: Optional[List[str]] = None,
    seeds_limit: Optional[int] = 5,
    **kwargs,
) -> Dict[str, Any]:
    """
    Extract local neighborhoods via curated DB templates (cheap, DB-backed).

    Modes:
    - protein_ids (list) -> per-seed flanking (or k-step) neighborhoods, aggregated
    - protein_id (+ optional k, limit) -> 'protein_neighbors_k' (k) or flanking (default)
    - contig + start + end (+ optional limit) -> 'neighbors_by_window'
    """
    try:
        logger.info(
            f"TOOL_INVOCATION neighborhood_extractor: protein_id={protein_id} protein_ids={bool(protein_ids)} contig={contig} k={k} limit={limit}"
        )
        # Helper to detect placeholder protein IDs that won't exist in DB
        def _is_placeholder_pid(pid: Optional[str]) -> bool:
            if not pid:
                return True
            low = str(pid).lower()
            # Placeholder heuristics: example/placeholder tokens only (no domain-specific assumptions)
            if ("example" in low) or ("placeholder" in low) or ("sample" in low):
                return True
            return False

        debug_info: Dict[str, Any] = {}

        # Batch mode: explicit list of seeds
        if protein_ids and isinstance(protein_ids, list):
            seeds = [pid for pid in protein_ids if isinstance(pid, str) and not _is_placeholder_pid(pid)]
            logger.info(f"NEIGHBORHOOD_BATCH: seeds={len(seeds)} (showing up to 5) sample={seeds[:5]}")
            if seeds_limit:
                try:
                    seeds = seeds[: int(seeds_limit)]
                except Exception:
                    seeds = seeds[:5]
            if not seeds:
                return ToolResultEnvelope(
                    tool_name="neighborhood_extractor",
                    success=False,
                    message="protein_ids provided but none valid",
                ).dict()
            aggregated: List[Dict[str, Any]] = []
            total_rows = 0
            for pid in seeds:
                seed_debug = {"seed_protein_id": pid}
                try:
                    ctx = await rag_system.neo4j_processor.execute_named_template(
                        "protein_gene_context", {"protein_id": pid}
                    )
                    if ctx.results:
                        seed = ctx.results[0]
                        seed_debug.update({
                            "seed_gene_id": seed.get("gene_id"),
                            "seed_contig": seed.get("contig"),
                            "seed_start": seed.get("start"),
                            "seed_end": seed.get("end"),
                            "seed_strand": seed.get("strand"),
                        })
                        try:
                            nxt = await rag_system.neo4j_processor.execute_named_template(
                                "gene_next_degree", {"gene_id": seed.get("gene_id")}
                            )
                            if nxt.results:
                                seed_debug["seed_next_degree"] = nxt.results[0].get("next_degree")
                        except Exception:
                            pass
                except Exception:
                    pass

                if isinstance(k, int):
                    name = "protein_neighbors_k"
                    slots = {"protein_id": pid, "k": k}
                    if isinstance(limit, int):
                        slots["limit"] = limit
                else:
                    name = "protein_flanking_genes_5"
                    slots = {"protein_id": pid}
                qres = await rag_system.neo4j_processor.execute_named_template(name, slots)
                rows = qres.results or []
                if len(rows) == 0:
                    try:
                        contig = seed_debug.get("seed_contig")
                        s0 = int(seed_debug.get("seed_start") or 0)
                        e0 = int(seed_debug.get("seed_end") or 0)
                        window = 10000
                        start_w = max(0, s0 - window)
                        end_w = e0 + window
                        slots2 = {"contig": contig, "start": start_w, "end": end_w}
                        if isinstance(limit, int):
                            slots2["limit"] = limit
                        q2 = await rag_system.neo4j_processor.execute_named_template("neighbors_by_window", slots2)
                        rows = q2.results or []
                        seed_debug["fallback"] = {"template": "neighbors_by_window", "slots": slots2, "row_count": len(rows)}
                    except Exception:
                        pass
                aggregated.append({
                    "seed_protein_id": pid,
                    "template": name,
                    "rows": rows,
                    "debug": seed_debug,
                })
                total_rows += len(rows)

            # Advisory for very large batches
            try:
                import os as _os
                advisory_threshold = int(_os.getenv('NEIGHBORHOOD_BATCH_ADVISORY_SEEDS', '50'))
            except Exception:
                advisory_threshold = 50
            advisory = None
            if len(seeds) >= advisory_threshold:
                advisory = {
                    "type": "large_batch",
                    "message": f"Processing {len(seeds)} seeds; consider narrowing or confirming batch."
                }

            # Summary table (seed → row_count)
            summary_table = [{"seed": a["seed_protein_id"], "row_count": len(a["rows"]) } for a in aggregated]

            return ToolResultEnvelope(
                tool_name="neighborhood_extractor",
                success=True,
                display_text=f"batch_neighborhoods seeds={len(aggregated)} total_rows={total_rows}",
                structured_data=aggregated,
                summary={
                    "mode": "batch",
                    "seeds": [a["seed_protein_id"] for a in aggregated],
                    "total_rows": total_rows,
                    "summary_table": summary_table,
                    "advisory": advisory,
                },
            ).dict()

        if protein_id and not _is_placeholder_pid(protein_id):
            # If caller specified k, use adjacency-expansion; otherwise return fixed 5 upstream/downstream by contig order
            if isinstance(k, int):
                name = "protein_neighbors_k"
                slots = {"protein_id": protein_id, "k": k}
                if isinstance(limit, int):
                    slots["limit"] = limit
            else:
                name = "protein_flanking_genes_5"
                slots = {"protein_id": protein_id}
        elif contig and isinstance(start, int) and isinstance(end, int):
            name = "neighbors_by_window"
            slots = {"contig": contig, "start": int(start), "end": int(end)}
            if isinstance(limit, int):
                slots["limit"] = limit
        else:
            # Attempt auto-seeding from the most recent database_query result (session cache)
            seeds: List[str] = []
            try:
                from pathlib import Path
                import json, re
                session_path = getattr(getattr(rag_system, 'note_keeper', None), 'session_path', None)
                if session_path:
                    tool_dir = Path(session_path) / 'tool_results'
                    if tool_dir.exists():
                        db_files = sorted(tool_dir.glob('db_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
                        for f in db_files[:5]:
                            data = json.loads(f.read_text())
                            rows = (data.get('tool_result') or {}).get('structured_data') or []
                            for row in rows:
                                pid0 = row.get('protein_id')
                                if isinstance(pid0, str) and pid0 not in seeds and not _is_placeholder_pid(pid0):
                                    seeds.append(pid0)
                                s = row.get('p') or row.get('protein') or ''
                                m = re.search(r"'id':\s*'([^']+)'", str(s))
                                if m:
                                    pid = m.group(1)
                                    if pid not in seeds and not _is_placeholder_pid(pid):
                                        seeds.append(pid)
                            if seeds:
                                break
            except Exception:
                seeds = []

            if seeds:
                # Use batch path for up to seeds_limit proteins
                try:
                    limit_n = int(seeds_limit) if seeds_limit is not None else 5
                except Exception:
                    limit_n = 5
                return await neighborhood_extractor_tool(
                    rag_system=rag_system,
                    protein_ids=seeds[:limit_n],
                    k=k,
                    limit=limit,
                    seeds_limit=limit_n,
                )
            else:
                return ToolResultEnvelope(
                    tool_name="neighborhood_extractor",
                    success=False,
                    message=(
                        "Provide either protein_id (optionally k, limit) OR contig+start+end (optionally limit). "
                        "Hint: use a real protein id from prior database_query results; avoid placeholders."
                    ),
                ).dict()

        # Always collect schema/context debug info for protein-based extractions
        try:
            if slots.get("protein_id"):
                ctx = await rag_system.neo4j_processor.execute_named_template(
                    "protein_gene_context", {"protein_id": slots["protein_id"]}
                )
                if ctx.results:
                    seed = ctx.results[0]
                    debug_info.update({
                        "seed_protein_id": slots["protein_id"],
                        "seed_gene_id": seed.get("gene_id"),
                        "seed_contig": seed.get("contig"),
                        "seed_start": seed.get("start"),
                        "seed_end": seed.get("end"),
                        "seed_strand": seed.get("strand"),
                    })
                    # NEXT degree
                    try:
                        nxt = await rag_system.neo4j_processor.execute_named_template(
                            "gene_next_degree", {"gene_id": seed.get("gene_id")}
                        )
                        if nxt.results:
                            debug_info["seed_next_degree"] = nxt.results[0].get("next_degree")
                    except Exception:
                        pass
                    # Contig gene count + index
                    try:
                        gi = await rag_system.neo4j_processor.execute_named_template(
                            "contig_gene_index", {"contig": seed.get("contig"), "gene_id": seed.get("gene_id")}
                        )
                        if gi.results:
                            debug_info.update(gi.results[0])
                    except Exception:
                        pass
        except Exception:
            pass

        logger.info(f"🧭 Neighborhood extractor executing template={name} slots={slots}")
        qres = await rag_system.neo4j_processor.execute_named_template(name, slots)
        rows = qres.results or []
        try:
            logger.info(f"🧭 Neighborhood result rows={len(rows)} debug={debug_info}")
        except Exception:
            pass

        # Fallback: if no rows returned, try a coordinate window
        if len(rows) == 0 and name in ("protein_neighbors_k", "protein_flanking_genes_5"):
            try:
                ctx = await rag_system.neo4j_processor.execute_named_template(
                    "protein_gene_context", {"protein_id": slots["protein_id"]}
                )
                if ctx.results:
                    contig = ctx.results[0].get("contig")
                    s0 = int(ctx.results[0].get("start") or 0)
                    e0 = int(ctx.results[0].get("end") or 0)
                    # Use a generous ±10kb window
                    window = 10000
                    start_w = max(0, s0 - window)
                    end_w = e0 + window
                    slots2 = {"contig": contig, "start": start_w, "end": end_w}
                    if isinstance(limit, int):
                        slots2["limit"] = limit
                    logger.info(f"🧭 Neighborhood fallback neighbors_by_window slots={slots2}")
                    q2 = await rag_system.neo4j_processor.execute_named_template("neighbors_by_window", slots2)
                    rows2 = q2.results or []
                    logger.info(f"🧭 Neighborhood fallback rows={len(rows2)}")
                    debug_info["fallback"] = {"template": "neighbors_by_window", "slots": slots2, "row_count": len(rows2)}
                    return ToolResultEnvelope(
                        tool_name="neighborhood_extractor",
                        success=True,
                        display_text=f"neighbors_by_window rows={len(rows2)}",
                        structured_data=rows2,
                        summary={"template": "neighbors_by_window", "slots": slots2, "row_count": len(rows2), "debug": debug_info},
                    ).dict()
            except Exception as _e:
                # If fallback fails, proceed to return the original empty result
                pass

        return ToolResultEnvelope(
            tool_name="neighborhood_extractor",
            success=True,
            display_text=f"{name} rows={len(rows)}",
            structured_data=rows,
            summary={"template": name, "slots": slots, "row_count": len(rows), "debug": debug_info},
        ).dict()
    except Exception as e:
        logger.error(f"Neighborhood extractor failed: {e}")
        return ToolResultEnvelope(
            tool_name="neighborhood_extractor",
            success=False,
            message=str(e),
        ).dict()

async def annotation_discovery_tool(rag_system, keyword: Optional[str] = None, limit: int = 100, protein_limit: int = 100, **kwargs) -> Dict[str, Any]:
    """
    Discover candidate PFAM and KOFAM annotations matching a keyword, then fetch proteins annotated with any of them.

    - Searches Domain (PFAM) and KEGGOrtholog for case-insensitive matches to 'keyword'.
    - Unions proteins with any of those PFAMs or KOs.
    - Returns deduplicated proteins and the list of candidate annotations used.
    """
    try:
        if not keyword or not str(keyword).strip():
            return ToolResultEnvelope(
                tool_name="annotation_discovery",
                success=False,
                message="annotation_discovery requires a 'keyword' parameter (case-insensitive substring)",
            ).dict()
        kw = str(keyword).strip()
        # Helper: extract protein_id robustly from Neo4j row
        import re as _re
        def _extract_pid(row: Dict[str, Any]) -> Optional[str]:
            if not isinstance(row, dict):
                return None
            # Common forms: explicit field, nested node under 'p' or 'protein'
            pid = row.get('protein_id') or row.get('id')
            if isinstance(pid, str) and pid:
                return pid
            p = row.get('p') or row.get('protein') or None
            if p is not None:
                # Try mapping-like access first
                try:
                    val = p.get('id')  # type: ignore[attr-defined]
                    if isinstance(val, str) and val:
                        return val
                except Exception:
                    pass
                try:
                    val = p['id']  # type: ignore[index]
                    if isinstance(val, str) and val:
                        return val
                except Exception:
                    pass
                # Fallback: parse string representation
                try:
                    m = _re.search(r"'id':\s*'([^']+)'", str(p))
                    if m:
                        return m.group(1)
                except Exception:
                    pass
            # Last resort: scan all values
            try:
                text = str(row)
                m = _re.search(r"protein:?['\"]?id['\"]?[:=]\s*['\"]([^'\"]+)['\"]", text)
                if m:
                    return m.group(1)
            except Exception:
                pass
            return None

        # Find candidate PFAMs
        pfam_slots = {"q": kw, "limit": int(limit)}
        pfam_res = await rag_system.neo4j_processor.execute_named_template("pfam_search", pfam_slots)
        pfams = []
        for r in pfam_res.results or []:
            if r.get("pfam"):
                pfams.append(r["pfam"])  # accession preferred
            elif r.get("id"):
                pfams.append(r["id"])   # fallback to id
        pfams = list({x for x in pfams})

        # Find candidate KOs
        ko_slots = {"q": kw, "limit": int(limit)}
        ko_res = await rag_system.neo4j_processor.execute_named_template("kofam_search", ko_slots)
        kos = [r["id"] for r in (ko_res.results or []) if r.get("id")]
        kos = list({x for x in kos})

        proteins: Dict[str, Dict[str, Any]] = {}
        # Fetch proteins by PFAMs
        if pfams:
            pfp = await rag_system.neo4j_processor.execute_named_template(
                "proteins_with_pfams", {"pfams": pfams, "limit": int(protein_limit)}
            )
            for row in pfp.results or []:
                pid = _extract_pid(row)
                if pid and pid not in proteins:
                    proteins[pid] = {"protein_id": pid}
        # Fetch proteins by KOs
        if kos:
            pko = await rag_system.neo4j_processor.execute_named_template(
                "proteins_with_kos", {"kos": kos, "limit": int(protein_limit)}
            )
            for row in pko.results or []:
                pid = _extract_pid(row)
                if pid and pid not in proteins:
                    proteins[pid] = {"protein_id": pid}

        summary = {
            "keyword": kw,
            "pfam_candidates": pfams,
            "ko_candidates": kos,
            "protein_count": len(proteins),
        }

        return ToolResultEnvelope(
            tool_name="annotation_discovery",
            success=True,
            display_text=f"annotation_discovery proteins={len(proteins)} pfams={len(pfams)} kos={len(kos)}",
            structured_data=list(proteins.values()),
            summary=summary,
        ).dict()
    except Exception as e:
        logger.error(f"annotation_discovery failed: {e}")
        return ToolResultEnvelope(
            tool_name="annotation_discovery",
            success=False,
            message=str(e),
        ).dict()

# Tool registry for agentic workflows
AVAILABLE_TOOLS = {
    "literature_search": literature_search,
    "code_interpreter": code_interpreter_tool,
    "genome_selector": genome_selector_tool,
    "whole_genome_reader": whole_genome_reader_tool,
    "neighborhood_extractor": neighborhood_extractor_tool,
    "annotation_discovery": annotation_discovery_tool,
    "concept_discovery": None,  # placeholder; set below after definition
    "report_synthesis": report_synthesis_tool,
    "lancedb_knn": lancedb_knn_tool,
}

# Enhanced tool capabilities for agent-based selection
TOOL_CAPABILITIES = {
    'database_query': {
        'description': 'Execute curated Cypher templates (compiled; no free-form queries)',
        'when_to_use': [
            'Look up proteins/genes/domains via registry templates',
            'Count or list entities with deterministic limits',
            'Seed IDs for downstream tools (neighborhoods, LanceDB)'
        ],
        'when_NOT_to_use': [
            'Large spatial scans (use whole_genome_reader)',
            'Neighborhood extraction (use neighborhood_extractor)'
        ],
        'biological_scope': 'discovery|listing|counts',
        'query_indicators': ['find proteins', 'with pfam', 'with ko', 'in pathway'],
        'biological_functions': ['annotation_lookup', 'entity_listing'],
        'input_types': ['template', 'slots'],
        'output_types': ['table'],
        'analysis_types': ['discovery']
    },
    'whole_genome_reader': {
        'description': 'Read complete genome(s) in spatial coordinate order for discovery-based genomic analysis',
        'when_to_use': [
            'Global prophage/phage discovery across ALL genomes',
            'Operon identification requiring broad spatial context (small genomes)',
            'Cross-genome comparative spatial patterns',
            'Analysis requiring reading genes in genomic coordinate order'
        ],
        'when_NOT_to_use': [
            'Simple functional annotation lookups (use database_query)',
            'Counting specific protein types (use database_query)',
            'Direct database searches for known annotations',
            'Questions with specific protein/gene IDs already identified',
            'Neighborhood extraction around specific genes/proteins (use neighborhood_extractor)'
        ],
        'biological_scope': 'global_discovery|spatial_analysis|prophage_discovery',
        'query_indicators': ['prophage', 'phage', 'operon', 'spatial', 'across all genomes'],
        'biological_functions': [
            'spatial_genomic_analysis',
            'prophage_discovery',
            'operon_detection',
            'hypothetical_protein_clustering',
            'genomic_coordinate_analysis'
        ],
        'input_types': ['genome_sequences', 'annotation_data', 'spatial_coordinates'],
        'output_types': ['spatial_clusters', 'prophage_candidates', 'operon_predictions', 'genomic_context'],
        'analysis_types': ['discovery', 'exploration', 'spatial', 'contextual']
    },
    'neighborhood_extractor': {
        'description': 'Extract local gene/protein neighborhoods via curated DB templates',
        'when_to_use': [
            'Neighborhood context around specific integrase/functional candidates',
            'Windowed neighborhoods on a contig when coordinates are known',
            'Adjacency-based k-step neighborhoods from a seed gene/protein'
        ],
        'when_NOT_to_use': [
            'Global spatial scans across large genomes (consider whole_genome_reader for small genomes)',
        ],
        'biological_scope': 'neighborhood_context|local_spatial_analysis',
        'query_indicators': ['neighborhood', 'around', 'upstream', 'downstream', 'window', 'k-step'],
        'biological_functions': [
            'gene_neighborhood_analysis',
            'local_spatial_context'
        ],
        'input_types': ['protein_id', 'protein_ids', 'contig+start+end'],
        'output_types': ['genes', 'proteins'],
        'analysis_types': ['contextual', 'local']
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
    'annotation_discovery': {
        'description': 'Discover candidate annotations (PFAM + KOFAM) matching a keyword and fetch proteins',
        'when_to_use': [
            'Need proteins via multiple annotation sources (PFAM/KOFAM)',
            'Case-insensitive keyword search across PFAM/KOFAM catalogs',
            'Union discovery across PFAM+KOFAM then neighborhood extraction'
        ],
        'biological_scope': 'annotation_lookup|functional_search',
        'query_indicators': ['annotation keyword', 'functional term', 'domain or KO name'],
        'biological_functions': ['annotation_discovery'],
        'input_types': ['keyword'],
        'output_types': ['proteins', 'annotation_candidates'],
        'analysis_types': ['discovery']
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

async def concept_discovery_tool(
    rag_system,
    concept: str,
    n: int = 5,
    flank_k: int = 5,
    max_rounds: int = 2,
    anchor_limit: int = 8,
    protein_limit: int = 200,
    seeds_limit: int = 100,
    **kwargs,
) -> Dict[str, Any]:
    """
    Concept → anchors → PFAM/KO IDs → proteins → batch neighborhoods → select ≥ n loci.
    Deterministic DB retrieval; LLM only proposes short anchors.
    """
    from .tool_schemas import ToolResultEnvelope
    from .memory.model_allocation import get_model_allocator
    from .dspy_signatures import AnnotationAnchorPlanner
    import json as _json
    import logging as _logging
    import re as _re

    logger = _logging.getLogger(__name__)

    concept = (concept or "").strip()
    if not concept:
        return ToolResultEnvelope(
            tool_name="concept_discovery",
            success=False,
            message="concept_discovery requires a non-empty 'concept' string",
        ).dict()

    allocator = get_model_allocator()

    def _plan_anchors(prev: list[str]) -> list[str]:
        hints = _json.dumps({"prior": prev, "max": int(anchor_limit)})
        def call(module):
            return module(concept=concept, hints=hints)
        res = allocator.create_context_managed_call(
            task_name="query_classification",  # cheap tier
            signature_class=AnnotationAnchorPlanner,
            module_call_func=call,
            query=concept,
            task_context="anchor_planning",
        )
        if not res:
            return []
        # Prefer new field 'anchors'; fallback to legacy 'anchors_json'
        text = getattr(res, "anchors", None)
        if not text:
            text = getattr(res, "anchors_json", "[]") or "[]"
        try:
            arr = _json.loads(text)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
        return []

    used_anchors: list[str] = []
    proteins: dict[str, dict] = {}
    rounds_summary: list[dict] = []
    picked_loci: list[dict] = []

    for rnd in range(int(max_rounds)):
        anchors = [a for a in _plan_anchors(used_anchors) if a.lower() not in {x.lower() for x in used_anchors}]
        anchors = anchors[: int(anchor_limit)]
        used_anchors.extend(anchors)
        if not anchors:
            rounds_summary.append({"round": rnd + 1, "anchors": [], "proteins": 0})
            continue

        # Collect PFAM/KOs and proteins per anchor
        pfam_total = 0
        ko_total = 0
        def _extract_pid(row: Dict[str, Any]) -> Optional[str]:
            if not isinstance(row, dict):
                return None
            pid = row.get('protein_id') or row.get('id')
            if isinstance(pid, str) and pid:
                return pid
            p = row.get('p') or row.get('protein') or None
            if p is not None:
                try:
                    val = p.get('id')  # type: ignore[attr-defined]
                    if isinstance(val, str) and val:
                        return val
                except Exception:
                    pass
                try:
                    val = p['id']  # type: ignore[index]
                    if isinstance(val, str) and val:
                        return val
                except Exception:
                    pass
                try:
                    m = _re.search(r"'id':\s*'([^']+)'", str(p))
                    if m:
                        return m.group(1)
                except Exception:
                    pass
            return None

        for a in anchors:
            # PFAM candidates
            pfam_rows = await rag_system.neo4j_processor.execute_named_template("pfam_search", {"q": a, "limit": 200})
            pfam_ids = []
            for r in (pfam_rows.results or []):
                if r.get("pfam"):
                    pfam_ids.append(str(r["pfam"]))  # keep case to match template exact comparison
                elif r.get("id"):
                    pfam_ids.append(str(r["id"]))
            pfam_ids = list(dict.fromkeys(pfam_ids))
            pfam_total += len(pfam_ids)
            # KO candidates
            ko_rows = await rag_system.neo4j_processor.execute_named_template("kofam_search", {"q": a, "limit": 200})
            ko_ids = [str(r.get("id")) for r in (ko_rows.results or []) if r.get("id")]
            ko_ids = list(dict.fromkeys(ko_ids))
            ko_total += len(ko_ids)
            # Proteins by PFAMs
            if pfam_ids:
                pfp = await rag_system.neo4j_processor.execute_named_template(
                    "proteins_with_pfams", {"pfams": pfam_ids, "limit": int(protein_limit)}
                )
                for row in pfp.results or []:
                    pid = _extract_pid(row)
                    if pid and pid not in proteins:
                        proteins[pid] = {"protein_id": pid, "anchor": a}
            # Proteins by KOs
            if ko_ids:
                pko = await rag_system.neo4j_processor.execute_named_template(
                    "proteins_with_kos", {"kos": ko_ids, "limit": int(protein_limit)}
                )
                for row in pko.results or []:
                    pid = _extract_pid(row)
                    if pid and pid not in proteins:
                        proteins[pid] = {"protein_id": pid, "anchor": a}

        rounds_summary.append({
            "round": rnd + 1,
            "anchors": anchors,
            "pfam_candidates": pfam_total,
            "ko_candidates": ko_total,
            "proteins": len(proteins),
        })

        # If we have seeds, extract annotated neighborhoods in batch (pfams/kos) and pick loci
        if proteins:
            seeds = list(proteins.keys())[: int(seeds_limit)]
            try:
                from ..options.template_runner import FileCypherRunner
                runner = FileCypherRunner(rag_system.neo4j_processor.driver)
                seeds_payload = [{
                    'seed_protein_id': pid,
                    'contig_len': 1,
                    'orf_count': 1,
                    'genome_id': '',
                    'contig_id': '',
                } for pid in seeds]
                rows = runner.run_template(
                    "batched_neighborhoods_gated.cypher",
                    {
                        "seeds": seeds_payload,
                        "min_contig_len": 0,
                        "min_orf": 0,
                        "k_window": int(flank_k),
                    },
                )
                agg = rows or []
                for r in agg:
                    sid = r.get('seed_protein_id')
                    neigh = r.get('neighbors') or []
                    if sid and isinstance(neigh, list) and len(neigh) > 0:
                        picked_loci.append({
                            'seed_protein_id': sid,
                            'neighbor_count': len(neigh),
                            'from_anchor': proteins.get(sid, {}).get('anchor'),
                        })
                        if len(picked_loci) >= int(n):
                            break
                if len(picked_loci) >= int(n):
                    break
            except Exception as e:
                logger.info(f"concept_discovery annotated neighborhood fallback failed: {e}")

    display = f"concept_discovery loci={len(picked_loci)} anchors={len(used_anchors)}"
    summary = {
        "concept": concept,
        "n": int(n),
        "flank_k": int(flank_k),
        "rounds": rounds_summary,
        "anchors_used": used_anchors,
    }
    # Ensure 'agg' is defined even when annotated neighborhoods step is skipped or returns nothing
    try:
        agg
    except NameError:
        agg = []
    return ToolResultEnvelope(
        tool_name="concept_discovery",
        success=True,
        display_text=display,
        structured_data=[
            {"loci": picked_loci},
            {"aggregated": agg},
        ],
        summary=summary,
    ).dict()

# Register concept_discovery now that it's defined
AVAILABLE_TOOLS["concept_discovery"] = concept_discovery_tool

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
