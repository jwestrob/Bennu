"""
Progressive synthesis system for handling large multi-task agentic workflows.

Uses a Map-Reduce architecture to process task notes and raw data efficiently,
with token-based decision making for optimal model utilization.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import os
import tiktoken
import concurrent.futures

from .note_keeper import NoteKeeper
from .note_schemas import TaskNote, SynthesisNote, ConfidenceLevel
from .memory_utils import generate_session_id
from .model_allocation import get_model_allocator
from .tool_result_cache import ToolResultCache

logger = logging.getLogger(__name__)


class ProgressiveSynthesizer:
    """
    Map-Reduce based progressive synthesis system for genomic analysis workflows.
    
    Architecture:
    - Unified entry point that processes both raw_data and task_notes
    - Token-based decision making (not keyword or count based)
    - Direct synthesis for data that fits within model limits
    - Map-Reduce pipeline for larger datasets:
      * Map: Split data into chunks, summarize each chunk
      * Reduce: Combine chunk summaries into final synthesis
    """
    
    def __init__(self, note_keeper: NoteKeeper, chunk_size: int = 8, target_tokens: int = 15000, max_concurrent_calls: int = 6):
        """
        Initialize progressive synthesizer with Map-Reduce architecture and parallel processing.
        
        Args:
            note_keeper: NoteKeeper instance for accessing notes
            chunk_size: Number of tasks to process per chunk (legacy parameter)
            target_tokens: Target token count for each synthesis chunk
            max_concurrent_calls: Maximum number of concurrent API calls (default: 6 for safe rate limiting)
        """
        self.note_keeper = note_keeper
        self.chunk_size = chunk_size
        self.target_tokens = target_tokens
        self.max_concurrent_calls = max_concurrent_calls
        
        # Map-Reduce configuration - Model-aware limits
        # We'll set final limits after model allocator is available
        # Aim for large, near-limit chunks by default
        self.chunk_utilization = 0.93  # target fraction of model context per chunk
        self.direct_synthesis_limit = 28000  # placeholder until updated
        self.map_chunk_limit = 26000  # placeholder until updated
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception as e:
            logger.warning(f"Failed to initialize tokenizer: {e}")
            self.tokenizer = None
        
        # Initialize model allocator for intelligent model selection
        self.model_allocator = get_model_allocator()
        
        # Initialize tool result cache for reference loading
        if hasattr(note_keeper, 'session_path'):
            self.tool_cache = ToolResultCache(str(note_keeper.session_path))
        else:
            self.tool_cache = None
        
        # Caching system to reduce API calls
        self.synthesis_cache = {}  # Cache for synthesis results
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Update chunk limits based on actual model capabilities
        self._update_model_aware_limits()
        
        logger.info("🏗️ ProgressiveSynthesizer initialized with Map-Reduce architecture and caching")

        # Guardrails to prevent unverifiable claims in synthesis
        self._guardrails_text = (
            "CRITICAL GUARDRAILS:\n"
            "- Only mention coverage/read depth, genomic coordinates, contig/scaffold IDs, or 'representative contigs' if they appear explicitly in the provided data. Do not infer or invent.\n"
            "- Whenever you mention any locus or cluster, include the full, unabbreviated contig/scaffold identifier exactly as it appears in the data. If such an identifier is not present, do not mention a locus.\n"
            "- If locus/coverage information is missing, say 'not provided' and avoid locus/coverage claims.\n"
        )

        # Example display cap for compact mode (env override via SUMMARY_EXAMPLE_CAP)
        try:
            self.example_cap = max(1, int(os.getenv("SUMMARY_EXAMPLE_CAP", "10")))
        except Exception:
            self.example_cap = 10

    def _with_guardrails(self, context: str) -> str:
        try:
            return f"{self._guardrails_text}\n\n{context}" if context else self._guardrails_text
        except Exception:
            return context
    
    def _update_model_aware_limits(self):
        """Update chunk limits based on actual model capabilities."""
        try:
            # Get limits for the models we'll be using
            _, final_synthesis_model = self.model_allocator.get_model_for_task("final_synthesis", "")
            # Tune limits close to model context to reduce number of chunks
            max_ctx = int(getattr(final_synthesis_model, 'max_context', 30000) or 30000)
            util = float(getattr(self, 'chunk_utilization', 0.93) or 0.93)
            self.direct_synthesis_limit = max(1000, int(max_ctx * util))
            self.map_chunk_limit = max(1000, int(max_ctx * util))
            
            logger.info(f"📊 Model-aware limits updated: direct={self.direct_synthesis_limit:,}, chunk={self.map_chunk_limit:,}")
            logger.info(f"📊 Models: final_synthesis={final_synthesis_model.model_name}, map_step={map_step_model.model_name}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not update model-aware limits: {e}, using defaults")
    
    def synthesize_progressive(self, 
                             task_notes: List[TaskNote],
                             question: str,
                             synthesis_mode: str = "report",
                             dspy_synthesizer=None,
                             raw_data: List[Dict[str, Any]] = None,
                             rag_system=None) -> str:
        """
        Main entry point for progressive synthesis with hybrid mode support.
        
        Args:
            task_notes: List of TaskNote objects
            question: Original user question
            synthesis_mode: "guidance" for lightweight agent guidance or "report" for comprehensive final analysis
            dspy_synthesizer: Legacy parameter (kept for compatibility; ignored)
            raw_data: Raw data from task execution (prioritized over task_notes)
            rag_system: Legacy parameter (not used in Map-Reduce architecture)
            
        Returns:
            Final comprehensive synthesis or brief guidance summary
        """
        # Warn about deprecated parameters
        if dspy_synthesizer is not None:
            logger.info("ℹ️ dspy_synthesizer legacy parameter provided; it is ignored")
        if rag_system is not None:
            logger.info("ℹ️ rag_system legacy parameter provided; it is ignored")
        
        # HYBRID MODEL: Branch based on synthesis mode
        if synthesis_mode == "guidance":
            logger.info(f"🧭 Guidance synthesis: {len(task_notes)} recent notes")
            return self._guidance_synthesis(task_notes, question)
        else:
            logger.info(f"📊 Report synthesis: {len(task_notes)} notes, {len(raw_data) if raw_data else 0} raw items")
            return self._report_synthesis(task_notes, question, raw_data)
    
    def _guidance_synthesis(self, task_notes: List[TaskNote], question: str) -> str:
        """
        Lightweight guidance synthesis for agent situational awareness.
        
        Args:
            task_notes: Recent task notes (typically last 3 steps)
            question: Original user question
            
        Returns:
            Brief guidance summary (2-3 sentences)
        """
        logger.info("🧭 Running lightweight guidance synthesis")
        
        # Simple format for guidance - just recent findings
        if not task_notes:
            return "Continue exploration - no recent notes available for guidance."
        
        # Format recent findings 
        recent_findings = []
        for note in task_notes[-3:]:  # Last 3 notes max
            findings = " | ".join(note.key_findings) if note.key_findings else note.description
            recent_findings.append(f"Step {note.task_id}: {findings}")
        
        context = "RECENT PROGRESS:\n" + "\n".join(recent_findings)
        
        # Use fast model for guidance
        try:
            guidance = self._call_synthesis_model(
                context=context,
                question=question,
                task_name="guidance_synthesis",  # Maps to MEDIUM = gpt-4.1-mini
                focus="brief guidance for next steps (2-3 sentences max)",
                synthesis_type="summarization"
            )
            
            return guidance
            
        except Exception as e:
            logger.warning(f"⚠️ Guidance synthesis failed: {e}")
            return f"Recent findings: {len(task_notes)} steps completed. Continue systematic exploration."
    
    def _report_synthesis(self, task_notes: List[TaskNote], question: str, raw_data: List[Dict[str, Any]] = None) -> str:
        """
        Comprehensive report synthesis using full Map-Reduce architecture.
        
        Args:
            task_notes: All task notes from session
            question: Original user question  
            raw_data: Raw data from task execution
            
        Returns:
            Comprehensive final report
        """
        logger.info("📊 Running comprehensive report synthesis")
        
        # Step 1: Determine primary data source (prioritize raw_data)
        unified_data = self._prepare_unified_data(raw_data, task_notes)
        
        if not unified_data:
            return "No data available for synthesis."
        
        # Step 2: Smart synthesis mode selection
        synthesis_strategy = self._choose_optimal_synthesis_strategy(unified_data, question)
        
        if synthesis_strategy == "key_findings_only":
            logger.info("🎯 Using key-findings-only synthesis (sufficient discoveries detected)")
            return self._synthesize_from_key_findings_only(unified_data, question)
        
        # Step 3: Token-based decision making for full context synthesis
        total_tokens = self._count_data_tokens(unified_data)
        logger.info(f"📊 Total input tokens: {total_tokens} (strategy: {synthesis_strategy})")
        
        if total_tokens <= self.direct_synthesis_limit:
            logger.info("🎯 Using direct synthesis (data fits within model limits)")
            return self._direct_synthesis(unified_data, question)
        else:
            logger.info("🗂️ Using Map-Reduce synthesis (data exceeds model limits)")
            return self._map_reduce_synthesis(unified_data, question)
    
    def _prepare_unified_data(self, raw_data: Optional[List[Dict[str, Any]]], 
                             task_notes: List[TaskNote]) -> List[Dict[str, Any]]:
        """
        Prepare unified data format for processing.
        
        Args:
            raw_data: Raw data from task execution
            task_notes: Task notes for context
            
        Returns:
            Unified data format (list of dictionaries)
        """
        unified_data = []
        
        # Priority 1: Use raw_data if available
        if raw_data:
            logger.info(f"✅ Using raw_data as primary source ({len(raw_data)} items)")
            unified_data.extend(raw_data)
        
        # Priority 2: Convert task_notes to unified format if no raw_data
        if not raw_data and task_notes:
            logger.info(f"📝 Converting task_notes to unified format ({len(task_notes)} notes)")
            for note in task_notes:
                # REVOLUTIONARY CHANGE: Expand tool result references
                quantitative_data = dict(note.quantitative_data) if note.quantitative_data else {}
                
                # Check if this note has a tool result reference
                if 'tool_result_ref' in quantitative_data and self.tool_cache:
                    result_id = quantitative_data['tool_result_ref']
                    logger.info(f"🔗 Expanding tool result reference: {result_id}")
                    
                    # Load the referenced tool result
                    tool_result = self.tool_cache.retrieve_tool_result(result_id)
                    
                    if tool_result:
                        # Add expanded tool result to quantitative data
                        quantitative_data['expanded_tool_result'] = tool_result
                        logger.info(f"✅ Loaded referenced tool result for {note.task_id}")
                    else:
                        logger.warning(f"⚠️ Failed to load tool result reference: {result_id}")
                
                unified_data.append({
                    'type': 'task_note',
                    'task_id': note.task_id,
                    'description': note.description,
                    'observations': note.observations,
                    'key_findings': note.key_findings,
                    'confidence_level': note.confidence_level.value,
                    'quantitative_data': quantitative_data,
                    'cross_task_connections': [
                        {
                            'connected_task': conn.connected_task,
                            'connection_type': conn.connection_type.value,
                            'description': conn.description
                        } for conn in note.cross_task_connections
                    ]
                })
        
        # Add task_notes as metadata if we have raw_data
        if raw_data and task_notes:
            logger.info(f"📋 Adding task_notes as metadata ({len(task_notes)} notes)")
            unified_data.append({
                'type': 'task_metadata',
                'task_count': len(task_notes),
                'key_insights': [finding for note in task_notes for finding in note.key_findings],
                'cross_connections': [
                    {
                        'from_task': note.task_id,
                        'to_task': conn.connected_task,
                        'relationship': conn.connection_type.value,
                        'description': conn.description
                    } for note in task_notes for conn in note.cross_task_connections
                ]
            })
        
        # DEBUG: Save pre-compression data  
        self._save_debug_data("pre_compression_data", unified_data, "Unified data before intelligent context management")
        
        # CRITICAL OPTIMIZATION: Apply intelligent context management
        if self.tool_cache:
            unified_data = self._apply_intelligent_context_management(unified_data)
        
        # DEBUG: Save post-compression data
        self._save_debug_data("post_compression_data", unified_data, "Unified data after intelligent context management")
        
        return unified_data
    
    def _apply_intelligent_context_management(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply intelligent context management to avoid token explosion.
        
        This method decides whether to include full tool results or just summaries
        based on available token budget and content importance.
        
        Args:
            data: Unified data with potentially expanded tool results
            
        Returns:
            Optimized data with smart context management
        """
        logger.info("🧠 Applying intelligent context management")
        
        # Calculate current token usage
        current_tokens = self._count_data_tokens(data)
        logger.info(f"📊 Current context size: {current_tokens:,} tokens")
        
        # If we're within limits, keep everything
        if current_tokens <= self.direct_synthesis_limit:
            logger.info("✅ Context size acceptable, keeping full tool results")
            return data
        
        # Otherwise, apply compression by replacing large tool results with summaries
        compressed_data = []
        tokens_saved = 0
        
        for item in data:
            if item.get('type') == 'task_note' and 'quantitative_data' in item:
                quant_data = item['quantitative_data']
                
                if 'expanded_tool_result' in quant_data:
                    # Calculate size of expanded result
                    expanded_result = quant_data['expanded_tool_result']
                    expanded_tokens = self._count_data_tokens([expanded_result])
                    
                    # Model-aware compression threshold
                    compression_threshold = self._get_compression_threshold()
                    
                    # If expanded result is large, replace with summary + key findings
                    if expanded_tokens > compression_threshold:
                        logger.info(f"🗜️ Compressing large tool result in {item['task_id']} ({expanded_tokens:,} tokens > {compression_threshold:,} threshold)")
                        
                        # Keep the summary and biological discoveries
                        compressed_quant = dict(quant_data)
                        del compressed_quant['expanded_tool_result']  # Remove large result
                        
                        # Extract and preserve key biological information
                        if 'tool_result_ref' in compressed_quant:
                            result_id = compressed_quant['tool_result_ref']
                            tool_name = result_id.split('_')[0]  # Extract tool name from ID
                            
                            # For WGR results, preserve detailed loci information
                            if tool_name == 'wgr' and isinstance(expanded_result, dict):
                                loci_summary = self._extract_detailed_loci_summary(expanded_result)
                                if loci_summary:
                                    compressed_quant['detailed_loci_summary'] = loci_summary
                                    logger.info(f"🧬 Preserved detailed summary of {len(loci_summary)} loci")
                            
                            # Also extract general discoveries
                            discoveries = self.tool_cache.extract_key_discoveries(tool_name, expanded_result)
                            if discoveries:
                                # Add discoveries to key_findings for preservation
                                item['key_findings'] = list(item.get('key_findings', [])) + discoveries
                                logger.info(f"🔬 Preserved {len(discoveries)} biological discoveries")
                        
                        # Update item with compressed data
                        item = dict(item)
                        item['quantitative_data'] = compressed_quant
                        tokens_saved += expanded_tokens
            
            compressed_data.append(item)
        
        if tokens_saved > 0:
            final_tokens = self._count_data_tokens(compressed_data)
            logger.info(f"🎯 Context compression complete: {tokens_saved:,} tokens saved, final size: {final_tokens:,} tokens")
        
        return compressed_data
    
    def _extract_detailed_loci_summary(self, wgr_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract detailed loci summary from WGR results for synthesis.
        
        This preserves the essential loci information (coordinates, gene counts, 
        biological features) while reducing token count significantly.
        
        Args:
            wgr_result: Whole genome reader result with hierarchical analysis
            
        Returns:
            List of detailed loci summaries for synthesis
        """
        loci_summaries = []
        
        try:
            if 'interesting_loci' in wgr_result:
                interesting_loci = wgr_result['interesting_loci']
                
                # Handle both string and object representations
                for locus_data in interesting_loci:
                    try:
                        if isinstance(locus_data, str):
                            # Parse string representation of InterestingLocus
                            locus_summary = self._parse_locus_string(locus_data)
                        elif isinstance(locus_data, dict):
                            # Already structured data
                            locus_summary = {
                                'contig_id': locus_data.get('contig_id', 'unknown'),
                                'coordinates': f"{locus_data.get('start', 0)}-{locus_data.get('end', 0)}",
                                'gene_count': locus_data.get('gene_count', 0),
                                'hypothetical_count': locus_data.get('hypothetical_count', 0),
                                'locus_type': locus_data.get('locus_type', 'unknown'),
                                'biological_features': locus_data.get('biological_features', [])[:3]  # Top 3 features
                            }
                        else:
                            continue
                            
                        if locus_summary:
                            loci_summaries.append(locus_summary)
                            
                    except Exception as e:
                        logger.warning(f"Error parsing locus data: {e}")
                        continue
                        
        except Exception as e:
            logger.warning(f"Error extracting detailed loci summary: {e}")
        
        # Sort by gene count (largest first) and return top 10 for synthesis
        loci_summaries.sort(key=lambda x: x.get('gene_count', 0), reverse=True)
        return loci_summaries[:10]
    
    def _parse_locus_string(self, locus_str: str) -> Dict[str, Any]:
        """Parse InterestingLocus string representation into structured data."""
        try:
            # Extract key information using regex
            import re
            
            # Parse contig_id
            contig_match = re.search(r"contig_id='([^']+)'", locus_str)
            contig_id = contig_match.group(1) if contig_match else 'unknown'
            
            # Parse coordinates
            start_match = re.search(r"start=(\d+)", locus_str)
            end_match = re.search(r"end=(\d+)", locus_str)
            start = int(start_match.group(1)) if start_match else 0
            end = int(end_match.group(1)) if end_match else 0
            
            # Parse gene counts
            gene_count_match = re.search(r"gene_count=(\d+)", locus_str)
            hypo_count_match = re.search(r"hypothetical_count=(\d+)", locus_str)
            gene_count = int(gene_count_match.group(1)) if gene_count_match else 0
            hypo_count = int(hypo_count_match.group(1)) if hypo_count_match else 0
            
            # Parse locus type
            type_match = re.search(r"locus_type='([^']+)'", locus_str)
            locus_type = type_match.group(1) if type_match else 'unknown'
            
            # Parse biological features
            features_match = re.search(r"biological_features=\[(.*?)\]", locus_str)
            features = []
            if features_match:
                features_str = features_match.group(1)
                # Simple extraction of quoted strings
                feature_matches = re.findall(r"'([^']+)'", features_str)
                features = feature_matches[:3]  # Top 3 features
            
            return {
                'contig_id': contig_id,
                'coordinates': f"{start}-{end}",
                'gene_count': gene_count,
                'hypothetical_count': hypo_count,
                'locus_type': locus_type,
                'biological_features': features
            }
            
        except Exception as e:
            logger.warning(f"Error parsing locus string: {e}")
            return None
    
    def _get_compression_threshold(self) -> int:
        """
        Get compression threshold respecting OpenAI's 30K TPM rate limit.
        
        Returns:
            Token threshold above which tool results should be compressed
        """
        # Respect OpenAI's ~30K token budget for premium (gpt-5)
        # Use conservative limit to avoid rate limiting errors
        return 20000
    
    def _save_debug_data(self, stage_name: str, data: Any, description: str) -> None:
        """
        Save debug data to session notes for data flow analysis.
        
        Args:
            stage_name: Name of the processing stage (e.g., "pre_compression_data")
            data: Data to save for debugging
            description: Human-readable description of this data
        """
        try:
            # Create debug directory if it doesn't exist
            if hasattr(self.note_keeper, 'session_path'):
                debug_dir = self.note_keeper.session_path / "debug_data_flow"
                debug_dir.mkdir(exist_ok=True)
                
                # Create debug file
                from datetime import datetime
                timestamp = datetime.now().strftime("%H%M%S")
                debug_file = debug_dir / f"{stage_name}_{timestamp}.json"
                
                # Prepare debug payload
                debug_payload = {
                    "stage_name": stage_name,
                    "description": description,
                    "timestamp": datetime.now().isoformat(),
                    "data_type": type(data).__name__,
                    "data_size_chars": len(str(data)),
                    "token_estimate": self._count_data_tokens([data]) if isinstance(data, (dict, list)) else 0,
                    "data": data
                }
                
                # Write debug file
                import json
                with open(debug_file, 'w') as f:
                    json.dump(debug_payload, f, indent=2, default=str)
                
                logger.debug(f"🐛 DEBUG: Saved {stage_name} to {debug_file.name} ({debug_payload['data_size_chars']} chars)")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to save debug data for {stage_name}: {e}")
    
    def _count_data_tokens(self, data: List[Dict[str, Any]]) -> int:
        """
        Count tokens in unified data format.
        
        Args:
            data: Unified data to count tokens for
            
        Returns:
            Total token count
        """
        if not self.tokenizer:
            # Fallback estimation: 3.5 characters per token for English
            total_chars = sum(len(str(item)) for item in data)
            return int(total_chars / 3.5)
        
        try:
            # More accurate token counting
            full_text = "\n".join(str(item) for item in data)
            return len(self.tokenizer.encode(full_text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Fallback to character-based estimation
            total_chars = sum(len(str(item)) for item in data)
            return int(total_chars / 3.5)
    
    def _direct_synthesis(self, data: List[Dict[str, Any]], question: str) -> str:
        """
        Direct synthesis for data that fits within model limits.
        
        Args:
            data: Unified data to synthesize
            question: Original user question
            
        Returns:
            Direct synthesis result
        """
        logger.info("🎯 Performing direct synthesis with full context")
        
        # DEBUG: Save pre-synthesis data for debugging
        self._save_debug_data("pre_synthesis_data", data, "Raw unified data before formatting")
        
        # Format data for synthesis
        formatted_data = self._format_data_for_synthesis(data)
        # Extract task graph if present in raw items (e.g., MacroPlanner plan)
        task_graph_text = self._extract_task_graph(data)
        
        # DEBUG: Save formatted synthesis input
        self._save_debug_data("formatted_synthesis_input", formatted_data, "Formatted data sent to LLM")
        
        prefix = f"QUESTION: {question}\n\n"
        tg = ("TASK GRAPH:\n" + task_graph_text + "\n\n") if task_graph_text else ""
        body = f"DATA:\n{formatted_data}"
        synthesis_context = self._with_guardrails(prefix + tg + body)
        # Save exact final synthesis context for auditing
        self._save_debug_data("final_synthesis_context_direct", synthesis_context, "Exact input to final LLM (direct mode)")
        return self._call_synthesis_model(
            context=synthesis_context,
            question=question,
            task_name="final_synthesis",  # Use gpt-5 for complex biological reasoning
            focus="comprehensive biological analysis with full context"
        )
    
    def _map_reduce_synthesis(self, data: List[Dict[str, Any]], question: str) -> str:
        """
        Map-Reduce synthesis for large datasets with sequential processing.
        
        Args:
            data: Unified data to synthesize
            question: Original user question
            
        Returns:
            Map-Reduce synthesis result
        """
        logger.info("🗂️ Performing Map-Reduce synthesis with sequential processing")
        
        # Store question for pre-summarization context
        self._current_question = question
        
        # MAP STEP: Split data into chunks and summarize each sequentially  
        chunks = self._create_chunks(data)
        
        # Log chunk sizes for debugging
        chunk_sizes = [self._count_data_tokens(chunk) for chunk in chunks]
        logger.info(f"📦 Created {len(chunks)} chunks for parallel Map step processing")
        logger.info(f"📊 Chunk sizes: {chunk_sizes} tokens (limit: {self.map_chunk_limit})")
        
        # SINGLE-CHUNK BYPASS: If only one chunk, skip Map-Reduce and go directly to final synthesis
        if len(chunks) == 1:
            logger.info("🎯 Single chunk detected - bypassing Map-Reduce and proceeding directly to final synthesis")
            single_chunk_data = chunks[0]
            formatted_context = self._format_data_for_synthesis(single_chunk_data)
            task_graph_text = self._extract_task_graph(data)
            prefix = f"QUESTION: {question}\n\n"
            tg = ("TASK GRAPH:\n" + task_graph_text + "\n\n") if task_graph_text else ""
            body = f"DATA:\n{formatted_context}"
            synthesis_context = self._with_guardrails(prefix + tg + body)
            # Save exact final synthesis context for auditing
            self._save_debug_data("final_synthesis_context_single_chunk", synthesis_context, "Exact input to final LLM (single-chunk bypass)")
            
            # Use high-capability model for direct synthesis
            final_result = self._call_synthesis_model(
                context=synthesis_context,
                question=question,
                task_name="final_synthesis",  # Use gpt-5 for complex synthesis
                focus="comprehensive biological analysis with full data context",
                synthesis_type="summarization"  # Use regular synthesis, not Map-Reduce
            )
            
            # DEBUG: Save single-chunk bypass output
            self._save_debug_data("single_chunk_bypass_output", final_result, "Direct synthesis from single chunk")
            
            return final_result
        
        # MAP STEP: Parallelize chunk processing
        logger.info(f"🚀 Starting parallel chunk processing with {self.max_concurrent_calls} workers")
        chunk_summaries = [None] * len(chunks)  # Pre-allocate results list
        
        def process_chunk(index_and_chunk):
            index, chunk = index_and_chunk
            try:
                logger.info(f"📝 Processing chunk {index+1}/{len(chunks)}")
                summary = self._map_step(chunk, question, index+1)
                return index, summary
            except Exception as e:
                logger.error(f"❌ Chunk {index+1} processing failed: {e}")
                return index, None
        
        # CRITICAL FIX: Replace parallel with SEQUENTIAL processing to prevent TPM overflow
        logger.info(f"🔄 Processing chunks SEQUENTIALLY to prevent rate limits")
        
        for i, chunk in enumerate(chunks):
            try:
                logger.info(f"📝 Processing chunk {i+1}/{len(chunks)}")
                
                # Add rate limiting between main chunk API calls
                if i > 0:  # Don't delay before first call
                    import time
                    delay = 2.0  # 2 second delay between main chunk API calls
                    logger.info(f"⏳ Rate limiting: waiting {delay}s before next chunk")
                    time.sleep(delay)
                
                summary = self._map_step(chunk, question, i+1)
                if summary:  # Only store successful summaries
                    chunk_summaries[i] = summary
                    logger.info(f"✅ Completed chunk {i+1}/{len(chunks)}")
                else:
                    logger.warning(f"⚠️ Chunk {i+1} returned no summary")
                    
            except Exception as e:
                logger.error(f"❌ Chunk {i+1} failed: {e}")
        
        # Filter out None values from failed chunks
        chunk_summaries = [s for s in chunk_summaries if s is not None]
        logger.info(f"🎯 Sequential chunk processing complete: {len(chunk_summaries)}/{len(chunks)} successful")
        
        # REDUCE STEP: Combine chunk summaries into final synthesis
        logger.info(f"🔄 Reduce step: combining {len(chunk_summaries)} summaries")
        return self._reduce_step(chunk_summaries, question)
    
    def _create_chunks(self, data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Split data into chunks for Map step processing with pre-summarization of oversized items.
        
        Args:
            data: Unified data to chunk
            
        Returns:
            List of data chunks
        """
        # Step 1: Pre-processing - handle oversized items
        logger.info(f"🔍 Pre-processing {len(data)} items for oversized content")
        processed_data = []
        
        for i, item in enumerate(data):
            # Pre-compact large macro_result items before token counting
            try:
                if isinstance(item, dict) and item.get('type') == 'macro_result' and item.get('format') != 'full':
                    name = item.get('name', 'result')
                    rows = item.get('rows') or []
                    total = len(rows)
                    # Keep only up to N examples to cap size
                    examples = rows[: self.example_cap]
                    item = {
                        'type': 'macro_result',
                        'name': name,
                        'rows': examples,
                        'total_rows': total,
                        'format': 'compact',
                        'note': 'pre-compacted for context control (set return_full_rows=true for full JSON on small targets)'
                    }
            except Exception:
                pass

            item_tokens = self._count_data_tokens([item])
            
            if item_tokens > self.map_chunk_limit:
                # This item is oversized, pre-summarize it (NO TRUNCATION - ALWAYS PRESERVE BIOLOGICAL DATA)
                logger.warning(f"📝 Item {i+1}/{len(data)} is oversized ({item_tokens:,} tokens > {self.map_chunk_limit:,}) - will summarize to preserve all biological content")
                
                # Get the question from context (we need to pass this through)
                question = getattr(self, '_current_question', 'biological analysis')
                summarized_item = self._summarize_oversized_item(item, question)
                processed_data.append(summarized_item)
            else:
                # Item is fine as-is
                processed_data.append(item)
        
        logger.info(f"✅ Pre-processing complete: {len(processed_data)} items ready for chunking")
        
        # Step 2: Size-aware packing to produce fewer, larger chunks (Next-Fit Decreasing)
        sized_items = []
        for item in processed_data:
            sized_items.append((item, self._count_data_tokens([item])))
        # Sort by descending size
        sized_items.sort(key=lambda x: x[1], reverse=True)

        chunks: List[List[Dict[str, Any]]] = []
        current_chunk: List[Dict[str, Any]] = []
        current_tokens = 0

        for item, sz in sized_items:
            if current_chunk and current_tokens + sz > self.map_chunk_limit:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            current_chunk.append(item)
            current_tokens += sz

        if current_chunk:
            chunks.append(current_chunk)
        
        # B. Compress any chunks that are still too big
        compressed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_tokens = self._count_data_tokens(chunk)
            if chunk_tokens > self.map_chunk_limit:
                logger.warning(f"🗜️ Chunk {i+1} exceeds limit ({chunk_tokens:,} > {self.map_chunk_limit:,} tokens) - compressing")
                compressed_chunk = self._compress_oversized_chunk(chunk, i+1)
                compressed_chunks.append(compressed_chunk)
            else:
                compressed_chunks.append(chunk)
        
        return compressed_chunks
    
    def _summarize_oversized_item(self, item: Dict[str, Any], question: str) -> Dict[str, Any]:
        """
        Summarize oversized data items using recursive Map-Reduce.
        
        Args:
            item: Oversized data item to summarize
            question: Original user question for context
            
        Returns:
            Summarized item that fits within token limits
        """
        item_tokens = self._count_data_tokens([item])
        logger.warning(f"🔄 Pre-summarizing oversized item ({item_tokens:,} tokens → target: {self.map_chunk_limit:,})")
        
        # Extract the main content to summarize
        content = ""
        if 'result' in item and isinstance(item['result'], dict):
            result = item['result']
            if 'tool_output' in result:
                content = str(result['tool_output'])
            else:
                content = str(result)
        else:
            content = str(item)
        
        # Sub-chunking: Split content into manageable pieces
        sub_chunks = self._split_content_into_subchunks(content)
        logger.info(f"📦 Split oversized item into {len(sub_chunks)} sub-chunks for pre-summarization")
        
        # Sub-Map: SEQUENTIAL sub-chunk summarization to prevent API flood
        logger.info(f"🔄 Starting SEQUENTIAL sub-chunk summarization ({len(sub_chunks)} chunks)")
        sub_summaries = []
        
        for i, sub_chunk in enumerate(sub_chunks):
            try:
                logger.info(f"🔄 Pre-summarizing sub-chunk {i+1}/{len(sub_chunks)}")
                
                # Add aggressive rate limiting between API calls
                if i > 0:  # Don't delay before first call
                    import time
                    delay = 3.0  # 3 second delay between sub-chunk API calls
                    logger.info(f"⏳ Rate limiting: waiting {delay}s before next API call")
                    time.sleep(delay)
                
                summary = self._call_synthesis_model(
                    context=sub_chunk,
                    question=question,
                    task_name="genomic_summarization", 
                    focus=f"key biological patterns and insights from data chunk {i+1}"
                )
                sub_summaries.append(summary)
                logger.info(f"✅ Completed sub-chunk {i+1}/{len(sub_chunks)}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to summarize sub-chunk {i+1}: {e}")
                # Include truncated version as fallback
                truncated = sub_chunk[:5000] + "...[truncated for synthesis]"
                sub_summaries.append(f"Sub-chunk {i+1} (summarization failed): {truncated}")
        
        logger.info(f"🎯 Sequential sub-chunk summarization complete: {len(sub_summaries)}/{len(sub_chunks)} successful")
        
        # Sub-Reduce: Combine all sub-summaries
        if sub_summaries:
            combined_summary = "\n\n".join([
                f"=== Summary Part {i+1} ===\n{summary}" 
                for i, summary in enumerate(sub_summaries)
            ])
            
            # Final synthesis of combined summaries
            final_summary = self._call_synthesis_model(
                context=combined_summary,
                question=question,
                task_name="genomic_summarization",
                focus="integrate biological insights from all data sections"
            )
        else:
            final_summary = f"Pre-summarization failed for oversized item ({item_tokens:,} tokens)"
        
        # Create summarized item preserving structure
        summarized_item = item.copy()
        
        # Replace the large content with summary
        if 'result' in summarized_item and isinstance(summarized_item['result'], dict):
            summarized_item['result'] = summarized_item['result'].copy()
            summarized_item['result']['tool_output'] = final_summary
            summarized_item['result']['_original_token_count'] = item_tokens
            summarized_item['result']['_summarization_applied'] = True
        else:
            summarized_item = {
                'original_type': str(type(item).__name__),
                'summarized_content': final_summary,
                '_original_token_count': item_tokens,
                '_summarization_applied': True
            }
        
        final_tokens = self._count_data_tokens([summarized_item])
        logger.info(f"✅ Pre-summarization complete: {item_tokens:,} → {final_tokens:,} tokens ({(1-final_tokens/item_tokens)*100:.1f}% reduction)")
        
        return summarized_item
    
    def _split_content_into_subchunks(self, content: str) -> List[str]:
        """
        Split large content into smaller sub-chunks for pre-summarization.
        
        Args:
            content: Large content string to split
            
        Returns:
            List of content sub-chunks (maximum 10 chunks to prevent API flood)
        """
        # OPTIMIZED: Create fewer, larger sub-chunks for better performance while preserving all biological data
        max_subchunks = 5  # Even fewer sub-chunks for faster processing
        min_chunk_size = len(content) // max_subchunks  # Ensure we don't exceed max chunks
        
        # Calculate safe sub-chunk size based on model limits
        # Get model that will be used for summarization (genomic_summarization task uses gpt-4.1-mini)
        model_name, model_config = self.model_allocator.get_model_for_task("genomic_summarization", "")
        
        # Use 40% of model context for sub-chunks (leaving room for system messages, response, etc.)
        max_safe_tokens = int(model_config.max_context * 0.4)
        target_subchunk_tokens = min(25000, max_safe_tokens)  # Use smaller of 25k or 40% of model limit
        target_chars = int(target_subchunk_tokens * 3.5)
        
        logger.debug(f"📊 Sub-chunk sizing: {target_subchunk_tokens:,} tokens max for {model_name} (context: {model_config.max_context:,})")
        
        if len(content) <= target_chars:
            return [content]
        
        # Split content into chunks, trying to break at natural boundaries
        sub_chunks = []
        current_pos = 0
        
        while current_pos < len(content):
            end_pos = min(current_pos + target_chars, len(content))
            
            # Try to find a good break point near the target position
            if end_pos < len(content):
                # Look for natural breaks (newlines, periods, etc.)
                break_positions = []
                search_range = min(500, target_chars // 4)  # Search within reasonable range
                
                for offset in range(search_range):
                    if end_pos - offset > current_pos:
                        char = content[end_pos - offset]
                        if char in ['\n\n', '\n', '.', '!', '?']:  # Prefer paragraph/sentence breaks
                            break_positions.append(end_pos - offset + 1)
                
                if break_positions:
                    end_pos = break_positions[0]  # Use the closest good break point
            
            chunk = content[current_pos:end_pos].strip()
            if chunk:
                sub_chunks.append(chunk)
            
            current_pos = end_pos
        
        # ENFORCE HARD LIMIT: If we still have too many chunks, merge them to preserve all data
        if len(sub_chunks) > max_subchunks:
            logger.warning(f"🔄 Too many sub-chunks ({len(sub_chunks)} > {max_subchunks}), merging to stay under limit while preserving all biological data")
            merged_chunks = []
            chunks_per_merge = (len(sub_chunks) + max_subchunks - 1) // max_subchunks  # Ceiling division
            
            for i in range(0, len(sub_chunks), chunks_per_merge):
                merged_chunk = "\n\n".join(sub_chunks[i:i + chunks_per_merge])
                merged_chunks.append(merged_chunk)
            
            sub_chunks = merged_chunks
        
        logger.info(f"📝 Split {len(content):,} chars into {len(sub_chunks)} sub-chunks (max: {max_subchunks}, avg: {len(content)//len(sub_chunks):,} chars each)")
        return sub_chunks
    
    def _map_step(self, chunk: List[Dict[str, Any]], question: str, chunk_id: int) -> Optional[str]:
        """
        Map step: Summarize a single chunk of data.
        
        Args:
            chunk: Data chunk to summarize
            question: Original user question for context
            chunk_id: Chunk identifier for logging
            
        Returns:
            Chunk summary or None if failed
        """
        try:
            # Format chunk for processing
            formatted_chunk = self._format_data_for_synthesis(chunk)
            
            # Use cheaper model for Map step (data extraction task)
            focus = (
                f"preserve specific loci details and identifiers from chunk {chunk_id}; "
                "do not mention loci/coverage unless present; always include full contig IDs when present"
            )
            summary = self._call_synthesis_model(
                context=formatted_chunk,
                question=question,
                task_name="genomic_summarization",  # Use cheaper model for chunk summarization
                focus=focus,
                synthesis_type="map_extraction"
            )
            
            # DEBUG: Save Map step output
            self._save_debug_data(f"map_step_chunk_{chunk_id}_output", summary, f"Map step summary for chunk {chunk_id}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Map step failed for chunk {chunk_id}: {e}")
            return None
    
    def _reduce_step(self, chunk_summaries: List[str], question: str) -> str:
        """
        Reduce step: Combine chunk summaries into final synthesis.
        
        Args:
            chunk_summaries: List of chunk summaries from Map step
            question: Original user question
            
        Returns:
            Final synthesis result
        """
        if not chunk_summaries:
            return "No chunk summaries available for final synthesis."
        
        # Combine all chunk summaries
        combined_context = "\n\n".join([
            f"=== Chunk {i+1} Summary ===\n{summary}" 
            for i, summary in enumerate(chunk_summaries)
        ])
        
        # Add synthesis metadata
        # Include task graph if available from previously unified data (may not be present in reduce step)
        task_graph_text = getattr(self, '_last_task_graph', '') if hasattr(self, '_last_task_graph') else ''
        header = f"QUESTION: {question}\n\nCHUNK SUMMARIES ({len(chunk_summaries)} chunks):\n{combined_context}\n\n"
        tg = ("TASK GRAPH:\n" + task_graph_text + "\n") if task_graph_text else ""
        footer = "SYNTHESIS TASK: Integrate the above chunk summaries into a comprehensive, coherent analysis that addresses the original question.\n"
        synthesis_context = self._with_guardrails(header + tg + footer)
        # Save exact final synthesis context for auditing
        self._save_debug_data("final_synthesis_context_reduce", synthesis_context, "Exact input to final LLM (reduce phase)")
        
        # Use original question directly for intelligent selection
        
        # Use high-capability model for final synthesis
        final_synthesis = self._call_synthesis_model(
            context=synthesis_context,
            question=question,
            task_name="final_synthesis",  # Use gpt-5 for complex integration
            focus="intelligent biological prioritization",
            synthesis_type="reduce_selection"
        )
        
        # DEBUG: Save Reduce step output
        self._save_debug_data("reduce_step_final_output", final_synthesis, f"Final synthesis from {len(chunk_summaries)} chunk summaries")
        
        return final_synthesis

    def _extract_task_graph(self, data: List[Dict[str, Any]]) -> str:
        """Extract and pretty-print a task graph (operator plan) if present in raw items.

        Recognizes a task_note with quantitative_data.plan from MacroPlanner execution.
        """
        try:
            plan = None
            for item in data:
                if isinstance(item, dict) and item.get('type') == 'task_note':
                    qd = item.get('quantitative_data') or {}
                    if isinstance(qd, dict) and qd.get('plan'):
                        plan = qd.get('plan')
                        break
            if not plan:
                return ''
            steps = plan.get('steps', []) if isinstance(plan, dict) else []
            lines = []
            for idx, st in enumerate(steps, 1):
                if not isinstance(st, dict):
                    continue
                op = st.get('op')
                bind = st.get('bind')
                params = st.get('params') or {}
                inputs = st.get('inputs') or {}
                line = f"{idx}. op={op}"
                if bind:
                    line += f" -> {bind}"
                if inputs:
                    line += f" inputs={inputs}"
                if params:
                    line += f" params={params}"
                lines.append(line)
            text = "\n".join(lines)
            # store for reduce step
            self._last_task_graph = text
            return text
        except Exception:
            return ''
    
    
    def _format_data_for_synthesis(self, data: List[Dict[str, Any]]) -> str:
        """
        Format unified data for synthesis model.
        
        Args:
            data: Unified data to format
            
        Returns:
            Formatted string for synthesis
        """
        if not data:
            return "No data available"
        
        formatted_items = []
        for i, item in enumerate(data):
            # Format based on item type
            if item.get('type') == 'task_note':
                formatted_item = f"Task {item['task_id']}: {item['description']}\n"
                formatted_item += f"Observations: {'; '.join(item['observations'])}\n"
                formatted_item += f"Key Findings: {'; '.join(item['key_findings'])}\n"
                if item.get('quantitative_data'):
                    formatted_item += f"Data: {item['quantitative_data']}\n"
                if item.get('cross_task_connections'):
                    connections = [f"{conn['connected_task']} ({conn['connection_type']})" 
                                 for conn in item['cross_task_connections']]
                    formatted_item += f"Connections: {'; '.join(connections)}\n"
            elif item.get('type') == 'macro_result':
                # Compact by default; allow full JSON when explicitly requested and reasonably small
                name = item.get('name', 'result')
                rows = item.get('rows') or []
                # Prefer true total if provided by pre-compaction
                try:
                    total = int(item.get('total_rows')) if item.get('total_rows') is not None else len(rows)
                except Exception:
                    total = len(rows)
                want_full = (item.get('format') == 'full')
                FULL_MAX_ROWS = 2000  # guardrail to avoid blow-ups
                formatted_item = f"Result: {name} (rows={total})\n"
                if want_full and total <= FULL_MAX_ROWS:
                    try:
                        import json
                        formatted_item += "Full rows (JSONL):\n"
                        for ex in rows:
                            try:
                                formatted_item += json.dumps(ex, default=str)[:2000] + "\n"
                            except Exception:
                                formatted_item += str(ex)[:2000] + "\n"
                    except Exception:
                        want_full = False
                if not want_full or total > FULL_MAX_ROWS:
                    # Show up to N compact examples
                    try:
                        examples = rows[: self.example_cap]
                        if examples:
                            formatted_item += f"Examples (up to {self.example_cap}):\n"
                            for ex in examples:
                                if isinstance(ex, dict):
                                    gid = ex.get('genome_id', '')
                                    pid = ex.get('protein_id', '')
                                    pf = ex.get('pfams', [])
                                    ko = ex.get('kos', [])
                                    pf_show = ",".join(pf[:2]) if isinstance(pf, list) else ""
                                    ko_show = ",".join(ko[:2]) if isinstance(ko, list) else ""
                                    formatted_item += f"  - genome={gid} protein={pid} pfams=[{pf_show}] kos=[{ko_show}]\n"
                                else:
                                    formatted_item += f"  - {str(ex)[:200]}\n"
                        formatted_item += ("Note: For small, targeted queries you may request full JSON rows by setting return_full_rows=true on AnnotationDiscovery. "
                                           "For larger datasets, compact summaries are used to avoid excessive context.")
                    except Exception:
                        pass
            elif item.get('type') == 'task_metadata':
                formatted_item = f"Task Metadata ({item['task_count']} tasks):\n"
                formatted_item += f"Key Insights: {'; '.join(item['key_insights'][:5])}\n"
                if item.get('cross_connections'):
                    connections = [f"{conn['from_task']} → {conn['to_task']}" 
                                 for conn in item['cross_connections'][:3]]
                    formatted_item += f"Cross-connections: {'; '.join(connections)}\n"
            elif item.get('type') == 'followup_request':
                formatted_item = "NEXT PASS PROPOSAL:\n"
                try:
                    reason = item.get('reason', '')
                    if reason:
                        formatted_item += f"Reason: {reason}\n"
                    steps = item.get('next_task', {}).get('steps', [])
                    if steps:
                        formatted_item += "Suggested Steps:\n"
                        for idx, st in enumerate(steps, 1):
                            op = st.get('op')
                            params = st.get('params', {})
                            bind = st.get('bind', '')
                            formatted_item += f"  {idx}. {op} params={params} bind={bind}\n"
                    inputs = item.get('inputs_needed', [])
                    if inputs:
                        formatted_item += "Inputs Needed:\n"
                        for inp in inputs:
                            nm = inp.get('name', '?')
                            desc = inp.get('desc', '')
                            ex = inp.get('examples', [])
                            exs = ", ".join(ex[:3]) if isinstance(ex, list) else ""
                            formatted_item += f"  - {nm}: {desc}"
                            if exs:
                                formatted_item += f" (e.g., {exs})"
                            formatted_item += "\n"
                except Exception:
                    formatted_item += str(item) + "\n"
            elif isinstance(item, dict) and 'cards' in item:
                # Structured locus cards from fast path; include PFAM/KO when present
                cards = item.get('cards') or []
                formatted_lines = [f"LocusDiscovery: {len(cards)} loci contextualized"]
                # Scope & provenance: be explicit about ±k windows and seed selection
                try:
                    meta = item.get('meta') if isinstance(item.get('meta'), dict) else {}
                    scope = meta.get('analysis_scope') or {}
                    seedsel = meta.get('seed_selection') or {}
                    window_k = scope.get('window_k')
                    sel_method = seedsel.get('method')
                    if isinstance(window_k, int) or isinstance(sel_method, str):
                        scope_bits = []
                        if isinstance(window_k, int):
                            scope_bits.append(f"±{window_k} genes around seed")
                        if isinstance(sel_method, str):
                            if sel_method == 'id_resolution':
                                scope_bits.append("seeded by PFAM/KO IDs")
                            elif sel_method == 'concept_anchors':
                                scope_bits.append("seeded via concept anchors → PFAM/KO IDs")
                            elif sel_method == 'substring_fallback':
                                scope_bits.append("seeded by substring fallback")
                        if scope_bits:
                            formatted_lines.append("Scope: " + "; ".join(scope_bits) + " (locus windows, not complete islands)")
                except Exception:
                    pass
                # Show marker context if available
                try:
                    marker = item.get('meta', {}).get('marker') if isinstance(item.get('meta'), dict) else None
                    if marker:
                        formatted_lines.append(f"Marker: {marker}")
                except Exception:
                    pass
                for idx, card in enumerate(cards, 1):
                    try:
                        seed = card.get('seed_protein_id') if isinstance(card, dict) else getattr(card, 'seed_protein_id', '?')
                        contig = card.get('contig_id') if isinstance(card, dict) else getattr(card, 'contig_id', '')
                        genome = card.get('genome_id') if isinstance(card, dict) else getattr(card, 'genome_id', '')
                        neighbors = card.get('neighbors') if isinstance(card, dict) else getattr(card, 'neighbors', [])
                        ncount = len(neighbors or [])
                        formatted_lines.append(f"{idx}. seed={seed} contig={contig} genome={genome} neighbors={ncount}")
                        # Seed annotations (PFAM/KO) if present
                        try:
                            spf = card.get('seed_pfams', []) if isinstance(card, dict) else []
                            sko = card.get('seed_kos', []) if isinstance(card, dict) else []
                            spf_show = ", ".join(spf[:3]) if spf else "-"
                            sko_show = ", ".join(sko[:3]) if sko else "-"
                            formatted_lines.append(f"   seed_ann: pfams=[{spf_show}] kos=[{sko_show}]")
                        except Exception:
                            pass
                        # Show up to 3 neighbors with PFAM/KO details when present
                        for nb in (neighbors or [])[:3]:
                            pid = nb.get('protein_id') if isinstance(nb, dict) else nb
                            pfams = nb.get('pfams', []) if isinstance(nb, dict) else []
                            kos = nb.get('kos', []) if isinstance(nb, dict) else []
                            # Show up to 2 identifiers for each to keep concise
                            p_show = ", ".join(pfams[:2]) if pfams else "-"
                            k_show = ", ".join(kos[:2]) if kos else "-"
                            formatted_lines.append(f"   • {pid} pfams=[{p_show}] kos=[{k_show}]")
                    except Exception:
                        # Be robust to shape drift
                        continue
                # Summarize kNN neighbors if present in payload
                nbrs_full = item.get('neighbors_full') if isinstance(item, dict) else None
                stats = item.get('knn_stats') if isinstance(item, dict) else None
                # Always include a LanceDB section if the stage ran (stats present) even if empty
                if isinstance(stats, dict):
                    counts = stats.get('neighbors_counts') or {}
                    topk = stats.get('topk')
                    ns = stats.get('exclude_namespace')
                    needle = stats.get('filter_needle')
                    markers = stats.get('filter_markers') or []
                    formatted_lines.append(f"LanceDB kNN: topk={topk} seeds={len(counts) if isinstance(counts, dict) else 0}")
                    # Briefly show filter criteria
                    try:
                        m_show = ", ".join(markers[:3]) if isinstance(markers, list) and markers else "-"
                        n_show = needle if isinstance(needle, str) and needle else "-"
                        ns_show = ns if isinstance(ns, str) and ns else "-"
                        formatted_lines.append(f"filter: ns={ns_show} needle='{n_show}' markers=[{m_show}]")
                        # Include criteria summary (if provided)
                        ins = stats.get('include_namespace')
                        ineed = stats.get('include_needle')
                        imarks = stats.get('include_markers') or []
                        if ins or ineed or imarks:
                            ins_show = ins if isinstance(ins, str) and ins else "-"
                            in_show = ineed if isinstance(ineed, str) and ineed else "-"
                            im_show = ", ".join(imarks[:3]) if isinstance(imarks, list) and imarks else "-"
                            formatted_lines.append(f"include: ns={ins_show} needle='{in_show}' markers=[{im_show}]")
                    except Exception:
                        pass
                    # Summarize counts (even if detailed neighbors are present)
                    if isinstance(counts, dict) and counts:
                        formatted_lines.append("kNN counts:")
                        shown = 0
                        for sid, cnt in counts.items():
                            if shown >= 5:
                                break
                            formatted_lines.append(f"  - {sid}: {cnt}")
                            shown += 1
                formatted_item = "\n".join(formatted_lines) + "\n"
            else:
                # Raw data item
                formatted_item = f"Data Item {i+1}: {str(item)}\n"
            
            formatted_items.append(formatted_item)
        
        return "\n".join(formatted_items)
    
    def _enforce_context_limits(self, context: str, task_name: str, question: str = "") -> str:
        """
        Enforce strict context limits for the selected model to prevent context overflow.
        
        Args:
            context: Input context to validate/compress
            task_name: Task name for model selection
            question: Question for context (used for compression if needed)
            
        Returns:
            Context guaranteed to fit within model limits
        """
        # Get the model that will be used for this task
        model_name, model_config = self.model_allocator.get_model_for_task(task_name, question)
        
        # Calculate safe limits (leave room for system messages, response, etc.)
        safety_margin = 2000  # Reserve tokens for system messages and response
        max_input_tokens = model_config.max_context - safety_margin
        
        # Count actual tokens in context
        context_tokens = self._count_data_tokens([context]) if isinstance(context, str) else self._count_data_tokens(context)
        
        logger.info(f"🔍 Context check: {context_tokens:,} tokens for {model_name} (limit: {max_input_tokens:,})")
        
        if context_tokens <= max_input_tokens:
            return context  # Safe to use as-is
        
        # Context is too large - need to compress intelligently
        logger.warning(f"⚠️ Context exceeds {model_name} limits ({context_tokens:,} > {max_input_tokens:,}) - applying intelligent compression")
        
        # Calculate compression ratio needed
        compression_ratio = max_input_tokens / context_tokens
        target_chars = int(len(context) * compression_ratio * 0.95)  # Use 95% of target for safety
        
        # Apply intelligent compression that preserves biological information
        compressed_context = self._intelligent_compress(context, target_chars, question)
        
        # Verify the compressed context fits
        final_tokens = self._count_data_tokens([compressed_context])
        logger.info(f"✅ Compression complete: {context_tokens:,} → {final_tokens:,} tokens ({(1-final_tokens/context_tokens)*100:.1f}% reduction)")
        
        if final_tokens > max_input_tokens:
            logger.error(f"❌ CRITICAL: Compressed context still exceeds limits ({final_tokens:,} > {max_input_tokens:,})")
            raise ValueError(f"Cannot compress context enough for {model_name} (limit: {max_input_tokens:,} tokens)")
        
        return compressed_context
    
    def _intelligent_compress(self, context: str, target_chars: int, question: str = "") -> str:
        """
        Intelligently compress context while preserving biological information.
        
        Args:
            context: Original context to compress
            target_chars: Target character count
            question: Question context for relevance scoring
            
        Returns:
            Compressed context preserving key biological information
        """
        if len(context) <= target_chars:
            return context
        
        # Split into logical sections
        sections = context.split('\n\n')
        
        # Score sections by biological relevance
        scored_sections = []
        for i, section in enumerate(sections):
            score = self._score_biological_relevance(section, question)
            scored_sections.append((score, i, section))
        
        # Sort by relevance (highest first)
        scored_sections.sort(reverse=True)
        
        # Build compressed context by adding highest-scoring sections
        compressed_parts = []
        current_chars = 0
        
        for score, original_index, section in scored_sections:
            if current_chars + len(section) + 2 <= target_chars:  # +2 for \n\n
                compressed_parts.append((original_index, section))
                current_chars += len(section) + 2
            else:
                # Try to fit a truncated version of this section
                remaining_chars = target_chars - current_chars - 100  # Leave room for truncation message
                if remaining_chars > 200:  # Only if we have meaningful space
                    truncated = section[:remaining_chars] + "...[section continues]"
                    compressed_parts.append((original_index, truncated))
                break
        
        # Sort by original order and rebuild
        compressed_parts.sort()  # Sort by original_index
        compressed_context = '\n\n'.join([section for _, section in compressed_parts])
        
        # Add compression notice if context was significantly compressed
        reduction = (len(context) - len(compressed_context)) / len(context)
        if reduction > 0.3:  # If >30% reduction
            header = f"[CONTEXT INTELLIGENTLY COMPRESSED - {reduction*100:.1f}% reduction while preserving biological relevance]\n\n"
            compressed_context = header + compressed_context
        
        return compressed_context
    
    def _score_biological_relevance(self, section: str, question: str = "") -> float:
        """
        Score a text section for biological relevance.
        
        Args:
            section: Text section to score
            question: Question context for relevance
            
        Returns:
            Relevance score (higher = more important to preserve)
        """
        score = 0.0
        section_lower = section.lower()
        
        # High priority biological terms
        high_priority_terms = [
            'protein', 'gene', 'enzyme', 'pathway', 'metabolism', 'biosynthesis',
            'annotation', 'function', 'domain', 'pfam', 'kegg', 'go:', 'ec:',
            'prophage', 'phage', 'viral', 'integrase', 'operon', 'cluster',
            'transport', 'binding', 'kinase', 'synthase', 'oxidase', 'reductase'
        ]
        
        # Medium priority terms
        medium_priority_terms = [
            'sequence', 'blast', 'similarity', 'identity', 'coverage',
            'structure', 'motif', 'region', 'site', 'residue', 'amino acid'
        ]
        
        # Score based on biological term density
        for term in high_priority_terms:
            score += section_lower.count(term) * 2.0
        
        for term in medium_priority_terms:
            score += section_lower.count(term) * 1.0
        
        # Bonus for question relevance if question provided
        if question:
            question_lower = question.lower()
            question_words = set(question_lower.split())
            section_words = set(section_lower.split())
            overlap = len(question_words.intersection(section_words))
            score += overlap * 1.5
        
        # Bonus for structured data (likely annotations)
        if any(indicator in section for indicator in [':', '=>', '|', '\t']):
            score += 1.0
        
        # Penalty for very short sections (likely noise)
        if len(section) < 50:
            score *= 0.5
        
        return score
    
    def _call_synthesis_model(self, context: str, question: str, task_name: str, focus: str, synthesis_type: str = "summarization") -> str:
        """
        Call synthesis model using model allocation system with caching.
        
        Args:
            context: Formatted context for synthesis
            question: Original user question
            task_name: Task name for model allocation
            focus: Focus areas for synthesis
            synthesis_type: Type of synthesis ("map_extraction", "reduce_selection", "summarization")
            
        Returns:
            Synthesis result
        """
        # Create cache key based on context hash and parameters
        import hashlib
        cache_key = hashlib.md5(f"{context[:1000]}{question}{task_name}{focus}".encode()).hexdigest()
        
        # Check cache first
        if cache_key in self.synthesis_cache:
            self.cache_hits += 1
            logger.info(f"📋 Cache hit for synthesis (key: {cache_key[:8]}...)")
            return self.synthesis_cache[cache_key]
        
        self.cache_misses += 1
        logger.info(f"🔄 Cache miss - making API call (key: {cache_key[:8]}...)")
        
        try:
            # CRITICAL: Enforce context limits before API call
            safe_context = self._enforce_context_limits(context, task_name, question)
            
            # Select appropriate signature based on synthesis type
            if synthesis_type == "map_extraction":
                from ..dspy_signatures import GenomicDataExtractor
                signature_class = GenomicDataExtractor
                
                def synthesize_call(module):
                    result = module(
                        genomic_data=safe_context,
                        focus_areas=focus
                    )
                    # Return combined output for compatibility
                    return f"KEY LOCI:\n{result.key_loci}\n\nBIOLOGICAL CONTEXT:\n{result.biological_context}\n\nQUANTITATIVE METRICS:\n{result.quantitative_metrics}"
                    
            elif synthesis_type == "reduce_selection":
                from ..dspy_signatures import GenomicSelector
                signature_class = GenomicSelector
                
                def synthesize_call(module):
                    result = module(
                        question=question,
                        chunk_extractions=safe_context
                    )
                    # Return combined output with data validation fields
                    return f"FINAL REPORT:\n{result.final_report}\n\nSELECTION REASONING:\n{result.selection_reasoning}\n\nBIOLOGICAL SIGNIFICANCE:\n{result.biological_significance}\n\nDATA SOURCES:\n{result.data_sources}\n\nUNSUPPORTED CLAIMS:\n{result.unsupported_claims}"
                    
            else:  # Default to summarization
                from ..dspy_signatures import GenomicSummarizer
                signature_class = GenomicSummarizer
                
                def synthesize_call(module):
                    result = module(
                        genomic_data=safe_context,
                        target_length="detailed",
                        focus_areas=focus
                    )
                    # Return combined output for compatibility
                    return f"SUMMARY:\n{result.summary}\n\nKEY FINDINGS:\n{result.key_findings}\n\nDATA STATISTICS:\n{result.data_statistics}"
            
            # ENHANCED retry logic with better rate limit detection
            max_retries = 5  # More retries for rate limits
            retry_delay = 5  # Start with longer delay (5 seconds)
            
            for attempt in range(max_retries):
                try:
                    result = self.model_allocator.create_context_managed_call(
                        task_name=task_name,
                        signature_class=signature_class,
                        module_call_func=synthesize_call,
                        query=question,
                        task_context=f"Progressive synthesis: {focus}"
                    )
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit = any(indicator in error_str for indicator in [
                        "429", "rate limit", "too many requests", "quota", "tokens per minute"
                    ])
                    
                    if is_rate_limit and attempt < max_retries - 1:
                        logger.warning(f"⏳ Rate limited detected, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                        logger.warning(f"🔍 Error details: {error_str[:200]}...")
                        import time
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 1.5, 30)  # Cap at 30 seconds
                        continue
                    else:
                        raise  # Not rate limited or max retries reached
            
            # Handle different result formats based on synthesis type
            if isinstance(result, str):
                # synthesize_call function returned formatted string directly
                synthesis_result = result
            elif result and hasattr(result, 'summary'):
                # Old GenomicSummarizer format
                synthesis_result = result.summary
            else:
                logger.warning(f"Model allocation returned unexpected result format: {type(result)}")
                fallback_result = f"Synthesis completed but result format unexpected. Context: {context[:500]}..."
                self.synthesis_cache[cache_key] = fallback_result
                return fallback_result
            
            # If a task graph was extracted earlier, prepend it to ensure visibility in final output
            try:
                tg = getattr(self, '_last_task_graph', '') if hasattr(self, '_last_task_graph') else ''
                if tg:
                    synthesis_result = f"TASK GRAPH:\n{tg}\n\n" + synthesis_result
            except Exception:
                pass
            # Cache and return the result
            self.synthesis_cache[cache_key] = synthesis_result
            return synthesis_result
                
        except Exception as e:
            logger.error(f"Synthesis model call failed: {e}")
            error_result = f"Synthesis failed: {str(e)}"
            # Don't cache errors
            return error_result
    
    def _choose_optimal_synthesis_strategy(self, unified_data: List[Dict[str, Any]], question: str) -> str:
        """
        Choose the optimal synthesis strategy based on data quality and question type.
        
        Args:
            unified_data: Unified data for synthesis
            question: User's question
            
        Returns:
            Synthesis strategy: "key_findings_only" or "full_context"
        """
        logger.info("🧠 Analyzing synthesis strategy requirements")
        
        # Assess key findings completeness and quality
        key_findings_quality = self._assess_key_findings_completeness(unified_data)
        
        # Check question complexity requirements
        requires_detailed_context = self._question_requires_detailed_context(question)
        
        # FORCE MAP-REDUCE: Always use full_context to ensure anti-hallucination constraints
        # and detailed Map-Reduce analysis instead of lightweight key-findings-only mode
        logger.info(f"🎯 Forcing Map-Reduce synthesis (key_findings_quality: {key_findings_quality:.2f}, detailed_context: {requires_detailed_context})")
        return "full_context"
    
    def _assess_key_findings_completeness(self, unified_data: List[Dict[str, Any]]) -> float:
        """
        Assess how complete and useful the key findings are.
        
        Args:
            unified_data: Unified data to assess
            
        Returns:
            Quality score 0.0-1.0 (higher = more complete findings)
        """
        total_findings = 0
        quality_indicators = 0
        
        for item in unified_data:
            if item.get('type') == 'task_note' and 'key_findings' in item:
                findings = item['key_findings']
                if findings:
                    total_findings += len(findings)
                    
                    # Check for quality indicators
                    for finding in findings:
                        finding_lower = str(finding).lower()
                        
                        # High-quality indicators
                        if any(term in finding_lower for term in [
                            'identified', 'detected', 'found', 'analyzed', 'discovered',
                            'protein', 'gene', 'loci', 'coordinates', 'domain', 
                            'prophage', 'operon', 'pathway', 'annotation'
                        ]):
                            quality_indicators += 1
                        
                        # Quantitative indicators (numbers suggest real analysis)
                        if any(char.isdigit() for char in finding):
                            quality_indicators += 0.5
        
        # Calculate quality score
        if total_findings == 0:
            return 0.0
        
        quality_ratio = quality_indicators / total_findings
        
        # Adjust based on total findings count (more findings = higher confidence)
        findings_bonus = min(total_findings / 20, 0.2)  # Max 20% bonus
        
        final_score = min(quality_ratio + findings_bonus, 1.0)
        logger.info(f"📊 Key findings assessment: {total_findings} findings, {quality_indicators} quality indicators, score: {final_score:.2f}")
        
        return final_score
    
    def _question_requires_detailed_context(self, question: str) -> bool:
        """
        Determine if question requires detailed context beyond key findings.
        
        Args:
            question: User's question
            
        Returns:
            True if detailed context needed, False if key findings sufficient
        """
        question_lower = question.lower()
        
        # Detailed context required for these patterns
        detailed_patterns = [
            'detailed report', 'comprehensive', 'full analysis', 'all details',
            'coordinates', 'exact', 'precise', 'specific location', 'sequence',
            'show me everything', 'maximum detail', 'complete analysis'
        ]
        
        # Simple patterns that can use key findings only
        simple_patterns = [
            'quick', 'summary', 'overview', 'brief', 'what are', 'how many',
            'list', 'find', 'identify', 'detect', 'simple'
        ]
        
        requires_detailed = any(pattern in question_lower for pattern in detailed_patterns)
        is_simple = any(pattern in question_lower for pattern in simple_patterns)
        
        # Default to detailed if unclear, but simple patterns override
        return requires_detailed or (not is_simple and len(question.split()) > 15)
    
    def _synthesize_from_key_findings_only(self, unified_data: List[Dict[str, Any]], question: str) -> str:
        """
        Lightweight synthesis using only key findings and discoveries.
        
        Args:
            unified_data: Unified data (only key findings will be used)
            question: User's question
            
        Returns:
            Synthesis result from key findings only
        """
        logger.info("🎯 Performing key-findings-only synthesis (lightweight mode)")
        
        # Extract all key findings
        all_findings = []
        task_summaries = []
        
        for item in unified_data:
            if item.get('type') == 'task_note':
                task_id = item.get('task_id', 'unknown')
                description = item.get('description', '')
                findings = item.get('key_findings', [])
                
                if findings:
                    task_summaries.append(f"Task {task_id}: {description[:100]}...")
                    all_findings.extend(findings)
        
        # Build lightweight context
        lightweight_context = f"""
BIOLOGICAL DISCOVERIES FROM ANALYSIS:

Task Overview:
{chr(10).join(task_summaries)}

Key Findings and Discoveries:
{chr(10).join(f"• {finding}" for finding in all_findings)}

ANALYSIS COMPLETE - {len(all_findings)} key discoveries identified
"""
        
        # Use direct synthesis with lightweight context
        from ..dspy_signatures import GenomicSynthesizer
        
        def synthesis_call(module):
            return module(
                question=question,
                context=lightweight_context,
                synthesis_mode="discovery_summary"
            )
        
        result = self.model_allocator.create_context_managed_call(
            task_name="biological_interpretation",
            signature_class=GenomicSynthesizer,
            module_call_func=synthesis_call,
            query=question,
            task_context="Key findings synthesis"
        )
        
        if result and hasattr(result, 'summary'):
            logger.info(f"✅ Key-findings-only synthesis complete ({len(lightweight_context)} characters)")
            return result.summary
        else:
            logger.warning("Key-findings synthesis failed, using fallback")
            return f"Analysis complete. Identified {len(all_findings)} key biological discoveries from the genomic analysis."

    def get_synthesis_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the synthesis process.
        
        Returns:
            Dictionary with synthesis statistics
        """
        total_calls = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_calls * 100) if total_calls > 0 else 0
        
        return {
            "architecture": "Map-Reduce",
            "direct_synthesis_limit": self.direct_synthesis_limit,
            "map_chunk_limit": self.map_chunk_limit,
            "tokenizer_available": self.tokenizer is not None,
            "model_allocator_available": self.model_allocator is not None,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate_percent": cache_hit_rate,
            "api_call_reduction": f"{cache_hit_rate:.1f}% fewer API calls"
        }
