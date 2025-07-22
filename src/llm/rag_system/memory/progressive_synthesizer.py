"""
Progressive synthesis system for handling large multi-task agentic workflows.

Uses a Map-Reduce architecture to process task notes and raw data efficiently,
with token-based decision making for optimal model utilization.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import tiktoken
import concurrent.futures

from .note_keeper import NoteKeeper
from .note_schemas import TaskNote, SynthesisNote, ConfidenceLevel
from .memory_utils import generate_session_id
from .model_allocation import get_model_allocator

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
        self.direct_synthesis_limit = 20000  # Will be updated based on actual model limits
        self.map_chunk_limit = 15000  # Will be updated based on actual model limits
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception as e:
            logger.warning(f"Failed to initialize tokenizer: {e}")
            self.tokenizer = None
        
        # Initialize model allocator for intelligent model selection
        self.model_allocator = get_model_allocator()
        
        # Caching system to reduce API calls
        self.synthesis_cache = {}  # Cache for synthesis results
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Update chunk limits based on actual model capabilities
        self._update_model_aware_limits()
        
        logger.info("🏗️ ProgressiveSynthesizer initialized with Map-Reduce architecture and caching")
    
    def _update_model_aware_limits(self):
        """Update chunk limits based on actual model capabilities."""
        try:
            # Get limits for the models we'll be using
            _, final_synthesis_model = self.model_allocator.get_model_for_task("final_synthesis", "")
            _, map_step_model = self.model_allocator.get_model_for_task("genomic_summarization", "")
            
            # Set direct synthesis limit to 30% of final synthesis model capacity 
            self.direct_synthesis_limit = int(final_synthesis_model.max_context * 0.3)
            
            # Set map chunk limit to 40% of map step model capacity
            self.map_chunk_limit = int(map_step_model.max_context * 0.4)
            
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
            dspy_synthesizer: DEPRECATED - uses model allocation (kept for compatibility)
            raw_data: Raw data from task execution (prioritized over task_notes)
            rag_system: DEPRECATED - not used in Map-Reduce architecture
            
        Returns:
            Final comprehensive synthesis or brief guidance summary
        """
        # Warn about deprecated parameters
        if dspy_synthesizer is not None:
            logger.warning("⚠️ dspy_synthesizer parameter is deprecated and will be ignored")
        if rag_system is not None:
            logger.warning("⚠️ rag_system parameter is deprecated and will be ignored")
        
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
                focus="brief guidance for next steps (2-3 sentences max)"
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
        
        # Step 2: Token-based decision making
        total_tokens = self._count_data_tokens(unified_data)
        logger.info(f"📊 Total input tokens: {total_tokens}")
        
        # Step 3: Choose synthesis strategy based on token count
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
                unified_data.append({
                    'type': 'task_note',
                    'task_id': note.task_id,
                    'description': note.description,
                    'observations': note.observations,
                    'key_findings': note.key_findings,
                    'confidence_level': note.confidence_level.value,
                    'quantitative_data': note.quantitative_data,
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
        
        return unified_data
    
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
        
        # Format data for synthesis
        formatted_data = self._format_data_for_synthesis(data)
        
        return self._call_synthesis_model(
            context=formatted_data,
            question=question,
            task_name="final_synthesis",  # Use o3 for complex biological reasoning
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
        
        # Step 2: Main chunking logic (now guaranteed that no item exceeds chunk limit)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for item in processed_data:
            item_tokens = self._count_data_tokens([item])
            
            # Check if adding this item would exceed chunk limit
            if current_tokens + item_tokens > self.map_chunk_limit and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(item)
            current_tokens += item_tokens
        
        # Add final chunk if not empty
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
            
            # Use cheaper model for Map step (summarization task)
            summary = self._call_synthesis_model(
                context=formatted_chunk,
                question=question,
                task_name="genomic_summarization",  # Use cheaper model for chunk summarization
                focus=f"key insights and biological patterns from chunk {chunk_id}"
            )
            
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
        synthesis_context = f"""
QUESTION: {question}

CHUNK SUMMARIES ({len(chunk_summaries)} chunks):
{combined_context}

SYNTHESIS TASK: Integrate the above chunk summaries into a comprehensive, coherent analysis that addresses the original question.
"""
        
        # Use high-capability model for final synthesis
        return self._call_synthesis_model(
            context=synthesis_context,
            question=question,
            task_name="final_synthesis",  # Use o3 for complex integration
            focus="comprehensive integration of chunk summaries with biological insights"
        )
    
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
            elif item.get('type') == 'task_metadata':
                formatted_item = f"Task Metadata ({item['task_count']} tasks):\n"
                formatted_item += f"Key Insights: {'; '.join(item['key_insights'][:5])}\n"
                if item.get('cross_connections'):
                    connections = [f"{conn['from_task']} → {conn['to_task']}" 
                                 for conn in item['cross_connections'][:3]]
                    formatted_item += f"Cross-connections: {'; '.join(connections)}\n"
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
    
    def _call_synthesis_model(self, context: str, question: str, task_name: str, focus: str) -> str:
        """
        Call synthesis model using model allocation system with caching.
        
        Args:
            context: Formatted context for synthesis
            question: Original user question
            task_name: Task name for model allocation
            focus: Focus areas for synthesis
            
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
            
            from ..dspy_signatures import GenomicSummarizer
            
            def synthesize_call(module):
                return module(
                    genomic_data=safe_context,  # Use context-limit-enforced version
                    target_length="detailed",
                    focus_areas=focus
                )
            
            # ENHANCED retry logic with better rate limit detection
            max_retries = 5  # More retries for rate limits
            retry_delay = 5  # Start with longer delay (5 seconds)
            
            for attempt in range(max_retries):
                try:
                    result = self.model_allocator.create_context_managed_call(
                        task_name=task_name,
                        signature_class=GenomicSummarizer,
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
            
            if result and hasattr(result, 'summary'):
                synthesis_result = result.summary
                # Cache the result
                self.synthesis_cache[cache_key] = synthesis_result
                return synthesis_result
            else:
                logger.warning("Model allocation returned unexpected result format")
                fallback_result = f"Synthesis completed but result format unexpected. Context: {context[:500]}..."
                self.synthesis_cache[cache_key] = fallback_result
                return fallback_result
                
        except Exception as e:
            logger.error(f"Synthesis model call failed: {e}")
            error_result = f"Synthesis failed: {str(e)}"
            # Don't cache errors
            return error_result
    
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