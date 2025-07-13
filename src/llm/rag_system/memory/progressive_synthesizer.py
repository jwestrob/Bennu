"""
Progressive synthesis system for handling large multi-task agentic workflows.

Processes task notes in chunks to maintain memory persistence across token limits
and generates comprehensive final synthesis from accumulated insights.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import tiktoken

from .note_keeper import NoteKeeper
from .note_schemas import TaskNote, SynthesisNote, ConfidenceLevel
from .memory_utils import generate_session_id
from .model_allocation import get_model_allocator

logger = logging.getLogger(__name__)


class ProgressiveSynthesizer:
    """
    Handles progressive synthesis of task notes into comprehensive final answers.
    
    Processes notes in chunks to avoid token limits while maintaining cross-task
    insights and building comprehensive understanding across complex workflows.
    """
    
    def __init__(self, note_keeper: NoteKeeper, chunk_size: int = 8, target_tokens: int = 15000):
        """
        Initialize progressive synthesizer.
        
        Args:
            note_keeper: NoteKeeper instance for accessing notes
            chunk_size: Number of tasks to process per chunk
            target_tokens: Target token count for each synthesis chunk
        """
        self.note_keeper = note_keeper
        self.chunk_size = chunk_size
        self.target_tokens = target_tokens
        self.synthesis_chunks = []
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception as e:
            logger.warning(f"Failed to initialize tokenizer: {e}")
            self.tokenizer = None
        
        # Initialize model allocator for intelligent model selection
        self.model_allocator = get_model_allocator()
    
    def synthesize_progressive(self, 
                             task_notes: List[TaskNote],
                             dspy_synthesizer,
                             question: str,
                             raw_data: List[Dict[str, Any]] = None,
                             rag_system = None) -> str:
        """
        Perform progressive synthesis prioritizing raw task results over compressed notes.
        
        Args:
            task_notes: List of TaskNote objects (used as metadata)
            dspy_synthesizer: DSPy synthesizer module
            question: Original user question
            raw_data: Raw data from task execution (PRIMARY DATA SOURCE)
            rag_system: Optional RAG system for task-based processing
            
        Returns:
            Final comprehensive synthesis
        """
        logger.info(f"🔄 REDESIGNED SYNTHESIS: Prioritizing raw data over compressed notes")
        logger.info(f"📊 Input: {len(task_notes)} task notes, {len(raw_data) if raw_data else 0} raw data items")
        
        if not task_notes and not raw_data:
            return "No data available for synthesis."
        
        # NEW APPROACH: Prioritize raw data from task execution
        if raw_data and len(raw_data) > 0:
            logger.info(f"✅ Using RAW DATA as primary source ({len(raw_data)} items)")
            return self._synthesize_from_raw_data(raw_data, task_notes, dspy_synthesizer, question)
        
        # Fallback: If no raw data, use notes-based synthesis
        logger.warning("⚠️ No raw data available, falling back to notes-only synthesis")
        return self._synthesize_standard(task_notes, dspy_synthesizer, question)
    
    def _should_use_multipart_report(self, raw_data: List[Dict[str, Any]], question: str) -> bool:
        """
        Determine if multi-part report synthesis should be used.
        
        Args:
            raw_data: Raw data for analysis
            question: User question
            
        Returns:
            True if multi-part report should be used
        """
        # Use multi-part for medium datasets (but task-based for very large)
        if 50 < len(raw_data) <= 1000:
            return True
        
        # Use multi-part for specific report types
        multipart_keywords = [
            'crispr', 'comprehensive', 'all genomes', 'complete analysis',
            'detailed report', 'full analysis', 'compare across genomes'
        ]
        
        question_lower = question.lower()
        if any(keyword in question_lower for keyword in multipart_keywords) and len(raw_data) <= 1000:
            return True
        
        return False
    
    def _synthesize_multipart_report(self, 
                                   task_notes: List[TaskNote],
                                   dspy_synthesizer,
                                   question: str,
                                   raw_data: List[Dict[str, Any]]) -> str:
        """
        Synthesize using multi-part report generation.
        
        Args:
            task_notes: List of TaskNote objects
            dspy_synthesizer: DSPy synthesizer module
            question: Original user question
            raw_data: Raw data for multi-part report
            
        Returns:
            Multi-part report synthesis
        """
        try:
            # Import here to avoid circular imports
            from .multipart_synthesizer import MultiPartReportSynthesizer
            
            # Initialize multi-part synthesizer
            multipart_synthesizer = MultiPartReportSynthesizer(
                note_keeper=self.note_keeper,
                chunk_size=self.chunk_size,
                max_part_tokens=18000
            )
            
            # Initialize DSPy modules (now uses global config)
            multipart_synthesizer.initialize_dspy_modules(None)  # Pass None to avoid serialization issues
            
            # Generate multi-part report
            return multipart_synthesizer.synthesize_multipart_report(
                task_notes=task_notes,
                question=question,
                data=raw_data
            )
            
        except Exception as e:
            logger.error(f"Multi-part report synthesis failed: {e}")
            logger.info("Falling back to standard progressive synthesis")
            return self._synthesize_standard(task_notes, dspy_synthesizer, question)
    
    def _synthesize_standard(self, 
                           task_notes: List[TaskNote],
                           dspy_synthesizer,
                           question: str) -> str:
        """
        Perform standard progressive synthesis.
        
        Args:
            task_notes: List of TaskNote objects to synthesize
            dspy_synthesizer: DSPy synthesizer module
            question: Original user question
            
        Returns:
            Final comprehensive synthesis
        """
        # Process notes in chunks
        chunks = self._create_note_chunks(task_notes)
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i+1:03d}"
            logger.info(f"Processing synthesis chunk {chunk_id} with {len(chunk)} tasks")
            
            # Synthesize this chunk
            synthesis_result = self._synthesize_chunk(chunk, dspy_synthesizer, chunk_id)
            
            if synthesis_result:
                self.synthesis_chunks.append(synthesis_result)
                
                # Record synthesis notes
                self.note_keeper.record_synthesis_notes(
                    chunk_id=chunk_id,
                    source_tasks=[note.task_id for note in chunk],
                    chunk_theme=synthesis_result.get("theme", ""),
                    integrated_findings=synthesis_result.get("integrated_findings", []),
                    cross_task_synthesis=synthesis_result.get("cross_task_synthesis", []),
                    emergent_insights=synthesis_result.get("emergent_insights", []),
                    confidence=ConfidenceLevel(synthesis_result.get("confidence", "medium")),
                    tokens_used=synthesis_result.get("tokens_used", 0)
                )
        
        # Generate final synthesis
        final_synthesis = self._generate_final_synthesis(dspy_synthesizer, question)
        
        logger.info(f"Completed progressive synthesis with {len(self.synthesis_chunks)} chunks")
        return final_synthesis
    
    def _synthesize_with_task_system(self, 
                                   task_notes: List[TaskNote],
                                   dspy_synthesizer,
                                   question: str,
                                   raw_data: List[Dict[str, Any]],
                                   rag_system) -> str:
        """
        Use task-based synthesis for very large datasets.
        
        Args:
            task_notes: List of TaskNote objects
            dspy_synthesizer: DSPy synthesizer module
            question: Original user question
            raw_data: Complete raw data (no size limits)
            rag_system: RAG system for task execution
            
        Returns:
            Comprehensive synthesis using task management
        """
        try:
            from .task_based_synthesizer import TaskBasedSynthesizer
            
            # Initialize task-based synthesizer with large but manageable chunks
            task_synthesizer = TaskBasedSynthesizer(
                note_keeper=self.note_keeper,
                chunk_size=self.chunk_size,
                max_items_per_task=2000  # Large chunks for detailed analysis without bypassing multi-part structure
            )
            
            # Use task-based synthesis for unlimited dataset processing
            return task_synthesizer.synthesize_unlimited_dataset(
                task_notes=task_notes,
                dspy_synthesizer=dspy_synthesizer,
                question=question,
                raw_data=raw_data,
                rag_system=rag_system
            )
            
        except Exception as e:
            logger.error(f"Task-based synthesis failed: {e}")
            logger.info("Falling back to multi-part report synthesis")
            return self._synthesize_multipart_report(task_notes, dspy_synthesizer, question, raw_data)
    
    def _create_note_chunks(self, task_notes: List[TaskNote]) -> List[List[TaskNote]]:
        """
        Create chunks of task notes for processing.
        
        Args:
            task_notes: List of TaskNote objects
            
        Returns:
            List of note chunks
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for note in task_notes:
            # Estimate token count for this note
            note_tokens = self._estimate_note_tokens(note)
            
            # Check if adding this note would exceed chunk limits
            if (len(current_chunk) >= self.chunk_size or 
                (current_tokens + note_tokens > self.target_tokens and current_chunk)):
                
                # Start new chunk
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0
            
            current_chunk.append(note)
            current_tokens += note_tokens
        
        # Add final chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _estimate_note_tokens(self, note: TaskNote) -> int:
        """
        Estimate token count for a task note.
        
        Args:
            note: TaskNote to estimate
            
        Returns:
            Estimated token count
        """
        if not self.tokenizer:
            # Fallback estimation: ~4 characters per token
            text = f"{note.description} {' '.join(note.observations)} {' '.join(note.key_findings)}"
            return len(text) // 4
        
        try:
            # More accurate token counting
            text_content = self._format_note_for_counting(note)
            return len(self.tokenizer.encode(text_content))
        except Exception as e:
            logger.warning(f"Token counting failed for note {note.task_id}: {e}")
            return 500  # Conservative fallback
    
    def _format_note_for_counting(self, note: TaskNote) -> str:
        """Format note content for token counting."""
        parts = [
            f"Task: {note.description}",
            f"Observations: {'; '.join(note.observations)}",
            f"Key Findings: {'; '.join(note.key_findings)}",
            f"Confidence: {note.confidence_level.value}"
        ]
        
        if note.quantitative_data:
            parts.append(f"Data: {str(note.quantitative_data)}")
        
        if note.cross_task_connections:
            connections = [f"{conn.connected_task}:{conn.connection_type.value}" 
                          for conn in note.cross_task_connections]
            parts.append(f"Connections: {'; '.join(connections)}")
        
        return " | ".join(parts)
    
    def _synthesize_chunk(self, 
                         chunk: List[TaskNote],
                         dspy_synthesizer,
                         chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Synthesize a chunk of task notes.
        
        Args:
            chunk: List of TaskNote objects to synthesize
            dspy_synthesizer: DSPy synthesizer module
            chunk_id: Identifier for this chunk
            
        Returns:
            Synthesis result dictionary
        """
        try:
            # Prepare chunk context
            chunk_context = self._format_chunk_context(chunk)
            
            # Generate synthesis theme
            theme = self._generate_chunk_theme(chunk)
            
            # Use model allocation system with context manager approach
            logger.info(f"🔥 Using model allocation for chunk synthesis: {chunk_id}")
            
            from ..dspy_signatures import GenomicSummarizer
            
            def synthesize_call(module):
                return module(
                    genomic_data=chunk_context,
                    target_length="medium",
                    focus_areas="cross-task connections, biological patterns, quantitative insights"
                )
            
            synthesis_result = self.model_allocator.create_context_managed_call(
                task_name="biological_interpretation",
                signature_class=GenomicSummarizer,
                module_call_func=synthesize_call
            )
            
            if synthesis_result is None:
                # Fallback to provided dspy_synthesizer if allocation fails
                logger.warning("Model allocation failed, falling back to default synthesizer")
                synthesis_result = dspy_synthesizer(
                    genomic_data=chunk_context,
                    target_length="medium",
                    focus_areas="cross-task connections, biological patterns, quantitative insights"
                )
            
            # Extract and structure results
            integrated_findings = self._extract_findings(synthesis_result.summary)
            cross_task_synthesis = self._extract_cross_task_insights(chunk)
            emergent_insights = self._extract_emergent_insights(synthesis_result.key_findings)
            
            # Count tokens used
            tokens_used = self._estimate_synthesis_tokens(synthesis_result)
            
            return {
                "chunk_id": chunk_id,
                "theme": theme,
                "integrated_findings": integrated_findings,
                "cross_task_synthesis": cross_task_synthesis,
                "emergent_insights": emergent_insights,
                "confidence": getattr(synthesis_result, 'confidence', 'medium'),
                "tokens_used": tokens_used,
                "raw_synthesis": synthesis_result.summary
            }
            
        except Exception as e:
            logger.error(f"Failed to synthesize chunk {chunk_id}: {e}")
            return None
    
    def _format_chunk_context(self, chunk: List[TaskNote]) -> str:
        """Format chunk notes for DSPy synthesis."""
        context_parts = []
        
        for note in chunk:
            note_context = [
                f"Task: {note.description}",
                f"Observations: {'; '.join(note.observations)}",
                f"Key Findings: {'; '.join(note.key_findings)}"
            ]
            
            if note.quantitative_data:
                note_context.append(f"Data: {str(note.quantitative_data)}")
            
            if note.cross_task_connections:
                connections = [f"{conn.connected_task} ({conn.connection_type.value}: {conn.description})"
                              for conn in note.cross_task_connections]
                note_context.append(f"Connections: {'; '.join(connections)}")
            
            context_parts.append(" | ".join(note_context))
        
        return "\n\n".join(context_parts)
    
    def _generate_chunk_theme(self, chunk: List[TaskNote]) -> str:
        """Generate a theme for the chunk based on task content."""
        # Extract common themes from task descriptions
        descriptions = [note.description for note in chunk]
        
        # Simple theme generation based on common terms
        common_terms = {}
        for desc in descriptions:
            words = desc.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    common_terms[word] = common_terms.get(word, 0) + 1
        
        # Get most common meaningful terms
        if common_terms:
            top_terms = sorted(common_terms.items(), key=lambda x: x[1], reverse=True)[:3]
            theme_words = [term[0] for term in top_terms]
            return f"Analysis of {', '.join(theme_words)}"
        
        return f"Task Analysis (Tasks {chunk[0].task_id}-{chunk[-1].task_id})"
    
    def _extract_findings(self, synthesis_text: str) -> List[str]:
        """Extract key findings from synthesis text."""
        # Simple extraction - split by sentences and filter for key insights
        sentences = synthesis_text.split('. ')
        findings = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 50 and 
                any(keyword in sentence.lower() for keyword in 
                    ['significant', 'important', 'reveals', 'indicates', 'shows', 'demonstrates'])):
                findings.append(sentence)
        
        return findings[:5]  # Top 5 findings
    
    def _extract_cross_task_insights(self, chunk: List[TaskNote]) -> List[Dict[str, Any]]:
        """Extract cross-task insights from chunk."""
        insights = []
        
        for note in chunk:
            for connection in note.cross_task_connections:
                insight = {
                    "connection": f"{note.task_id} {connection.connection_type.value} {connection.connected_task}",
                    "insight": connection.description,
                    "confidence": connection.confidence.value
                }
                insights.append(insight)
        
        return insights
    
    def _extract_emergent_insights(self, key_findings: str) -> List[str]:
        """Extract emergent insights from key findings."""
        # Simple extraction of insights that suggest emergent patterns
        sentences = key_findings.split('. ')
        emergent = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 40 and 
                any(keyword in sentence.lower() for keyword in 
                    ['pattern', 'trend', 'correlation', 'relationship', 'connection', 'emerges'])):
                emergent.append(sentence)
        
        return emergent[:3]  # Top 3 emergent insights
    
    def _estimate_synthesis_tokens(self, synthesis_result) -> int:
        """Estimate tokens used in synthesis."""
        text = f"{synthesis_result.summary} {synthesis_result.key_findings}"
        
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        return len(text) // 4  # Fallback estimation
    
    def _synthesize_from_raw_data(self, 
                                raw_data: List[Dict[str, Any]], 
                                task_notes: List[TaskNote],
                                dspy_synthesizer,
                                question: str) -> str:
        """
        NEW PRIMARY SYNTHESIS METHOD: Process raw task execution data directly.
        
        Args:
            raw_data: Raw data from task execution (primary source)
            task_notes: Task notes for context and cross-task insights
            dspy_synthesizer: DSPy synthesizer module
            question: Original user question
            
        Returns:
            Comprehensive synthesis based on raw data
        """
        logger.info(f"🎯 SYNTHESIZING FROM RAW DATA: {len(raw_data)} items")
        
        # Organize raw data by type and significance
        organized_data = self._organize_raw_data_by_significance(raw_data)
        
        # Extract high-level insights from task notes for context
        cross_task_context = self._extract_cross_task_context(task_notes)
        
        # Determine synthesis strategy based on data size and complexity
        if len(raw_data) > 1000:
            logger.info("📚 Large dataset detected - using chunked synthesis approach")
            return self._synthesize_large_raw_dataset(organized_data, cross_task_context, dspy_synthesizer, question)
        elif len(raw_data) > 100:
            logger.info("📊 Medium dataset detected - using structured synthesis approach") 
            return self._synthesize_medium_raw_dataset(organized_data, cross_task_context, dspy_synthesizer, question)
        else:
            logger.info("📝 Small dataset detected - using detailed synthesis approach")
            return self._synthesize_small_raw_dataset(organized_data, cross_task_context, dspy_synthesizer, question)
    
    def _organize_raw_data_by_significance(self, raw_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Organize raw data by biological significance and data type.
        
        Args:
            raw_data: Raw execution results
            
        Returns:
            Organized data structure
        """
        # Don't pre-categorize data - let the LLM see everything and decide what's important
        # This preserves the agentic architecture principle of model-driven analysis
        organized = {
            'all_data': raw_data,  # Everything goes to the LLM for analysis
            'metadata': {}
        }
        
        logger.info(f"📋 all_data: {len(raw_data)} items (no pre-filtering)")
        
        return organized
    
    def _is_high_significance_item(self, item: Dict[str, Any]) -> bool:
        """Check if item represents high biological significance."""
        if not isinstance(item, dict):
            return False
        
        # Check for novelty indicators
        novelty_keywords = ['novel', 'uncharacterized', 'hypothetical', 'unknown', 'rare']
        
        # Check descriptions for novelty
        desc = item.get('ko_description', '') or item.get('description', '')
        if desc and any(keyword in desc.lower() for keyword in novelty_keywords):
            return True
        
        # Check for BGC-related data (biosynthetic gene clusters are often novel)
        if any(key in item for key in ['bgc', 'biosynthetic', 'cluster', 'secondary_metabolite']):
            return True
        
        # Check for CAZyme data (carbohydrate-active enzymes can be novel)
        if any(key in item for key in ['cazyme', 'carbohydrate', 'glycoside']):
            return True
        
        return False
    
    def _is_functional_annotation(self, item: Dict[str, Any]) -> bool:
        """Check if item is functional annotation data."""
        annotation_keys = ['ko_id', 'ko_description', 'pfam', 'protein_id', 'function']
        return isinstance(item, dict) and any(key in item for key in annotation_keys)
    
    def _is_comparative_data(self, item: Dict[str, Any]) -> bool:
        """Check if item is comparative analysis data."""
        comparative_keys = ['genome_id', 'comparison', 'across_genomes', 'distribution']
        return isinstance(item, dict) and any(key in item for key in comparative_keys)
    
    def _is_tool_result(self, item: Dict[str, Any]) -> bool:
        """Check if item is from external tool execution."""
        return isinstance(item, dict) and 'tool_result' in str(item)
    
    def _extract_cross_task_context(self, task_notes: List[TaskNote]) -> Dict[str, Any]:
        """Extract cross-task insights from notes for context."""
        context = {
            'key_insights': [],
            'cross_connections': [],
            'execution_summary': {}
        }
        
        for note in task_notes:
            # Extract key findings
            context['key_insights'].extend(note.key_findings)
            
            # Extract cross-task connections
            for connection in note.cross_task_connections:
                context['cross_connections'].append({
                    'from_task': note.task_id,
                    'to_task': connection.connected_task,
                    'relationship': connection.connection_type.value,
                    'description': connection.description
                })
        
        return context
    
    def _synthesize_large_raw_dataset(self, 
                                    organized_data: Dict[str, Any],
                                    cross_task_context: Dict[str, Any], 
                                    dspy_synthesizer,
                                    question: str) -> str:
        """Synthesize large datasets (>1000 items) using chunked approach."""
        logger.info("🏗️ Large dataset synthesis: Chunking by significance")
        
        synthesis_parts = []
        
        # Let the LLM analyze ALL data without any pre-filtering or categorization
        if organized_data['all_data']:
            all_data_synthesis = self._synthesize_significance_chunk(
                organized_data['all_data'], 
                f"Complete Dataset Analysis ({len(organized_data['all_data'])} items)",
                dspy_synthesizer, 
                question
            )
            synthesis_parts.append(all_data_synthesis)
        
        # Add cross-task insights
        if cross_task_context['key_insights']:
            context_summary = f"Cross-task insights: {'; '.join(cross_task_context['key_insights'][:5])}"
            synthesis_parts.append(context_summary)
        
        return "\n\n".join(synthesis_parts)
    
    def _synthesize_medium_raw_dataset(self,
                                     organized_data: Dict[str, Any],
                                     cross_task_context: Dict[str, Any],
                                     dspy_synthesizer, 
                                     question: str) -> str:
        """Synthesize medium datasets (100-1000 items) with structured approach."""
        logger.info("📊 Medium dataset synthesis: Structured analysis")
        
        # Combine all data for comprehensive analysis
        all_data = []
        for category, items in organized_data.items():
            if isinstance(items, list):
                all_data.extend(items)
        
        # Format for DSPy synthesis
        formatted_data = self._format_raw_data_for_synthesis(all_data[:300])  # Manageable chunk
        
        # Include cross-task context
        context_summary = self._format_cross_task_context(cross_task_context)
        
        full_context = f"Raw Data Analysis:\n{formatted_data}\n\nCross-Task Context:\n{context_summary}"
        
        # Use model allocation for synthesis
        return self._synthesize_with_model_allocation(full_context, dspy_synthesizer, question, "medium_dataset")
    
    def _synthesize_small_raw_dataset(self,
                                    organized_data: Dict[str, Any],
                                    cross_task_context: Dict[str, Any],
                                    dspy_synthesizer,
                                    question: str) -> str:
        """Synthesize small datasets (<100 items) with detailed analysis."""
        logger.info("📝 Small dataset synthesis: Detailed analysis")
        
        # Include all data for comprehensive analysis
        all_data = []
        for category, items in organized_data.items():
            if isinstance(items, list):
                all_data.extend(items)
        
        # Format all data for synthesis
        formatted_data = self._format_raw_data_for_synthesis(all_data)
        context_summary = self._format_cross_task_context(cross_task_context)
        
        full_context = f"Complete Data Analysis:\n{formatted_data}\n\nTask Execution Context:\n{context_summary}"
        
        # Use model allocation for detailed synthesis
        return self._synthesize_with_model_allocation(full_context, dspy_synthesizer, question, "detailed_analysis")
    
    def _synthesize_significance_chunk(self,
                                     data_chunk: List[Dict[str, Any]],
                                     chunk_name: str,
                                     dspy_synthesizer,
                                     question: str) -> str:
        """Synthesize a chunk of significant data."""
        # Let the LLM see all data - no filtering (agentic architecture principle)
        filtered_chunk = data_chunk
        formatted_chunk = self._format_raw_data_for_synthesis(filtered_chunk)
        
        context = f"{chunk_name}:\n{formatted_chunk}"
        
        return self._synthesize_with_model_allocation(context, dspy_synthesizer, question, "significance_analysis")
    
    def _synthesize_tool_results_chunk(self,
                                     tool_results: List[Dict[str, Any]],
                                     dspy_synthesizer,
                                     question: str) -> str:
        """Synthesize external tool execution results."""
        formatted_tools = []
        
        for result in tool_results:
            formatted_tools.append(f"Tool Result: {str(result)[:1000]}")  # Include substantial detail
        
        context = f"External Tool Analysis:\n" + "\n".join(formatted_tools)
        
        return self._synthesize_with_model_allocation(context, dspy_synthesizer, question, "tool_analysis")
    
    def _format_raw_data_for_synthesis(self, raw_data: List[Dict[str, Any]]) -> str:
        """Format raw data for DSPy synthesis."""
        if not raw_data:
            return "No data available"
        
        formatted_items = []
        for i, item in enumerate(raw_data):
            formatted_items.append(f"Item {i+1}: {str(item)}")
        
        return "\n".join(formatted_items)
    
    def _format_cross_task_context(self, context: Dict[str, Any]) -> str:
        """Format cross-task context for synthesis."""
        parts = []
        
        if context.get('key_insights'):
            parts.append(f"Key Insights: {'; '.join(context['key_insights'][:5])}")
        
        if context.get('cross_connections'):
            connections = [f"{conn['from_task']} → {conn['to_task']}: {conn['description']}" 
                         for conn in context['cross_connections'][:3]]
            parts.append(f"Cross-Task Connections: {'; '.join(connections)}")
        
        return "\n".join(parts) if parts else "No additional context available"
    
    def _intelligent_data_filter(self, data_chunk: List[Dict[str, Any]], max_items: int = 100, question: str = "") -> List[Dict[str, Any]]:
        """
        Intelligently filter data to focus on task-relevant findings while staying under token limits.
        
        Args:
            data_chunk: Raw data to filter
            max_items: Maximum items to include
            question: User question for context-aware filtering
            
        Returns:
            Filtered data prioritizing task-relevant findings
        """
        if len(data_chunk) <= max_items:
            return data_chunk
            
        # Determine and log filtering strategy
        filter_type = "generic novelty"
        if is_phage_query:
            filter_type = "phage relevance"
        elif is_transport_query:
            filter_type = "transport relevance"
        elif is_crispr_query:
            filter_type = "CRISPR relevance"
        elif is_metabolic_query:
            filter_type = "metabolic relevance"
            
        logger.info(f"🎯 Intelligent filtering ({filter_type}): {len(data_chunk)} items → {max_items} items")
        
        # Task-specific relevance scoring
        scored_items = []
        question_lower = question.lower() if question else ""
        
        # Determine query focus
        is_phage_query = any(term in question_lower for term in ['phage', 'prophage', 'virus', 'viral', 'operons'])
        is_transport_query = any(term in question_lower for term in ['transport', 'transporter', 'permease'])
        is_crispr_query = any(term in question_lower for term in ['crispr', 'cas'])
        is_metabolic_query = any(term in question_lower for term in ['metabolic', 'pathway', 'enzyme'])
        
        for item in data_chunk:
            score = 0
            item_str = str(item).lower()
            
            # TASK-SPECIFIC HIGH PRIORITY SCORING
            if is_phage_query:
                # Phage-specific keywords get massive boost
                if any(keyword in item_str for keyword in ['phage', 'prophage', 'viral', 'virus', 'integrase', 'terminase', 'capsid', 'tail', 'lysis', 'holin']):
                    score += 50
                # Hypothetical proteins in operon context
                if any(keyword in item_str for keyword in ['hypothetical', 'uncharacterized', 'unknown']) and 'gene' in item_str:
                    score += 25
                # Other proteins get minimal score
                else:
                    score += 1
                    
            elif is_transport_query:
                # Transport-specific keywords
                if any(keyword in item_str for keyword in ['transport', 'transporter', 'permease', 'channel', 'efflux', 'influx']):
                    score += 50
                elif any(keyword in item_str for keyword in ['membrane', 'abc', 'mfs']):
                    score += 25
                else:
                    score += 1
                    
            elif is_crispr_query:
                # CRISPR-specific keywords
                if any(keyword in item_str for keyword in ['crispr', 'cas', 'spacer', 'repeat']):
                    score += 50
                else:
                    score += 1
                    
            else:
                # Generic novelty scoring (fallback)
                if any(keyword in item_str for keyword in ['unknown', 'hypothetical', 'uncharacterized', 'duf']):
                    score += 10
                elif any(keyword in item_str for keyword in ['bgc', 'cluster', 'biosynthetic']):
                    score += 8
                elif any(keyword in item_str for keyword in ['transport', 'regulator', 'sensor', 'kinase']):
                    score += 5
                elif any(keyword in item_str for keyword in ['dehydrogenase', 'oxidase', 'reductase', 'synthase']):
                    score += 4
                elif any(keyword in item_str for keyword in ['ribosomal', 'translation', 'replication']):
                    score += 1
                
            # Boost for longer proteins (more likely to be interesting)
            if 'length' in item and isinstance(item.get('length'), (int, str)):
                try:
                    length = int(item['length'])
                    if length > 500:  # Large proteins often more interesting
                        score += 3
                except:
                    pass
                    
            scored_items.append((score, item))
        
        # Sort by score (descending) and take top items
        scored_items.sort(key=lambda x: x[0], reverse=True)
        filtered_items = [item for score, item in scored_items[:max_items]]
        
        # Log filtering results
        avg_score = sum(score for score, _ in scored_items[:max_items]) / max_items if max_items > 0 else 0
        logger.info(f"✅ Filtered to top {len(filtered_items)} items (avg novelty score: {avg_score:.1f})")
        
        return filtered_items
    
    def _compress_context_for_synthesis(self, context: str, max_tokens: int = 18000) -> str:
        """
        Intelligently compress context for synthesis while preserving biological insights.
        Uses AGGRESSIVE token-aware compression to stay well under o3 limits.
        
        Args:
            context: Original context string
            max_tokens: Maximum tokens allowed (conservative limit for o3)
            
        Returns:
            Compressed context focusing on novel findings
        """
        # Use actual tokenizer for accurate token counting
        def count_tokens(text: str) -> int:
            if self.tokenizer:
                try:
                    return len(self.tokenizer.encode(text))
                except:
                    pass
            # Fallback: more accurate estimate (3.5 chars per token for English)
            return int(len(text) / 3.5)
        
        logger.info(f"🗜️ AGGRESSIVE COMPRESSION: {count_tokens(context)} → target {max_tokens} tokens")
        
        # Step 1: Parse structured data for intelligent filtering
        lines = context.split('\n')
        
        # Priority categories for biological analysis (ENHANCED FOR COMPREHENSIVE NOVELTY)
        ultra_high_priority = [
            'unknown', 'hypothetical', 'uncharacterized', 'novel', 'duf', 'unusual',
            'bgc', 'cluster', 'biosynthetic', 'unique', 'rare', 'cryptic',
            'novelty', 'stands out', 'interesting', 'unusual', 'orphan',
            'domain of unknown function', 'no functional annotation', 'no annotation',
            'putative', 'predicted protein', 'hypothetical protein'
        ]
        
        high_priority_keywords = [
            'transport', 'regulator', 'sensor', 'kinase', 'dehydrogenase',
            'oxidase', 'reductase', 'synthase', 'transferase', 'recombinase',
            'toxin', 'antitoxin', 'resistance', 'virulence'
        ]
        
        # Step 2: AGGRESSIVE filtering - only keep highest priority data
        scored_lines = []
        for line in lines:
            if not line.strip() or len(line) < 10:  # Skip short/empty lines
                continue
                
            score = 0
            line_lower = line.lower()
            
            # ULTRA HIGH priority for novel/unknown functions
            if any(keyword in line_lower for keyword in ultra_high_priority):
                score += 50  # Much higher weighting
                
            # High priority for interesting functions
            elif any(keyword in line_lower for keyword in high_priority_keywords):
                score += 20
                
            # Medium priority for genomic loci with coordinates
            elif any(keyword in line_lower for keyword in ['coordinate', 'scaffold', 'contig']):
                score += 10
                
            # Low priority for basic metadata
            elif any(keyword in line_lower for keyword in ['genome_id', 'protein_id']):
                score += 1
            
            # Skip very low priority lines entirely
            if score >= 5:  # Only keep moderately interesting or better
                scored_lines.append((score, line))
        
        # Step 3: Sort by priority and apply AGGRESSIVE token budgeting
        scored_lines.sort(key=lambda x: x[0], reverse=True)
        
        compressed_lines = []
        current_tokens = 0
        
        # Reserve tokens for essential context
        header_tokens = count_tokens("Genomic analysis data summary:\n")
        footer_tokens = count_tokens("\n\nAnalysis complete.")
        available_tokens = max_tokens - header_tokens - footer_tokens - 500  # Safety buffer
        
        for score, line in scored_lines:
            line_tokens = count_tokens(line + '\n')
            
            if current_tokens + line_tokens <= available_tokens:
                compressed_lines.append(line)
                current_tokens += line_tokens
            else:
                # Stop when we hit token limit
                break
        
        # Step 4: Build compressed context with essential structure
        compressed_context = "Genomic analysis data summary:\n"
        compressed_context += '\n'.join(compressed_lines[:200])  # Limit to top 200 lines max
        compressed_context += "\n\nAnalysis complete."
        
        final_tokens = count_tokens(compressed_context)
        compression_ratio = final_tokens / count_tokens(context) * 100
        
        logger.info(f"✅ AGGRESSIVE COMPRESSION: {final_tokens} tokens ({compression_ratio:.1f}% of original)")
        
        return compressed_context
    
    def _synthesize_with_model_allocation(self,
                                        context: str,
                                        dspy_synthesizer,
                                        question: str,
                                        task_type: str) -> str:
        """Synthesize using model allocation system with intelligent token management."""
        try:
            from ..dspy_signatures import GenomicSummarizer
            
            # Estimate token count and compress if needed
            estimated_tokens = int(len(context) / 3.5)  # More accurate estimate: 3.5 chars per token
            max_safe_tokens = 18000  # VERY conservative limit for o3 (30K per minute limit)
            
            if estimated_tokens > max_safe_tokens:
                logger.warning(f"🚫 Context too large ({estimated_tokens} tokens), applying intelligent compression")
                context = self._compress_context_for_synthesis(context, max_tokens=max_safe_tokens)
                logger.info(f"✅ Context compressed to ~{int(len(context) / 3.5)} tokens")
            
            def synthesize_call(module):
                return module(
                    genomic_data=context,
                    target_length="detailed",
                    focus_areas="biological insights, functional analysis, novelty detection"
                )
            
            # Use biological interpretation task for o3 allocation
            result = self.model_allocator.create_context_managed_call(
                task_name="biological_interpretation",
                signature_class=GenomicSummarizer, 
                module_call_func=synthesize_call,
                query=question,
                task_context=f"Synthesis of {task_type} data"
            )
            
            if result:
                return result.summary
            else:
                # Fallback to provided synthesizer
                fallback_result = dspy_synthesizer(
                    genomic_data=context,
                    target_length="detailed", 
                    focus_areas="biological insights, functional analysis, novelty detection"
                )
                return fallback_result.summary
                
        except Exception as e:
            logger.error(f"Model allocation synthesis failed: {e}")
            return f"Synthesis error for {task_type}: {str(e)}"
    
    def _generate_final_synthesis(self, dspy_synthesizer, question: str) -> str:
        """
        Generate final synthesis from all chunks.
        
        Args:
            dspy_synthesizer: DSPy synthesizer module
            question: Original user question
            
        Returns:
            Final synthesis text
        """
        if not self.synthesis_chunks:
            return "No synthesis chunks available for final synthesis."
        
        # Combine all chunk insights
        all_findings = []
        all_insights = []
        all_connections = []
        
        for chunk in self.synthesis_chunks:
            all_findings.extend(chunk.get("integrated_findings", []))
            all_insights.extend(chunk.get("emergent_insights", []))
            all_connections.extend(chunk.get("cross_task_synthesis", []))
        
        # Create comprehensive context
        final_context = {
            "original_question": question,
            "total_chunks": len(self.synthesis_chunks),
            "integrated_findings": all_findings,
            "emergent_insights": all_insights,
            "cross_task_connections": all_connections,
            "chunk_summaries": [chunk.get("raw_synthesis", "") for chunk in self.synthesis_chunks]
        }
        
        # Format for DSPy
        formatted_context = self._format_final_context(final_context)
        
        try:
            # Use model allocation system with context manager for final synthesis
            logger.info("🔥 Using model allocation for final synthesis (o3 for complex task)")
            
            from ..dspy_signatures import GenomicSummarizer
            
            def final_synthesize_call(module):
                return module(
                    genomic_data=formatted_context,
                    target_length="detailed",
                    focus_areas="comprehensive biological insights, cross-task integration, quantitative analysis"
                )
            
            final_result = self.model_allocator.create_context_managed_call(
                task_name="final_synthesis",  # Maps to COMPLEX = o3
                signature_class=GenomicSummarizer,
                module_call_func=final_synthesize_call
            )
            
            if final_result is not None:
                return final_result.summary
            else:
                # Fallback to provided dspy_synthesizer if allocation fails
                logger.warning("Model allocation failed for final synthesis, falling back to default")
                final_result = dspy_synthesizer(
                    genomic_data=formatted_context,
                    target_length="detailed",
                    focus_areas="comprehensive biological insights, cross-task integration, quantitative analysis"
                )
                return final_result.summary
            
        except Exception as e:
            logger.error(f"Failed to generate final synthesis: {e}")
            
            # Fallback: create manual synthesis
            return self._create_fallback_synthesis(final_context)
    
    def _format_final_context(self, context: Dict[str, Any]) -> str:
        """Format final context for DSPy synthesis."""
        parts = [
            f"Original Question: {context['original_question']}",
            f"Analysis completed in {context['total_chunks']} synthesis chunks",
            "",
            "Integrated Findings:",
            *[f"- {finding}" for finding in context['integrated_findings'][:10]],
            "",
            "Emergent Insights:",
            *[f"- {insight}" for insight in context['emergent_insights'][:5]],
            "",
            "Cross-Task Connections:",
            *[f"- {conn.get('insight', '')}" for conn in context['cross_task_connections'][:5]],
            "",
            "Chunk Summaries:",
            *[f"Chunk {i+1}: {summary[:200]}..." for i, summary in enumerate(context['chunk_summaries'])]
        ]
        
        return "\n".join(parts)
    
    def _create_fallback_synthesis(self, context: Dict[str, Any]) -> str:
        """Create fallback synthesis when DSPy fails."""
        parts = [
            f"Based on analysis of {context['total_chunks']} synthesis chunks:",
            "",
            "Key Findings:",
            *[f"• {finding}" for finding in context['integrated_findings'][:8]],
            "",
            "Emergent Insights:",
            *[f"• {insight}" for insight in context['emergent_insights'][:4]],
            "",
            f"Analysis completed through progressive synthesis to handle complex multi-task workflow."
        ]
        
        return "\n".join(parts)
    
    def get_synthesis_statistics(self) -> Dict[str, Any]:
        """Get statistics about the synthesis process."""
        total_tokens = sum(chunk.get("tokens_used", 0) for chunk in self.synthesis_chunks)
        
        return {
            "total_chunks": len(self.synthesis_chunks),
            "total_tokens_used": total_tokens,
            "average_tokens_per_chunk": total_tokens / len(self.synthesis_chunks) if self.synthesis_chunks else 0,
            "chunk_themes": [chunk.get("theme", "") for chunk in self.synthesis_chunks]
        }