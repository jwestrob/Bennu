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
    
    def synthesize_progressive(self, 
                             task_notes: List[TaskNote],
                             dspy_synthesizer,
                             question: str) -> str:
        """
        Perform progressive synthesis of task notes.
        
        Args:
            task_notes: List of TaskNote objects to synthesize
            dspy_synthesizer: DSPy synthesizer module
            question: Original user question
            
        Returns:
            Final comprehensive synthesis
        """
        logger.info(f"Starting progressive synthesis of {len(task_notes)} task notes")
        
        if not task_notes:
            return "No task notes available for synthesis."
        
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
            
            # Use DSPy to synthesize insights
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
                "confidence": "medium",
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
            # Generate final synthesis
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