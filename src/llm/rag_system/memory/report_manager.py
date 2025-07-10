"""
Multi-part report generation system for handling large-scale genomic analyses.

Provides intelligent chunking, progressive generation, and navigation for comprehensive
reports that exceed token limits while maintaining scientific rigor and user accessibility.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import tiktoken

from .note_schemas import TaskNote, ConfidenceLevel

logger = logging.getLogger(__name__)


class ReportType(str, Enum):
    """Types of reports that can be generated."""
    COMPREHENSIVE = "comprehensive"
    FUNCTIONAL_COMPARISON = "functional_comparison"
    PATHWAY_ANALYSIS = "pathway_analysis"
    CRISPR_SYSTEMS = "crispr_systems"
    TRANSPORT_SYSTEMS = "transport_systems"
    METABOLIC_RECONSTRUCTION = "metabolic_reconstruction"
    PROTEIN_FAMILIES = "protein_families"
    COMPARATIVE_GENOMICS = "comparative_genomics"


class ChunkingStrategy(str, Enum):
    """Strategies for chunking large reports."""
    BY_GENOME = "by_genome"
    BY_CATEGORY = "by_category"
    BY_PATHWAY = "by_pathway"
    BY_FUNCTION = "by_function"
    BY_SYSTEM = "by_system"


@dataclass
class ReportChunk:
    """Represents a chunk of a multi-part report."""
    chunk_id: str
    title: str
    data_subset: List[Dict[str, Any]]
    context: str
    estimated_tokens: int
    chunk_type: ChunkingStrategy


@dataclass
class ReportPlan:
    """Plan for generating a multi-part report."""
    report_type: ReportType
    requires_chunking: bool
    total_estimated_tokens: int
    chunks: List[ReportChunk]
    executive_summary_tokens: int
    synthesis_tokens: int
    max_tokens_per_part: int


class ReportPlanner:
    """
    Plans multi-part reports based on data size and complexity.
    
    Estimates token requirements and creates chunking strategies for large datasets
    while maintaining scientific coherence and user accessibility.
    """
    
    def __init__(self, max_tokens_per_part: int = 18000):
        """
        Initialize report planner.
        
        Args:
            max_tokens_per_part: Maximum tokens per report part (leaving buffer for o3)
        """
        self.max_tokens_per_part = max_tokens_per_part
        self.base_tokens = 2000  # Introduction, conclusions, formatting
        
        # Token estimates per item by report type
        self.token_estimates = {
            ReportType.COMPREHENSIVE: 200,
            ReportType.FUNCTIONAL_COMPARISON: 150,
            ReportType.PATHWAY_ANALYSIS: 180,
            ReportType.CRISPR_SYSTEMS: 300,
            ReportType.TRANSPORT_SYSTEMS: 250,
            ReportType.METABOLIC_RECONSTRUCTION: 220,
            ReportType.PROTEIN_FAMILIES: 170,
            ReportType.COMPARATIVE_GENOMICS: 160
        }
        
        # Initialize tokenizer for accurate estimation
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception as e:
            logger.warning(f"Failed to initialize tokenizer: {e}")
            self.tokenizer = None
    
    def plan_report(self, question: str, data: List[Dict[str, Any]], 
                   task_notes: List[TaskNote] = None) -> ReportPlan:
        """
        Create a plan for generating a report.
        
        Args:
            question: Original user question
            data: Dataset to analyze
            task_notes: Optional task notes for context
            
        Returns:
            ReportPlan with chunking strategy
        """
        # Determine report type
        report_type = self._classify_report_type(question)
        
        # Estimate total size
        total_estimated_tokens = self._estimate_total_tokens(data, report_type, task_notes)
        
        # Determine if chunking is needed
        requires_chunking = total_estimated_tokens > self.max_tokens_per_part
        
        if requires_chunking:
            logger.info(f"📄 Planning multi-part report: {total_estimated_tokens} tokens estimated")
            chunks = self._create_chunks(data, report_type, question)
        else:
            logger.info(f"📄 Planning single-part report: {total_estimated_tokens} tokens estimated")
            chunks = []
        
        return ReportPlan(
            report_type=report_type,
            requires_chunking=requires_chunking,
            total_estimated_tokens=total_estimated_tokens,
            chunks=chunks,
            executive_summary_tokens=1500,
            synthesis_tokens=2000,
            max_tokens_per_part=self.max_tokens_per_part
        )
    
    def _classify_report_type(self, question: str) -> ReportType:
        """Classify the type of report based on the question."""
        question_lower = question.lower()
        
        # Check for specific system types
        if any(term in question_lower for term in ['crispr', 'cas protein', 'cas gene']):
            return ReportType.CRISPR_SYSTEMS
        elif any(term in question_lower for term in ['transport', 'transporter', 'permease']):
            return ReportType.TRANSPORT_SYSTEMS
        elif any(term in question_lower for term in ['pathway', 'metabolic', 'metabolism']):
            return ReportType.METABOLIC_RECONSTRUCTION
        elif any(term in question_lower for term in ['protein families', 'protein family', 'domains']):
            return ReportType.PROTEIN_FAMILIES
        elif any(term in question_lower for term in ['functional', 'function', 'categories']):
            return ReportType.FUNCTIONAL_COMPARISON
        elif any(term in question_lower for term in ['comparative', 'compare', 'comparison']):
            return ReportType.COMPARATIVE_GENOMICS
        else:
            return ReportType.COMPREHENSIVE
    
    def _estimate_total_tokens(self, data: List[Dict[str, Any]], 
                              report_type: ReportType, 
                              task_notes: List[TaskNote] = None) -> int:
        """Estimate total tokens needed for the report."""
        # Base tokens for structure
        total_tokens = self.base_tokens
        
        # Tokens per data item
        tokens_per_item = self.token_estimates.get(report_type, 200)
        data_tokens = len(data) * tokens_per_item
        
        # Additional tokens from task notes
        if task_notes:
            note_tokens = sum(len(note.observations) * 50 + len(note.key_findings) * 75 
                            for note in task_notes)
            total_tokens += note_tokens
        
        total_tokens += data_tokens
        
        # Add overhead for synthesis and formatting
        total_tokens = int(total_tokens * 1.2)  # 20% overhead
        
        return total_tokens
    
    def _create_chunks(self, data: List[Dict[str, Any]], 
                      report_type: ReportType, 
                      question: str) -> List[ReportChunk]:
        """Create chunks for multi-part report."""
        chunking_strategy = self._determine_chunking_strategy(report_type, question)
        
        if chunking_strategy == ChunkingStrategy.BY_GENOME:
            return self._chunk_by_genome(data, report_type)
        elif chunking_strategy == ChunkingStrategy.BY_CATEGORY:
            return self._chunk_by_category(data, report_type)
        elif chunking_strategy == ChunkingStrategy.BY_FUNCTION:
            return self._chunk_by_function(data, report_type)
        elif chunking_strategy == ChunkingStrategy.BY_SYSTEM:
            return self._chunk_by_system(data, report_type)
        else:
            # Default: simple size-based chunking
            return self._chunk_by_size(data, report_type)
    
    def _determine_chunking_strategy(self, report_type: ReportType, question: str) -> ChunkingStrategy:
        """Determine the best chunking strategy for the report type."""
        if report_type == ReportType.CRISPR_SYSTEMS:
            return ChunkingStrategy.BY_SYSTEM
        elif report_type == ReportType.TRANSPORT_SYSTEMS:
            return ChunkingStrategy.BY_SYSTEM
        elif report_type == ReportType.FUNCTIONAL_COMPARISON:
            return ChunkingStrategy.BY_CATEGORY
        elif report_type == ReportType.PATHWAY_ANALYSIS:
            return ChunkingStrategy.BY_PATHWAY
        elif report_type == ReportType.COMPARATIVE_GENOMICS:
            return ChunkingStrategy.BY_GENOME
        else:
            return ChunkingStrategy.BY_GENOME
    
    def _chunk_by_genome(self, data: List[Dict[str, Any]], 
                        report_type: ReportType) -> List[ReportChunk]:
        """Chunk data by genome groups."""
        chunks = []
        tokens_per_item = self.token_estimates.get(report_type, 200)
        
        # Group data by genome
        genome_groups = {}
        for item in data:
            genome_id = item.get('genome_id', item.get('genome', 'unknown'))
            if genome_id not in genome_groups:
                genome_groups[genome_id] = []
            genome_groups[genome_id].append(item)
        
        # Create chunks of genomes that fit within token limits
        current_chunk_data = []
        current_chunk_genomes = []
        current_tokens = 0
        chunk_id = 1
        
        for genome_id, genome_data in genome_groups.items():
            genome_tokens = len(genome_data) * tokens_per_item
            
            if current_tokens + genome_tokens > self.max_tokens_per_part and current_chunk_data:
                # Create chunk with current data
                chunks.append(ReportChunk(
                    chunk_id=f"chunk_{chunk_id}",
                    title=f"Genomes {', '.join(current_chunk_genomes[:3])}{'...' if len(current_chunk_genomes) > 3 else ''}",
                    data_subset=current_chunk_data,
                    context=f"Analysis of {len(current_chunk_genomes)} genomes",
                    estimated_tokens=current_tokens,
                    chunk_type=ChunkingStrategy.BY_GENOME
                ))
                
                # Start new chunk
                current_chunk_data = []
                current_chunk_genomes = []
                current_tokens = 0
                chunk_id += 1
            
            current_chunk_data.extend(genome_data)
            current_chunk_genomes.append(genome_id)
            current_tokens += genome_tokens
        
        # Add final chunk
        if current_chunk_data:
            chunks.append(ReportChunk(
                chunk_id=f"chunk_{chunk_id}",
                title=f"Genomes {', '.join(current_chunk_genomes[:3])}{'...' if len(current_chunk_genomes) > 3 else ''}",
                data_subset=current_chunk_data,
                context=f"Analysis of {len(current_chunk_genomes)} genomes",
                estimated_tokens=current_tokens,
                chunk_type=ChunkingStrategy.BY_GENOME
            ))
        
        return chunks
    
    def _chunk_by_category(self, data: List[Dict[str, Any]], 
                          report_type: ReportType) -> List[ReportChunk]:
        """Chunk data by functional categories."""
        chunks = []
        tokens_per_item = self.token_estimates.get(report_type, 200)
        
        # Group by functional category
        category_groups = {}
        for item in data:
            category = item.get('category', item.get('ko_description', 'unknown'))
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(item)
        
        # Create chunks by category
        current_chunk_data = []
        current_chunk_categories = []
        current_tokens = 0
        chunk_id = 1
        
        for category, category_data in category_groups.items():
            category_tokens = len(category_data) * tokens_per_item
            
            if current_tokens + category_tokens > self.max_tokens_per_part and current_chunk_data:
                chunks.append(ReportChunk(
                    chunk_id=f"chunk_{chunk_id}",
                    title=f"Functional Categories: {', '.join(current_chunk_categories[:2])}{'...' if len(current_chunk_categories) > 2 else ''}",
                    data_subset=current_chunk_data,
                    context=f"Analysis of {len(current_chunk_categories)} functional categories",
                    estimated_tokens=current_tokens,
                    chunk_type=ChunkingStrategy.BY_CATEGORY
                ))
                
                current_chunk_data = []
                current_chunk_categories = []
                current_tokens = 0
                chunk_id += 1
            
            current_chunk_data.extend(category_data)
            current_chunk_categories.append(category)
            current_tokens += category_tokens
        
        # Add final chunk
        if current_chunk_data:
            chunks.append(ReportChunk(
                chunk_id=f"chunk_{chunk_id}",
                title=f"Functional Categories: {', '.join(current_chunk_categories[:2])}{'...' if len(current_chunk_categories) > 2 else ''}",
                data_subset=current_chunk_data,
                context=f"Analysis of {len(current_chunk_categories)} functional categories",
                estimated_tokens=current_tokens,
                chunk_type=ChunkingStrategy.BY_CATEGORY
            ))
        
        return chunks
    
    def _chunk_by_function(self, data: List[Dict[str, Any]], 
                          report_type: ReportType) -> List[ReportChunk]:
        """Chunk data by functional groups."""
        # Similar to category but more specific grouping
        return self._chunk_by_category(data, report_type)
    
    def _chunk_by_system(self, data: List[Dict[str, Any]], 
                        report_type: ReportType) -> List[ReportChunk]:
        """Chunk data by biological systems (e.g., CRISPR types, transport families)."""
        chunks = []
        tokens_per_item = self.token_estimates.get(report_type, 200)
        
        # Group by system type
        system_groups = {}
        for item in data:
            # Try to extract system type from various fields
            system_type = (item.get('system_type') or 
                          item.get('family_type') or 
                          item.get('cazyme_type') or 
                          item.get('transport_type') or
                          'unknown')
            
            if system_type not in system_groups:
                system_groups[system_type] = []
            system_groups[system_type].append(item)
        
        # Create chunks by system
        for system_type, system_data in system_groups.items():
            system_tokens = len(system_data) * tokens_per_item
            
            if system_tokens > self.max_tokens_per_part:
                # Split large systems into sub-chunks
                sub_chunks = self._split_large_system(system_data, system_type, tokens_per_item)
                chunks.extend(sub_chunks)
            else:
                chunks.append(ReportChunk(
                    chunk_id=f"system_{system_type}",
                    title=f"{system_type.title()} Systems",
                    data_subset=system_data,
                    context=f"Analysis of {system_type} biological systems",
                    estimated_tokens=system_tokens,
                    chunk_type=ChunkingStrategy.BY_SYSTEM
                ))
        
        return chunks
    
    def _split_large_system(self, system_data: List[Dict[str, Any]], 
                           system_type: str, 
                           tokens_per_item: int) -> List[ReportChunk]:
        """Split a large system into multiple chunks."""
        sub_chunks = []
        items_per_chunk = self.max_tokens_per_part // tokens_per_item
        
        for i in range(0, len(system_data), items_per_chunk):
            chunk_data = system_data[i:i + items_per_chunk]
            sub_chunk_id = i // items_per_chunk + 1
            
            sub_chunks.append(ReportChunk(
                chunk_id=f"system_{system_type}_part_{sub_chunk_id}",
                title=f"{system_type.title()} Systems - Part {sub_chunk_id}",
                data_subset=chunk_data,
                context=f"Analysis of {system_type} systems (Part {sub_chunk_id})",
                estimated_tokens=len(chunk_data) * tokens_per_item,
                chunk_type=ChunkingStrategy.BY_SYSTEM
            ))
        
        return sub_chunks
    
    def _chunk_by_size(self, data: List[Dict[str, Any]], 
                      report_type: ReportType) -> List[ReportChunk]:
        """Simple size-based chunking as fallback."""
        chunks = []
        tokens_per_item = self.token_estimates.get(report_type, 200)
        items_per_chunk = self.max_tokens_per_part // tokens_per_item
        
        for i in range(0, len(data), items_per_chunk):
            chunk_data = data[i:i + items_per_chunk]
            chunk_id = i // items_per_chunk + 1
            
            chunks.append(ReportChunk(
                chunk_id=f"chunk_{chunk_id}",
                title=f"Analysis Part {chunk_id}",
                data_subset=chunk_data,
                context=f"Part {chunk_id} of comprehensive analysis",
                estimated_tokens=len(chunk_data) * tokens_per_item,
                chunk_type=ChunkingStrategy.BY_GENOME
            ))
        
        return chunks
    
    def estimate_chunk_tokens(self, chunk: ReportChunk) -> int:
        """Estimate tokens for a specific chunk."""
        if self.tokenizer:
            try:
                # More accurate estimation if tokenizer available
                sample_text = str(chunk.data_subset[:5])  # Sample first 5 items
                sample_tokens = len(self.tokenizer.encode(sample_text))
                scale_factor = len(chunk.data_subset) / min(5, len(chunk.data_subset))
                return int(sample_tokens * scale_factor * 1.5)  # Add overhead
            except Exception:
                pass
        
        # Fallback to simple estimation
        return chunk.estimated_tokens