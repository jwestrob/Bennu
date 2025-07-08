"""
Context Compression System for Genomic RAG

This module provides intelligent context compression for large genomic datasets
that exceed token limits. It uses a multi-stage approach to compress context
while preserving essential biological information.

Key Features:
- Token counting with tiktoken or fallback length-based estimation
- Tiered compression strategy based on context size
- Genomic-specific compression preserving functional annotations
- Iterative compression with configurable quality levels
- Chunked processing for very large datasets
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import re
import asyncio

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available - using fallback token estimation")

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    logging.warning("DSPy not available - compression functionality limited")

logger = logging.getLogger(__name__)

class CompressionLevel(Enum):
    """Compression levels with different quality/size tradeoffs."""
    LIGHT = "light"      # 10-20% reduction, preserve most details
    MEDIUM = "medium"    # 30-50% reduction, balance details/size
    AGGRESSIVE = "aggressive"  # 60-80% reduction, essential info only

@dataclass
class CompressionResult:
    """Result of context compression operation."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    compressed_content: str
    compression_level: CompressionLevel
    chunks_processed: int
    summary_stats: Dict[str, Any]
    preserved_elements: List[str]

class TokenCounter:
    """Token counting with tiktoken or fallback estimation."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.encoder = None
        
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoder = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # Fallback to cl100k_base for unknown models
                self.encoder = tiktoken.get_encoding("cl100k_base")
                logger.warning(f"Unknown model {model_name}, using cl100k_base encoding")
        else:
            logger.warning("Using fallback token estimation (4 chars per token)")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Fallback: approximate 4 characters per token
            return len(text) // 4
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens."""
        if self.encoder:
            tokens = self.encoder.encode(text)
            if len(tokens) <= max_tokens:
                return text
            truncated_tokens = tokens[:max_tokens]
            return self.encoder.decode(truncated_tokens)
        else:
            # Fallback: approximate character truncation
            max_chars = max_tokens * 4
            if len(text) <= max_chars:
                return text
            return text[:max_chars] + "..."

class GenomicDataCompressor:
    """Genomic-specific data compression utilities."""
    
    @staticmethod
    def compress_protein_annotations(proteins: List[Dict[str, Any]], 
                                   level: CompressionLevel = CompressionLevel.MEDIUM) -> Tuple[str, Dict[str, Any]]:
        """
        Compress protein annotation data while preserving essential biological information.
        
        Args:
            proteins: List of protein annotation dictionaries
            level: Compression level to apply
            
        Returns:
            Tuple of (compressed_string, stats_dict)
        """
        if not proteins:
            return "", {}
        
        # Group proteins by function for compression
        function_groups = {}
        unique_proteins = []
        
        for protein in proteins:
            # Extract key functional annotations
            kegg_function = protein.get('kegg_function', 'Unknown')
            pfam_domains = protein.get('pfam_domains', [])
            
            # Create function signature
            function_key = f"{kegg_function}|{len(pfam_domains)}"
            
            if function_key not in function_groups:
                function_groups[function_key] = []
            function_groups[function_key].append(protein)
        
        # Compress based on level
        compressed_parts = []
        stats = {
            'total_proteins': len(proteins),
            'unique_functions': len(function_groups),
            'compression_level': level.value
        }
        
        for function_key, group in function_groups.items():
            if level == CompressionLevel.LIGHT:
                # Light compression: show first few examples of each function
                compressed_parts.append(GenomicDataCompressor._compress_function_group_light(function_key, group))
            elif level == CompressionLevel.MEDIUM:
                # Medium compression: summarize with key examples
                compressed_parts.append(GenomicDataCompressor._compress_function_group_medium(function_key, group))
            else:  # AGGRESSIVE
                # Aggressive compression: counts and summaries only
                compressed_parts.append(GenomicDataCompressor._compress_function_group_aggressive(function_key, group))
        
        compressed_string = "\n".join(compressed_parts)
        return compressed_string, stats
    
    @staticmethod
    def _compress_function_group_light(function_key: str, proteins: List[Dict]) -> str:
        """Light compression for function group."""
        kegg_function = function_key.split('|')[0]
        count = len(proteins)
        
        # Show first 3 examples with coordinates
        examples = []
        for i, protein in enumerate(proteins[:3]):
            coord_info = GenomicDataCompressor._extract_coordinate_info(protein)
            examples.append(f"  {protein.get('protein_id', 'Unknown')} {coord_info}")
        
        result = f"Function: {kegg_function} ({count} proteins)\n"
        result += "\n".join(examples)
        
        if count > 3:
            result += f"\n  ... and {count - 3} more proteins"
        
        return result
    
    @staticmethod
    def _compress_function_group_medium(function_key: str, proteins: List[Dict]) -> str:
        """Medium compression for function group."""
        kegg_function = function_key.split('|')[0]
        count = len(proteins)
        
        # Extract genome distribution
        genome_counts = {}
        for protein in proteins:
            genome = protein.get('genome_id', 'Unknown')
            genome_counts[genome] = genome_counts.get(genome, 0) + 1
        
        # Show distribution and one example
        result = f"Function: {kegg_function} ({count} proteins across {len(genome_counts)} genomes)\n"
        
        # Genome distribution
        for genome, count in sorted(genome_counts.items()):
            result += f"  {genome}: {count} proteins\n"
        
        # One representative example
        if proteins:
            example = proteins[0]
            coord_info = GenomicDataCompressor._extract_coordinate_info(example)
            result += f"  Example: {example.get('protein_id', 'Unknown')} {coord_info}"
        
        return result
    
    @staticmethod
    def _compress_function_group_aggressive(function_key: str, proteins: List[Dict]) -> str:
        """Aggressive compression for function group."""
        kegg_function = function_key.split('|')[0]
        count = len(proteins)
        
        # Just counts and basic stats
        genome_counts = {}
        for protein in proteins:
            genome = protein.get('genome_id', 'Unknown')
            genome_counts[genome] = genome_counts.get(genome, 0) + 1
        
        result = f"{kegg_function}: {count} proteins in {len(genome_counts)} genomes"
        return result
    
    @staticmethod
    def _extract_coordinate_info(protein: Dict[str, Any]) -> str:
        """Extract coordinate information from protein."""
        start = protein.get('start', '')
        end = protein.get('end', '')
        strand = protein.get('strand', '')
        
        if start and end:
            return f"({start}-{end}{strand})"
        return ""
    
    @staticmethod
    def compress_genomic_context(context_data: List[Dict[str, Any]], 
                               level: CompressionLevel = CompressionLevel.MEDIUM) -> Tuple[str, Dict[str, Any]]:
        """
        Compress genomic context data (neighborhoods, operons, etc.).
        
        Args:
            context_data: List of genomic context dictionaries
            level: Compression level to apply
            
        Returns:
            Tuple of (compressed_string, stats_dict)
        """
        if not context_data:
            return "", {}
        
        # Group by genomic regions
        compressed_parts = []
        stats = {
            'total_contexts': len(context_data),
            'compression_level': level.value
        }
        
        for i, context in enumerate(context_data):
            if level == CompressionLevel.LIGHT:
                # Full context with minor cleanup
                compressed_parts.append(GenomicDataCompressor._format_context_light(context))
            elif level == CompressionLevel.MEDIUM:
                # Summarized context with key neighbors
                compressed_parts.append(GenomicDataCompressor._format_context_medium(context))
            else:  # AGGRESSIVE
                # Just essential context info
                compressed_parts.append(GenomicDataCompressor._format_context_aggressive(context))
        
        compressed_string = "\n".join(compressed_parts)
        return compressed_string, stats
    
    @staticmethod
    def _format_context_light(context: Dict[str, Any]) -> str:
        """Light formatting for genomic context."""
        result = []
        
        # Main protein info
        if 'protein_id' in context:
            result.append(f"Protein: {context['protein_id']}")
        
        # Neighbors (limit to closest 5)
        if 'neighbors' in context:
            neighbors = context['neighbors'][:5]
            result.append(f"Neighbors ({len(neighbors)}):")
            for neighbor in neighbors:
                distance = neighbor.get('distance', 'Unknown')
                function = neighbor.get('function', 'Unknown')
                result.append(f"  {function} (distance: {distance})")
        
        return "\n".join(result)
    
    @staticmethod
    def _format_context_medium(context: Dict[str, Any]) -> str:
        """Medium formatting for genomic context."""
        result = []
        
        # Main protein info
        if 'protein_id' in context:
            result.append(f"Protein: {context['protein_id']}")
        
        # Summarized neighbors
        if 'neighbors' in context:
            neighbors = context['neighbors']
            close_neighbors = [n for n in neighbors if n.get('distance', 1000) < 200]
            result.append(f"Context: {len(close_neighbors)} close neighbors, {len(neighbors)} total")
            
            # Show closest 2 neighbors
            for neighbor in neighbors[:2]:
                function = neighbor.get('function', 'Unknown')
                distance = neighbor.get('distance', 'Unknown')
                result.append(f"  {function} ({distance}bp)")
        
        return "\n".join(result)
    
    @staticmethod
    def _format_context_aggressive(context: Dict[str, Any]) -> str:
        """Aggressive formatting for genomic context."""
        protein_id = context.get('protein_id', 'Unknown')
        neighbor_count = len(context.get('neighbors', []))
        return f"{protein_id}: {neighbor_count} neighbors"

class ContextCompressor:
    """
    Main context compression system with multi-stage processing.
    
    Provides intelligent compression of genomic context data to fit within
    token limits while preserving essential biological information.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the context compressor.
        
        Args:
            model_name: Model name for token counting
        """
        self.token_counter = TokenCounter(model_name)
        self.genomic_compressor = GenomicDataCompressor()
        
        # Token limits for different tiers
        self.SMALL_LIMIT = 10000   # Direct synthesis
        self.MEDIUM_LIMIT = 30000  # Single compression
        self.LARGE_LIMIT = 100000  # Multi-stage compression
        
        logger.info(f"ContextCompressor initialized with model: {model_name}")
    
    def get_compression_strategy(self, context_size: int) -> Tuple[str, CompressionLevel]:
        """
        Determine compression strategy based on context size.
        
        Args:
            context_size: Size of context in tokens
            
        Returns:
            Tuple of (strategy, compression_level)
        """
        if context_size <= self.SMALL_LIMIT:
            return "direct", CompressionLevel.LIGHT
        elif context_size <= self.MEDIUM_LIMIT:
            return "single_pass", CompressionLevel.MEDIUM
        else:
            return "multi_stage", CompressionLevel.AGGRESSIVE
    
    async def compress_context(self, context_data: Dict[str, Any], 
                              target_tokens: int = 25000) -> CompressionResult:
        """
        Compress context data to fit within target token limit.
        
        Args:
            context_data: Dictionary containing context data to compress
            target_tokens: Target token limit for compressed content
            
        Returns:
            CompressionResult with compressed content and metadata
        """
        logger.info(f"Starting context compression with target: {target_tokens} tokens")
        
        # Convert context to string for initial measurement
        original_content = self._format_context_for_compression(context_data)
        original_size = self.token_counter.count_tokens(original_content)
        
        logger.info(f"Original context size: {original_size} tokens")
        
        # Determine compression strategy
        strategy, initial_level = self.get_compression_strategy(original_size)
        
        if original_size <= target_tokens:
            # No compression needed
            return CompressionResult(
                original_size=original_size,
                compressed_size=original_size,
                compression_ratio=1.0,
                compressed_content=original_content,
                compression_level=CompressionLevel.LIGHT,
                chunks_processed=1,
                summary_stats={"strategy": "no_compression"},
                preserved_elements=["all"]
            )
        
        if strategy == "single_pass":
            return await self._single_pass_compression(context_data, target_tokens, initial_level)
        else:
            return await self._multi_stage_compression(context_data, target_tokens)
    
    async def _single_pass_compression(self, context_data: Dict[str, Any], 
                                     target_tokens: int, 
                                     level: CompressionLevel) -> CompressionResult:
        """Single-pass compression for medium-sized contexts."""
        logger.debug(f"Applying single-pass compression at level: {level.value}")
        
        compressed_parts = []
        stats = {"chunks_processed": 0, "compression_level": level.value}
        preserved_elements = []
        
        # Process different data types
        for data_type, data in context_data.items():
            if data_type == "structured_data" and isinstance(data, list):
                compressed_content, chunk_stats = self.genomic_compressor.compress_protein_annotations(
                    data, level
                )
                if compressed_content:
                    compressed_parts.append(f"=== {data_type.upper()} ===\n{compressed_content}")
                    preserved_elements.append(data_type)
                    stats["chunks_processed"] += 1
                    stats.update(chunk_stats)
            
            elif data_type == "semantic_data" and isinstance(data, list):
                # Compress semantic similarity data
                compressed_content = self._compress_semantic_data(data, level)
                if compressed_content:
                    compressed_parts.append(f"=== {data_type.upper()} ===\n{compressed_content}")
                    preserved_elements.append(data_type)
                    stats["chunks_processed"] += 1
            
            elif data_type == "genomic_context" and isinstance(data, list):
                compressed_content, chunk_stats = self.genomic_compressor.compress_genomic_context(
                    data, level
                )
                if compressed_content:
                    compressed_parts.append(f"=== GENOMIC CONTEXT ===\n{compressed_content}")
                    preserved_elements.append(data_type)
                    stats["chunks_processed"] += 1
                    stats.update(chunk_stats)
            
            elif data_type == "tool_results" and isinstance(data, list):
                # Compress tool execution results
                compressed_content = self._compress_tool_results(data, level)
                if compressed_content:
                    compressed_parts.append(f"=== TOOL RESULTS ===\n{compressed_content}")
                    preserved_elements.append(data_type)
                    stats["chunks_processed"] += 1
            
            elif data_type == "metadata" and isinstance(data, dict):
                # Include essential metadata
                compressed_content = self._compress_metadata(data, level)
                if compressed_content:
                    compressed_parts.append(f"=== METADATA ===\n{compressed_content}")
                    preserved_elements.append(data_type)
                    stats["chunks_processed"] += 1
        
        # Combine compressed parts
        compressed_content = "\n\n".join(compressed_parts)
        compressed_size = self.token_counter.count_tokens(compressed_content)
        
        # Check if further compression needed
        if compressed_size > target_tokens:
            logger.warning(f"Single-pass compression insufficient: {compressed_size} > {target_tokens}")
            # If already at aggressive level, truncate to target
            if level == CompressionLevel.AGGRESSIVE:
                logger.warning("Already at aggressive compression, truncating to target tokens")
                compressed_content = self.token_counter.truncate_to_tokens(compressed_content, target_tokens)
                compressed_size = self.token_counter.count_tokens(compressed_content)
            else:
                # Apply more aggressive compression
                return await self._single_pass_compression(context_data, target_tokens, CompressionLevel.AGGRESSIVE)
        
        original_content = self._format_context_for_compression(context_data)
        original_size = self.token_counter.count_tokens(original_content)
        
        return CompressionResult(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compressed_size / original_size,
            compressed_content=compressed_content,
            compression_level=level,
            chunks_processed=stats["chunks_processed"],
            summary_stats=stats,
            preserved_elements=preserved_elements
        )
    
    async def _multi_stage_compression(self, context_data: Dict[str, Any], 
                                     target_tokens: int) -> CompressionResult:
        """Multi-stage compression for very large contexts."""
        logger.info("Applying multi-stage compression")
        
        # Stage 1: Chunk data into manageable pieces
        chunks = self._chunk_context_data(context_data, chunk_size=5000)
        
        # Stage 2: Compress each chunk
        compressed_chunks = []
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            # Log progress periodically
            if i % 50 == 0 or i == total_chunks - 1:
                logger.info(f"Compressing chunk {i+1}/{total_chunks}")
            
            # Compress chunk with aggressive level
            chunk_result = await self._single_pass_compression(
                chunk, target_tokens // total_chunks, CompressionLevel.AGGRESSIVE
            )
            compressed_chunks.append(chunk_result.compressed_content)
        
        # Stage 3: Combine and potentially compress further
        combined_content = "\n\n=== CHUNK SEPARATOR ===\n\n".join(compressed_chunks)
        combined_size = self.token_counter.count_tokens(combined_content)
        
        # Stage 4: Final compression if still too large
        if combined_size > target_tokens:
            logger.info("Applying final compression to combined chunks")
            combined_content = await self._summarize_compressed_chunks(compressed_chunks, target_tokens)
            combined_size = self.token_counter.count_tokens(combined_content)
        
        original_content = self._format_context_for_compression(context_data)
        original_size = self.token_counter.count_tokens(original_content)
        
        return CompressionResult(
            original_size=original_size,
            compressed_size=combined_size,
            compression_ratio=combined_size / original_size,
            compressed_content=combined_content,
            compression_level=CompressionLevel.AGGRESSIVE,
            chunks_processed=total_chunks,
            summary_stats={
                "strategy": "multi_stage",
                "total_chunks": total_chunks,
                "final_compression": combined_size < target_tokens
            },
            preserved_elements=["essential_summary"]
        )
    
    def _chunk_context_data(self, context_data: Dict[str, Any], chunk_size: int = 5000) -> List[Dict[str, Any]]:
        """
        Chunk large context data into manageable pieces.
        
        Args:
            context_data: Context data to chunk
            chunk_size: Approximate size of each chunk in tokens
            
        Returns:
            List of chunked context dictionaries
        """
        chunks = []
        
        for data_type, data in context_data.items():
            if isinstance(data, list) and len(data) > 10:
                # Split large lists into chunks
                chunk_count = max(1, len(data) * 100 // chunk_size)  # Estimate based on average item size
                items_per_chunk = len(data) // chunk_count
                
                for i in range(0, len(data), items_per_chunk):
                    chunk_data = data[i:i + items_per_chunk]
                    chunks.append({data_type: chunk_data})
            else:
                # Small data goes into its own chunk
                chunks.append({data_type: data})
        
        return chunks
    
    async def _summarize_compressed_chunks(self, compressed_chunks: List[str], target_tokens: int) -> str:
        """
        Summarize compressed chunks using DSPy if available.
        
        Args:
            compressed_chunks: List of compressed chunk strings
            target_tokens: Target token count for final summary
            
        Returns:
            Summarized content
        """
        if not DSPY_AVAILABLE:
            logger.warning("DSPy not available - using simple truncation")
            combined = "\n".join(compressed_chunks)
            return self.token_counter.truncate_to_tokens(combined, target_tokens)
        
        # Use DSPy for intelligent summarization
        try:
            from .dspy_signatures import GenomicSummarizer
            summarizer = dspy.Predict(GenomicSummarizer)
            
            # Summarize each chunk first
            chunk_summaries = []
            for i, chunk in enumerate(compressed_chunks):
                chunk_size = self.token_counter.count_tokens(chunk)
                if chunk_size > 2000:  # Only summarize large chunks
                    summary = summarizer(
                        genomic_data=chunk,
                        target_length="brief",
                        focus_areas="functional annotations, key findings"
                    )
                    chunk_summaries.append(f"Chunk {i+1}: {summary.summary}")
                else:
                    chunk_summaries.append(f"Chunk {i+1}: {chunk}")
            
            # Combine summaries
            combined_summary = "\n\n".join(chunk_summaries)
            
            # Final check and truncation if needed
            if self.token_counter.count_tokens(combined_summary) > target_tokens:
                combined_summary = self.token_counter.truncate_to_tokens(combined_summary, target_tokens)
            
            return combined_summary
            
        except Exception as e:
            logger.error(f"DSPy summarization failed: {e}")
            # Fallback to simple truncation
            combined = "\n".join(compressed_chunks)
            return self.token_counter.truncate_to_tokens(combined, target_tokens)
    
    def _compress_semantic_data(self, semantic_data: List[Dict[str, Any]], 
                               level: CompressionLevel) -> str:
        """
        Compress semantic similarity data.
        
        Args:
            semantic_data: List of semantic similarity results
            level: Compression level
            
        Returns:
            Compressed semantic data string
        """
        if not semantic_data:
            return ""
        
        if level == CompressionLevel.LIGHT:
            # Show top similarities with full details
            results = []
            for i, item in enumerate(semantic_data[:10]):
                similarity = item.get('similarity', 0)
                protein_id = item.get('protein_id', 'Unknown')
                function = item.get('function', 'Unknown')
                results.append(f"  {i+1}. {protein_id} (similarity: {similarity:.3f}) - {function}")
            return "\n".join(results)
        
        elif level == CompressionLevel.MEDIUM:
            # Show top similarities with summary
            top_results = semantic_data[:5]
            results = []
            for i, item in enumerate(top_results):
                similarity = item.get('similarity', 0)
                protein_id = item.get('protein_id', 'Unknown')
                results.append(f"  {i+1}. {protein_id} (sim: {similarity:.3f})")
            
            if len(semantic_data) > 5:
                results.append(f"  ... and {len(semantic_data) - 5} more similar proteins")
            
            return "\n".join(results)
        
        else:  # AGGRESSIVE
            # Just summary statistics
            if semantic_data:
                top_sim = max(item.get('similarity', 0) for item in semantic_data)
                count = len(semantic_data)
                return f"  {count} similar proteins (max similarity: {top_sim:.3f})"
            return ""
    
    def _format_context_for_compression(self, context_data: Dict[str, Any]) -> str:
        """
        Format context data dictionary as string for token counting.
        
        Args:
            context_data: Context data dictionary
            
        Returns:
            Formatted string representation
        """
        formatted_parts = []
        
        for data_type, data in context_data.items():
            if isinstance(data, list):
                formatted_parts.append(f"=== {data_type.upper()} ({len(data)} items) ===")
                for item in data:
                    formatted_parts.append(str(item))
            else:
                formatted_parts.append(f"=== {data_type.upper()} ===")
                formatted_parts.append(str(data))
        
        return "\n".join(formatted_parts)
    
    def _compress_tool_results(self, tool_results: List[Dict[str, Any]], 
                              level: CompressionLevel) -> str:
        """
        Compress tool execution results.
        
        Args:
            tool_results: List of tool execution results
            level: Compression level
            
        Returns:
            Compressed tool results string
        """
        if not tool_results:
            return ""
        
        compressed_parts = []
        
        for tool_result in tool_results:
            task_id = tool_result.get('task_id', 'unknown')
            tool_name = tool_result.get('tool_name', 'unknown')
            result = tool_result.get('result', '')
            
            if level == CompressionLevel.LIGHT:
                # Full tool output with minor formatting
                compressed_parts.append(f"Task {task_id} ({tool_name}):")
                compressed_parts.append(f"  {str(result)}")
            
            elif level == CompressionLevel.MEDIUM:
                # Summarized tool output
                result_str = str(result)
                if len(result_str) > 200:
                    result_str = result_str[:200] + "..."
                compressed_parts.append(f"Task {task_id} ({tool_name}): {result_str}")
            
            else:  # AGGRESSIVE
                # Just tool and task info
                compressed_parts.append(f"{task_id}: {tool_name} executed")
        
        return "\n".join(compressed_parts)
    
    def _compress_metadata(self, metadata: Dict[str, Any], 
                          level: CompressionLevel) -> str:
        """
        Compress metadata information.
        
        Args:
            metadata: Metadata dictionary
            level: Compression level
            
        Returns:
            Compressed metadata string
        """
        if not metadata:
            return ""
        
        if level == CompressionLevel.LIGHT:
            # Include most metadata
            parts = []
            for key, value in metadata.items():
                if isinstance(value, dict):
                    parts.append(f"{key}: {len(value)} items")
                else:
                    parts.append(f"{key}: {str(value)[:100]}")
            return "\n".join(parts)
        
        elif level == CompressionLevel.MEDIUM:
            # Essential metadata only
            essential_keys = ['task_type', 'query_type', 'search_strategy', 'total_results']
            parts = []
            for key in essential_keys:
                if key in metadata:
                    parts.append(f"{key}: {metadata[key]}")
            return "\n".join(parts)
        
        else:  # AGGRESSIVE
            # Just count of metadata items
            return f"Metadata: {len(metadata)} items"

# Test function
async def test_context_compressor():
    """Test the context compression system."""
    
    # Create test data
    test_proteins = [
        {
            'protein_id': 'protein_001',
            'kegg_function': 'K00001 - alcohol dehydrogenase',
            'pfam_domains': ['PF00107', 'PF08240'],
            'start': 1000,
            'end': 2000,
            'strand': '+',
            'genome_id': 'genome_A'
        },
        {
            'protein_id': 'protein_002',
            'kegg_function': 'K00001 - alcohol dehydrogenase',
            'pfam_domains': ['PF00107'],
            'start': 3000,
            'end': 4000,
            'strand': '-',
            'genome_id': 'genome_B'
        }
    ] * 50  # Create 100 proteins
    
    test_context = {
        'structured_data': test_proteins,
        'semantic_data': [
            {'protein_id': 'sim_001', 'similarity': 0.85, 'function': 'similar enzyme'},
            {'protein_id': 'sim_002', 'similarity': 0.72, 'function': 'related protein'}
        ],
        'genomic_context': [
            {
                'protein_id': 'context_001',
                'neighbors': [
                    {'function': 'upstream gene', 'distance': 150},
                    {'function': 'downstream gene', 'distance': 200}
                ]
            }
        ]
    }
    
    # Test compression
    compressor = ContextCompressor()
    result = await compressor.compress_context(test_context, target_tokens=5000)
    
    print("=== Context Compression Test ===")
    print(f"Original size: {result.original_size} tokens")
    print(f"Compressed size: {result.compressed_size} tokens")
    print(f"Compression ratio: {result.compression_ratio:.2f}")
    print(f"Compression level: {result.compression_level.value}")
    print(f"Chunks processed: {result.chunks_processed}")
    print(f"Preserved elements: {result.preserved_elements}")
    print("\nCompressed content preview:")
    print(result.compressed_content[:500] + "..." if len(result.compressed_content) > 500 else result.compressed_content)

if __name__ == "__main__":
    asyncio.run(test_context_compressor())