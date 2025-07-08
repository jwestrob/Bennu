#!/usr/bin/env python3
"""
Context compression system for handling large result sets intelligently.
Fixes the corrupted "Unknown: 50 proteins" issue with better summarization.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json
from collections import Counter

logger = logging.getLogger(__name__)

@dataclass
class CompressionStats:
    """Statistics about compression operation."""
    original_count: int
    compressed_count: int
    compression_ratio: float
    compression_method: str
    preserved_fields: List[str]
    summary_stats: Dict[str, Any]

class ContextCompressor:
    """
    Intelligent context compression that preserves essential biological information
    while reducing context size for large datasets.
    """
    
    def __init__(self):
        self.essential_fields = [
            "protein_id", "gene_id", "genome_id", "ko_id", "ko_description",
            "pfam_accessions", "domain_descriptions", "kegg_functions",
            "start_coordinate", "end_coordinate", "strand", "cazyme_family",
            "bgc_product", "function_description"
        ]
        
        self.summarizable_fields = [
            "domain_descriptions", "kegg_descriptions", "pfam_accessions",
            "neighbor_details", "detailed_neighbors", "protein_families"
        ]
    
    def compress_context(self, results: List[Dict[str, Any]], 
                        target_size: int = 50, 
                        preserve_diversity: bool = True) -> Tuple[str, CompressionStats]:
        """
        Compress context results while preserving essential biological information.
        
        Args:
            results: List of query results to compress
            target_size: Target number of results to show in detail
            preserve_diversity: Whether to preserve diversity in sampling
            
        Returns:
            Tuple of (compressed_context_string, compression_stats)
        """
        logger.info(f"🗜️ Compressing {len(results)} results to target size {target_size}")
        
        if not results:
            return "No results to compress.", CompressionStats(
                original_count=0,
                compressed_count=0,
                compression_ratio=0.0,
                compression_method="empty",
                preserved_fields=[],
                summary_stats={}
            )
        
        if len(results) <= target_size:
            # No compression needed
            context = self._format_full_results(results)
            return context, CompressionStats(
                original_count=len(results),
                compressed_count=len(results),
                compression_ratio=1.0,
                compression_method="no_compression",
                preserved_fields=list(results[0].keys()) if results else [],
                summary_stats={}
            )
        
        # Perform intelligent compression
        if preserve_diversity:
            compressed_results = self._diverse_sampling(results, target_size)
            method = "diverse_sampling"
        else:
            compressed_results = results[:target_size]
            method = "simple_truncation"
        
        # Generate summary statistics for remaining results
        remaining_results = results[target_size:]
        summary_stats = self._generate_summary_stats(results, remaining_results)
        
        # Format compressed context
        context = self._format_compressed_context(compressed_results, summary_stats, len(results))
        
        stats = CompressionStats(
            original_count=len(results),
            compressed_count=len(compressed_results),
            compression_ratio=len(compressed_results) / len(results),
            compression_method=method,
            preserved_fields=self._get_preserved_fields(compressed_results),
            summary_stats=summary_stats
        )
        
        logger.info(f"✅ Compressed {len(results)} → {len(compressed_results)} results ({stats.compression_ratio:.2%})")
        
        return context, stats
    
    def _diverse_sampling(self, results: List[Dict[str, Any]], target_size: int) -> List[Dict[str, Any]]:
        """
        Perform diverse sampling to preserve biological diversity in results.
        
        Args:
            results: Full result set
            target_size: Target number of results
            
        Returns:
            Diversely sampled results
        """
        logger.info("🎯 Performing diverse sampling")
        
        # Try to sample based on different criteria
        sampled_results = []
        
        # 1. Sample by genome diversity
        genome_buckets = {}
        for result in results:
            genome_id = result.get("genome_id", "unknown")
            if genome_id not in genome_buckets:
                genome_buckets[genome_id] = []
            genome_buckets[genome_id].append(result)
        
        # Distribute target size across genomes
        genomes = list(genome_buckets.keys())
        if len(genomes) > 1:
            per_genome = max(1, target_size // len(genomes))
            remainder = target_size % len(genomes)
            
            for i, genome_id in enumerate(genomes):
                genome_results = genome_buckets[genome_id]
                take_count = per_genome + (1 if i < remainder else 0)
                
                # Sample functions diversity within genome
                sampled_results.extend(self._sample_by_function_diversity(genome_results, take_count))
        else:
            # Single genome - sample by function diversity
            sampled_results = self._sample_by_function_diversity(results, target_size)
        
        # Ensure we don't exceed target size
        return sampled_results[:target_size]
    
    def _sample_by_function_diversity(self, results: List[Dict[str, Any]], target_size: int) -> List[Dict[str, Any]]:
        """Sample results to maximize functional diversity."""
        if len(results) <= target_size:
            return results
        
        # Group by function descriptions
        function_buckets = {}
        for result in results:
            # Try different function fields
            function_key = (
                result.get("ko_description", "") or 
                result.get("function_description", "") or 
                result.get("domain_descriptions", [""])[0] if isinstance(result.get("domain_descriptions"), list) else 
                str(result.get("domain_descriptions", "")) or
                "unknown_function"
            )
            
            if function_key not in function_buckets:
                function_buckets[function_key] = []
            function_buckets[function_key].append(result)
        
        # Sample from each function bucket
        sampled_results = []
        functions = list(function_buckets.keys())
        
        if len(functions) > 1:
            per_function = max(1, target_size // len(functions))
            remainder = target_size % len(functions)
            
            for i, function_key in enumerate(functions):
                function_results = function_buckets[function_key]
                take_count = per_function + (1 if i < remainder else 0)
                sampled_results.extend(function_results[:take_count])
        else:
            # Single function - just take first N
            sampled_results = results[:target_size]
        
        return sampled_results[:target_size]
    
    def _generate_summary_stats(self, all_results: List[Dict[str, Any]], 
                               remaining_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for the full dataset."""
        logger.info("📊 Generating summary statistics")
        
        stats = {
            "total_results": len(all_results),
            "shown_results": len(all_results) - len(remaining_results),
            "remaining_results": len(remaining_results)
        }
        
        if not all_results:
            return stats
        
        # Count by genome
        genome_counts = Counter()
        for result in all_results:
            genome_id = result.get("genome_id", "unknown")
            genome_counts[genome_id] += 1
        
        stats["genome_distribution"] = dict(genome_counts.most_common())
        
        # Count by function categories
        function_counts = Counter()
        for result in all_results:
            function_desc = (
                result.get("ko_description", "") or 
                result.get("function_description", "") or 
                "unknown_function"
            )
            
            # Categorize functions
            if "transport" in function_desc.lower():
                function_counts["transport"] += 1
            elif "metabol" in function_desc.lower():
                function_counts["metabolism"] += 1
            elif "binding" in function_desc.lower():
                function_counts["binding"] += 1
            elif "synthase" in function_desc.lower() or "synthetase" in function_desc.lower():
                function_counts["synthesis"] += 1
            elif "kinase" in function_desc.lower():
                function_counts["phosphorylation"] += 1
            elif "dehydrogenase" in function_desc.lower():
                function_counts["oxidation_reduction"] += 1
            else:
                function_counts["other"] += 1
        
        stats["function_categories"] = dict(function_counts.most_common())
        
        # Domain statistics if available
        if any("pfam_accessions" in result for result in all_results):
            domain_counts = Counter()
            for result in all_results:
                accessions = result.get("pfam_accessions", [])
                if isinstance(accessions, list):
                    domain_counts.update(accessions)
                elif accessions:
                    domain_counts[str(accessions)] += 1
            
            stats["top_domains"] = dict(domain_counts.most_common(10))
        
        return stats
    
    def _format_compressed_context(self, compressed_results: List[Dict[str, Any]], 
                                  summary_stats: Dict[str, Any], 
                                  total_count: int) -> str:
        """Format compressed context with summary statistics."""
        logger.info("📝 Formatting compressed context")
        
        # Header with statistics
        context_parts = [
            f"📊 **COMPRESSED RESULTS: Showing {len(compressed_results)} of {total_count} results**",
            ""
        ]
        
        # Add summary statistics
        if summary_stats.get("genome_distribution"):
            context_parts.append("**Genome Distribution:**")
            for genome_id, count in summary_stats["genome_distribution"].items():
                context_parts.append(f"  - {genome_id}: {count} results")
            context_parts.append("")
        
        if summary_stats.get("function_categories"):
            context_parts.append("**Function Categories:**")
            for category, count in summary_stats["function_categories"].items():
                context_parts.append(f"  - {category}: {count} results")
            context_parts.append("")
        
        # Add detailed results
        context_parts.append("**DETAILED RESULTS (Sample):**")
        context_parts.append("")
        
        for i, result in enumerate(compressed_results, 1):
            context_parts.append(f"**Result {i}:**")
            
            # Format essential fields
            for field in self.essential_fields:
                if field in result and result[field] is not None:
                    value = result[field]
                    
                    # Handle different value types
                    if isinstance(value, list):
                        if len(value) <= 3:
                            formatted_value = ", ".join(str(v) for v in value)
                        else:
                            formatted_value = f"{', '.join(str(v) for v in value[:3])} ... ({len(value)} total)"
                    else:
                        formatted_value = str(value)
                    
                    # Limit field length
                    if len(formatted_value) > 100:
                        formatted_value = formatted_value[:97] + "..."
                    
                    context_parts.append(f"  {field}: {formatted_value}")
            
            context_parts.append("")
        
        # Add summary footer
        if len(compressed_results) < total_count:
            remaining_count = total_count - len(compressed_results)
            context_parts.append(f"**... and {remaining_count} more results not shown in detail**")
            context_parts.append("")
            context_parts.append("💡 **Note:** This is a compressed view. Use more specific queries or aggregation functions for complete analysis.")
        
        return "\n".join(context_parts)
    
    def _format_full_results(self, results: List[Dict[str, Any]]) -> str:
        """Format full results without compression."""
        logger.info("📝 Formatting full results")
        
        context_parts = [f"📊 **FULL RESULTS: {len(results)} results**", ""]
        
        for i, result in enumerate(results, 1):
            context_parts.append(f"**Result {i}:**")
            
            for key, value in result.items():
                if value is not None:
                    # Handle different value types
                    if isinstance(value, list):
                        if len(value) <= 5:
                            formatted_value = ", ".join(str(v) for v in value)
                        else:
                            formatted_value = f"{', '.join(str(v) for v in value[:5])} ... ({len(value)} total)"
                    else:
                        formatted_value = str(value)
                    
                    # Limit field length
                    if len(formatted_value) > 150:
                        formatted_value = formatted_value[:147] + "..."
                    
                    context_parts.append(f"  {key}: {formatted_value}")
            
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _get_preserved_fields(self, results: List[Dict[str, Any]]) -> List[str]:
        """Get list of preserved fields from results."""
        if not results:
            return []
        
        return list(results[0].keys())
    
    def get_compression_recommendation(self, result_count: int) -> Dict[str, Any]:
        """
        Get compression recommendation based on result count.
        
        Args:
            result_count: Number of results to potentially compress
            
        Returns:
            Dictionary with compression recommendations
        """
        if result_count <= 50:
            return {
                "compress": False,
                "method": "no_compression",
                "target_size": result_count,
                "reasoning": "Small result set - no compression needed"
            }
        elif result_count <= 200:
            return {
                "compress": True,
                "method": "diverse_sampling",
                "target_size": 50,
                "reasoning": "Medium result set - compress with diversity preservation"
            }
        else:
            return {
                "compress": True,
                "method": "diverse_sampling",
                "target_size": 30,
                "reasoning": "Large result set - aggressive compression with summary statistics"
            }