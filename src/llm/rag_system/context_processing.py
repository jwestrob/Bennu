#!/usr/bin/env python3
"""
Context processing and retrieval for genomic RAG system.
Handles context formatting, size detection, and intelligent routing.
"""

import logging
from typing import Dict, Any, List, Optional
import asyncio

from ..query_processor import QueryResult, Neo4jQueryProcessor
from .data_scaling import ScalingRouter, convert_to_count_query, convert_to_aggregated_query
from .utils import safe_log_data

logger = logging.getLogger(__name__)

class ContextProcessor:
    """Handles context retrieval and formatting with intelligent scaling."""
    
    def __init__(self, neo4j_processor: Neo4jQueryProcessor):
        self.neo4j_processor = neo4j_processor
        self.scaling_router = ScalingRouter()
    
    async def retrieve_context(self, query_result: QueryResult, **kwargs) -> str:
        """
        Retrieve and format context with intelligent size handling.
        
        Args:
            query_result: Result from query execution
            **kwargs: Additional context parameters
            
        Returns:
            Formatted context string ready for LLM
        """
        logger.info("🔍 Retrieving context with intelligent scaling")
        
        try:
            if not query_result.results:
                return "No results found for the query."
            
            # Check if this is a large dataset that needs special handling
            result_count = len(query_result.results)
            
            if result_count <= 100:
                # Small dataset - return full context
                return self._format_full_context(query_result.results)
            elif result_count <= 1000:
                # Medium dataset - return sample with summary
                return self._format_medium_context(query_result.results)
            else:
                # Large dataset - return aggregated summary
                return self._format_large_context(query_result.results)
                
        except Exception as e:
            logger.error(f"❌ Context retrieval failed: {e}")
            return f"Error retrieving context: {str(e)}"
    
    def _format_full_context(self, results: List[Dict[str, Any]]) -> str:
        """Format complete context for small datasets."""
        logger.info(f"📊 Formatting full context for {len(results)} results")
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            result_summary = []
            for key, value in result.items():
                if value is not None:
                    # Handle different data types appropriately
                    if isinstance(value, list) and len(value) > 5:
                        value_str = f"[{len(value)} items: {', '.join(map(str, value[:3]))}...]"
                    else:
                        value_str = str(value)[:100]  # Limit individual field length
                    result_summary.append(f"  {key}: {value_str}")
            
            formatted_results.append(f"Result {i}:\n" + "\n".join(result_summary))
        
        context = f"Found {len(results)} results:\n\n" + "\n\n".join(formatted_results)
        
        logger.debug(f"📄 Full context length: {len(context)} characters")
        return context
    
    def _format_medium_context(self, results: List[Dict[str, Any]]) -> str:
        """Format summarized context for medium datasets."""
        logger.info(f"📊 Formatting medium context for {len(results)} results")
        
        # Show first 50 results in detail, then summary
        detailed_results = results[:50]
        remaining_count = len(results) - 50
        
        # Format detailed section
        detailed_context = self._format_full_context(detailed_results)
        
        # Add summary for remaining results
        if remaining_count > 0:
            summary_context = f"\n\n[Additional {remaining_count} results not shown - use aggregation queries for full analysis]\n"
            
            # Try to provide some summary statistics
            if results and isinstance(results[0], dict):
                # Count unique values for key fields
                key_stats = {}
                for key in results[0].keys():
                    if key in ['genome_id', 'cazyme_family', 'ko_id', 'bgc_product']:
                        unique_values = set(str(r.get(key, '')) for r in results if r.get(key))
                        if unique_values:
                            key_stats[key] = len(unique_values)
                
                if key_stats:
                    stats_summary = ", ".join([f"{k}: {v} unique" for k, v in key_stats.items()])
                    summary_context += f"Summary statistics: {stats_summary}\n"
            
            context = detailed_context + summary_context
        else:
            context = detailed_context
        
        logger.debug(f"📄 Medium context length: {len(context)} characters")
        return context
    
    def _format_large_context(self, results: List[Dict[str, Any]]) -> str:
        """Format aggregated context for large datasets."""
        logger.info(f"📊 Formatting large context summary for {len(results)} results")
        
        # For large datasets, provide statistical summary
        context_parts = [
            f"Large dataset detected: {len(results)} total results",
            "",
            "📊 STATISTICAL SUMMARY:",
            ""
        ]
        
        if results and isinstance(results[0], dict):
            # Analyze key fields for summary statistics
            sample_result = results[0]
            
            for key in sample_result.keys():
                if key in ['genome_id', 'cazyme_family', 'ko_id', 'bgc_product', 'protein_id']:
                    values = [str(r.get(key, '')) for r in results if r.get(key)]
                    unique_values = set(values)
                    
                    context_parts.append(f"  {key}: {len(unique_values)} unique values")
                    
                    # Show top 5 most common values
                    from collections import Counter
                    value_counts = Counter(values)
                    top_values = value_counts.most_common(5)
                    if top_values:
                        top_str = ", ".join([f"{v}({c})" for v, c in top_values])
                        context_parts.append(f"    Top values: {top_str}")
                    
                    context_parts.append("")
        
        # Add first few examples
        context_parts.extend([
            "🔍 SAMPLE RESULTS (first 5):",
            ""
        ])
        
        for i, result in enumerate(results[:5], 1):
            result_summary = []
            for key, value in result.items():
                if value is not None:
                    value_str = str(value)[:50]  # Shorter for large datasets
                    result_summary.append(f"    {key}: {value_str}")
            context_parts.append(f"  Example {i}:")
            context_parts.extend(result_summary)
            context_parts.append("")
        
        context_parts.extend([
            f"[Showing 5 of {len(results)} total results]",
            "",
            "💡 RECOMMENDATION: Use code interpreter with CSV files or aggregation queries for detailed analysis of large datasets."
        ])
        
        context = "\n".join(context_parts)
        logger.debug(f"📄 Large context summary length: {len(context)} characters")
        return context
    
    async def check_result_size_and_choose_strategy(self, cypher_query: str) -> Dict[str, Any]:
        """
        Check result size and choose appropriate processing strategy.
        
        Args:
            cypher_query: Original Cypher query
            
        Returns:
            Dict with strategy information and size estimates
        """
        logger.info("🔍 Checking result size for strategy selection")
        
        try:
            # Get count estimate by modifying the query
            count_query = convert_to_count_query(cypher_query)
            logger.debug(f"🔢 Count query: {safe_log_data(count_query, 500)}")
            
            count_result = await self.neo4j_processor.process_query(count_query, query_type="cypher")
            
            if count_result.results and len(count_result.results) > 0:
                estimated_count = count_result.results[0].get('estimated_count', 0)
            else:
                logger.warning("⚠️ Count query returned no results, using fallback estimate")
                estimated_count = 0
            
            # Choose strategy based on size
            strategy = self.scaling_router.choose_strategy(estimated_count)
            
            strategy_info = {
                "estimated_count": estimated_count,
                "strategy": strategy.get_strategy_name(),
                "original_query": cypher_query,
                "count_query": count_query,
                "needs_aggregation": estimated_count > 1000,
                "max_proteins_for_code": self.scaling_router.get_protein_limit_for_code(estimated_count)
            }
            
            logger.info(f"📊 Size check complete: {estimated_count} results → {strategy.get_strategy_name()} strategy")
            
            # If large dataset, also prepare aggregated query
            if strategy_info["needs_aggregation"]:
                aggregated_query = convert_to_aggregated_query(cypher_query)
                strategy_info["aggregated_query"] = aggregated_query
                logger.info("📊 Prepared aggregated query for large dataset")
            
            return strategy_info
            
        except Exception as e:
            logger.error(f"❌ Size check failed: {e}")
            return {
                "estimated_count": 0,
                "strategy": "small_dataset", 
                "error": str(e),
                "original_query": cypher_query,
                "needs_aggregation": False,
                "max_proteins_for_code": 100
            }

class ContextFormatter:
    """Specialized formatters for different types of genomic data."""
    
    @staticmethod
    def format_protein_context(proteins: List[Dict[str, Any]]) -> str:
        """Format protein results with genomic context."""
        if not proteins:
            return "No proteins found."
        
        formatted_proteins = []
        for protein in proteins:
            protein_info = []
            
            # Core protein information
            protein_id = protein.get('protein_id', 'unknown')
            protein_info.append(f"Protein: {protein_id}")
            
            # Functional annotations
            if protein.get('ko_description'):
                protein_info.append(f"  Function: {protein.get('ko_description')}")
            if protein.get('ko_id'):
                protein_info.append(f"  KEGG: {protein.get('ko_id')}")
            
            # Genomic location
            if protein.get('start_coordinate') and protein.get('end_coordinate'):
                coords = f"{protein.get('start_coordinate')}-{protein.get('end_coordinate')}"
                if protein.get('strand'):
                    coords += f" ({protein.get('strand')} strand)"
                protein_info.append(f"  Location: {coords}")
            
            # Genome association
            if protein.get('genome_id'):
                protein_info.append(f"  Genome: {protein.get('genome_id')}")
            
            formatted_proteins.append("\n".join(protein_info))
        
        return f"Found {len(proteins)} proteins:\n\n" + "\n\n".join(formatted_proteins)
    
    @staticmethod
    def format_cazyme_context(cazymes: List[Dict[str, Any]]) -> str:
        """Format CAZyme results with family and substrate information."""
        if not cazymes:
            return "No CAZymes found."
        
        # Group by family for better organization
        family_groups = {}
        for cazyme in cazymes:
            family = cazyme.get('cazyme_family', 'Unknown')
            if family not in family_groups:
                family_groups[family] = []
            family_groups[family].append(cazyme)
        
        formatted_sections = []
        for family, family_cazymes in family_groups.items():
            section = [f"CAZyme Family {family} ({len(family_cazymes)} proteins):"]
            
            for cazyme in family_cazymes[:5]:  # Show first 5 per family
                cazyme_info = []
                if cazyme.get('protein_id'):
                    cazyme_info.append(f"  Protein: {cazyme.get('protein_id')}")
                if cazyme.get('substrate'):
                    cazyme_info.append(f"    Substrate: {cazyme.get('substrate')}")
                if cazyme.get('genome_id'):
                    cazyme_info.append(f"    Genome: {cazyme.get('genome_id')}")
                section.extend(cazyme_info)
            
            if len(family_cazymes) > 5:
                section.append(f"  ... and {len(family_cazymes) - 5} more proteins")
            
            formatted_sections.append("\n".join(section))
        
        return f"Found {len(cazymes)} CAZymes in {len(family_groups)} families:\n\n" + "\n\n".join(formatted_sections)
    
    @staticmethod
    def format_bgc_context(bgcs: List[Dict[str, Any]]) -> str:
        """Format BGC results with product and probability information."""
        if not bgcs:
            return "No BGCs found."
        
        formatted_bgcs = []
        for bgc in bgcs:
            bgc_info = []
            
            # Core BGC information
            bgc_id = bgc.get('bgcId', bgc.get('bgc_id', 'unknown'))
            bgc_info.append(f"BGC: {bgc_id}")
            
            # Product prediction
            if bgc.get('bgcProduct') or bgc.get('bgc_product'):
                product = bgc.get('bgcProduct') or bgc.get('bgc_product')
                bgc_info.append(f"  Product: {product}")
            
            # Probability scores
            if bgc.get('averageProbability') or bgc.get('avg_probability'):
                prob = bgc.get('averageProbability') or bgc.get('avg_probability')
                bgc_info.append(f"  Confidence: {float(prob):.3f}")
            
            # Location
            if bgc.get('startCoordinate') and bgc.get('endCoordinate'):
                coords = f"{bgc.get('startCoordinate')}-{bgc.get('endCoordinate')}"
                bgc_info.append(f"  Location: {coords}")
            
            # Genome
            if bgc.get('genome_id') or bgc.get('genomeId'):
                genome = bgc.get('genome_id') or bgc.get('genomeId')
                bgc_info.append(f"  Genome: {genome}")
            
            formatted_bgcs.append("\n".join(bgc_info))
        
        return f"Found {len(bgcs)} BGCs:\n\n" + "\n\n".join(formatted_bgcs)