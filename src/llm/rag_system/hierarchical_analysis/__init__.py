"""
Hierarchical Analysis System for Genomic Data.

This package implements the hierarchical analysis architecture that replaces
brute-force context stuffing with intelligent sub-agent analysis.

Components:
- GenomicChunkAnalyzer: Analyzes genomic chunks to identify interesting loci
- HierarchicalGenomeAnalyzer: Orchestrates the entire hierarchical workflow
"""

from .genomic_chunk_analyzer import GenomicChunkAnalyzer, InterestingLocus
from .hierarchical_genome_analyzer import HierarchicalGenomeAnalyzer, LocusAnalysis, HierarchicalAnalysisResult

__all__ = [
    'GenomicChunkAnalyzer',
    'InterestingLocus', 
    'HierarchicalGenomeAnalyzer',
    'LocusAnalysis',
    'HierarchicalAnalysisResult'
]