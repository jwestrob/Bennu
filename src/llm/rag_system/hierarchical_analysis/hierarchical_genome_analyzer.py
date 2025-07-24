"""
Hierarchical Genome Analyzer - Orchestrator for Hierarchical Analysis System.

This component orchestrates the entire hierarchical analysis workflow:
1. Splits large genomic datasets into manageable chunks
2. Analyzes each chunk with GenomicChunkAnalyzer sub-agents
3. Returns curated loci findings directly from LLM analysis
4. Provides detailed analysis of interesting loci

This replaces the broken context-stuffing approach with intelligent
sub-agent analysis that actually answers user questions correctly.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import math

from .genomic_chunk_analyzer import GenomicChunkAnalyzer, InterestingLocus

logger = logging.getLogger(__name__)


@dataclass
class LocusAnalysis:
    """Detailed analysis of an interesting locus."""
    locus: InterestingLocus
    detailed_genes: List[Any]
    functional_predictions: List[str]
    novelty_assessment: str


@dataclass 
class HierarchicalAnalysisResult:
    """Complete result of hierarchical genome analysis."""
    interesting_loci: List[InterestingLocus]
    detailed_analyses: List[LocusAnalysis]
    analysis_summary: Dict[str, Any]
    processing_stats: Dict[str, Any]


class HierarchicalGenomeAnalyzer:
    """
    Orchestrates hierarchical analysis of genomic data.
    
    This is the main orchestrator that coordinates:
    - GenomicChunkAnalyzer sub-agents for chunk analysis
    - Direct collection of LLM-identified interesting loci
    - Detailed analysis of interesting loci
    
    Replaces brute-force context stuffing with intelligent hierarchical workflow.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initialize the hierarchical genome analyzer.
        
        Args:
            model_name: LLM model to use for analysis
        """
        self.model_name = model_name
        self.chunk_analyzer = GenomicChunkAnalyzer(model_name)
        
        # Configuration for chunking strategy
        self.chunking_config = {
            "target_chunk_size": 1000,      # genes per chunk
            "min_chunk_size": 500,          # minimum genes per chunk
            "max_chunk_size": 1500,         # maximum genes per chunk
            "overlap_size": 50              # gene overlap between chunks
        }
        
        logger.info(f"🏗️ HierarchicalGenomeAnalyzer initialized with {model_name}")
    
    def analyze_genome_hierarchically(self, genome_contexts: List[Any], 
                                    user_question: str) -> HierarchicalAnalysisResult:
        """
        Perform hierarchical analysis of genomic data.
        
        Args:
            genome_contexts: List of GenomeContext objects containing genomic data
            user_question: Original user question to guide analysis
            
        Returns:
            HierarchicalAnalysisResult with curated loci findings
        """
        try:
            logger.info(f"🚀 Starting hierarchical analysis for {len(genome_contexts)} genomes")
            
            # Phase 1: Intelligent chunking
            chunks = self._create_biological_chunks(genome_contexts)
            logger.info(f"📊 Created {len(chunks)} biological chunks for analysis")
            
            # Phase 2: Sub-agent chunk analysis
            all_candidate_loci = []
            chunk_stats = {"successful": 0, "failed": 0, "total_loci": 0}
            
            for i, chunk in enumerate(chunks):
                try:
                    chunk_loci = self.chunk_analyzer.analyze_genomic_chunk(
                        chunk_data=chunk,
                        user_question=user_question
                    )
                    all_candidate_loci.extend(chunk_loci)
                    chunk_stats["successful"] += 1
                    chunk_stats["total_loci"] += len(chunk_loci)
                    
                    logger.info(f"✅ Chunk {i+1}/{len(chunks)}: {len(chunk_loci)} loci identified")
                    
                except Exception as e:
                    logger.warning(f"❌ Chunk {i+1} analysis failed: {e}")
                    chunk_stats["failed"] += 1
            
            # Phase 3: Take all interesting loci (LLM already identified what's interesting)
            interesting_loci = all_candidate_loci
            
            logger.info(f"🎯 Selected {len(interesting_loci)} interesting loci from {len(all_candidate_loci)} candidates")
            
            # Phase 4: Detailed analysis of interesting loci
            detailed_analyses = []
            for locus in interesting_loci:
                try:
                    detailed_analysis = self._create_detailed_locus_analysis(
                        locus, genome_contexts, user_question
                    )
                    if detailed_analysis:
                        detailed_analyses.append(detailed_analysis)
                except Exception as e:
                    logger.warning(f"Failed to create detailed analysis for locus: {e}")
            
            # Phase 5: Generate analysis summary
            analysis_summary = self._generate_analysis_summary(
                interesting_loci, all_candidate_loci, user_question
            )
            
            # Compile processing statistics
            processing_stats = {
                "total_chunks": len(chunks),
                "successful_chunks": chunk_stats["successful"],
                "failed_chunks": chunk_stats["failed"],
                "total_candidate_loci": len(all_candidate_loci),
                "interesting_loci": len(interesting_loci),
                "detailed_analyses": len(detailed_analyses)
            }
            
            result = HierarchicalAnalysisResult(
                interesting_loci=interesting_loci,
                detailed_analyses=detailed_analyses,
                analysis_summary=analysis_summary,
                processing_stats=processing_stats
            )
            
            logger.info(f"🏁 Hierarchical analysis complete: {len(interesting_loci)} loci delivered")
            return result
            
        except Exception as e:
            logger.error(f"❌ Hierarchical analysis failed: {e}")
            # Return empty result rather than crashing
            return HierarchicalAnalysisResult(
                interesting_loci=[],
                detailed_analyses=[],
                analysis_summary={"error": str(e)},
                processing_stats={"failed": True}
            )
    
    def _create_biological_chunks(self, genome_contexts: List[Any]) -> List[Dict[str, Any]]:
        """
        Create biologically-aware chunks of genomic data.
        
        Args:
            genome_contexts: List of GenomeContext objects
            
        Returns:
            List of chunk dictionaries for sub-agent analysis
        """
        chunks = []
        
        try:
            for genome_ctx in genome_contexts:
                genome_id = getattr(genome_ctx, 'genome_id', 'unknown')
                contigs = getattr(genome_ctx, 'contigs', [])
                
                # Strategy 1: Chunk by contigs (preserve biological boundaries)
                contig_chunks = self._chunk_by_contigs(contigs, genome_ctx)
                chunks.extend(contig_chunks)
                
                logger.debug(f"📋 Genome {genome_id}: {len(contig_chunks)} contig-based chunks")
        
        except Exception as e:
            logger.error(f"Error creating biological chunks: {e}")
        
        return chunks
    
    def _chunk_by_contigs(self, contigs: List[Any], genome_ctx: Any) -> List[Dict[str, Any]]:
        """Chunk genomic data by contigs to preserve biological boundaries."""
        chunks = []
        
        try:
            # Group small contigs together, keep large contigs separate
            current_chunk_contigs = []
            current_gene_count = 0
            
            for contig in contigs:
                contig_gene_count = getattr(contig, 'total_genes', 0)
                
                # If this contig alone exceeds target size, make it its own chunk
                if contig_gene_count >= self.chunking_config["target_chunk_size"]:
                    # Finalize current chunk if it exists
                    if current_chunk_contigs:
                        chunk = self._create_chunk_dict(current_chunk_contigs, genome_ctx)
                        chunks.append(chunk)
                        current_chunk_contigs = []
                        current_gene_count = 0
                    
                    # Create chunk for large contig
                    large_contig_chunk = self._create_chunk_dict([contig], genome_ctx)
                    chunks.append(large_contig_chunk)
                
                # If adding this contig would exceed max size, finalize current chunk
                elif (current_gene_count + contig_gene_count > self.chunking_config["max_chunk_size"] 
                      and current_chunk_contigs):
                    chunk = self._create_chunk_dict(current_chunk_contigs, genome_ctx)
                    chunks.append(chunk)
                    current_chunk_contigs = [contig]
                    current_gene_count = contig_gene_count
                
                # Add contig to current chunk
                else:
                    current_chunk_contigs.append(contig)
                    current_gene_count += contig_gene_count
            
            # Don't forget the last chunk
            if current_chunk_contigs:
                chunk = self._create_chunk_dict(current_chunk_contigs, genome_ctx)
                chunks.append(chunk)
        
        except Exception as e:
            logger.warning(f"Error chunking by contigs: {e}")
        
        return chunks
    
    def _create_chunk_dict(self, contigs: List[Any], genome_ctx: Any) -> Dict[str, Any]:
        """Create a chunk dictionary for sub-agent analysis."""
        try:
            # Calculate chunk statistics
            total_genes = sum(getattr(contig, 'total_genes', 0) for contig in contigs)
            total_length = sum(getattr(contig, 'length', 0) for contig in contigs)
            
            # Create simplified genome context for this chunk
            chunk_genome_ctx = type('ChunkGenomeContext', (), {
                'genome_id': getattr(genome_ctx, 'genome_id', 'unknown'),
                'contigs': contigs,
                'total_genes': total_genes,
                'total_contigs': len(contigs)
            })()
            
            return {
                "genome_contexts": [chunk_genome_ctx],
                "chunk_stats": {
                    "total_genes": total_genes,
                    "total_contigs": len(contigs),
                    "total_length": total_length
                }
            }
        
        except Exception as e:
            logger.warning(f"Error creating chunk dict: {e}")
            return {"genome_contexts": [], "chunk_stats": {}}
    
    def _create_detailed_locus_analysis(self, locus: InterestingLocus, 
                                      genome_contexts: List[Any],
                                      user_question: str) -> Optional[LocusAnalysis]:
        """Create detailed analysis of an interesting locus."""
        try:
            # Find the genes within this locus
            detailed_genes = self._extract_locus_genes(locus, genome_contexts)
            
            # Generate functional predictions
            functional_predictions = self._predict_locus_function(locus, detailed_genes, user_question)
            
            return LocusAnalysis(
                locus=locus,
                detailed_genes=detailed_genes,
                functional_predictions=functional_predictions,
                novelty_assessment=""
            )
        
        except Exception as e:
            logger.warning(f"Error creating detailed locus analysis: {e}")
            return None
    
    def _extract_locus_genes(self, locus: InterestingLocus, genome_contexts: List[Any]) -> List[Any]:
        """Extract the actual gene objects within a locus region."""
        genes_in_locus = []
        
        try:
            # Find the genome containing this locus
            target_genome = None
            for genome_ctx in genome_contexts:
                contigs = getattr(genome_ctx, 'contigs', [])
                for contig in contigs:
                    if getattr(contig, 'contig_id', '') == locus.contig_id:
                        target_genome = genome_ctx
                        break
                if target_genome:
                    break
            
            if not target_genome:
                logger.warning(f"Could not find genome for locus {locus.genomic_coordinates}")
                return []
            
            # Extract genes within locus boundaries
            contigs = getattr(target_genome, 'contigs', [])
            for contig in contigs:
                if getattr(contig, 'contig_id', '') == locus.contig_id:
                    plus_genes = getattr(contig, 'plus_strand_genes', [])
                    minus_genes = getattr(contig, 'minus_strand_genes', [])
                    
                    for gene in plus_genes + minus_genes:
                        gene_start = getattr(gene, 'start', 0)
                        gene_end = getattr(gene, 'end', 0)
                        
                        # Check if gene overlaps with locus
                        if (gene_start <= locus.end and gene_end >= locus.start):
                            genes_in_locus.append(gene)
                    break
        
        except Exception as e:
            logger.warning(f"Error extracting locus genes: {e}")
        
        return genes_in_locus
    
    def _predict_locus_function(self, locus: InterestingLocus, genes: List[Any], 
                              user_question: str) -> List[str]:
        """No hardcoded functional predictions - let the LLM analyze what's interesting."""
        return []
    
    def _assess_locus_novelty(self, locus: InterestingLocus, genes: List[Any]) -> str:
        """No hardcoded novelty assessment - let the LLM judge what's interesting."""
        return ""
    
    def _generate_analysis_summary(self, interesting_loci: List[InterestingLocus],
                                 all_candidates: List[InterestingLocus],
                                 user_question: str) -> Dict[str, Any]:
        """Generate a summary of the hierarchical analysis results."""
        try:
            if not interesting_loci:
                return {
                    "message": "No interesting loci identified",
                    "total_candidates_screened": len(all_candidates)
                }
            
            # Loci type distribution
            loci_types = {}
            for locus in interesting_loci:
                locus_type = locus.locus_type
                loci_types[locus_type] = loci_types.get(locus_type, 0) + 1
            
            # Size statistics
            gene_counts = [locus.gene_count for locus in interesting_loci]
            total_genes_analyzed = sum(gene_counts)
            
            # Hypothetical protein analysis
            total_hypothetical = sum(locus.hypothetical_count for locus in interesting_loci)
            hypothetical_percentages = [(locus.hypothetical_count / locus.gene_count * 100) if locus.gene_count > 0 else 0 for locus in interesting_loci]
            avg_hypothetical_pct = sum(hypothetical_percentages) / len(interesting_loci) if interesting_loci else 0
            
            summary = {
                "analysis_approach": "Hierarchical sub-agent analysis",
                "total_candidates_screened": len(all_candidates),
                "interesting_loci_count": len(interesting_loci),
                "loci_type_distribution": loci_types,
                "genomic_coverage": {
                    "total_genes_in_loci": total_genes_analyzed,
                    "min_locus_size": min(gene_counts) if gene_counts else 0,
                    "max_locus_size": max(gene_counts) if gene_counts else 0,
                    "avg_locus_size": total_genes_analyzed / len(interesting_loci) if interesting_loci else 0
                },
                "hypothetical_protein_analysis": {
                    "total_hypothetical_proteins": total_hypothetical,
                    "average_hypothetical_percentage": avg_hypothetical_pct
                },
                "user_question_context": user_question[:100] + "..." if len(user_question) > 100 else user_question
            }
            
            return summary
        
        except Exception as e:
            logger.warning(f"Error generating analysis summary: {e}")
            return {"error": f"Summary generation failed: {e}"}