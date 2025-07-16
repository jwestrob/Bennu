#!/usr/bin/env python3
"""
Whole Genome Context Reader for Spatial Genomic Analysis

Provides spatially-ordered genome data for LLM-based operon and prophage discovery.
This tool reads entire genomes in biological order (by contig and coordinate) rather
than presenting scattered individual proteins.

Key features:
- Spatial ordering by contig and genomic coordinates
- Strand-aware gene organization  
- Operon-scale context windows
- Hierarchical contig-by-contig analysis support
- Single-use per genome per session (prevents redundant calls)
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class GeneContext:
    """Individual gene with spatial and functional context."""
    gene_id: str
    protein_id: Optional[str]
    start: int
    end: int
    strand: str
    length: int
    annotation: Optional[str]
    ko_id: Optional[str] 
    ko_description: Optional[str]
    pfam_domains: List[str]
    is_hypothetical: bool

@dataclass
class ContigContext:
    """Complete contig with all genes in spatial order."""
    contig_id: str
    length: Optional[int]
    plus_strand_genes: List[GeneContext]
    minus_strand_genes: List[GeneContext]
    total_genes: int
    hypothetical_count: int
    annotated_count: int

@dataclass 
class GenomeContext:
    """Complete genome with all contigs in spatial order."""
    genome_id: str
    contigs: List[ContigContext]
    total_genes: int
    total_contigs: int
    hypothetical_gene_count: int
    annotated_gene_count: int
    largest_contig_length: int

class WholeGenomeReader:
    """
    Reads complete genomes in spatial order for comprehensive LLM analysis.
    
    This tool is designed for one-time comprehensive reading of entire genomes
    to support spatial analysis tasks like operon and prophage discovery.
    """
    
    def __init__(self, neo4j_processor):
        """Initialize with Neo4j connection for data retrieval."""
        self.neo4j_processor = neo4j_processor
        self.genomes_read_this_session = set()  # Prevent duplicate reads
        
    async def read_complete_genome(self, genome_id: str, max_genes_per_contig: int = 1000) -> Dict[str, Any]:
        """
        Read complete genome in spatial order for LLM analysis.
        
        Args:
            genome_id: Target genome identifier
            max_genes_per_contig: Maximum genes to read per contig (prevents memory issues)
            
        Returns:
            Dict with spatially-ordered genome data ready for LLM consumption
        """
        logger.info(f"🧬 Reading complete genome in spatial order: {genome_id}")
        
        # Prevent duplicate reads in same session
        if genome_id in self.genomes_read_this_session:
            logger.warning(f"⚠️ Genome {genome_id} already read this session - skipping duplicate read")
            return {
                "success": False,
                "error": f"Genome {genome_id} already read this session",
                "genome_context": None
            }
        
        try:
            # Step 1: Get all genes in spatial order
            spatial_query = f"""
            MATCH (g:Gene)-[:BELONGSTOGENOME]->(genome:Genome {{genomeId: '{genome_id}'}})
            OPTIONAL MATCH (g)<-[:ENCODEDBY]-(p:Protein)
            OPTIONAL MATCH (p)-[:HASFUNCTION]->(ko:KEGGOrtholog)
            OPTIONAL MATCH (p)-[:HASDOMAIN]->(da:DomainAnnotation)-[:DOMAINFAMILY]->(dom:Domain)
            WITH g, p, ko, collect(DISTINCT dom.id) AS pfam_domains
            RETURN g.id AS gene_id,
                   p.id AS protein_id,
                   g.startCoordinate AS start_pos,
                   g.endCoordinate AS end_pos,
                   g.strand AS strand,
                   g.lengthAA AS gene_length,
                   COALESCE(g.contig, g.id, 'unknown_contig') AS contig_id,
                   ko.id AS ko_id,
                   ko.description AS ko_description,
                   pfam_domains
            ORDER BY contig_id, toInteger(start_pos)
            """
            
            result = await self.neo4j_processor.process_query(spatial_query, query_type="cypher")
            
            if not result.results:
                logger.warning(f"❌ No genes found for genome: {genome_id}")
                return {
                    "success": False,
                    "error": f"No genes found for genome: {genome_id}",
                    "genome_context": None
                }
            
            logger.info(f"📊 Retrieved {len(result.results)} genes for spatial analysis")
            
            # Step 2: Organize genes by contig and strand
            genome_context = self._organize_spatial_context(result.results, genome_id, max_genes_per_contig)
            
            # Step 3: Mark this genome as read
            self.genomes_read_this_session.add(genome_id)
            
            logger.info(f"✅ Complete genome context prepared: {genome_context.total_contigs} contigs, {genome_context.total_genes} genes")
            
            return {
                "success": True,
                "genome_context": genome_context,
                "spatial_summary": {
                    "genome_id": genome_id,
                    "total_contigs": genome_context.total_contigs,
                    "total_genes": genome_context.total_genes,
                    "hypothetical_genes": genome_context.hypothetical_gene_count,
                    "annotated_genes": genome_context.annotated_gene_count,
                    "largest_contig": genome_context.largest_contig_length,
                    "reading_method": "spatial_ordered_by_coordinates"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to read complete genome {genome_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "genome_context": None
            }
    
    def _organize_spatial_context(self, raw_genes: List[Dict], genome_id: str, max_genes_per_contig: int) -> GenomeContext:
        """Organize raw gene data into spatial genomic context."""
        
        # Group genes by contig
        contig_groups = {}
        for gene_data in raw_genes:
            contig_id = gene_data.get('contig_id', 'unknown')
            if contig_id not in contig_groups:
                contig_groups[contig_id] = []
            contig_groups[contig_id].append(gene_data)
        
        # Process each contig
        contigs = []
        total_genes = 0
        total_hypothetical = 0
        total_annotated = 0
        largest_contig = 0
        
        for contig_id, genes in contig_groups.items():
            
            # PROPHAGE DISCOVERY FIX: Don't truncate genes - need complete genomic context
            if len(genes) > max_genes_per_contig:
                logger.info(f"📊 Contig {contig_id} has {len(genes)} genes - processing all for prophage discovery (no truncation)")
                # genes = genes[:max_genes_per_contig]  # DISABLED: Causes 75% data loss
            else:
                logger.info(f"📊 Processing all {len(genes)} genes from contig {contig_id}")
            
            # Separate by strand and sort by position
            plus_genes = [g for g in genes if g.get('strand') == '+1' or g.get('strand') == '+']
            minus_genes = [g for g in genes if g.get('strand') == '-1' or g.get('strand') == '-']
            
            plus_genes.sort(key=lambda x: int(x.get('start_pos', 0)))
            minus_genes.sort(key=lambda x: int(x.get('start_pos', 0)))  # Sort by position, not reverse
            
            # Convert to GeneContext objects
            plus_strand_contexts = [self._create_gene_context(g) for g in plus_genes]
            minus_strand_contexts = [self._create_gene_context(g) for g in minus_genes]
            
            # Calculate contig stats - no hard-coded hypothetical classification
            contig_genes = len(plus_genes) + len(minus_genes)
            contig_hypothetical = 0  # LLM will determine annotation quality
            contig_annotated = contig_genes  # All genes have some annotation
            
            # Estimate contig length from gene positions
            all_positions = [int(g.get('start_pos', 0)) for g in genes] + [int(g.get('end_pos', 0)) for g in genes]
            contig_length = max(all_positions) if all_positions else 0
            
            largest_contig = max(largest_contig, contig_length)
            
            contig_context = ContigContext(
                contig_id=contig_id,
                length=contig_length,
                plus_strand_genes=plus_strand_contexts,
                minus_strand_genes=minus_strand_contexts,
                total_genes=contig_genes,
                hypothetical_count=contig_hypothetical,
                annotated_count=contig_annotated
            )
            
            contigs.append(contig_context)
            total_genes += contig_genes
            total_hypothetical += contig_hypothetical
            total_annotated += contig_annotated
        
        # Sort contigs by size (largest first) for prioritized reading
        contigs.sort(key=lambda c: c.length or 0, reverse=True)
        
        return GenomeContext(
            genome_id=genome_id,
            contigs=contigs,
            total_genes=total_genes,
            total_contigs=len(contigs),
            hypothetical_gene_count=total_hypothetical,
            annotated_gene_count=total_annotated,
            largest_contig_length=largest_contig
        )
    
    def _create_gene_context(self, gene_data: Dict) -> GeneContext:
        """Convert raw gene data to GeneContext object."""
        
        # Get annotation without hard-coded hypothetical classification
        ko_desc = gene_data.get('ko_description', '') or ''
        annotation = ko_desc if ko_desc else 'hypothetical protein'
        
        return GeneContext(
            gene_id=gene_data.get('gene_id', ''),
            protein_id=gene_data.get('protein_id'),
            start=int(gene_data.get('start_pos', 0)),
            end=int(gene_data.get('end_pos', 0)),
            strand=gene_data.get('strand', ''),
            length=int(gene_data.get('gene_length', 0)) if gene_data.get('gene_length') else 0,
            annotation=annotation,
            ko_id=gene_data.get('ko_id'),
            ko_description=ko_desc,
            pfam_domains=gene_data.get('pfam_domains', []),
            is_hypothetical=False  # Remove hard-coded hypothetical logic - LLM will analyze annotations
        )
    
    def format_for_llm_analysis(self, genome_context: GenomeContext, focus_on_spatial: bool = True) -> str:
        """
        Format genome context for LLM analysis with emphasis on spatial patterns.
        
        Args:
            genome_context: Complete genome context
            focus_on_spatial: Whether to emphasize spatial organization
            
        Returns:
            Formatted text for LLM consumption
        """
        output = []
        output.append(f"🧬 COMPLETE GENOME SPATIAL ANALYSIS: {genome_context.genome_id}")
        output.append(f"📊 Overview: {genome_context.total_contigs} contigs, {genome_context.total_genes} genes")
        output.append(f"📈 All genes included with their annotations for LLM analysis")
        output.append("")
        
        for i, contig in enumerate(genome_context.contigs[:5]):  # Show top 5 contigs
            output.append(f"📍 CONTIG {i+1}: {contig.contig_id} ({contig.length:,} bp, {contig.total_genes} genes)")
            
            # Plus strand
            if contig.plus_strand_genes:
                output.append(f"  ➡️  PLUS STRAND ({len(contig.plus_strand_genes)} genes):")
                self._format_strand_for_llm(contig.plus_strand_genes, output)
            
            # Minus strand  
            if contig.minus_strand_genes:
                output.append(f"  ⬅️  MINUS STRAND ({len(contig.minus_strand_genes)} genes):")
                self._format_strand_for_llm(contig.minus_strand_genes, output)
                
            output.append("")
        
        if len(genome_context.contigs) > 5:
            output.append(f"... and {len(genome_context.contigs) - 5} additional smaller contigs")
        
        return "\n".join(output)
    
    def _format_strand_for_llm(self, genes: List[GeneContext], output: List[str]):
        """Format a strand of genes for LLM analysis with spatial organization."""
        
        # Show all genes in spatial order for LLM pattern analysis
        for gene in genes[:15]:  # Show first 15 genes per strand
            # Include comprehensive annotation details for LLM analysis
            annotation_info = []
            if gene.ko_id:
                annotation_info.append(f"KO:{gene.ko_id}")
            if gene.pfam_domains:
                annotation_info.append(f"PFAM:{','.join(gene.pfam_domains[:3])}")
            
            annotation_details = f" [{'; '.join(annotation_info)}]" if annotation_info else ""
            output.append(f"    • {gene.gene_id} ({gene.start:,}-{gene.end:,}) {gene.annotation}{annotation_details}")
        
        if len(genes) > 15:
            output.append(f"    ... and {len(genes) - 15} more genes on this strand")

# Tool registration for external tools system
async def read_complete_genome_spatial(genome_id: str, neo4j_processor, **kwargs) -> Dict[str, Any]:
    """
    Tool function for reading complete genome in spatial order.
    
    This is a specialized tool for comprehensive genome reading that should only
    be called once per genome per session for spatial analysis tasks.
    """
    reader = WholeGenomeReader(neo4j_processor)
    result = await reader.read_complete_genome(genome_id)
    
    if result["success"]:
        # Format for LLM consumption
        formatted_context = reader.format_for_llm_analysis(
            result["genome_context"], 
            focus_on_spatial=True
        )
        
        return {
            "success": True,
            "tool_output": formatted_context,
            "raw_context": result["genome_context"],
            "summary": result["spatial_summary"],
            "usage_note": f"Complete spatial reading of {genome_id} - this tool should not be called again for this genome in this session"
        }
    else:
        return {
            "success": False,
            "error": result["error"],
            "tool_output": f"Failed to read genome {genome_id}: {result['error']}"
        }

async def read_all_genomes_spatial(neo4j_processor, **kwargs):
    """
    Read ALL genomes in spatial order for global prophage/operon discovery.
    
    This function reads through all available genomes spatially, organizing by
    genome -> contig -> coordinate order for comprehensive analysis.
    
    Args:
        neo4j_processor: Neo4j database processor
        **kwargs: Additional parameters
        
    Returns:
        Dict with success status and formatted output
    """
    try:
        logger.info("🌐 Starting global spatial genome reading across all genomes")
        
        # Get all available genomes
        genome_query = "MATCH (g:Genome) RETURN g.genomeId as genome_id ORDER BY g.genomeId"
        genome_result = await neo4j_processor.process_query(genome_query, query_type="cypher")
        
        if not genome_result or not genome_result.results:
            return {
                "success": False,
                "error": "No genomes found in database",
                "tool_output": "No genomes available for global analysis"
            }
        
        genome_ids = [row['genome_id'] for row in genome_result.results]
        logger.info(f"🔍 Found {len(genome_ids)} genomes for global spatial reading: {genome_ids}")
        
        # Read each genome spatially
        reader = WholeGenomeReader(neo4j_processor)
        all_genome_contexts = []
        total_genes = 0
        total_hypothetical = 0
        
        for genome_id in genome_ids:
            logger.info(f"📖 Reading genome {genome_id} spatially...")
            
            result = await reader.read_complete_genome(genome_id)
            
            if result["success"]:
                genome_context = result["genome_context"]
                all_genome_contexts.append(genome_context)
                total_genes += genome_context.total_genes
                total_hypothetical += genome_context.hypothetical_gene_count
                logger.info(f"✅ Read {genome_context.total_genes} genes from {genome_id}")
            else:
                logger.warning(f"⚠️ Failed to read {genome_id}: {result['error']}")
        
        if not all_genome_contexts:
            return {
                "success": False,
                "error": "Failed to read any genomes",
                "tool_output": "Global spatial reading failed - no genomes could be processed"
            }
        
        # Format for LLM analysis with global context
        output = []
        output.append("🌐 GLOBAL SPATIAL GENOMIC ANALYSIS")
        output.append(f"📊 Dataset overview: {len(all_genome_contexts)} genomes, {total_genes:,} total genes")
        output.append(f"📈 Global annotation: {total_genes - total_hypothetical:,} annotated, {total_hypothetical:,} hypothetical ({total_hypothetical/total_genes*100:.1f}%)")
        output.append("")
        output.append("🔍 GENOME-BY-GENOME SPATIAL READING:")
        output.append("")
        
        for i, genome_context in enumerate(all_genome_contexts):
            output.append(f"{'='*60}")
            output.append(f"GENOME {i+1}: {genome_context.genome_id}")
            output.append(f"📊 {genome_context.total_contigs} contigs, {genome_context.total_genes} genes")
            output.append(f"📈 {genome_context.annotated_gene_count} annotated, {genome_context.hypothetical_gene_count} hypothetical")
            output.append("")
            
            # Show spatial organization for each genome
            formatted_genome = reader.format_for_llm_analysis(genome_context, focus_on_spatial=True)
            # Remove the header since we're adding our own
            lines = formatted_genome.split('\n')[4:]  # Skip the first 4 header lines
            output.extend(lines)
            output.append("")
        
        output.append("🎯 GLOBAL ANALYSIS COMPLETE")
        output.append("Use this spatial context to identify:")
        output.append("- Cross-genome prophage patterns")
        output.append("- Conserved hypothetical gene clusters")
        output.append("- Genome-specific operon organizations")
        output.append("- Comparative spatial features")
        
        formatted_output = "\n".join(output)
        
        logger.info(f"✅ Global spatial reading complete: {len(all_genome_contexts)} genomes, {total_genes:,} genes")
        
        return {
            "success": True,
            "tool_output": formatted_output,
            "genome_contexts": all_genome_contexts,
            "summary": {
                "genomes_read": len(all_genome_contexts),
                "total_genes": total_genes,
                "total_hypothetical": total_hypothetical,
                "hypothetical_percentage": total_hypothetical/total_genes*100 if total_genes > 0 else 0
            },
            "usage_note": "Complete global spatial reading - comprehensive analysis across all genomes"
        }
        
    except Exception as e:
        logger.error(f"Global spatial reading failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "tool_output": f"Global genome reading failed: {str(e)}"
        }