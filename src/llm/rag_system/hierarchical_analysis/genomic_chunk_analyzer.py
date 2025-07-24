"""
Genomic Chunk Analyzer for Hierarchical Analysis System.

This component analyzes chunks of genomic data to identify regions of interest,
replacing the brute-force context stuffing approach with intelligent sub-agent analysis.

Key Features:
- Analyzes genomic chunks of variable size to identify interesting loci
- Uses LLM-guided analysis based on user questions, not hardcoded keywords
- Returns structured InterestingLocus objects with biological significance
- Enables hierarchical analysis instead of raw data dumping
"""

import logging
from dataclasses import dataclass
from typing import List, Any, Optional, Dict
import dspy

logger = logging.getLogger(__name__)


@dataclass
class InterestingLocus:
    """Represents a genomically interesting region identified by analysis."""
    contig_id: str
    start: int
    end: int
    gene_count: int
    hypothetical_count: int
    biological_features: List[str]
    flanking_genes: List[str]
    locus_type: str  # "gene_cluster", "single_protein", or LLM-determined type
    
    @property
    def genomic_coordinates(self) -> str:
        """Human-readable genomic coordinates."""
        return f"{self.contig_id}:{self.start}-{self.end}"
    
    @property
    def hypothetical_percentage(self) -> float:
        """Percentage of genes that are hypothetical."""
        if self.gene_count == 0:
            return 0.0
        return (self.hypothetical_count / self.gene_count) * 100


class GenomicRegionIdentifier(dspy.Signature):
    """Identify interesting genomic regions from chunk analysis."""
    
    genomic_chunk_data = dspy.InputField(desc="Structured genomic data for a chunk of genes with coordinates, annotations, and hypothetical status")
    analysis_criteria = dspy.InputField(desc="User question and criteria for identifying interesting regions")
    
    interesting_regions = dspy.OutputField(desc="JSON list of interesting loci, each with: contig_id, start_coordinate, end_coordinate, gene_ids (array of individual gene_id strings from input data), locus_type, biological_significance_reasoning. CRITICAL: gene_ids must be an array of individual gene_id values EXACTLY as they appear in the input data. Do NOT append coordinates or create composite identifiers. Each gene_id should be a separate string in the array.")


class GenomicChunkAnalyzer:
    """
    Analyzes chunks of genomic data to identify biologically interesting regions.
    
    This is a core component of the hierarchical analysis system that replaces
    brute-force context stuffing with intelligent loci identification based on
    user questions rather than hardcoded keyword filters.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initialize the genomic chunk analyzer.
        
        Args:
            model_name: LLM model to use for analysis
        """
        self.model_name = model_name
        self.analyzer = dspy.Predict(GenomicRegionIdentifier)
        
        # No more hardcoded parameters - let the LLM decide what's interesting
        
        logger.info(f"🧬 GenomicChunkAnalyzer initialized with {model_name}")
    
    def analyze_genomic_chunk(self, chunk_data: Dict[str, Any], 
                            user_question: str) -> List[InterestingLocus]:
        """
        Analyze a chunk of genomic data to identify regions relevant to user question.
        
        Args:
            chunk_data: Structured genomic data (subset of genome_contexts)
            user_question: The original user question to guide analysis
            
        Returns:
            List of InterestingLocus objects representing interesting regions
        """
        
        try:
            # Extract contigs from chunk data
            contigs = self._extract_contigs_from_chunk(chunk_data)
            if not contigs:
                logger.warning("No contigs found in chunk data")
                return []
            
            # Use LLM to analyze genomic data in context of user question
            interesting_loci = self._llm_guided_analysis(contigs, user_question)
                        
            # Return all interesting loci (let prioritizer handle filtering)
            # No arbitrary score thresholds - biological data is too messy for hardcoded cutoffs
            
            logger.info(f"🎯 Identified {len(interesting_loci)} interesting loci in genomic chunk")
            return interesting_loci
            
        except Exception as e:
            logger.error(f"❌ Error analyzing genomic chunk: {e}")
            return []
    
    def _extract_contigs_from_chunk(self, chunk_data: Dict[str, Any]) -> List[Any]:
        """Extract contig objects from chunk data structure."""
        contigs = []
        
        try:
            # Handle different chunk data structures
            if "genome_contexts" in chunk_data:
                for genome_ctx in chunk_data["genome_contexts"]:
                    if hasattr(genome_ctx, 'contigs'):
                        contigs.extend(genome_ctx.contigs)
            elif "contigs" in chunk_data:
                contigs = chunk_data["contigs"]
            elif "genome_context" in chunk_data:
                genome_ctx = chunk_data["genome_context"]
                if hasattr(genome_ctx, 'contigs'):
                    contigs = genome_ctx.contigs
                    
        except Exception as e:
            logger.warning(f"Error extracting contigs: {e}")
        
        return contigs
    
    def _llm_guided_analysis(self, contigs: List[Any], user_question: str) -> List[InterestingLocus]:
        """Use LLM to identify interesting loci based on user question and genomic data."""
        try:
            # Prepare structured genomic data for LLM analysis
            genomic_data = self._prepare_structured_genomic_data(contigs)
            
            # Create analysis criteria based on user question
            analysis_criteria = f"""
            User question: {user_question}
            
            Please identify genomically interesting regions. For each region, provide:
            - contig_id: The contig name where the region is located (NOT a gene ID - just the bare contig identifier)
            - start_coordinate: genomic start position of the region
            - end_coordinate: genomic end position of the region  
            - gene_ids: array of individual gene_id strings from the input data
            - locus_type: description of what makes this region interesting
            - biological_significance_reasoning: why this region is relevant to the user's question
            
            CRITICAL INSTRUCTIONS FOR gene_ids:
            1. gene_ids should be an array like ["gene:ABC_001", "gene:ABC_002", "gene:ABC_003"]
            2. Each entry should be an individual gene_id EXACTLY as it appears in the input data
            3. Do NOT append coordinates to gene IDs (NO ":0-12000" or similar suffixes)
            4. Do NOT create synthetic identifiers by combining gene IDs with coordinate ranges
            5. If a region contains 3 genes, list all 3 individual gene_id values separately
            
            EXAMPLE of CORRECT output format:
            [
              {{
                "contig_id": "scaffold_1",
                "start_coordinate": 1000,
                "end_coordinate": 5000,
                "gene_ids": ["gene:scaffold_1_001", "gene:scaffold_1_002", "gene:scaffold_1_003"],
                "locus_type": "prophage candidate",
                "biological_significance_reasoning": "Contains multiple hypothetical proteins"
              }}
            ]
            
            EXAMPLE of INCORRECT format (DO NOT DO THIS):
            [
              {{
                "gene_ids": ["gene:scaffold_1_001:0-5000"]
              }}
            ]
            
            Focus on the user's specific question and biological context.
            """
            
            # Use DSPy signature to get LLM analysis
            result = self.analyzer(
                genomic_chunk_data=genomic_data,
                analysis_criteria=analysis_criteria
            )
            
            # Parse LLM response into InterestingLocus objects
            interesting_loci = self._parse_structured_llm_response(result.interesting_regions, contigs)
            
            return interesting_loci
            
        except Exception as e:
            logger.error(f"Error in LLM-guided analysis: {e}")
            # If LLM analysis fails, return empty list rather than broken fallback
            logger.warning("LLM analysis failed - returning no results to avoid hardcoded assumptions")
            return []
    
    def _prepare_structured_genomic_data(self, contigs: List[Any]) -> str:
        """Prepare structured genomic data with contig-based chunking for biological integrity."""
        try:
            structured_data = []
            total_genes_processed = 0
            max_genes_per_chunk = 200  # Conservative limit to avoid token overflow
            
            for contig in contigs:
                contig_id = getattr(contig, 'contig_id', 'unknown')
                
                # Collect all genes from both strands
                plus_genes = getattr(contig, 'plus_strand_genes', [])
                minus_genes = getattr(contig, 'minus_strand_genes', [])
                all_genes = plus_genes + minus_genes
                
                # Check if adding this entire contig would exceed limit
                contig_gene_count = len(all_genes)
                if total_genes_processed > 0 and (total_genes_processed + contig_gene_count) > max_genes_per_chunk:
                    logger.info(f"📊 Stopping at {total_genes_processed} genes to preserve contig boundaries")
                    break
                
                # Process entire contig to maintain biological integrity
                contig_info = {
                    "contig_id": contig_id,
                    "length": getattr(contig, 'length', 0),
                    "genes": []
                }
                
                # Sort genes by position for spatial analysis
                all_genes.sort(key=lambda g: getattr(g, 'start', 0))
                
                # Add all genes from this contig
                for gene in all_genes:
                    gene_info = {
                        "gene_id": getattr(gene, 'gene_id', 'unknown'),
                        "protein_id": getattr(gene, 'protein_id', 'unknown'),
                        "start": getattr(gene, 'start', 0),
                        "end": getattr(gene, 'end', 0),
                        "strand": getattr(gene, 'strand', '+'),
                        "is_hypothetical": getattr(gene, 'is_hypothetical', False),
                        "ko_description": getattr(gene, 'ko_description', ''),
                        "pfam_domains": getattr(gene, 'pfam_domains', [])
                    }
                    contig_info["genes"].append(gene_info)
                
                # Add complete contig
                structured_data.append(contig_info)
                total_genes_processed += contig_gene_count
                
                logger.debug(f"🧱 Added contig {contig_id} with {contig_gene_count} genes")
            
            logger.info(f"📊 Prepared {total_genes_processed} genes across {len(structured_data)} contigs (contig-boundary preserved)")
            return str(structured_data)  # LLM can parse this structure
            
        except Exception as e:
            logger.warning(f"Error preparing structured genomic data: {e}")
            return "[]"
    
    def _summarize_gene(self, gene: Any) -> str:
        """Create a concise summary of a gene for LLM analysis."""
        try:
            protein_id = getattr(gene, 'protein_id', 'unknown')
            start = getattr(gene, 'start', 0)
            end = getattr(gene, 'end', 0)
            ko_desc = getattr(gene, 'ko_description', '')
            pfam_domains = getattr(gene, 'pfam_domains', [])
            is_hypothetical = getattr(gene, 'is_hypothetical', False)
            
            # Build concise description
            desc_parts = [f"{protein_id} ({start}-{end})"]
            
            if is_hypothetical:
                desc_parts.append("hypothetical protein")
            elif ko_desc:
                desc_parts.append(ko_desc)
            else:
                desc_parts.append("uncharacterized")
            
            if pfam_domains:
                domain_str = str(pfam_domains)  
                desc_parts.append(f"domains: {domain_str}")
            
            return " | ".join(desc_parts)
            
        except Exception as e:
            logger.warning(f"Error summarizing gene: {e}")
            return "gene summary unavailable"
    
    def _parse_structured_llm_response(self, llm_response: str, contigs: List[Any]) -> List[InterestingLocus]:
        """Parse structured LLM response into InterestingLocus objects."""
        try:
            import json
            import re
            
            # Try to extract JSON from the response
            json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in LLM response")
                return []
            
            # DEBUG: Log what the LLM actually returned
            logger.warning(f"🐛 DEBUG: Raw LLM response: {llm_response[:500]}...")
            
            try:
                loci_data = json.loads(json_match.group())
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from LLM response")
                return []
            
            interesting_loci = []
            
            for locus_data in loci_data:
                try:
                    # Extract required fields
                    contig_id = locus_data.get('contig_id', 'unknown')
                    start_coord = int(locus_data.get('start_coordinate', 0))
                    end_coord = int(locus_data.get('end_coordinate', 0))
                    gene_ids = locus_data.get('gene_ids', [])
                    
                    # VALIDATE: Reject synthetic gene IDs with coordinate ranges
                    valid_gene_ids = []
                    for gene_id in gene_ids:
                        if isinstance(gene_id, str) and ':' in gene_id and '-' in gene_id.split(':')[-1]:
                            # This looks like a synthetic ID with coordinates (e.g. "gene:ABC:0-5000")
                            logger.warning(f"🚫 Rejecting synthetic gene ID with coordinates: {gene_id}")
                            continue
                        valid_gene_ids.append(gene_id)
                    
                    if not valid_gene_ids:
                        logger.warning(f"🚫 Skipping locus - all gene IDs were synthetic with coordinates")
                        continue
                        
                    gene_ids = valid_gene_ids
                    locus_type = locus_data.get('locus_type', 'unknown')
                    reasoning = locus_data.get('biological_significance_reasoning', '')
                    
                    # Find the actual genes based on gene_ids
                    genes_in_locus = self._find_genes_by_ids(gene_ids, contigs)
                    
                    if not genes_in_locus:
                        logger.warning(f"No genes found for locus {contig_id}:{start_coord}-{end_coord}")
                        continue
                    
                    # Count hypothetical proteins
                    hypothetical_count = sum(1 for gene in genes_in_locus 
                                           if getattr(gene, 'is_hypothetical', False))
                    
                    # Extract biological features from reasoning and gene annotations
                    biological_features = [reasoning]
                    for gene in genes_in_locus:
                        ko_desc = getattr(gene, 'ko_description', '')
                        if ko_desc and ko_desc not in biological_features:
                            biological_features.append(ko_desc)
                    
                    # Get flanking information
                    flanking_genes = []
                    if len(genes_in_locus) >= 2:
                        first_desc = getattr(genes_in_locus[0], 'ko_description', 'unknown')
                        last_desc = getattr(genes_in_locus[-1], 'ko_description', 'unknown')
                        flanking_genes = [f"5': {first_desc}", f"3': {last_desc}"]
                    elif len(genes_in_locus) == 1:
                        gene_desc = getattr(genes_in_locus[0], 'ko_description', 'unknown')
                        flanking_genes = [f"Single gene: {gene_desc}"]
                    

                    # Create InterestingLocus object
                    locus = InterestingLocus(
                        contig_id=contig_id,
                        start=start_coord,
                        end=end_coord,
                        gene_count=len(genes_in_locus),
                        hypothetical_count=hypothetical_count,
                        biological_features=biological_features,
                        flanking_genes=flanking_genes,
                        locus_type=locus_type
                    )
                    
                    interesting_loci.append(locus)
                    
                except Exception as e:
                    logger.warning(f"Error processing locus data: {e}")
                    continue
            
            logger.info(f"🎯 Parsed {len(interesting_loci)} loci from LLM response")
            return interesting_loci
            
        except Exception as e:
            logger.error(f"Error parsing structured LLM response: {e}")
            return []
    
    def _find_genes_by_ids(self, gene_ids: List[str], contigs: List[Any]) -> List[Any]:
        """
        Gene ID resolution that handles:
        1. Exact gene ID matches
        2. Protein ID matches  
        3. Partial ID matches as fallback
        """
        found_genes = []
        
        try:
            for gene_id in gene_ids:
                logger.debug(f"🔍 Resolving gene ID: {gene_id}")
                
                # Strategy 1: Exact gene ID match
                exact_match = self._find_by_exact_gene_id(gene_id, contigs)
                if exact_match:
                    found_genes.extend(exact_match)
                    continue
                    
                # Strategy 2: Exact protein ID match  
                protein_match = self._find_by_exact_protein_id(gene_id, contigs)
                if protein_match:
                    found_genes.extend(protein_match)
                    continue
                    
                # Strategy 3: Partial ID matching (fuzzy)
                partial_match = self._find_by_partial_id(gene_id, contigs)
                if partial_match:
                    logger.debug(f"✅ Found by partial ID match: {len(partial_match)} genes")
                    found_genes.extend(partial_match)
                    continue
                    
                logger.warning(f"❌ No match found for gene ID: {gene_id}")
            
            # Remove duplicates while preserving order
            unique_genes = []
            seen_ids = set()
            for gene in found_genes:
                gene_id = getattr(gene, 'gene_id', '')
                if gene_id not in seen_ids:
                    unique_genes.append(gene)
                    seen_ids.add(gene_id)
            
            # Sort by genomic position
            unique_genes.sort(key=lambda g: getattr(g, 'start', 0))
            
            return unique_genes
            
        except Exception as e:
            logger.warning(f"Error in robust gene ID resolution: {e}")
            return []
    
    def _find_by_exact_gene_id(self, gene_id: str, contigs: List[Any]) -> List[Any]:
        """Find genes by exact gene ID match."""
        for contig in contigs:
            all_genes = getattr(contig, 'plus_strand_genes', []) + getattr(contig, 'minus_strand_genes', [])
            for gene in all_genes:
                if getattr(gene, 'gene_id', '') == gene_id:
                    return [gene]
        return []
    
    def _find_by_exact_protein_id(self, gene_id: str, contigs: List[Any]) -> List[Any]:
        """Find genes by exact protein ID match."""
        for contig in contigs:
            all_genes = getattr(contig, 'plus_strand_genes', []) + getattr(contig, 'minus_strand_genes', [])
            for gene in all_genes:
                if getattr(gene, 'protein_id', '') == gene_id:
                    return [gene]
        return []
    
    
    def _find_by_partial_id(self, gene_id: str, contigs: List[Any]) -> List[Any]:
        """Find genes by partial ID matching (last resort)."""
        # Extract meaningful parts from the gene ID
        parts = gene_id.replace('gene:', '').replace('protein:', '').split('_')
        if len(parts) < 2:
            return []
        
        # Look for genes with similar scaffold/contig patterns
        candidates = []
        for contig in contigs:
            all_genes = getattr(contig, 'plus_strand_genes', []) + getattr(contig, 'minus_strand_genes', [])
            for gene in all_genes:
                gene_id_actual = getattr(gene, 'gene_id', '')
                
                # Use whole string matching instead of partial
                if gene_id.replace('gene:', '').replace('protein:', '') == gene_id_actual.replace('gene:', '').replace('protein:', ''):
                    candidates.append(gene)
        
        return candidates
    
