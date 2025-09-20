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
class DetailedGeneInfo:
    """Detailed gene information with full annotations preserved."""
    gene_id: str
    protein_id: Optional[str]
    start: int
    end: int
    strand: str
    length: int
    annotation: str
    ko_id: Optional[str]
    ko_description: Optional[str]
    pfam_domains: List[str]
    is_hypothetical: bool


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
    detailed_genes: List[DetailedGeneInfo]  # Full gene annotations preserved
    
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
    """
    Identify interesting genomic regions from chunk analysis.
    
    CRITICAL ANTI-HALLUCINATION CONSTRAINTS:
    - ONLY use gene IDs that appear EXACTLY in the input genomic_chunk_data
    - DO NOT fabricate, assume, or invent gene IDs that are not explicitly provided
    - DO NOT create sequential gene numbering (e.g., if you see genes 1,2,3 do NOT assume 4,5,6 exist)
    - DO NOT extrapolate missing genes or fill in gaps in gene numbering
    - If unsure whether a gene ID exists, DO NOT include it
    - VIOLATION OF THESE CONSTRAINTS WILL RESULT IN ANALYSIS REJECTION
    """
    
    genomic_chunk_data = dspy.InputField(desc="Structured genomic data for a chunk of genes with coordinates, annotations, and hypothetical status")
    analysis_criteria = dspy.InputField(desc="User question and criteria for identifying interesting regions")
    
<<<<<<< HEAD
    interesting_regions = dspy.OutputField(desc="JSON list of interesting loci, each with: contig_id (COMPLETE scaffold/contig identifier as it appears in the data), start_coordinate, end_coordinate, gene_ids (array of individual gene_id strings from input data), locus_type, biological_significance_reasoning. VERIFICATION CRITICAL: Use the complete, unabbreviated contig_id for traceability (e.g., 'RIFCSPLOWO2_01_FULL_OD1_41_220_rifcsplowo2_01_scaffold_1705' not 'scaffold_1705'). Gene_ids must contain ONLY values that appear EXACTLY in the genomic_chunk_data input - DO NOT fabricate or assume any gene IDs not explicitly provided. HALLUCINATED GENE IDS WILL BE REJECTED.")
=======
    interesting_regions = dspy.OutputField(desc="JSON list of interesting loci, each with: contig_id (COMPLETE scaffold/contig identifier as it appears in the data), start_coordinate, end_coordinate, gene_ids (array of individual gene_id strings from input data), locus_type, biological_significance_reasoning. VERIFICATION CRITICAL: Use the complete, unabbreviated contig_id for traceability (e.g., 'EXAMPLE_CONTIG_ID' not 'scaffold_XXXX'). Gene_ids must contain ONLY values that appear EXACTLY in the genomic_chunk_data input - DO NOT fabricate or assume any gene IDs not explicitly provided. HALLUCINATED GENE IDS WILL BE REJECTED.")
>>>>>>> feat/agent-router-typed


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
            
            # Extract available gene IDs for explicit instruction
            available_gene_ids = self._extract_all_gene_ids_from_chunk_data(contigs)
            gene_id_sample = list(available_gene_ids)[:20]  # Show first 20 as examples
            
            # Create analysis criteria based on user question
            analysis_criteria = f"""
            User question: {user_question}
            
            AVAILABLE GENE IDs IN THIS CHUNK (ONLY these can be used):
            {gene_id_sample}
            
            CRITICAL: You can ONLY reference gene IDs that appear in the genomic_chunk_data. DO NOT create or assume additional gene IDs.
            
            Please identify genomically interesting regions. For each region, provide:
<<<<<<< HEAD
            - contig_id: COMPLETE scaffold/contig identifier as it appears in the input data (e.g., 'RIFCSPLOWO2_01_FULL_OD1_41_220_rifcsplowo2_01_scaffold_1705' - DO NOT abbreviate to just 'scaffold_1705')
=======
            - contig_id: COMPLETE scaffold/contig identifier as it appears in the input data (e.g., 'EXAMPLE_CONTIG_ID' - DO NOT abbreviate to just 'scaffold_XXXX')
>>>>>>> feat/agent-router-typed
            - start_coordinate: genomic nucleotide position where the region begins (use the START coordinate of the first gene in your selection)
            - end_coordinate: genomic nucleotide position where the region ends (use the END coordinate of the last gene in your selection)
            - gene_ids: array of individual gene_id strings from the input data
            - locus_type: description of what makes this region interesting
            - biological_significance_reasoning: why this region is relevant to the user's question
            
            COORDINATE CALCULATION GUIDELINES:
            - start_coordinate should be the genomic START position of the earliest gene you selected (e.g., if gene at position 15234-16890 is your first gene, use 15234)
            - end_coordinate should be the genomic END position of the latest gene you selected (e.g., if gene at position 28456-29123 is your last gene, use 29123)
            - These should be actual nucleotide positions on the contig, NOT gene numbers or indices
            - Coordinates should make biological sense (start < end, and span the actual genomic region)
            
            CRITICAL INSTRUCTIONS FOR gene_ids:
            1. gene_ids should be an array like ["gene:ABC_001", "gene:ABC_002", "gene:ABC_003"]
            2. Each entry should be an individual gene_id EXACTLY as it appears in the input data
            3. Do NOT append coordinates to gene IDs (NO ":0-12000" or similar suffixes)
            4. Do NOT create synthetic identifiers by combining gene IDs with coordinate ranges
            5. If a region contains 3 genes, list all 3 individual gene_id values separately
            
<<<<<<< HEAD
            EXAMPLE of CORRECT output format:
            [
              {{
                "contig_id": "RIFCSPLOWO2_01_FULL_OD1_41_220_rifcsplowo2_01_scaffold_1705",
=======
            EXAMPLE of CORRECT output format (with placeholders):
            [
              {{
                "contig_id": "EXAMPLE_CONTIG_ID",
>>>>>>> feat/agent-router-typed
                "start_coordinate": 15234,
                "end_coordinate": 29123,
                "gene_ids": ["gene:scaffold_1_001", "gene:scaffold_1_002", "gene:scaffold_1_003"],
                "locus_type": "prophage candidate",
                "biological_significance_reasoning": "Contains multiple hypothetical proteins spanning 13.9kb region"
              }}
            ]
            
            NOTE: In this example, the first gene (scaffold_1_001) starts at position 15234, the last gene (scaffold_1_003) ends at position 29123, so the locus spans 15234-29123.
            
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
        """Prepare structured genomic data with token-aware contig-based chunking."""
        try:
            # Token limits for different models (conservative estimates)
            model_token_limits = {
                "gpt-4o-mini": 8000,  # Conservative limit for input context
                "gpt-4o": 8000,
                "gpt-4": 6000
            }
            
            max_tokens = model_token_limits.get(self.model_name, 8000)
            logger.info(f"🎯 Token-aware chunking: max {max_tokens} tokens for {self.model_name}")
            
            structured_data = []
            current_tokens = 0
            total_genes_processed = 0
            
            for contig in contigs:
                contig_id = getattr(contig, 'contig_id', 'unknown')
                
                # Collect all genes from both strands
                plus_genes = getattr(contig, 'plus_strand_genes', [])
                minus_genes = getattr(contig, 'minus_strand_genes', [])
                all_genes = plus_genes + minus_genes
                
                if not all_genes:
                    continue
                
                # Build contig data structure
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
                
                # Estimate tokens for this contig (improved approximation)
                contig_tokens = self._estimate_contig_tokens(contig_info)
                
                # Check if adding this contig would exceed token limit
                if current_tokens > 0 and (current_tokens + contig_tokens) > max_tokens:
                    logger.info(f"🎯 Token limit reached: {current_tokens} tokens, stopping at {total_genes_processed} genes to preserve contig boundaries")
                    break
                
                # Add complete contig (preserve biological integrity)
                structured_data.append(contig_info)
                current_tokens += contig_tokens
                total_genes_processed += len(all_genes)
                
                logger.debug(f"🧱 Added contig {contig_id}: {len(all_genes)} genes, ~{contig_tokens} tokens (total: {current_tokens})")
            
            logger.info(f"📊 Token-aware chunking complete: {total_genes_processed} genes across {len(structured_data)} contigs (~{current_tokens} tokens)")
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
            logger.warning(f"🐛 DEBUG: Extracted JSON: {json_match.group()[:300]}...")
            
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
                    
                    # VALIDATE: Strict hallucination prevention
                    valid_gene_ids = []
                    available_gene_ids = self._extract_all_gene_ids_from_chunk_data(contigs)
                    
                    for gene_id in gene_ids:
                        # Check 1: Reject synthetic gene IDs with coordinate ranges
                        if isinstance(gene_id, str) and ':' in gene_id and '-' in gene_id.split(':')[-1]:
                            logger.warning(f"🚫 Rejecting synthetic gene ID with coordinates: {gene_id}")
                            continue
                        
                        # Check 2: STRICT - Gene ID must exist in input data
                        if gene_id not in available_gene_ids:
                            logger.error(f"🚨 HALLUCINATION DETECTED: Gene ID '{gene_id}' not found in input data!")
                            logger.error(f"   Available gene IDs sample: {list(available_gene_ids)[:10]}")
                            continue
                        
                        valid_gene_ids.append(gene_id)
                    
                    if not valid_gene_ids:
                        logger.warning(f"🚫 Skipping locus - all gene IDs were hallucinated or synthetic")
                        continue
                        
                    gene_ids = valid_gene_ids
                    locus_type = locus_data.get('locus_type', 'unknown')
                    reasoning = locus_data.get('biological_significance_reasoning', '')
                    
                    # Find the actual genes based on gene_ids
                    genes_in_locus = self._find_genes_by_ids(gene_ids, contigs)
                    
                    if not genes_in_locus:
                        logger.warning(f"No genes found for locus {contig_id}:{start_coord}-{end_coord}")
                        continue
                    
                    # CRITICAL FIX: Recalculate coordinates from actual gene positions
                    # The LLM often returns gene indices or nonsensical coordinates
                    # We need to use the actual genomic start/end positions of the found genes
                    actual_starts = [getattr(gene, 'start', 0) for gene in genes_in_locus]
                    actual_ends = [getattr(gene, 'end', 0) for gene in genes_in_locus]
                    
                    if actual_starts and actual_ends:
                        # Calculate the span of the genomic region
                        recalculated_start = min(actual_starts)
                        recalculated_end = max(actual_ends)
                        
                        # Log the fix for debugging
                        if start_coord != recalculated_start or end_coord != recalculated_end:
                            logger.warning(f"🔧 COORDINATE FIX: LLM returned {start_coord}-{end_coord}, "
                                         f"using actual gene coordinates {recalculated_start}-{recalculated_end}")
                        
                        # Use the corrected coordinates
                        start_coord = recalculated_start
                        end_coord = recalculated_end
                    else:
                        logger.warning(f"Could not extract gene coordinates for locus {contig_id}")
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
                    

                    # Create detailed gene info objects preserving all annotations
                    detailed_genes = self._create_detailed_gene_info(genes_in_locus)
                    
                    # Create InterestingLocus object with full gene details
                    locus = InterestingLocus(
                        contig_id=contig_id,
                        start=start_coord,
                        end=end_coord,
                        gene_count=len(genes_in_locus),
                        hypothetical_count=hypothetical_count,
                        biological_features=biological_features,
                        flanking_genes=flanking_genes,
                        locus_type=locus_type,
                        detailed_genes=detailed_genes
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
                    logger.debug(f"✅ Found by exact gene ID match: {len(exact_match)} genes")
                    found_genes.extend(exact_match)
                    continue
                    
                # Strategy 2: Exact protein ID match  
                protein_match = self._find_by_exact_protein_id(gene_id, contigs)
                if protein_match:
                    logger.debug(f"✅ Found by exact protein ID match: {len(protein_match)} genes")
                    found_genes.extend(protein_match)
                    continue
                    
                # Strategy 3: Partial ID matching (fuzzy)
                partial_match = self._find_by_partial_id(gene_id, contigs)
                if partial_match:
                    logger.debug(f"✅ Found by partial ID match: {len(partial_match)} genes")
                    found_genes.extend(partial_match)
                    continue
                    
                # Debug: Show what gene IDs are actually available in the first contig
                if contigs:
                    first_contig = contigs[0]
                    sample_genes = (getattr(first_contig, 'plus_strand_genes', [])[:3] + 
                                  getattr(first_contig, 'minus_strand_genes', [])[:3])
                    sample_ids = [getattr(g, 'gene_id', 'no_id') for g in sample_genes]
                    logger.warning(f"❌ No match found for gene ID: {gene_id}")
                    logger.debug(f"   Sample gene IDs in data: {sample_ids[:5]}")
            
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
        """Find genes by exact gene ID match, handling prefix variations."""
        # Try exact match first
        for contig in contigs:
            all_genes = getattr(contig, 'plus_strand_genes', []) + getattr(contig, 'minus_strand_genes', [])
            for gene in all_genes:
                actual_gene_id = getattr(gene, 'gene_id', '')
                if actual_gene_id == gene_id:
                    return [gene]
        
        # Try without gene: prefix on both sides
        clean_search_id = gene_id.replace('gene:', '') if gene_id.startswith('gene:') else gene_id
        for contig in contigs:
            all_genes = getattr(contig, 'plus_strand_genes', []) + getattr(contig, 'minus_strand_genes', [])
            for gene in all_genes:
                actual_gene_id = getattr(gene, 'gene_id', '')
                clean_actual_id = actual_gene_id.replace('gene:', '') if actual_gene_id.startswith('gene:') else actual_gene_id
                if clean_actual_id == clean_search_id:
                    return [gene]
        
        return []
    
    def _find_by_exact_protein_id(self, gene_id: str, contigs: List[Any]) -> List[Any]:
        """Find genes by exact protein ID match, handling prefix variations."""
        # Try exact match first
        for contig in contigs:
            all_genes = getattr(contig, 'plus_strand_genes', []) + getattr(contig, 'minus_strand_genes', [])
            for gene in all_genes:
                actual_protein_id = getattr(gene, 'protein_id', '')
                if actual_protein_id == gene_id:
                    return [gene]
        
        # Try without protein: prefix on both sides
        clean_search_id = gene_id.replace('protein:', '').replace('gene:', '') 
        for contig in contigs:
            all_genes = getattr(contig, 'plus_strand_genes', []) + getattr(contig, 'minus_strand_genes', [])
            for gene in all_genes:
                actual_protein_id = getattr(gene, 'protein_id', '')
                clean_actual_id = actual_protein_id.replace('protein:', '').replace('gene:', '') if actual_protein_id else ''
                if clean_actual_id and clean_actual_id == clean_search_id:
                    return [gene]
        
        return []
    
    
    def _find_by_partial_id(self, gene_id: str, contigs: List[Any]) -> List[Any]:
        """Find genes by partial ID matching (last resort)."""
        # Clean the search ID
        clean_search_id = gene_id.replace('gene:', '').replace('protein:', '')
        
        if not clean_search_id:
            return []
        
        # Look for genes with matching cleaned IDs
        candidates = []
        for contig in contigs:
            all_genes = getattr(contig, 'plus_strand_genes', []) + getattr(contig, 'minus_strand_genes', [])
            for gene in all_genes:
                gene_id_actual = getattr(gene, 'gene_id', '')
                protein_id_actual = getattr(gene, 'protein_id', '')
                
                # Clean actual IDs and compare
                clean_gene_id = gene_id_actual.replace('gene:', '').replace('protein:', '') if gene_id_actual else ''
                clean_protein_id = protein_id_actual.replace('gene:', '').replace('protein:', '') if protein_id_actual else ''
                
                # Exact match on cleaned IDs
                if clean_search_id == clean_gene_id or clean_search_id == clean_protein_id:
                    candidates.append(gene)
                    continue
                
                # Substring matching as last resort (if the search ID is contained in the actual ID)
                if clean_search_id in clean_gene_id or clean_search_id in clean_protein_id:
                    candidates.append(gene)
        
        return candidates
    
    def _create_detailed_gene_info(self, genes_in_locus: List[Any]) -> List[DetailedGeneInfo]:
        """Convert GeneContext objects to DetailedGeneInfo objects preserving all annotations."""
        detailed_genes = []
        
        try:
            for gene in genes_in_locus:
                detailed_gene = DetailedGeneInfo(
                    gene_id=getattr(gene, 'gene_id', ''),
                    protein_id=getattr(gene, 'protein_id', None),
                    start=getattr(gene, 'start', 0),
                    end=getattr(gene, 'end', 0),
                    strand=getattr(gene, 'strand', ''),
                    length=getattr(gene, 'length', 0),
                    annotation=getattr(gene, 'annotation', ''),
                    ko_id=getattr(gene, 'ko_id', None),
                    ko_description=getattr(gene, 'ko_description', None),
                    pfam_domains=getattr(gene, 'pfam_domains', []),
                    is_hypothetical=getattr(gene, 'is_hypothetical', False)
                )
                detailed_genes.append(detailed_gene)
                
        except Exception as e:
            logger.warning(f"Error creating detailed gene info: {e}")
        
        return detailed_genes
    
    def _estimate_contig_tokens(self, contig_info: Dict[str, Any]) -> int:
        """Estimate token count for a contig data structure."""
        try:
            # Base tokens for contig structure
            base_tokens = 50  # contig_id, length, metadata
            
            # Tokens per gene (empirically estimated)
            genes = contig_info.get("genes", [])
            gene_tokens = 0
            
            for gene in genes:
                # Base gene structure: ~30 tokens
                gene_base = 30
                
                # Gene ID and protein ID: ~10 tokens each
                gene_ids = 20
                
                # Coordinates and strand: ~10 tokens
                coordinates = 10
                
                # Functional annotations
                ko_desc = gene.get("ko_description", "")
                ko_tokens = len(ko_desc.split()) * 1.3 if ko_desc else 5  # "hypothetical protein"
                
                # PFAM domains
                pfam_domains = gene.get("pfam_domains", [])
                pfam_tokens = len(pfam_domains) * 3  # domain names
                
                gene_total = gene_base + gene_ids + coordinates + ko_tokens + pfam_tokens
                gene_tokens += gene_total
            
            total_estimated = base_tokens + gene_tokens
            
            # Add 10% buffer for JSON formatting
            return int(total_estimated * 1.1)
            
        except Exception as e:
            logger.debug(f"Error estimating tokens, using fallback: {e}")
            # Fallback to character-based estimation
            contig_str = str(contig_info)
            return len(contig_str) // 4
    
    def _extract_all_gene_ids_from_chunk_data(self, contigs: List[Any]) -> set:
        """Extract all gene IDs that are actually present in the input data."""
        available_gene_ids = set()
        
        try:
            for contig in contigs:
                plus_genes = getattr(contig, 'plus_strand_genes', [])
                minus_genes = getattr(contig, 'minus_strand_genes', [])
                
                for gene in plus_genes + minus_genes:
                    gene_id = getattr(gene, 'gene_id', '')
                    protein_id = getattr(gene, 'protein_id', '')
                    
                    if gene_id:
                        available_gene_ids.add(gene_id)
                        # Also add cleaned version without prefix
                        if gene_id.startswith('gene:'):
                            available_gene_ids.add(gene_id.replace('gene:', ''))
                    
                    if protein_id:
                        available_gene_ids.add(protein_id)
                        # Also add cleaned version without prefix
                        if protein_id.startswith('protein:'):
                            available_gene_ids.add(protein_id.replace('protein:', ''))
        
        except Exception as e:
            logger.warning(f"Error extracting gene IDs from chunk data: {e}")
        
        return available_gene_ids
    
