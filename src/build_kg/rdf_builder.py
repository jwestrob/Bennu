#!/usr/bin/env python3
"""
RDF triple generation for genome knowledge graph construction.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import rdflib
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from src.build_kg.annotation_processors import process_astra_results
from src.build_kg.functional_enrichment import add_functional_enrichment_to_pipeline
from src.build_kg.pathway_integration import integrate_pathways

logger = logging.getLogger(__name__)


def parse_prodigal_header(header_line: str) -> Dict[str, Any]:
    """
    Parse prodigal FASTA header to extract genomic coordinates and metadata.
    
    Example header:
    >RIFCSPHIGHO2_01_FULL_Acidovorax_64_960_rifcsphigho2_01_scaffold_15917_1 # 76 # 171 # -1 # ID=1_1;partial=00;start_type=ATG;rbs_motif=AGGAG;rbs_spacer=5-10bp;gc_cont=0.573
    
    Returns:
        Dict with parsed gene information including coordinates, strand, length, GC content
    """
    parts = header_line.strip()[1:].split(' # ')  # Remove '>' and split by ' # '
    
    if len(parts) < 4:
        # Basic header without coordinates
        protein_id = parts[0].split()[0]
        return {'gene_id': protein_id}
    
    protein_id = parts[0]
    start_coord = int(parts[1])
    end_coord = int(parts[2])
    strand = int(parts[3])
    
    gene_data = {
        'gene_id': protein_id,
        'start': start_coord,
        'end': end_coord,
        'strand': strand,
        'length_nt': abs(end_coord - start_coord) + 1,
        'length_aa': (abs(end_coord - start_coord) + 1) // 3
    }
    
    # Parse additional metadata if present
    if len(parts) > 4:
        metadata_str = parts[4]
        
        # Extract GC content
        gc_match = re.search(r'gc_cont=([0-9.]+)', metadata_str)
        if gc_match:
            gene_data['gc_content'] = float(gc_match.group(1))
        
        # Note: Additional fields like start_type, rbs_motif available for future integration
        # as documented in CLAUDE.md
    
    return gene_data


def build_protein_to_genome_mapping(protein_uris: Dict[str, URIRef], 
                                   genome_uris: Dict[str, URIRef]) -> Dict[str, str]:
    """
    Build mapping from protein header IDs to correct filename-based genome IDs.
    
    This resolves the mismatch between:
    - Protein IDs: RIFCSPHIGHO2_01_FULL_Acidovorax_64_960_rifcsphigho2_01_scaffold_X_Y
    - Genome IDs: Burkholderiales_bacterium_RIFCSPHIGHO2_01_FULL_64_960_contigs
    
    Args:
        protein_uris: Map of protein_id -> protein_uri from RDF building
        genome_uris: Map of genome_id -> genome_uri from RDF building
        
    Returns:
        Dict mapping protein_id -> correct_genome_id
    """
    protein_to_genome = {}
    
    # Extract common identifiers from genome IDs for matching
    genome_patterns = {}
    for genome_id in genome_uris.keys():
        # Special handling for PLM0 genomes: use "PLM0_60" pattern
        if genome_id.startswith('PLM0_'):
            plm_parts = genome_id.split('_')
            if len(plm_parts) >= 2 and plm_parts[1].isdigit():
                pattern = f"PLM0_{plm_parts[1]}"
                genome_patterns[pattern] = genome_id
                logger.debug(f"PLM0 pattern mapping: {pattern} -> {genome_id}")
        
        # For RIFCS genomes, use the RIFCS prefix + number pattern  
        elif 'RIFCS' in genome_id:
            # Extract patterns like "RIFCSPHIGHO2_01_FULL_64_960" or "RIFCSPLOWO2_01_FULL_41_220"
            parts = genome_id.split('_')
            rifcs_pattern = None
            for i, part in enumerate(parts):
                if part.startswith('RIFCS') and i + 4 < len(parts):
                    # Build pattern like "RIFCSPHIGHO2_01_FULL_64_960"
                    if parts[i+3].isdigit() and parts[i+4].isdigit():
                        rifcs_pattern = f"{parts[i]}_{parts[i+1]}_{parts[i+2]}_{parts[i+3]}_{parts[i+4]}"
                        genome_patterns[rifcs_pattern] = genome_id
                        logger.debug(f"RIFCS pattern mapping: {rifcs_pattern} -> {genome_id}")
                        break
        
        # Fallback: Extract pattern like "64_960" from any genome
        if genome_id not in genome_patterns.values():
            parts = genome_id.split('_')
            for i in range(len(parts) - 1):
                if parts[i].isdigit() and parts[i+1].isdigit():
                    pattern = f"{parts[i]}_{parts[i+1]}"
                    if pattern not in genome_patterns:  # Avoid conflicts
                        genome_patterns[pattern] = genome_id
                        logger.debug(f"Genome pattern mapping: {pattern} -> {genome_id}")
                        break
    
    # Map each protein ID to correct genome ID using pattern matching
    for protein_id in protein_uris.keys():
        # Extract pattern from protein ID like "RIFCSPHIGHO2_01_FULL_Acidovorax_64_960_..." or "PLM0_60_b1_sep16_..."
        
        # Handle PLM0 proteins first
        if protein_id.startswith('PLM0_'):
            plm_parts = protein_id.split('_')
            if len(plm_parts) >= 2 and plm_parts[1].isdigit():
                pattern = f"PLM0_{plm_parts[1]}"
                if pattern in genome_patterns:
                    correct_genome_id = genome_patterns[pattern]
                    protein_to_genome[protein_id] = correct_genome_id
                    logger.debug(f"PLM0 protein mapping: {protein_id} -> {correct_genome_id}")
                    continue
        
        # Handle RIFCS proteins with full RIFCS pattern matching
        elif 'RIFCS' in protein_id:
            parts = protein_id.split('_')
            for i, part in enumerate(parts):
                if part.startswith('RIFCS') and i + 4 < len(parts):
                    # Look for pattern like "RIFCSPHIGHO2_01_FULL_64_960" in the protein ID
                    if parts[i+3].isdigit() and parts[i+4].isdigit():
                        rifcs_pattern = f"{parts[i]}_{parts[i+1]}_{parts[i+2]}_{parts[i+3]}_{parts[i+4]}"
                        if rifcs_pattern in genome_patterns:
                            correct_genome_id = genome_patterns[rifcs_pattern]
                            protein_to_genome[protein_id] = correct_genome_id
                            logger.debug(f"RIFCS protein mapping: {protein_id} -> {correct_genome_id}")
                            break
        
        # Fallback: Handle other protein patterns with consecutive digit sequences
        if protein_id not in protein_to_genome:
            parts = protein_id.split('_')
            for i in range(len(parts) - 1):
                if parts[i].isdigit() and parts[i+1].isdigit():
                    pattern = f"{parts[i]}_{parts[i+1]}"
                    if pattern in genome_patterns:
                        correct_genome_id = genome_patterns[pattern]
                        protein_to_genome[protein_id] = correct_genome_id
                        logger.debug(f"Protein mapping: {protein_id} -> {correct_genome_id}")
                        break
        
        if protein_id not in protein_to_genome:
            logger.warning(f"Could not map protein {protein_id} to any genome")
    
    logger.info(f"Built protein-to-genome mapping: {len(protein_to_genome)} proteins mapped to correct genomes")
    return protein_to_genome


def build_contig_to_genome_index_from_proteins(graph: Graph, protein_uris: Dict[str, URIRef]) -> Dict[str, URIRef]:
    """Build efficient contig -> genome URI mapping by querying existing protein-genome relationships.
    
    Args:
        graph: RDF graph containing protein-genome relationships  
        protein_uris: Map of protein_id -> protein_uri
        
    Returns:
        Dict mapping contig_id -> genome_uri for efficient BGC assignment
    """
    contig_to_genome = {}
    
    # Query the graph for protein-genome relationships that already exist
    query = """
    PREFIX kg: <http://genome-kg.org/ontology/>
    
    SELECT ?protein ?genome WHERE {
        ?protein kg:encodedBy ?gene .
        ?gene kg:belongsToGenome ?genome .
    }
    """
    
    # Execute query and build mapping
    for row in graph.query(query):
        protein_uri = str(row.protein)
        genome_uri = row.genome
        
        # Extract protein ID from URI
        protein_id = protein_uri.split('/')[-1] if '/' in protein_uri else protein_uri
        
        # Extract contig from protein ID
        parts = protein_id.split('_')
        if len(parts) >= 2:
            contig_id = '_'.join(parts[:-1])  # Remove last part (protein number)
            if contig_id not in contig_to_genome:
                contig_to_genome[contig_id] = genome_uri
    
    logger.info(f"Built contig-to-genome index from existing relationships: {len(contig_to_genome)} contigs mapped")
    return contig_to_genome


def assign_bgc_to_correct_genome(bgc_data: Dict[str, Any], 
                                contig_to_genome: Dict[str, URIRef]) -> Dict[str, URIRef]:
    """Assign each BGC to its correct genome URI based on contig mapping.
    
    Args:
        bgc_data: BGC results from GECCO
        contig_to_genome: Mapping from contig_id -> genome_uri
        
    Returns:
        Dict mapping cluster_id -> genome_uri
    """
    bgc_genome_assignments = {}
    
    for cluster in bgc_data.get("clusters", []):
        contig_id = cluster.get("contig", "")
        cluster_id = cluster.get("cluster_id", "unknown")
        
        if contig_id in contig_to_genome:
            bgc_genome_assignments[cluster_id] = contig_to_genome[contig_id]
            logger.debug(f"Assigned BGC {cluster_id} to genome via contig {contig_id}")
        else:
            logger.warning(f"Could not assign BGC {cluster_id} - contig {contig_id} not found in mapping")
    
    logger.info(f"BGC genome assignments: {len(bgc_genome_assignments)} BGCs assigned to genomes")
    return bgc_genome_assignments


# Define ontology namespaces
KG = Namespace("http://genome-kg.org/ontology/")
GENOME = Namespace("http://genome-kg.org/genomes/")
GENE = Namespace("http://genome-kg.org/genes/")
PROTEIN = Namespace("http://genome-kg.org/proteins/")
PFAM = Namespace("http://pfam.xfam.org/family/")
KO = Namespace("http://www.genome.jp/kegg/ko/")
CAZYME = Namespace("http://genome-kg.org/cazyme/")
PROV = Namespace("http://www.w3.org/ns/prov#")


class GenomeKGBuilder:
    """Builder for genome knowledge graph RDF triples."""
    
    def __init__(self):
        self.graph = Graph()
        self._bind_namespaces()
        self._add_ontology_definitions()
    
    def _bind_namespaces(self):
        """Bind namespace prefixes for cleaner serialization."""
        self.graph.bind("kg", KG)
        self.graph.bind("genome", GENOME)
        self.graph.bind("gene", GENE)
        self.graph.bind("protein", PROTEIN)
        self.graph.bind("pfam", PFAM)
        self.graph.bind("ko", KO)
        self.graph.bind("cazyme", CAZYME)
        self.graph.bind("prov", PROV)
    
    def _add_ontology_definitions(self):
        """Add core ontology class definitions."""
        classes = [
            (KG.Genome, "Genome assembly"),
            (KG.Gene, "Protein-coding gene"),
            (KG.Protein, "Protein sequence"),
            (KG.DomainAnnotation, "Protein domain annotation instance"),
            (KG.FunctionalAnnotation, "Functional annotation"),
            (KG.KEGGOrtholog, "KEGG Orthologous group"),
            (KG.Domain, "Protein domain family (PFAM)"),
            (KG.BGC, "Biosynthetic Gene Cluster"),
            (KG.BGCGene, "Gene within a biosynthetic gene cluster"),
            (KG.CAZymeFamily, "Carbohydrate-Active Enzyme family"),
            (KG.CAZymeAnnotation, "CAZyme domain annotation instance")
        ]
        
        properties = [
            (KG.belongsToGenome, "belongs to genome"),
            (KG.encodedBy, "protein encoded by gene"),
            (KG.hasDomain, "protein has domain"),
            (KG.hasFunction, "protein has function"),
            (KG.domainFamily, "domain belongs to family"),
            (KG.hasQualityMetrics, "genome has quality metrics"),
            (KG.partOfBGC, "gene is part of biosynthetic gene cluster"),
            (KG.hasBGC, "genome has biosynthetic gene cluster"),
            (KG.produces, "BGC produces metabolite"),
            (KG.hasCAZyme, "protein has CAZyme annotation"),
            (KG.cazymeFamily, "CAZyme annotation belongs to family")
        ]
        
        for class_uri, label in classes:
            self.graph.add((class_uri, RDF.type, RDFS.Class))
            self.graph.add((class_uri, RDFS.label, Literal(label)))
        
        for prop_uri, label in properties:
            self.graph.add((prop_uri, RDF.type, RDF.Property))
            self.graph.add((prop_uri, RDFS.label, Literal(label)))
    
    def add_genome_entity(self, genome_data: Dict[str, Any]) -> URIRef:
        """Add genome entity and quality metrics."""
        genome_id = genome_data['genome_id']
        genome_uri = GENOME[genome_id]
        
        # Core genome properties
        self.graph.add((genome_uri, RDF.type, KG.Genome))
        self.graph.add((genome_uri, KG.genomeId, Literal(genome_id)))
        
        # Add quality metrics if available
        if 'quality_metrics' in genome_data:
            metrics = genome_data['quality_metrics']
            metrics_uri = GENOME[f"{genome_id}/quality"]
            
            self.graph.add((genome_uri, KG.hasQualityMetrics, metrics_uri))
            self.graph.add((metrics_uri, RDF.type, KG.QualityMetrics))
            
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    datatype = XSD.integer if isinstance(value, int) else XSD.float
                    self.graph.add((metrics_uri, KG[f"quast_{metric}"], Literal(value, datatype=datatype)))
                else:
                    self.graph.add((metrics_uri, KG[f"quast_{metric}"], Literal(str(value))))
        
        logger.info(f"Added genome entity: {genome_id}")
        return genome_uri
    
    def add_gene_protein_entities(self, gene_data: List[Dict[str, Any]], 
                                 genome_uri: URIRef) -> Dict[str, URIRef]:
        """Add gene and protein entities with their relationships."""
        protein_uris = {}
        
        for gene in gene_data:
            gene_id = gene['gene_id']
            gene_uri = GENE[gene_id]
            protein_uri = PROTEIN[gene_id]  # 1:1 mapping for bacterial genes
            
            # Gene properties
            self.graph.add((gene_uri, RDF.type, KG.Gene))
            self.graph.add((gene_uri, KG.belongsToGenome, genome_uri))
            self.graph.add((gene_uri, KG.geneId, Literal(gene_id)))
            
            # Add genomic coordinates from prodigal
            if 'start' in gene and 'end' in gene:
                self.graph.add((gene_uri, KG.startCoordinate, Literal(gene['start'], datatype=XSD.integer)))
                self.graph.add((gene_uri, KG.endCoordinate, Literal(gene['end'], datatype=XSD.integer)))
                
                # Legacy location format for compatibility
                location = f":{gene['start']}-{gene['end']}"
                self.graph.add((gene_uri, KG.hasLocation, Literal(location)))
            
            if 'strand' in gene:
                self.graph.add((gene_uri, KG.strand, Literal(gene['strand'], datatype=XSD.integer)))
            
            if 'length_nt' in gene:
                self.graph.add((gene_uri, KG.lengthNt, Literal(gene['length_nt'], datatype=XSD.integer)))
            
            if 'length_aa' in gene:
                self.graph.add((gene_uri, KG.lengthAA, Literal(gene['length_aa'], datatype=XSD.integer)))
            
            if 'gc_content' in gene:
                self.graph.add((gene_uri, KG.gcContent, Literal(gene['gc_content'], datatype=XSD.float)))
            
            # Protein properties
            self.graph.add((protein_uri, RDF.type, KG.Protein))
            self.graph.add((protein_uri, KG.encodedBy, gene_uri))
            self.graph.add((protein_uri, KG.proteinId, Literal(gene_id)))
            
            # Add protein length from gene data if sequence not available
            if 'length_aa' in gene:
                self.graph.add((protein_uri, KG.length, Literal(gene['length_aa'], datatype=XSD.integer)))
            
            if 'protein_sequence' in gene:
                seq = gene['protein_sequence']
                self.graph.add((protein_uri, KG.sequence, Literal(seq)))
                # Override length with actual sequence length if available
                self.graph.add((protein_uri, KG.length, Literal(len(seq), datatype=XSD.integer)))
            
            protein_uris[gene_id] = protein_uri
        
        logger.info(f"Added {len(gene_data)} gene-protein pairs with genomic coordinates")
        return protein_uris
    
    def add_pfam_domains(self, domains: List[Dict[str, Any]], 
                        protein_uris: Dict[str, URIRef]):
        """Add PFAM domain annotations."""
        domain_count = 0
        
        for domain in domains:
            protein_id = domain['protein_id']
            if protein_id not in protein_uris:
                logger.warning(f"Protein {protein_id} not found for PFAM domain")
                continue
            
            protein_uri = protein_uris[protein_id]
            domain_uri = PROTEIN[domain['domain_id']]
            pfam_uri = PFAM[domain['pfam_id']]
            
            # Domain annotation instance
            self.graph.add((domain_uri, RDF.type, KG.DomainAnnotation))
            self.graph.add((domain_uri, KG.belongsToProtein, protein_uri))
            self.graph.add((domain_uri, KG.domainFamily, pfam_uri))
            self.graph.add((domain_uri, KG.domainStart, Literal(domain['start_pos'], datatype=XSD.integer)))
            self.graph.add((domain_uri, KG.domainEnd, Literal(domain['end_pos'], datatype=XSD.integer)))
            self.graph.add((domain_uri, KG.bitscore, Literal(domain['bitscore'], datatype=XSD.float)))
            self.graph.add((domain_uri, KG.evalue, Literal(domain['evalue'], datatype=XSD.double)))
            
            # PFAM domain family reference
            self.graph.add((pfam_uri, RDF.type, KG.Domain))
            self.graph.add((pfam_uri, KG.pfamAccession, Literal(domain['pfam_id'])))
            
            # Link protein to domain
            self.graph.add((protein_uri, KG.hasDomain, domain_uri))
            
            domain_count += 1
        
        logger.info(f"Added {domain_count} PFAM domain annotations")
    
    def add_kofam_functions(self, functions: List[Dict[str, Any]], 
                           protein_uris: Dict[str, URIRef]):
        """Add KOFAM functional annotations."""
        function_count = 0
        
        for function in functions:
            protein_id = function['protein_id']
            if protein_id not in protein_uris:
                logger.warning(f"Protein {protein_id} not found for KOFAM function")
                continue
            
            protein_uri = protein_uris[protein_id]
            annotation_uri = PROTEIN[function['annotation_id']]
            ko_uri = KO[function['ko_id']]
            
            # Functional annotation instance
            self.graph.add((annotation_uri, RDF.type, KG.FunctionalAnnotation))
            self.graph.add((annotation_uri, KG.annotatesProtein, protein_uri))
            self.graph.add((annotation_uri, KG.assignedFunction, ko_uri))
            self.graph.add((annotation_uri, KG.confidence, Literal(function['confidence'])))
            self.graph.add((annotation_uri, KG.bitscore, Literal(function['bitscore'], datatype=XSD.float)))
            self.graph.add((annotation_uri, KG.evalue, Literal(function['evalue'], datatype=XSD.double)))
            
            # KEGG Ortholog reference
            self.graph.add((ko_uri, RDF.type, KG.KEGGOrtholog))
            self.graph.add((ko_uri, KG.koId, Literal(function['ko_id'])))
            
            # Link protein to function
            self.graph.add((protein_uri, KG.hasFunction, ko_uri))
            
            function_count += 1
        
        logger.info(f"Added {function_count} KOFAM functional annotations")
    
    def add_bgc_annotations(self, bgc_data: Dict[str, Any], 
                           genome_uri: URIRef, protein_uris: Dict[str, URIRef]):
        """Add biosynthetic gene cluster annotations from AntiSMASH."""
        bgc_count = 0
        gene_count = 0
        
        # Define BGC namespace
        BGC_NS = Namespace("http://genome-kg.org/bgc/")
        
        # Process clusters
        for cluster in bgc_data.get("clusters", []):
            cluster_id = f"{cluster.get('record_id', 'unknown')}_{cluster.get('cluster_number', bgc_count)}"
            bgc_uri = BGC_NS[cluster_id]
            
            # BGC properties
            self.graph.add((bgc_uri, RDF.type, KG.BGC))
            self.graph.add((bgc_uri, KG.belongsToGenome, genome_uri))
            self.graph.add((bgc_uri, KG.bgcId, Literal(cluster_id)))
            
            if "start" in cluster and "end" in cluster:
                self.graph.add((bgc_uri, KG.startCoordinate, Literal(cluster["start"], datatype=XSD.integer)))
                self.graph.add((bgc_uri, KG.endCoordinate, Literal(cluster["end"], datatype=XSD.integer)))
                
                # Calculate cluster length
                cluster_length = cluster["end"] - cluster["start"] + 1
                self.graph.add((bgc_uri, KG.lengthNt, Literal(cluster_length, datatype=XSD.integer)))
            
            if "product" in cluster:
                self.graph.add((bgc_uri, KG.bgcProduct, Literal(cluster["product"])))
            
            if "qualifiers" in cluster:
                qualifiers = cluster["qualifiers"]
                if "cluster_number" in qualifiers:
                    self.graph.add((bgc_uri, KG.clusterNumber, Literal(qualifiers["cluster_number"][0])))
            
            # Link genome to BGC
            self.graph.add((genome_uri, KG.hasBGC, bgc_uri))
            
            bgc_count += 1
        
        # Process genes within BGCs
        for gene in bgc_data.get("genes", []):
            if gene.get("feature_type") == "CDS" and "protein_id" in gene:
                protein_id = gene["protein_id"]
                
                # Try to find matching protein URI
                matching_protein_uri = None
                for existing_protein_id, protein_uri in protein_uris.items():
                    if protein_id in existing_protein_id or existing_protein_id in protein_id:
                        matching_protein_uri = protein_uri
                        break
                
                if matching_protein_uri:
                    # Create BGC gene annotation
                    bgc_gene_uri = BGC_NS[f"gene_{protein_id}"]
                    
                    self.graph.add((bgc_gene_uri, RDF.type, KG.BGCGene))
                    self.graph.add((bgc_gene_uri, KG.annotatesProtein, matching_protein_uri))
                    
                    if "product" in gene:
                        self.graph.add((bgc_gene_uri, KG.geneProduct, Literal(gene["product"])))
                    
                    if "gene_kind" in gene:
                        self.graph.add((bgc_gene_uri, KG.geneKind, Literal(gene["gene_kind"])))
                    
                    if "sec_met_domains" in gene:
                        for domain in gene["sec_met_domains"]:
                            self.graph.add((bgc_gene_uri, KG.secMetDomain, Literal(domain)))
                    
                    # Link protein to BGC gene annotation
                    self.graph.add((matching_protein_uri, KG.partOfBGC, bgc_gene_uri))
                    
                    gene_count += 1
                else:
                    logger.warning(f"Could not find matching protein for BGC gene: {protein_id}")
        
        logger.info(f"Added {bgc_count} BGC clusters and {gene_count} BGC gene annotations")
    
    def add_bgc_annotations_with_assignments(self, bgc_data: Dict[str, Any], 
                                           bgc_genome_assignments: Dict[str, URIRef], 
                                           protein_uris: Dict[str, URIRef]):
        """Add biosynthetic gene cluster annotations with per-BGC genome assignments."""
        bgc_count = 0
        gene_count = 0
        
        # Define BGC namespace
        BGC_NS = Namespace("http://genome-kg.org/bgc/")
        
        # Process clusters
        for cluster in bgc_data.get("clusters", []):
            cluster_id = cluster.get('cluster_id', f"unknown_{bgc_count}")
            bgc_uri = BGC_NS[cluster_id]
            
            # Get the correct genome URI for this BGC
            genome_uri = bgc_genome_assignments.get(cluster_id)
            if not genome_uri:
                logger.warning(f"No genome assignment found for BGC {cluster_id}, skipping")
                continue
            
            # BGC properties
            self.graph.add((bgc_uri, RDF.type, KG.BGC))
            self.graph.add((bgc_uri, KG.belongsToGenome, genome_uri))
            self.graph.add((bgc_uri, KG.bgcId, Literal(cluster_id)))
            
            # Add coordinates if available
            if "start" in cluster and "end" in cluster:
                self.graph.add((bgc_uri, KG.startCoordinate, Literal(cluster["start"], datatype=XSD.integer)))
                self.graph.add((bgc_uri, KG.endCoordinate, Literal(cluster["end"], datatype=XSD.integer)))
                
                # Calculate cluster length
                cluster_length = cluster["end"] - cluster["start"] + 1
                self.graph.add((bgc_uri, KG.lengthNt, Literal(cluster_length, datatype=XSD.integer)))
            
            # Add BGC type/product
            if "bgc_type" in cluster:
                self.graph.add((bgc_uri, KG.bgcProduct, Literal(cluster["bgc_type"])))
            
            # Add contig information
            if "contig" in cluster:
                self.graph.add((bgc_uri, KG.contig, Literal(cluster["contig"])))
            
            # Add protein count
            if "proteins" in cluster:
                self.graph.add((bgc_uri, KG.proteinCount, Literal(cluster["proteins"], datatype=XSD.integer)))
            
            # Add GECCO probability scores
            if "average_p" in cluster:
                self.graph.add((bgc_uri, KG.averageProbability, Literal(cluster["average_p"], datatype=XSD.float)))
            if "max_p" in cluster:
                self.graph.add((bgc_uri, KG.maxProbability, Literal(cluster["max_p"], datatype=XSD.float)))
            if "alkaloid_probability" in cluster:
                self.graph.add((bgc_uri, KG.alkaloidProbability, Literal(cluster["alkaloid_probability"], datatype=XSD.float)))
            if "nrp_probability" in cluster:
                self.graph.add((bgc_uri, KG.nrpProbability, Literal(cluster["nrp_probability"], datatype=XSD.float)))
            if "polyketide_probability" in cluster:
                self.graph.add((bgc_uri, KG.polyketideProbability, Literal(cluster["polyketide_probability"], datatype=XSD.float)))
            if "ripp_probability" in cluster:
                self.graph.add((bgc_uri, KG.rippProbability, Literal(cluster["ripp_probability"], datatype=XSD.float)))
            if "saccharide_probability" in cluster:
                self.graph.add((bgc_uri, KG.saccharideProbability, Literal(cluster["saccharide_probability"], datatype=XSD.float)))
            if "terpene_probability" in cluster:
                self.graph.add((bgc_uri, KG.terpeneProbability, Literal(cluster["terpene_probability"], datatype=XSD.float)))
            
            # Add domain information
            if "domains" in cluster:
                self.graph.add((bgc_uri, KG.domains, Literal(cluster["domains"])))
            
            # Link genome to BGC
            self.graph.add((genome_uri, KG.hasBGC, bgc_uri))
            
            # Link BGC genes if available
            if "protein_list" in cluster:
                for protein_id in cluster["protein_list"]:
                    if protein_id in protein_uris:
                        # Convert protein URI to gene URI since partOfBGC should be gene->BGC
                        gene_uri = URIRef(f"http://genome-kg.org/genes/{protein_id}")
                        self.graph.add((gene_uri, KG.partOfBGC, bgc_uri))
                        gene_count += 1
            
            bgc_count += 1
        
        logger.info(f"Added {bgc_count} BGC clusters with correct genome assignments and {gene_count} BGC protein links")
    
    def add_cazyme_annotations(self, cazyme_data: Dict[str, Any], 
                              genome_uri: URIRef, protein_uris: Dict[str, URIRef]):
        """Add CAZyme family annotations from dbCAN."""
        annotation_count = 0
        family_count = 0
        families_added = set()
        
        # Define CAZyme namespace
        CAZYME_NS = Namespace("http://genome-kg.org/cazyme/")
        
        # Process CAZyme annotations
        for annotation in cazyme_data.get("annotations", []):
            protein_id = annotation.get("protein_id")
            cazyme_family = annotation.get("cazyme_family")
            
            if not protein_id or not cazyme_family:
                continue
                
            # Find matching protein URI
            matching_protein_uri = None
            for existing_protein_id, protein_uri in protein_uris.items():
                if existing_protein_id == protein_id or existing_protein_id.endswith(protein_id):
                    matching_protein_uri = protein_uri
                    break
            
            if matching_protein_uri:
                # Create CAZyme annotation instance
                annotation_id = f"{protein_id}_{cazyme_family}_{annotation_count}"
                annotation_uri = CAZYME_NS[annotation_id]
                
                self.graph.add((annotation_uri, RDF.type, KG.CAZymeAnnotation))
                self.graph.add((annotation_uri, KG.annotationId, Literal(annotation_id)))
                
                # Add annotation properties
                if "family_type" in annotation:
                    self.graph.add((annotation_uri, KG.cazymeType, Literal(annotation["family_type"])))
                
                if "evalue" in annotation:
                    self.graph.add((annotation_uri, KG.evalue, Literal(annotation["evalue"], datatype=XSD.float)))
                
                if "coverage" in annotation:
                    self.graph.add((annotation_uri, KG.coverage, Literal(annotation["coverage"], datatype=XSD.float)))
                
                if "start_pos" in annotation and "end_pos" in annotation:
                    self.graph.add((annotation_uri, KG.startPosition, Literal(annotation["start_pos"], datatype=XSD.integer)))
                    self.graph.add((annotation_uri, KG.endPosition, Literal(annotation["end_pos"], datatype=XSD.integer)))
                
                if "ec_number" in annotation and annotation["ec_number"]:
                    self.graph.add((annotation_uri, KG.ecNumber, Literal(annotation["ec_number"])))
                
                # Enhanced: Add substrate prediction if available
                if "substrate_prediction" in annotation and annotation["substrate_prediction"]:
                    self.graph.add((annotation_uri, KG.substrateSpecificity, Literal(annotation["substrate_prediction"])))
                
                # Enhanced: Add HMM length if available
                if "hmm_length" in annotation:
                    self.graph.add((annotation_uri, KG.hmmLength, Literal(annotation["hmm_length"], datatype=XSD.integer)))
                
                # Create CAZyme family if not already added
                if cazyme_family not in families_added:
                    family_uri = CAZYME_NS[f"family_{cazyme_family}"]
                    self.graph.add((family_uri, RDF.type, KG.CAZymeFamily))
                    self.graph.add((family_uri, KG.familyId, Literal(cazyme_family)))
                    
                    if "family_type" in annotation:
                        self.graph.add((family_uri, KG.cazymeType, Literal(annotation["family_type"])))
                    
                    # Enhanced: Add substrate prediction to family level too
                    if "substrate_prediction" in annotation and annotation["substrate_prediction"]:
                        self.graph.add((family_uri, KG.substrateSpecificity, Literal(annotation["substrate_prediction"])))
                    
                    families_added.add(cazyme_family)
                    family_count += 1
                
                # Link annotation to family and protein
                family_uri = CAZYME_NS[f"family_{cazyme_family}"]
                self.graph.add((annotation_uri, KG.cazymeFamily, family_uri))
                self.graph.add((matching_protein_uri, KG.hasCAZyme, annotation_uri))
                
                annotation_count += 1
            else:
                logger.warning(f"Could not find matching protein for CAZyme annotation: {protein_id}")
        
        logger.info(f"Added {annotation_count} CAZyme annotations and {family_count} CAZyme families")
    
    def add_cazyme_annotations_with_correct_genomes(self, cazyme_data: Dict[str, Any], 
                                                   genome_uris: Dict[str, URIRef],
                                                   protein_uris: Dict[str, URIRef],
                                                   protein_to_genome: Dict[str, str]):
        """Add CAZyme family annotations with correct protein-to-genome mapping."""
        annotation_count = 0
        family_count = 0
        families_added = set()
        mapping_stats = {'mapped': 0, 'unmapped': 0}
        
        # Define CAZyme namespace
        CAZYME_NS = Namespace("http://genome-kg.org/cazyme/")
        
        # Process CAZyme annotations
        for annotation in cazyme_data.get("annotations", []):
            protein_id = annotation.get("protein_id")
            cazyme_family = annotation.get("cazyme_family")
            
            if not protein_id or not cazyme_family:
                continue
                
            # Find matching protein URI using exact matching or suffix matching
            matching_protein_uri = None
            for existing_protein_id, protein_uri in protein_uris.items():
                if existing_protein_id == protein_id or existing_protein_id.endswith(protein_id):
                    matching_protein_uri = protein_uri
                    break
            
            if matching_protein_uri:
                # Use protein-to-genome mapping to get the correct genome
                correct_genome_id = None
                for existing_protein_id in protein_uris.keys():
                    if existing_protein_id == protein_id or existing_protein_id.endswith(protein_id):
                        correct_genome_id = protein_to_genome.get(existing_protein_id)
                        break
                
                if correct_genome_id:
                    mapping_stats['mapped'] += 1
                    logger.debug(f"Mapped CAZyme protein {protein_id} to genome {correct_genome_id}")
                else:
                    mapping_stats['unmapped'] += 1
                    logger.warning(f"Could not map CAZyme protein {protein_id} to any genome")
                
                # Create CAZyme annotation instance
                annotation_id = f"{protein_id}_{cazyme_family}_{annotation_count}"
                annotation_uri = CAZYME_NS[annotation_id]
                
                self.graph.add((annotation_uri, RDF.type, KG.CAZymeAnnotation))
                self.graph.add((annotation_uri, KG.annotationId, Literal(annotation_id)))
                
                # Add annotation properties
                if "family_type" in annotation:
                    self.graph.add((annotation_uri, KG.cazymeType, Literal(annotation["family_type"])))
                
                if "evalue" in annotation:
                    self.graph.add((annotation_uri, KG.evalue, Literal(annotation["evalue"], datatype=XSD.float)))
                
                if "coverage" in annotation:
                    self.graph.add((annotation_uri, KG.coverage, Literal(annotation["coverage"], datatype=XSD.float)))
                
                if "start_pos" in annotation and "end_pos" in annotation:
                    self.graph.add((annotation_uri, KG.startPosition, Literal(annotation["start_pos"], datatype=XSD.integer)))
                    self.graph.add((annotation_uri, KG.endPosition, Literal(annotation["end_pos"], datatype=XSD.integer)))
                
                if "ec_number" in annotation and annotation["ec_number"]:
                    self.graph.add((annotation_uri, KG.ecNumber, Literal(annotation["ec_number"])))
                
                # Enhanced: Add substrate prediction if available
                if "substrate_prediction" in annotation and annotation["substrate_prediction"]:
                    self.graph.add((annotation_uri, KG.substrateSpecificity, Literal(annotation["substrate_prediction"])))
                
                # Enhanced: Add HMM length if available
                if "hmm_length" in annotation:
                    self.graph.add((annotation_uri, KG.hmmLength, Literal(annotation["hmm_length"], datatype=XSD.integer)))
                
                # Create CAZyme family if not already added
                if cazyme_family not in families_added:
                    family_uri = CAZYME_NS[f"family_{cazyme_family}"]
                    self.graph.add((family_uri, RDF.type, KG.CAZymeFamily))
                    self.graph.add((family_uri, KG.familyId, Literal(cazyme_family)))
                    
                    if "family_type" in annotation:
                        self.graph.add((family_uri, KG.cazymeType, Literal(annotation["family_type"])))
                    
                    # Enhanced: Add substrate prediction to family level too
                    if "substrate_prediction" in annotation and annotation["substrate_prediction"]:
                        self.graph.add((family_uri, KG.substrateSpecificity, Literal(annotation["substrate_prediction"])))
                    
                    families_added.add(cazyme_family)
                    family_count += 1
                
                # Link annotation to family and protein
                family_uri = CAZYME_NS[f"family_{cazyme_family}"]
                self.graph.add((annotation_uri, KG.cazymeFamily, family_uri))
                self.graph.add((matching_protein_uri, KG.hasCAZyme, annotation_uri))
                
                annotation_count += 1
            else:
                logger.warning(f"Could not find matching protein for CAZyme annotation: {protein_id}")
                mapping_stats['unmapped'] += 1
        
        logger.info(f"Added {annotation_count} CAZyme annotations and {family_count} CAZyme families")
        logger.info(f"Protein mapping stats: {mapping_stats['mapped']} mapped, {mapping_stats['unmapped']} unmapped")
    
    def add_provenance(self, pipeline_data: Dict[str, Any]):
        """Add provenance information for the knowledge graph."""
        kg_uri = URIRef("http://genome-kg.org/this-kg")
        
        self.graph.add((kg_uri, RDF.type, PROV.Entity))
        self.graph.add((kg_uri, PROV.wasGeneratedBy, Literal("genome-kg-pipeline")))
        self.graph.add((kg_uri, PROV.generatedAtTime, Literal(datetime.now(), datatype=XSD.dateTime)))
        
        if 'version' in pipeline_data:
            self.graph.add((kg_uri, KG.pipelineVersion, Literal(pipeline_data['version'])))
        
        if 'astra_databases' in pipeline_data:
            for db in pipeline_data['astra_databases']:
                self.graph.add((kg_uri, KG.usedDatabase, Literal(db)))
    
    def save_graph(self, output_file: Path, format: str = 'turtle'):
        """Save the knowledge graph to file."""
        try:
            # Serialize to string first, then write to file
            serialized = self.graph.serialize(format=format)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(serialized)
            
            triple_count = len(self.graph)
            logger.info(f"Saved knowledge graph with {triple_count:,} triples to {output_file}")
            
            return {
                "output_file": str(output_file),
                "format": format,
                "triple_count": triple_count,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
            raise


def build_knowledge_graph_from_pipeline(stage03_dir: Path, stage04_dir: Path, 
                                       output_dir: Path) -> Dict[str, Any]:
    """
    Build complete knowledge graph from pipeline results.
    
    Args:
        stage03_dir: Prodigal output directory
        stage04_dir: Astra annotation output directory  
        output_dir: Output directory for knowledge graph
        
    Returns:
        Dict containing build statistics and output files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize builder
    builder = GenomeKGBuilder()
    
    # Load prodigal manifest to get genome/gene information
    prodigal_manifest_file = stage03_dir / "processing_manifest.json"
    with open(prodigal_manifest_file) as f:
        prodigal_manifest = json.load(f)
    
    # Load astra results
    annotation_results = process_astra_results(stage04_dir)
    
    # Build protein URIs mapping for linking annotations
    protein_uris = {}
    genome_uris = {}  # Track genome ID -> URI mapping for protein mapping
    
    # Process each genome from prodigal results
    for genome_result in prodigal_manifest['genomes']:
        if genome_result['execution_status'] != 'success':
            continue
            
        genome_id = genome_result['genome_id']
        
        # Add genome entity
        genome_data = {
            'genome_id': genome_id,
            'quality_metrics': {}  # TODO: Load from QUAST results
        }
        genome_uri = builder.add_genome_entity(genome_data)
        genome_uris[genome_id] = genome_uri  # Store for protein mapping
        
        # Load protein sequences from prodigal output for this genome
        protein_file = stage03_dir / "genomes" / genome_id / f"{genome_id}.faa"
        gene_data = []
        
        if protein_file.exists():
            with open(protein_file, 'r') as f:
                for line in f:
                    if line.startswith('>'):
                        # Parse full prodigal header including genomic coordinates
                        gene_info = parse_prodigal_header(line)
                        gene_data.append(gene_info)
        
        genome_protein_uris = builder.add_gene_protein_entities(gene_data, genome_uri)
        protein_uris.update(genome_protein_uris)
    
    # Add PFAM domain annotations
    builder.add_pfam_domains(annotation_results['pfam_domains'], protein_uris)
    
    # Add KOFAM functional annotations  
    builder.add_kofam_functions(annotation_results['kofam_functions'], protein_uris)
    
    # Add provenance
    builder.add_provenance({
        'version': '0.1.0',
        'astra_databases': ['PFAM', 'KOFAM']
    })
    
    # Enrich with functional annotations from reference databases
    enriched_graph, enrichment_stats = add_functional_enrichment_to_pipeline(builder.graph)
    builder.graph = enriched_graph
    
    # Integrate KEGG pathways
    logger.info("Integrating KEGG pathways...")
    repo_root = Path(__file__).parent.parent.parent
    ko_pathway_file = repo_root / "data/reference/ko_pathway.list"
    
    # Extract KO IDs that are actually found in our proteins
    found_ko_ids = set()
    for func in annotation_results['kofam_functions']:
        found_ko_ids.add(func['ko_id'])
    
    logger.info(f"Found {len(found_ko_ids)} unique KO IDs in protein annotations")
    
    pathway_stats = {'pathways_integrated': 0, 'ko_pathway_relationships': 0}
    if ko_pathway_file.exists():
        # Create pathway integration in temporary directory  
        pathway_temp_dir = output_dir / "temp_pathways"
        pathway_rdf_file = integrate_pathways(ko_pathway_file, pathway_temp_dir, found_ko_ids)
        
        # Load and merge pathway graph into main graph
        pathway_graph = Graph()
        pathway_graph.parse(str(pathway_rdf_file), format='turtle')
        
        # Merge pathway graph into main graph
        for triple in pathway_graph:
            builder.graph.add(triple)
        
        # Get pathway statistics
        pathway_stats['pathways_integrated'] = len([s for s in pathway_graph.subjects(RDF.type, None) 
                                                   if 'pathway/' in str(s)])
        pathway_stats['ko_pathway_relationships'] = len(list(pathway_graph.triples((None, URIRef("http://genomics.ai/kg/participatesIn"), None))))
        
        logger.info(f"Integrated {pathway_stats['pathways_integrated']} pathways with {pathway_stats['ko_pathway_relationships']} relationships")
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(pathway_temp_dir, ignore_errors=True)
    else:
        logger.warning(f"ko_pathway.list not found at {ko_pathway_file}, skipping pathway integration")
    
    # Save knowledge graph
    kg_file = output_dir / "knowledge_graph.ttl"
    save_stats = builder.save_graph(kg_file, format='turtle')
    
    # Generate summary statistics
    stats = {
        'total_triples': save_stats['triple_count'],
        'genomes_processed': len([g for g in prodigal_manifest['genomes'] 
                                if g['execution_status'] == 'success']),
        'proteins_annotated': len(protein_uris),
        'pfam_domains': len(annotation_results['pfam_domains']),
        'kofam_functions': len(annotation_results['kofam_functions']),
        'functional_enrichment': enrichment_stats,
        'pathway_integration': pathway_stats,
        'output_files': {
            'knowledge_graph': str(kg_file)
        }
    }
    
    # Save statistics
    stats_file = output_dir / "build_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Knowledge graph build completed: {stats}")
    return stats


def build_knowledge_graph_with_extended_annotations(stage03_dir: Path, stage04_dir: Path, 
                                                   stage05a_dir: Optional[Path], 
                                                   stage05b_dir: Optional[Path],
                                                   output_dir: Path) -> Dict[str, Any]:
    """
    Build complete knowledge graph from pipeline results including BGC and CAZyme annotations.
    
    Args:
        stage03_dir: Prodigal output directory
        stage04_dir: Astra annotation output directory
        stage05a_dir: AntiSMASH BGC output directory (optional)
        stage05b_dir: dbCAN CAZyme output directory (optional)
        output_dir: Output directory for knowledge graph
        
    Returns:
        Dict containing build statistics and output files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize builder
    builder = GenomeKGBuilder()
    
    # Load prodigal manifest to get genome/gene information
    prodigal_manifest_file = stage03_dir / "processing_manifest.json"
    with open(prodigal_manifest_file) as f:
        prodigal_manifest = json.load(f)
    
    # Load astra results
    annotation_results = process_astra_results(stage04_dir)
    
    # Load BGC results if available
    bgc_results = {}
    if stage05a_dir and stage05a_dir.exists():
        bgc_manifest_file = stage05a_dir / "processing_manifest.json"
        if bgc_manifest_file.exists():
            with open(bgc_manifest_file) as f:
                bgc_manifest = json.load(f)
            
            # Load combined BGC data
            bgc_data_file = stage05a_dir / "combined_bgc_data.json"
            if bgc_data_file.exists():
                with open(bgc_data_file) as f:
                    bgc_results = json.load(f)
                logger.info(f"Loaded BGC data: {len(bgc_results.get('clusters', []))} clusters, {len(bgc_results.get('genes', []))} genes")
            else:
                logger.warning(f"BGC data file not found: {bgc_data_file}")
        else:
            logger.warning(f"BGC manifest not found: {bgc_manifest_file}")
    else:
        logger.info("No BGC directory provided, skipping BGC annotations")
    
    # Load CAZyme results if available
    cazyme_results = {}
    if stage05b_dir and stage05b_dir.exists():
        cazyme_manifest_file = stage05b_dir / "processing_manifest.json"
        if cazyme_manifest_file.exists():
            with open(cazyme_manifest_file) as f:
                cazyme_manifest = json.load(f)
            
            # Load CAZyme summary data
            cazyme_summary_file = stage05b_dir / "dbcan_summary.json"
            if cazyme_summary_file.exists():
                with open(cazyme_summary_file) as f:
                    cazyme_summary = json.load(f)
                
                # Combine all individual CAZyme results
                all_annotations = []
                for genome_id in cazyme_summary.get('genome_results', {}):
                    result_file = stage05b_dir / f"{genome_id}_cazyme_results.json"
                    if result_file.exists():
                        with open(result_file) as f:
                            genome_cazyme_data = json.load(f)
                            all_annotations.extend(genome_cazyme_data.get('annotations', []))
                
                cazyme_results = {'annotations': all_annotations}
                logger.info(f"Loaded CAZyme data: {len(all_annotations)} annotations from {len(cazyme_summary.get('genome_results', {}))} genomes")
            else:
                logger.warning(f"CAZyme summary file not found: {cazyme_summary_file}")
        else:
            logger.warning(f"CAZyme manifest not found: {cazyme_manifest_file}")
    else:
        logger.info("No CAZyme directory provided, skipping CAZyme annotations")
    
    # Build protein URIs mapping for linking annotations
    protein_uris = {}
    genome_uris = {}
    
    # Process each genome from prodigal results
    for genome_result in prodigal_manifest['genomes']:
        if genome_result['execution_status'] != 'success':
            continue
            
        genome_id = genome_result['genome_id']
        
        # Add genome entity
        genome_data = {
            'genome_id': genome_id,
            'quality_metrics': {}  # TODO: Load from QUAST results
        }
        genome_uri = builder.add_genome_entity(genome_data)
        genome_uris[genome_id] = genome_uri
        
        # Load protein sequences from prodigal output for this genome
        protein_file = stage03_dir / "genomes" / genome_id / f"{genome_id}.faa"
        gene_data = []
        
        if protein_file.exists():
            with open(protein_file, 'r') as f:
                for line in f:
                    if line.startswith('>'):
                        # Parse full prodigal header including genomic coordinates
                        gene_info = parse_prodigal_header(line)
                        gene_data.append(gene_info)
        
        genome_protein_uris = builder.add_gene_protein_entities(gene_data, genome_uri)
        protein_uris.update(genome_protein_uris)
    
    # Add PFAM domain annotations
    builder.add_pfam_domains(annotation_results['pfam_domains'], protein_uris)
    
    # Add KOFAM functional annotations  
    builder.add_kofam_functions(annotation_results['kofam_functions'], protein_uris)
    
    # Add BGC annotations if available with efficient genome mapping
    bgc_stats = {'clusters': 0, 'genes': 0}
    if bgc_results:
        if genome_uris:
            # Build efficient contig-to-genome mapping from existing graph relationships
            logger.info("Building contig-to-genome index from existing protein-genome relationships...")
            contig_to_genome = build_contig_to_genome_index_from_proteins(builder.graph, protein_uris)
            
            # Assign each BGC to its correct genome
            bgc_genome_assignments = assign_bgc_to_correct_genome(bgc_results, contig_to_genome)
            
            # Add BGC annotations with correct assignments
            if bgc_genome_assignments:
                builder.add_bgc_annotations_with_assignments(bgc_results, bgc_genome_assignments, protein_uris)
                bgc_stats['clusters'] = len(bgc_results.get('clusters', []))
                bgc_stats['genes'] = len(bgc_results.get('genes', []))
            else:
                logger.warning("No BGC genome assignments could be made - BGCs will be skipped")
        else:
            logger.warning("No genome URIs available for BGC assignment")
    
    # Add CAZyme annotations if available
    cazyme_stats = {'annotations': 0, 'families': 0}
    if cazyme_results:
        if genome_uris:
            # Build protein-to-genome mapping to correctly assign CAZyme annotations
            logger.info("Building protein-to-genome mapping for CAZyme annotations...")
            protein_to_genome = build_protein_to_genome_mapping(protein_uris, genome_uris)
            
            # Add CAZyme annotations with correct genome assignments
            builder.add_cazyme_annotations_with_correct_genomes(cazyme_results, genome_uris, protein_uris, protein_to_genome)
            cazyme_stats['annotations'] = len(cazyme_results.get('annotations', []))
            # Count unique families
            families = set(ann.get('cazyme_family') for ann in cazyme_results.get('annotations', []))
            cazyme_stats['families'] = len(families)
    
    # Add provenance
    databases_used = ['PFAM', 'KOFAM']
    if bgc_results:
        databases_used.append('AntiSMASH')
    if cazyme_results:
        databases_used.append('dbCAN')
    
    builder.add_provenance({
        'version': '0.1.0',
        'astra_databases': databases_used
    })
    
    # Enrich with functional annotations from reference databases
    enriched_graph, enrichment_stats = add_functional_enrichment_to_pipeline(builder.graph)
    builder.graph = enriched_graph
    
    # Integrate KEGG pathways
    logger.info("Integrating KEGG pathways...")
    repo_root = Path(__file__).parent.parent.parent
    ko_pathway_file = repo_root / "data/reference/ko_pathway.list"
    
    # Extract KO IDs that are actually found in our proteins
    found_ko_ids = set()
    for func in annotation_results['kofam_functions']:
        found_ko_ids.add(func['ko_id'])
    
    logger.info(f"Found {len(found_ko_ids)} unique KO IDs in protein annotations")
    
    pathway_stats = {'pathways_integrated': 0, 'ko_pathway_relationships': 0}
    if ko_pathway_file.exists():
        # Create pathway integration in temporary directory  
        pathway_temp_dir = output_dir / "temp_pathways"
        pathway_rdf_file = integrate_pathways(ko_pathway_file, pathway_temp_dir, found_ko_ids)
        
        # Load and merge pathway graph into main graph
        pathway_graph = Graph()
        pathway_graph.parse(str(pathway_rdf_file), format='turtle')
        
        # Merge pathway graph into main graph
        for triple in pathway_graph:
            builder.graph.add(triple)
        
        # Get pathway statistics
        pathway_stats['pathways_integrated'] = len([s for s in pathway_graph.subjects(RDF.type, None) 
                                                   if 'pathway/' in str(s)])
        pathway_stats['ko_pathway_relationships'] = len(list(pathway_graph.triples((None, URIRef("http://genomics.ai/kg/participatesIn"), None))))
        
        logger.info(f"Integrated {pathway_stats['pathways_integrated']} pathways with {pathway_stats['ko_pathway_relationships']} relationships")
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(pathway_temp_dir, ignore_errors=True)
    else:
        logger.warning(f"ko_pathway.list not found at {ko_pathway_file}, skipping pathway integration")
    
    # Save knowledge graph
    kg_file = output_dir / "knowledge_graph.ttl"
    save_stats = builder.save_graph(kg_file, format='turtle')
    
    # Generate summary statistics
    stats = {
        'total_triples': save_stats['triple_count'],
        'genomes_processed': len([g for g in prodigal_manifest['genomes'] 
                                if g['execution_status'] == 'success']),
        'proteins_annotated': len(protein_uris),
        'pfam_domains': len(annotation_results['pfam_domains']),
        'kofam_functions': len(annotation_results['kofam_functions']),
        'bgc_clusters': bgc_stats['clusters'],
        'bgc_genes': bgc_stats['genes'],
        'cazyme_annotations': cazyme_stats['annotations'],
        'cazyme_families': cazyme_stats['families'],
        'functional_enrichment': enrichment_stats,
        'pathway_integration': pathway_stats,
        'output_files': {
            'knowledge_graph': str(kg_file)
        }
    }
    
    # Save statistics
    stats_file = output_dir / "build_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Knowledge graph build completed with extended annotations: {stats}")
    return stats


def build_knowledge_graph_with_bgc(stage03_dir: Path, stage04_dir: Path, 
                                  stage05a_dir: Optional[Path], output_dir: Path) -> Dict[str, Any]:
    """
    Backward compatibility wrapper for BGC-only knowledge graph building.
    
    This function maintains compatibility with existing code that only uses BGC annotations.
    For new code, use build_knowledge_graph_with_extended_annotations() instead.
    """
    return build_knowledge_graph_with_extended_annotations(
        stage03_dir=stage03_dir,
        stage04_dir=stage04_dir, 
        stage05a_dir=stage05a_dir,
        stage05b_dir=None,  # No CAZyme annotations
        output_dir=output_dir
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python rdf_builder.py <stage03_dir> <stage04_dir> <output_dir>")
        sys.exit(1)
    
    stage03_dir = Path(sys.argv[1])
    stage04_dir = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])
    
    logging.basicConfig(level=logging.INFO)
    
    stats = build_knowledge_graph_from_pipeline(stage03_dir, stage04_dir, output_dir)
    print(f"Knowledge graph built successfully: {stats['total_triples']:,} triples")