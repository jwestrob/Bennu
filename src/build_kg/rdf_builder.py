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


# Define ontology namespaces
KG = Namespace("http://genome-kg.org/ontology/")
GENOME = Namespace("http://genome-kg.org/genomes/")
GENE = Namespace("http://genome-kg.org/genes/")
PROTEIN = Namespace("http://genome-kg.org/proteins/")
PFAM = Namespace("http://pfam.xfam.org/family/")
KO = Namespace("http://www.genome.jp/kegg/ko/")
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
            (KG.Domain, "Protein domain family (PFAM)")
        ]
        
        properties = [
            (KG.belongsToGenome, "belongs to genome"),
            (KG.encodedBy, "protein encoded by gene"),
            (KG.hasDomain, "protein has domain"),
            (KG.hasFunction, "protein has function"),
            (KG.domainFamily, "domain belongs to family"),
            (KG.hasQualityMetrics, "genome has quality metrics")
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