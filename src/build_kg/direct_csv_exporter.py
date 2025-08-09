#!/usr/bin/env python3
"""
Direct CSV Export from RDF Graph

Generates Neo4j CSV files directly from rdflib.Graph without RDF serialization.
This eliminates the expensive serialize→parse→convert pipeline bottleneck.
"""

import csv
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import rdflib
from rdflib import Graph, URIRef, Literal, Namespace
from rich.console import Console
from rich.progress import Progress, TaskID

console = Console()
logger = logging.getLogger(__name__)

# Define namespaces (matching rdf_builder.py exactly)
KG = Namespace("http://genome-kg.org/ontology/")
GENOME = Namespace("http://genome-kg.org/genomes/")
GENE = Namespace("http://genome-kg.org/genes/")
PROTEIN = Namespace("http://genome-kg.org/proteins/")
PFAM = Namespace("http://pfam.xfam.org/family/")
KO = Namespace("http://www.genome.jp/kegg/ko/")
CAZYME = Namespace("http://genome-kg.org/cazyme/")
PROV = Namespace("http://www.w3.org/ns/prov#")


class DirectCSVExporter:
    """Generate Neo4j CSV files directly from rdflib.Graph without serialization."""
    
    def __init__(self, graph: Graph, output_dir: Path):
        """
        Initialize CSV exporter.
        
        Args:
            graph: In-memory rdflib.Graph containing all RDF data
            output_dir: Directory to write CSV files for Neo4j bulk import
        """
        self.graph = graph
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track export statistics
        self.stats = {
            "export_start_time": None,
            "export_end_time": None,
            "files_generated": [],
            "total_nodes": 0,
            "total_relationships": 0,
            "nodes_by_type": {},
            "relationships_by_type": {}
        }
        
    def export_all(self) -> Dict[str, Any]:
        """Export all CSV files needed for Neo4j bulk import."""
        logger.info("Starting direct CSV export from in-memory RDF graph...")
        self.stats["export_start_time"] = time.time()
        
        with Progress(console=console) as progress:
            overall_task = progress.add_task("Exporting CSV files...", total=22)
            
            # Export node CSV files
            progress.update(overall_task, description="Exporting genomes...")
            self._export_genomes()
            progress.advance(overall_task)
            
            progress.update(overall_task, description="Exporting contigs...")
            self._export_contigs()
            progress.advance(overall_task)
            
            progress.update(overall_task, description="Exporting genes...")
            self._export_genes()
            progress.advance(overall_task)
            
            progress.update(overall_task, description="Exporting proteins...")
            self._export_proteins()
            progress.advance(overall_task)
            
            progress.update(overall_task, description="Exporting PFAM domains...")
            self._export_pfam_domains()
            progress.advance(overall_task)
            
            progress.update(overall_task, description="Exporting KEGG functions...")
            self._export_kegg_functions()
            progress.advance(overall_task)
            
            progress.update(overall_task, description="Exporting pathways...")
            self._export_pathways()
            progress.advance(overall_task)
            
            progress.update(overall_task, description="Exporting domain annotations...")
            self._export_domain_annotations()
            progress.advance(overall_task)
            
            progress.update(overall_task, description="Exporting functional annotations...")
            self._export_functional_annotations()
            progress.advance(overall_task)
            
            progress.update(overall_task, description="Exporting quality metrics...")
            self._export_quality_metrics()
            progress.advance(overall_task)
            
            # Skip datasets and entities - these RDF types are not created by current pipeline
            # progress.update(overall_task, description="Exporting datasets...")
            # self._export_datasets()
            # progress.advance(overall_task)
            # 
            # progress.update(overall_task, description="Exporting entities...")
            # self._export_entities()
            # progress.advance(overall_task)
            
            # Export relationship CSV files
            progress.update(overall_task, description="Exporting encoded-by relationships...")
            self._export_encodedby_relationships()
            progress.advance(overall_task)
            
            progress.update(overall_task, description="Exporting belongs-to-genome relationships...")
            self._export_belongstogenome_relationships()
            progress.advance(overall_task)
            
            progress.update(overall_task, description="Exporting gene-to-contig relationships...")
            self._export_belongstocontig_relationships()
            progress.advance(overall_task)
            
            progress.update(overall_task, description="Exporting belongs-to-protein relationships...")
            self._export_belongstoprotein_relationships()
            progress.advance(overall_task)
            
            progress.update(overall_task, description="Exporting domain-family relationships...")
            self._export_domainfamily_relationships()
            progress.advance(overall_task)
            
            progress.update(overall_task, description="Exporting annotates-protein relationships...")
            self._export_annotatesprotein_relationships()
            progress.advance(overall_task)
            
            progress.update(overall_task, description="Exporting assigned-function relationships...")
            self._export_assignedfunction_relationships()
            progress.advance(overall_task)
            
            progress.update(overall_task, description="Exporting has-participant relationships...")
            self._export_hasparticipant_relationships()
            progress.advance(overall_task)
            
            progress.update(overall_task, description="Exporting has-quality-metrics relationships...")
            self._export_hasqualitymetrics_relationships()
            progress.advance(overall_task)
            
            progress.update(overall_task, description="Exporting protein-domain relationships...")
            self._export_protein_domain_relationships()
            progress.advance(overall_task)
            
            progress.update(overall_task, description="Exporting protein-function relationships...")
            self._export_protein_function_relationships()
            progress.advance(overall_task)
            
            progress.update(overall_task, description="Exporting function-pathway relationships...")
            self._export_function_pathway_relationships()
            progress.advance(overall_task)
            
        
        self.stats["export_end_time"] = time.time()
        export_time = self.stats["export_end_time"] - self.stats["export_start_time"]
        
        logger.info(f"Direct CSV export completed in {export_time:.1f} seconds")
        logger.info(f"Generated {len(self.stats['files_generated'])} CSV files")
        logger.info(f"Total nodes: {self.stats['total_nodes']:,}")
        logger.info(f"Total relationships: {self.stats['total_relationships']:,}")
        
        self.stats["export_time_seconds"] = export_time
        return self.stats
    
    def _export_genomes(self):
        """Export genome nodes to genomes.csv"""
        filename = "genomes.csv"
        filepath = self.output_dir / filename
        
        # SPARQL query to extract genome data
        query = """
        PREFIX kg: <http://genome-kg.org/ontology/>
        SELECT ?genome_id WHERE {
            ?genome rdf:type kg:Genome .
            ?genome kg:genomeId ?genome_id .
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Neo4j CSV headers
            writer.writerow(['id:ID', 'genomeId'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    str(row.genome_id) if row.genome_id else '',
                    str(row.genome_id) if row.genome_id else ''
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["nodes_by_type"]["Genome"] = count
        self.stats["total_nodes"] += count
        logger.info(f"Exported {count:,} genomes to {filename}")
    
    def _export_genes(self):
        """Export gene nodes to genes.csv"""
        filename = "genes.csv"
        filepath = self.output_dir / filename
        
        query = """
        PREFIX kg: <http://genome-kg.org/ontology/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?gene_id ?start_coord ?end_coord ?strand ?gc_content ?length_aa ?length_nt ?location ?contig WHERE {
            ?gene rdf:type kg:Gene .
            ?gene kg:geneId ?gene_id .
            OPTIONAL { ?gene kg:startCoordinate ?start_coord }
            OPTIONAL { ?gene kg:endCoordinate ?end_coord }
            OPTIONAL { ?gene kg:strand ?strand }
            OPTIONAL { ?gene kg:gcContent ?gc_content }
            OPTIONAL { ?gene kg:lengthAA ?length_aa }
            OPTIONAL { ?gene kg:lengthNt ?length_nt }
            OPTIONAL { ?gene kg:hasLocation ?location }
            OPTIONAL { ?gene kg:contig ?contig }
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id:ID', 'endCoordinate', 'gcContent', 'geneId', 'hasLocation', 'lengthAA', 'lengthNt', 'startCoordinate', 'strand', 'contig'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    f"gene:{row.gene_id}" if row.gene_id else '',
                    int(row.end_coord) if row.end_coord else 0,
                    float(row.gc_content) if row.gc_content else 0.0,
                    str(row.gene_id) if row.gene_id else '',
                    str(row.location) if row.location else '',
                    int(row.length_aa) if row.length_aa else 0,
                    int(row.length_nt) if row.length_nt else 0,
                    int(row.start_coord) if row.start_coord else 0,
                    int(row.strand) if row.strand else 0,
                    str(row.contig) if row.contig else ''
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["nodes_by_type"]["Gene"] = count
        self.stats["total_nodes"] += count
        logger.info(f"Exported {count:,} genes to {filename}")
    
    def _export_proteins(self):
        """Export protein nodes to proteins.csv"""
        filename = "proteins.csv"
        filepath = self.output_dir / filename
        
        query = """
        PREFIX kg: <http://genome-kg.org/ontology/>
        SELECT ?protein_id ?length WHERE {
            ?protein rdf:type kg:Protein .
            ?protein kg:proteinId ?protein_id .
            OPTIONAL { ?protein kg:length ?length }
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id:ID', 'length', 'proteinId'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    f"protein:{row.protein_id}" if row.protein_id else '',
                    int(row.length) if row.length else 0,
                    str(row.protein_id) if row.protein_id else ''
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["nodes_by_type"]["Protein"] = count
        self.stats["total_nodes"] += count
        logger.info(f"Exported {count:,} proteins to {filename}")
    
    def _export_pfam_domains(self):
        """Export PFAM domain nodes to domains.csv"""
        filename = "domains.csv"
        filepath = self.output_dir / filename
        
        query = """
        PREFIX kg: <http://genome-kg.org/ontology/>
        PREFIX pfam: <http://pfam.xfam.org/family/>
        SELECT DISTINCT ?pfam_id ?name ?description WHERE {
            ?domain_uri rdf:type kg:Domain .
            ?domain_uri kg:pfamAccession ?pfam_id .
            OPTIONAL { ?domain_uri kg:name ?name }
            OPTIONAL { ?domain_uri kg:description ?description }
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id:ID', 'pfamAccession', 'description', 'familyType'])
            
            for row in self.graph.query(query):
                pfam_id = str(row.pfam_id) if row.pfam_id else ''
                writer.writerow([
                    pfam_id,
                    pfam_id,  # pfamAccession same as ID
                    str(row.description) if row.description else '',
                    'Domain'  # Default familyType
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["nodes_by_type"]["PFAMDomain"] = count
        self.stats["total_nodes"] += count
        logger.info(f"Exported {count:,} PFAM domains to {filename}")
    
    def _export_kegg_functions(self):
        """Export KEGG function nodes to keggorthologs.csv"""
        filename = "keggorthologs.csv"
        filepath = self.output_dir / filename
        
        query = """
        PREFIX kg: <http://genome-kg.org/ontology/>
        PREFIX ko: <http://www.genome.jp/kegg/ko/>
        SELECT DISTINCT ?kegg_id ?definition WHERE {
            ?ko_uri rdf:type kg:KEGGOrtholog .
            ?ko_uri kg:koId ?kegg_id .
            OPTIONAL { ?ko_uri kg:description ?definition }
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id:ID', 'description', 'koId'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    str(row.kegg_id) if row.kegg_id else '',
                    str(row.definition) if row.definition else '',
                    str(row.kegg_id) if row.kegg_id else ''
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["nodes_by_type"]["KEGGFunction"] = count  
        self.stats["total_nodes"] += count
        logger.info(f"Exported {count:,} KEGG functions to {filename}")
    
    def _export_pathways(self):
        """Export pathway nodes to pathways.csv"""
        filename = "pathways.csv"
        filepath = self.output_dir / filename
        
        query = """
        PREFIX kg: <http://genomics.ai/kg/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?pathway_id ?name ?description WHERE {
            ?pathway_uri ?pred ?obj .
            FILTER(STRSTARTS(STR(?pathway_uri), "http://genomics.ai/kg/pathway/"))
            BIND(STRAFTER(STR(?pathway_uri), "http://genomics.ai/kg/pathway/") AS ?pathway_id)
            OPTIONAL { ?pathway_uri rdfs:label ?name }
            OPTIONAL { ?pathway_uri kg:description ?description }
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id:ID', 'pathwayNumber', 'description'])
            
            for row in self.graph.query(query):
                pathway_id = str(row.pathway_id) if row.pathway_id else ''
                # Extract pathway number from ID (e.g., map00541 -> 00541)
                pathway_number = pathway_id.replace('map', '') if pathway_id.startswith('map') else pathway_id
                writer.writerow([
                    f"pathway:{pathway_id}",
                    pathway_number,
                    str(row.description) if row.description else ''
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["nodes_by_type"]["Pathway"] = count
        self.stats["total_nodes"] += count
        logger.info(f"Exported {count:,} pathways to {filename}")
    
    def _export_domain_annotations(self):
        """Export domain annotation nodes to domainannotations.csv"""
        filename = "domainannotations.csv"
        filepath = self.output_dir / filename
        
        query = """
        PREFIX kg: <http://genome-kg.org/ontology/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT DISTINCT ?annotation_id ?protein_id ?evalue ?bitscore ?domain_start ?domain_end 
        (GROUP_CONCAT(DISTINCT ?pfam_id; separator="|") AS ?pfam_ids) WHERE {
            ?annotation rdf:type kg:DomainAnnotation .
            ?annotation kg:belongsToProtein ?protein_uri .
            ?protein_uri kg:proteinId ?protein_id .
            OPTIONAL { ?annotation kg:domainFamily ?pfam_family .
                       ?pfam_family kg:pfamAccession ?pfam_id }
            OPTIONAL { ?annotation kg:evalue ?evalue }
            OPTIONAL { ?annotation kg:bitscore ?bitscore }
            OPTIONAL { ?annotation kg:domainStart ?domain_start }
            OPTIONAL { ?annotation kg:domainEnd ?domain_end }
            BIND(STRAFTER(STR(?annotation), "http://genome-kg.org/proteins/") AS ?annotation_id)
        }
        GROUP BY ?annotation_id ?protein_id ?evalue ?bitscore ?domain_start ?domain_end
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id:ID', 'bitscore', 'domainEnd', 'domainStart', 'evalue'])
            
            for row in self.graph.query(query):
                # annotation_id already has "proteins/" stripped, add "protein:" prefix
                annotation_id = f"protein:{row.annotation_id}" if row.annotation_id else ''
                writer.writerow([
                    annotation_id,
                    float(row.bitscore) if row.bitscore else '',
                    int(row.domain_end) if row.domain_end else '',
                    int(row.domain_start) if row.domain_start else '',
                    float(row.evalue) if row.evalue else ''
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["nodes_by_type"]["DomainAnnotation"] = count
        self.stats["total_nodes"] += count
        logger.info(f"Exported {count:,} domain annotations to {filename}")
    
    def _export_functional_annotations(self):
        """Export functional annotation nodes to functionalannotations.csv"""
        filename = "functionalannotations.csv"
        filepath = self.output_dir / filename
        
        query = """
        PREFIX kg: <http://genome-kg.org/ontology/>
        SELECT DISTINCT ?annotation_uri ?bitscore ?confidence ?evalue WHERE {
            ?annotation_uri kg:bitscore ?bitscore .
            ?annotation_uri kg:confidence ?confidence .
            ?annotation_uri kg:evalue ?evalue .
            FILTER(CONTAINS(STR(?annotation_uri), "/function/"))
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id:ID', 'bitscore', 'confidence', 'evalue'])
            
            for row in self.graph.query(query):
                annotation_id = str(row.annotation_uri).replace("http://genome-kg.org/proteins/", "function:")
                writer.writerow([
                    annotation_id,
                    float(row.bitscore) if row.bitscore else '',
                    row.confidence or '',
                    float(row.evalue) if row.evalue else ''
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["nodes_by_type"]["FunctionalAnnotation"] = count
        self.stats["total_nodes"] += count
        logger.info(f"Exported {count:,} functional annotations to {filename}")
    
    def _export_quality_metrics(self):
        """Export quality metrics nodes to qualitymetrics.csv"""
        filename = "qualitymetrics.csv"
        filepath = self.output_dir / filename
        
        query = """
        PREFIX kg: <http://genome-kg.org/ontology/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?metric_id ?total_length ?n50 ?num_contigs ?gc_content ?largest_contig WHERE {
            ?metric rdf:type kg:QualityMetrics .
            BIND(STRAFTER(STR(?metric), "http://genome-kg.org/genomes/") AS ?metric_id)
            OPTIONAL { ?metric kg:quast_totalLength ?total_length }
            OPTIONAL { ?metric kg:quast_n50 ?n50 }
            OPTIONAL { ?metric kg:quast_numContigs ?num_contigs }
            OPTIONAL { ?metric kg:quast_gcContent ?gc_content }
            OPTIONAL { ?metric kg:quast_largestContig ?largest_contig }
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id:ID', 'totalLength:long', 'n50:long', 'numContigs:int', 'gcContent:float', 'largestContig:long'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    str(row.metric_id) if row.metric_id else '',
                    int(row.total_length) if row.total_length else 0,
                    int(row.n50) if row.n50 else 0,
                    int(row.num_contigs) if row.num_contigs else 0,
                    float(row.gc_content) if row.gc_content else 0.0,
                    int(row.largest_contig) if row.largest_contig else 0
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["nodes_by_type"]["QualityMetric"] = count
        self.stats["total_nodes"] += count
        logger.info(f"Exported {count:,} quality metrics to {filename}")
    
    def _export_datasets(self):
        """Export dataset nodes to datasets.csv"""
        filename = "datasets.csv"
        filepath = self.output_dir / filename
        
        query = """
        PREFIX kg: <http://genome-kg.org/ontology/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?dataset_id ?name ?description WHERE {
            ?dataset rdf:type kg:Dataset .
            ?dataset kg:datasetId ?dataset_id .
            OPTIONAL { ?dataset kg:name ?name }
            OPTIONAL { ?dataset kg:description ?description }
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id:ID', 'name', 'description'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    str(row.dataset_id) if row.dataset_id else '',
                    str(row.name) if row.name else '',
                    str(row.description) if row.description else ''
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["nodes_by_type"]["Dataset"] = count
        self.stats["total_nodes"] += count
        logger.info(f"Exported {count:,} datasets to {filename}")
    
    def _export_entities(self):
        """Export entity nodes to entities.csv"""
        filename = "entities.csv"
        filepath = self.output_dir / filename
        
        query = """
        PREFIX kg: <http://genome-kg.org/ontology/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?entity_id ?type ?name WHERE {
            ?entity rdf:type kg:Entity .
            ?entity kg:entityId ?entity_id .
            OPTIONAL { ?entity kg:entityType ?type }
            OPTIONAL { ?entity kg:name ?name }
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id:ID', 'type', 'name'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    str(row.entity_id) if row.entity_id else '',
                    str(row.type) if row.type else '',
                    str(row.name) if row.name else ''
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["nodes_by_type"]["Entity"] = count
        self.stats["total_nodes"] += count
        logger.info(f"Exported {count:,} entities to {filename}")
    
    def _export_encodedby_relationships(self):
        """Export protein-gene relationships to encodedby_relationships.csv"""
        filename = "encodedby_relationships.csv"
        filepath = self.output_dir / filename
        
        query = """
        PREFIX kg: <http://genome-kg.org/ontology/>
        SELECT ?protein_id ?gene_id WHERE {
            ?protein_uri kg:encodedBy ?gene_uri .
            ?protein_uri kg:proteinId ?protein_id .
            ?gene_uri kg:geneId ?gene_id .
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([':START_ID', ':END_ID'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    f"protein:{row.protein_id}" if row.protein_id else '',
                    f"gene:{row.gene_id}" if row.gene_id else ''
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["relationships_by_type"]["ENCODED_BY"] = count
        self.stats["total_relationships"] += count
        logger.info(f"Exported {count:,} protein-gene encoded-by relationships to {filename}")
    
    def _export_belongstogenome_relationships(self):
        """Export gene-genome relationships to belongstogenome_relationships.csv"""
        filename = "belongstogenome_relationships.csv"
        filepath = self.output_dir / filename
        
        query = """
        PREFIX kg: <http://genome-kg.org/ontology/>
        SELECT ?gene_id ?genome_id WHERE {
            ?gene_uri kg:belongsToGenome ?genome_uri .
            ?gene_uri kg:geneId ?gene_id .
            ?genome_uri kg:genomeId ?genome_id .
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([':START_ID', ':END_ID'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    f"gene:{row.gene_id}" if row.gene_id else '',
                    str(row.genome_id) if row.genome_id else ''
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["relationships_by_type"]["BELONGS_TO_GENOME"] = count
        self.stats["total_relationships"] += count
        logger.info(f"Exported {count:,} gene-genome belongs-to relationships to {filename}")
    
    def _export_belongstoprotein_relationships(self):
        """Export annotation-protein relationships to belongstoprotein_relationships.csv"""
        filename = "belongstoprotein_relationships.csv"
        filepath = self.output_dir / filename
        
        query = """
        PREFIX kg: <http://genome-kg.org/ontology/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT DISTINCT ?annotation_id ?protein_id WHERE {
            ?annotation_uri rdf:type kg:DomainAnnotation .
            ?annotation_uri kg:belongsToProtein ?protein_uri .
            ?protein_uri kg:proteinId ?protein_id .
            BIND(STRAFTER(STR(?annotation_uri), "http://genome-kg.org/proteins/") AS ?annotation_id)
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([':START_ID', ':END_ID'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    f"protein:{row.annotation_id}" if row.annotation_id else '',
                    f"protein:{row.protein_id}" if row.protein_id else ''
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["relationships_by_type"]["BELONGS_TO_PROTEIN"] = count
        self.stats["total_relationships"] += count
        logger.info(f"Exported {count:,} annotation-protein belongs-to relationships to {filename}")
    
    def _export_domainfamily_relationships(self):
        """Export domain-family relationships to domainfamily_relationships.csv"""
        filename = "domainfamily_relationships.csv"
        filepath = self.output_dir / filename
        
        query = """
        PREFIX kg: <http://genome-kg.org/ontology/>
        SELECT ?domain_annotation_id ?pfam_id WHERE {
            ?domain_annotation kg:domainFamily ?pfam_family .
            ?pfam_family kg:pfamAccession ?pfam_id .
            BIND(STRAFTER(STR(?domain_annotation), "http://genome-kg.org/proteins/") AS ?domain_annotation_id)
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([':START_ID', ':END_ID'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    f"protein:{row.domain_annotation_id}" if row.domain_annotation_id else '',
                    str(row.pfam_id) if row.pfam_id else ''
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["relationships_by_type"]["DOMAIN_FAMILY"] = count
        self.stats["total_relationships"] += count
        logger.info(f"Exported {count:,} domain-family relationships to {filename}")
    
    def _export_annotatesprotein_relationships(self):
        """Export annotation-protein relationships to annotatesprotein_relationships.csv"""
        filename = "annotatesprotein_relationships.csv"
        filepath = self.output_dir / filename
        
        query = """
        PREFIX kg: <http://genome-kg.org/ontology/>
        SELECT DISTINCT ?annotation_id ?protein_id WHERE {
            ?annotation_uri kg:annotatesProtein ?protein_uri .
            ?protein_uri kg:proteinId ?protein_id .
            BIND(STRAFTER(STR(?annotation_uri), "http://genome-kg.org/proteins/") AS ?annotation_id)
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([':START_ID', ':END_ID'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    f"function:{row.annotation_id}" if row.annotation_id else '',
                    f"protein:{row.protein_id}" if row.protein_id else ''
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["relationships_by_type"]["ANNOTATES_PROTEIN"] = count
        self.stats["total_relationships"] += count
        logger.info(f"Exported {count:,} annotation-protein annotates relationships to {filename}")
    
    def _export_assignedfunction_relationships(self):
        """Export function assignment relationships to assignedfunction_relationships.csv"""
        filename = "assignedfunction_relationships.csv"
        filepath = self.output_dir / filename
        
        query = """
        PREFIX kg: <http://genome-kg.org/ontology/>
        SELECT DISTINCT ?annotation_id ?function_id WHERE {
            ?annotation_uri kg:assignedFunction ?function_uri .
            ?function_uri kg:koId ?function_id .
            BIND(STRAFTER(STR(?annotation_uri), "http://genome-kg.org/proteins/") AS ?annotation_id)
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([':START_ID', ':END_ID'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    f"function:{row.annotation_id}" if row.annotation_id else '',
                    str(row.function_id) if row.function_id else ''
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["relationships_by_type"]["ASSIGNED_FUNCTION"] = count
        self.stats["total_relationships"] += count
        logger.info(f"Exported {count:,} function assignment relationships to {filename}")
    
    def _export_hasparticipant_relationships(self):
        """Export pathway participant relationships to hasparticipant_relationships.csv"""
        filename = "hasparticipant_relationships.csv"
        filepath = self.output_dir / filename
        
        query = """
        PREFIX kg2: <http://genomics.ai/kg/>
        SELECT DISTINCT ?pathway_id ?function_id WHERE {
            ?pathway_uri kg2:hasParticipant ?function_uri .
            BIND(STRAFTER(STR(?pathway_uri), "http://genomics.ai/kg/pathway/") AS ?pathway_id)
            BIND(STRAFTER(STR(?function_uri), "http://genomics.ai/kg/kegg/") AS ?function_id)
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([':START_ID', ':END_ID'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    f"pathway:{row.pathway_id}" if row.pathway_id else '',
                    str(row.function_id) if row.function_id else ''
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["relationships_by_type"]["HAS_PARTICIPANT"] = count
        self.stats["total_relationships"] += count
        logger.info(f"Exported {count:,} pathway participant relationships to {filename}")
    
    def _export_hasqualitymetrics_relationships(self):
        """Export quality metrics relationships to hasqualitymetrics_relationships.csv"""
        filename = "hasqualitymetrics_relationships.csv"
        filepath = self.output_dir / filename
        
        query = """
        PREFIX kg: <http://genome-kg.org/ontology/>
        SELECT DISTINCT ?genome_id ?metric_id WHERE {
            ?genome_uri kg:hasQualityMetrics ?metric_uri .
            ?genome_uri kg:genomeId ?genome_id .
            BIND(STRAFTER(STR(?metric_uri), "http://genome-kg.org/genomes/") AS ?metric_id)
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([':START_ID', ':END_ID'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    str(row.genome_id) if row.genome_id else '',
                    str(row.metric_id) if row.metric_id else ''
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["relationships_by_type"]["HAS_QUALITY_METRICS"] = count
        self.stats["total_relationships"] += count
        logger.info(f"Exported {count:,} genome quality metrics relationships to {filename}")
    
    def _export_bgc_clusters(self):
        """Export BGC cluster nodes to bgc_clusters.csv"""
        filename = "bgc_clusters.csv"
        filepath = self.output_dir / filename
        
        query = """
        SELECT ?cluster_id ?cluster_type ?start_pos ?end_pos ?length WHERE {
            ?cluster rdf:type kg:BGCCluster .
            ?cluster kg:clusterId ?cluster_id .
            OPTIONAL { ?cluster kg:clusterType ?cluster_type }
            OPTIONAL { ?cluster kg:startPosition ?start_pos }
            OPTIONAL { ?cluster kg:endPosition ?end_pos }
            OPTIONAL { ?cluster kg:length ?length }
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['cluster_id:ID', 'cluster_type', 'start_position:long', 'end_position:long', 'length:int', ':LABEL'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    str(row.cluster_id) if row.cluster_id else '',
                    str(row.cluster_type) if row.cluster_type else '',
                    int(row.start_pos) if row.start_pos else 0,
                    int(row.end_pos) if row.end_pos else 0,
                    int(row.length) if row.length else 0,
                    'BGCCluster'
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["nodes_by_type"]["BGCCluster"] = count
        self.stats["total_nodes"] += count
        logger.info(f"Exported {count:,} BGC clusters to {filename}")
    
    def _export_protein_domain_relationships(self):
        """Export protein-domain relationships to hasdomain_relationships.csv"""
        filename = "hasdomain_relationships.csv"
        filepath = self.output_dir / filename
        
        query = """
        PREFIX kg: <http://genome-kg.org/ontology/>
        PREFIX protein: <http://genome-kg.org/proteins/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT DISTINCT ?protein_id ?annotation_id WHERE {
            ?domain_annotation rdf:type kg:DomainAnnotation .
            ?domain_annotation kg:belongsToProtein ?protein_uri .
            ?protein_uri kg:proteinId ?protein_id .
            BIND(STRAFTER(STR(?domain_annotation), "http://genome-kg.org/proteins/") AS ?annotation_id)
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([':START_ID', ':END_ID'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    f"protein:{row.protein_id}" if row.protein_id else '',
                    f"protein:{row.annotation_id}" if row.annotation_id else ''
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["relationships_by_type"]["HAS_DOMAIN"] = count
        self.stats["total_relationships"] += count
        logger.info(f"Exported {count:,} protein-domain relationships to {filename}")
    
    def _export_protein_function_relationships(self):
        """Export protein-function relationships to hasfunction_relationships.csv"""
        filename = "hasfunction_relationships.csv"
        filepath = self.output_dir / filename
        
        query = """
        PREFIX kg: <http://genome-kg.org/ontology/>
        SELECT DISTINCT ?protein_id ?kegg_id WHERE {
            ?annotation_uri kg:annotatesProtein ?protein_uri .
            ?annotation_uri kg:assignedFunction ?ko_uri .
            ?protein_uri kg:proteinId ?protein_id .
            ?ko_uri kg:koId ?kegg_id .
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([':START_ID', ':END_ID'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    f"protein:{row.protein_id}" if row.protein_id else '',
                    str(row.kegg_id) if row.kegg_id else ''
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["relationships_by_type"]["HAS_FUNCTION"] = count
        self.stats["total_relationships"] += count
        logger.info(f"Exported {count:,} protein-function relationships to {filename}")
    
    def _export_function_pathway_relationships(self):
        """Export function-pathway relationships to participatesin_relationships.csv"""
        filename = "participatesin_relationships.csv"
        filepath = self.output_dir / filename
        
        query = """
        PREFIX kg2: <http://genomics.ai/kg/>
        SELECT DISTINCT ?kegg_id ?pathway_id WHERE {
            ?ko_uri kg2:participatesIn ?pathway .
            BIND(STRAFTER(STR(?ko_uri), "http://genomics.ai/kg/kegg/") AS ?kegg_id)
            BIND(STRAFTER(STR(?pathway), "http://genomics.ai/kg/pathway/") AS ?pathway_id)
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([':START_ID', ':END_ID'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    str(row.kegg_id) if row.kegg_id else '',
                    f"pathway:{row.pathway_id}" if row.pathway_id else ''
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["relationships_by_type"]["PARTICIPATES_IN"] = count
        self.stats["total_relationships"] += count
        logger.info(f"Exported {count:,} function-pathway relationships to {filename}")
    
    def _export_genome_relationships(self):
        """Export genome-related relationships to genome_rels.csv"""
        filename = "genome_rels.csv"
        filepath = self.output_dir / filename
        
        query = """
        SELECT ?protein_id ?genome_id WHERE {
            ?protein kg:fromGenome ?genome .
            ?protein kg:proteinId ?protein_id .
            ?genome kg:genomeId ?genome_id .
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([':START_ID', ':END_ID'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    str(row.protein_id) if row.protein_id else '',
                    str(row.genome_id) if row.genome_id else ''
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["relationships_by_type"]["FROM_GENOME"] = count
        self.stats["total_relationships"] += count
        logger.info(f"Exported {count:,} protein-genome relationships to {filename}")


    def _export_contigs(self):
        """Export contig nodes to contigs.csv"""
        filename = "contigs.csv"
        filepath = self.output_dir / filename
        
        query = """
        PREFIX kg: <http://genome-kg.org/ontology/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT DISTINCT ?contig_id WHERE {
            ?gene rdf:type kg:Gene .
            ?gene kg:contig ?contig_id .
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id:ID', 'contigId', 'length:int', 'coverage:float'])
            
            for row in self.graph.query(query):
                if row.contig_id:
                    contig_name = str(row.contig_id)
                    # Extract length and coverage from contig name pattern: NODE_X_length_Y_cov_Z
                    length = 0
                    coverage = 0.0
                    try:
                        parts = contig_name.split('_')
                        if 'length' in parts:
                            length_idx = parts.index('length') + 1
                            if length_idx < len(parts):
                                length = int(parts[length_idx])
                        if 'cov' in parts:
                            cov_idx = parts.index('cov') + 1
                            if cov_idx < len(parts):
                                coverage = float(parts[cov_idx])
                    except (ValueError, IndexError):
                        pass  # Keep defaults if parsing fails
                    
                    writer.writerow([
                        f"contig:{contig_name}",
                        contig_name,
                        length,
                        coverage
                    ])
                    count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["nodes_by_type"]["Contig"] = count
        self.stats["total_nodes"] += count
        logger.info(f"Exported {count:,} contigs to {filename}")
    
    def _export_belongstocontig_relationships(self):
        """Export gene-contig relationships to belongstocontig_relationships.csv"""
        filename = "belongstocontig_relationships.csv"
        filepath = self.output_dir / filename
        
        query = """
        PREFIX kg: <http://genome-kg.org/ontology/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT DISTINCT ?gene_id ?contig_id WHERE {
            ?gene rdf:type kg:Gene .
            ?gene kg:geneId ?gene_id .
            ?gene kg:contig ?contig_id .
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([':START_ID', ':END_ID'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    f"gene:{row.gene_id}" if row.gene_id else '',
                    f"contig:{row.contig_id}" if row.contig_id else ''
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["relationships_by_type"]["BELONGS_TO_CONTIG"] = count
        self.stats["total_relationships"] += count
        logger.info(f"Exported {count:,} gene-contig relationships to {filename}")


def main():
    """Test the DirectCSVExporter with a sample RDF graph."""
    # This would typically be called from rdf_builder.py with the real graph
    logger.info("DirectCSVExporter test - would need real RDF graph to run")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()