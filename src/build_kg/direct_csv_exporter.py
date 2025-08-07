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

# Define namespaces (matching rdf_builder.py)
KG = Namespace("http://genome-kg.org/")
PROTEIN_NS = Namespace("http://genome-kg.org/protein/")
PFAM_NS = Namespace("http://genome-kg.org/pfam/")
KEGG_NS = Namespace("http://genome-kg.org/kegg/")
PATHWAY_NS = Namespace("http://genome-kg.org/pathway/")
BGC_NS = Namespace("http://genome-kg.org/bgc/")
GENOME_NS = Namespace("http://genome-kg.org/genome/")


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
            overall_task = progress.add_task("Exporting CSV files...", total=10)
            
            # Export node CSV files
            progress.update(overall_task, description="Exporting genomes...")
            self._export_genomes()
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
            
            progress.update(overall_task, description="Exporting BGC clusters...")
            self._export_bgc_clusters()
            progress.advance(overall_task)
            
            # Export relationship CSV files
            progress.update(overall_task, description="Exporting protein-domain relationships...")
            self._export_protein_domain_relationships()
            progress.advance(overall_task)
            
            progress.update(overall_task, description="Exporting protein-function relationships...")
            self._export_protein_function_relationships()
            progress.advance(overall_task)
            
            progress.update(overall_task, description="Exporting function-pathway relationships...")
            self._export_function_pathway_relationships()
            progress.advance(overall_task)
            
            progress.update(overall_task, description="Exporting genome relationships...")
            self._export_genome_relationships()
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
        SELECT ?genome_id ?assembly_file ?total_contigs ?total_length ?gc_content WHERE {
            ?genome rdf:type kg:Genome .
            ?genome kg:genomeId ?genome_id .
            OPTIONAL { ?genome kg:assemblyFile ?assembly_file }
            OPTIONAL { ?genome kg:totalContigs ?total_contigs }
            OPTIONAL { ?genome kg:totalLength ?total_length }  
            OPTIONAL { ?genome kg:gcContent ?gc_content }
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Neo4j CSV headers
            writer.writerow(['genome_id:ID', 'assembly_file', 'total_contigs:int', 'total_length:long', 'gc_content:float', ':LABEL'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    str(row.genome_id) if row.genome_id else '',
                    str(row.assembly_file) if row.assembly_file else '',
                    int(row.total_contigs) if row.total_contigs else 0,
                    int(row.total_length) if row.total_length else 0,
                    float(row.gc_content) if row.gc_content else 0.0,
                    'Genome'
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["nodes_by_type"]["Genome"] = count
        self.stats["total_nodes"] += count
        logger.info(f"Exported {count:,} genomes to {filename}")
    
    def _export_proteins(self):
        """Export protein nodes to proteins.csv"""
        filename = "proteins.csv"
        filepath = self.output_dir / filename
        
        query = """
        SELECT ?protein_id ?sequence ?length ?start_pos ?end_pos ?strand ?genome_id WHERE {
            ?protein rdf:type kg:Protein .
            ?protein kg:proteinId ?protein_id .
            OPTIONAL { ?protein kg:sequence ?sequence }
            OPTIONAL { ?protein kg:length ?length }
            OPTIONAL { ?protein kg:startPosition ?start_pos }
            OPTIONAL { ?protein kg:endPosition ?end_pos }
            OPTIONAL { ?protein kg:strand ?strand }
            OPTIONAL { 
                ?protein kg:fromGenome ?genome .
                ?genome kg:genomeId ?genome_id 
            }
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['protein_id:ID', 'sequence', 'length:int', 'start_position:long', 'end_position:long', 'strand', 'genome_id', ':LABEL'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    str(row.protein_id) if row.protein_id else '',
                    str(row.sequence) if row.sequence else '',
                    int(row.length) if row.length else 0,
                    int(row.start_pos) if row.start_pos else 0,
                    int(row.end_pos) if row.end_pos else 0,
                    str(row.strand) if row.strand else '',
                    str(row.genome_id) if row.genome_id else '',
                    'Protein'
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["nodes_by_type"]["Protein"] = count
        self.stats["total_nodes"] += count
        logger.info(f"Exported {count:,} proteins to {filename}")
    
    def _export_pfam_domains(self):
        """Export PFAM domain nodes to pfam_domains.csv"""
        filename = "pfam_domains.csv"
        filepath = self.output_dir / filename
        
        query = """
        SELECT DISTINCT ?pfam_id ?name ?description WHERE {
            ?domain rdf:type kg:PFAMDomain .
            ?domain kg:pfamId ?pfam_id .
            OPTIONAL { ?domain kg:name ?name }
            OPTIONAL { ?domain kg:description ?description }
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['pfam_id:ID', 'name', 'description', ':LABEL'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    str(row.pfam_id) if row.pfam_id else '',
                    str(row.name) if row.name else '',
                    str(row.description) if row.description else '',
                    'PFAMDomain'
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["nodes_by_type"]["PFAMDomain"] = count
        self.stats["total_nodes"] += count
        logger.info(f"Exported {count:,} PFAM domains to {filename}")
    
    def _export_kegg_functions(self):
        """Export KEGG function nodes to kegg_functions.csv"""
        filename = "kegg_functions.csv"
        filepath = self.output_dir / filename
        
        query = """
        SELECT DISTINCT ?kegg_id ?name ?definition WHERE {
            ?function rdf:type kg:KEGGFunction .
            ?function kg:keggId ?kegg_id .
            OPTIONAL { ?function kg:name ?name }
            OPTIONAL { ?function kg:definition ?definition }
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['kegg_id:ID', 'name', 'definition', ':LABEL'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    str(row.kegg_id) if row.kegg_id else '',
                    str(row.name) if row.name else '',
                    str(row.definition) if row.definition else '',
                    'KEGGFunction'
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
        SELECT DISTINCT ?pathway_id ?name ?category WHERE {
            ?pathway rdf:type kg:Pathway .
            ?pathway kg:pathwayId ?pathway_id .
            OPTIONAL { ?pathway kg:name ?name }
            OPTIONAL { ?pathway kg:category ?category }
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['pathway_id:ID', 'name', 'category', ':LABEL'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    str(row.pathway_id) if row.pathway_id else '',
                    str(row.name) if row.name else '',
                    str(row.category) if row.category else '',
                    'Pathway'
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["nodes_by_type"]["Pathway"] = count
        self.stats["total_nodes"] += count
        logger.info(f"Exported {count:,} pathways to {filename}")
    
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
        """Export protein-domain relationships to protein_domain_rels.csv"""
        filename = "protein_domain_rels.csv"
        filepath = self.output_dir / filename
        
        query = """
        SELECT ?protein_id ?pfam_id ?evalue ?score WHERE {
            ?protein kg:hasPFAMDomain ?annotation .
            ?protein kg:proteinId ?protein_id .
            ?annotation kg:domain ?domain .
            ?domain kg:pfamId ?pfam_id .
            OPTIONAL { ?annotation kg:evalue ?evalue }
            OPTIONAL { ?annotation kg:score ?score }
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([':START_ID', ':END_ID', 'evalue:double', 'score:double', ':TYPE'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    str(row.protein_id) if row.protein_id else '',
                    str(row.pfam_id) if row.pfam_id else '',
                    float(row.evalue) if row.evalue else 1.0,
                    float(row.score) if row.score else 0.0,
                    'HAS_DOMAIN'
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["relationships_by_type"]["HAS_DOMAIN"] = count
        self.stats["total_relationships"] += count
        logger.info(f"Exported {count:,} protein-domain relationships to {filename}")
    
    def _export_protein_function_relationships(self):
        """Export protein-function relationships to protein_function_rels.csv"""
        filename = "protein_function_rels.csv"
        filepath = self.output_dir / filename
        
        query = """
        SELECT ?protein_id ?kegg_id ?evalue ?score WHERE {
            ?protein kg:hasKEGGFunction ?annotation .
            ?protein kg:proteinId ?protein_id .
            ?annotation kg:function ?function .
            ?function kg:keggId ?kegg_id .
            OPTIONAL { ?annotation kg:evalue ?evalue }
            OPTIONAL { ?annotation kg:score ?score }
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([':START_ID', ':END_ID', 'evalue:double', 'score:double', ':TYPE'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    str(row.protein_id) if row.protein_id else '',
                    str(row.kegg_id) if row.kegg_id else '',
                    float(row.evalue) if row.evalue else 1.0,
                    float(row.score) if row.score else 0.0,
                    'HAS_FUNCTION'
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["relationships_by_type"]["HAS_FUNCTION"] = count
        self.stats["total_relationships"] += count
        logger.info(f"Exported {count:,} protein-function relationships to {filename}")
    
    def _export_function_pathway_relationships(self):
        """Export function-pathway relationships to function_pathway_rels.csv"""
        filename = "function_pathway_rels.csv"
        filepath = self.output_dir / filename
        
        query = """
        SELECT ?kegg_id ?pathway_id WHERE {
            ?function kg:participatesInPathway ?pathway .
            ?function kg:keggId ?kegg_id .
            ?pathway kg:pathwayId ?pathway_id .
        }
        """
        
        count = 0
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([':START_ID', ':END_ID', ':TYPE'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    str(row.kegg_id) if row.kegg_id else '',
                    str(row.pathway_id) if row.pathway_id else '',
                    'PARTICIPATES_IN'
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
            writer.writerow([':START_ID', ':END_ID', ':TYPE'])
            
            for row in self.graph.query(query):
                writer.writerow([
                    str(row.protein_id) if row.protein_id else '',
                    str(row.genome_id) if row.genome_id else '',
                    'FROM_GENOME'
                ])
                count += 1
        
        self.stats["files_generated"].append(filename)
        self.stats["relationships_by_type"]["FROM_GENOME"] = count
        self.stats["total_relationships"] += count
        logger.info(f"Exported {count:,} protein-genome relationships to {filename}")


def main():
    """Test the DirectCSVExporter with a sample RDF graph."""
    # This would typically be called from rdf_builder.py with the real graph
    logger.info("DirectCSVExporter test - would need real RDF graph to run")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()