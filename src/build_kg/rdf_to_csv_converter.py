#!/usr/bin/env python3
"""
Convert RDF knowledge graph to Neo4j-compatible CSV files for bulk import.
Designed for 100x faster loading than current Python-based approach.
"""

import csv
import rdflib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Set
import logging
from rich.console import Console
from rich.progress import track

console = Console()
logger = logging.getLogger(__name__)


class RDFToCSVConverter:
    """Convert RDF triples to Neo4j bulk import CSV format."""
    
    def __init__(self, rdf_file: Path, output_dir: Path):
        self.rdf_file = rdf_file
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define namespaces
        self.namespaces = {
            "genome": "http://genome-kg.org/genomes/",
            "protein": "http://genome-kg.org/proteins/", 
            "gene": "http://genome-kg.org/genes/",
            "pfam": "http://pfam.xfam.org/family/",
            "ko": "http://www.genome.jp/kegg/ko/",
            "kg": "http://genome-kg.org/ontology/"
        }
        
        # Track all nodes and their properties
        self.nodes = defaultdict(dict)  # {node_id: {property: value}}
        self.node_types = {}  # {node_id: type}
        self.relationships = []  # [(from_id, rel_type, to_id)]
        
    def convert(self) -> Dict[str, Any]:
        """Convert RDF to CSV files and return statistics."""
        console.print(f"[bold blue]Converting RDF to CSV for bulk import[/bold blue]")
        
        # Load RDF
        console.print("Loading RDF graph...")
        g = rdflib.Graph()
        g.parse(self.rdf_file, format="turtle")
        console.print(f"Loaded {len(g):,} triples")
        
        # Parse triples
        self._parse_triples(g)
        
        # Write CSV files
        stats = self._write_csv_files()
        
        console.print(f"[green]✓ CSV conversion complete![/green]")
        return stats
    
    def _parse_triples(self, g: rdflib.Graph):
        """Parse RDF triples into nodes and relationships."""
        console.print("Parsing triples...")
        
        for subj, pred, obj in track(g, description="Processing triples"):
            subj_id = self._uri_to_id(str(subj))
            pred_name = self._uri_to_property(str(pred))
            
            # Handle rdf:type declarations
            if str(pred) == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
                node_type = self._get_node_type(str(obj))
                if node_type not in ["Property", "Class"]:  # Skip ontology declarations
                    self.node_types[subj_id] = node_type
                continue
            
            # Handle relationships vs properties
            if isinstance(obj, rdflib.URIRef):
                # This is a relationship
                obj_id = self._uri_to_id(str(obj))
                self.relationships.append((subj_id, pred_name, obj_id))
            else:
                # This is a node property
                obj_value = self._convert_literal(obj)
                self.nodes[subj_id][pred_name] = obj_value
    
    def _write_csv_files(self) -> Dict[str, Any]:
        """Write separate CSV files for each node type and relationships."""
        stats = {"nodes": {}, "relationships": {}}
        
        # Group nodes by type
        nodes_by_type = defaultdict(list)
        for node_id, node_type in self.node_types.items():
            nodes_by_type[node_type].append(node_id)
        
        # Write node CSV files
        console.print("Writing node CSV files...")
        for node_type, node_ids in nodes_by_type.items():
            if not node_ids:
                continue
                
            filename = f"{node_type.lower()}s.csv"
            filepath = self.output_dir / filename
            
            # Collect all possible properties for this node type
            all_properties = set()
            for node_id in node_ids:
                all_properties.update(self.nodes[node_id].keys())
            
            # Keep all properties - we don't want to remove legitimate properties
            
            # Write CSV
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                # Neo4j bulk import expects ID column to be labeled with :ID
                # Use a unique name that won't conflict with properties
                id_column = 'id:ID'
                fieldnames = [id_column] + sorted(all_properties)
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for node_id in node_ids:
                    row = {id_column: node_id}
                    row.update(self.nodes[node_id])
                    writer.writerow(row)
            
            stats["nodes"][node_type] = len(node_ids)
            console.print(f"  ✓ {filename}: {len(node_ids):,} nodes")
        
        # Write relationship CSV files
        console.print("Writing relationship CSV files...")
        rels_by_type = defaultdict(list)
        for from_id, rel_type, to_id in self.relationships:
            rels_by_type[rel_type].append((from_id, to_id))
        
        for rel_type, rels in rels_by_type.items():
            filename = f"{rel_type.lower()}_relationships.csv"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([':START_ID', ':END_ID'])  # Neo4j bulk import format
                writer.writerows(rels)
            
            stats["relationships"][rel_type] = len(rels)
            console.print(f"  ✓ {filename}: {len(rels):,} relationships")
        
        return stats
    
    def _uri_to_id(self, uri: str) -> str:
        """Convert URI to readable ID, preserving namespace for nodes to avoid conflicts."""
        for prefix, namespace in self.namespaces.items():
            if uri.startswith(namespace):
                local_id = uri.replace(namespace, "")
                # Preserve namespace prefix for nodes to distinguish protein:X from gene:X
                if prefix in ['protein', 'gene']:
                    return f"{prefix}:{local_id}"
                return local_id
        return uri.split("/")[-1]
    
    def _uri_to_property(self, uri: str) -> str:
        """Convert property URI to readable name."""
        if uri.startswith(self.namespaces["kg"]):
            return uri.replace(self.namespaces["kg"], "")
        return uri.split("/")[-1].split("#")[-1]
    
    def _get_node_type(self, type_uri: str) -> str:
        """Determine node type from RDF type URI."""
        if type_uri.startswith(self.namespaces["kg"]):
            return type_uri.replace(self.namespaces["kg"], "")
        return type_uri.split("/")[-1].split("#")[-1]
    
    def _convert_literal(self, literal: rdflib.Literal) -> Any:
        """Convert RDF literal to appropriate Python type."""
        if literal.datatype == rdflib.XSD.integer:
            return int(literal)
        elif literal.datatype in [rdflib.XSD.decimal, rdflib.XSD.double, rdflib.XSD.float]:
            return float(literal)
        elif literal.datatype == rdflib.XSD.boolean:
            return str(literal).lower() == 'true'
        else:
            return str(literal)


def main():
    """Convert RDF to CSV for bulk import."""
    rdf_file = Path("data/stage05_kg/knowledge_graph.ttl")
    csv_dir = Path("data/stage05_kg/csv")
    
    if not rdf_file.exists():
        console.print(f"[red]RDF file not found: {rdf_file}[/red]")
        return 1
    
    converter = RDFToCSVConverter(rdf_file, csv_dir)
    stats = converter.convert()
    
    console.print(f"\n[bold]Conversion Summary:[/bold]")
    console.print(f"Output directory: {csv_dir}")
    
    total_nodes = sum(stats["nodes"].values())
    total_rels = sum(stats["relationships"].values())
    
    console.print(f"Total nodes: {total_nodes:,}")
    console.print(f"Total relationships: {total_rels:,}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())