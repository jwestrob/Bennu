#!/usr/bin/env python3
"""
KEGG Pathway Integration Module

Integrates KEGG pathway information into the knowledge graph by:
1. Parsing ko_pathway.list file
2. Creating Pathway nodes
3. Adding PARTICIPATES_IN relationships between KEGGOrtholog and Pathway nodes
4. Enriching with pathway names and descriptions
"""

import logging
from pathlib import Path
from typing import Dict, Set, Tuple, List
import requests
import time
from dataclasses import dataclass

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS

logger = logging.getLogger(__name__)

# Define our knowledge graph namespace
KG = Namespace("http://genomics.ai/kg/")

@dataclass
class PathwayInfo:
    """Information about a KEGG pathway."""
    pathway_id: str  # e.g., "map00010" or "ko00010"
    pathway_number: str  # e.g., "00010"
    pathway_type: str  # "map" or "ko"
    name: str = ""
    description: str = ""

class KEGGPathwayIntegrator:
    """Integrates KEGG pathway information into the knowledge graph."""
    
    def __init__(self, ko_pathway_file: Path, output_dir: Path):
        self.ko_pathway_file = ko_pathway_file
        self.output_dir = output_dir
        self.graph = Graph()
        
        # Bind namespaces
        self.graph.bind("kg", KG)
        self.graph.bind("rdfs", RDFS)
        
        # Cache for pathway information
        self.pathways: Dict[str, PathwayInfo] = {}
        self.ko_pathway_relationships: List[Tuple[str, str]] = []
        
        logger.info(f"Initialized KEGGPathwayIntegrator with file: {ko_pathway_file}")
    
    def parse_ko_pathway_file(self, found_ko_ids: set = None) -> None:
        """Parse the ko_pathway.list file to extract KO-pathway relationships."""
        logger.info("Parsing ko_pathway.list file...")
        
        ko_count = 0
        pathway_count = 0
        filtered_count = 0
        
        with open(self.ko_pathway_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    ko_id, pathway_id = line.split('\t')
                    
                    # Clean the IDs
                    ko_id = ko_id.replace('ko:', '')  # K00001
                    pathway_id = pathway_id.replace('path:', '')  # map00010 or ko00010
                    
                    # CRITICAL FIX: Skip KO pathways (ko00010, etc.) - these indicate KOs NOT in pathways
                    if pathway_id.startswith('ko'):
                        # KO pathways mean the KO is NOT in a pathway - skip entirely
                        continue
                    
                    # Only process MAP pathways (map00010, etc.) - these are actual pathways
                    if not pathway_id.startswith('map'):
                        continue
                    
                    # FILTER: Only include KOs that are actually found in our proteins
                    if found_ko_ids is not None and ko_id not in found_ko_ids:
                        filtered_count += 1
                        continue
                    
                    # Extract pathway info
                    pathway_type = 'map'
                    pathway_number = pathway_id[3:]  # Remove 'map' prefix
                    
                    # Store the relationship
                    self.ko_pathway_relationships.append((ko_id, pathway_id))
                    
                    # Store pathway info if not seen before
                    if pathway_id not in self.pathways:
                        self.pathways[pathway_id] = PathwayInfo(
                            pathway_id=pathway_id,
                            pathway_number=pathway_number,
                            pathway_type=pathway_type
                        )
                        pathway_count += 1
                    
                    ko_count += 1
                    
                except ValueError as e:
                    logger.warning(f"Line {line_num}: Could not parse '{line}': {e}")
                    continue
        
        if found_ko_ids is not None:
            logger.info(f"Filtered out {filtered_count:,} KO-pathway relationships for KOs not found in proteins")
            logger.info(f"Using {len(found_ko_ids):,} KO IDs found in protein annotations")
        
        logger.info(f"Parsed {ko_count:,} KO-pathway relationships")
        logger.info(f"Found {pathway_count:,} unique pathways")
        logger.info(f"Map pathways: {sum(1 for p in self.pathways.values() if p.pathway_type == 'map')}")
        logger.info(f"KO pathways: {sum(1 for p in self.pathways.values() if p.pathway_type == 'ko')}")
    
    def enrich_pathway_names(self) -> None:
        """Enrich pathway information with names from KEGG REST API or static mapping."""
        logger.info("Enriching pathway names...")
        
        # Common pathway names (static mapping for reliability)
        pathway_names = {
            "00010": "Glycolysis / Gluconeogenesis",
            "00020": "Citrate cycle (TCA cycle)",
            "00030": "Pentose phosphate pathway",
            "00040": "Pentose and glucuronate interconversions",
            "00051": "Fructose and mannose metabolism",
            "00052": "Galactose metabolism",
            "00053": "Ascorbate and aldarate metabolism",
            "00061": "Fatty acid biosynthesis",
            "00062": "Fatty acid elongation",
            "00071": "Fatty acid degradation",
            "00130": "Ubiquinone and other terpenoid-quinone biosynthesis",
            "00190": "Oxidative phosphorylation",
            "00195": "Photosynthesis",
            "00220": "Arginine biosynthesis",
            "00230": "Purine metabolism",
            "00240": "Pyrimidine metabolism",
            "00250": "Alanine, aspartate and glutamate metabolism",
            "00260": "Glycine, serine and threonine metabolism",
            "00270": "Cysteine and methionine metabolism",
            "00280": "Valine, leucine and isoleucine degradation",
            "00290": "Valine, leucine and isoleucine biosynthesis",
            "00300": "Lysine biosynthesis",
            "00310": "Lysine degradation",
            "00330": "Arginine and proline metabolism",
            "00340": "Histidine metabolism",
            "00350": "Tyrosine metabolism",
            "00360": "Phenylalanine metabolism",
            "00380": "Tryptophan metabolism",
            "00400": "Phenylalanine, tyrosine and tryptophan biosynthesis",
            "00500": "Starch and sucrose metabolism",
            "00520": "Amino sugar and nucleotide sugar metabolism",
            "00550": "Peptidoglycan biosynthesis",
            "00561": "Glycerolipid metabolism",
            "00564": "Glycerophospholipid metabolism",
            "00620": "Pyruvate metabolism",
            "00630": "Glyoxylate and dicarboxylate metabolism",
            "00640": "Propanoate metabolism",
            "00650": "Butanoate metabolism",
            "00660": "C5-Branched dibasic acid metabolism",
            "00670": "One carbon pool by folate",
            "00680": "Methane metabolism",
            "00710": "Carbon fixation in photosynthetic organisms",
            "00720": "Carbon fixation pathways in prokaryotes",
            "00730": "Thiamine metabolism",
            "00740": "Riboflavin metabolism",
            "00750": "Vitamin B6 metabolism",
            "00760": "Nicotinate and nicotinamide metabolism",
            "00770": "Pantothenate and CoA biosynthesis",
            "00780": "Biotin metabolism",
            "00790": "Folate biosynthesis",
            "00860": "Porphyrin and chlorophyll metabolism",
            "00900": "Terpenoid backbone biosynthesis",
            "00910": "Nitrogen metabolism",
            "00920": "Sulfur metabolism",
            "02010": "ABC transporters",
            "02020": "Two-component system",
            "02030": "Bacterial chemotaxis",
            "02040": "Flagellar assembly",
            "02060": "Phosphotransferase system (PTS)",
            "03010": "Ribosome",
            "03018": "RNA degradation",
            "03020": "RNA polymerase",
            "03030": "DNA replication",
            "03060": "Protein export",
            "03070": "Bacterial secretion system",
            "03410": "Base excision repair",
            "03420": "Nucleotide excision repair",
            "03430": "Mismatch repair",
            "03440": "Homologous recombination"
        }
        
        enriched_count = 0
        for pathway in self.pathways.values():
            if pathway.pathway_number in pathway_names:
                pathway.name = pathway_names[pathway.pathway_number]
                pathway.description = f"KEGG pathway {pathway.pathway_id}: {pathway.name}"
                enriched_count += 1
        
        logger.info(f"Enriched {enriched_count:,} pathways with names")
        
        # For pathways without names, create generic descriptions
        for pathway in self.pathways.values():
            if not pathway.name:
                pathway.name = f"KEGG pathway {pathway.pathway_number}"
                pathway.description = f"KEGG pathway {pathway.pathway_id}"
    
    def build_pathway_graph(self) -> None:
        """Build RDF graph with pathway nodes and relationships."""
        logger.info("Building pathway RDF graph...")
        
        # Create Pathway nodes
        pathway_count = 0
        for pathway in self.pathways.values():
            pathway_uri = KG[f"pathway/{pathway.pathway_id}"]
            
            # Add pathway node
            self.graph.add((pathway_uri, RDF.type, KG.Pathway))
            #self.graph.add((pathway_uri, KG.id, Literal(pathway.pathway_id)))
            self.graph.add((pathway_uri, KG.pathwayNumber, Literal(pathway.pathway_number)))
            self.graph.add((pathway_uri, KG.pathwayType, Literal(pathway.pathway_type)))
            
            if pathway.name:
                self.graph.add((pathway_uri, RDFS.label, Literal(pathway.name)))
                self.graph.add((pathway_uri, KG.name, Literal(pathway.name)))
            
            if pathway.description:
                self.graph.add((pathway_uri, KG.description, Literal(pathway.description)))
            
            pathway_count += 1
        
        logger.info(f"Created {pathway_count:,} pathway nodes")
        
        # Create KO-pathway relationships
        relationship_count = 0
        missing_ko_count = 0
        ko_nodes_created = set()  # Track KOs we've created nodes for

        for ko_id, pathway_id in self.ko_pathway_relationships:
            ko_uri = KG[f"kegg/{ko_id}"]
            pathway_uri = KG[f"pathway/{pathway_id}"]
            if ko_id not in ko_nodes_created:
                self.graph.add((ko_uri, RDF.type, KG.KEGGOrtholog))
                self.graph.add((ko_uri, KG.koId, Literal(ko_id)))
                ko_nodes_created.add(ko_id)
            # Add the relationship
            self.graph.add((ko_uri, KG.participatesIn, pathway_uri))
            relationship_count += 1
            
            # Also add reverse relationship for easier querying
            self.graph.add((pathway_uri, KG.hasParticipant, ko_uri))
        
        logger.info(f"Created {relationship_count:,} KO-pathway relationships")
        
        # Add some useful metadata
        self.graph.add((KG.PathwayGraph, RDF.type, KG.Dataset))
        self.graph.add((KG.PathwayGraph, KG.pathwayCount, Literal(pathway_count)))
        self.graph.add((KG.PathwayGraph, KG.relationshipCount, Literal(relationship_count)))
    
    def save_graph(self) -> Path:
        """Save the pathway graph to RDF file."""
        output_file = self.output_dir / "pathway_integration.ttl"
        
        logger.info(f"Saving pathway graph to {output_file}")
        self.graph.serialize(destination=str(output_file), format="turtle")
        
        # Save statistics
        stats_file = self.output_dir / "pathway_statistics.txt"
        with open(stats_file, 'w') as f:
            f.write(f"Pathway Integration Statistics\n")
            f.write(f"============================\n\n")
            f.write(f"Total pathways: {len(self.pathways):,}\n")
            f.write(f"Map pathways: {sum(1 for p in self.pathways.values() if p.pathway_type == 'map'):,}\n")
            f.write(f"KO pathways: {sum(1 for p in self.pathways.values() if p.pathway_type == 'ko'):,}\n")
            f.write(f"KO-pathway relationships: {len(self.ko_pathway_relationships):,}\n")
            f.write(f"Pathways with names: {sum(1 for p in self.pathways.values() if p.name):,}\n")
            f.write(f"RDF triples: {len(self.graph):,}\n")
        
        logger.info(f"Saved statistics to {stats_file}")
        logger.info(f"Generated {len(self.graph):,} RDF triples")
        
        return output_file


def integrate_pathways(ko_pathway_file: Path, output_dir: Path, found_ko_ids: set = None) -> Path:
    """
    Main function to integrate KEGG pathways into knowledge graph.
    
    Args:
        ko_pathway_file: Path to ko_pathway.list file
        output_dir: Directory to save output files
        found_ko_ids: Set of KO IDs actually found in proteins (filter to these only)
        
    Returns:
        Path to generated RDF file
    """
    logger.info("Starting KEGG pathway integration...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize integrator
    integrator = KEGGPathwayIntegrator(ko_pathway_file, output_dir)
    
    # Parse the input file with filtering
    integrator.parse_ko_pathway_file(found_ko_ids)
    
    # Enrich with pathway names
    integrator.enrich_pathway_names()
    
    # Build the RDF graph
    integrator.build_pathway_graph()
    
    # Save the graph
    output_file = integrator.save_graph()
    
    logger.info("KEGG pathway integration completed successfully!")
    return output_file


if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get file paths
    repo_root = Path(__file__).parent.parent.parent
    ko_pathway_file = repo_root / "ko_pathway.list"
    output_dir = repo_root / "data" / "pathway_integration"
    
    if not ko_pathway_file.exists():
        logger.error(f"ko_pathway.list file not found at {ko_pathway_file}")
        sys.exit(1)
    
    # Run integration
    output_file = integrate_pathways(ko_pathway_file, output_dir)
    print(f"Pathway integration complete! Output: {output_file}")