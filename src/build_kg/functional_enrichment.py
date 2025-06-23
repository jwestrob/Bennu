#!/usr/bin/env python3
"""
Functional Enrichment Module for Knowledge Graph Construction.
Integrates PFAM and KOFAM reference data to add functional descriptions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from dataclasses import dataclass

import rdflib
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class PfamEntry:
    """PFAM family entry with functional information."""
    id: str
    accession: str
    description: str
    type: str
    model_length: Optional[int] = None
    clan: Optional[str] = None


@dataclass
class KoEntry:
    """KEGG Ortholog entry with functional information."""
    knum: str
    definition: str
    simplified_definition: str
    threshold: float
    score_type: str
    profile_type: str


class FunctionalEnrichment:
    """Enrich knowledge graph with functional annotations from reference databases."""
    
    def __init__(self, pfam_file: Path, ko_file: Path):
        """Initialize with reference database files."""
        self.pfam_file = pfam_file
        self.ko_file = ko_file
        self.pfam_data = {}
        self.ko_data = {}
        
    def load_reference_data(self) -> None:
        """Load PFAM and KOFAM reference data."""
        console.print("[bold blue]Loading functional reference databases...[/bold blue]")
        
        self.pfam_data = self._parse_pfam_stockholm()
        self.ko_data = self._parse_ko_list()
        
        console.print(f"✓ Loaded {len(self.pfam_data)} PFAM families")
        console.print(f"✓ Loaded {len(self.ko_data)} KEGG orthologs")
    
    def _parse_pfam_stockholm(self) -> Dict[str, PfamEntry]:
        """Parse PFAM Stockholm format file."""
        pfam_entries = {}
        
        if not self.pfam_file.exists():
            logger.warning(f"PFAM file not found: {self.pfam_file}")
            return pfam_entries
        
        current_entry = {}
        
        with open(self.pfam_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('#=GF ID'):
                    current_entry['id'] = line.split(None, 2)[-1]
                elif line.startswith('#=GF AC'):
                    current_entry['accession'] = line.split(None, 2)[-1]
                elif line.startswith('#=GF DE'):
                    current_entry['description'] = line.split(None, 2)[-1]
                elif line.startswith('#=GF TP'):
                    current_entry['type'] = line.split(None, 2)[-1]
                elif line.startswith('#=GF ML'):
                    try:
                        current_entry['model_length'] = int(line.split(None, 2)[-1])
                    except ValueError:
                        pass
                elif line.startswith('#=GF CL'):
                    current_entry['clan'] = line.split(None, 2)[-1]
                elif line == '//':
                    # End of entry
                    if 'id' in current_entry and 'description' in current_entry:
                        entry = PfamEntry(
                            id=current_entry['id'],
                            accession=current_entry.get('accession', ''),
                            description=current_entry['description'],
                            type=current_entry.get('type', 'Unknown'),
                            model_length=current_entry.get('model_length'),
                            clan=current_entry.get('clan')
                        )
                        pfam_entries[entry.id] = entry
                    current_entry = {}
        
        return pfam_entries
    
    def _parse_ko_list(self) -> Dict[str, KoEntry]:
        """Parse KEGG Ortholog list file."""
        ko_entries = {}
        
        if not self.ko_file.exists():
            logger.warning(f"KO file not found: {self.ko_file}")
            return ko_entries
        
        with open(self.ko_file, 'r') as f:
            header = f.readline().strip().split('\t')
            
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= len(header):
                    try:
                        entry = KoEntry(
                            knum=parts[0],
                            threshold=float(parts[1]) if parts[1] != '-' else float('nan'),
                            score_type=parts[2],
                            profile_type=parts[3],
                            definition=parts[11] if len(parts) > 11 else '',
                            simplified_definition=parts[12] if len(parts) > 12 else ''
                        )
                        ko_entries[entry.knum] = entry
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing KO line: {line[:50]}... - {e}")
                        continue
        
        return ko_entries
    
    def enrich_rdf_graph(self, graph: rdflib.Graph) -> Tuple[rdflib.Graph, Dict[str, int]]:
        """Enrich RDF graph with functional annotations."""
        console.print("[bold blue]Enriching knowledge graph with functional annotations...[/bold blue]")
        
        if not self.pfam_data and not self.ko_data:
            self.load_reference_data()
        
        stats = {'pfam_enriched': 0, 'ko_enriched': 0, 'missing_pfam': 0, 'missing_ko': 0}
        
        # Define namespaces
        kg = rdflib.Namespace("http://genome-kg.org/ontology/")
        pfam_ns = rdflib.Namespace("http://pfam.xfam.org/family/")
        ko_ns = rdflib.Namespace("http://www.genome.jp/kegg/ko/")
        
        # Enrich PFAM families
        stats.update(self._enrich_pfam_families(graph, kg, pfam_ns))
        
        # Enrich KEGG orthologs
        stats.update(self._enrich_ko_functions(graph, kg, ko_ns))
        
        console.print(f"✓ Enriched {stats['pfam_enriched']} PFAM families")
        console.print(f"✓ Enriched {stats['ko_enriched']} KEGG orthologs")
        if stats['missing_pfam'] > 0:
            console.print(f"⚠️  {stats['missing_pfam']} PFAM families without reference data")
        if stats['missing_ko'] > 0:
            console.print(f"⚠️  {stats['missing_ko']} KEGG orthologs without reference data")
        
        return graph, stats
    
    def _enrich_pfam_families(self, graph: rdflib.Graph, kg: rdflib.Namespace, pfam_ns: rdflib.Namespace) -> Dict[str, int]:
        """Add functional descriptions to PFAM families."""
        stats = {'pfam_enriched': 0, 'missing_pfam': 0}
        
        # Find all PFAM family nodes
        pfam_families = set()
        for subj, pred, obj in graph:
            if isinstance(subj, rdflib.URIRef) and str(subj).startswith(str(pfam_ns)):
                family_id = str(subj).replace(str(pfam_ns), "")
                pfam_families.add((subj, family_id))
        
        # Enrich each family
        for family_uri, family_id in pfam_families:
            if family_id in self.pfam_data:
                entry = self.pfam_data[family_id]
                
                # Add functional description
                graph.add((family_uri, rdflib.RDFS.label, rdflib.Literal(entry.description)))
                graph.add((family_uri, kg.description, rdflib.Literal(entry.description)))
                graph.add((family_uri, kg.familyType, rdflib.Literal(entry.type)))
                
                if entry.accession:
                    graph.add((family_uri, kg.pfamAccession, rdflib.Literal(entry.accession)))
                
                if entry.model_length:
                    graph.add((family_uri, kg.modelLength, rdflib.Literal(entry.model_length)))
                
                if entry.clan:
                    graph.add((family_uri, kg.clan, rdflib.Literal(entry.clan)))
                
                stats['pfam_enriched'] += 1
            else:
                stats['missing_pfam'] += 1
                logger.debug(f"No PFAM reference data for: {family_id}")
        
        return stats
    
    def _enrich_ko_functions(self, graph: rdflib.Graph, kg: rdflib.Namespace, ko_ns: rdflib.Namespace) -> Dict[str, int]:
        """Add functional descriptions to KEGG orthologs."""
        stats = {'ko_enriched': 0, 'missing_ko': 0}
        
        # Find all KEGG ortholog nodes
        ko_functions = set()
        for subj, pred, obj in graph:
            if isinstance(subj, rdflib.URIRef) and str(subj).startswith(str(ko_ns)):
                ko_id = str(subj).replace(str(ko_ns), "")
                ko_functions.add((subj, ko_id))
        
        # Enrich each function
        for ko_uri, ko_id in ko_functions:
            if ko_id in self.ko_data:
                entry = self.ko_data[ko_id]
                
                # Add functional description
                graph.add((ko_uri, rdflib.RDFS.label, rdflib.Literal(entry.simplified_definition)))
                graph.add((ko_uri, kg.description, rdflib.Literal(entry.definition)))
                graph.add((ko_uri, kg.simplifiedDescription, rdflib.Literal(entry.simplified_definition)))
                graph.add((ko_uri, kg.threshold, rdflib.Literal(entry.threshold)))
                graph.add((ko_uri, kg.scoreType, rdflib.Literal(entry.score_type)))
                graph.add((ko_uri, kg.profileType, rdflib.Literal(entry.profile_type)))
                
                # Extract EC numbers if present
                ec_matches = re.findall(r'\[EC:([\d\.-]+)\]', entry.definition)
                for ec in ec_matches:
                    graph.add((ko_uri, kg.ecNumber, rdflib.Literal(ec)))
                
                stats['ko_enriched'] += 1
            else:
                stats['missing_ko'] += 1
                logger.debug(f"No KO reference data for: {ko_id}")
        
        return stats


def add_functional_enrichment_to_pipeline(graph: rdflib.Graph, 
                                        pfam_file: Path = Path("data/reference/Pfam-A.hmm.dat.stockholm"),
                                        ko_file: Path = Path("data/reference/ko_list")) -> Tuple[rdflib.Graph, Dict[str, int]]:
    """
    Main function to add functional enrichment to the knowledge graph pipeline.
    
    Args:
        graph: RDF graph to enrich
        pfam_file: Path to PFAM reference file
        ko_file: Path to KEGG Ortholog reference file
        
    Returns:
        Tuple of enriched graph and enrichment statistics
    """
    enricher = FunctionalEnrichment(pfam_file, ko_file)
    return enricher.enrich_rdf_graph(graph)