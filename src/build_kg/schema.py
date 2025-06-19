"""
Knowledge Graph Schema Definitions
Pydantic models mapping genomic entities to Biolink Model identifiers.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class BiologicalSequenceType(str, Enum):
    """Biolink biological sequence types."""
    GENOME = "biolink:Genome"
    GENE = "biolink:Gene"  
    PROTEIN = "biolink:Protein"
    TRANSCRIPT = "biolink:Transcript"


class TaxonomicRank(str, Enum):
    """Taxonomic rank levels."""
    DOMAIN = "biolink:Domain"
    KINGDOM = "biolink:Kingdom"
    PHYLUM = "biolink:Phylum"
    CLASS = "biolink:Class"
    ORDER = "biolink:Order"
    FAMILY = "biolink:Family"
    GENUS = "biolink:Genus"
    SPECIES = "biolink:Species"
    STRAIN = "biolink:OrganismTaxon"


class GenomeEntity(BaseModel):
    """
    Represents a genome assembly with quality metrics.
    Maps to biolink:Genome.
    
    TODO: Add complete Biolink-compliant schema
    """
    id: str = Field(..., description="Unique genome identifier")
    name: str = Field(..., description="Genome name/label")
    organism_taxon: Optional[str] = Field(None, description="GTDB taxonomic ID")
    assembly_accession: Optional[str] = Field(None, description="NCBI assembly accession")
    
    # Quality metrics from QUAST
    total_length: Optional[int] = Field(None, description="Total assembly length")
    n50: Optional[int] = Field(None, description="N50 contig length")
    num_contigs: Optional[int] = Field(None, description="Number of contigs")
    
    # CheckM metrics
    completeness: Optional[float] = Field(None, description="Genome completeness %")
    contamination: Optional[float] = Field(None, description="Genome contamination %")


class GeneEntity(BaseModel):
    """
    Represents a predicted gene with coordinates and annotations.
    Maps to biolink:Gene.
    
    TODO: Add complete gene feature schema
    """
    id: str = Field(..., description="Unique gene identifier")
    genome_id: str = Field(..., description="Parent genome identifier")
    contig_id: str = Field(..., description="Contig identifier")
    start: int = Field(..., description="Start coordinate")
    end: int = Field(..., description="End coordinate")
    strand: str = Field(..., description="Strand (+/-)")
    product: Optional[str] = Field(None, description="Gene product description")


class ProteinEntity(BaseModel):
    """
    Represents a protein sequence with functional annotations.
    Maps to biolink:Protein.
    
    TODO: Add domain and pathway annotations
    """
    id: str = Field(..., description="Unique protein identifier")
    gene_id: str = Field(..., description="Parent gene identifier")
    sequence: str = Field(..., description="Amino acid sequence")
    length: int = Field(..., description="Sequence length")
    domains: List[str] = Field(default_factory=list, description="Domain IDs")


class TaxonomicEntity(BaseModel):
    """
    Represents a taxonomic classification.
    Maps to biolink:OrganismTaxon.
    
    TODO: Add GTDB-specific fields
    """
    id: str = Field(..., description="Taxonomic identifier")
    name: str = Field(..., description="Taxonomic name")
    rank: TaxonomicRank = Field(..., description="Taxonomic rank")
    parent_id: Optional[str] = Field(None, description="Parent taxon ID")


# TODO: Add additional entity types:
# - FunctionalDomain
# - MetabolicPathway
# - BiologicalProcess
# - MolecularFunction
# - CellularComponent
