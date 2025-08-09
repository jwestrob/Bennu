#!/usr/bin/env python3
"""
Schema-locked query builder for parameterized Cypher generation.
Always starts from Protein and expands outward via allowed edges/directions.
"""

import logging
from typing import Dict, List, Tuple, Any
from pydantic import BaseModel

from .schema_map import SchemaMap

logger = logging.getLogger(__name__)


class QueryPlan(BaseModel):
    """Structured query plan with Cypher and parameters."""
    cypher: str
    params: Dict[str, Any]
    producer_type: str  # "kegg_driven", "pfam_driven", "combined"


class QueryBuilder:
    """
    Schema-locked, parameterized Cypher compiler.
    Always starts from Protein and expands outward using allowed edges/directions.
    """
    
    def __init__(self, schema_map: SchemaMap):
        self.schema_map = schema_map
        
        # Validate required schema components exist
        self._validate_schema()
        
        logger.info("🏗️  QueryBuilder initialized with schema-locked compilation")
    
    def _validate_schema(self):
        """Validate that required schema components exist."""
        # Required labels - only check those we absolutely need
        required_labels = ["Protein", "KEGGOrtholog", "Domain"]
        for label in required_labels:
            if not self.schema_map.has_label(label):
                logger.warning(f"Required label {label} not found in schema")
        
        # Check for critical relationships (don't assert, just warn)
        critical_relationships = [
            ("Protein", "HASFUNCTION", "KEGGOrtholog"),
            ("Protein", "ENCODEDBY", "Gene"),
        ]
        
        for src, rel, dst in critical_relationships:
            if not self.schema_map.has_edge(src, rel, dst):
                logger.warning(f"Critical relationship {src}-[{rel}]->{dst} not found in schema")
        
        # Required properties - check what's actually available
        required_props = [
            ("Protein", "id"),
            ("KEGGOrtholog", "id"),  
            ("KEGGOrtholog", "description"),
            ("Domain", "id"),
        ]
        
        for label, prop in required_props:
            if not self.schema_map.has_property(label, prop):
                logger.warning(f"Required property {prop} not found for label {label}")
        
        # Check optional properties
        optional_props = [
            ("Gene", "startCoordinate"),
            ("Gene", "endCoordinate"), 
            ("Gene", "strand"),
            ("Genome", "genomeId"),
        ]
        
        for label, prop in optional_props:
            if self.schema_map.has_property(label, prop):
                logger.debug(f"✅ Optional property {prop} available for {label}")
            else:
                logger.debug(f"⚠️  Optional property {prop} not available for {label}")
    
    def build(self, ko_ids: List[str], pfam_ids: List[str], k: int = 20) -> List[QueryPlan]:
        """
        Build parameterized query plans for given detector sets.
        
        Args:
            ko_ids: List of KEGGOrtholog IDs to query
            pfam_ids: List of PFAM Domain IDs to query  
            k: Result limit (applied post-merge in Python)
            
        Returns:
            List of QueryPlan objects with Cypher and parameters
        """
        query_plans = []
        
        try:
            # KEGG-driven producer if KO IDs present
            if ko_ids:
                kegg_plan = self._build_kegg_driven_query(ko_ids)
                query_plans.append(kegg_plan)
            
            # PFAM-driven producer if PFAM IDs present  
            if pfam_ids:
                pfam_plan = self._build_pfam_driven_query(pfam_ids)
                query_plans.append(pfam_plan)
            
            # Log query plan summary
            producer_types = [plan.producer_type for plan in query_plans]
            logger.info(f"🏗️  Built {len(query_plans)} query plans: {producer_types}")
            
            return query_plans
            
        except Exception as e:
            logger.error(f"❌ Query building failed: {e}")
            return []
    
    def _build_kegg_driven_query(self, ko_ids: List[str]) -> QueryPlan:
        """
        Build KEGG-driven query starting from KEGGOrtholog -> Protein.
        Expands to domains, genes, and genomes via schema-validated paths.
        """
        # Use genomeId if available, otherwise fall back to id
        genome_id_field = "genomeId" if self.schema_map.has_property("Genome", "genomeId") else "id"
        
        # Build query using only available properties
        gene_props = []
        if self.schema_map.has_property("Gene", "startCoordinate"):
            gene_props.append("g.startCoordinate AS start_coordinate")
        else:
            gene_props.append("null AS start_coordinate")
            
        if self.schema_map.has_property("Gene", "endCoordinate"):
            gene_props.append("g.endCoordinate AS end_coordinate")
        else:
            gene_props.append("null AS end_coordinate")
            
        if self.schema_map.has_property("Gene", "strand"):
            gene_props.append("g.strand AS strand")
        else:
            gene_props.append("null AS strand")
        
        gene_props_str = ",\n               ".join(gene_props)
        
        cypher = f"""
        MATCH (ko:KEGGOrtholog)
        WHERE ko.id IN $ko_ids
        MATCH (p:Protein)-[:HASFUNCTION]->(ko)
        OPTIONAL MATCH (p)-[:DOMAINFAMILY]->(dom:Domain)
        OPTIONAL MATCH (p)-[:ENCODEDBY]->(g:Gene)
        RETURN p.id AS protein_id,
               ko.id AS ko_id,
               ko.description AS ko_description,
               collect(DISTINCT dom.id) AS pfam_accessions,
               {gene_props_str},
               g.id AS genome_id
        """
        
        params = {"ko_ids": ko_ids}
        
        logger.debug(f"🔍 KEGG-driven query built for {len(ko_ids)} KO IDs")
        
        return QueryPlan(
            cypher=cypher,
            params=params,
            producer_type="kegg_driven"
        )
    
    def _build_pfam_driven_query(self, pfam_ids: List[str]) -> QueryPlan:
        """
        Build PFAM-driven query starting from Domain -> DomainAnnotation -> Protein.
        Expands to functions, genes, and genomes via schema-validated paths.
        """
        # Use genomeId if available, otherwise fall back to id
        genome_id_field = "genomeId" if self.schema_map.has_property("Genome", "genomeId") else "id"
        
        # Build query using only available properties
        gene_props = []
        if self.schema_map.has_property("Gene", "startCoordinate"):
            gene_props.append("g.startCoordinate AS start_coordinate")
        else:
            gene_props.append("null AS start_coordinate")
            
        if self.schema_map.has_property("Gene", "endCoordinate"):
            gene_props.append("g.endCoordinate AS end_coordinate")
        else:
            gene_props.append("null AS end_coordinate")
            
        if self.schema_map.has_property("Gene", "strand"):
            gene_props.append("g.strand AS strand")
        else:
            gene_props.append("null AS strand")
        
        gene_props_str = ",\n               ".join(gene_props)
        
        cypher = f"""
        MATCH (dom:Domain)
        WHERE dom.id IN $pfam_ids
        MATCH (p:Protein)-[:DOMAINFAMILY]->(dom)
        OPTIONAL MATCH (p)-[:HASFUNCTION]->(ko:KEGGOrtholog)
        OPTIONAL MATCH (p)-[:ENCODEDBY]->(g:Gene)
        RETURN p.id AS protein_id,
               collect(DISTINCT dom.id) AS pfam_accessions,
               ko.id AS ko_id,
               ko.description AS ko_description,
               {gene_props_str},
               g.id AS genome_id
        """
        
        params = {"pfam_ids": pfam_ids}
        
        logger.debug(f"🔍 PFAM-driven query built for {len(pfam_ids)} PFAM IDs")
        
        return QueryPlan(
            cypher=cypher,
            params=params,
            producer_type="pfam_driven"
        )
    
    def build_neighborhood_expansion(self, protein_ids: List[str], k: int = 20) -> QueryPlan:
        """
        Build neighborhood expansion query for spatial analysis around proteins.
        Optional: can be used for spatial neighborhood queries.
        """
        if not protein_ids:
            return QueryPlan(cypher="", params={}, producer_type="neighborhood_empty")
        
        # Use genomeId if available
        genome_id_field = "genomeId" if self.schema_map.has_property("Genome", "genomeId") else "id"
        
        # EFFICIENT CONTIG-BASED NEIGHBORHOOD: Get ~10 ORFs around anchor on same contig only
        cypher = f"""
        MATCH (p:Protein)
        WHERE p.id IN $protein_ids
        MATCH (p)-[:ENCODEDBY]->(g:Gene)-[:BELONGSTOCONTIG]->(contig:Contig)
        
        WITH g, contig, toInteger(g.startCoordinate) as anchor_start
        WHERE anchor_start IS NOT NULL
        
        // Find neighboring genes on SAME CONTIG within ~10kb window (roughly 10 genes)
        MATCH (neighbor_gene:Gene)-[:BELONGSTOCONTIG]->(contig)
        WITH g, neighbor_gene, contig, anchor_start, toInteger(neighbor_gene.startCoordinate) as neighbor_start
        WHERE neighbor_gene.id <> g.id
          AND neighbor_start IS NOT NULL
          AND neighbor_start >= (anchor_start - 10000)
          AND neighbor_start <= (anchor_start + 10000)
        
        MATCH (neighbor_protein:Protein)-[:ENCODEDBY]->(neighbor_gene)
        OPTIONAL MATCH (neighbor_protein)-[:HASFUNCTION]->(neighbor_ko:KEGGOrtholog)
        OPTIONAL MATCH (neighbor_protein)-[:HASDOMAIN]->(neighbor_da:DomainAnnotation)-[:DOMAINFAMILY]->(neighbor_dom:Domain)
        
        RETURN neighbor_protein.id AS protein_id,
               neighbor_ko.id AS ko_id,
               neighbor_ko.description AS ko_description,
               collect(DISTINCT neighbor_dom.id) AS pfam_accessions,
               neighbor_gene.startCoordinate AS start_coordinate,
               neighbor_gene.endCoordinate AS end_coordinate,
               neighbor_gene.strand AS strand,
               contig.id AS contig_id,
               abs(neighbor_start - anchor_start) AS distance_from_anchor
        ORDER BY distance_from_anchor
        LIMIT $k
        """
        
        params = {
            "protein_ids": protein_ids,
            "k": k
        }
        
        logger.debug(f"🔍 Contig-based neighborhood expansion built for {len(protein_ids)} anchor proteins (~10 ORFs each)")
        
        return QueryPlan(
            cypher=cypher,
            params=params,
            producer_type="neighborhood_expansion"
        )