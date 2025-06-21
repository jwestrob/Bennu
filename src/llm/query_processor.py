#!/usr/bin/env python3
"""
Query processors for different data sources.
Modular design for containerized deployment.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
import numpy as np

from neo4j import GraphDatabase
import lancedb
import h5py
from rich.console import Console

from .config import LLMConfig

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Standard result format for all query processors."""
    source: str  # "neo4j", "lancedb", "hybrid"
    query_type: str  # "structural", "semantic", "hybrid"
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    execution_time: float


class BaseQueryProcessor(ABC):
    """Abstract base class for query processors."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    async def process_query(self, query: str, **kwargs) -> QueryResult:
        """Process a query and return standardized results."""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if the data source is available."""
        pass


class Neo4jQueryProcessor(BaseQueryProcessor):
    """Process structured queries against Neo4j knowledge graph."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Establish Neo4j connection."""
        try:
            self.driver = GraphDatabase.driver(
                self.config.database.neo4j_uri,
                auth=(self.config.database.neo4j_user, self.config.database.neo4j_password)
            )
            logger.info(f"Connected to Neo4j: {self.config.database.neo4j_uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check Neo4j connection."""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1").single()
            return True
        except:
            return False
    
    async def process_query(self, query: str, query_type: str = "cypher", **kwargs) -> QueryResult:
        """Process Neo4j queries."""
        import time
        start_time = time.time()
        
        try:
            if query_type == "cypher":
                # Direct Cypher query
                results = await self._execute_cypher(query)
            elif query_type == "genome_overview":
                results = await self._get_genome_overview(query)
            elif query_type == "protein_info":
                results = await self._get_protein_info(query)
            elif query_type == "functional_annotation":
                results = await self._get_functional_annotations(query)
            else:
                # Auto-detect query type and generate appropriate Cypher
                results = await self._auto_query(query)
            
            execution_time = time.time() - start_time
            
            return QueryResult(
                source="neo4j",
                query_type=query_type,
                results=results,
                metadata={"cypher_query": query, "result_count": len(results)},
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")
            execution_time = time.time() - start_time
            return QueryResult(
                source="neo4j",
                query_type=query_type,
                results=[],
                metadata={"error": str(e)},
                execution_time=execution_time
            )
    
    async def _execute_cypher(self, cypher: str) -> List[Dict[str, Any]]:
        """Execute raw Cypher query."""
        with self.driver.session() as session:
            result = session.run(cypher)
            return [dict(record) for record in result]
    
    async def _get_genome_overview(self, genome_id: str) -> List[Dict[str, Any]]:
        """Get comprehensive genome information."""
        cypher = """
        MATCH (g:Genome {id: $genome_id})
        OPTIONAL MATCH (g)<-[:belongsToGenome]-(p:Protein)
        OPTIONAL MATCH (p)-[:hasDomain]->(d:DomainAnnotation)-[:domainFamily]->(pf:Domain)
        OPTIONAL MATCH (p)-[:hasFunction]->(ko:KEGGOrtholog)
        RETURN g.id as genome_id,
               count(DISTINCT p) as protein_count,
               count(DISTINCT pf) as domain_family_count,
               collect(DISTINCT pf.description)[0..5] as sample_family_descriptions,
               count(DISTINCT ko) as kegg_function_count,
               collect(DISTINCT pf.id)[0..5] as sample_families,
               collect(DISTINCT ko.id)[0..5] as sample_functions
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, genome_id=genome_id)
            return [dict(record) for record in result]
    
    async def _get_protein_info(self, protein_id: str) -> List[Dict[str, Any]]:
        """Get comprehensive protein information with genomic context."""
        cypher = """
        MATCH (p:Protein {id: $protein_id})
        OPTIONAL MATCH (p)<-[:encodedBy]-(gene:Gene)-[:belongsToGenome]->(g:Genome)
        OPTIONAL MATCH (p)-[:hasDomain]->(d:DomainAnnotation)-[:domainFamily]->(pf:Domain)
        OPTIONAL MATCH (p)-[:hasFunction]->(ko:KEGGOrtholog)
        
        // Get genomic neighborhood (genes on same scaffold)
        OPTIONAL MATCH (gene)-[:belongsToGenome]->(g)<-[:belongsToGenome]-(neighbor_gene:Gene)
        WHERE neighbor_gene.id CONTAINS split(gene.id, '_')[0] + '_' + split(gene.id, '_')[1] + '_' + split(gene.id, '_')[2] + '_' + split(gene.id, '_')[3] + '_' + split(gene.id, '_')[4]
        OPTIONAL MATCH (neighbor_gene)<-[:encodedBy]-(neighbor_protein:Protein)-[:hasDomain]->(neighbor_domain:DomainAnnotation)-[:domainFamily]->(neighbor_pf:Domain)
        
        RETURN p.id as protein_id,
               gene.id as gene_id,
               g.id as genome_id,
               collect(DISTINCT pf.id) as protein_families,
               collect(DISTINCT pf.description) as domain_descriptions,
               collect(DISTINCT pf.pfamAccession) as pfam_accessions,
               collect(DISTINCT ko.id) as kegg_functions,
               collect(DISTINCT ko.description) as kegg_descriptions,
               collect(DISTINCT d.id) as domain_ids,
               collect(DISTINCT d.bitscore) as domain_scores,
               collect(DISTINCT (d.domainStart + '-' + d.domainEnd)) as domain_positions,
               count(DISTINCT d) as domain_count,
               collect(DISTINCT neighbor_protein.id)[0..5] as neighboring_proteins,
               collect(DISTINCT neighbor_pf.id)[0..10] as neighborhood_families
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, protein_id=protein_id)
            return [dict(record) for record in result]
    
    async def _get_functional_annotations(self, function_query: str) -> List[Dict[str, Any]]:
        """Search for proteins by functional annotation."""
        cypher = """
        MATCH (ko:KEGGOrtholog)
        WHERE ko.id CONTAINS $query OR ko.description CONTAINS $query
        MATCH (p:Protein)-[:hasFunction]->(ko)
        OPTIONAL MATCH (p)-[:belongsToGenome]->(g:Genome)
        RETURN ko.id as kegg_id,
               ko.description as function_description,
               collect(p.id) as proteins,
               collect(DISTINCT g.id) as genomes
        LIMIT 20
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, query=function_query)
            return [dict(record) for record in result]
    
    async def _auto_query(self, natural_query: str) -> List[Dict[str, Any]]:
        """Auto-generate Cypher from natural language (simplified)."""
        # Simple pattern matching for common queries
        query_lower = natural_query.lower()
        
        if "genome" in query_lower and any(word in query_lower for word in ["how many", "count", "number"]):
            cypher = "MATCH (g:Genome) RETURN count(g) as genome_count"
        elif "protein" in query_lower and any(word in query_lower for word in ["how many", "count", "number"]):
            cypher = "MATCH (p:Protein) RETURN count(p) as protein_count"
        elif "function" in query_lower or "kegg" in query_lower:
            cypher = "MATCH (ko:KEGGOrtholog) RETURN ko.id, ko.description LIMIT 10"
        else:
            # Default: show database overview
            cypher = """
            MATCH (n)
            RETURN labels(n) as node_type, count(n) as count
            ORDER BY count DESC
            LIMIT 10
            """
        
        return await self._execute_cypher(cypher)
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()


class LanceDBQueryProcessor(BaseQueryProcessor):
    """Process semantic similarity queries against LanceDB protein embeddings."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.db = None
        self.table = None
        self.embeddings_file = None
        self._connect()
    
    def _connect(self):
        """Establish LanceDB connection."""
        try:
            db_path = self.config.database.lancedb_path
            self.db = lancedb.connect(db_path)
            self.table = self.db.open_table("protein_embeddings")
            
            # Also load HDF5 file for embedding metadata
            embeddings_h5 = f"{db_path}/../protein_embeddings.h5"
            self.embeddings_file = embeddings_h5
            
            logger.info(f"Connected to LanceDB: {db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to LanceDB: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check LanceDB connection."""
        try:
            len(self.table)  # Simple check
            return True
        except:
            return False
    
    async def process_query(self, query: Union[str, np.ndarray], query_type: str = "similarity", **kwargs) -> QueryResult:
        """Process LanceDB similarity queries."""
        import time
        start_time = time.time()
        
        try:
            if query_type == "similarity":
                if isinstance(query, str):
                    # Find protein by ID and get similar proteins
                    results = await self._find_similar_by_id(query, **kwargs)
                else:
                    # Direct embedding similarity search
                    results = await self._find_similar_by_embedding(query, **kwargs)
            elif query_type == "protein_lookup":
                results = await self._lookup_protein(query)
            else:
                results = await self._find_similar_by_id(query, **kwargs)
            
            execution_time = time.time() - start_time
            
            return QueryResult(
                source="lancedb",
                query_type=query_type,
                results=results,
                metadata={"query": str(query)[:100], "result_count": len(results)},
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"LanceDB query failed: {e}")
            execution_time = time.time() - start_time
            return QueryResult(
                source="lancedb",
                query_type=query_type,
                results=[],
                metadata={"error": str(e)},
                execution_time=execution_time
            )
    
    async def _find_similar_by_id(self, protein_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find proteins similar to a given protein ID (excluding self)."""
        # First find the protein's embedding
        results = self.table.search().where(f"protein_id = '{protein_id}'").limit(1).to_pandas()
        
        if results.empty:
            return []
        
        query_embedding = results.iloc[0]['vector']
        # Get more results than needed to filter out self-similarity
        similar_results = await self._find_similar_by_embedding(query_embedding, limit=limit+5)
        
        # Filter out self-similarity and very high similarity (likely identical)
        filtered_results = []
        for result in similar_results:
            if (result['protein_id'] != protein_id and 
                result['distance'] > 0.001 and  # Exclude nearly identical
                result['similarity'] < 0.999):  # Exclude near-perfect matches
                filtered_results.append(result)
                if len(filtered_results) >= limit:
                    break
        
        return filtered_results
    
    async def _find_similar_by_embedding(self, embedding: np.ndarray, limit: int = 10) -> List[Dict[str, Any]]:
        """Find proteins similar to a given embedding."""
        results = self.table.search(embedding).limit(limit).to_pandas()
        
        return [
            {
                "protein_id": row['protein_id'],
                "genome_id": row['genome_id'],
                "sequence_length": row['sequence_length'],
                "distance": row['_distance'],
                "similarity": 1.0 - row['_distance']  # Convert distance to similarity
            }
            for _, row in results.iterrows()
        ]
    
    async def _lookup_protein(self, protein_id: str) -> List[Dict[str, Any]]:
        """Look up a specific protein by ID."""
        results = self.table.search().where(f"protein_id = '{protein_id}'").limit(1).to_pandas()
        
        if results.empty:
            return []
        
        row = results.iloc[0]
        return [{
            "protein_id": row['protein_id'],
            "genome_id": row['genome_id'],
            "sequence_length": row['sequence_length'],
            "embedding_available": True
        }]


class HybridQueryProcessor(BaseQueryProcessor):
    """Combine results from Neo4j and LanceDB for comprehensive answers."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.neo4j_processor = Neo4jQueryProcessor(config)
        self.lancedb_processor = LanceDBQueryProcessor(config)
    
    def health_check(self) -> bool:
        """Check both data sources."""
        return (self.neo4j_processor.health_check() and 
                self.lancedb_processor.health_check())
    
    async def process_query(self, query: str, query_type: str = "hybrid", **kwargs) -> QueryResult:
        """Process queries that require both structured and semantic search."""
        import time
        start_time = time.time()
        
        try:
            # Run both queries concurrently
            neo4j_task = self.neo4j_processor.process_query(query, **kwargs)
            lancedb_task = self.lancedb_processor.process_query(query, **kwargs)
            
            neo4j_result, lancedb_result = await asyncio.gather(neo4j_task, lancedb_task)
            
            # Combine results
            combined_results = {
                "structured_data": neo4j_result.results,
                "semantic_data": lancedb_result.results,
                "sources": ["neo4j", "lancedb"]
            }
            
            execution_time = time.time() - start_time
            
            return QueryResult(
                source="hybrid",
                query_type=query_type,
                results=[combined_results],
                metadata={
                    "neo4j_time": neo4j_result.execution_time,
                    "lancedb_time": lancedb_result.execution_time,
                    "total_results": len(neo4j_result.results) + len(lancedb_result.results)
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Hybrid query failed: {e}")
            execution_time = time.time() - start_time
            return QueryResult(
                source="hybrid",
                query_type=query_type,
                results=[],
                metadata={"error": str(e)},
                execution_time=execution_time
            )
    
    def close(self):
        """Close all connections."""
        self.neo4j_processor.close()
        # LanceDB doesn't need explicit closing