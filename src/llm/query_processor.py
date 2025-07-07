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
from .task_repair_agent import TaskRepairAgent
from .repair_types import RepairResult

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
        self.task_repair_agent = TaskRepairAgent()
        self.last_repair_result = None
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
    
    def get_last_repair_result(self) -> Optional[RepairResult]:
        """Get the result of the last repair attempt."""
        return self.last_repair_result
    
    async def process_query(self, query: str, query_type: str = "cypher", **kwargs) -> QueryResult:
        """Process Neo4j queries."""
        import time
        start_time = time.time()
        
        try:
            if query_type == "cypher":
                # Check if this is a domain query that needs count enhancement
                if "GGDEF" in query or "TPR" in query or "/domain/" in query:
                    results = await self._execute_domain_query_with_count(query)
                else:
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
            
            # NEW: Attempt repair with TaskRepairAgent
            repair_result = self.task_repair_agent.detect_and_repair(query, e)
            self.last_repair_result = repair_result
            
            if repair_result.success and repair_result.repaired_query:
                logger.info(f"TaskRepairAgent repaired query using: {repair_result.repair_strategy_used}")
                try:
                    # Retry with repaired query
                    if query_type == "cypher":
                        results = await self._execute_cypher(repair_result.repaired_query)
                    else:
                        results = await self._auto_query(repair_result.repaired_query)
                    
                    execution_time = time.time() - start_time
                    return QueryResult(
                        source="neo4j",
                        query_type=query_type,
                        results=results,
                        metadata={
                            "cypher_query": repair_result.repaired_query,
                            "result_count": len(results),
                            "repaired_by": str(repair_result.repair_strategy_used),
                            "original_error": str(e)
                        },
                        execution_time=execution_time
                    )
                except Exception as retry_error:
                    logger.error(f"Repaired query also failed: {retry_error}")
            
            # If repair failed or no repaired query, return error result
            execution_time = time.time() - start_time
            error_metadata = {"error": str(e)}
            if repair_result.success:
                error_metadata["repair_message"] = repair_result.user_message
                error_metadata["repair_strategy"] = str(repair_result.repair_strategy_used)
            
            return QueryResult(
                source="neo4j",
                query_type=query_type,
                results=[],
                metadata=error_metadata,
                execution_time=execution_time
            )
    
    async def _execute_cypher(self, cypher: str) -> List[Dict[str, Any]]:
        """Execute raw Cypher query."""
        # Proactive repair: Fix BELONGSTO -> BELONGSTOGENOME
        if "BELONGSTO" in cypher and "BELONGSTOGENOME" not in cypher:
            logger.info("Detected BELONGSTO relationship, auto-repairing to BELONGSTOGENOME")
            cypher = cypher.replace("[:BELONGSTO]", "[:BELONGSTOGENOME]")
            cypher = cypher.replace("-[:BELONGSTO]->", "-[:BELONGSTOGENOME]->")
        
        # Proactive repair: Fix HASGENE -> BELONGSTOGENOME (reverse direction)
        if "HASGENE" in cypher:
            logger.info("Detected HASGENE relationship, auto-repairing to BELONGSTOGENOME")
            cypher = cypher.replace("[:HASGENE]", "[:BELONGSTOGENOME]")
            cypher = cypher.replace("-[:HASGENE]->", "<-[:BELONGSTOGENOME]-")  # Reverse direction
        
        with self.driver.session() as session:
            result = session.run(cypher)
            return [dict(record) for record in result]
    
    async def _get_genome_overview(self, genome_id: str) -> List[Dict[str, Any]]:
        """Get comprehensive genome information."""
        cypher = """
        MATCH (g:Genome {id: $genome_id})
        OPTIONAL MATCH (g)<-[:BELONGSTOGENOME]-(p:Protein)
        OPTIONAL MATCH (p)-[:HASDOMAIN]->(d:DomainAnnotation)-[:DOMAINFAMILY]->(dom:Domain)
        OPTIONAL MATCH (p)-[:HASFUNCTION]->(ko:KEGGOrtholog)
        RETURN g.id as genome_id,
               count(DISTINCT p) as protein_count,
               count(DISTINCT dom) as domain_family_count,
               collect(DISTINCT dom.description)[0..5] as sample_family_descriptions,
               count(DISTINCT ko) as kegg_function_count,
               collect(DISTINCT dom.id)[0..5] as sample_families,
               collect(DISTINCT ko.id)[0..5] as sample_functions
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, genome_id=genome_id)
            return [dict(record) for record in result]
    
    async def _get_protein_info(self, protein_id: str) -> List[Dict[str, Any]]:
        """Get comprehensive protein information with genomic context."""
        # Ensure protein_id has the protein: prefix for exact matching
        if not protein_id.startswith("protein:"):
            protein_id = f"protein:{protein_id}"
        
        cypher = """
        MATCH (p:Protein {id: $protein_id})
        OPTIONAL MATCH (p)-[:ENCODEDBY]->(gene:Gene)-[:BELONGSTOGENOME]->(g:Genome)
        OPTIONAL MATCH (p)-[:HASDOMAIN]->(da:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)
        OPTIONAL MATCH (p)-[:HASFUNCTION]->(ko:KEGGOrtholog)
        
        // Get genomic neighborhood using coordinates (5kb window)
        OPTIONAL MATCH (neighbor_gene:Gene)-[:BELONGSTOGENOME]->(g)
        WHERE neighbor_gene.id <> gene.id
          AND neighbor_gene.startCoordinate IS NOT NULL 
          AND gene.startCoordinate IS NOT NULL
          AND abs(toInteger(neighbor_gene.startCoordinate) - toInteger(gene.startCoordinate)) < 5000
        OPTIONAL MATCH (neighbor_protein:Protein)-[:ENCODEDBY]->(neighbor_gene)
        OPTIONAL MATCH (neighbor_protein)-[:HASDOMAIN]->(neighbor_da:DomainAnnotation)-[:DOMAINFAMILY]->(neighbor_d:Domain)
        OPTIONAL MATCH (neighbor_protein)-[:HASFUNCTION]->(neighbor_ko:KEGGOrtholog)
        
        RETURN p.id as protein_id,
               gene.id as gene_id,
               g.id as genome_id,
               gene.startCoordinate as gene_start,
               gene.endCoordinate as gene_end,
               gene.strand as gene_strand,
               gene.lengthAA as gene_length_aa,
               gene.gcContent as gene_gc_content,
               collect(DISTINCT d.id) as protein_families,
               collect(DISTINCT d.description) as domain_descriptions,
               collect(DISTINCT d.pfamAccession) as pfam_accessions,
               collect(DISTINCT ko.id) as kegg_functions,
               collect(DISTINCT ko.description) as kegg_descriptions,
               collect(DISTINCT da.id) as domain_ids,
               collect(DISTINCT da.bitscore) as domain_scores,
               collect(DISTINCT (da.domainStart + '-' + da.domainEnd)) as domain_positions,
               count(DISTINCT da) as domain_count,
               collect(DISTINCT {
                   neighbor_id: neighbor_protein.id,
                   neighbor_start: neighbor_gene.startCoordinate,
                   neighbor_end: neighbor_gene.endCoordinate,
                   neighbor_strand: neighbor_gene.strand,
                   distance: abs(toInteger(neighbor_gene.startCoordinate) - toInteger(gene.startCoordinate)),
                   direction: CASE WHEN toInteger(neighbor_gene.startCoordinate) > toInteger(gene.startCoordinate) THEN 'downstream' ELSE 'upstream' END,
                   function: neighbor_ko.description
               }) as neighbor_details,
               collect(DISTINCT {
                   protein_id: neighbor_protein.id,
                   gene_id: neighbor_gene.id,
                   position: toInteger(neighbor_gene.startCoordinate),
                   strand: neighbor_gene.strand,
                   pfam_ids: neighbor_d.id,
                   pfam_desc: neighbor_d.description,
                   kegg_id: neighbor_ko.id,
                   kegg_desc: neighbor_ko.description
               }) as detailed_neighbors
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, protein_id=protein_id)
            return [dict(record) for record in result]
    
    async def _get_functional_annotations(self, function_query: str) -> List[Dict[str, Any]]:
        """Search for proteins by functional annotation."""
        cypher = """
        MATCH (ko:KEGGOrtholog)
        WHERE ko.id CONTAINS $query OR ko.description CONTAINS $query
        MATCH (p:Protein)-[:HASFUNCTION]->(ko)
        OPTIONAL MATCH (p)-[:BELONGSTOGENOME]->(g:Genome)
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
    
    async def _execute_domain_query_with_count(self, query: str) -> List[Dict[str, Any]]:
        """Execute domain query with accurate count information."""
        # Extract domain name from query with enhanced pattern matching
        domain_name = None
        import re
        
        # Direct domain name mentions
        if "GGDEF" in query:
            domain_name = "GGDEF"
        elif "TPR" in query:
            domain_name = "TPR"
        elif "/domain/" in query:
            # Extract domain name from pattern like "/domain/DOMAIN_NAME/"
            match = re.search(r'/domain/([^/]+)/', query)
            if match:
                domain_name = match.group(1)
        
        # Get accurate count first with flexible domain matching
        total_count = 0
        if domain_name:
            # Use flexible matching for domain families like TPR_1, TPR_2, etc.
            if domain_name == "TPR":
                count_query = "MATCH (da:DomainAnnotation) WHERE da.id CONTAINS '/domain/TPR' RETURN count(da) as total_count"
            else:
                count_query = f"MATCH (da:DomainAnnotation) WHERE da.id CONTAINS '/domain/{domain_name}/' RETURN count(da) as total_count"
            
            count_result = await self._execute_cypher(count_query)
            if count_result:
                total_count = count_result[0]['total_count']
        
        # Execute original query for sample results
        sample_results = await self._execute_cypher(query)
        
        # Add count metadata to each result
        for result in sample_results:
            result['_domain_total_count'] = total_count
            result['_domain_name'] = domain_name
            result['_is_sample'] = len(sample_results) < total_count
            result['_sample_size'] = len(sample_results)
        
        return sample_results
    
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
        # For cosine similarity: 1.0 = identical, values closer to 1.0 are more similar
        filtered_results = []
        for result in similar_results:
            if (result['protein_id'] != protein_id and 
                result['distance'] > 0.001 and  # Exclude nearly identical (cosine distance)
                result['similarity'] < 0.999):  # Exclude near-perfect matches (cosine similarity)
                filtered_results.append(result)
                if len(filtered_results) >= limit:
                    break
        
        return filtered_results
    
    async def _find_similar_by_embedding(self, embedding: np.ndarray, limit: int = 10) -> List[Dict[str, Any]]:
        """Find proteins similar to a given embedding."""
        # Use cosine distance which is proper for sequence embeddings
        results = self.table.search(embedding).metric("cosine").limit(limit).to_pandas()
        
        # Convert cosine distance to cosine similarity
        # Cosine distance = 1 - cosine similarity, so similarity = 1 - distance
        # This gives proper values in range [-1, 1] where 1 = identical
        return [
            {
                "protein_id": row['protein_id'],
                "genome_id": row['genome_id'],
                "sequence_length": row['sequence_length'],
                "distance": row['_distance'],
                "similarity": float(1.0 - row['_distance'])  # Cosine distance to cosine similarity
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