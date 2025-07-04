"""
Batch Query Processor for Large-Scale Genomic Analyses

Implements efficient processing of large-scale genomic analyses with:
- Queue system for large analyses  
- Parallel Neo4j queries
- Progress tracking with rich
- Graceful partial failure handling
- Batch result aggregation

Part of Phase 2: Performance Optimization
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from uuid import uuid4

from rich.console import Console
from rich.progress import Progress, TaskID, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

# Import genomic system components
from neo4j import GraphDatabase
from ..llm.config import LLMConfig


def get_neo4j_driver():
    """Get Neo4j driver instance."""
    config = LLMConfig.from_env()
    return GraphDatabase.driver(
        config.database.neo4j_uri,
        auth=(config.database.neo4j_user, config.database.neo4j_password)
    )


@dataclass
class BatchQuery:
    """Individual query in a batch processing job."""
    id: str
    query: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # Higher number = higher priority
    timeout: int = 60  # Query timeout in seconds
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result of a batch query execution."""
    query_id: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchJob:
    """Container for a batch processing job."""
    id: str
    name: str
    queries: List[BatchQuery]
    created_at: float = field(default_factory=time.time)
    status: str = "pending"  # pending, running, completed, failed
    progress: int = 0
    total_queries: int = 0
    results: List[BatchResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BatchQueryProcessor:
    """
    High-performance batch query processor for genomic analyses.
    
    Features:
    - Parallel Neo4j query execution
    - Priority-based queue system
    - Progress tracking and monitoring
    - Graceful error handling and retries
    - Result aggregation and export
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        max_concurrent_queries: int = 10,
        default_timeout: int = 60,
        output_dir: Path = Path("data/batch_outputs")
    ):
        self.max_workers = max_workers
        self.max_concurrent_queries = max_concurrent_queries
        self.default_timeout = default_timeout
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.console = Console()
        self.logger = logging.getLogger(__name__)
        
        # Job management
        self.jobs: Dict[str, BatchJob] = {}
        self.job_queue: List[str] = []
        self.running_jobs: Dict[str, asyncio.Task] = {}
        
        # Performance tracking
        self.stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_execution_time": 0.0
        }
        
    def create_job(
        self,
        name: str,
        queries: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new batch job."""
        job_id = str(uuid4())
        
        # Convert query dictionaries to BatchQuery objects
        batch_queries = []
        for i, query_data in enumerate(queries):
            query_id = query_data.get('id', f"{job_id}_query_{i}")
            batch_query = BatchQuery(
                id=query_id,
                query=query_data['query'],
                parameters=query_data.get('parameters', {}),
                priority=query_data.get('priority', 1),
                timeout=query_data.get('timeout', self.default_timeout),
                metadata=query_data.get('metadata', {})
            )
            batch_queries.append(batch_query)
        
        # Sort by priority (higher priority first)
        batch_queries.sort(key=lambda q: q.priority, reverse=True)
        
        job = BatchJob(
            id=job_id,
            name=name,
            queries=batch_queries,
            total_queries=len(batch_queries),
            metadata=metadata or {}
        )
        
        self.jobs[job_id] = job
        self.job_queue.append(job_id)
        self.stats["total_jobs"] += 1
        self.stats["total_queries"] += len(batch_queries)
        
        self.logger.info(f"Created batch job '{name}' with {len(batch_queries)} queries")
        return job_id
    
    async def execute_job(self, job_id: str) -> BatchJob:
        """Execute a batch job with parallel query processing."""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.jobs[job_id]
        job.status = "running"
        
        self.console.print(f"[bold blue]Starting batch job: {job.name}[/bold blue]")
        
        # Progress tracking setup
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task(f"Processing {job.name}", total=job.total_queries)
            
            # Execute queries with controlled concurrency
            semaphore = asyncio.Semaphore(self.max_concurrent_queries)
            driver = get_neo4j_driver()
            
            async def execute_single_query(query: BatchQuery) -> BatchResult:
                """Execute a single query with error handling and retries."""
                async with semaphore:
                    start_time = time.time()
                    
                    for attempt in range(query.max_retries + 1):
                        try:
                            # Execute Neo4j query
                            with driver.session() as session:
                                result = session.run(query.query, query.parameters)
                                records = [record.data() for record in result]
                                
                            execution_time = time.time() - start_time
                            
                            return BatchResult(
                                query_id=query.id,
                                success=True,
                                result=records,
                                execution_time=execution_time,
                                retry_count=attempt,
                                metadata=query.metadata
                            )
                            
                        except Exception as e:
                            query.retry_count = attempt
                            if attempt == query.max_retries:
                                execution_time = time.time() - start_time
                                self.logger.error(f"Query {query.id} failed after {attempt + 1} attempts: {e}")
                                
                                return BatchResult(
                                    query_id=query.id,
                                    success=False,
                                    error=str(e),
                                    execution_time=execution_time,
                                    retry_count=attempt,
                                    metadata=query.metadata
                                )
                            else:
                                self.logger.warning(f"Query {query.id} attempt {attempt + 1} failed: {e}. Retrying...")
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            # Create tasks for all queries
            tasks = [execute_single_query(query) for query in job.queries]
            
            # Process tasks as they complete
            for task_future in asyncio.as_completed(tasks):
                result = await task_future
                job.results.append(result)
                job.progress += 1
                
                # Update progress
                progress.update(task, advance=1)
                
                # Update statistics
                if result.success:
                    self.stats["successful_queries"] += 1
                else:
                    self.stats["failed_queries"] += 1
                self.stats["total_execution_time"] += result.execution_time
        
        # Finalize job
        job.status = "completed" if all(r.success for r in job.results) else "failed"
        if job.status == "completed":
            self.stats["completed_jobs"] += 1
        else:
            self.stats["failed_jobs"] += 1
        
        # Save results
        await self._save_job_results(job)
        
        self.console.print(f"[bold green]Completed batch job: {job.name}[/bold green]")
        self._print_job_summary(job)
        
        return job
    
    def queue_job(self, job_id: str) -> None:
        """Add a job to the processing queue."""
        if job_id not in self.job_queue:
            self.job_queue.append(job_id)
    
    async def process_queue(self) -> None:
        """Process all jobs in the queue."""
        while self.job_queue:
            job_id = self.job_queue.pop(0)
            if job_id in self.jobs:
                try:
                    await self.execute_job(job_id)
                except Exception as e:
                    self.logger.error(f"Failed to execute job {job_id}: {e}")
                    if job_id in self.jobs:
                        self.jobs[job_id].status = "failed"
                        self.stats["failed_jobs"] += 1
    
    async def _save_job_results(self, job: BatchJob) -> None:
        """Save job results to disk."""
        output_file = self.output_dir / f"{job.id}_{job.name.replace(' ', '_')}_results.json"
        
        # Prepare results for serialization
        job_data = {
            "job_id": job.id,
            "name": job.name,
            "status": job.status,
            "created_at": job.created_at,
            "total_queries": job.total_queries,
            "progress": job.progress,
            "metadata": job.metadata,
            "results": [
                {
                    "query_id": r.query_id,
                    "success": r.success,
                    "result": r.result,
                    "error": r.error,
                    "execution_time": r.execution_time,
                    "retry_count": r.retry_count,
                    "metadata": r.metadata
                }
                for r in job.results
            ],
            "summary": {
                "successful_queries": sum(1 for r in job.results if r.success),
                "failed_queries": sum(1 for r in job.results if not r.success),
                "total_execution_time": sum(r.execution_time for r in job.results),
                "average_execution_time": sum(r.execution_time for r in job.results) / len(job.results) if job.results else 0
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(job_data, f, indent=2, default=str)
        
        self.logger.info(f"Saved job results to {output_file}")
    
    def _print_job_summary(self, job: BatchJob) -> None:
        """Print a summary of job execution."""
        table = Table(title=f"Job Summary: {job.name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        successful = sum(1 for r in job.results if r.success)
        failed = sum(1 for r in job.results if not r.success)
        total_time = sum(r.execution_time for r in job.results)
        avg_time = total_time / len(job.results) if job.results else 0
        
        table.add_row("Status", job.status.upper())
        table.add_row("Total Queries", str(job.total_queries))
        table.add_row("Successful", str(successful))
        table.add_row("Failed", str(failed))
        table.add_row("Success Rate", f"{(successful/job.total_queries)*100:.1f}%" if job.total_queries > 0 else "0%")
        table.add_row("Total Time", f"{total_time:.2f}s")
        table.add_row("Average Time", f"{avg_time:.3f}s")
        
        self.console.print(table)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return self.stats.copy()
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        return {
            "id": job.id,
            "name": job.name,
            "status": job.status,
            "progress": job.progress,
            "total_queries": job.total_queries,
            "created_at": job.created_at
        }
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs."""
        return [self.get_job_status(job_id) for job_id in self.jobs.keys()]


# Convenience functions for common genomic analyses

async def batch_protein_analysis(
    protein_ids: List[str],
    analysis_type: str = "functional_annotation",
    batch_size: int = 100,
    processor: Optional[BatchQueryProcessor] = None
) -> str:
    """
    Perform batch analysis on a list of proteins.
    
    Args:
        protein_ids: List of protein IDs to analyze
        analysis_type: Type of analysis ('functional_annotation', 'domain_analysis', 'pathway_analysis')
        batch_size: Number of proteins to process per query
        processor: BatchQueryProcessor instance (created if None)
    
    Returns:
        Job ID for tracking
    """
    if processor is None:
        processor = BatchQueryProcessor()
    
    # Define query templates for different analysis types
    query_templates = {
        "functional_annotation": """
            MATCH (p:Protein)-[:HASFUNCTION]->(ko:KEGGOrtholog)
            WHERE p.id IN $protein_ids
            RETURN p.id as protein_id, p.sequence_length, ko.id as ko_id, ko.definition
            ORDER BY p.id
        """,
        "domain_analysis": """
            MATCH (p:Protein)-[:HASDOMAIN]->(d:Domain)
            WHERE p.id IN $protein_ids
            RETURN p.id as protein_id, d.accession, d.name, d.description, d.score, d.evalue
            ORDER BY p.id, d.score DESC
        """,
        "pathway_analysis": """
            MATCH (p:Protein)-[:HASFUNCTION]->(ko:KEGGOrtholog)-[:PARTICIPATESIN]->(pw:Pathway)
            WHERE p.id IN $protein_ids
            RETURN p.id as protein_id, ko.id as ko_id, pw.id as pathway_id, pw.name as pathway_name
            ORDER BY p.id
        """
    }
    
    if analysis_type not in query_templates:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    query_template = query_templates[analysis_type]
    
    # Split protein IDs into batches
    protein_batches = [protein_ids[i:i + batch_size] for i in range(0, len(protein_ids), batch_size)]
    
    # Create queries for each batch
    queries = []
    for i, batch in enumerate(protein_batches):
        query_data = {
            "id": f"{analysis_type}_batch_{i}",
            "query": query_template,
            "parameters": {"protein_ids": batch},
            "priority": 1,
            "metadata": {
                "analysis_type": analysis_type,
                "batch_number": i,
                "protein_count": len(batch)
            }
        }
        queries.append(query_data)
    
    # Create and queue job
    job_name = f"{analysis_type}_{len(protein_ids)}_proteins"
    job_id = processor.create_job(job_name, queries, {
        "analysis_type": analysis_type,
        "total_proteins": len(protein_ids),
        "batch_size": batch_size
    })
    
    return job_id


async def batch_genome_comparison(
    genome_ids: List[str],
    processor: Optional[BatchQueryProcessor] = None
) -> str:
    """
    Perform comparative analysis across multiple genomes.
    
    Args:
        genome_ids: List of genome IDs to compare
        processor: BatchQueryProcessor instance (created if None)
    
    Returns:
        Job ID for tracking
    """
    if processor is None:
        processor = BatchQueryProcessor()
    
    queries = []
    
    # Query 1: Basic genome statistics
    queries.append({
        "id": "genome_statistics",
        "query": """
            MATCH (g:Genome)
            WHERE g.id IN $genome_ids
            OPTIONAL MATCH (g)<-[:BELONGSTOGENOME]-(gene:Gene)
            OPTIONAL MATCH (gene)-[:ENCODEDBY]->(p:Protein)
            OPTIONAL MATCH (p)-[:HASFUNCTION]->(ko:KEGGOrtholog)
            OPTIONAL MATCH (p)-[:HASDOMAIN]->(d:Domain)
            RETURN g.id as genome_id, g.assembly_length, g.gc_content,
                   count(DISTINCT gene) as gene_count,
                   count(DISTINCT p) as protein_count,
                   count(DISTINCT ko) as function_count,
                   count(DISTINCT d) as domain_count
        """,
        "parameters": {"genome_ids": genome_ids},
        "priority": 3,
        "metadata": {"analysis": "basic_statistics"}
    })
    
    # Query 2: Functional profile comparison
    queries.append({
        "id": "functional_profiles",
        "query": """
            MATCH (g:Genome)<-[:BELONGSTOGENOME]-(gene:Gene)-[:ENCODEDBY]->(p:Protein)-[:HASFUNCTION]->(ko:KEGGOrtholog)
            WHERE g.id IN $genome_ids
            RETURN g.id as genome_id, ko.id as ko_id, ko.definition, count(p) as protein_count
            ORDER BY g.id, protein_count DESC
        """,
        "parameters": {"genome_ids": genome_ids},
        "priority": 2,
        "metadata": {"analysis": "functional_profiles"}
    })
    
    # Query 3: Domain architecture comparison  
    queries.append({
        "id": "domain_architectures",
        "query": """
            MATCH (g:Genome)<-[:BELONGSTOGENOME]-(gene:Gene)-[:ENCODEDBY]->(p:Protein)-[:HASDOMAIN]->(d:Domain)
            WHERE g.id IN $genome_ids
            RETURN g.id as genome_id, d.accession as domain_accession, d.name as domain_name, 
                   count(p) as protein_count, avg(d.score) as avg_score
            ORDER BY g.id, protein_count DESC
        """,
        "parameters": {"genome_ids": genome_ids},
        "priority": 2,
        "metadata": {"analysis": "domain_architectures"}
    })
    
    # Query 4: Pathway completeness
    queries.append({
        "id": "pathway_completeness",
        "query": """
            MATCH (g:Genome)<-[:BELONGSTOGENOME]-(gene:Gene)-[:ENCODEDBY]->(p:Protein)-[:HASFUNCTION]->(ko:KEGGOrtholog)-[:PARTICIPATESIN]->(pw:Pathway)
            WHERE g.id IN $genome_ids
            RETURN g.id as genome_id, pw.id as pathway_id, pw.name as pathway_name,
                   count(DISTINCT ko) as present_kos,
                   collect(DISTINCT ko.id) as ko_list
            ORDER BY g.id, present_kos DESC
        """,
        "parameters": {"genome_ids": genome_ids},
        "priority": 1,
        "metadata": {"analysis": "pathway_completeness"}
    })
    
    job_name = f"genome_comparison_{len(genome_ids)}_genomes"
    job_id = processor.create_job(job_name, queries, {
        "analysis_type": "genome_comparison",
        "genome_count": len(genome_ids),
        "genome_ids": genome_ids
    })
    
    return job_id


# Export main classes and functions
__all__ = [
    "BatchQueryProcessor",
    "BatchQuery", 
    "BatchResult",
    "BatchJob",
    "batch_protein_analysis",
    "batch_genome_comparison"
]