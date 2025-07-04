"""
Tests for the Batch Query Processor

Tests batch processing capabilities for large-scale genomic analyses.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import json

from src.build_kg.batch_processor import (
    BatchQueryProcessor,
    BatchQuery,
    BatchResult,
    BatchJob,
    batch_protein_analysis,
    batch_genome_comparison
)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver for testing."""
    with patch('src.build_kg.batch_processor.get_neo4j_driver') as mock_driver:
        # Mock session and result
        mock_session = Mock()
        mock_result = Mock()
        mock_result.data.return_value = [
            {"protein_id": "protein_1", "ko_id": "K00001", "definition": "alcohol dehydrogenase"},
            {"protein_id": "protein_2", "ko_id": "K00002", "definition": "aldehyde dehydrogenase"}
        ]
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        
        mock_driver_instance = Mock()
        mock_driver_instance.session.return_value = mock_session
        mock_driver.return_value = mock_driver_instance
        
        yield mock_driver


class TestBatchQueryProcessor:
    """Test the BatchQueryProcessor class."""
    
    def test_create_job(self, temp_output_dir):
        """Test job creation."""
        processor = BatchQueryProcessor(output_dir=temp_output_dir)
        
        queries = [
            {
                "id": "test_query_1",
                "query": "MATCH (p:Protein) RETURN count(p)",
                "priority": 2
            },
            {
                "id": "test_query_2", 
                "query": "MATCH (g:Genome) RETURN count(g)",
                "priority": 1
            }
        ]
        
        job_id = processor.create_job("test_job", queries, {"test": True})
        
        assert job_id in processor.jobs
        job = processor.jobs[job_id]
        assert job.name == "test_job"
        assert len(job.queries) == 2
        assert job.total_queries == 2
        assert job.status == "pending"
        
        # Check priority sorting (higher priority first)
        assert job.queries[0].priority == 2
        assert job.queries[1].priority == 1
        
        # Check statistics
        assert processor.stats["total_jobs"] == 1
        assert processor.stats["total_queries"] == 2
    
    @pytest.mark.asyncio
    async def test_execute_job_success(self, temp_output_dir, mock_neo4j_driver):
        """Test successful job execution."""
        processor = BatchQueryProcessor(output_dir=temp_output_dir, max_workers=2)
        
        queries = [
            {
                "query": "MATCH (p:Protein) RETURN p.id, p.sequence_length",
                "parameters": {"limit": 10}
            }
        ]
        
        job_id = processor.create_job("test_execution", queries)
        result_job = await processor.execute_job(job_id)
        
        assert result_job.status == "completed"
        assert result_job.progress == 1
        assert len(result_job.results) == 1
        assert result_job.results[0].success
        assert result_job.results[0].result is not None
        
        # Check that results were saved
        result_files = list(temp_output_dir.glob("*_results.json"))
        assert len(result_files) == 1
        
        # Verify result file content
        with open(result_files[0]) as f:
            saved_data = json.load(f)
        assert saved_data["job_id"] == job_id
        assert saved_data["status"] == "completed"
        assert len(saved_data["results"]) == 1
    
    @pytest.mark.asyncio
    async def test_execute_job_with_retries(self, temp_output_dir):
        """Test job execution with query failures and retries."""
        processor = BatchQueryProcessor(output_dir=temp_output_dir)
        
        # Mock driver that fails first few times then succeeds
        with patch('src.build_kg.batch_processor.get_neo4j_driver') as mock_driver:
            call_count = 0
            
            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 3:  # Fail first 2 attempts
                    raise Exception("Connection error")
                # Success on 3rd attempt
                mock_result = Mock()
                mock_result.data.return_value = [{"count": 42}]
                return mock_result
            
            mock_session = Mock()
            mock_session.run.side_effect = side_effect
            mock_session.__enter__ = Mock(return_value=mock_session)
            mock_session.__exit__ = Mock(return_value=False)
            
            mock_driver_instance = Mock()
            mock_driver_instance.session.return_value = mock_session
            mock_driver.return_value = mock_driver_instance
            
            queries = [{"query": "MATCH (n) RETURN count(n)", "max_retries": 3}]
            job_id = processor.create_job("retry_test", queries)
            
            result_job = await processor.execute_job(job_id)
            
            assert result_job.status == "completed"
            assert result_job.results[0].success
            assert result_job.results[0].retry_count == 2  # Succeeded on 3rd attempt (retry_count 2)
    
    def test_job_status_tracking(self, temp_output_dir):
        """Test job status and listing functionality."""
        processor = BatchQueryProcessor(output_dir=temp_output_dir)
        
        # Create multiple jobs
        job1_id = processor.create_job("job1", [{"query": "MATCH (n) RETURN n"}])
        job2_id = processor.create_job("job2", [{"query": "MATCH (m) RETURN m"}])
        
        # Test individual job status
        status1 = processor.get_job_status(job1_id)
        assert status1["name"] == "job1"
        assert status1["status"] == "pending"
        
        # Test job listing
        all_jobs = processor.list_jobs()
        assert len(all_jobs) == 2
        job_names = [job["name"] for job in all_jobs]
        assert "job1" in job_names
        assert "job2" in job_names
    
    def test_statistics_tracking(self, temp_output_dir):
        """Test statistics tracking."""
        processor = BatchQueryProcessor(output_dir=temp_output_dir)
        
        initial_stats = processor.get_stats()
        assert initial_stats["total_jobs"] == 0
        assert initial_stats["total_queries"] == 0
        
        # Create job with 3 queries
        queries = [{"query": f"MATCH (n) WHERE n.id = {i} RETURN n"} for i in range(3)]
        processor.create_job("stats_test", queries)
        
        updated_stats = processor.get_stats()
        assert updated_stats["total_jobs"] == 1
        assert updated_stats["total_queries"] == 3


class TestBatchAnalysisFunctions:
    """Test convenience functions for batch analyses."""
    
    @pytest.mark.asyncio
    async def test_batch_protein_analysis(self, temp_output_dir):
        """Test batch protein analysis function."""
        processor = BatchQueryProcessor(output_dir=temp_output_dir)
        
        protein_ids = [f"protein_{i}" for i in range(25)]  # 25 proteins
        
        job_id = await batch_protein_analysis(
            protein_ids=protein_ids,
            analysis_type="functional_annotation",
            batch_size=10,
            processor=processor
        )
        
        assert job_id in processor.jobs
        job = processor.jobs[job_id]
        
        # Should create 3 batches (10, 10, 5 proteins)
        assert len(job.queries) == 3
        assert "functional_annotation" in job.name
        assert job.metadata["total_proteins"] == 25
        assert job.metadata["batch_size"] == 10
        
        # Check query parameters
        batch_sizes = [len(query.parameters["protein_ids"]) for query in job.queries]
        assert batch_sizes == [10, 10, 5]
    
    @pytest.mark.asyncio
    async def test_batch_genome_comparison(self, temp_output_dir):
        """Test batch genome comparison function."""
        processor = BatchQueryProcessor(output_dir=temp_output_dir)
        
        genome_ids = ["genome_1", "genome_2", "genome_3"]
        
        job_id = await batch_genome_comparison(
            genome_ids=genome_ids,
            processor=processor
        )
        
        assert job_id in processor.jobs
        job = processor.jobs[job_id]
        
        # Should create 4 different analysis queries
        assert len(job.queries) == 4
        assert "genome_comparison" in job.name
        assert job.metadata["genome_count"] == 3
        
        # Check query types
        query_ids = [query.id for query in job.queries]
        expected_analyses = ["genome_statistics", "functional_profiles", "domain_architectures", "pathway_completeness"]
        assert all(analysis in query_ids for analysis in expected_analyses)
        
        # All queries should have same genome_ids parameter
        for query in job.queries:
            assert query.parameters["genome_ids"] == genome_ids
    
    @pytest.mark.asyncio
    async def test_invalid_analysis_type(self):
        """Test error handling for invalid analysis type."""
        with pytest.raises(ValueError, match="Unknown analysis type"):
            await batch_protein_analysis(
                protein_ids=["protein_1"],
                analysis_type="invalid_analysis",
            )


@pytest.mark.integration
class TestBatchProcessorIntegration:
    """Integration tests requiring actual database connection."""
    
    @pytest.mark.asyncio
    async def test_real_database_query(self, temp_output_dir):
        """Test with real database connection (requires Neo4j running)."""
        # Skip if no real database available
        try:
            from src.llm.config import get_neo4j_driver
            driver = get_neo4j_driver()
            with driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as count LIMIT 1")
                list(result)  # Consume result
        except Exception:
            pytest.skip("Neo4j database not available")
        
        processor = BatchQueryProcessor(output_dir=temp_output_dir)
        
        # Simple count query that should work on any database
        queries = [{"query": "MATCH (n) RETURN count(n) as total_nodes"}]
        job_id = processor.create_job("integration_test", queries)
        
        result_job = await processor.execute_job(job_id)
        
        assert result_job.status == "completed"
        assert len(result_job.results) == 1
        assert result_job.results[0].success
        assert result_job.results[0].result is not None