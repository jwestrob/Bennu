#!/usr/bin/env python3
"""
Unit tests for ResultStreamer functionality.
Tests chunking of large datasets to prevent context window overflow.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from typing import Iterator, Dict, Any
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.llm.rag_system import ResultStreamer


class TestResultStreamer:
    """Test suite for ResultStreamer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "test_outputs"
        self.streamer = ResultStreamer(chunk_context_size=4096, output_dir=str(self.output_dir))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def generate_synthetic_data(self, target_tokens: int, record_size_tokens: int = 100) -> Iterator[Dict[str, Any]]:
        """
        Generate synthetic data iterator with approximately target_tokens total.
        
        Args:
            target_tokens: Target total token count
            record_size_tokens: Approximate tokens per record
            
        Yields:
            Dict records with synthetic genomic data
        """
        # Calculate approximate number of records needed
        num_records = target_tokens // record_size_tokens
        
        for i in range(num_records):
            # Create a record that's approximately record_size_tokens when JSON-encoded
            # Assuming ~4 chars per token, create strings of appropriate length
            protein_id = f"protein_{i:06d}"
            
            # Create description that fills most of the token budget
            desc_chars = (record_size_tokens - 20) * 4  # Leave room for other fields
            description = f"Synthetic protein {i} with function " + "X" * max(0, desc_chars - 50)
            
            record = {
                "protein_id": protein_id,
                "genome_id": f"genome_{i % 10}",
                "start": i * 1000,
                "end": (i * 1000) + 300,
                "strand": 1 if i % 2 == 0 else -1,
                "description": description,
                "pfam_domains": [f"PF{(i % 1000):05d}"],
                "kegg_orthologs": [f"K{(i % 10000):05d}"],
                "sequence": "M" + "A" * 99,  # 100 AA sequence
                "metadata": {
                    "gc_content": 0.5 + (i % 100) / 1000,
                    "length": 300,
                    "partial": False
                }
            }
            yield record
    
    def test_token_counting(self):
        """Test token counting functionality."""
        test_text = "This is a test string for token counting."
        token_count = self.streamer._count_tokens(test_text)
        
        # Should return a reasonable token count (not zero, not way too high)
        assert token_count > 0
        assert token_count < len(test_text)  # Should be less than character count
        assert token_count < 20  # Reasonable upper bound for this short text
    
    def test_session_creation(self):
        """Test session directory creation."""
        session_id = self.streamer._create_session()
        
        assert session_id is not None
        assert session_id.startswith("session_")
        assert self.streamer.session_dir.exists()
        assert self.streamer.session_dir.is_dir()
    
    def test_chunk_writing(self):
        """Test JSONL chunk file writing."""
        # Create test data
        test_data = [
            {"id": 1, "name": "test1", "value": "data1"},
            {"id": 2, "name": "test2", "value": "data2"}
        ]
        
        # Create session and write chunk
        self.streamer._create_session()
        chunk_file = self.streamer.session_dir / "test_chunk.jsonl"
        self.streamer._write_chunk(chunk_file, test_data)
        
        # Verify file exists and content is correct
        assert chunk_file.exists()
        
        # Read back and verify content
        with open(chunk_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 2
        
        # Parse each line as JSON
        record1 = json.loads(lines[0].strip())
        record2 = json.loads(lines[1].strip())
        
        assert record1 == test_data[0]
        assert record2 == test_data[1]
    
    def test_small_dataset_streaming(self):
        """Test streaming with small dataset (single chunk)."""
        # Generate small dataset
        data_iterator = self.generate_synthetic_data(target_tokens=2000, record_size_tokens=100)
        
        # Stream the data
        summaries = self.streamer.stream_results(data_iterator)
        
        # Should create exactly one chunk
        assert len(summaries) == 1
        assert summaries[0].startswith("chunk 000:")
        assert "rows" in summaries[0]
        
        # Verify session info
        session_info = self.streamer.get_session_info()
        assert session_info["chunks"] == 1
        assert session_info["session_id"] is not None
    
    def test_large_dataset_streaming_25k_tokens(self):
        """
        Test streaming with 25k token dataset at 4k chunk size.
        Should create approximately 7 chunks.
        """
        # Generate 25k token dataset
        data_iterator = self.generate_synthetic_data(target_tokens=25000, record_size_tokens=200)
        
        # Stream with 4k chunk size
        summaries = self.streamer.stream_results(data_iterator)
        
        # Should create approximately 7 chunks (25k / 4k ≈ 6.25, rounded up)
        assert len(summaries) >= 6
        assert len(summaries) <= 8  # Allow some variance due to token counting approximation
        
        # Verify chunk naming
        for i, summary in enumerate(summaries):
            expected_prefix = f"chunk {i:03d}:"
            assert summary.startswith(expected_prefix)
            assert "rows" in summary
        
        # Verify all chunk files exist
        session_info = self.streamer.get_session_info()
        assert session_info["chunks"] == len(summaries)
        
        # Verify chunk files are properly named
        chunk_files = session_info["chunk_files"]
        for i in range(len(summaries)):
            expected_filename = f"chunk_{i:03d}.jsonl"
            assert expected_filename in chunk_files
    
    def test_exact_7_chunks_expectation(self):
        """
        Test that we get approximately 7 chunks with 25k token dataset.
        This is the specific test case mentioned in the requirements.
        """
        # Create data that should result in approximately 7 chunks
        # 25k tokens / 4k chunk size ≈ 6.25, so expect 6-8 chunks
        data_iterator = self.generate_synthetic_data(target_tokens=25000, record_size_tokens=150)
        
        # Stream with 4k chunk size
        summaries = self.streamer.stream_results(data_iterator)
        
        # Should create approximately 7 chunks (allow 6-8 due to token estimation variance)
        assert len(summaries) >= 6, f"Expected at least 6 chunks, got {len(summaries)}: {summaries}"
        assert len(summaries) <= 8, f"Expected at most 8 chunks, got {len(summaries)}: {summaries}"
        
        # Verify chunk numbering
        for i, summary in enumerate(summaries):
            expected_prefix = f"chunk {i:03d}:"
            assert summary.startswith(expected_prefix)
    
    def test_no_message_output_length_error(self):
        """
        Test that streaming prevents MessageOutputLengthError by chunking large datasets.
        This simulates the scenario where a large result set would overflow the context window.
        """
        # Generate very large dataset that would definitely cause context overflow
        large_data_iterator = self.generate_synthetic_data(target_tokens=50000, record_size_tokens=300)
        
        # This should NOT raise any MessageOutputLengthError or similar exceptions
        try:
            summaries = self.streamer.stream_results(large_data_iterator)
            
            # Should successfully create multiple chunks
            assert len(summaries) > 10  # Should be many chunks for 50k tokens
            
            # Each summary should be short and manageable
            for summary in summaries:
                assert len(summary) < 100  # Summaries should be very short
                assert "chunk" in summary
                assert "rows" in summary
                
        except Exception as e:
            # Should not raise MessageOutputLengthError or similar context window errors
            error_msg = str(e).lower()
            assert "length" not in error_msg
            assert "context" not in error_msg
            assert "token" not in error_msg
            # Re-raise if it's a different type of error
            raise
    
    def test_session_info_accuracy(self):
        """Test that session info provides accurate metadata."""
        # Generate test data
        data_iterator = self.generate_synthetic_data(target_tokens=10000, record_size_tokens=200)
        
        # Stream the data
        summaries = self.streamer.stream_results(data_iterator)
        
        # Get session info
        session_info = self.streamer.get_session_info()
        
        # Verify accuracy
        assert session_info["chunks"] == len(summaries)
        assert session_info["session_id"] is not None
        assert session_info["total_size"] > 0
        assert len(session_info["chunk_files"]) == len(summaries)
        
        # Verify chunk files actually exist
        for chunk_file in session_info["chunk_files"]:
            chunk_path = Path(session_info["session_dir"]) / chunk_file
            assert chunk_path.exists()
            assert chunk_path.stat().st_size > 0  # Files should not be empty
    
    def test_empty_iterator(self):
        """Test streaming with empty iterator."""
        empty_iterator = iter([])
        
        summaries = self.streamer.stream_results(empty_iterator)
        
        # Should return empty list for empty input
        assert summaries == []
        
        # Session should still be created but with no chunks
        session_info = self.streamer.get_session_info()
        assert session_info["chunks"] == 0


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])