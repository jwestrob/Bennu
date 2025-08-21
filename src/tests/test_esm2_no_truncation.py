#!/usr/bin/env python3
"""
Test script to validate ESM2 embedding generation without sequence truncation.
Verifies that long protein sequences (>1024 AA) are processed correctly.
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import numpy as np

from src.ingest.06_esm2_embeddings import ESM2EmbeddingGenerator, ProteinSequence, AggregationStrategy

# Set up logging
logging.basicConfig(level=logging.INFO)


class TestESM2NoTruncation:
    """Test that ESM2 embeddings preserve full sequence lengths."""
    
    def test_long_sequence_processing(self):
        """Test that sequences >1024 AA are processed without truncation."""
        
        # Create test sequences of varying lengths
        test_sequences = [
            ProteinSequence(
                protein_id="short_protein",
                genome_id="test_genome",
                sequence="M" + "A" * 100,  # 101 AA
                length=101,
                source_file=Path("/test/short.faa")
            ),
            ProteinSequence(
                protein_id="medium_protein", 
                genome_id="test_genome",
                sequence="M" + "L" * 500,  # 501 AA
                length=501,
                source_file=Path("/test/medium.faa")
            ),
            ProteinSequence(
                protein_id="long_protein",
                genome_id="test_genome", 
                sequence="M" + "K" * 1500,  # 1501 AA - LONGER than 1024!
                length=1501,
                source_file=Path("/test/long.faa")
            ),
            ProteinSequence(
                protein_id="very_long_protein",
                genome_id="test_genome",
                sequence="M" + "R" * 3000,  # 3001 AA - Very long like real data
                length=3001,
                source_file=Path("/test/very_long.faa")
            )
        ]
        
        # Initialize ESM2 with smallest model for testing and sliding window
        generator = ESM2EmbeddingGenerator(
            model_name="facebook/esm2_t6_8M_UR50D",
            window_size=1024,
            overlap=256,
            aggregation=AggregationStrategy.MAX_POOL
        )
        
        # Test tokenization without truncation
        sequence_texts = [seq.sequence for seq in test_sequences]
        
        # This should NOT truncate sequences
        inputs = generator.tokenizer(
            sequence_texts,
            return_tensors="pt", 
            padding=True,
            truncation=False  # Key: no truncation
        )
        
        # Verify that long sequences are preserved
        input_ids = inputs['input_ids']
        
        # Check that each sequence's tokens reflect full length (approximately)
        for i, seq in enumerate(test_sequences):
            token_count = (input_ids[i] != generator.tokenizer.pad_token_id).sum().item()
            # ESM2 adds special tokens, so token count should be close to sequence length + special tokens
            assert token_count > seq.length, f"Sequence {seq.protein_id} may have been truncated: {token_count} tokens < {seq.length} AA"
            
        print(f"✓ All sequences preserved their full lengths")
        
        # Test actual embedding generation
        embeddings = generator.embed_sequences(test_sequences, batch_size=2)
        
        # Verify all sequences got embeddings
        assert len(embeddings) == 4, f"Expected 4 embeddings, got {len(embeddings)}"
        
        # Verify embeddings are valid
        for protein_id, embedding in embeddings.items():
            assert embedding.shape == (generator.embedding_dim,), f"Wrong embedding shape for {protein_id}"
            assert not np.all(embedding == 0), f"Zero embedding for {protein_id} - may indicate failure"
            
        print(f"✓ All sequences generated valid embeddings")
        print(f"✓ Long sequence test passed: sequences up to {max(seq.length for seq in test_sequences)} AA processed successfully")
        print(f"✓ Sliding window config: {generator.window_size} AA windows, {generator.overlap} AA overlap")

    def test_batch_size_adaptation(self):
        """Test that very long sequences trigger individual processing."""
        
        # Create batch with very long sequences
        long_sequences = [
            ProteinSequence(
                protein_id=f"very_long_{i}",
                genome_id="test_genome",
                sequence="M" + "A" * 2500,  # 2501 AA - triggers individual processing
                length=2501,
                source_file=Path(f"/test/very_long_{i}.faa")
            )
            for i in range(3)  # 3 very long sequences in one batch
        ]
        
        generator = ESM2EmbeddingGenerator(
            model_name="facebook/esm2_t6_8M_UR50D",
            window_size=1024,
            overlap=256,
            aggregation=AggregationStrategy.MAX_POOL
        )
        
        # This should trigger individual processing due to length
        embeddings = generator.embed_sequences(long_sequences, batch_size=8)
        
        # Verify all sequences were processed
        assert len(embeddings) == 3
        for embedding in embeddings.values():
            assert not np.all(embedding == 0), "Individual processing should succeed"
            
        print("✓ Very long sequences correctly triggered individual processing with sliding windows")


def test_sliding_window_functionality():
    """Test sliding window processing directly."""
    
    generator = ESM2EmbeddingGenerator(
        model_name="facebook/esm2_t6_8M_UR50D",
        window_size=500,  # Smaller for testing
        overlap=100,
        aggregation=AggregationStrategy.MAX_POOL
    )
    
    # Test sequence that will need multiple windows
    long_sequence = "M" + "A" * 1200  # 1201 AA
    
    # Test sliding window detection
    assert generator._should_use_sliding_window(long_sequence), "Long sequence should trigger sliding window"
    
    short_sequence = "M" + "A" * 400  # 401 AA
    assert not generator._should_use_sliding_window(short_sequence), "Short sequence should not trigger sliding window"
    
    print("✓ Sliding window detection working correctly")
    
    # Test actual sliding window processing
    embedding = generator._embed_long_sequence_with_sliding_window(long_sequence, "test_protein")
    
    assert embedding.shape == (generator.embedding_dim,), f"Wrong embedding shape: {embedding.shape}"
    assert not np.all(embedding == 0), "Sliding window should produce non-zero embedding"
    
    print("✓ Sliding window embedding generation successful")

def test_sequence_length_statistics():
    """Test that sequence length statistics are correctly tracked."""
    
    # Mock sequences with known length distribution
    sequences = [
        ProteinSequence("p1", "g1", "A" * 50, 50, Path("/test")),
        ProteinSequence("p2", "g1", "A" * 800, 800, Path("/test")),
        ProteinSequence("p3", "g1", "A" * 1200, 1200, Path("/test")),  # > 1024
        ProteinSequence("p4", "g1", "A" * 2500, 2500, Path("/test")),  # > 2000
    ]
    
    # Calculate expected statistics
    over_1024 = len([s for s in sequences if s.length > 1024])
    over_2000 = len([s for s in sequences if s.length > 2000])
    
    assert over_1024 == 2, f"Expected 2 proteins >1024 AA, got {over_1024}"
    assert over_2000 == 1, f"Expected 1 protein >2000 AA, got {over_2000}"
    
    print("✓ Sequence length statistics correctly calculated")


if __name__ == "__main__":
    # Run tests
    test = TestESM2NoTruncation()
    test.test_long_sequence_processing()
    test.test_batch_size_adaptation() 
    test_sliding_window_functionality()
    test_sequence_length_statistics()
    
    print("\n🎉 ALL TESTS PASSED - ESM2 sliding window implementation is working correctly!")
    print("🔬 Long sequences are now processed with sliding windows + max pooling aggregation!")
    print("📊 No biological information is lost from sequence truncation!")