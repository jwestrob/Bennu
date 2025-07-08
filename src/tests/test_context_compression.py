#!/usr/bin/env python3
"""
Test script for context compression system.

This tests the multi-stage context compression for large genomic datasets.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.context_compression import ContextCompressor, CompressionLevel, GenomicDataCompressor

async def test_token_counting():
    """Test token counting functionality."""
    print("=== Testing Token Counting ===")
    
    compressor = ContextCompressor("gpt-3.5-turbo")
    
    test_text = "This is a test sentence with multiple words for token counting."
    token_count = compressor.token_counter.count_tokens(test_text)
    
    print(f"Test text: {test_text}")
    print(f"Token count: {token_count}")
    
    # Test truncation
    truncated = compressor.token_counter.truncate_to_tokens(test_text, 5)
    print(f"Truncated to 5 tokens: {truncated}")
    print()

def test_genomic_data_compression():
    """Test genomic-specific data compression."""
    print("=== Testing Genomic Data Compression ===")
    
    # Create test protein data
    test_proteins = []
    for i in range(100):
        protein = {
            'protein_id': f'protein_{i:03d}',
            'kegg_function': 'K00001 - alcohol dehydrogenase' if i % 3 == 0 else 'K00002 - aldehyde dehydrogenase',
            'pfam_domains': ['PF00107', 'PF08240'] if i % 2 == 0 else ['PF00107'],
            'start': 1000 + i * 1000,
            'end': 2000 + i * 1000,
            'strand': '+' if i % 2 == 0 else '-',
            'genome_id': f'genome_{chr(65 + i % 4)}'  # A, B, C, D
        }
        test_proteins.append(protein)
    
    print(f"Created {len(test_proteins)} test proteins")
    
    # Test different compression levels
    for level in [CompressionLevel.LIGHT, CompressionLevel.MEDIUM, CompressionLevel.AGGRESSIVE]:
        compressed, stats = GenomicDataCompressor.compress_protein_annotations(test_proteins, level)
        print(f"\n{level.value.upper()} compression:")
        print(f"  Stats: {stats}")
        print(f"  Length: {len(compressed)} characters")
        print(f"  Preview: {compressed[:200]}...")
    
    print()

async def test_context_compression_tiers():
    """Test different compression tiers."""
    print("=== Testing Context Compression Tiers ===")
    
    compressor = ContextCompressor("gpt-3.5-turbo")
    
    # Create test data of different sizes
    small_proteins = [{'protein_id': f'protein_{i}', 'kegg_function': 'K00001'} for i in range(10)]
    medium_proteins = [{'protein_id': f'protein_{i}', 'kegg_function': 'K00001'} for i in range(100)]
    large_proteins = [{'protein_id': f'protein_{i}', 'kegg_function': 'K00001'} for i in range(1000)]
    
    test_cases = [
        ("small", {"structured_data": small_proteins}),
        ("medium", {"structured_data": medium_proteins}),
        ("large", {"structured_data": large_proteins})
    ]
    
    for case_name, context_data in test_cases:
        print(f"\nTesting {case_name} dataset:")
        
        # Test compression
        result = await compressor.compress_context(context_data, target_tokens=5000)
        
        print(f"  Original size: {result.original_size} tokens")
        print(f"  Compressed size: {result.compressed_size} tokens")
        print(f"  Compression ratio: {result.compression_ratio:.2f}")
        print(f"  Compression level: {result.compression_level.value}")
        print(f"  Chunks processed: {result.chunks_processed}")
        print(f"  Strategy: {compressor.get_compression_strategy(result.original_size)[0]}")
    
    print()

async def test_multi_stage_compression():
    """Test multi-stage compression for very large datasets."""
    print("=== Testing Multi-Stage Compression ===")
    
    compressor = ContextCompressor("gpt-3.5-turbo")
    
    # Create very large dataset
    huge_proteins = []
    for i in range(2000):
        protein = {
            'protein_id': f'huge_protein_{i:04d}',
            'kegg_function': f'K{(i % 100):05d} - test function {i}',
            'pfam_domains': [f'PF{j:05d}' for j in range(i % 5 + 1)],
            'start': i * 1000,
            'end': (i + 1) * 1000,
            'strand': '+' if i % 2 == 0 else '-',
            'genome_id': f'genome_{chr(65 + i % 10)}',
            'additional_data': f'Extra annotation data for protein {i} ' * 10
        }
        huge_proteins.append(protein)
    
    context_data = {
        "structured_data": huge_proteins,
        "semantic_data": [
            {'protein_id': f'sim_{i}', 'similarity': 0.9 - i * 0.1, 'function': f'similar protein {i}'}
            for i in range(50)
        ],
        "tool_results": [
            {'task_id': f'task_{i}', 'tool_name': 'analysis', 'result': f'Analysis result {i} ' * 20}
            for i in range(10)
        ]
    }
    
    print(f"Created dataset with {len(huge_proteins)} proteins, {len(context_data['semantic_data'])} semantic results, {len(context_data['tool_results'])} tool results")
    
    # Test multi-stage compression
    result = await compressor.compress_context(context_data, target_tokens=8000)
    
    print(f"Multi-stage compression results:")
    print(f"  Original size: {result.original_size} tokens")
    print(f"  Compressed size: {result.compressed_size} tokens")
    print(f"  Compression ratio: {result.compression_ratio:.2f}")
    print(f"  Compression level: {result.compression_level.value}")
    print(f"  Chunks processed: {result.chunks_processed}")
    print(f"  Preserved elements: {result.preserved_elements}")
    print(f"  Summary stats: {result.summary_stats}")
    
    print(f"\nCompressed content preview:")
    print(result.compressed_content[:500] + "..." if len(result.compressed_content) > 500 else result.compressed_content)
    print()

async def main():
    """Run all compression tests."""
    print("🧬 Context Compression System Tests")
    print("=" * 50)
    
    await test_token_counting()
    test_genomic_data_compression()
    await test_context_compression_tiers()
    await test_multi_stage_compression()
    
    print("✅ All compression tests completed!")

if __name__ == "__main__":
    asyncio.run(main())