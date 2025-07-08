#!/usr/bin/env python3
"""
Integration test for RAG system with context compression.

Tests that the context compression integrates properly with the main RAG system.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.rag_system.core import GenomicRAG
from llm.config import LLMConfig

async def test_rag_compression_integration():
    """Test that RAG system can handle large contexts with compression."""
    print("🧬 Testing RAG System with Context Compression")
    print("=" * 50)
    
    # Create RAG system
    config = LLMConfig()
    rag = GenomicRAG(config)
    
    try:
        # Test health check
        health = rag.health_check()
        print(f"System health: {health}")
        
        # Create mock large context data for testing compression
        large_context_data = {
            'structured_data': [
                {
                    'protein_id': f'test_protein_{i:04d}',
                    'kegg_function': f'K{(i % 100):05d} - test enzyme {i}',
                    'genome_id': f'test_genome_{chr(65 + i % 4)}',
                    'start': i * 1000,
                    'end': (i + 1) * 1000,
                    'pfam_domains': [f'PF{j:05d}' for j in range(i % 3 + 1)],
                    'description': f'Test protein {i} with extensive annotation data ' * 10
                }
                for i in range(500)  # Large dataset to trigger compression
            ],
            'semantic_data': [
                {
                    'protein_id': f'similar_{i}',
                    'similarity': 0.9 - i * 0.01,
                    'function': f'Similar protein {i}'
                }
                for i in range(50)
            ],
            'tool_results': [
                {
                    'task_id': f'analysis_task_{i}',
                    'tool_name': 'test_analyzer',
                    'result': f'Analysis result {i}: ' + 'detailed findings ' * 20
                }
                for i in range(10)
            ]
        }
        
        # Test organizing results for compression
        organized_data = rag._organize_results_for_compression({'test_task': large_context_data})
        print(f"Organized data keys: {organized_data.keys()}")
        print(f"Structured data count: {len(organized_data['structured_data'])}")
        print(f"Semantic data count: {len(organized_data['semantic_data'])}")
        print(f"Tool results count: {len(organized_data['tool_results'])}")
        
        # Import compression module to test directly
        from llm.context_compression import ContextCompressor
        
        compressor = ContextCompressor('gpt-3.5-turbo')
        original_size = compressor.token_counter.count_tokens(
            compressor._format_context_for_compression(organized_data)
        )
        print(f"\nOriginal context size: {original_size} tokens")
        
        # Test compression
        compression_result = await compressor.compress_context(organized_data, target_tokens=15000)
        
        print(f"\nCompression Results:")
        print(f"  Original: {compression_result.original_size} tokens")
        print(f"  Compressed: {compression_result.compressed_size} tokens")
        print(f"  Ratio: {compression_result.compression_ratio:.3f}")
        print(f"  Level: {compression_result.compression_level.value}")
        print(f"  Chunks: {compression_result.chunks_processed}")
        print(f"  Preserved: {compression_result.preserved_elements}")
        
        # Test that compression preserves essential information
        compressed_content = compression_result.compressed_content
        
        # Check for preserved information
        has_protein_info = 'protein' in compressed_content.lower()
        has_function_info = any(term in compressed_content.lower() for term in ['k00000', 'enzyme', 'function'])
        has_genome_info = 'genome' in compressed_content.lower()
        
        print(f"\nInformation Preservation Check:")
        print(f"  Protein information preserved: {has_protein_info}")
        print(f"  Function information preserved: {has_function_info}")
        print(f"  Genome information preserved: {has_genome_info}")
        
        # Show preview of compressed content
        print(f"\nCompressed Content Preview:")
        preview_length = min(500, len(compressed_content))
        print(compressed_content[:preview_length])
        if len(compressed_content) > preview_length:
            print("...")
        
        print(f"\n✅ Integration test completed successfully!")
        
        # Test compression strategy determination
        strategies = []
        for size in [5000, 15000, 50000]:
            strategy, level = compressor.get_compression_strategy(size)
            strategies.append((size, strategy, level.value))
        
        print(f"\nCompression Strategy Tests:")
        for size, strategy, level in strategies:
            print(f"  {size} tokens -> {strategy} ({level})")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        rag.close()

if __name__ == "__main__":
    asyncio.run(test_rag_compression_integration())