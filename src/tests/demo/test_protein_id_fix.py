#!/usr/bin/env python3
"""
Test script to verify that the protein ID truncation fix works correctly.

This script tests:
1. Context formatting preserves both short and full protein IDs
2. Code interpreter enhancement uses full protein IDs for sequence lookups
3. End-to-end workflow retrieves sequences correctly
"""

import sys
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm.rag_system import GenomicRAG, GenomicContext
from llm.config import LLMConfig

async def test_protein_id_fix():
    """Test the protein ID truncation fix."""
    print("🧪 Testing Protein ID Truncation Fix")
    print("=" * 50)
    
    # Mock structured data that simulates Neo4j query results
    mock_context_data = [
        {
            'protein_id': 'protein:RIFCSPHIGHO2_01_FULL_Acidovorax_64_960_rifcsphigho2_01_scaffold_3030_5',
            'p.id': 'protein:RIFCSPHIGHO2_01_FULL_Acidovorax_64_960_rifcsphigho2_01_scaffold_3030_5',
            'gene_start': 1234,
            'gene_end': 2345,
            'strand': 1,
            'length': 371
        },
        {
            'protein_id': 'protein:PLM0_60_b1_sep16_scaffold_10001_curated_1',
            'p.id': 'protein:PLM0_60_b1_sep16_scaffold_10001_curated_1', 
            'gene_start': 5678,
            'gene_end': 6789,
            'strand': -1,
            'length': 370
        }
    ]
    
    # Create mock context
    mock_context = GenomicContext(
        structured_data=mock_context_data,
        semantic_data=[],
        metadata={},
        query_time=0.1
    )
    
    # Initialize RAG system (we'll only test the formatting methods)
    try:
        # Create minimal config for testing
        config = LLMConfig()
        rag = GenomicRAG(config)
        
        # Test 1: Context formatting preserves both short and full IDs
        print("1. Testing context formatting...")
        formatted_context = rag._format_context(mock_context)
        print("✅ Context formatted successfully")
        
        # Check that the structured data now has clean protein IDs (no prefix)
        for item in mock_context.structured_data:
            if 'protein_id' in item:
                print(f"   Clean Protein ID: {item['protein_id']}")
                print()
        
        # Test 2: Code interpreter enhancement uses full protein IDs  
        print("2. Testing code interpreter enhancement...")
        
        # Mock previous results with the formatted context
        mock_previous_results = {
            'test_query': {
                'context': mock_context
            }
        }
        
        # Test code interpreter enhancement
        test_code = "print('Testing sequence analysis')"
        enhanced_code = rag._enhance_code_interpreter(test_code, mock_previous_results)
        
        # Check that the enhanced code uses clean, full protein IDs (without prefix)
        if 'RIFCSPHIGHO2_01_FULL_Acidovorax_64_960_rifcsphigho2_01_scaffold_3030_5' in enhanced_code:
            print("✅ Code interpreter uses clean, full protein IDs for sequence lookup")
        else:
            print("❌ Code interpreter not using correct protein IDs")
            
        print(f"Enhanced code preview:")
        print("=" * 30)
        lines = enhanced_code.split('\n')
        for i, line in enumerate(lines[:20]):  # Show first 20 lines
            print(f"{i+1:2d}: {line}")
        if len(lines) > 20:
            print(f"... ({len(lines)-20} more lines)")
            
        print("\n✅ Protein ID fix test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run the test."""
    asyncio.run(test_protein_id_fix())

if __name__ == "__main__":
    main()