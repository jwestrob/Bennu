#!/usr/bin/env python3
"""
Debug Protein ID Extraction in Code Interpreter Enhancement

This script helps us understand what protein IDs are being extracted
from the context for the code interpreter.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm.rag_system import GenomicRAG, GenomicContext
from llm.config import LLMConfig

def debug_protein_extraction():
    """Debug what protein IDs are extracted from context."""
    print("🔍 Debugging Protein ID Extraction")
    print("=" * 50)
    
    # Create mock context that simulates what we get from the transport protein query
    mock_structured_data = [
        {
            'protein_id': 'RIFCSPHIGHO2_01_FULL_Acidovorax_64_960_rifcsphigho2_01_scaffold_3030_5',
            'ko_id': 'K02115',
            'ko_description': 'F-type H+-transporting ATPase subunit gamma',
            'start_coordinate': '3090',
            'end_coordinate': '3956',
            'strand': '-1',
            'pfam_accessions': ['PF00231.24']
        },
        {
            'protein_id': 'RIFCSPLOWO2_01_FULL_OD1_41_220_rifcsplowo2_01_scaffold_2623_17',
            'ko_id': 'K02115', 
            'ko_description': 'F-type H+-transporting ATPase subunit gamma',
            'start_coordinate': '11392',
            'end_coordinate': '12318',
            'strand': '1',
            'pfam_accessions': ['PF00231.24']
        }
    ]
    
    # Simulate what context formatting does (clean the protein IDs)
    for item in mock_structured_data:
        original_id = item['protein_id']
        clean_id = original_id.replace('protein:', '') if original_id.startswith('protein:') else original_id
        item['protein_id'] = clean_id
        print(f"Original: {original_id}")
        print(f"Clean:    {clean_id}")
        print()
    
    mock_context = GenomicContext(
        structured_data=mock_structured_data,
        semantic_data=[],
        metadata={},
        query_time=0.1
    )
    
    # Mock previous results structure
    mock_previous_results = {
        'get_transport_proteins': {
            'context': mock_context
        }
    }
    
    # Initialize RAG system and test protein extraction
    config = LLMConfig()
    rag = GenomicRAG(config)
    
    print("🧪 Testing protein ID extraction...")
    test_code = "print('Testing sequence analysis')"
    enhanced_code = rag._enhance_code_interpreter(test_code, mock_previous_results)
    
    print("📋 Enhanced Code:")
    print("=" * 30)
    lines = enhanced_code.split('\n')
    
    # Look for the protein_ids line specifically
    for i, line in enumerate(lines):
        print(f"{i+1:2d}: {line}")
        if 'protein_ids =' in line:
            print(f"    ^^^ FOUND PROTEIN IDS LINE!")
    
    print("\n🔍 Analysis:")
    if 'RIFCSPHIGHO2_01_FULL_Acidovorax_64_960_rifcsphigho2_01_scaffold_3030_5' in enhanced_code:
        print("✅ First protein ID found in enhanced code")
    else:
        print("❌ First protein ID NOT found in enhanced code")
        
    if 'RIFCSPLOWO2_01_FULL_OD1_41_220_rifcsplowo2_01_scaffold_2623_17' in enhanced_code:
        print("✅ Second protein ID found in enhanced code")
    else:
        print("❌ Second protein ID NOT found in enhanced code")

if __name__ == "__main__":
    debug_protein_extraction()