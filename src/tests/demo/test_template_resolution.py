#!/usr/bin/env python3
"""
Test the template resolution fixes for task dependencies
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm.rag_system import GenomicRAG
from llm.config import LLMConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_template_resolution():
    """Test that template variables are properly resolved"""
    print("🧪 Testing Template Resolution...")
    
    config = LLMConfig()
    rag = GenomicRAG(config)
    
    # Test the _resolve_template_variables method directly
    print("\n📋 Testing template resolution patterns...")
    
    # Mock previous results with protein data
    mock_previous_results = {
        "query_central_metabolism_proteins": {
            "context": type('MockContext', (), {
                'structured_data': [
                    {'protein_id': 'protein_1', 'ko_id': 'K00001', 'ko_description': 'test enzyme 1'},
                    {'protein_id': 'protein_2', 'ko_id': 'K00002', 'ko_description': 'test enzyme 2'},
                    {'protein_id': 'protein_3', 'ko_id': 'K00003', 'ko_description': 'test enzyme 3'}
                ]
            })()
        }
    }
    
    # Test different template patterns
    test_cases = [
        {
            "name": "DSPy angle bracket pattern",
            "tool_args": {"protein_ids": "<ids_from_query_central_metabolism_proteins>"},
            "expected": ["protein_1", "protein_2", "protein_3"]
        },
        {
            "name": "Legacy 'from' pattern", 
            "tool_args": {"protein_ids": "from query_central_metabolism_proteins"},
            "expected": ["protein_1", "protein_2", "protein_3"]
        },
        {
            "name": "List with template",
            "tool_args": {"protein_ids": ["<ids_from_query_central_metabolism_proteins>"]},
            "expected": ["protein_1", "protein_2", "protein_3"]
        },
        {
            "name": "No template (should pass through)",
            "tool_args": {"protein_ids": ["manual_protein_1", "manual_protein_2"]},
            "expected": ["manual_protein_1", "manual_protein_2"]
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🔍 Testing: {test_case['name']}")
        print(f"  Input: {test_case['tool_args']}")
        
        resolved = rag._resolve_template_variables(test_case['tool_args'], mock_previous_results)
        
        print(f"  Output: {resolved}")
        print(f"  Expected: {test_case['expected']}")
        
        if resolved.get('protein_ids') == test_case['expected']:
            print(f"  ✅ PASS")
        else:
            print(f"  ❌ FAIL")
    
    # Test protein ID extraction
    print(f"\n📊 Testing protein ID extraction...")
    extracted_ids = rag._extract_protein_ids_from_task("query_central_metabolism_proteins", mock_previous_results)
    print(f"  Extracted IDs: {extracted_ids}")
    print(f"  Expected: ['protein_1', 'protein_2', 'protein_3']")
    
    if extracted_ids == ['protein_1', 'protein_2', 'protein_3']:
        print(f"  ✅ PASS")
    else:
        print(f"  ❌ FAIL")

async def test_with_real_query():
    """Test with a real query that should find proteins"""
    print("\n🧪 Testing with real database query...")
    
    config = LLMConfig()
    rag = GenomicRAG(config)
    
    # Use a query that should find proteins in our database
    query = "Find proteins with dehydrogenase function and show their sequences"
    
    try:
        print(f"📋 Query: {query}")
        result = await rag.ask(query)
        
        print(f"✅ Query completed")
        print(f"📊 Answer length: {len(result['answer'])} characters")
        print(f"🎯 Confidence: {result['confidence']}")
        
        # Check if we got actual protein data
        if "No relevant genomic context found" in result['answer']:
            print("❌ Still getting 'No relevant genomic context found'")
        else:
            print("✅ Got actual genomic context!")
            
    except Exception as e:
        print(f"❌ Query failed: {e}")

async def main():
    """Run all template resolution tests"""
    print("🚀 Starting Template Resolution Tests")
    print("=" * 60)
    
    try:
        await test_template_resolution()
        await test_with_real_query()
        
        print("\n" + "=" * 60)
        print("✅ Template resolution tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())