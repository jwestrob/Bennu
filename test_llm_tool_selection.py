#!/usr/bin/env python3
"""
Test script for the new LLM-first tool selection architecture.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.llm.rag_system.agent_tool_selector import get_tool_selector

async def test_prophage_discovery():
    """Test that prophage discovery queries select whole_genome_reader with global analysis."""
    
    tool_selector = get_tool_selector()
    
    test_cases = [
        {
            "query": "find prophage across all genomes",
            "task": "Discover prophage segments in the complete genomic dataset",
            "expected_tool": "whole_genome_reader",
            "expected_global": True
        },
        {
            "query": "how many transport proteins are there?",
            "task": "Count transport proteins in database",
            "expected_tool": None,  # database_query
            "expected_global": False
        },
        {
            "query": "explore operons containing hypothetical proteins",
            "task": "Identify operon structures with hypothetical protein clusters",
            "expected_tool": "whole_genome_reader", 
            "expected_global": True
        }
    ]
    
    print("🧪 Testing LLM-First Tool Selection")
    print("=" * 50)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {case['query']}")
        print(f"Task: {case['task']}")
        
        try:
            result = await tool_selector.select_tool_for_task(
                task_description=case['task'],
                original_user_query=case['query'],
                previous_task_context=""
            )
            
            print(f"✅ Selected Tool: {result.selected_tool or 'database_query'}")
            print(f"🧠 Reasoning: {result.reasoning[:200]}...")
            print(f"🎯 Confidence: {result.confidence:.2f}")
            
            # Check tool selection
            if case['expected_tool'] is None:
                assert result.selected_tool is None, f"Expected database_query, got {result.selected_tool}"
                print("✅ Correctly selected database_query")
            else:
                assert result.selected_tool == case['expected_tool'], f"Expected {case['expected_tool']}, got {result.selected_tool}"
                print(f"✅ Correctly selected {case['expected_tool']}")
            
            # Check global analysis parameter
            if case['expected_tool'] == "whole_genome_reader":
                global_analysis = result.tool_arguments.get('global_analysis', False)
                if case['expected_global']:
                    assert global_analysis, "Expected global_analysis=True"
                    print("✅ Correctly detected global analysis requirement")
                else:
                    assert not global_analysis, "Expected global_analysis=False"
                    print("✅ Correctly detected single-genome analysis")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    asyncio.run(test_prophage_discovery())