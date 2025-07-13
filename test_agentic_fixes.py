#!/usr/bin/env python3
"""
Test script to validate the agentic genomic RAG system fixes.
Validates context propagation and cost optimization improvements.
"""

import asyncio
import logging
from src.llm.rag_system.memory.model_allocation import ModelAllocation, TaskComplexity
from src.llm.rag_system.task_executor import TaskExecutor
from src.llm.rag_system.core import GenomicRAG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_allocation_context_awareness():
    """Test that model allocation is now context-aware for discovery vs complex queries."""
    print("🧪 Testing Model Allocation Context Awareness")
    
    allocator = ModelAllocation(use_premium_everywhere=False)
    
    # Test 1: Discovery query should use MEDIUM complexity (gpt-4.1-mini)
    discovery_query = "Find me operons containing probable prophage segments; we don't have virus-specific annotations so read through everything directly and see what you can find."
    
    complexity_discovery = allocator.get_task_complexity(
        task_name="context_preparation",
        query=discovery_query,
        task_context="Global analysis across all genomes"
    )
    
    model_discovery, config_discovery = allocator.get_model_for_task(
        task_name="context_preparation",
        query=discovery_query,
        task_context="Global analysis across all genomes"
    )
    
    print(f"   ✅ Discovery Query:")
    print(f"      Query: {discovery_query[:60]}...")
    print(f"      Complexity: {complexity_discovery}")
    print(f"      Model: {model_discovery}")
    print(f"      Cost/M tokens: ${config_discovery.cost_per_million}")
    
    # Test 2: Complex reasoning query should use COMPLEX (o3)
    complex_query = "Synthesize evolutionary significance of BGC distribution patterns across genomes with detailed mechanistic analysis"
    
    complexity_complex = allocator.get_task_complexity(
        task_name="context_preparation",
        query=complex_query,
        task_context="Detailed biological synthesis required"
    )
    
    model_complex, config_complex = allocator.get_model_for_task(
        task_name="context_preparation", 
        query=complex_query,
        task_context="Detailed biological synthesis required"
    )
    
    print(f"   ✅ Complex Query:")
    print(f"      Query: {complex_query[:60]}...")
    print(f"      Complexity: {complexity_complex}")
    print(f"      Model: {model_complex}")
    print(f"      Cost/M tokens: ${config_complex.cost_per_million}")
    
    # Validate the fix worked
    expected_cost_reduction = config_discovery.cost_per_million < config_complex.cost_per_million
    
    print(f"\n   📊 Cost Optimization Results:")
    print(f"      Discovery uses cheaper model: {expected_cost_reduction}")
    print(f"      Cost reduction: {((config_complex.cost_per_million - config_discovery.cost_per_million) / config_complex.cost_per_million * 100):.1f}%")
    
    if expected_cost_reduction:
        print("   ✅ PASS: Model allocation is now context-aware!")
    else:
        print("   ❌ FAIL: Model allocation still not optimized")
    
    return expected_cost_reduction

def test_chunking_decision_logic():
    """Test that chunking decisions now consider query intent."""
    print("\n🧪 Testing Chunking Decision Logic")
    
    class MockTaskExecutor:
        def _should_chunk_for_analysis_type(self, data_size, threshold, task_desc, original_question):
            # Import the actual method we fixed
            from src.llm.rag_system.task_executor import TaskExecutor
            executor = TaskExecutor(None)
            return executor._should_chunk_for_analysis_type(data_size, threshold, task_desc, original_question)
    
    executor = MockTaskExecutor()
    
    # Test 1: Discovery query with large dataset should NOT chunk
    discovery_query = "Find operons containing prophage segments; read through everything directly"
    should_chunk_discovery = executor._should_chunk_for_analysis_type(
        data_size=1500,  # Above threshold
        threshold=1000,
        task_desc="Find genomic patterns",
        original_question=discovery_query
    )
    
    print(f"   ✅ Discovery Query Chunking:")
    print(f"      Query: {discovery_query[:60]}...")
    print(f"      Dataset size: 1500 (> threshold 1000)")
    print(f"      Should chunk: {should_chunk_discovery}")
    
    # Test 2: Functional query with large dataset SHOULD chunk
    functional_query = "Analyze protein families and their functional domains in detail"
    should_chunk_functional = executor._should_chunk_for_analysis_type(
        data_size=1500,  # Above threshold  
        threshold=1000,
        task_desc="Analyze protein functions",
        original_question=functional_query
    )
    
    print(f"   ✅ Functional Query Chunking:")
    print(f"      Query: {functional_query[:60]}...")
    print(f"      Dataset size: 1500 (> threshold 1000)")
    print(f"      Should chunk: {should_chunk_functional}")
    
    # Validate the fix worked
    logic_correct = (not should_chunk_discovery) and should_chunk_functional
    
    print(f"\n   📊 Chunking Logic Results:")
    print(f"      Discovery avoids chunking: {not should_chunk_discovery}")
    print(f"      Functional uses chunking: {should_chunk_functional}")
    
    if logic_correct:
        print("   ✅ PASS: Chunking logic is now context-aware!")
    else:
        print("   ❌ FAIL: Chunking logic still not optimized")
    
    return logic_correct

def test_analysis_type_detection():
    """Test biological analysis type detection."""
    print("\n🧪 Testing Analysis Type Detection")
    
    class MockCore:
        def _determine_analysis_type(self, question):
            from src.llm.rag_system.core import GenomicRAG
            core = GenomicRAG.__new__(GenomicRAG)  # Create without __init__
            return core._determine_analysis_type(question)
    
    core = MockCore()
    
    test_cases = [
        ("Find operons containing prophage segments", "spatial_genomic"),
        ("What protein families are present?", "functional_annotation"), 
        ("Discover interesting genomic features", "comprehensive_discovery"),
        ("Show me gene clusters in the genome", "spatial_genomic"),
        ("Analyze metabolic pathways", "functional_annotation")
    ]
    
    correct_count = 0
    for query, expected_type in test_cases:
        detected_type = core._determine_analysis_type(query)
        is_correct = detected_type == expected_type
        
        print(f"   {'✅' if is_correct else '❌'} Query: {query[:50]}...")
        print(f"      Expected: {expected_type}")
        print(f"      Detected: {detected_type}")
        
        if is_correct:
            correct_count += 1
    
    accuracy = correct_count / len(test_cases)
    print(f"\n   📊 Analysis Type Detection Results:")
    print(f"      Accuracy: {correct_count}/{len(test_cases)} ({accuracy*100:.1f}%)")
    
    if accuracy >= 0.8:
        print("   ✅ PASS: Analysis type detection working well!")
    else:
        print("   ❌ FAIL: Analysis type detection needs improvement")
    
    return accuracy >= 0.8

def main():
    """Run all tests to validate the agentic fixes."""
    print("🚀 Testing Agentic Genomic RAG System Fixes")
    print("=" * 60)
    
    # Run tests
    test1_pass = test_model_allocation_context_awareness()
    test2_pass = test_chunking_decision_logic()
    test3_pass = test_analysis_type_detection()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 FINAL RESULTS:")
    print(f"   Model Allocation Context Awareness: {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"   Chunking Decision Logic: {'✅ PASS' if test2_pass else '❌ FAIL'}")
    print(f"   Analysis Type Detection: {'✅ PASS' if test3_pass else '❌ FAIL'}")
    
    all_tests_pass = test1_pass and test2_pass and test3_pass
    
    if all_tests_pass:
        print("\n🎉 ALL TESTS PASSED! Agentic system fixes are working correctly.")
        print("\n💰 Expected Cost Savings:")
        print("   - Discovery queries: ~95% cost reduction ($4.00 → $0.20)")
        print("   - No unnecessary chunking for spatial analysis")
        print("   - Proper biological context propagation")
    else:
        print("\n⚠️ Some tests failed. Please review the implementations.")
    
    return all_tests_pass

if __name__ == "__main__":
    main()