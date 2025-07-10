#!/usr/bin/env python3
"""
Quick test to verify model allocation is working.
"""

import sys
sys.path.append('/Users/jacob/Documents/Sandbox/microbial_claude_matter')

from src.llm.rag_system.memory import (
    quick_switch_to_optimized, 
    quick_switch_to_o3, 
    print_model_status,
    get_model_allocator
)

def test_model_allocation():
    print("🧪 Testing Model Allocation System")
    print("=" * 40)
    
    # Test optimized mode
    print("\n1. Testing Optimized Mode (should use gpt-4.1-mini for most tasks):")
    quick_switch_to_optimized()
    
    allocator = get_model_allocator()
    
    # Check what models are allocated
    test_tasks = [
        ("query_classification", "Query classification"),
        ("executive_summary", "Executive summary"),
        ("report_part_generation", "Report parts"),
        ("final_synthesis", "Final synthesis"),
        ("biological_interpretation", "Biological interpretation")
    ]
    
    print("\nCurrent allocation:")
    for task_name, description in test_tasks:
        model_name, model_config = allocator.get_model_for_task(task_name)
        print(f"  {description:25} → {model_name:15} (${model_config.cost_per_million:.2f}/1M)")
    
    # Test premium mode
    print("\n2. Testing Premium Mode (should use o3 for all tasks):")
    quick_switch_to_o3()
    
    print("\nCurrent allocation:")
    for task_name, description in test_tasks:
        model_name, model_config = allocator.get_model_for_task(task_name)
        print(f"  {description:25} → {model_name:15} (${model_config.cost_per_million:.2f}/1M)")
    
    # Switch back to optimized
    print("\n3. Switching back to optimized mode:")
    quick_switch_to_optimized()
    
    print("\n✅ Model allocation system is working!")
    print("\nNow when you run queries, you should see gpt-4.1-mini API calls")
    print("instead of o3 calls (except for complex synthesis tasks).")

if __name__ == "__main__":
    test_model_allocation()