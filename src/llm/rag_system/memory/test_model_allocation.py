"""
Test script for model allocation system.

Demonstrates how to use the model allocation system for cost-optimized
multi-part report generation.
"""

import logging
from typing import Dict, Any

from .model_config import (
    quick_switch_to_optimized,
    quick_switch_to_o3,
    quick_switch_to_testing,
    print_model_status,
    get_current_mode
)
from .model_allocation import get_model_allocator

logger = logging.getLogger(__name__)


def test_model_allocation():
    """Test the model allocation system."""
    print("🧪 Testing Model Allocation System")
    print("=" * 50)
    
    # Test 1: Default optimized mode
    print("\n1. Testing Default Optimized Mode:")
    quick_switch_to_optimized()
    
    # Test 2: Premium mode
    print("\n2. Testing Premium Mode:")
    quick_switch_to_o3()
    
    # Test 3: Testing mode
    print("\n3. Testing Testing Mode:")
    quick_switch_to_testing()
    
    # Test 4: Model allocation for specific tasks
    print("\n4. Testing Task-Specific Model Allocation:")
    quick_switch_to_optimized()
    
    allocator = get_model_allocator()
    
    # Test different task types
    test_tasks = [
        ("report_type_classification", "Simple classification"),
        ("executive_summary", "Executive summary generation"),
        ("final_synthesis", "Complex synthesis"),
        ("report_part_generation", "Report part generation"),
        ("biological_interpretation", "Biological interpretation")
    ]
    
    print("\nTask-specific model allocation:")
    for task_name, description in test_tasks:
        model_name, model_config = allocator.get_model_for_task(task_name)
        cost_per_1k = model_config.cost_per_million / 1000
        print(f"  {description:25} → {model_name:15} (${cost_per_1k:.3f}/1K tokens)")
    
    # Test 5: Cost estimation
    print("\n5. Testing Cost Estimation:")
    sample_token_counts = [1000, 5000, 10000, 50000]
    
    for task_name, description in test_tasks[:3]:  # Test first 3 tasks
        print(f"\n  {description}:")
        for tokens in sample_token_counts:
            cost = allocator.get_cost_estimate(task_name, tokens)
            print(f"    {tokens:5} tokens → ${cost:.4f}")
    
    # Test 6: Allocation summary
    print("\n6. Current Allocation Summary:")
    summary = allocator.get_allocation_summary()
    print(f"  Mode: {summary['mode']}")
    
    if summary['mode'] == 'optimized':
        print(f"  Cost savings: {summary['cost_savings_vs_premium']}")
        print(f"  Average cost: ${summary['average_cost_per_million']:.2f}/1M tokens")
    
    print("\n✅ Model allocation system test completed!")


def demo_switching_for_development():
    """Demonstrate model switching for development workflow."""
    print("\n🔄 Development Workflow Demo")
    print("=" * 40)
    
    print("\n📝 Scenario: Developing multi-part reports")
    
    # Step 1: Development phase
    print("\n1. Development Phase - Use testing mode:")
    quick_switch_to_testing()
    print("   → Fast iteration with cheap models")
    
    # Step 2: Quality testing
    print("\n2. Quality Testing - Use optimized mode:")
    quick_switch_to_optimized()
    print("   → Better quality with reasonable costs")
    
    # Step 3: Final production
    print("\n3. Production Run - Use premium mode:")
    quick_switch_to_o3()
    print("   → Maximum quality for final results")
    
    print("\n✅ Development workflow demo completed!")


def estimate_cost_comparison():
    """Estimate cost comparison between modes."""
    print("\n💰 Cost Comparison Analysis")
    print("=" * 35)
    
    # Simulate a large multi-part report
    estimated_tokens = {
        "report_type_classification": 500,
        "executive_summary": 3000,
        "report_part_generation": 25000,  # 5 parts × 5000 tokens each
        "final_synthesis": 8000,
        "biological_interpretation": 5000
    }
    
    # Calculate costs in optimized mode
    quick_switch_to_optimized()
    allocator = get_model_allocator()
    
    optimized_cost = 0
    print("\nOptimized Mode Costs:")
    for task_name, tokens in estimated_tokens.items():
        cost = allocator.get_cost_estimate(task_name, tokens)
        optimized_cost += cost
        print(f"  {task_name:25} → ${cost:.4f}")
    
    print(f"\nTotal Optimized Cost: ${optimized_cost:.4f}")
    
    # Calculate costs in premium mode
    quick_switch_to_o3()
    
    premium_cost = 0
    print("\nPremium Mode Costs:")
    for task_name, tokens in estimated_tokens.items():
        cost = allocator.get_cost_estimate(task_name, tokens)
        premium_cost += cost
        print(f"  {task_name:25} → ${cost:.4f}")
    
    print(f"\nTotal Premium Cost: ${premium_cost:.4f}")
    
    # Compare
    savings = (premium_cost - optimized_cost) / premium_cost * 100
    print(f"\n📊 Cost Comparison:")
    print(f"  Optimized Mode: ${optimized_cost:.4f}")
    print(f"  Premium Mode:   ${premium_cost:.4f}")
    print(f"  Savings:        ${premium_cost - optimized_cost:.4f} ({savings:.1f}%)")
    
    # Reset to optimized
    quick_switch_to_optimized()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    test_model_allocation()
    demo_switching_for_development()
    estimate_cost_comparison()
    
    print("\n🎉 All tests completed successfully!")
    print("\nTo use in your code:")
    print("  from src.llm.rag_system.memory import quick_switch_to_optimized, quick_switch_to_o3")
    print("  quick_switch_to_optimized()  # For cost-effective analysis")
    print("  quick_switch_to_o3()         # For maximum quality")