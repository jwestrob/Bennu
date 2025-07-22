#!/usr/bin/env python3
"""
Test script for cached tool selection to demonstrate API call reduction.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.llm.rag_system.agent_tool_selector import get_cached_tool_selector
from src.llm.rag_system.task_management import Task, TaskType

async def test_cached_tool_selection():
    """Test cached tool selection with main tasks and sub-tasks."""
    
    cached_selector = get_cached_tool_selector()
    
    print("🧪 Testing Cached Tool Selection - API Call Reduction")
    print("=" * 60)
    
    # Simulate main task
    main_task = Task(
        task_id="main_task_1",
        task_type=TaskType.TOOL_CALL,
        description="Find prophage across all genomes using spatial analysis",
        is_main_task=True,
        parent_task_id=None
    )
    
    print(f"\n📋 MAIN TASK: {main_task.description}")
    print("Expected: Full LLM call")
    
    try:
        result = await cached_selector.select_tool_for_task_with_caching(
            task=main_task,
            original_user_query="find prophage across all genomes",
            previous_task_context=""
        )
        
        print(f"✅ Selected Tool: {result.selected_tool or 'database_query'}")
        print(f"🧠 Selection Source: {main_task.tool_selection_source}")
        print(f"🎯 Confidence: {result.confidence:.2f}")
        
        # Simulate 3 sub-tasks (chunks)
        sub_tasks = []
        for i in range(3):
            sub_task = Task(
                task_id=f"chunk_task_{i+1}",
                task_type=TaskType.TOOL_CALL,
                description=f"Process genome chunk {i+1} for prophage analysis",
                is_main_task=False,
                parent_task_id=main_task.task_id
            )
            sub_tasks.append(sub_task)
        
        print(f"\n📦 SUB-TASKS (Chunks): Processing {len(sub_tasks)} chunks")
        print("Expected: Inherit from main task (no LLM calls)")
        
        for i, sub_task in enumerate(sub_tasks, 1):
            print(f"\n   Chunk {i}: {sub_task.description}")
            
            result = await cached_selector.select_tool_for_task_with_caching(
                task=sub_task,
                original_user_query="find prophage across all genomes",
                previous_task_context=""
            )
            
            print(f"   ✅ Tool: {result.selected_tool or 'database_query'}")
            print(f"   📋 Source: {sub_task.tool_selection_source}")
            print(f"   🔗 Inherited from: {sub_task.parent_task_id}")
        
        # Simulate synthesis task
        synthesis_task = Task(
            task_id="synthesis_task_1",
            task_type=TaskType.SYNTHESIS,
            description="Synthesize prophage discovery results into comprehensive report",
            is_main_task=False,
            parent_task_id=main_task.task_id
        )
        
        print(f"\n🔄 SYNTHESIS TASK: {synthesis_task.description}")
        print("Expected: Conditional LLM call (may inherit or get new tool)")
        
        result = await cached_selector.select_tool_for_task_with_caching(
            task=synthesis_task,
            original_user_query="find prophage across all genomes",
            previous_task_context=""
        )
        
        print(f"✅ Selected Tool: {result.selected_tool or 'database_query'}")
        print(f"🧠 Selection Source: {synthesis_task.tool_selection_source}")
        
        # Show caching statistics
        stats = cached_selector.get_stats()
        print(f"\n📊 CACHING STATISTICS:")
        print(f"   Total Requests: {stats['total_requests']}")
        print(f"   LLM Calls: {stats['llm_calls']}")
        print(f"   Cache Hits: {stats['cache_hits']}")
        print(f"   Cache Hit Rate: {stats['cache_hit_rate_percent']:.1f}%")
        print(f"   🎯 API Call Reduction: {stats['api_call_reduction']}")
        
        print(f"\n🎉 Without caching: {stats['total_requests']} LLM calls")
        print(f"🚀 With caching: {stats['llm_calls']} LLM calls")
        reduction = stats['total_requests'] - stats['llm_calls']
        print(f"💰 Saved {reduction} API calls!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_cached_tool_selection())