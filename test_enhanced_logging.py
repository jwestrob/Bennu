#!/usr/bin/env python3
"""
Test script for enhanced task graph logging and clean output formatting.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.llm.rag_system.log_formatter import enable_clean_logging, export_task_summary
from src.llm.rag_system.task_management import TaskGraph, Task, TaskType, TaskStatus
from src.llm.rag_system.agent_tool_selector import get_cached_tool_selector

async def test_enhanced_logging():
    """Test the enhanced logging system with a realistic task workflow."""
    
    # Enable clean logging
    logger = enable_clean_logging()
    
    print("\n" + "="*80)
    print("🧪 TESTING ENHANCED TASK GRAPH LOGGING")
    print("="*80)
    
    # Create TaskGraph with user query
    user_query = "Find prophage across all genomes and analyze their distribution"
    graph = TaskGraph(user_query=user_query)
    
    # Simulate pipeline phases
    graph.set_phase("Planning & Tool Selection")
    
    # Create main task
    main_task = Task(
        task_id="main_prophage_discovery",
        task_type=TaskType.TOOL_CALL,
        description="Find prophage across all genomes using spatial analysis",
        tool_name="whole_genome_reader",
        tool_args={"global_analysis": True, "focus_on_spatial": True},
        is_main_task=True,
        tool_selection_source="planned"
    )
    
    graph.add_task(main_task, source="dspy_planning")
    
    # Create sub-tasks (chunks)
    chunk_tasks = []
    for i in range(3):
        chunk_task = Task(
            task_id=f"chunk_analysis_{i+1}",
            task_type=TaskType.TOOL_CALL,
            description=f"Analyze genome chunk {i+1} for prophage segments using inherited spatial analysis",
            tool_name="whole_genome_reader",
            tool_args={"chunk_id": i+1},
            is_main_task=False,
            parent_task_id=main_task.task_id,
            tool_selection_source="inherited"
        )
        chunk_tasks.append(chunk_task)
        graph.add_task(chunk_task, source="intelligent_chunking")
    
    # Create synthesis task
    synthesis_task = Task(
        task_id="synthesis_prophage_report",
        task_type=TaskType.SYNTHESIS,
        description="Synthesize prophage discovery results into comprehensive distribution analysis",
        is_main_task=False,
        parent_task_id=main_task.task_id,
        tool_selection_source="synthesized"
    )
    
    graph.add_task(synthesis_task, source="synthesis_planning")
    
    # Simulate execution phase
    graph.set_phase("Task Execution")
    
    # Simulate task execution with timing
    import time
    
    # Execute main task
    graph.mark_task_started(main_task.task_id)
    await asyncio.sleep(0.1)  # Simulate work
    graph.mark_task_completed(
        main_task.task_id, 
        result={"prophage_candidates": 15, "genomes_analyzed": 5},
        result_summary="Found 15 prophage candidates across 5 genomes"
    )
    
    # Execute chunk tasks
    for i, chunk_task in enumerate(chunk_tasks):
        graph.mark_task_started(chunk_task.task_id)
        await asyncio.sleep(0.05)  # Simulate work
        graph.mark_task_completed(
            chunk_task.task_id,
            result={"chunk_prophages": 3 + i, "chunk_size": 500},
            result_summary=f"Chunk {i+1}: Found {3+i} prophage segments in 500-gene chunk"
        )
    
    # Execute synthesis task
    graph.mark_task_started(synthesis_task.task_id)
    await asyncio.sleep(0.08)  # Simulate work
    graph.mark_task_completed(
        synthesis_task.task_id,
        result={"final_report": "Comprehensive prophage analysis complete"},
        result_summary="Generated comprehensive prophage distribution report"
    )
    
    # Final summary phase
    graph.set_phase("Results & Summary")
    execution_summary = graph.get_execution_summary()
    
    # Export logs and summary
    log_file = graph.export_log()
    summary_file = export_task_summary(graph)
    
    print(f"\n📄 Files Generated:")
    print(f"   Execution Log: {log_file}")
    print(f"   Task Summary: {summary_file}")
    
    print(f"\n🎯 Test completed successfully!")
    return graph

async def test_cached_tool_selection_logging():
    """Test the cached tool selection with enhanced logging."""
    
    print(f"\n{'='*80}")
    print("🧪 TESTING CACHED TOOL SELECTION LOGGING")
    print("="*80)
    
    cached_selector = get_cached_tool_selector()
    
    # Create main task
    main_task = Task(
        task_id="main_global_analysis",
        task_type=TaskType.TOOL_CALL,
        description="Perform global spatial analysis to discover prophage patterns",
        is_main_task=True
    )
    
    print("\n🎯 Testing Main Task Tool Selection (Expected: LLM call)")
    result = await cached_selector.select_tool_for_task_with_caching(
        task=main_task,
        original_user_query="find prophage across all genomes",
        previous_task_context=""
    )
    
    print(f"Selected Tool: {result.selected_tool or 'database_query'}")
    print(f"Selection Source: {main_task.tool_selection_source}")
    
    # Create sub-tasks
    sub_tasks = []
    for i in range(3):
        sub_task = Task(
            task_id=f"sub_analysis_{i+1}",
            task_type=TaskType.TOOL_CALL,
            description=f"Process sub-analysis {i+1} using inherited tool selection",
            is_main_task=False,
            parent_task_id=main_task.task_id
        )
        sub_tasks.append(sub_task)
    
    print(f"\n📦 Testing Sub-Task Tool Selection (Expected: Inheritance)")
    for sub_task in sub_tasks:
        result = await cached_selector.select_tool_for_task_with_caching(
            task=sub_task,
            original_user_query="find prophage across all genomes",
            previous_task_context=""
        )
        print(f"   {sub_task.task_id}: {result.selected_tool or 'database_query'} (source: {sub_task.tool_selection_source})")
    
    # Show caching statistics
    stats = cached_selector.get_stats()
    print(f"\n📊 Caching Statistics:")
    print(f"   Total Requests: {stats['total_requests']}")
    print(f"   LLM Calls: {stats['llm_calls']}")
    print(f"   Cache Hits: {stats['cache_hits']}")
    print(f"   🚀 {stats['api_call_reduction']}")

if __name__ == "__main__":
    async def main():
        # Test enhanced logging
        graph = await test_enhanced_logging()
        
        # Test cached tool selection logging
        await test_cached_tool_selection_logging()
        
        print(f"\n✨ All logging tests completed successfully!")
    
    asyncio.run(main())