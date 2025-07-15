#!/usr/bin/env python3
"""
Test script for the agent-based tool selector.

This script tests whether the new agent-based tool selection correctly routes
prophage discovery tasks to the whole_genome_reader tool instead of database queries.
"""

import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm.rag_system.task_plan_parser import TaskPlanParser
from llm.rag_system.agent_tool_selector import get_tool_selector

async def test_prophage_discovery_routing():
    """Test that prophage discovery tasks are correctly routed to whole_genome_reader."""
    
    print("🧪 Testing Agent-Based Tool Selection for Prophage Discovery")
    print("=" * 60)
    
    # Original user query that should trigger spatial analysis
    original_query = """Find me operons containing probable prophage segments; we don't have virus-specific 
    annotations so read through everything directly and see what you can find. Large stretches of 
    unannotated/hypothetical proteins are good signals for phage! At the end, make a report on at least 
    five loci most likely to be phage, based on their novelty."""
    
    # Task descriptions that previously failed (from session 0b350a2a-82ed-47eb-8bae-b7f80cb9a21c)
    problematic_tasks = [
        "Retrieve full genome annotation table",
        "Define operons by clustering neighboring genes", 
        "For each operon, score phage likelihood",
        "Compute novelty of each operon (e.g., BLAST analysis)",
        "Rank operons by combined phage-likelihood and novelty",
        "Select top ≥5 operons",
        "For each selected locus gather details",
        "Draft a narrative report summarizing methodology"
    ]
    
    # Test direct tool selector
    tool_selector = get_tool_selector()
    
    print("\n🤖 Testing Direct Agent Tool Selection:")
    print("-" * 40)
    
    for i, task_desc in enumerate(problematic_tasks, 1):
        print(f"\nTask {i}: {task_desc}")
        
        try:
            result = await tool_selector.select_tool_for_task(
                task_description=task_desc,
                original_user_query=original_query,
                previous_task_context=""
            )
            
            print(f"  📤 Selected Tool: {result.selected_tool or 'database_query'}")
            print(f"  🎯 Confidence: {result.confidence:.2f}")
            print(f"  💭 Reasoning: {result.reasoning[:100]}...")
            
            # Check if spatial analysis tasks are correctly routed
            if any(keyword in task_desc.lower() for keyword in ['operon', 'clustering', 'phage', 'spatial']):
                if result.selected_tool == 'whole_genome_reader':
                    print(f"  ✅ CORRECT: Spatial analysis task routed to whole_genome_reader")
                else:
                    print(f"  ❌ WRONG: Spatial analysis task routed to {result.selected_tool or 'database_query'}")
            
        except Exception as e:
            print(f"  💥 Error: {e}")
    
    # Test through task plan parser
    print("\n📋 Testing Task Plan Parser Integration:")
    print("-" * 40)
    
    parser = TaskPlanParser()
    
    # Create a sample DSPy plan
    sample_plan = """
    1. Retrieve full genome annotation table for spatial analysis
    2. Define operons by clustering neighboring genes  
    3. For each operon, score phage likelihood based on hypothetical proteins
    4. Compute novelty of each operon using BLAST analysis
    5. Rank operons by combined phage-likelihood and novelty scores
    6. Select top 5 operons most likely to contain prophage segments
    7. For each selected locus gather detailed genomic context
    8. Draft a narrative report summarizing prophage discovery methodology and results
    """
    
    try:
        parsed_plan = parser.parse_dspy_plan(sample_plan, original_query)
        
        print(f"\n📊 Plan Parsing Results:")
        print(f"  Success: {parsed_plan.parsing_success}")
        print(f"  Tasks Created: {len(parsed_plan.tasks)}")
        print(f"  Errors: {len(parsed_plan.errors)}")
        
        if parsed_plan.errors:
            print(f"  Error Details: {parsed_plan.errors}")
        
        print(f"\n📝 Task Classifications:")
        tool_call_count = 0
        atomic_query_count = 0
        
        for task in parsed_plan.tasks:
            task_type_symbol = "🔧" if task.task_type.value == "tool_call" else "🗄️"
            tool_name = task.tool_name if hasattr(task, 'tool_name') and task.tool_name else "database_query"
            
            print(f"  {task_type_symbol} {task.task_id[:30]}... → {tool_name}")
            
            if task.task_type.value == "tool_call":
                tool_call_count += 1
                
                # Check if spatial analysis tasks are using whole_genome_reader
                if any(keyword in task.description.lower() for keyword in ['operon', 'clustering', 'phage', 'spatial']):
                    if task.tool_name == 'whole_genome_reader':
                        print(f"    ✅ Correct routing for spatial analysis")
                    else:
                        print(f"    ❌ Wrong routing: should use whole_genome_reader")
            else:
                atomic_query_count += 1
        
        print(f"\n📈 Summary:")
        print(f"  🔧 Tool Calls: {tool_call_count}")
        print(f"  🗄️ Database Queries: {atomic_query_count}")
        
        # Ideal outcome: spatial analysis tasks should be tool calls to whole_genome_reader
        if tool_call_count >= 4:  # At least half should be tool calls for this prophage workflow
            print(f"  ✅ GOOD: {tool_call_count} tasks correctly identified as tool calls")
        else:
            print(f"  ⚠️ CONCERN: Only {tool_call_count} tool calls, may need tuning")
            
    except Exception as e:
        print(f"💥 Task plan parsing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_prophage_discovery_routing())