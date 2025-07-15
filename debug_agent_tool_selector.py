#!/usr/bin/env python3
"""
Debug script for the agent-based tool selector issue.
"""

import asyncio
import sys
import os
import logging

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

# Limit to our specific modules
logging.getLogger('src.llm.rag_system.agent_tool_selector').setLevel(logging.DEBUG)
logging.getLogger('src.llm.rag_system.task_plan_parser').setLevel(logging.DEBUG)

# Reduce noise from other loggers
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('LiteLLM').setLevel(logging.WARNING)

async def test_single_task():
    """Test a single task to see where it fails."""
    
    print("🐛 Debug: Testing Single Task Classification")
    print("=" * 50)
    
    from llm.rag_system.task_plan_parser import TaskPlanParser
    
    # Original user query that should trigger spatial analysis
    original_query = """Find me operons containing probable prophage segments; we don't have virus-specific 
    annotations so read through everything directly and see what you can find."""
    
    # Test with detailed report request that should trigger multipart synthesis
    task_description = "Find operons containing prophage segments and give me a detailed report"
    
    print(f"\n🎯 Testing Task: {task_description}")
    print(f"📝 Original Query: {original_query[:100]}...")
    
    # Test through task plan parser
    parser = TaskPlanParser()
    
    try:
        print(f"\n🔄 Calling _classify_task_type_with_args...")
        task_type, tool_name, tool_args = parser._classify_task_type_with_args(task_description)
        
        print(f"\n✅ Result:")
        print(f"  📋 Task Type: {task_type}")
        print(f"  🛠️ Tool Name: {tool_name}")
        print(f"  ⚙️ Tool Args: {tool_args}")
        
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_single_task())