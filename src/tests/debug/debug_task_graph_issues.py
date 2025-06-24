#!/usr/bin/env python3
"""
Quick debug script to test task graph issues and agentic planning fixes.
Focuses on the specific JSON parsing and planning validation problems.

Usage: python -m src.tests.debug.debug_task_graph_issues
"""

import asyncio
import os
from rich.console import Console
from rich.panel import Panel

from src.llm.config import LLMConfig
from src.llm.rag_system import GenomicRAG

console = Console()

async def test_agentic_planning_fixes():
    """Test the fixes for agentic planning issues."""
    console.print(Panel.fit(
        "[bold cyan]🔧 TASK GRAPH & AGENTIC PLANNING FIXES TEST[/bold cyan]\n"
        "Testing JSON parsing and planning validation improvements",
        title="Debug Task Graph Issues"
    ))
    
    if not os.getenv('OPENAI_API_KEY'):
        console.print("[red]❌ Please set OPENAI_API_KEY environment variable[/red]")
        return
    
    try:
        config = LLMConfig.from_env()
        rag = GenomicRAG(config)
        
        # Test cases designed to trigger different planning scenarios
        test_cases = [
            {
                "query": "How many proteins are in the database?",
                "description": "Simple query - should use traditional mode",
                "expected_mode": "traditional"
            },
            {
                "query": "What does recent literature say about CRISPR proteins?",
                "description": "Literature search - should use agentic mode with proper JSON",
                "expected_mode": "agentic"
            },
            {
                "query": "Find proteins similar to heme transporters and search for recent papers about their function",
                "description": "Complex multi-step - should use agentic mode",
                "expected_mode": "agentic"
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            console.print(f"\n[bold blue]TEST {i}/3: {test_case['description']}[/bold blue]")
            console.print(f"Query: {test_case['query']}")
            console.print("─" * 60)
            
            try:
                response = await rag.ask(test_case['query'])
                
                actual_mode = response.get('query_metadata', {}).get('execution_mode', 'unknown')
                expected_mode = test_case['expected_mode']
                
                success = actual_mode == expected_mode
                status = "✅" if success else "⚠️"
                
                console.print(f"{status} [bold]Result:[/bold] {actual_mode} mode (expected: {expected_mode})")
                
                if actual_mode == 'agentic':
                    metadata = response.get('query_metadata', {})
                    console.print(f"   Tasks completed: {metadata.get('tasks_completed', 0)}")
                    console.print(f"   Tasks failed: {metadata.get('tasks_failed', 0)}")
                    console.print(f"   Iterations: {metadata.get('total_iterations', 0)}")
                
                console.print(f"   Answer length: {len(response.get('answer', ''))} characters")
                console.print(f"   Confidence: {response.get('confidence', 'unknown')}")
                
                results.append({
                    'test': test_case['description'],
                    'success': success,
                    'actual_mode': actual_mode,
                    'expected_mode': expected_mode,
                    'answer_length': len(response.get('answer', '')),
                    'confidence': response.get('confidence', 'unknown')
                })
                
            except Exception as e:
                console.print(f"❌ [red]ERROR: {str(e)}[/red]")
                results.append({
                    'test': test_case['description'],
                    'success': False,
                    'error': str(e)
                })
        
        # Summary
        console.print(f"\n[bold green]📊 SUMMARY[/bold green]")
        console.print("─" * 40)
        
        successful_tests = sum(1 for r in results if r.get('success', False))
        total_tests = len(results)
        
        console.print(f"Successful tests: {successful_tests}/{total_tests}")
        
        traditional_mode_count = sum(1 for r in results if r.get('actual_mode') == 'traditional')
        agentic_mode_count = sum(1 for r in results if r.get('actual_mode') == 'agentic')
        
        console.print(f"Traditional mode: {traditional_mode_count} queries")
        console.print(f"Agentic mode: {agentic_mode_count} queries")
        
        high_confidence = sum(1 for r in results if r.get('confidence') == 'high')
        console.print(f"High confidence answers: {high_confidence}/{total_tests}")
        
        if successful_tests == total_tests:
            console.print(f"\n[bold green]🎉 ALL FIXES WORKING CORRECTLY![/bold green]")
            console.print("✅ Task graph construction is functional")
            console.print("✅ Agentic planning routing is working")
            console.print("✅ JSON parsing and validation fixes applied successfully")
        elif successful_tests >= total_tests * 0.7:
            console.print(f"\n[bold yellow]⚠️ MOSTLY WORKING ({successful_tests}/{total_tests})[/bold yellow]")
            console.print("✅ Core functionality is operational")
            console.print("⚠️ Some edge cases may need attention")
        else:
            console.print(f"\n[bold red]❌ ISSUES REMAINING ({successful_tests}/{total_tests})[/bold red]")
            console.print("❌ Task graph issues persist")
            console.print("❌ Further debugging needed")
        
        console.print(f"\n[bold cyan]DETAILED RESULTS:[/bold cyan]")
        for result in results:
            status = "✅" if result.get('success', False) else "❌"
            console.print(f"{status} {result['test']}")
            if 'actual_mode' in result:
                console.print(f"   Mode: {result['actual_mode']} (expected: {result['expected_mode']})")
                console.print(f"   Confidence: {result['confidence']}, Length: {result['answer_length']} chars")
            if 'error' in result:
                console.print(f"   Error: {result['error']}")
        
        rag.close()
        
    except Exception as e:
        console.print(f"[red]FATAL ERROR: {str(e)}[/red]")

if __name__ == "__main__":
    asyncio.run(test_agentic_planning_fixes())