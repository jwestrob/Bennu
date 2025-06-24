#!/usr/bin/env python3
"""
Debug script to test agentic RAG integration and examine LLM context.
Shows exactly what the LLM sees and how agentic planning affects responses.

Usage: python -m src.tests.debug.debug_agentic_integration
"""

import asyncio
import os
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
import json

from src.llm.config import LLMConfig
from src.llm.rag_system import GenomicRAG

console = Console()

async def debug_query_with_context(rag: GenomicRAG, query: str, description: str):
    """Debug a query and show the exact context the LLM sees."""
    console.print(f"\n[bold blue]🧬 {description}[/bold blue]")
    console.print(f"[dim]Query: {query}[/dim]")
    console.print("="*80)
    
    try:
        # Patch the answerer to capture context before it's processed
        original_answerer = rag.answerer
        captured_context = None
        
        def capture_context_answerer(question, context):
            nonlocal captured_context
            captured_context = context
            return original_answerer(question=question, context=context)
        
        rag.answerer = capture_context_answerer
        
        # Execute the query
        response = await rag.ask(query)
        
        # Show planning decision
        console.print(f"[yellow]Execution Mode:[/yellow] {response.get('query_metadata', {}).get('execution_mode', 'unknown')}")
        
        if 'query_metadata' in response:
            metadata = response['query_metadata']
            if metadata.get('execution_mode') == 'agentic':
                console.print(f"[cyan]Tasks Completed:[/cyan] {metadata.get('tasks_completed', 0)}")
                console.print(f"[cyan]Tasks Failed:[/cyan] {metadata.get('tasks_failed', 0)}")
                console.print(f"[cyan]Total Iterations:[/cyan] {metadata.get('total_iterations', 0)}")
            else:
                console.print(f"[cyan]Query Type:[/cyan] {metadata.get('query_type', 'unknown')}")
                console.print(f"[cyan]Search Strategy:[/cyan] {metadata.get('search_strategy', 'unknown')}")
        
        # Show what the LLM actually sees
        console.print(f"\n[bold green]📋 LLM CONTEXT (what the answerer receives):[/bold green]")
        if captured_context:
            # Format the context nicely for display
            context_display = str(captured_context)
            if len(context_display) > 3000:
                context_display = context_display[:3000] + "\n\n[... TRUNCATED FOR DISPLAY ...]"
            
            syntax = Syntax(context_display, "text", theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title="Raw LLM Context", expand=False))
        else:
            console.print("[red]No context captured[/red]")
        
        # Show the final answer
        console.print(f"\n[bold green]🤖 LLM RESPONSE:[/bold green]")
        console.print(Panel(response['answer'], title=f"Answer (confidence: {response.get('confidence', 'unknown')})", expand=False))
        
        if response.get('citations'):
            console.print(f"[dim]Citations: {response['citations']}[/dim]")
        
        # Restore original answerer
        rag.answerer = original_answerer
        
        return response
        
    except Exception as e:
        console.print(f"[red]ERROR: {str(e)}[/red]")
        return None

async def main():
    """Main debug function to test agentic integration."""
    console.print(Panel.fit(
        "[bold cyan]🔬 AGENTIC RAG INTEGRATION DEBUG[/bold cyan]\n"
        "Examining LLM context and agentic vs traditional execution",
        title="Debug Session"
    ))
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        console.print("[red]❌ Please set OPENAI_API_KEY environment variable[/red]")
        return
    
    try:
        # Initialize the system
        config = LLMConfig.from_env()
        rag = GenomicRAG(config)
        
        # Health check
        health = rag.health_check()
        console.print(f"[dim]System Health: {health}[/dim]")
        
        if not all(health.values()):
            console.print("[yellow]⚠️ Some components unhealthy - continuing anyway[/yellow]")
        
        # Test cases designed to trigger different execution paths
        test_cases = [
            {
                "query": "How many proteins are in the database?",
                "description": "Simple Count Query (Should use Traditional Mode)",
                "expected_mode": "traditional"
            },
            {
                "query": "Find proteins similar to heme transporters",
                "description": "Functional Similarity Query (Should use Traditional/Semantic Mode)",
                "expected_mode": "traditional"
            },
            {
                "query": "What does recent literature say about CRISPR proteins and how do they compare to our genomic data?",
                "description": "Literature + Database Integration (Should use Agentic Mode)",
                "expected_mode": "agentic"
            },
            {
                "query": "Search for research papers about bacterial heme transport systems and analyze how they relate to proteins in our database",
                "description": "Complex Multi-Step Analysis (Should use Agentic Mode)",
                "expected_mode": "agentic"
            }
        ]
        
        results = {}
        
        for i, test_case in enumerate(test_cases, 1):
            console.print(f"\n[bold white]TEST {i}/4[/bold white]")
            
            response = await debug_query_with_context(
                rag, 
                test_case["query"], 
                test_case["description"]
            )
            
            if response:
                actual_mode = response.get('query_metadata', {}).get('execution_mode', 'unknown')
                expected_mode = test_case["expected_mode"]
                
                mode_check = "✅" if actual_mode == expected_mode else "⚠️"
                console.print(f"\n{mode_check} [bold]Mode Prediction:[/bold] Expected {expected_mode}, got {actual_mode}")
                
                results[test_case["query"]] = {
                    "actual_mode": actual_mode,
                    "expected_mode": expected_mode,
                    "answer_length": len(response.get('answer', '')),
                    "confidence": response.get('confidence', 'unknown')
                }
            
            console.print("\n" + "─"*80)
        
        # Summary analysis
        console.print(f"\n[bold green]📊 INTEGRATION ANALYSIS SUMMARY[/bold green]")
        
        traditional_count = sum(1 for r in results.values() if r['actual_mode'] == 'traditional')
        agentic_count = sum(1 for r in results.values() if r['actual_mode'] == 'agentic')
        
        console.print(f"Traditional Mode: {traditional_count}/4 queries")
        console.print(f"Agentic Mode: {agentic_count}/4 queries")
        
        # Mode prediction accuracy
        correct_predictions = sum(1 for r in results.values() if r['actual_mode'] == r['expected_mode'])
        console.print(f"Mode Prediction Accuracy: {correct_predictions}/4 ({correct_predictions/4*100:.1f}%)")
        
        # Answer quality indicators
        avg_answer_length = sum(r['answer_length'] for r in results.values()) / len(results)
        high_confidence = sum(1 for r in results.values() if r['confidence'] == 'high')
        
        console.print(f"Average Answer Length: {avg_answer_length:.0f} characters")
        console.print(f"High Confidence Answers: {high_confidence}/4")
        
        # Show detailed results
        console.print(f"\n[bold cyan]DETAILED RESULTS:[/bold cyan]")
        for query, result in results.items():
            status = "✅" if result['actual_mode'] == result['expected_mode'] else "⚠️"
            console.print(f"{status} {query[:60]}...")
            console.print(f"   Mode: {result['actual_mode']} (expected: {result['expected_mode']})")
            console.print(f"   Confidence: {result['confidence']}, Length: {result['answer_length']} chars")
        
        # Integration assessment
        if correct_predictions >= 3 and high_confidence >= 2:
            console.print(f"\n[bold green]🎉 INTEGRATION STATUS: EXCELLENT[/bold green]")
            console.print("✅ Proper mode routing working")
            console.print("✅ High-quality responses generated")
            console.print("✅ Agentic capabilities properly integrated")
        elif correct_predictions >= 2:
            console.print(f"\n[bold yellow]⚠️ INTEGRATION STATUS: GOOD[/bold yellow]")
            console.print("✅ Basic functionality working")
            console.print("⚠️ Some mode prediction issues")
        else:
            console.print(f"\n[bold red]❌ INTEGRATION STATUS: NEEDS WORK[/bold red]")
            console.print("❌ Mode routing issues detected")
            console.print("❌ Review agentic planning logic")
        
        rag.close()
        
    except Exception as e:
        console.print(f"[red]FATAL ERROR: {str(e)}[/red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())