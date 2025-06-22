#!/usr/bin/env python3
"""
Debug script to see exactly what context gets formatted for the LLM.
"""

import asyncio
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.syntax import Syntax

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.config import LLMConfig
from src.llm.rag_system import GenomicRAG

console = Console()

async def debug_context_format():
    """Debug what context gets formatted for the LLM."""
    
    console.print("[bold green]🔍 Debug Context Formatting[/bold green]")
    console.print("="*60)
    
    # Check if we have an API key
    if not os.getenv('OPENAI_API_KEY'):
        console.print("[red]❌ No OPENAI_API_KEY found. Please set it to test LLM integration.[/red]")
        return False
    
    try:
        # Load configuration
        config = LLMConfig.from_env()
        config.database.neo4j_password = "your_new_password"
        
        # Initialize RAG system
        rag = GenomicRAG(config)
        
        # Test question with a protein that has rich domain annotations (27 domains!)
        question = "Tell me about protein RIFCSPHIGHO2_01_FULL_Acidovorax_64_960_rifcsphigho2_01_scaffold_14_66"
        
        console.print(f"[cyan]Question:[/cyan] {question}")
        console.print("\n[yellow]Step 1: Classifying query...[/yellow]")
        
        # Step 1: Classify the query type
        classification = rag.classifier(question=question)
        console.print(f"Query type: {classification.query_type}")
        console.print(f"Reasoning: {classification.reasoning}")
        
        console.print("\n[yellow]Step 2: Generating retrieval plan...[/yellow]")
        
        # Step 2: Generate retrieval strategy
        retrieval_plan = rag.retriever(
            question=question,
            query_type=classification.query_type
        )
        console.print(f"Neo4j Query: {retrieval_plan.neo4j_query}")
        console.print(f"Protein Search: {retrieval_plan.protein_search}")
        console.print(f"Search Strategy: {retrieval_plan.search_strategy}")
        
        console.print("\n[yellow]Step 3: Retrieving raw context...[/yellow]")
        
        # Step 3: Execute database queries
        context = await rag._retrieve_context(classification.query_type, retrieval_plan)
        
        console.print(f"Structured data items: {len(context.structured_data)}")
        console.print(f"Semantic data items: {len(context.semantic_data)}")
        console.print(f"Query time: {context.query_time:.3f}s")
        
        # Show raw data structure
        console.print("\n[yellow]Step 4: Raw structured data (first 2 items):[/yellow]")
        for i, item in enumerate(context.structured_data[:2]):
            console.print(f"\nItem {i+1}:")
            syntax = Syntax(str(item), "json", theme="monokai", line_numbers=True)
            console.print(syntax)
        
        console.print("\n[yellow]Step 5: Formatted context (what the LLM sees):[/yellow]")
        
        # Step 4: Format context
        formatted_context = rag._format_context(context)
        
        # Show formatted context with syntax highlighting
        console.print("\n[bold red]FORMATTED CONTEXT FOR LLM:[/bold red]")
        console.print("-" * 60)
        syntax = Syntax(formatted_context, "text", theme="monokai", line_numbers=True)
        console.print(syntax)
        console.print("-" * 60)
        
        # Close RAG system
        rag.close()
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Debug failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(debug_context_format())
    exit(0 if success else 1)