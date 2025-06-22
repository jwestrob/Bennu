#!/usr/bin/env python3
"""
Debug genomic context formatting specifically.
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

async def debug_genomic_context():
    """Debug genomic context retrieval and formatting."""
    
    console.print("[bold green]🔍 Debug Genomic Context[/bold green]")
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
        
        # Test protein-specific question
        question = "Tell me about protein RIFCSPHIGHO2_01_FULL_Acidovorax_64_960_rifcsphigho2_01_scaffold_4_159"
        
        console.print(f"[cyan]Question:[/cyan] {question}")
        
        # Step 1: Classify the query type
        classification = rag.classifier(question=question)
        console.print(f"\n[yellow]Classification:[/yellow]")
        console.print(f"Query type: {classification.query_type}")
        console.print(f"Reasoning: {classification.reasoning}")
        
        # Step 2: Generate retrieval strategy
        retrieval_plan = rag.retriever(
            question=question,
            query_type=classification.query_type
        )
        console.print(f"\n[yellow]Retrieval Plan:[/yellow]")
        console.print(f"Neo4j Query: {retrieval_plan.neo4j_query}")
        console.print(f"Protein Search: {retrieval_plan.protein_search}")
        console.print(f"Search Strategy: {retrieval_plan.search_strategy}")
        
        # Step 3: Execute database queries
        console.print(f"\n[yellow]Executing Queries...[/yellow]")
        context = await rag._retrieve_context(classification.query_type, retrieval_plan)
        
        console.print(f"Structured data items: {len(context.structured_data)}")
        console.print(f"Semantic data items: {len(context.semantic_data)}")
        
        # Show first few raw data items
        console.print(f"\n[yellow]Raw Context Data (first 2 items):[/yellow]")
        for i, item in enumerate(context.structured_data[:2], 1):
            console.print(f"\n[cyan]Item {i}:[/cyan]")
            syntax = Syntax(str(item), "json", theme="monokai", line_numbers=False)
            console.print(syntax)
        
        # Step 4: Format context
        formatted_context = rag._format_context(context)
        
        console.print(f"\n[yellow]Formatted Context for LLM:[/yellow]")
        console.print("-" * 60)
        console.print(formatted_context)
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
    success = asyncio.run(debug_genomic_context())
    exit(0 if success else 1)