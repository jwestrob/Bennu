#!/usr/bin/env python3
"""
Test TPR context formatting to see why count isn't showing.
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.syntax import Syntax

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm.config import LLMConfig
from src.llm.rag_system import GenomicRAG

console = Console()

@pytest.mark.asyncio
async def test_tpr_context_format():
    """Test TPR context formatting."""
    
    console.print("[bold green]🔍 Testing TPR Context Formatting[/bold green]")
    console.print("=" * 60)
    
    if not os.getenv('OPENAI_API_KEY'):
        console.print("[red]❌ No OPENAI_API_KEY found.[/red]")
        pytest.skip("No OPENAI_API_KEY found")
    
    try:
        # Initialize system
        config = LLMConfig.from_env()
        config.database.neo4j_password = "your_new_password"
        
        rag = GenomicRAG(config)
        
        question = "How many TPR domains are found in the dataset and what do they do?"
        
        # Get classification and retrieval plan
        classification = rag.classifier(question=question)
        retrieval_plan = rag.retriever(question=question, query_type=classification.query_type)
        
        console.print(f"[cyan]Generated query:[/cyan] {retrieval_plan.neo4j_query}")
        
        # Get context
        context = await rag._retrieve_context(classification.query_type, retrieval_plan)
        console.print(f"\nStructured data items: {len(context.structured_data)}")
        
        if context.structured_data:
            console.print(f"\nFirst structured data item:")
            console.print(context.structured_data[0])
        
        # Format context
        formatted_context = rag._format_context(context)
        console.print(f"\n[bold red]FORMATTED CONTEXT FOR LLM:[/bold red]")
        console.print("-" * 60)
        syntax = Syntax(formatted_context, "text", theme="monokai", line_numbers=True)
        console.print(syntax)
        console.print("-" * 60)
        
        rag.close()
        
    except Exception as e:
        console.print(f"[red]❌ Test failed: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_tpr_context_format())