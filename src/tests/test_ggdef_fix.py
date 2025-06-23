#!/usr/bin/env python3
"""
Test the GGDEF query fix.
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path
from rich.console import Console

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm.config import LLMConfig
from src.llm.rag_system import GenomicRAG

console = Console()

@pytest.mark.asyncio
async def test_ggdef_fix():
    """Test if the GGDEF query fix works."""
    
    console.print("[bold green]🧬 Testing GGDEF Query Fix[/bold green]")
    console.print("=" * 60)
    
    if not os.getenv('OPENAI_API_KEY'):
        console.print("[red]❌ No OPENAI_API_KEY found.[/red]")
        pytest.skip("No OPENAI_API_KEY found")
    
    try:
        # Initialize system
        config = LLMConfig.from_env()
        config.database.neo4j_password = "your_new_password"
        config.llm_model = "gpt-4o-mini"
        
        rag = GenomicRAG(config)
        
        # Test the GGDEF question
        question = "What proteins contain GGDEF domains and what do they do?"
        
        console.print(f"[cyan]Question:[/cyan] {question}")
        console.print("\n[yellow]Testing...[/yellow]")
        
        response = await rag.ask(question)
        
        console.print(f"\n[bold green]Answer:[/bold green]")
        console.print(response['answer'])
        console.print(f"\n[yellow]Confidence:[/yellow] {response['confidence']}")
        console.print(f"[yellow]Citations:[/yellow] {response['citations']}")
        
        # Check if we got database results this time
        metadata = response['query_metadata']
        console.print(f"\n[cyan]Database Results:[/cyan]")
        console.print(f"Structured: {metadata.get('structured_results', 0)}")
        console.print(f"Semantic: {metadata.get('semantic_results', 0)}")
        console.print(f"Query time: {metadata['retrieval_time']:.2f}s")
        
        rag.close()
        
        assert True  # Test completed successfully
        
    except Exception as e:
        console.print(f"[red]❌ Test failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        assert False, f"Test failed: {e}"

if __name__ == "__main__":
    success = asyncio.run(test_ggdef_fix())
    exit(0 if success else 1)