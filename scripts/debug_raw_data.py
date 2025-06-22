#!/usr/bin/env python3
"""
Debug script to see the exact raw data being retrieved.
"""

import asyncio
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.json import JSON

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.config import LLMConfig
from src.llm.rag_system import GenomicRAG

console = Console()

async def debug_raw_data():
    """Debug the exact raw data being retrieved."""
    
    console.print("[bold green]🔍 Debug Raw Data Retrieval[/bold green]")
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
        
        # Test the protein with 27 domains
        protein_id = "RIFCSPHIGHO2_01_FULL_Acidovorax_64_960_rifcsphigho2_01_scaffold_14_66"
        
        console.print(f"[cyan]Testing protein:[/cyan] {protein_id}")
        
        # Test direct Neo4j protein_info query
        console.print("\n[yellow]1. Testing direct Neo4j protein_info query:[/yellow]")
        neo4j_result = await rag.neo4j_processor.process_query(
            protein_id,
            query_type="protein_info"
        )
        
        console.print(f"Results returned: {len(neo4j_result.results)}")
        console.print(f"Execution time: {neo4j_result.execution_time:.3f}s")
        
        if neo4j_result.results:
            for i, result in enumerate(neo4j_result.results):
                console.print(f"\n[bold]Result {i+1}:[/bold]")
                console.print(JSON.from_data(result, indent=2))
        else:
            console.print("No results returned from protein_info query")
        
        # Close RAG system
        rag.close()
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Debug failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(debug_raw_data())
    exit(0 if success else 1)