#!/usr/bin/env python3
"""
Test the LLM integration system with basic questions.
"""

import asyncio
import os
import sys
from pathlib import Path
from rich.console import Console

# Add parent directory to Python path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.config import LLMConfig
from src.llm.rag_system import GenomicRAG

console = Console()


async def test_llm_integration():
    """Test the LLM integration with simple questions."""
    
    console.print("[bold green]🧬 Testing LLM Integration[/bold green]")
    console.print("="*50)
    
    # Check if we have an API key
    if not os.getenv('OPENAI_API_KEY'):
        console.print("[red]❌ No OPENAI_API_KEY found. Please set it to test LLM integration.[/red]")
        console.print("[yellow]💡 You can get an API key from https://platform.openai.com/api-keys[/yellow]")
        return False
    
    try:
        # Load configuration
        config = LLMConfig.from_env()
        
        # Override Neo4j password for our setup
        config.database.neo4j_password = "your_new_password"
        
        console.print(f"[dim]Using LLM: {config.llm_provider} - {config.llm_model}[/dim]")
        
        # Validate configuration
        status = config.validate_configuration()
        console.print(f"Configuration status: {status}")
        
        if not all(status.values()):
            console.print("[yellow]⚠️  Some components are not configured. Testing will continue with available components.[/yellow]")
        
        # Initialize RAG system
        rag = GenomicRAG(config)
        
        # Health check
        health = rag.health_check()
        console.print(f"System health: {health}")
        
        # Test questions (start simple)
        test_questions = [
            "How many genomes are in the database?",
            "What is the structure of the knowledge graph?",
            "List some protein families found in the data"
        ]
        
        for i, question in enumerate(test_questions, 1):
            console.print(f"\n[bold cyan]Test {i}:[/bold cyan] {question}")
            
            try:
                response = await rag.ask(question)
                
                console.print(f"[green]✅ Answer:[/green] {response['answer']}")
                console.print(f"[dim]Confidence: {response.get('confidence', 'unknown')}[/dim]")
                
                if 'query_metadata' in response:
                    metadata = response['query_metadata']
                    console.print(f"[dim]Query type: {metadata.get('query_type', 'unknown')} | "
                                 f"Time: {metadata.get('retrieval_time', 0):.2f}s[/dim]")
                
            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")
                continue
            
            console.print("-" * 40)
        
        # Close RAG system
        rag.close()
        
        console.print(f"\n[green]✅ LLM integration test completed![/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Test failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_llm_integration())
    exit(0 if success else 1)