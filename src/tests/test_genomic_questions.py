#!/usr/bin/env python3
"""
Test script for genomic RAG system with realistic bacterial genome questions.
"""

import asyncio
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm.config import LLMConfig
from src.llm.rag_system import GenomicRAG

console = Console()

# Test cases covering different aspects of bacterial genomes
TEST_QUESTIONS = [
    # 1. Domain-based questions (we know GGDEF exists)
    "What proteins contain GGDEF domains and what do they do?",
    
    # 2. Specific protein analysis (test neighborhood analysis)
    "Tell me about protein RIFCSPHIGHO2_01_FULL_Acidovorax_64_960_rifcsphigho2_01_scaffold_14_66",
    
    # 3. Functional pathway questions
    "What ATP-binding proteins are present in these genomes?",
    
    # 4. Metabolic function questions
    "Find proteins involved in DNA repair mechanisms",
    
    # 5. Transport proteins (common in bacteria)
    "What transport proteins or channels are annotated in the dataset?",
    
    # 6. Signal transduction (GGDEF domains are part of this)
    "What proteins are involved in bacterial signal transduction?",
    
    # 7. Similarity search question
    "Find proteins similar to DNA polymerase",
    
    # 8. Domain family analysis
    "What are the most common protein domain families in these genomes?",
]

async def test_genomic_rag():
    """Test the genomic RAG system with various bacterial genome questions."""
    
    console.print("[bold green]🧬 Testing Genomic RAG System[/bold green]")
    console.print("=" * 80)
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        console.print("[red]❌ No OPENAI_API_KEY found. Please set it to test.[/red]")
        return False
    
    try:
        # Initialize system
        config = LLMConfig.from_env()
        config.database.neo4j_password = "your_new_password"
        config.llm_model = "gpt-4o-mini"  # Use 4o-mini for testing
        
        rag = GenomicRAG(config)
        
        # Health check
        health = rag.health_check()
        console.print(f"[cyan]System Health:[/cyan] Neo4j: {health['neo4j']}, LanceDB: {health['lancedb']}")
        
        if not all(health.values()):
            console.print("[red]⚠️  System not healthy. Check configuration.[/red]")
            return False
        
        console.print(f"[green]✅ System ready! Testing with {config.llm_model}[/green]\n")
        
        # Test each question
        for i, question in enumerate(TEST_QUESTIONS, 1):
            console.print(Panel(
                f"[bold cyan]Question {i}:[/bold cyan] {question}",
                title="Test Case",
                border_style="blue"
            ))
            
            try:
                # Ask the question
                response = await rag.ask(question)
                
                # Display results
                console.print(f"[bold green]Answer:[/bold green]")
                console.print(response['answer'])
                console.print(f"\n[yellow]Confidence:[/yellow] {response['confidence']}")
                console.print(f"[yellow]Citations:[/yellow] {response['citations']}")
                console.print(f"[dim]Query time: {response['query_metadata']['retrieval_time']:.2f}s[/dim]")
                
                # Show query details for debugging
                metadata = response['query_metadata']
                console.print(f"[dim]Query type: {metadata.get('query_type', 'unknown')}[/dim]")
                console.print(f"[dim]Results: {metadata.get('structured_results', 0)} structured, {metadata.get('semantic_results', 0)} semantic[/dim]")
                
            except Exception as e:
                console.print(f"[red]❌ Question {i} failed: {e}[/red]")
            
            console.print("\n" + "─" * 80 + "\n")
            
            # Small delay between questions
            await asyncio.sleep(1)
        
        rag.close()
        console.print("[green]🎉 Testing completed![/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ System test failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_genomic_rag())
    exit(0 if success else 1)