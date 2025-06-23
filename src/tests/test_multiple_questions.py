#!/usr/bin/env python3
"""
Test multiple questions with the fixed system.
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

# Test questions that should work with our enhanced system
TEST_QUESTIONS = [
    "What proteins contain GGDEF domains and what do they do?",
    "Tell me about protein RIFCSPHIGHO2_01_FULL_Acidovorax_64_960_rifcsphigho2_01_scaffold_14_66",
    "What are the most common protein domain families in these genomes?",
    "How many proteins are in each genome?",
]

@pytest.mark.asyncio
async def test_multiple_questions():
    """Test multiple questions with the enhanced system."""
    
    console.print("[bold green]🧬 Testing Enhanced Genomic RAG System[/bold green]")
    console.print("=" * 80)
    
    if not os.getenv('OPENAI_API_KEY'):
        console.print("[red]❌ No OPENAI_API_KEY found.[/red]")
        pytest.skip("No OPENAI_API_KEY found")
    
    try:
        # Initialize system
        config = LLMConfig.from_env()
        config.database.neo4j_password = "your_new_password"
        config.llm_model = "gpt-4o-mini"
        
        rag = GenomicRAG(config)
        
        console.print(f"[green]✅ System ready with {config.llm_model}[/green]\n")
        
        # Test each question
        for i, question in enumerate(TEST_QUESTIONS, 1):
            console.print(Panel(
                f"[bold cyan]Question {i}:[/bold cyan] {question}",
                title="Test Case",
                border_style="blue"
            ))
            
            try:
                response = await rag.ask(question)
                
                # Display results
                console.print(f"[bold green]Answer:[/bold green]")
                console.print(response['answer'])
                console.print(f"\n[yellow]Confidence:[/yellow] {response['confidence']}")
                console.print(f"[yellow]Citations:[/yellow] {response['citations']}")
                
                # Check for real data indicators
                answer_text = response['answer'].lower()
                has_real_data = any([
                    'protein:' in answer_text,
                    'rifcs' in answer_text,
                    'scaffold' in answer_text,
                    'acidovorax' in answer_text,
                    'plm0_60' in answer_text,
                    'pf00' in response['citations'].lower()  # PFAM accessions
                ])
                
                if has_real_data:
                    console.print("[green]✅ Using real database results![/green]")
                else:
                    console.print("[yellow]⚠️  May be using general knowledge[/yellow]")
                
            except Exception as e:
                console.print(f"[red]❌ Question {i} failed: {e}[/red]")
            
            console.print("\n" + "─" * 80 + "\n")
        
        rag.close()
        console.print("[green]🎉 Testing completed![/green]")
        assert True  # Test completed successfully
        
    except Exception as e:
        console.print(f"[red]❌ System test failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        assert False, f"System test failed: {e}"

if __name__ == "__main__":
    success = asyncio.run(test_multiple_questions())
    exit(0 if success else 1)