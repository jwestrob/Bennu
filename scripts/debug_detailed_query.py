#!/usr/bin/env python3
"""
Debug script to test detailed queries that should return rich context.
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

async def test_detailed_queries():
    """Test queries that should return detailed protein information."""
    
    console.print("[bold green]🔍 Testing Detailed Queries[/bold green]")
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
        
        # Test questions that should return detailed data
        test_questions = [
            "What proteins have GGDEF domains?",  # Should return protein list
            "Show me details about GGDEF domain proteins",  # Should return protein details
            "Tell me about RIFCSPHIGHO2_01_FULL_Gammaproteobacteria_61_200_rifcsphigho2_01_scaffold_29964_1",  # Specific protein
        ]
        
        for i, question in enumerate(test_questions, 1):
            console.print(f"\n[bold cyan]Test {i}: {question}[/bold cyan]")
            
            # Classify and retrieve
            classification = rag.classifier(question=question)
            console.print(f"Query type: {classification.query_type}")
            
            retrieval_plan = rag.retriever(
                question=question,
                query_type=classification.query_type
            )
            console.print(f"Neo4j Query: {retrieval_plan.neo4j_query}")
            
            # Get context
            context = await rag._retrieve_context(classification.query_type, retrieval_plan)
            console.print(f"Structured data items: {len(context.structured_data)}")
            
            # Show first raw item
            if context.structured_data:
                console.print(f"\nFirst raw data item:")
                syntax = Syntax(str(context.structured_data[0]), "json", theme="monokai")
                console.print(syntax)
            
            # Show formatted context
            formatted_context = rag._format_context(context)
            console.print(f"\n[yellow]Formatted context:[/yellow]")
            console.print(formatted_context[:500] + "..." if len(formatted_context) > 500 else formatted_context)
            
            console.print("-" * 80)
        
        rag.close()
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Test failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_detailed_queries())
    exit(0 if success else 1)