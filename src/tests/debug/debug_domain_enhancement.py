#!/usr/bin/env python3
"""
Debug the domain query enhancement pipeline.
"""

import asyncio
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.json import JSON

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm.config import LLMConfig
from src.llm.rag_system import GenomicRAG

console = Console()

async def debug_domain_enhancement():
    """Debug the domain enhancement pipeline step by step."""
    
    console.print("[bold green]🔍 Debugging Domain Enhancement Pipeline[/bold green]")
    console.print("=" * 70)
    
    try:
        # Initialize system
        config = LLMConfig.from_env()
        config.database.neo4j_password = "your_new_password"
        
        rag = GenomicRAG(config)
        
        question = "What proteins contain GGDEF domains and what do they do?"
        
        console.print(f"[cyan]Question:[/cyan] {question}")
        
        # Step 1: Classification
        console.print("\n[yellow]Step 1: Classification[/yellow]")
        classification = rag.classifier(question=question)
        console.print(f"Query type: {classification.query_type}")
        
        # Step 2: Retrieval plan
        console.print("\n[yellow]Step 2: Retrieval Plan[/yellow]")
        retrieval_plan = rag.retriever(
            question=question,
            query_type=classification.query_type
        )
        console.print(f"Neo4j Query: {retrieval_plan.neo4j_query}")
        console.print(f"Protein Search: '{retrieval_plan.protein_search}'")
        
        # Step 3: Test enhanced domain query directly
        console.print("\n[yellow]Step 3: Test Enhanced Domain Query[/yellow]")
        enhanced_result = await rag.neo4j_processor._execute_domain_query_with_count(retrieval_plan.neo4j_query)
        console.print(f"Enhanced query results: {len(enhanced_result)}")
        
        if enhanced_result:
            first_result = enhanced_result[0]
            console.print(f"Sample result metadata:")
            console.print(f"  - _domain_total_count: {first_result.get('_domain_total_count')}")
            console.print(f"  - _domain_name: {first_result.get('_domain_name')}")
            console.print(f"  - _is_sample: {first_result.get('_is_sample')}")
            console.print(f"  - _sample_size: {first_result.get('_sample_size')}")
        
        # Step 4: Test full retrieve_context pipeline
        console.print("\n[yellow]Step 4: Test Full Pipeline[/yellow]")
        context = await rag._retrieve_context(classification.query_type, retrieval_plan)
        console.print(f"Pipeline structured_data: {len(context.structured_data)}")
        console.print(f"Pipeline semantic_data: {len(context.semantic_data)}")
        console.print(f"Pipeline metadata: {context.metadata}")
        
        # Step 5: Check protein_search logic
        console.print("\n[yellow]Step 5: Check Protein Search Logic[/yellow]")
        protein_search = retrieval_plan.protein_search
        is_actual_protein_id = (
            "RIFCS" in protein_search or 
            any(id_part in protein_search for id_part in ["scaffold", "contigs"]) or
            (len(protein_search) > 15 and "_" in protein_search and not " " in protein_search)
        )
        console.print(f"Protein search: '{protein_search}'")
        console.print(f"Is actual protein ID: {is_actual_protein_id}")
        console.print(f"Length > 15: {len(protein_search) > 15}")
        console.print(f"Has underscore: {'_' in protein_search}")
        console.print(f"No spaces: {' ' not in protein_search}")
        
        rag.close()
        
    except Exception as e:
        console.print(f"[red]❌ Debug failed: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_domain_enhancement())