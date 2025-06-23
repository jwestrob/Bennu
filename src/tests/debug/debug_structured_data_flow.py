#!/usr/bin/env python3
"""
Debug what happens to structured_data through the _retrieve_context method.
"""

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

async def debug_structured_data_flow():
    """Debug structured_data flow through _retrieve_context."""
    
    console.print("[bold green]🔍 Debugging Structured Data Flow[/bold green]")
    console.print("=" * 60)
    
    try:
        # Initialize system
        config = LLMConfig.from_env()
        config.database.neo4j_password = "your_new_password"
        
        rag = GenomicRAG(config)
        
        question = "What proteins contain GGDEF domains and what do they do?"
        
        # Get the classification and retrieval plan
        classification = rag.classifier(question=question)
        retrieval_plan = rag.retriever(question=question, query_type=classification.query_type)
        
        # Now let me manually trace through _retrieve_context with debug prints
        console.print(f"[cyan]Tracing _retrieve_context for hybrid query...[/cyan]")
        
        query_type = classification.query_type
        structured_data = []
        semantic_data = []
        metadata = {}
        
        console.print(f"1. Initial state: structured_data={len(structured_data)}")
        
        # Execute Neo4j query if one was generated
        if hasattr(retrieval_plan, 'neo4j_query') and retrieval_plan.neo4j_query.strip():
            console.print(f"2. Executing Neo4j query...")
            
            # Check if this is a protein-specific query
            protein_search = getattr(retrieval_plan, 'protein_search', '')
            is_protein_query = (
                'Protein {id:' in retrieval_plan.neo4j_query or
                ('RIFCS' in protein_search or 'scaffold' in protein_search) or
                len(protein_search) > 15
            )
            console.print(f"   is_protein_query: {is_protein_query}")
            console.print(f"   protein_search: '{protein_search}'")
            
            if is_protein_query and protein_search.strip():
                console.print(f"   Using protein_info query")
                neo4j_result = await rag.neo4j_processor.process_query(
                    protein_search,
                    query_type="protein_info"
                )
            else:
                console.print(f"   Using cypher query")
                neo4j_result = await rag.neo4j_processor.process_query(
                    retrieval_plan.neo4j_query,
                    query_type="cypher"
                )
            
            structured_data = neo4j_result.results
            metadata['neo4j_execution_time'] = neo4j_result.execution_time
            console.print(f"3. After Neo4j query: structured_data={len(structured_data)}")
        
        # Check semantic/hybrid logic
        if query_type in ["semantic", "hybrid"]:
            console.print(f"4. Processing hybrid/semantic logic...")
            protein_search = retrieval_plan.protein_search if hasattr(retrieval_plan, 'protein_search') else ""
            
            is_actual_protein_id = (
                "RIFCS" in protein_search or 
                any(id_part in protein_search for id_part in ["scaffold", "contigs"]) or
                (len(protein_search) > 15 and "_" in protein_search and not " " in protein_search)
            )
            console.print(f"   is_actual_protein_id: {is_actual_protein_id}")
            
            if is_actual_protein_id:
                console.print(f"   Would extend with protein_info...")
            else:
                console.print(f"   Skipping protein_info (not a protein ID)")
            
            console.print(f"5. After hybrid logic: structured_data={len(structured_data)}")
        
        console.print(f"6. Final: structured_data={len(structured_data)}, semantic_data={len(semantic_data)}")
        
        if structured_data:
            console.print(f"First structured item keys: {list(structured_data[0].keys())}")
        
        rag.close()
        
    except Exception as e:
        console.print(f"[red]❌ Debug failed: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_structured_data_flow())