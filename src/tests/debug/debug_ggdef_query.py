#!/usr/bin/env python3
"""
Debug why GGDEF domain query isn't returning results.
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

async def debug_ggdef_query():
    """Debug why GGDEF query returns no results."""
    
    console.print("[bold green]🔍 Debugging GGDEF Query[/bold green]")
    console.print("=" * 60)
    
    try:
        # Initialize system
        config = LLMConfig.from_env()
        config.database.neo4j_password = "your_new_password"
        
        rag = GenomicRAG(config)
        
        question = "What proteins contain GGDEF domains and what do they do?"
        
        # Step 1: See what DSPy generates
        console.print("[yellow]Step 1: Query Classification[/yellow]")
        classification = rag.classifier(question=question)
        console.print(f"Query type: {classification.query_type}")
        console.print(f"Reasoning: {classification.reasoning}")
        
        console.print("\n[yellow]Step 2: Retrieval Strategy[/yellow]")
        retrieval_plan = rag.retriever(
            question=question,
            query_type=classification.query_type
        )
        console.print(f"Neo4j Query: {retrieval_plan.neo4j_query}")
        console.print(f"Protein Search: {retrieval_plan.protein_search}")
        
        # Step 3: Test the Neo4j query directly
        console.print("\n[yellow]Step 3: Testing Neo4j Query Directly[/yellow]")
        try:
            neo4j_result = await rag.neo4j_processor._execute_cypher(retrieval_plan.neo4j_query)
            console.print(f"Direct Neo4j results: {len(neo4j_result)}")
            
            if neo4j_result:
                console.print("Sample results:")
                console.print(JSON.from_data(neo4j_result[:2], indent=2))
            else:
                console.print("No results from Neo4j query!")
                
        except Exception as e:
            console.print(f"Neo4j query failed: {e}")
        
        # Step 4: Test if GGDEF domains exist at all
        console.print("\n[yellow]Step 4: Testing if GGDEF exists in database[/yellow]")
        test_query = "MATCH (d:Domain) WHERE d.id CONTAINS 'GGDEF' RETURN d.id, d.description LIMIT 5"
        try:
            direct_ggdef = await rag.neo4j_processor._execute_cypher(test_query)
            console.print(f"Direct GGDEF search results: {len(direct_ggdef)}")
            
            if direct_ggdef:
                console.print("GGDEF domains found:")
                console.print(JSON.from_data(direct_ggdef, indent=2))
            else:
                console.print("No GGDEF domains found in database!")
                
        except Exception as e:
            console.print(f"Direct GGDEF search failed: {e}")
            
        # Step 5: Check domain annotation structure
        console.print("\n[yellow]Step 5: Check DomainAnnotation structure[/yellow]")
        structure_query = "MATCH (da:DomainAnnotation) WHERE da.id CONTAINS 'GGDEF' RETURN da.id LIMIT 5"
        try:
            domain_annotations = await rag.neo4j_processor._execute_cypher(structure_query)
            console.print(f"GGDEF domain annotations: {len(domain_annotations)}")
            
            if domain_annotations:
                console.print("GGDEF domain annotations found:")
                console.print(JSON.from_data(domain_annotations, indent=2))
                
        except Exception as e:
            console.print(f"Domain annotation search failed: {e}")
        
        rag.close()
        
    except Exception as e:
        console.print(f"[red]❌ Debug failed: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_ggdef_query())