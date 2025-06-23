#!/usr/bin/env python3
"""
Debug to see if there's a hidden LIMIT being applied to queries.
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

async def debug_query_limit():
    """Check if there's a hidden limit being applied."""
    
    console.print("[bold green]🔍 Debugging Query Limits[/bold green]")
    console.print("=" * 60)
    
    try:
        # Initialize system
        config = LLMConfig.from_env()
        config.database.neo4j_password = "your_new_password"
        
        rag = GenomicRAG(config)
        
        # Test 1: Count total GGDEF domains without limit
        console.print("[yellow]Test 1: Count all GGDEF domain annotations[/yellow]")
        count_query = "MATCH (da:DomainAnnotation) WHERE da.id CONTAINS 'GGDEF' RETURN count(da) as total_count"
        count_result = await rag.neo4j_processor._execute_cypher(count_query)
        console.print(f"Total GGDEF domain annotations: {count_result[0]['total_count']}")
        
        # Test 2: Get first 10 results explicitly
        console.print("\n[yellow]Test 2: Get first 10 GGDEF domains explicitly[/yellow]")
        limit_10_query = """
        MATCH (p:Protein)-[:HASDOMAIN]->(d:DomainAnnotation)-[:DOMAINFAMILY]->(pf:Domain) 
        WHERE d.id CONTAINS '/domain/GGDEF/' 
        MATCH (p)-[:ENCODEDBY]->(g:Gene) 
        OPTIONAL MATCH (p)-[:HASFUNCTION]->(ko:KEGGOrtholog) 
        RETURN p.id, d.id
        LIMIT 10
        """
        limit_result = await rag.neo4j_processor._execute_cypher(limit_10_query)
        console.print(f"Results with LIMIT 10: {len(limit_result)}")
        
        # Test 3: Get all results without explicit limit
        console.print("\n[yellow]Test 3: Get all GGDEF domains without LIMIT[/yellow]")
        no_limit_query = """
        MATCH (p:Protein)-[:HASDOMAIN]->(d:DomainAnnotation)-[:DOMAINFAMILY]->(pf:Domain) 
        WHERE d.id CONTAINS '/domain/GGDEF/' 
        MATCH (p)-[:ENCODEDBY]->(g:Gene) 
        OPTIONAL MATCH (p)-[:HASFUNCTION]->(ko:KEGGOrtholog) 
        RETURN p.id, d.id
        """
        all_results = await rag.neo4j_processor._execute_cypher(no_limit_query)
        console.print(f"Results without LIMIT: {len(all_results)}")
        
        # Test 4: Get first 150 results to see if we can exceed 100
        console.print("\n[yellow]Test 4: Try to get 150 results[/yellow]")
        limit_150_query = """
        MATCH (p:Protein)-[:HASDOMAIN]->(d:DomainAnnotation)-[:DOMAINFAMILY]->(pf:Domain) 
        WHERE d.id CONTAINS '/domain/GGDEF/' 
        MATCH (p)-[:ENCODEDBY]->(g:Gene) 
        OPTIONAL MATCH (p)-[:HASFUNCTION]->(ko:KEGGOrtholog) 
        RETURN p.id, d.id
        LIMIT 150
        """
        limit_150_result = await rag.neo4j_processor._execute_cypher(limit_150_query)
        console.print(f"Results with LIMIT 150: {len(limit_150_result)}")
        
        # Check config
        console.print(f"\n[cyan]Config max_results_per_query:[/cyan] {config.max_results_per_query}")
        
        rag.close()
        
    except Exception as e:
        console.print(f"[red]❌ Debug failed: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_query_limit())