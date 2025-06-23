#!/usr/bin/env python3
"""
Test if the 100-limit applies to all queries.
"""

import pytest
import asyncio
import sys
from pathlib import Path
from rich.console import Console

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm.config import LLMConfig
from src.llm.rag_system import GenomicRAG

console = Console()

@pytest.mark.asyncio
async def test_simple_queries():
    """Test simple queries to see if 100-limit is universal."""
    
    console.print("[bold green]🔍 Testing Simple Queries for 100-Limit[/bold green]")
    console.print("=" * 60)
    
    try:
        # Initialize system
        config = LLMConfig.from_env()
        config.database.neo4j_password = "your_new_password"
        
        rag = GenomicRAG(config)
        
        # Test 1: Count all proteins
        console.print("[yellow]Test 1: Count all proteins[/yellow]")
        count_query = "MATCH (p:Protein) RETURN count(p) as total_proteins"
        count_result = await rag.neo4j_processor._execute_cypher(count_query)
        total_proteins = count_result[0]['total_proteins']
        console.print(f"Total proteins: {total_proteins}")
        
        # Test 2: Get all proteins without limit
        console.print(f"\n[yellow]Test 2: Get all proteins (expect {total_proteins})[/yellow]")
        all_proteins_query = "MATCH (p:Protein) RETURN p.id"
        all_proteins = await rag.neo4j_processor._execute_cypher(all_proteins_query)
        console.print(f"Proteins returned: {len(all_proteins)}")
        
        # Test 3: Try with explicit high limit
        console.print(f"\n[yellow]Test 3: Get proteins with LIMIT 1000[/yellow]")
        limit_query = "MATCH (p:Protein) RETURN p.id LIMIT 1000"
        limit_proteins = await rag.neo4j_processor._execute_cypher(limit_query)
        console.print(f"Proteins with LIMIT 1000: {len(limit_proteins)}")
        
        # Test 4: Check domain annotations
        console.print(f"\n[yellow]Test 4: Count all domain annotations[/yellow]")
        domain_count_query = "MATCH (da:DomainAnnotation) RETURN count(da) as total_domains"
        domain_count = await rag.neo4j_processor._execute_cypher(domain_count_query)
        total_domains = domain_count[0]['total_domains']
        console.print(f"Total domain annotations: {total_domains}")
        
        # Test 5: Get all domain annotations
        console.print(f"\n[yellow]Test 5: Get all domain annotations (expect {total_domains})[/yellow]")
        all_domains_query = "MATCH (da:DomainAnnotation) RETURN da.id"
        all_domains = await rag.neo4j_processor._execute_cypher(all_domains_query)
        console.print(f"Domain annotations returned: {len(all_domains)}")
        
        if len(all_domains) == 100 and total_domains > 100:
            console.print("[red]⚠️  Found the 100-limit bug! Affects all large result sets.[/red]")
        elif len(all_domains) == total_domains:
            console.print("[green]✅ No universal 100-limit found.[/green]")
        
        rag.close()
        
    except Exception as e:
        console.print(f"[red]❌ Test failed: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simple_queries())