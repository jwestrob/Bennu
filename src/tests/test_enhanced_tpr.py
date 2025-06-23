#!/usr/bin/env python3
"""
Test the enhanced TPR domain query processor.
"""

import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.json import JSON

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm.config import LLMConfig
from src.llm.rag_system import GenomicRAG

console = Console()

async def test_enhanced_tpr():
    """Test enhanced TPR domain processing."""
    
    console.print("[bold green]🔍 Testing Enhanced TPR Processing[/bold green]")
    console.print("=" * 60)
    
    try:
        # Initialize system
        config = LLMConfig.from_env()
        config.database.neo4j_password = "your_new_password"
        
        rag = GenomicRAG(config)
        
        # Test the enhanced domain query directly
        tpr_query = """
        MATCH (p:Protein)-[:HASDOMAIN]->(d:DomainAnnotation)-[:DOMAINFAMILY]->(pf:Domain) 
        WHERE pf.id CONTAINS 'TPR' 
        MATCH (p)-[:ENCODEDBY]->(g:Gene) 
        OPTIONAL MATCH (p)-[:HASFUNCTION]->(ko:KEGGOrtholog) 
        RETURN count(pf.id) as TPR_domain_count, pf.description, ko.id, ko.description
        """
        
        console.print("[yellow]Testing enhanced domain query processor...[/yellow]")
        enhanced_result = await rag.neo4j_processor._execute_domain_query_with_count(tpr_query)
        console.print(f"Enhanced query results: {len(enhanced_result)}")
        
        if enhanced_result:
            first_result = enhanced_result[0]
            console.print(f"Sample result metadata:")
            console.print(f"  - _domain_total_count: {first_result.get('_domain_total_count')}")
            console.print(f"  - _domain_name: {first_result.get('_domain_name')}")
            console.print(f"  - _is_sample: {first_result.get('_is_sample')}")
            console.print(f"  - _sample_size: {first_result.get('_sample_size')}")
            
            console.print(f"\nFirst few results:")
            for i, result in enumerate(enhanced_result[:3]):
                console.print(f"Result {i+1}: {result}")
        
        rag.close()
        
    except Exception as e:
        console.print(f"[red]❌ Test failed: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_enhanced_tpr())