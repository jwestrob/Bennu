#!/usr/bin/env python3
"""
Debug what query DSPy generates for TPR domains.
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

async def debug_tpr_query():
    """Debug what query DSPy generates for TPR domains."""
    
    console.print("[bold green]🔍 Debugging TPR Query Generation[/bold green]")
    console.print("=" * 60)
    
    if not os.getenv('OPENAI_API_KEY'):
        console.print("[red]❌ No OPENAI_API_KEY found.[/red]")
        return
    
    try:
        # Initialize system
        config = LLMConfig.from_env()
        config.database.neo4j_password = "your_new_password"
        
        rag = GenomicRAG(config)
        
        question = "How many TPR domains are found in the dataset and what do they do?"
        
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
        console.print(f"Generated Cypher: {retrieval_plan.neo4j_query}")
        
        # Step 3: Test what TPR domains actually exist
        console.print("\n[yellow]Step 3: Check actual TPR domains in database[/yellow]")
        check_query = "MATCH (d:Domain) WHERE d.id CONTAINS 'TPR' RETURN d.id, d.description LIMIT 10"
        tpr_domains = await rag.neo4j_processor._execute_cypher(check_query)
        console.print(f"TPR domains found: {len(tpr_domains)}")
        for domain in tpr_domains:
            console.print(f"  - {domain['d.id']}: {domain['d.description']}")
        
        # Step 4: Test what domain annotations exist
        console.print("\n[yellow]Step 4: Check TPR domain annotations[/yellow]")
        annotation_query = "MATCH (da:DomainAnnotation) WHERE da.id CONTAINS 'TPR' RETURN count(da) as tpr_count"
        tpr_annotations = await rag.neo4j_processor._execute_cypher(annotation_query)
        if tpr_annotations:
            console.print(f"TPR domain annotations: {tpr_annotations[0]['tpr_count']}")
        
        # Step 5: Test the generated query directly
        console.print("\n[yellow]Step 5: Test generated query[/yellow]")
        try:
            generated_results = await rag.neo4j_processor._execute_cypher(retrieval_plan.neo4j_query)
            console.print(f"Generated query results: {len(generated_results)}")
        except Exception as e:
            console.print(f"Generated query failed: {e}")
        
        # Step 6: Test a corrected query
        console.print("\n[yellow]Step 6: Test corrected query[/yellow]")
        corrected_query = """
        MATCH (p:Protein)-[:HASDOMAIN]->(d:DomainAnnotation)-[:DOMAINFAMILY]->(pf:Domain) 
        WHERE pf.id CONTAINS 'TPR'
        MATCH (p)-[:ENCODEDBY]->(g:Gene) 
        OPTIONAL MATCH (p)-[:HASFUNCTION]->(ko:KEGGOrtholog) 
        RETURN p.id, d.id, d.bitscore, pf.description, pf.pfamAccession, ko.id, 
               ko.description, g.startCoordinate, g.endCoordinate, g.strand, g.lengthAA
        """
        try:
            corrected_results = await rag.neo4j_processor._execute_cypher(corrected_query)
            console.print(f"Corrected query results: {len(corrected_results)}")
            if corrected_results:
                console.print(f"Sample result: {corrected_results[0]}")
        except Exception as e:
            console.print(f"Corrected query failed: {e}")
        
        rag.close()
        
    except Exception as e:
        console.print(f"[red]❌ Debug failed: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_tpr_query())