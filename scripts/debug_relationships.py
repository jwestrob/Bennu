#!/usr/bin/env python3
"""
Debug script to check the actual Neo4j relationships for domain annotations.
"""

from neo4j import GraphDatabase
from rich.console import Console
from rich.json import JSON

console = Console()

def debug_relationships():
    """Debug the actual Neo4j relationships."""
    
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "your_new_password"))
    
    # Test protein from our debug
    protein_id = "protein:RIFCSPHIGHO2_01_FULL_Acidovorax_64_960_rifcsphigho2_01_scaffold_14_66"
    
    console.print(f"[bold]Testing relationships for:[/bold] {protein_id}")
    
    with driver.session() as session:
        # 1. Check if protein exists
        console.print("\n[yellow]1. Check protein exists:[/yellow]")
        query = "MATCH (p:Protein {id: $protein_id}) RETURN p.id"
        result = session.run(query, protein_id=protein_id)
        records = list(result)
        console.print(f"Protein found: {len(records) > 0}")
        
        if not records:
            console.print("❌ Protein not found!")
            return
            
        # 2. Check HASDOMAIN relationships
        console.print("\n[yellow]2. Check HASDOMAIN relationships:[/yellow]")
        query = "MATCH (p:Protein {id: $protein_id})-[:HASDOMAIN]->(da) RETURN da LIMIT 5"
        result = session.run(query, protein_id=protein_id)
        records = list(result)
        console.print(f"HASDOMAIN relationships found: {len(records)}")
        
        for record in records[:2]:
            console.print(JSON.from_data(dict(record['da']), indent=2))
            
        # 3. Check what relationship names exist from this protein
        console.print("\n[yellow]3. Check all relationship types from protein:[/yellow]")
        query = """
        MATCH (p:Protein {id: $protein_id})-[r]->(target)
        RETURN type(r) as relationship_type, labels(target) as target_labels, count(*) as count
        ORDER BY count DESC
        """
        result = session.run(query, protein_id=protein_id)
        records = list(result)
        
        for record in records:
            console.print(f"  {record['relationship_type']} -> {record['target_labels']}: {record['count']}")
            
        # 4. Check the BELONGSTOPROTEIN relationship path
        console.print("\n[yellow]4. Check BELONGSTOPROTEIN path:[/yellow]")
        query = """
        MATCH (p:Protein {id: $protein_id})-[:HASDOMAIN]->(da:DomainAnnotation)-[:BELONGSTOPROTEIN]->(d:Domain)
        RETURN da.id, da.bitscore, d.id, d.description
        LIMIT 5
        """
        result = session.run(query, protein_id=protein_id)
        records = list(result)
        console.print(f"Domain annotations via BELONGSTOPROTEIN: {len(records)}")
        
        for record in records:
            console.print(f"  Domain: {record['d.id']}")
            console.print(f"  Description: {record['d.description']}")
            console.print(f"  Score: {record['da.bitscore']}")
            console.print()
            
        # 5. Check the DOMAINFAMILY relationship path (our current query uses this)
        console.print("\n[yellow]5. Check DOMAINFAMILY path (current query):[/yellow]")
        query = """
        MATCH (p:Protein {id: $protein_id})-[:HASDOMAIN]->(da:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)
        RETURN da.id, da.bitscore, d.id, d.description
        LIMIT 5
        """
        result = session.run(query, protein_id=protein_id)
        records = list(result)
        console.print(f"Domain annotations via DOMAINFAMILY: {len(records)}")
        
        for record in records:
            console.print(f"  Domain: {record['d.id']}")
            console.print(f"  Description: {record['d.description']}")
            console.print(f"  Score: {record['da.bitscore']}")
            console.print()
    
    driver.close()

if __name__ == "__main__":
    debug_relationships()