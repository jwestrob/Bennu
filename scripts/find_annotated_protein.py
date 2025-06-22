#!/usr/bin/env python3
"""
Find a protein with rich domain annotations for testing context formatting.
"""

from neo4j import GraphDatabase
from rich.console import Console

console = Console()

def find_annotated_protein():
    """Find proteins with the most domain annotations."""
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "your_new_password"))
    
    with driver.session() as session:
        # Find proteins with most domains
        query = """
        MATCH (p:Protein)-[:HASDOMAIN]->(da:DomainAnnotation)
        RETURN p.id, count(da) as domain_count
        ORDER BY domain_count DESC
        LIMIT 5
        """
        
        result = session.run(query)
        records = list(result)
        
        console.print("[bold]Proteins with most domain annotations:[/bold]")
        for record in records:
            console.print(f"  • {record['p.id']}: {record['domain_count']} domains")
        
        if records:
            # Get detailed info for the top protein
            top_protein = records[0]['p.id']
            console.print(f"\n[bold]Detailed info for {top_protein}:[/bold]")
            
            query = """
            MATCH (p:Protein {id: $protein_id})-[:HASDOMAIN]->(da:DomainAnnotation)-[:BELONGSTOPROTEIN]->(d:Domain)
            RETURN d.id, d.description, da.bitscore, da.domainStart, da.domainEnd
            ORDER BY da.bitscore DESC
            LIMIT 5
            """
            
            result = session.run(query, protein_id=top_protein)
            records = list(result)
            
            for record in records:
                console.print(f"    Domain: {record['d.id']}")
                console.print(f"    Description: {record['d.description']}")
                console.print(f"    Score: {record['da.bitscore']}")
                console.print(f"    Position: {record['da.domainStart']}-{record['da.domainEnd']}")
                console.print()
            
            return top_protein
    
    driver.close()
    return None

if __name__ == "__main__":
    find_annotated_protein()