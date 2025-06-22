#!/usr/bin/env python3
"""
Check the actual Neo4j schema to fix query mismatches.
"""

from neo4j import GraphDatabase
from rich.console import Console
from rich.table import Table

console = Console()

def check_schema():
    """Check the actual Neo4j schema."""
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "your_new_password"))
    
    with driver.session() as session:
        # Check all node labels
        console.print("\n[bold]Node Labels in Database:[/bold]")
        result = session.run("CALL db.labels()")
        labels = [record["label"] for record in result]
        for label in sorted(labels):
            console.print(f"  • {label}")
        
        # Check all relationship types
        console.print("\n[bold]Relationship Types in Database:[/bold]")
        result = session.run("CALL db.relationshipTypes()")
        rels = [record["relationshipType"] for record in result]
        for rel in sorted(rels):
            console.print(f"  • {rel}")
        
        # Check specific protein's relationships and properties
        console.print("\n[bold]Sample Protein Relationships and Properties:[/bold]")
        query = """
        MATCH (p:Protein)
        WHERE p.id CONTAINS 'scaffold_40828_6'
        OPTIONAL MATCH (p)-[r]->(connected)
        RETURN p.id, type(r) as relationship_type, labels(connected) as connected_labels, 
               keys(p) as protein_properties
        LIMIT 10
        """
        
        result = session.run(query)
        records = list(result)
        
        if records:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Protein ID", style="cyan")
            table.add_column("Relationship", style="yellow")
            table.add_column("Connected To", style="green")
            table.add_column("Properties", style="blue")
            
            for record in records:
                protein_id = str(record["p.id"])[:50] + "..." if len(str(record["p.id"])) > 50 else str(record["p.id"])
                rel_type = str(record["relationship_type"]) if record["relationship_type"] else "None"
                connected = str(record["connected_labels"]) if record["connected_labels"] else "None"
                properties = str(record["protein_properties"])[:30] + "..." if record["protein_properties"] and len(str(record["protein_properties"])) > 30 else str(record["protein_properties"])
                
                table.add_row(protein_id, rel_type, connected, properties)
            
            console.print(table)
        
        # Check Gene properties for our target protein
        console.print("\n[bold]Gene Properties for Target Protein:[/bold]")
        query = """
        MATCH (p:Protein)-[:ENCODEDBY]->(g:Gene)
        WHERE p.id CONTAINS 'scaffold_40828_6'
        RETURN keys(g) as gene_properties, g.startCoordinate, g.endCoordinate, g.strand
        """
        
        result = session.run(query)
        record = result.single()
        if record:
            console.print(f"Gene properties: {record['gene_properties']}")
            console.print(f"Coordinates: {record['g.startCoordinate']}-{record['g.endCoordinate']} (strand {record['g.strand']})")
        
        # Check functional annotations
        console.print("\n[bold]Functional Annotation Pattern:[/bold]")
        query = """
        MATCH (p:Protein)
        WHERE p.id CONTAINS 'scaffold_40828_6'
        OPTIONAL MATCH (p)-[r]->(func)
        WHERE type(r) CONTAINS 'FUNCTION'
        RETURN type(r) as function_rel, labels(func) as function_labels
        """
        
        result = session.run(query)
        records = list(result)
        for record in records:
            if record["function_rel"]:
                console.print(f"  Function relationship: {record['function_rel']} -> {record['function_labels']}")
    
    driver.close()

if __name__ == "__main__":
    check_schema()