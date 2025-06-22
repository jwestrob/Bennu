#!/usr/bin/env python3
"""
Check if GGDEF domain descriptions are actually stored in Neo4j.
"""

import sys
from pathlib import Path
from neo4j import GraphDatabase
from rich.console import Console
from rich.table import Table

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()

def check_ggdef_descriptions():
    """Check GGDEF domain data in Neo4j."""
    
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "your_new_password"))
    
    try:
        with driver.session() as session:
            console.print("[bold green]🔍 Checking GGDEF Domain Data[/bold green]")
            console.print("="*60)
            
            # 1. Check Domain nodes for GGDEF
            console.print("\n[bold]1. GGDEF Domain Family Nodes[/bold]")
            result = session.run("""
                MATCH (pf:Domain)
                WHERE pf.id CONTAINS 'GGDEF' OR pf.description CONTAINS 'GGDEF'
                RETURN pf.id as domain_id, pf.description, pf.pfamAccession, pf.familyType
                LIMIT 5
            """)
            
            for record in result:
                console.print(f"Domain ID: {record['domain_id']}")
                console.print(f"Description: {record['pf.description']}")
                console.print(f"PFAM Accession: {record['pf.pfamAccession']}")
                console.print(f"Family Type: {record['pf.familyType']}")
                console.print("-" * 40)
            
            # 2. Check DomainAnnotation → Domain relationship for GGDEF
            console.print("\n[bold]2. GGDEF Domain Annotations with Family Links[/bold]")
            result = session.run("""
                MATCH (d:DomainAnnotation)-[:DOMAINFAMILY]->(pf:Domain)
                WHERE d.id CONTAINS '/domain/GGDEF/'
                RETURN d.id as annotation_id, d.bitscore, pf.id as family_id, pf.description as family_description
                LIMIT 3
            """)
            
            for record in result:
                console.print(f"Annotation: {record['annotation_id']}")
                console.print(f"Bitscore: {record['d.bitscore']}")
                console.print(f"Family ID: {record['family_id']}")
                console.print(f"Family Description: {record['family_description']}")
                console.print("-" * 40)
            
            # 3. Check KEGG functions for GGDEF proteins
            console.print("\n[bold]3. KEGG Functions for GGDEF Proteins[/bold]")
            result = session.run("""
                MATCH (p:Protein)-[:HASDOMAIN]->(d:DomainAnnotation)
                WHERE d.id CONTAINS '/domain/GGDEF/'
                MATCH (p)-[:HASFUNCTION]->(ko:KEGGOrtholog)
                RETURN p.id as protein_id, ko.id as kegg_id, ko.description as kegg_description
                LIMIT 3
            """)
            
            kegg_found = False
            for record in result:
                kegg_found = True
                console.print(f"Protein: {record['protein_id']}")
                console.print(f"KEGG ID: {record['kegg_id']}")
                console.print(f"KEGG Description: {record['kegg_description']}")
                console.print("-" * 40)
            
            if not kegg_found:
                console.print("[yellow]No KEGG functions found for GGDEF proteins[/yellow]")
            
            # 4. Test the rich query that should be used
            console.print("\n[bold]4. Rich Query Test (what we should be using)[/bold]")
            result = session.run("""
                MATCH (p:Protein)-[:HASDOMAIN]->(d:DomainAnnotation)-[:DOMAINFAMILY]->(pf:Domain)
                WHERE d.id CONTAINS '/domain/GGDEF/'
                OPTIONAL MATCH (p)-[:HASFUNCTION]->(ko:KEGGOrtholog)
                RETURN p.id as protein_id, 
                       d.id as domain_id, 
                       d.bitscore,
                       pf.id as family_id,
                       pf.description as family_description,
                       pf.pfamAccession,
                       ko.id as kegg_id,
                       ko.description as kegg_description
                LIMIT 5
            """)
            
            for i, record in enumerate(result, 1):
                console.print(f"\n[cyan]Record {i}:[/cyan]")
                for key, value in record.items():
                    console.print(f"  {key}: {value}")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        driver.close()

if __name__ == "__main__":
    check_ggdef_descriptions()