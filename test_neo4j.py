#!/usr/bin/env python3
"""
Test Neo4j knowledge graph with basic queries.
"""

from neo4j import GraphDatabase
from rich.console import Console
from rich.table import Table

console = Console()


def test_neo4j_connection():
    """Test basic Neo4j queries on our genomic knowledge graph."""
    
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "your_new_password"))
    
    try:
        with driver.session() as session:
            console.print("[bold green]🧬 Neo4j Genomic Knowledge Graph Test[/bold green]")
            console.print("="*60)
            
            # 1. Database overview
            console.print("\n[bold]1. Database Overview[/bold]")
            result = session.run("MATCH (n) RETURN labels(n) as labels, count(n) as count ORDER BY count DESC")
            
            table = Table(title="Node Types")
            table.add_column("Node Type", style="cyan")
            table.add_column("Count", style="magenta", justify="right")
            
            for record in result:
                label = record["labels"][0] if record["labels"] else "Unknown"
                table.add_row(label, f"{record['count']:,}")
            
            console.print(table)
            
            # 2. Relationship overview
            console.print("\n[bold]2. Relationship Types[/bold]")
            result = session.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC")
            
            rel_table = Table(title="Relationship Types")
            rel_table.add_column("Relationship", style="cyan")
            rel_table.add_column("Count", style="magenta", justify="right")
            
            for record in result:
                rel_table.add_row(record["rel_type"], f"{record['count']:,}")
            
            console.print(rel_table)
            
            # 3. Sample genomes (exclude quality metrics)
            console.print("\n[bold]3. Sample Genomes[/bold]")
            result = session.run("""
                MATCH (g:Genome) 
                WHERE NOT g.id CONTAINS '/quality'
                RETURN g.id as genome_id 
                ORDER BY g.id
                LIMIT 5
            """)
            
            for record in result:
                console.print(f"  • {record['genome_id']}")
            
            # 4. Proteins per genome (via gene relationships)
            console.print("\n[bold]4. Proteins per Genome[/bold]")
            result = session.run("""
                MATCH (g:Genome)<-[:belongsToGenome]-(gene:Gene)<-[:encodedBy]-(p:Protein)
                WHERE NOT g.id CONTAINS '/quality'
                RETURN g.id as genome, count(DISTINCT p) as protein_count
                ORDER BY protein_count DESC
            """)
            
            for record in result:
                console.print(f"  • {record['genome']}: {record['protein_count']:,} proteins")
            
            # 5. Sample protein domains
            console.print("\n[bold]5. Sample Protein Families (PFAM)[/bold]")
            result = session.run("MATCH (pf:ProteinFamily) RETURN pf.id as family_id LIMIT 5")
            
            for record in result:
                console.print(f"  • {record['family_id']}")
            
            # 6. Protein with most domains
            console.print("\n[bold]6. Proteins with Most Domains[/bold]")
            result = session.run("""
                MATCH (p:Protein)-[:hasDomain]->(d:ProteinDomain)
                RETURN p.id as protein, count(d) as domain_count
                ORDER BY domain_count DESC
                LIMIT 3
            """)
            
            for record in result:
                console.print(f"  • {record['protein']}: {record['domain_count']} domains")
            
            console.print("\n[green]✓ Neo4j knowledge graph is operational![/green]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        driver.close()


if __name__ == "__main__":
    test_neo4j_connection()