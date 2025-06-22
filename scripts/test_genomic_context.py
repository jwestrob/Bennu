#!/usr/bin/env python3
"""
Test genomic context functionality end-to-end.
Verify that ENCODEDBY relationships are working and genomic coordinates are available.
"""

from neo4j import GraphDatabase
from rich.console import Console
from rich.table import Table

console = Console()

def test_encodedby_relationships():
    """Test that ENCODEDBY relationships are correctly linking proteins to genes."""
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "your_new_password"))
    
    console.print("[bold]Testing ENCODEDBY relationships...[/bold]")
    
    with driver.session() as session:
        # Test basic ENCODEDBY relationship
        query = """
        MATCH (p:Protein)-[:ENCODEDBY]->(g:Gene)
        RETURN p.id as protein_id, g.id as gene_id, 
               g.startCoordinate as start, g.endCoordinate as end, 
               g.strand as strand
        LIMIT 5
        """
        
        result = session.run(query)
        records = list(result)
        
        if not records:
            console.print("[red]❌ No ENCODEDBY relationships found![/red]")
            return False
        
        console.print(f"[green]✓ Found {len(records)} ENCODEDBY relationships[/green]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Protein ID", style="cyan")
        table.add_column("Gene ID", style="green") 
        table.add_column("Start", style="yellow")
        table.add_column("End", style="yellow")
        table.add_column("Strand", style="blue")
        
        for record in records:
            table.add_row(
                str(record["protein_id"])[:50] + "..." if len(str(record["protein_id"])) > 50 else str(record["protein_id"]),
                str(record["gene_id"])[:50] + "..." if len(str(record["gene_id"])) > 50 else str(record["gene_id"]),
                str(record["start"]),
                str(record["end"]),
                str(record["strand"])
            )
        
        console.print(table)
        return True

def test_genomic_neighborhood():
    """Test genomic neighborhood queries using coordinates."""
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "your_new_password"))
    
    console.print("\n[bold]Testing genomic neighborhood analysis...[/bold]")
    
    with driver.session() as session:
        # Test genomic neighborhood query (similar to what's in query_processor.py)
        query = """
        MATCH (p:Protein)-[:ENCODEDBY]->(gene:Gene)-[:BELONGSTOGENOME]->(g:Genome)
        WHERE p.id CONTAINS 'scaffold_4_'
        WITH p, gene, g LIMIT 1
        
        // Get genomic neighborhood using coordinates (5kb window)
        OPTIONAL MATCH (neighbor_gene:Gene)-[:BELONGSTOGENOME]->(g)
        WHERE neighbor_gene.id <> gene.id
          AND neighbor_gene.startCoordinate IS NOT NULL 
          AND gene.startCoordinate IS NOT NULL
          AND abs(toInteger(neighbor_gene.startCoordinate) - toInteger(gene.startCoordinate)) < 5000
        
        RETURN p.id as protein_id, gene.id as gene_id,
               gene.startCoordinate as gene_start, gene.endCoordinate as gene_end,
               gene.strand as gene_strand,
               collect(DISTINCT {
                 neighbor_id: neighbor_gene.id,
                 neighbor_start: neighbor_gene.startCoordinate,
                 neighbor_end: neighbor_gene.endCoordinate,
                 distance: abs(toInteger(neighbor_gene.startCoordinate) - toInteger(gene.startCoordinate))
               }) as neighbors
        """
        
        result = session.run(query)
        records = list(result)
        
        if not records:
            console.print("[red]❌ No genomic neighborhood data found![/red]")
            return False
        
        record = records[0]
        console.print(f"[green]✓ Found genomic neighborhood for protein: {record['protein_id'][:60]}...[/green]")
        console.print(f"Gene coordinates: {record['gene_start']}-{record['gene_end']} (strand {record['gene_strand']})")
        console.print(f"Neighbors found: {len(record['neighbors'])}")
        
        if record['neighbors']:
            neighbor_table = Table(show_header=True, header_style="bold magenta")
            neighbor_table.add_column("Neighbor Gene", style="cyan")
            neighbor_table.add_column("Coordinates", style="yellow")
            neighbor_table.add_column("Distance (bp)", style="green")
            
            for neighbor in record['neighbors'][:5]:  # Show first 5 neighbors
                if neighbor['neighbor_id']:
                    neighbor_table.add_row(
                        str(neighbor['neighbor_id'])[:40] + "..." if len(str(neighbor['neighbor_id'])) > 40 else str(neighbor['neighbor_id']),
                        f"{neighbor['neighbor_start']}-{neighbor['neighbor_end']}",
                        str(neighbor['distance'])
                    )
            
            console.print(neighbor_table)
        
        return True

def main():
    """Test genomic context functionality."""
    console.print("[bold blue]Testing Genomic Context Functionality[/bold blue]")
    console.print("=" * 50)
    
    success = True
    
    # Test ENCODEDBY relationships
    if not test_encodedby_relationships():
        success = False
    
    # Test genomic neighborhood
    if not test_genomic_neighborhood():
        success = False
    
    if success:
        console.print("\n[bold green]✅ All genomic context tests passed![/bold green]")
    else:
        console.print("\n[bold red]❌ Some genomic context tests failed![/bold red]")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())