#!/usr/bin/env python3
"""
Check if gene genomic coordinates are stored in Neo4j.
"""

import sys
from pathlib import Path
from neo4j import GraphDatabase
from rich.console import Console
from rich.table import Table

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()

def check_gene_coordinates():
    """Check gene coordinate data in Neo4j."""
    
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "your_new_password"))
    
    try:
        with driver.session() as session:
            console.print("[bold green]🔍 Checking Gene Coordinate Data[/bold green]")
            console.print("="*60)
            
            # 1. Check Gene node properties
            console.print("\n[bold]1. Gene Node Properties (sample)[/bold]")
            result = session.run("""
                MATCH (g:Gene)
                RETURN g.id as gene_id, 
                       g.startCoordinate as start, 
                       g.endCoordinate as end,
                       g.strand as strand,
                       g.lengthAA as length_aa,
                       g.lengthNt as length_nt,
                       g.gcContent as gc_content
                LIMIT 5
            """)
            
            table = Table(title="Gene Coordinates")
            table.add_column("Gene ID", style="cyan", max_width=40)
            table.add_column("Start", style="green")
            table.add_column("End", style="green") 
            table.add_column("Strand", style="yellow")
            table.add_column("Length AA", style="magenta")
            table.add_column("Length NT", style="magenta")
            table.add_column("GC Content", style="blue")
            
            for record in result:
                table.add_row(
                    str(record['gene_id'])[:40] + "..." if len(str(record['gene_id'])) > 40 else str(record['gene_id']),
                    str(record['start']) if record['start'] else "N/A",
                    str(record['end']) if record['end'] else "N/A",
                    str(record['strand']) if record['strand'] else "N/A", 
                    str(record['length_aa']) if record['length_aa'] else "N/A",
                    str(record['length_nt']) if record['length_nt'] else "N/A",
                    f"{float(record['gc_content']):.3f}" if record['gc_content'] else "N/A"
                )
            
            console.print(table)
            
            # 2. Check Gene coordinate statistics
            console.print("\n[bold]2. Gene Coordinate Statistics[/bold]")
            result = session.run("""
                MATCH (g:Gene)
                WHERE g.startCoordinate IS NOT NULL AND g.endCoordinate IS NOT NULL
                RETURN count(g) as genes_with_coords,
                       min(toInteger(g.startCoordinate)) as min_start,
                       max(toInteger(g.endCoordinate)) as max_end,
                       avg(toInteger(g.lengthAA)) as avg_aa_length,
                       avg(toFloat(g.gcContent)) as avg_gc_content
            """)
            
            for record in result:
                console.print(f"✅ Genes with coordinates: {record['genes_with_coords']:,}")
                console.print(f"✅ Coordinate range: {record['min_start']:,} - {record['max_end']:,}")
                console.print(f"✅ Average AA length: {record['avg_aa_length']:.1f}")
                console.print(f"✅ Average GC content: {record['avg_gc_content']:.3f}")
            
            # 3. Test genomic neighborhood query
            console.print("\n[bold]3. Genomic Neighborhood Test[/bold]")
            result = session.run("""
                MATCH (g1:Gene)-[:BELONGSTOGENOME]->(genome:Genome)
                WHERE g1.startCoordinate IS NOT NULL
                WITH g1, genome
                MATCH (g2:Gene)-[:BELONGSTOGENOME]->(genome)
                WHERE g2.startCoordinate IS NOT NULL 
                  AND g1.id <> g2.id
                  AND abs(g1.startCoordinate - g2.startCoordinate) < 5000
                RETURN g1.id as gene1, g1.startCoordinate as start1,
                       g2.id as gene2, g2.startCoordinate as start2,
                       abs(g1.startCoordinate - g2.startCoordinate) as distance
                ORDER BY distance
                LIMIT 5
            """)
            
            for record in result:
                console.print(f"  📍 {record['gene1'][:50]}... (pos: {record['start1']})")
                console.print(f"      ↕️ {record['distance']} bp apart")
                console.print(f"     {record['gene2'][:50]}... (pos: {record['start2']})")
                console.print()
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        driver.close()

if __name__ == "__main__":
    check_gene_coordinates()