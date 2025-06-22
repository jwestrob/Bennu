#!/usr/bin/env python3
"""
Test the protein_info query directly.
"""

import sys
from pathlib import Path
from neo4j import GraphDatabase
from rich.console import Console
import json

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()

def test_protein_query():
    """Test the protein_info query directly."""
    
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "your_new_password"))
    
    try:
        with driver.session() as session:
            console.print("[bold green]🔍 Testing Protein Info Query[/bold green]")
            console.print("="*60)
            
            protein_id = "RIFCSPHIGHO2_01_FULL_Acidovorax_64_960_rifcsphigho2_01_scaffold_4_159"
            
            # Test the enhanced protein query
            cypher = """
            MATCH (p:Protein {id: $protein_id})
            OPTIONAL MATCH (p)<-[:ENCODEDBY]-(gene:Gene)-[:BELONGSTOGENOME]->(g:Genome)
            OPTIONAL MATCH (p)-[:HASDOMAIN]->(d:DomainAnnotation)-[:DOMAINFAMILY]->(pf:Domain)
            OPTIONAL MATCH (p)-[:HASFUNCTION]->(ko:KEGGOrtholog)
            
            // Get genomic neighborhood using coordinates (5kb window)
            OPTIONAL MATCH (neighbor_gene:Gene)-[:BELONGSTOGENOME]->(g)
            WHERE neighbor_gene.id <> gene.id
              AND neighbor_gene.startCoordinate IS NOT NULL 
              AND gene.startCoordinate IS NOT NULL
              AND abs(toInteger(neighbor_gene.startCoordinate) - toInteger(gene.startCoordinate)) < 5000
            OPTIONAL MATCH (neighbor_gene)-[:ENCODEDBY]->(neighbor_protein:Protein)
            OPTIONAL MATCH (neighbor_protein)-[:HASDOMAIN]->(neighbor_domain:DomainAnnotation)-[:DOMAINFAMILY]->(neighbor_pf:Domain)
            
            RETURN p.id as protein_id,
                   gene.id as gene_id,
                   g.id as genome_id,
                   gene.startCoordinate as gene_start,
                   gene.endCoordinate as gene_end,
                   gene.strand as gene_strand,
                   gene.lengthAA as gene_length_aa,
                   gene.gcContent as gene_gc_content,
                   collect(DISTINCT pf.id) as protein_families,
                   collect(DISTINCT pf.description) as domain_descriptions,
                   collect(DISTINCT pf.pfamAccession) as pfam_accessions,
                   collect(DISTINCT ko.id) as kegg_functions,
                   collect(DISTINCT ko.description) as kegg_descriptions,
                   collect(DISTINCT d.id) as domain_ids,
                   collect(DISTINCT d.bitscore) as domain_scores,
                   collect(DISTINCT (d.domainStart + '-' + d.domainEnd)) as domain_positions,
                   count(DISTINCT d) as domain_count,
                   collect(DISTINCT neighbor_protein.id)[0..5] as neighboring_proteins,
                   collect(DISTINCT neighbor_pf.id)[0..10] as neighborhood_families,
                   collect(DISTINCT neighbor_gene.startCoordinate)[0..5] as neighbor_coordinates
            """
            
            result = session.run(cypher, protein_id=protein_id)
            
            for record in result:
                console.print(f"[cyan]Raw Query Result:[/cyan]")
                data = dict(record)
                console.print(json.dumps(data, indent=2, default=str))
                
                # Check specific fields
                console.print(f"\n[yellow]Key Fields:[/yellow]")
                console.print(f"Protein ID: {data.get('protein_id')}")
                console.print(f"Gene ID: {data.get('gene_id')}")
                console.print(f"Genome ID: {data.get('genome_id')}")
                console.print(f"Gene Start: {data.get('gene_start')}")
                console.print(f"Gene End: {data.get('gene_end')}")
                console.print(f"Neighbors: {len(data.get('neighboring_proteins', []))}")
                
            # Also test basic gene lookup
            console.print(f"\n[yellow]Testing Gene Lookup:[/yellow]")
            result2 = session.run("""
                MATCH (p:Protein {id: $protein_id})<-[:ENCODEDBY]-(g:Gene)
                RETURN p.id, g.id, g.startCoordinate, g.endCoordinate, g.strand
            """, protein_id=protein_id)
            
            for record in result2:
                console.print(f"Direct gene lookup: {dict(record)}")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()

if __name__ == "__main__":
    test_protein_query()