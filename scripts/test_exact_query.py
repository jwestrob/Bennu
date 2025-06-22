#!/usr/bin/env python3
"""
Test the exact enhanced query to see why it's returning empty results.
"""

from neo4j import GraphDatabase
from rich.console import Console
from rich.json import JSON

console = Console()

def test_exact_query():
    """Test the exact query from _get_protein_info method."""
    
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "your_new_password"))
    
    # The exact protein ID from our test
    protein_id = "protein:RIFCSPHIGHO2_01_FULL_Gammaproteobacteria_61_200_rifcsphigho2_01_scaffold_40828_6"
    
    console.print(f"Testing with exact protein ID: {protein_id}")
    
    with driver.session() as session:
        # Test basic MATCH first
        console.print("\n[bold]1. Testing basic MATCH:[/bold]")
        query = "MATCH (p:Protein {id: $protein_id}) RETURN p.id"
        result = session.run(query, protein_id=protein_id)
        records = list(result)
        console.print(f"Basic match found: {len(records)} records")
        
        if records:
            console.print(f"Found protein: {records[0]['p.id']}")
            
            # Test ENCODEDBY relationship
            console.print("\n[bold]2. Testing ENCODEDBY relationship:[/bold]")
            query = """
            MATCH (p:Protein {id: $protein_id})
            OPTIONAL MATCH (p)-[:ENCODEDBY]->(gene:Gene)
            RETURN p.id, gene.id as gene_id, gene.startCoordinate, gene.endCoordinate
            """
            result = session.run(query, protein_id=protein_id)
            record = result.single()
            if record and record['gene_id']:
                console.print(f"✓ ENCODEDBY works: {record['gene_id']} at {record['gene.startCoordinate']}-{record['gene.endCoordinate']}")
            else:
                console.print("✗ ENCODEDBY relationship not found")
            
            # Test the full enhanced query
            console.print("\n[bold]3. Testing full enhanced query:[/bold]")
            cypher = """
            MATCH (p:Protein {id: $protein_id})
            OPTIONAL MATCH (p)-[:ENCODEDBY]->(gene:Gene)-[:BELONGSTOGENOME]->(g:Genome)
            OPTIONAL MATCH (p)-[:HASDOMAIN]->(da:DomainAnnotation)-[:BELONGSTOPROTEIN]->(d:Domain)
            OPTIONAL MATCH (p)-[:HASFUNCTION]->(fa:Functionalannotation)-[:ANNOTATESPROTEIN]->(ko:KEGGOrtholog)
            
            // Get genomic neighborhood using coordinates (5kb window)
            OPTIONAL MATCH (neighbor_gene:Gene)-[:BELONGSTOGENOME]->(g)
            WHERE neighbor_gene.id <> gene.id
              AND neighbor_gene.startCoordinate IS NOT NULL 
              AND gene.startCoordinate IS NOT NULL
              AND abs(toInteger(neighbor_gene.startCoordinate) - toInteger(gene.startCoordinate)) < 5000
            OPTIONAL MATCH (neighbor_protein:Protein)-[:ENCODEDBY]->(neighbor_gene)
            OPTIONAL MATCH (neighbor_protein)-[:HASDOMAIN]->(neighbor_da:DomainAnnotation)-[:BELONGSTOPROTEIN]->(neighbor_d:Domain)
            
            RETURN p.id as protein_id,
                   gene.id as gene_id,
                   g.id as genome_id,
                   gene.startCoordinate as gene_start,
                   gene.endCoordinate as gene_end,
                   gene.strand as gene_strand,
                   gene.lengthAA as gene_length_aa,
                   gene.gcContent as gene_gc_content,
                   collect(DISTINCT d.id) as protein_families,
                   collect(DISTINCT d.description) as domain_descriptions,
                   collect(DISTINCT d.pfamAccession) as pfam_accessions,
                   collect(DISTINCT ko.id) as kegg_functions,
                   collect(DISTINCT ko.description) as kegg_descriptions,
                   collect(DISTINCT da.id) as domain_ids,
                   collect(DISTINCT da.bitscore) as domain_scores,
                   collect(DISTINCT (da.domainStart + '-' + da.domainEnd)) as domain_positions,
                   count(DISTINCT da) as domain_count,
                   collect(DISTINCT neighbor_protein.id)[0..5] as neighboring_proteins,
                   collect(DISTINCT neighbor_d.id)[0..10] as neighborhood_families,
                   collect(DISTINCT neighbor_gene.startCoordinate)[0..5] as neighbor_coordinates
            """
            
            result = session.run(cypher, protein_id=protein_id)
            records = list(result)
            
            console.print(f"Enhanced query returned: {len(records)} records")
            if records:
                record = records[0]
                console.print(JSON.from_data(dict(record), indent=2))
            else:
                console.print("Enhanced query returned empty results")
        else:
            console.print("Protein not found with exact ID match")
    
    driver.close()

if __name__ == "__main__":
    test_exact_query()