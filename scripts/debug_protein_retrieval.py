#!/usr/bin/env python3
"""
Debug protein retrieval to see if the enhanced _get_protein_info method is working.
"""

import sys
sys.path.append('/Users/jacob/Documents/Sandbox/microbial_claude_matter/src')

from llm.query_processor import Neo4jQueryProcessor
from llm.config import LLMConfig
from rich.console import Console
from rich.json import JSON
import asyncio

console = Console()

def test_protein_info_retrieval():
    """Test the enhanced protein info retrieval with genomic context."""
    
    console.print("[bold]Testing Enhanced Protein Info Retrieval[/bold]")
    
    # Initialize the query processor
    config = LLMConfig()
    processor = Neo4jQueryProcessor(config)
    
    # Test protein from our ENCODEDBY test
    protein_id = "RIFCSPHIGHO2_01_FULL_Gammaproteobacteria_61_200_rifcsphigho2_01_scaffold_40828_6"
    
    console.print(f"Testing protein: {protein_id}")
    
    # Call the enhanced _get_protein_info method
    result = asyncio.run(processor._get_protein_info(protein_id))
    
    console.print("\n[bold]Retrieved protein information:[/bold]")
    console.print(JSON.from_data(result, indent=2))
    
    return result

def test_direct_neo4j_query():
    """Test direct Neo4j query to verify data exists."""
    from neo4j import GraphDatabase
    
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "your_new_password"))
    
    console.print("\n[bold]Testing Direct Neo4j Query[/bold]")
    
    with driver.session() as session:
        # Check if the protein exists
        query = """
        MATCH (p:Protein)
        WHERE p.id CONTAINS 'scaffold_40828_6'
        RETURN p.id as protein_id
        LIMIT 5
        """
        
        result = session.run(query)
        records = list(result)
        
        console.print(f"Found {len(records)} proteins matching scaffold_40828_6:")
        for record in records:
            console.print(f"  • {record['protein_id']}")
        
        if records:
            # Test the enhanced query on the first protein
            test_protein = records[0]['protein_id']
            
            query = """
            MATCH (p:Protein)-[:ENCODEDBY]->(gene:Gene)-[:BELONGSTOGENOME]->(g:Genome)
            WHERE p.id = $protein_id
            
            // Get basic protein info
            OPTIONAL MATCH (p)-[:HASDOMAIN]->(da:DomainAnnotation)-[:BELONGSTOPROTEIN]->(d:Domain)
            OPTIONAL MATCH (p)-[:HASFUNCTION]->(fa:FunctionalAnnotation)-[:ANNOTATESPROTEIN]->(ko:KEGGOrtholog)
            
            // Get genomic neighborhood using coordinates (5kb window)
            OPTIONAL MATCH (neighbor_gene:Gene)-[:BELONGSTOGENOME]->(g)
            WHERE neighbor_gene.id <> gene.id
              AND neighbor_gene.startCoordinate IS NOT NULL 
              AND gene.startCoordinate IS NOT NULL
              AND abs(toInteger(neighbor_gene.startCoordinate) - toInteger(gene.startCoordinate)) < 5000
            
            RETURN p.id as protein_id, p.length as protein_length,
                   gene.id as gene_id, gene.startCoordinate as gene_start, 
                   gene.endCoordinate as gene_end, gene.strand as gene_strand,
                   g.id as genome_id,
                   collect(DISTINCT {domain_id: d.id, domain_name: d.name, bitscore: da.bitscore}) as domains,
                   collect(DISTINCT {kegg_id: ko.id, kegg_name: ko.name, definition: ko.definition}) as functions,
                   collect(DISTINCT {
                     neighbor_id: neighbor_gene.id,
                     neighbor_start: neighbor_gene.startCoordinate,
                     neighbor_end: neighbor_gene.endCoordinate,
                     distance: abs(toInteger(neighbor_gene.startCoordinate) - toInteger(gene.startCoordinate))
                   }) as neighbors
            """
            
            result = session.run(query, protein_id=test_protein)
            record = result.single()
            
            if record:
                console.print(f"\n[bold]Enhanced query result for {test_protein}:[/bold]")
                console.print(f"Gene coordinates: {record['gene_start']}-{record['gene_end']} (strand {record['gene_strand']})")
                console.print(f"Domains: {len([d for d in record['domains'] if d['domain_id']])}")
                console.print(f"Functions: {len([f for f in record['functions'] if f['kegg_id']])}")
                console.print(f"Neighbors: {len([n for n in record['neighbors'] if n['neighbor_id']])}")
            else:
                console.print(f"[red]No enhanced data found for {test_protein}[/red]")

def main():
    """Main test function."""
    
    # Test protein info retrieval
    test_protein_info_retrieval()
    
    # Test direct Neo4j query
    test_direct_neo4j_query()

if __name__ == "__main__":
    main()