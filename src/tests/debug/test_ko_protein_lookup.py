#!/usr/bin/env python3
"""
Test script to debug KO -> Protein lookup in Neo4j database
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm.query_processor import Neo4jQueryProcessor
from llm.config import LLMConfig

async def test_ko_protein_lookup():
    """Test direct Neo4j queries for proteins with specific KO annotations"""
    
    print("🔍 Testing KO -> Protein lookup in Neo4j database")
    
    try:
        # Initialize query processor
        config = LLMConfig()
        neo4j = Neo4jQueryProcessor(config)
        
        # Test 1: Check what KO orthologs we have
        print("\n📊 Step 1: Checking available KEGG orthologs...")
        ko_count_query = """
        MATCH (ko:KEGGOrtholog)
        RETURN count(ko) AS total_kos
        """
        
        result = await neo4j.process_query(ko_count_query, query_type="cypher")
        total_kos = result.results[0]['total_kos'] if result.results else 0
        print(f"✅ Found {total_kos} KEGG orthologs in database")
        
        # Test 2: Sample some KO IDs
        print("\n📋 Step 2: Sampling KO IDs...")
        sample_ko_query = """
        MATCH (ko:KEGGOrtholog)
        RETURN ko.id AS ko_id, ko.description AS description
        ORDER BY ko.id
        LIMIT 10
        """
        
        result = await neo4j.process_query(sample_ko_query, query_type="cypher")
        sample_kos = []
        for item in result.results:
            ko_id = item['ko_id']
            description = item['description']
            sample_kos.append(ko_id)
            print(f"  - {ko_id}: {description}")
        
        # Test 3: Check proteins with KO annotations
        print("\n🧬 Step 3: Checking proteins with KO annotations...")
        protein_ko_count_query = """
        MATCH (p:Protein)-[:HASFUNCTION]->(ko:KEGGOrtholog)
        RETURN count(DISTINCT p) AS proteins_with_ko, count(ko) AS total_annotations
        """
        
        result = await neo4j.process_query(protein_ko_count_query, query_type="cypher")
        if result.results:
            proteins_with_ko = result.results[0]['proteins_with_ko']
            total_annotations = result.results[0]['total_annotations']
            print(f"✅ Found {proteins_with_ko} proteins with KO annotations ({total_annotations} total annotations)")
        
        # Test 4: Try to find proteins for specific KOs
        if sample_kos:
            print(f"\n🎯 Step 4: Testing protein lookup for specific KOs...")
            test_kos = sample_kos[:3]  # Test first 3 KOs
            
            for ko_id in test_kos:
                print(f"\n  Testing KO: {ko_id}")
                
                # Single KO lookup
                single_ko_query = f"""
                MATCH (p:Protein)-[:HASFUNCTION]->(ko:KEGGOrtholog)
                WHERE ko.id = '{ko_id}'
                RETURN p.id AS protein_id, ko.id AS ko_id, ko.description AS ko_description
                LIMIT 5
                """
                
                result = await neo4j.process_query(single_ko_query, query_type="cypher")
                proteins_found = len(result.results)
                print(f"    Found {proteins_found} proteins with KO {ko_id}")
                
                for item in result.results[:2]:  # Show first 2
                    print(f"      - {item['protein_id']}")
            
            # Test 5: Multi-KO lookup (like pathway_based_protein_discovery does)
            print(f"\n🔗 Step 5: Testing multi-KO lookup...")
            ko_list_str = "', '".join(test_kos)
            multi_ko_query = f"""
            MATCH (p:Protein)-[:HASFUNCTION]->(ko:KEGGOrtholog)
            WHERE ko.id IN ['{ko_list_str}']
            RETURN p.id AS protein_id, ko.id AS ko_id, ko.description AS ko_description
            ORDER BY ko.id, p.id
            LIMIT 10
            """
            
            print(f"Query: {multi_ko_query}")
            result = await neo4j.process_query(multi_ko_query, query_type="cypher")
            proteins_found = len(result.results)
            print(f"✅ Multi-KO lookup found {proteins_found} proteins")
            
            for item in result.results:
                print(f"  - {item['protein_id']} -> {item['ko_id']}")
        
        # Test 6: Test transport-related KOs specifically
        print(f"\n🚛 Step 6: Testing transport-related KOs...")
        transport_query = """
        MATCH (ko:KEGGOrtholog)
        WHERE toLower(ko.description) CONTAINS 'transport' 
           OR toLower(ko.description) CONTAINS 'permease'
           OR toLower(ko.description) CONTAINS 'abc'
        RETURN ko.id AS ko_id, ko.description AS description
        LIMIT 5
        """
        
        result = await neo4j.process_query(transport_query, query_type="cypher")
        transport_kos = []
        for item in result.results:
            ko_id = item['ko_id']
            description = item['description']
            transport_kos.append(ko_id)
            print(f"  - {ko_id}: {description}")
        
        # Find proteins for transport KOs
        if transport_kos:
            print(f"\n🧬 Finding proteins for transport KOs...")
            ko_list_str = "', '".join(transport_kos)
            transport_protein_query = f"""
            MATCH (p:Protein)-[:HASFUNCTION]->(ko:KEGGOrtholog)
            WHERE ko.id IN ['{ko_list_str}']
            RETURN p.id AS protein_id, ko.id AS ko_id, ko.description AS ko_description
            ORDER BY ko.id, p.id
            LIMIT 15
            """
            
            result = await neo4j.process_query(transport_protein_query, query_type="cypher")
            proteins_found = len(result.results)
            print(f"✅ Found {proteins_found} transport proteins")
            
            for item in result.results:
                print(f"  - {item['protein_id']} -> {item['ko_id']}: {item['ko_description'][:50]}...")
        
        print(f"\n🎉 KO -> Protein lookup test complete!")
        
    except Exception as e:
        print(f"❌ Error in KO protein lookup test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_ko_protein_lookup())