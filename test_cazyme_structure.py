#!/usr/bin/env python3
"""
Test script to determine correct CAZyme node labels and relationships.
This will help us fix the DSPy signature patterns.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.llm.query_processor import Neo4jQueryProcessor

def test_cazyme_patterns():
    """Test different CAZyme query patterns to find the correct structure."""
    
    processor = Neo4jQueryProcessor()
    
    print("🔍 Testing CAZyme Database Structure\n")
    
    # Test 1: Check what node labels exist
    print("=== Test 1: Finding CAZyme-related node labels ===")
    query1 = """
    CALL db.labels() YIELD label
    WHERE toLower(label) CONTAINS 'cazyme' OR toLower(label) CONTAINS 'cazy'
    RETURN label
    ORDER BY label
    """
    
    try:
        result1 = processor.execute_query(query1)
        print(f"Found {len(result1.results)} CAZyme-related labels:")
        for record in result1.results:
            print(f"  - {record.get('label')}")
    except Exception as e:
        print(f"❌ Error in Test 1: {e}")
    
    print()
    
    # Test 2: Check relationships from Protein nodes
    print("=== Test 2: Finding relationships from Protein to CAZyme nodes ===")
    query2 = """
    MATCH (p:Protein)-[r]->(target)
    WHERE type(r) = 'HASCAZYME' OR toLower(type(r)) CONTAINS 'cazyme'
    RETURN DISTINCT type(r) as relationship_type, labels(target) as target_labels, count(*) as count
    ORDER BY count DESC
    LIMIT 10
    """
    
    try:
        result2 = processor.execute_query(query2)
        print(f"Found {len(result2.results)} relationship patterns:")
        for record in result2.results:
            print(f"  - {record.get('relationship_type')} -> {record.get('target_labels')} ({record.get('count')} times)")
    except Exception as e:
        print(f"❌ Error in Test 2: {e}")
    
    print()
    
    # Test 3: Sample CAZyme nodes to see their structure
    print("=== Test 3: Sampling CAZyme node structure ===")
    query3 = """
    MATCH (n)
    WHERE any(label in labels(n) WHERE toLower(label) CONTAINS 'cazyme')
    RETURN labels(n) as node_labels, keys(n) as properties, n.id as sample_id
    LIMIT 5
    """
    
    try:
        result3 = processor.execute_query(query3)
        print(f"Found {len(result3.results)} sample CAZyme nodes:")
        for record in result3.results:
            print(f"  - Labels: {record.get('node_labels')}")
            print(f"    Properties: {record.get('properties')}")
            print(f"    Sample ID: {record.get('sample_id')}")
            print()
    except Exception as e:
        print(f"❌ Error in Test 3: {e}")
    
    # Test 4: Count total CAZyme relationships
    print("=== Test 4: Counting total CAZyme relationships ===")
    query4 = """
    MATCH (p:Protein)-[:HASCAZYME]->(ca)
    RETURN count(ca) as total_cazymes, count(DISTINCT ca) as unique_cazymes
    """
    
    try:
        result4 = processor.execute_query(query4)
        for record in result4.results:
            print(f"Total CAZyme relationships: {record.get('total_cazymes')}")
            print(f"Unique CAZyme nodes: {record.get('unique_cazymes')}")
    except Exception as e:
        print(f"❌ Error in Test 4: {e}")
    
    print()
    
    # Test 5: Test current incorrect pattern
    print("=== Test 5: Testing current (incorrect) pattern ===")
    query5 = """
    MATCH (p:Protein)-[:HASCAZYME]->(ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily)
    RETURN count(*) as count_with_current_pattern
    """
    
    try:
        result5 = processor.execute_query(query5)
        for record in result5.results:
            print(f"Results with current pattern: {record.get('count_with_current_pattern')}")
    except Exception as e:
        print(f"❌ Error in Test 5 (expected): {e}")
    
    print()
    
    # Test 6: Test alternative patterns
    patterns_to_test = [
        ("Cazyme", "BELONGSTOFAMILY", "CAZymeFamily"),
        ("Cazyme", "HASFAMILY", "CAZymeFamily"), 
        ("CAZymeAnnotation", "CAZYMEFAMILY", "CAZymeFamily"),
        ("Cazyme", None, None),  # Single node pattern
    ]
    
    print("=== Test 6: Testing alternative patterns ===")
    for i, (ca_label, rel_type, cf_label) in enumerate(patterns_to_test, 1):
        if rel_type and cf_label:
            query = f"""
            MATCH (p:Protein)-[:HASCAZYME]->(ca:{ca_label})-[:{rel_type}]->(cf:{cf_label})
            RETURN count(*) as pattern_count
            LIMIT 5
            """
            pattern_desc = f"{ca_label} -[:{rel_type}]-> {cf_label}"
        else:
            query = f"""
            MATCH (p:Protein)-[:HASCAZYME]->(ca:{ca_label})
            RETURN count(*) as pattern_count
            LIMIT 5
            """
            pattern_desc = f"{ca_label} (single node)"
        
        try:
            result = processor.execute_query(query)
            for record in result.results:
                count = record.get('pattern_count')
                print(f"  Pattern {i} ({pattern_desc}): {count} results")
                if count > 0:
                    print(f"    ✅ FOUND WORKING PATTERN!")
        except Exception as e:
            print(f"  Pattern {i} ({pattern_desc}): ❌ Failed - {e}")
    
    processor.close()

if __name__ == "__main__":
    test_cazyme_patterns()