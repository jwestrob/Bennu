#!/usr/bin/env python3
"""
Direct Neo4j test for CAZyme structure without complex imports.
"""

try:
    from neo4j import GraphDatabase
    neo4j_available = True
except ImportError:
    neo4j_available = False

def test_cazyme_patterns():
    """Test different CAZyme query patterns directly with Neo4j."""
    
    if not neo4j_available:
        print("❌ Neo4j driver not available")
        return
    
    # Try different connection options
    connection_options = [
        ('bolt://localhost:7687', 'neo4j', 'password'),
        ('bolt://localhost:7687', 'neo4j', 'neo4j'),
        ('bolt://localhost:7687', '', ''),
        ('neo4j://localhost:7687', 'neo4j', 'password'),
    ]
    
    driver = None
    for uri, user, password in connection_options:
        try:
            print(f"🔍 Trying connection: {uri} with user '{user}'")
            driver = GraphDatabase.driver(uri, auth=(user, password) if user else None)
            with driver.session() as session:
                # Test connection with simple query
                result = session.run("RETURN 1 as test")
                list(result)  # Force execution
                print(f"✅ Connected successfully!")
                break
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            if driver:
                driver.close()
            driver = None
    
    if not driver:
        print("❌ Could not connect to Neo4j database")
        return
    
    print("\n🔍 Testing CAZyme Database Structure\n")
    
    try:
        with driver.session() as session:
            
            # Test 1: Check what node labels exist
            print("=== Test 1: Finding CAZyme-related node labels ===")
            try:
                result = session.run("""
                    CALL db.labels() YIELD label
                    WHERE toLower(label) CONTAINS 'cazyme' OR toLower(label) CONTAINS 'cazy'
                    RETURN label
                    ORDER BY label
                """)
                labels = [record['label'] for record in result]
                print(f"Found {len(labels)} CAZyme-related labels:")
                for label in labels:
                    print(f"  - {label}")
            except Exception as e:
                print(f"❌ Error in Test 1: {e}")
            
            print()
            
            # Test 2: Check relationships from Protein nodes
            print("=== Test 2: Finding relationships from Protein to CAZyme nodes ===")
            try:
                result = session.run("""
                    MATCH (p:Protein)-[r]->(target)
                    WHERE type(r) = 'HASCAZYME' OR toLower(type(r)) CONTAINS 'cazyme'
                    RETURN DISTINCT type(r) as relationship_type, labels(target) as target_labels, count(*) as count
                    ORDER BY count DESC
                    LIMIT 10
                """)
                relationships = list(result)
                print(f"Found {len(relationships)} relationship patterns:")
                for record in relationships:
                    print(f"  - {record['relationship_type']} -> {record['target_labels']} ({record['count']} times)")
            except Exception as e:
                print(f"❌ Error in Test 2: {e}")
            
            print()
            
            # Test 3: Count total CAZyme relationships
            print("=== Test 3: Counting total CAZyme relationships ===")
            try:
                result = session.run("""
                    MATCH (p:Protein)-[:HASCAZYME]->(ca)
                    RETURN count(ca) as total_cazymes, count(DISTINCT ca) as unique_cazymes
                """)
                for record in result:
                    print(f"Total CAZyme relationships: {record['total_cazymes']}")
                    print(f"Unique CAZyme nodes: {record['unique_cazymes']}")
            except Exception as e:
                print(f"❌ Error in Test 3: {e}")
            
            print()
            
            # Test 4: Sample CAZyme nodes to see their structure
            print("=== Test 4: Sampling CAZyme node structure ===")
            try:
                result = session.run("""
                    MATCH (p:Protein)-[:HASCAZYME]->(ca)
                    RETURN labels(ca) as node_labels, keys(ca) as properties, ca.id as sample_id
                    LIMIT 3
                """)
                samples = list(result)
                print(f"Found {len(samples)} sample CAZyme nodes:")
                for record in samples:
                    print(f"  - Labels: {record['node_labels']}")
                    print(f"    Properties: {record['properties']}")
                    print(f"    Sample ID: {record['sample_id']}")
                    print()
            except Exception as e:
                print(f"❌ Error in Test 4: {e}")
            
            # Test 5: Test current incorrect pattern
            print("=== Test 5: Testing current (incorrect) pattern ===")
            try:
                result = session.run("""
                    MATCH (p:Protein)-[:HASCAZYME]->(ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily)
                    RETURN count(*) as count_with_current_pattern
                """)
                for record in result:
                    print(f"Results with current pattern: {record['count_with_current_pattern']}")
            except Exception as e:
                print(f"❌ Error in Test 5 (expected): {e}")
            
            print()
            
            # Test 6: Look for family relationships
            print("=== Test 6: Finding family relationship patterns ===")
            try:
                result = session.run("""
                    MATCH (ca)-[r]->(target)
                    WHERE ca.id STARTS WITH "cazyme:" AND target.id STARTS WITH "cazyme:family_"
                    RETURN DISTINCT type(r) as family_relationship, count(*) as count
                    ORDER BY count DESC
                """)
                family_rels = list(result)
                print(f"Found {len(family_rels)} family relationship types:")
                for record in family_rels:
                    print(f"  - {record['family_relationship']}: {record['count']} times")
            except Exception as e:
                print(f"❌ Error in Test 6: {e}")
                
    finally:
        driver.close()

if __name__ == "__main__":
    test_cazyme_patterns()