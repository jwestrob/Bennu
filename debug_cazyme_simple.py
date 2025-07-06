#!/usr/bin/env python3
"""
Simple test to debug CAZyme query step by step.
"""

import subprocess
import sys

def test_simple_queries():
    """Test increasingly complex CAZyme queries to find the issue."""
    
    queries = [
        # Test 1: Just count CAZyme annotations
        ("Count CAZyme annotations", "MATCH (ca:Cazymeannotation) RETURN count(ca) as cazyme_count"),
        
        # Test 2: Just count CAZyme families  
        ("Count CAZyme families", "MATCH (cf:Cazymefamily) RETURN count(cf) as family_count"),
        
        # Test 3: Show sample CAZyme annotation properties
        ("Sample CAZyme annotation", "MATCH (ca:Cazymeannotation) RETURN ca.id, ca.cazymeType, ca.substrateSpecificity LIMIT 3"),
        
        # Test 4: Show sample CAZyme family properties
        ("Sample CAZyme family", "MATCH (cf:Cazymefamily) RETURN cf.id, cf.familyId, cf.cazymeType LIMIT 3"),
        
        # Test 5: Test the protein-to-annotation relationship
        ("Protein to CAZyme", "MATCH (p:Protein)-[:HASCAZYME]->(ca:Cazymeannotation) RETURN p.id, ca.id LIMIT 3"),
        
        # Test 6: Test the annotation-to-family relationship
        ("CAZyme to family", "MATCH (ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily) RETURN ca.id, cf.id LIMIT 3"),
        
        # Test 7: Full path
        ("Full path", "MATCH (p:Protein)-[:HASCAZYME]->(ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily) RETURN p.id, ca.id, cf.id LIMIT 3"),
    ]
    
    print("🧪 Testing CAZyme database structure step by step\n")
    
    for description, query in queries:
        print(f"=== {description} ===")
        print(f"Query: {query}")
        
        try:
            # Create a simple Cypher query file
            with open("temp_query.cypher", "w") as f:
                f.write(query)
            
            # Try to run it with cypher-shell (if available)
            result = subprocess.run([
                "cypher-shell", "-u", "neo4j", "-p", "password", 
                "--file", "temp_query.cypher"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("✅ Success:")
                print(result.stdout)
            else:
                print("❌ Failed:")
                print(result.stderr)
                
        except subprocess.TimeoutExpired:
            print("⏰ Query timed out")
        except FileNotFoundError:
            print("❌ cypher-shell not available")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    # Clean up
    try:
        import os
        os.remove("temp_query.cypher")
    except:
        pass

if __name__ == "__main__":
    test_simple_queries()