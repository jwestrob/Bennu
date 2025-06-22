#!/usr/bin/env python3
"""
Debug RDF relationship parsing to find ENCODEDBY bug.
"""

import rdflib
from pathlib import Path
from collections import defaultdict

def debug_rdf_relationships():
    """Debug RDF relationship parsing."""
    print("🔍 Debugging RDF Relationship Parsing")
    print("="*60)
    
    # Load RDF
    rdf_file = Path("data/stage05_kg/knowledge_graph.ttl")
    g = rdflib.Graph()
    g.parse(rdf_file, format="turtle")
    
    # Define namespaces (same as converter)
    namespaces = {
        "genome": "http://genome-kg.org/genomes/",
        "protein": "http://genome-kg.org/proteins/", 
        "gene": "http://genome-kg.org/genes/",
        "pfam": "http://pfam.xfam.org/family/",
        "ko": "http://www.genome.jp/kegg/ko/",
        "kg": "http://genome-kg.org/ontology/"
    }
    
    def uri_to_id(uri: str) -> str:
        """Convert URI to readable ID (same as converter)."""
        for prefix, namespace in namespaces.items():
            if uri.startswith(namespace):
                return uri.replace(namespace, "")
        return uri.split("/")[-1]
    
    def uri_to_property(uri: str) -> str:
        """Convert property URI to readable name (same as converter)."""
        if uri.startswith(namespaces["kg"]):
            return uri.replace(namespaces["kg"], "")
        return uri.split("/")[-1].split("#")[-1]
    
    # Track relationships
    relationships = []
    encodedby_relationships = []
    
    print("Parsing relationships...")
    for subj, pred, obj in g:
        if isinstance(obj, rdflib.URIRef):  # This is a relationship
            subj_id = uri_to_id(str(subj))
            pred_name = uri_to_property(str(pred))
            obj_id = uri_to_id(str(obj))
            
            relationships.append((subj_id, pred_name, obj_id))
            
            if pred_name == "encodedBy":
                encodedby_relationships.append((subj_id, pred_name, obj_id))
    
    print(f"Total relationships: {len(relationships)}")
    print(f"ENCODEDBY relationships: {len(encodedby_relationships)}")
    
    # Show sample ENCODEDBY relationships
    print("\nSample ENCODEDBY relationships (first 5):")
    for i, (subj, pred, obj) in enumerate(encodedby_relationships[:5], 1):
        print(f"  {i}. {subj} -[{pred}]-> {obj}")
        
        # Check if subject is protein and object is gene
        if subj == obj:
            print(f"     ❌ BUG: Subject and object are identical!")
        else:
            print(f"     ✅ Different: Protein -> Gene")
    
    # Check specific protein/gene pair
    test_protein = "PLM0_60_b1_sep16_scaffold_10001_curated_1"
    test_relationships = [r for r in encodedby_relationships if test_protein in r]
    print(f"\nRelationships for test protein {test_protein}:")
    for rel in test_relationships:
        print(f"  {rel}")

if __name__ == "__main__":
    debug_rdf_relationships()