#!/usr/bin/env python3
"""Debug RDF loader to see why descriptions are missing."""

import rdflib
from rich.console import Console

console = Console()

def debug_ggdef_in_rdf():
    """Debug what GGDEF properties exist in our RDF file."""
    print("=== DEBUGGING GGDEF in RDF ===")
    
    # Load RDF
    g = rdflib.Graph()
    g.parse("data/stage05_kg/knowledge_graph.ttl", format="turtle")
    
    # Define namespaces
    pfam_ns = rdflib.Namespace("http://pfam.xfam.org/family/")
    kg_ns = rdflib.Namespace("http://genome-kg.org/ontology/")
    rdfs_ns = rdflib.Namespace("http://www.w3.org/2000/01/rdf-schema#")
    
    ggdef_uri = pfam_ns["GGDEF"]
    
    print(f"Looking for: {ggdef_uri}")
    
    # Find all triples with GGDEF as subject
    ggdef_triples = []
    for subj, pred, obj in g:
        if str(subj) == str(ggdef_uri):
            ggdef_triples.append((subj, pred, obj))
    
    print(f"Found {len(ggdef_triples)} triples with GGDEF as subject:")
    for subj, pred, obj in ggdef_triples:
        print(f"  {pred} -> {obj}")
    
    # Test specific properties
    description_prop = kg_ns["description"]
    label_prop = rdfs_ns["label"]
    
    print(f"\nChecking specific properties:")
    print(f"  kg:description: {list(g.objects(ggdef_uri, description_prop))}")
    print(f"  rdfs:label: {list(g.objects(ggdef_uri, label_prop))}")
    
    # Test property conversion logic
    namespaces = {
        "kg": "http://genome-kg.org/ontology/"
    }
    
    def _uri_to_property(uri: str, namespaces) -> str:
        if uri.startswith(namespaces["kg"]):
            return uri.replace(namespaces["kg"], "")
        return uri.split("/")[-1].split("#")[-1]
    
    print(f"\nProperty name conversions:")
    for subj, pred, obj in ggdef_triples:
        prop_name = _uri_to_property(str(pred), namespaces)
        print(f"  {pred} -> '{prop_name}'")
        if isinstance(obj, rdflib.Literal):
            print(f"    Value: '{obj}' (type: {type(obj)})")

if __name__ == "__main__":
    debug_ggdef_in_rdf()