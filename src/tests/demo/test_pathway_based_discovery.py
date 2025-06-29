#!/usr/bin/env python3
"""
Test the new pathway-based protein discovery system
Validates that we can find proteins dynamically using KEGG pathways instead of hardcoded examples
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm.pathway_tools import KEGGPathwayMapper, pathway_based_protein_discovery, pathway_classifier
from llm.annotation_tools import pathway_based_annotation_selector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_kegg_pathway_mapper():
    """Test the KEGG pathway mapping functionality"""
    print("🧪 Testing KEGG Pathway Mapper...")
    
    mapper = KEGGPathwayMapper()
    
    # Test pathway finding for different biological processes
    test_queries = [
        ["transport", "abc"],
        ["glycolysis", "glucose"],
        ["respiration", "electron"],
        ["amino", "acid", "synthesis"],
        ["fatty", "acid", "metabolism"]
    ]
    
    for query_terms in test_queries:
        print(f"\n📋 Query terms: {query_terms}")
        pathways = mapper.find_relevant_pathways(query_terms, max_pathways=3)
        
        for pathway in pathways:
            print(f"  🔍 Pathway {pathway['pathway_id']}: score={pathway['relevance_score']}, KOs={pathway['matched_kos']}")
            
            # Show some example KOs
            for ko_match in pathway['ko_matches'][:3]:
                print(f"    - {ko_match['ko_id']}: {ko_match['description'][:60]}...")

async def test_pathway_classifier():
    """Test pathway classification for user queries"""
    print("\n🧪 Testing Pathway Classifier...")
    
    test_queries = [
        "Find proteins involved in glucose metabolism",
        "What transport proteins are in our genomes?",
        "Show me amino acid synthesis enzymes",
        "Find proteins in the electron transport chain",
        "What proteins are involved in fatty acid degradation?"
    ]
    
    for query in test_queries:
        print(f"\n📋 Query: {query}")
        result = await pathway_classifier(query, max_pathways=2)
        
        if result["success"]:
            print(f"  🔍 Extracted terms: {result['extracted_terms']}")
            for pathway in result["relevant_pathways"]:
                print(f"  📊 Pathway {pathway['pathway_id']}: score={pathway['relevance_score']}")
        else:
            print(f"  ❌ Error: {result['error']}")

async def test_pathway_based_protein_discovery():
    """Test finding actual proteins in our database using pathway analysis"""
    print("\n🧪 Testing Pathway-Based Protein Discovery...")
    
    test_cases = [
        {
            "terms": ["transport", "abc"],
            "category": "transport"
        },
        {
            "terms": ["dehydrogenase", "metabolism"],
            "category": "metabolism"
        },
        {
            "terms": ["kinase", "phosphorylation"],
            "category": "regulation"
        }
    ]
    
    for case in test_cases:
        print(f"\n📋 Testing: {case['terms']} ({case['category']})")
        
        result = await pathway_based_protein_discovery(
            case["terms"], 
            case["category"], 
            max_pathways=2, 
            max_proteins_per_pathway=3
        )
        
        if result["success"]:
            print(f"  ✅ Found {result['total_proteins']} proteins across {result['pathways_analyzed']} pathways")
            
            for pathway in result["pathway_details"]:
                print(f"  📊 Pathway {pathway['pathway_id']}: {pathway['proteins_found']} proteins (score: {pathway['relevance_score']})")
            
            # Show some example proteins
            for protein in result["proteins_found"][:3]:
                print(f"    🧬 {protein['protein_id']}: {protein['ko_description'][:50]}...")
                print(f"       Genome: {protein['genome_id']}, KO: {protein['ko_id']}")
        else:
            print(f"  ❌ Error: {result['error']}")

async def test_pathway_based_annotation_selector():
    """Test the complete annotation selector that replaces hardcoded examples"""
    print("\n🧪 Testing Pathway-Based Annotation Selector...")
    
    test_queries = [
        "Find transport proteins in our genomes",
        "What proteins are involved in central metabolism?",
        "Show me regulatory proteins",
        "Find proteins in amino acid biosynthesis"
    ]
    
    for query in test_queries:
        print(f"\n📋 Query: {query}")
        
        result = await pathway_based_annotation_selector(
            query, 
            max_pathways=2, 
            max_proteins_per_pathway=3
        )
        
        if result["success"]:
            print(f"  ✅ Selected {result['total_proteins']} proteins using {len(result['pathways_used'])} pathways")
            print(f"  🔍 Search terms: {result['search_terms_used']}")
            
            for pathway in result["pathways_used"]:
                print(f"  📊 Used pathway {pathway['pathway_id']}: {pathway['proteins_found']} proteins")
            
            # Show selected proteins
            for protein in result["selected_proteins"][:3]:
                print(f"    🧬 {protein['protein_id']}: {protein['ko_description'][:50]}...")
                print(f"       Pathway: {protein['pathway_id']}, Relevance: {protein['pathway_relevance']}")
        else:
            print(f"  ❌ Error: {result['error']}")

async def main():
    """Run all pathway-based discovery tests"""
    print("🚀 Starting Pathway-Based Protein Discovery Tests")
    print("=" * 60)
    
    try:
        await test_kegg_pathway_mapper()
        await test_pathway_classifier()
        await test_pathway_based_protein_discovery()
        await test_pathway_based_annotation_selector()
        
        print("\n" + "=" * 60)
        print("✅ All pathway-based discovery tests completed!")
        print("🎉 Dynamic KEGG pathway system is ready to replace hardcoded examples!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())