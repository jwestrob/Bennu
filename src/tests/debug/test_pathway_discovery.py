#!/usr/bin/env python3
"""
Test script to debug pathway-based protein discovery
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm.pathway_tools import pathway_based_protein_discovery, pathway_classifier

async def test_pathway_discovery():
    """Test the pathway-based protein discovery pipeline"""
    
    print("🔍 Testing pathway-based protein discovery pipeline")
    
    try:
        # Test 1: Pathway classification
        print("\n📊 Step 1: Testing pathway classification...")
        query = "Show me all proteins involved in central metabolism"
        classification_result = await pathway_classifier(query, max_pathways=3)
        
        print(f"Classification success: {classification_result['success']}")
        if classification_result['success']:
            print(f"Extracted terms: {classification_result['extracted_terms']}")
            print(f"Relevant pathways: {len(classification_result['relevant_pathways'])}")
            for pathway in classification_result['relevant_pathways']:
                print(f"  - {pathway['pathway_id']}: score={pathway['relevance_score']}, KOs={pathway['matched_kos']}")
        else:
            print(f"Classification error: {classification_result.get('error', 'Unknown error')}")
        
        # Test 2: Protein discovery with specific terms
        print("\n🧬 Step 2: Testing protein discovery with metabolism terms...")
        search_terms = ["metabolism", "glycolysis", "citrate", "pyruvate"]
        discovery_result = await pathway_based_protein_discovery(
            search_terms, 
            functional_category="central_metabolism",
            max_pathways=3,
            max_proteins_per_pathway=10
        )
        
        print(f"Discovery success: {discovery_result['success']}")
        if discovery_result['success']:
            print(f"Pathways analyzed: {discovery_result['pathways_analyzed']}")
            print(f"Total proteins found: {discovery_result['total_proteins']}")
            print(f"Pathway details:")
            for pathway in discovery_result['pathway_details']:
                print(f"  - {pathway['pathway_id']}: {pathway['proteins_found']} proteins, score={pathway['relevance_score']}")
            
            print(f"\nFirst 5 proteins found:")
            for protein in discovery_result['proteins_found'][:5]:
                print(f"  - {protein['protein_id']} -> {protein['ko_id']}: {protein['ko_description'][:50]}...")
        else:
            print(f"Discovery error: {discovery_result.get('error', 'Unknown error')}")
        
        # Test 3: Transport protein discovery
        print("\n🚛 Step 3: Testing transport protein discovery...")
        transport_terms = ["transport", "permease", "abc"]
        transport_result = await pathway_based_protein_discovery(
            transport_terms,
            functional_category="transport", 
            max_pathways=3,
            max_proteins_per_pathway=10
        )
        
        print(f"Transport discovery success: {transport_result['success']}")
        if transport_result['success']:
            print(f"Transport pathways analyzed: {transport_result['pathways_analyzed']}")
            print(f"Transport proteins found: {transport_result['total_proteins']}")
            
            print(f"\nFirst 5 transport proteins:")
            for protein in transport_result['proteins_found'][:5]:
                print(f"  - {protein['protein_id']} -> {protein['ko_id']}: {protein['ko_description'][:50]}...")
        else:
            print(f"Transport discovery error: {transport_result.get('error', 'Unknown error')}")
        
        print(f"\n🎉 Pathway discovery test complete!")
        
    except Exception as e:
        print(f"❌ Error in pathway discovery test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_pathway_discovery())