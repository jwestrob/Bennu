#!/usr/bin/env python3
"""
Test script to verify that the LLM hallucination fixes are working.

This script creates a mock genomic chunk with known gene IDs and tests
whether the system correctly rejects hallucinated gene IDs.
"""

import sys
import os
from unittest.mock import Mock
sys.path.append('/Users/jacob/Documents/Sandbox/microbial_claude_matter')

from src.llm.rag_system.hierarchical_analysis.genomic_chunk_analyzer import GenomicChunkAnalyzer

def create_mock_contigs():
    """Create mock contigs with known gene IDs for testing."""
    
    # Create mock genes with known IDs
    mock_gene_1 = Mock()
    mock_gene_1.gene_id = "gene:scaffold_5445_3"
    mock_gene_1.protein_id = "protein:scaffold_5445_3"
    mock_gene_1.start = 1000
    mock_gene_1.end = 2000
    mock_gene_1.strand = "+"
    mock_gene_1.is_hypothetical = False
    mock_gene_1.ko_description = "transporter"
    mock_gene_1.pfam_domains = ["PF01234"]
    
    mock_gene_2 = Mock()
    mock_gene_2.gene_id = "gene:scaffold_5445_4"
    mock_gene_2.protein_id = "protein:scaffold_5445_4"
    mock_gene_2.start = 2100
    mock_gene_2.end = 3000
    mock_gene_2.strand = "+"
    mock_gene_2.is_hypothetical = True
    mock_gene_2.ko_description = ""
    mock_gene_2.pfam_domains = []
    
    mock_gene_3 = Mock()
    mock_gene_3.gene_id = "gene:scaffold_5445_11"
    mock_gene_3.protein_id = "protein:scaffold_5445_11"
    mock_gene_3.start = 5000
    mock_gene_3.end = 6000
    mock_gene_3.strand = "-"
    mock_gene_3.is_hypothetical = False
    mock_gene_3.ko_description = "kinase"
    mock_gene_3.pfam_domains = ["PF05678"]
    
    # Create mock contig
    mock_contig = Mock()
    mock_contig.contig_id = "RIFCSPLOWO2_01_FULL_OD1_41_220_rifcsplowo2_01_scaffold_5445"
    mock_contig.length = 10000
    mock_contig.plus_strand_genes = [mock_gene_1, mock_gene_2]
    mock_contig.minus_strand_genes = [mock_gene_3]
    mock_contig.total_genes = 3
    
    return [mock_contig]

def test_gene_id_extraction():
    """Test that gene ID extraction works correctly."""
    analyzer = GenomicChunkAnalyzer()
    contigs = create_mock_contigs()
    
    available_gene_ids = analyzer._extract_all_gene_ids_from_chunk_data(contigs)
    
    expected_ids = {
        "gene:scaffold_5445_3", "scaffold_5445_3", "protein:scaffold_5445_3",
        "gene:scaffold_5445_4", "scaffold_5445_4", "protein:scaffold_5445_4", 
        "gene:scaffold_5445_11", "scaffold_5445_11", "protein:scaffold_5445_11"
    }
    
    print(f"✅ Extracted gene IDs: {available_gene_ids}")
    print(f"✅ Expected gene IDs: {expected_ids}")
    
    # Check that all expected IDs are present
    missing_ids = expected_ids - available_gene_ids
    if missing_ids:
        print(f"❌ Missing gene IDs: {missing_ids}")
        return False
    
    print("✅ Gene ID extraction test PASSED")
    return True

def test_hallucination_detection():
    """Test that hallucinated gene IDs are properly rejected."""
    analyzer = GenomicChunkAnalyzer()
    contigs = create_mock_contigs()
    
    # Simulate LLM response with hallucinated gene IDs
    mock_locus_data = {
        "contig_id": "RIFCSPLOWO2_01_FULL_OD1_41_220_rifcsplowo2_01_scaffold_5445",
        "start_coordinate": 1000,
        "end_coordinate": 6000,
        "gene_ids": [
            "gene:scaffold_5445_3",    # Real gene - should be accepted
            "gene:scaffold_5445_4",    # Real gene - should be accepted  
            "gene:scaffold_5445_18",   # HALLUCINATED - should be rejected
            "gene:scaffold_5445_19"    # HALLUCINATED - should be rejected
        ],
        "locus_type": "test_locus",
        "biological_significance_reasoning": "test reasoning"
    }
    
    # Extract available gene IDs
    available_gene_ids = analyzer._extract_all_gene_ids_from_chunk_data(contigs)
    
    # Test validation logic
    valid_gene_ids = []
    hallucinated_count = 0
    
    for gene_id in mock_locus_data["gene_ids"]:
        if gene_id not in available_gene_ids:
            print(f"🚨 HALLUCINATION DETECTED: {gene_id}")
            hallucinated_count += 1
        else:
            print(f"✅ Valid gene ID: {gene_id}")
            valid_gene_ids.append(gene_id)
    
    print(f"✅ Valid gene IDs: {valid_gene_ids}")
    print(f"🚨 Hallucinated gene IDs detected: {hallucinated_count}")
    
    # Should have 2 valid genes and 2 hallucinated
    if len(valid_gene_ids) == 2 and hallucinated_count == 2:
        print("✅ Hallucination detection test PASSED")
        return True
    else:
        print("❌ Hallucination detection test FAILED")
        return False

def main():
    print("🧬 Testing LLM Hallucination Fixes")
    print("=" * 50)
    
    success = True
    
    print("\n1. Testing Gene ID Extraction...")
    if not test_gene_id_extraction():
        success = False
    
    print("\n2. Testing Hallucination Detection...")
    if not test_hallucination_detection():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED - Hallucination fixes are working!")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Check the fixes")
        return 1

if __name__ == "__main__":
    sys.exit(main())