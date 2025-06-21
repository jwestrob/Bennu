#!/usr/bin/env python3
"""
Tests for annotation processors module.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import os

from src.build_kg.annotation_processors import (
    AnnotationProcessor, 
    PfamProcessor, 
    KofamProcessor,
    process_astra_results
)


class TestAnnotationProcessor:
    """Test base AnnotationProcessor class."""

    def test_init(self):
        """Test processor initialization."""
        processor = AnnotationProcessor("TEST", keep_multiple=True)
        assert processor.annotation_type == "TEST"
        assert processor.keep_multiple == True

    def test_load_hits_valid_file(self, temp_dir):
        """Test loading valid TSV hits file."""
        # Create test TSV file
        test_data = pd.DataFrame({
            'sequence_id': ['protein_1', 'protein_2'],
            'hmm_name': ['PF00001', 'PF00002'],
            'bitscore': [100.5, 85.3],
            'evalue': [1e-20, 1e-15]
        })
        
        hits_file = temp_dir / "test_hits.tsv"
        test_data.to_csv(hits_file, sep='\t', index=False)
        
        processor = AnnotationProcessor("TEST")
        result = processor.load_hits(hits_file)
        
        assert len(result) == 2
        assert 'sequence_id' in result.columns
        assert 'hmm_name' in result.columns
        assert result.iloc[0]['sequence_id'] == 'protein_1'

    def test_load_hits_missing_file(self, temp_dir):
        """Test loading non-existent file returns empty DataFrame."""
        processor = AnnotationProcessor("TEST")
        missing_file = temp_dir / "missing.tsv"
        
        result = processor.load_hits(missing_file)
        
        assert result.empty
        assert isinstance(result, pd.DataFrame)

    def test_filter_all_significant(self):
        """Test filtering hits by E-value significance."""
        test_data = pd.DataFrame({
            'sequence_id': ['protein_1', 'protein_2', 'protein_3'],
            'hmm_name': ['PF00001', 'PF00002', 'PF00003'],
            'bitscore': [100.5, 85.3, 45.2],
            'evalue': [1e-20, 1e-15, 1e-3]  # Third one should be filtered out
        })
        
        processor = AnnotationProcessor("TEST")
        result = processor.filter_all_significant(test_data, evalue_threshold=1e-5)
        
        assert len(result) == 2
        assert all(result['evalue'] <= 1e-5)

    def test_select_best_per_protein(self):
        """Test selecting best hit per protein."""
        test_data = pd.DataFrame({
            'sequence_id': ['protein_1', 'protein_1', 'protein_2'],
            'hmm_name': ['PF00001', 'PF00002', 'PF00003'],
            'bitscore': [100.5, 120.3, 85.3],  # Second hit should be selected for protein_1
            'evalue': [1e-20, 1e-25, 1e-15]
        })
        
        processor = AnnotationProcessor("TEST")
        result = processor.select_best_per_protein(test_data)
        
        assert len(result) == 2
        # Check that best hit for protein_1 was selected
        protein_1_hit = result[result['sequence_id'] == 'protein_1'].iloc[0]
        assert protein_1_hit['hmm_name'] == 'PF00002'
        assert protein_1_hit['bitscore'] == 120.3

    def test_process_hits_keep_multiple(self):
        """Test process_hits with keep_multiple=True."""
        test_data = pd.DataFrame({
            'sequence_id': ['protein_1', 'protein_2'],
            'hmm_name': ['PF00001', 'PF00002'],
            'bitscore': [100.5, 85.3],
            'evalue': [1e-20, 1e-15]
        })
        
        processor = AnnotationProcessor("TEST", keep_multiple=True)
        result = processor.process_hits(test_data)
        
        assert len(result) == 2  # All kept since they're significant

    def test_process_hits_best_only(self):
        """Test process_hits with keep_multiple=False."""
        test_data = pd.DataFrame({
            'sequence_id': ['protein_1', 'protein_1', 'protein_2'],
            'hmm_name': ['PF00001', 'PF00002', 'PF00003'],
            'bitscore': [100.5, 120.3, 85.3],
            'evalue': [1e-20, 1e-25, 1e-15]
        })
        
        processor = AnnotationProcessor("TEST", keep_multiple=False)
        result = processor.process_hits(test_data)
        
        assert len(result) == 2  # Best hit per protein


class TestPfamProcessor:
    """Test PFAM-specific processor."""

    def test_init(self):
        """Test PFAM processor initialization."""
        processor = PfamProcessor()
        assert processor.annotation_type == "PFAM"
        assert processor.keep_multiple == True

    def test_create_domain_entities(self):
        """Test creating PFAM domain entities."""
        test_data = pd.DataFrame({
            'sequence_id': ['protein_1', 'protein_2'],
            'hmm_name': ['PF00001', 'PF00002'],
            'env_from': [10, 25],
            'env_to': [50, 75],
            'bitscore': [100.5, 85.3],
            'evalue': [1e-20, 1e-15],
            'dom_bitscore': [98.2, 83.1]
        })
        
        processor = PfamProcessor()
        result = processor.create_domain_entities(test_data)
        
        assert len(result) == 2
        
        domain1 = result[0]
        assert domain1['protein_id'] == 'protein_1'
        assert domain1['pfam_id'] == 'PF00001'
        assert domain1['start_pos'] == 10
        assert domain1['end_pos'] == 50
        assert domain1['bitscore'] == 100.5
        assert domain1['evalue'] == 1e-20
        assert 'domain_id' in domain1
        assert 'PF00001' in domain1['domain_id']

    def test_create_domain_entities_missing_dom_bitscore(self):
        """Test creating domain entities when dom_bitscore is missing."""
        test_data = pd.DataFrame({
            'sequence_id': ['protein_1'],
            'hmm_name': ['PF00001'],
            'env_from': [10],
            'env_to': [50],
            'bitscore': [100.5],
            'evalue': [1e-20]
            # Missing dom_bitscore column
        })
        
        processor = PfamProcessor()
        result = processor.create_domain_entities(test_data)
        
        assert len(result) == 1
        assert result[0]['dom_bitscore'] == 100.5  # Falls back to bitscore


class TestKofamProcessor:
    """Test KOFAM-specific processor."""

    def test_init(self):
        """Test KOFAM processor initialization."""
        processor = KofamProcessor()
        assert processor.annotation_type == "KOFAM"
        assert processor.keep_multiple == False

    def test_create_functional_entities(self):
        """Test creating KOFAM functional entities."""
        test_data = pd.DataFrame({
            'sequence_id': ['protein_1', 'protein_2'],
            'hmm_name': ['K00001', 'K00002'],
            'bitscore': [200.5, 150.3],
            'evalue': [1e-30, 1e-8]
        })
        
        processor = KofamProcessor()
        result = processor.create_functional_entities(test_data)
        
        assert len(result) == 2
        
        func1 = result[0]
        assert func1['protein_id'] == 'protein_1'
        assert func1['ko_id'] == 'K00001'
        assert func1['bitscore'] == 200.5
        assert func1['evalue'] == 1e-30
        assert func1['confidence'] == 'high'  # E-value <= 1e-10
        assert 'annotation_id' in func1
        assert 'K00001' in func1['annotation_id']
        
        func2 = result[1]
        assert func2['confidence'] == 'medium'  # E-value > 1e-10

    def test_confidence_levels(self):
        """Test confidence level assignment."""
        test_data = pd.DataFrame({
            'sequence_id': ['protein_1', 'protein_2', 'protein_3'],
            'hmm_name': ['K00001', 'K00002', 'K00003'],
            'bitscore': [200.5, 150.3, 100.1],
            'evalue': [1e-15, 1e-10, 1e-5]  # high, high, medium
        })
        
        processor = KofamProcessor()
        result = processor.create_functional_entities(test_data)
        
        assert result[0]['confidence'] == 'high'
        assert result[1]['confidence'] == 'high'  # exactly at threshold
        assert result[2]['confidence'] == 'medium'


class TestProcessAstraResults:
    """Test complete astra results processing."""

    def test_process_astra_results_complete(self, temp_dir):
        """Test processing complete astra results with both PFAM and KOFAM."""
        # Create mock astra output directory structure
        astra_dir = temp_dir / "astra_output"
        pfam_dir = astra_dir / "pfam_results"
        kofam_dir = astra_dir / "kofam_results"
        pfam_dir.mkdir(parents=True)
        kofam_dir.mkdir(parents=True)
        
        # Create mock PFAM results
        pfam_data = pd.DataFrame({
            'sequence_id': ['protein_1', 'protein_1', 'protein_2'],
            'hmm_name': ['PF00001', 'PF00002', 'PF00003'],
            'env_from': [10, 60, 25],
            'env_to': [50, 100, 75],
            'bitscore': [100.5, 85.3, 120.2],
            'evalue': [1e-20, 1e-15, 1e-25],
            'dom_bitscore': [98.2, 83.1, 118.5]
        })
        pfam_file = pfam_dir / "PFAM_hits_df.tsv"
        pfam_data.to_csv(pfam_file, sep='\t', index=False)
        
        # Create mock KOFAM results
        kofam_data = pd.DataFrame({
            'sequence_id': ['protein_1', 'protein_1', 'protein_2'],
            'hmm_name': ['K00001', 'K00002', 'K00003'],
            'bitscore': [200.5, 180.3, 150.1],
            'evalue': [1e-30, 1e-25, 1e-20]
        })
        kofam_file = kofam_dir / "KOFAM_hits_df.tsv"
        kofam_data.to_csv(kofam_file, sep='\t', index=False)
        
        # Process results
        result = process_astra_results(astra_dir)
        
        # Check structure
        assert 'pfam_domains' in result
        assert 'kofam_functions' in result
        assert 'processing_stats' in result
        
        # Check PFAM results (should keep all 3 domains)
        assert len(result['pfam_domains']) == 3
        pfam_domain = result['pfam_domains'][0]
        assert 'domain_id' in pfam_domain
        assert 'protein_id' in pfam_domain
        assert 'pfam_id' in pfam_domain
        
        # Check KOFAM results (should keep best hit per protein = 2 functions)
        assert len(result['kofam_functions']) == 2
        kofam_func = result['kofam_functions'][0]
        assert 'annotation_id' in kofam_func
        assert 'protein_id' in kofam_func
        assert 'ko_id' in kofam_func
        
        # Check processing stats
        stats = result['processing_stats']
        assert 'pfam_total_hits' in stats
        assert 'pfam_significant_hits' in stats
        assert 'kofam_total_hits' in stats
        assert 'kofam_best_hits' in stats
        assert 'kofam_proteins_annotated' in stats
        
        assert stats['pfam_total_hits'] == 3
        assert stats['pfam_significant_hits'] == 3
        assert stats['kofam_total_hits'] == 3
        assert stats['kofam_best_hits'] == 2
        assert stats['kofam_proteins_annotated'] == 2

    def test_process_astra_results_pfam_only(self, temp_dir):
        """Test processing when only PFAM results exist."""
        astra_dir = temp_dir / "astra_output"
        pfam_dir = astra_dir / "pfam_results"
        pfam_dir.mkdir(parents=True)
        
        # Create only PFAM results
        pfam_data = pd.DataFrame({
            'sequence_id': ['protein_1'],
            'hmm_name': ['PF00001'],
            'env_from': [10],
            'env_to': [50],
            'bitscore': [100.5],
            'evalue': [1e-20],
            'dom_bitscore': [98.2]
        })
        pfam_file = pfam_dir / "PFAM_hits_df.tsv"
        pfam_data.to_csv(pfam_file, sep='\t', index=False)
        
        result = process_astra_results(astra_dir)
        
        assert len(result['pfam_domains']) == 1
        assert len(result['kofam_functions']) == 0
        assert 'pfam_total_hits' in result['processing_stats']
        assert 'kofam_total_hits' not in result['processing_stats']

    def test_process_astra_results_empty_directory(self, temp_dir):
        """Test processing empty astra output directory."""
        astra_dir = temp_dir / "empty_astra"
        astra_dir.mkdir()
        
        result = process_astra_results(astra_dir)
        
        assert len(result['pfam_domains']) == 0
        assert len(result['kofam_functions']) == 0
        assert result['processing_stats'] == {}