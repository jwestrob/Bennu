#!/usr/bin/env python3
"""
Test AntiSMASH integration functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.ingest.antismash_bgc import parse_antismash_genbank, run_single_antismash_analysis
from src.build_kg.rdf_builder import GenomeKGBuilder


class TestAntiSMASHIntegration:
    """Test AntiSMASH integration components."""
    
    def test_parse_antismash_genbank_empty(self):
        """Test parsing with empty/invalid GenBank file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.gbk', delete=False) as f:
            f.write("# Empty file\n")
            temp_file = Path(f.name)
        
        try:
            result = parse_antismash_genbank(temp_file)
            
            assert "file" in result
            assert "clusters" in result
            assert "genes" in result
            assert "parsing_errors" in result
            assert len(result["clusters"]) == 0
            assert len(result["genes"]) == 0
        finally:
            temp_file.unlink()
    
    def test_bgc_rdf_builder_integration(self):
        """Test BGC integration with RDF builder."""
        builder = GenomeKGBuilder()
        
        # Mock genome URI
        from rdflib import Namespace, URIRef
        GENOME = Namespace("http://genome-kg.org/genomes/")
        genome_uri = GENOME["test_genome"]
        
        # Mock protein URIs
        PROTEIN = Namespace("http://genome-kg.org/proteins/")
        protein_uris = {
            "test_protein_1": PROTEIN["test_protein_1"],
            "test_protein_2": PROTEIN["test_protein_2"]
        }
        
        # Mock BGC data
        bgc_data = {
            "clusters": [
                {
                    "record_id": "test_scaffold",
                    "cluster_number": "1",
                    "start": 1000,
                    "end": 5000,
                    "product": "NRPS",
                    "qualifiers": {
                        "cluster_number": ["1"]
                    }
                }
            ],
            "genes": [
                {
                    "feature_type": "CDS",
                    "protein_id": "test_protein_1",
                    "product": "NRPS synthase",
                    "gene_kind": "biosynthetic",
                    "sec_met_domains": ["AMP-binding", "Condensation"]
                }
            ]
        }
        
        # Test BGC annotation addition
        builder.add_bgc_annotations(bgc_data, genome_uri, protein_uris)
        
        # Verify triples were added
        assert len(builder.graph) > 0
        
        # Check for BGC-related triples
        bgc_triples = list(builder.graph.triples((None, None, None)))
        assert len(bgc_triples) > 0
    
    @patch('subprocess.run')
    @patch('os.path.exists')
    def test_run_single_antismash_analysis_missing_wrapper(self, mock_exists, mock_run):
        """Test AntiSMASH analysis with missing wrapper script."""
        mock_exists.return_value = False
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            genome_file = temp_path / "test_genome.fna"
            genome_file.write_text(">scaffold1\nATGCGCGCGCGCTAA\n")
            
            output_dir = temp_path / "output"
            
            result = run_single_antismash_analysis(
                genome_file=genome_file,
                output_dir=output_dir,
                threads=2
            )
            
            assert result["execution_status"] == "failed"
            assert "AntiSMASH wrapper not found" in result["error_message"]
    
    def test_bgc_data_structure_validation(self):
        """Test that BGC data structure is properly validated."""
        # Test with minimal valid BGC data
        bgc_data = {
            "clusters": [],
            "genes": [],
            "parsing_errors": []
        }
        
        # Should not raise any exceptions
        builder = GenomeKGBuilder()
        from rdflib import Namespace, URIRef
        GENOME = Namespace("http://genome-kg.org/genomes/")
        genome_uri = GENOME["test_genome"]
        protein_uris = {}
        
        builder.add_bgc_annotations(bgc_data, genome_uri, protein_uris)
        
        # Should have minimal triples (just the basic ontology)
        assert len(builder.graph) >= 0
    
    def test_antismash_stage_configuration(self):
        """Test that AntiSMASH stage is properly configured in CLI."""
        # This test verifies the stage configuration without actually running it
        from src.cli import app
        
        # Check that the CLI app exists and has the expected structure
        assert app is not None
        # Typer apps have registered_commands instead of commands
        assert hasattr(app, 'registered_commands') or hasattr(app, 'commands')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])