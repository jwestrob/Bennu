#!/usr/bin/env python3
"""
Tests for dbCAN CAZyme integration.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.ingest.dbcan_cazyme import (
    CAZymeAnnotation, CAZymeResult, parse_dbcan_overview, 
    get_cazyme_family_type, run_dbcan_analysis
)
from src.build_kg.rdf_builder import GenomeKGBuilder
from rdflib import Graph, Namespace


class TestDbcanIntegration(unittest.TestCase):
    """Test dbCAN CAZyme annotation integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Sample CAZyme annotation data
        self.sample_annotation = CAZymeAnnotation(
            protein_id="test_protein_1",
            cazyme_family="GH13",
            family_type="GH",
            evalue=1e-20,
            coverage=0.85,
            start_pos=1,
            end_pos=295,
            hmm_length=295,
            ec_number="3.2.1.1"
        )
        
        self.sample_result = CAZymeResult(
            genome_id="test_genome",
            total_proteins=1000,
            cazyme_proteins=50,
            annotations=[self.sample_annotation],
            family_counts={"GH13": 1},
            processing_time=120.5
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_cazyme_annotation_model(self):
        """Test CAZyme annotation data model."""
        annotation = self.sample_annotation
        
        self.assertEqual(annotation.protein_id, "test_protein_1")
        self.assertEqual(annotation.cazyme_family, "GH13")
        self.assertEqual(annotation.family_type, "GH")
        self.assertEqual(annotation.evalue, 1e-20)
        self.assertEqual(annotation.coverage, 0.85)
        self.assertEqual(annotation.ec_number, "3.2.1.1")
    
    def test_cazyme_result_model(self):
        """Test CAZyme result data model."""
        result = self.sample_result
        
        self.assertEqual(result.genome_id, "test_genome")
        self.assertEqual(result.total_proteins, 1000)
        self.assertEqual(result.cazyme_proteins, 50)
        self.assertEqual(len(result.annotations), 1)
        self.assertEqual(result.family_counts["GH13"], 1)
        self.assertAlmostEqual(result.processing_time, 120.5)
    
    def test_get_cazyme_family_type(self):
        """Test CAZyme family type classification."""
        test_cases = [
            ("GH13", "GH"),
            ("GT2", "GT"),
            ("PL1", "PL"),
            ("CE4", "CE"),
            ("AA9", "AA"),
            ("CBM20", "CBM"),
            ("Unknown_family", "Unknown")
        ]
        
        for family, expected_type in test_cases:
            with self.subTest(family=family):
                self.assertEqual(get_cazyme_family_type(family), expected_type)
    
    def test_parse_dbcan_overview(self):
        """Test parsing of dbCAN overview.txt file."""
        # Create sample overview file
        overview_content = """Gene ID\tEC#\tHMMER\teCAMI\tDIAMOND\t#ofTools
protein_1\t3.2.1.1\tGH13(1-295)\tGH13\t-\t2
protein_2\t-\tGT2(50-300)\tGT2\t-\t2
protein_3\t-\t-\t-\t-\t0"""
        
        overview_file = self.test_dir / "overview.txt"
        with open(overview_file, 'w') as f:
            f.write(overview_content)
        
        annotations = parse_dbcan_overview(overview_file)
        
        self.assertEqual(len(annotations), 2)  # protein_3 has no annotations
        
        # Check first annotation
        ann1 = annotations[0]
        self.assertEqual(ann1.protein_id, "protein_1")
        self.assertEqual(ann1.cazyme_family, "GH13")
        self.assertEqual(ann1.family_type, "GH")
        self.assertEqual(ann1.start_pos, 1)
        self.assertEqual(ann1.end_pos, 295)
        self.assertEqual(ann1.ec_number, "3.2.1.1")
        
        # Check second annotation
        ann2 = annotations[1]
        self.assertEqual(ann2.protein_id, "protein_2")
        self.assertEqual(ann2.cazyme_family, "GT2")
        self.assertEqual(ann2.family_type, "GT")
        self.assertEqual(ann2.start_pos, 50)
        self.assertEqual(ann2.end_pos, 300)
        self.assertIsNone(ann2.ec_number)
    
    @patch('src.ingest.dbcan_cazyme.subprocess.run')
    def test_run_dbcan_analysis_success(self, mock_subprocess):
        """Test successful dbCAN analysis execution."""
        # Mock successful subprocess execution
        mock_subprocess.return_value = MagicMock(returncode=0, stderr="")
        
        # Create test protein file
        protein_file = self.test_dir / "test_genome.faa"
        with open(protein_file, 'w') as f:
            f.write(">protein_1\nMKLLILGALLGAAVAQAAQE\n>protein_2\nMKLLILGALLGAAVAQAAQE\n")
        
        # Create mock overview file
        output_dir = self.test_dir / "output"
        output_dir.mkdir()
        genome_output_dir = output_dir / "test_genome"
        genome_output_dir.mkdir()
        
        overview_file = genome_output_dir / "test_genome.overview.txt"
        with open(overview_file, 'w') as f:
            f.write("Gene ID\tEC#\tHMMER\teCAMI\tDIAMOND\t#ofTools\n")
            f.write("protein_1\t3.2.1.1\tGH13(1-295)\tGH13\t-\t2\n")
        
        result = run_dbcan_analysis(protein_file, output_dir)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.genome_id, "test_genome")
        self.assertEqual(result.total_proteins, 2)
        self.assertEqual(result.cazyme_proteins, 1)
        self.assertEqual(len(result.annotations), 1)
    
    @patch('src.ingest.dbcan_cazyme.subprocess.run')
    def test_run_dbcan_analysis_failure(self, mock_subprocess):
        """Test dbCAN analysis failure handling."""
        # Mock failed subprocess execution
        mock_subprocess.return_value = MagicMock(returncode=1, stderr="dbCAN error")
        
        protein_file = self.test_dir / "test_genome.faa"
        with open(protein_file, 'w') as f:
            f.write(">protein_1\nMKLLILGALLGAAVAQAAQE\n")
        
        result = run_dbcan_analysis(protein_file, self.test_dir / "output")
        
        self.assertIsNone(result)
    
    def test_rdf_builder_cazyme_integration(self):
        """Test CAZyme annotation integration into RDF builder."""
        builder = GenomeKGBuilder()
        
        # Create test data
        cazyme_data = {
            "annotations": [
                {
                    "protein_id": "test_protein_1",
                    "cazyme_family": "GH13",
                    "family_type": "GH",
                    "evalue": 1e-20,
                    "coverage": 0.85,
                    "start_pos": 1,
                    "end_pos": 295,
                    "ec_number": "3.2.1.1"
                }
            ]
        }
        
        # Create mock genome and protein URIs
        KG = Namespace("http://genome-kg.org/ontology/")
        GENOME = Namespace("http://genome-kg.org/genomes/")
        PROTEIN = Namespace("http://genome-kg.org/proteins/")
        
        genome_uri = GENOME["test_genome"]
        protein_uris = {"test_protein_1": PROTEIN["test_protein_1"]}
        
        # Add CAZyme annotations
        builder.add_cazyme_annotations(cazyme_data, genome_uri, protein_uris)
        
        # Verify RDF triples were created
        graph = builder.graph
        
        # Check that CAZyme annotation was created
        cazyme_annotations = list(graph.subjects(None, KG.CAZymeAnnotation))
        self.assertGreater(len(cazyme_annotations), 0)
        
        # Check that protein has CAZyme annotation
        protein_cazyme_relations = list(graph.triples((PROTEIN["test_protein_1"], KG.hasCAZyme, None)))
        self.assertGreater(len(protein_cazyme_relations), 0)
        
        # Check that CAZyme family was created
        cazyme_families = list(graph.subjects(None, KG.CAZymeFamily))
        self.assertGreater(len(cazyme_families), 0)


if __name__ == '__main__':
    unittest.main()