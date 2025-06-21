#!/usr/bin/env python3
"""
Integration tests for knowledge graph construction using real pipeline data.
"""

import pytest
import json
from pathlib import Path
import tempfile
import shutil

from src.build_kg.rdf_builder import build_knowledge_graph_from_pipeline
from src.build_kg.annotation_processors import process_astra_results


class TestKnowledgeGraphIntegration:
    """Integration tests using real pipeline data if available."""

    @pytest.fixture
    def pipeline_data_dir(self):
        """Get pipeline data directory if it exists."""
        data_dir = Path("data")
        if data_dir.exists():
            return data_dir
        return None

    @pytest.fixture
    def real_stage03_dir(self, pipeline_data_dir):
        """Real stage03 directory if available."""
        if pipeline_data_dir:
            stage03 = pipeline_data_dir / "stage03_prodigal"
            if stage03.exists():
                return stage03
        return None

    @pytest.fixture
    def real_stage04_dir(self, pipeline_data_dir):
        """Real stage04 directory if available."""
        if pipeline_data_dir:
            stage04 = pipeline_data_dir / "stage04_astra"
            if stage04.exists():
                return stage04
        return None

    @pytest.mark.integration
    def test_build_kg_with_real_data(self, real_stage03_dir, real_stage04_dir, temp_dir):
        """Test knowledge graph building with real pipeline output."""
        if real_stage03_dir is None or real_stage04_dir is None:
            pytest.skip("Real pipeline data not available")
        
        # Use temporary output directory
        output_dir = temp_dir / "kg_output"
        
        # Build knowledge graph
        result = build_knowledge_graph_from_pipeline(
            real_stage03_dir, 
            real_stage04_dir, 
            output_dir
        )
        
        # Verify basic structure
        assert 'total_triples' in result
        assert 'genomes_processed' in result
        assert 'proteins_annotated' in result
        assert 'pfam_domains' in result
        assert 'kofam_functions' in result
        assert 'output_files' in result
        
        # Check that we actually processed some data
        assert result['total_triples'] > 0
        assert result['genomes_processed'] > 0
        assert result['proteins_annotated'] > 0
        
        # Check output files exist and are non-empty
        kg_file = Path(result['output_files']['knowledge_graph'])
        assert kg_file.exists()
        assert kg_file.stat().st_size > 1000  # Should be substantial
        
        # Verify it's valid Turtle format
        with open(kg_file) as f:
            content = f.read()
            assert '@prefix' in content
            assert 'kg:' in content
            assert 'genome:' in content
            assert 'protein:' in content

    @pytest.mark.integration  
    def test_annotation_processors_with_real_data(self, real_stage04_dir):
        """Test annotation processors with real astra output."""
        if real_stage04_dir is None:
            pytest.skip("Real astra data not available")
        
        # Process real astra results
        result = process_astra_results(real_stage04_dir)
        
        # Verify structure
        assert 'pfam_domains' in result
        assert 'kofam_functions' in result
        assert 'processing_stats' in result
        
        # Check that we got some results
        if (real_stage04_dir / "pfam_results" / "PFAM_hits_df.tsv").exists():
            assert len(result['pfam_domains']) > 0
            
            # Verify domain structure
            domain = result['pfam_domains'][0]
            required_fields = ['domain_id', 'protein_id', 'pfam_id', 'start_pos', 
                             'end_pos', 'bitscore', 'evalue']
            for field in required_fields:
                assert field in domain
                
        if (real_stage04_dir / "kofam_results" / "KOFAM_hits_df.tsv").exists():
            assert len(result['kofam_functions']) > 0
            
            # Verify function structure
            function = result['kofam_functions'][0]
            required_fields = ['annotation_id', 'protein_id', 'ko_id', 'bitscore', 
                             'evalue', 'confidence']
            for field in required_fields:
                assert field in function

    @pytest.mark.integration
    def test_knowledge_graph_completeness(self, real_stage03_dir, real_stage04_dir, temp_dir):
        """Test that knowledge graph contains expected entity relationships."""
        if real_stage03_dir is None or real_stage04_dir is None:
            pytest.skip("Real pipeline data not available")
        
        output_dir = temp_dir / "kg_completeness_test"
        
        result = build_knowledge_graph_from_pipeline(
            real_stage03_dir, 
            real_stage04_dir, 
            output_dir
        )
        
        # Load the generated knowledge graph
        from rdflib import Graph
        graph = Graph()
        kg_file = Path(result['output_files']['knowledge_graph'])
        graph.parse(kg_file, format='turtle')
        
        # Count different entity types
        from src.build_kg.rdf_builder import KG
        
        genome_count = len(list(graph.subjects(predicate=None, object=KG.Genome)))
        protein_count = len(list(graph.subjects(predicate=None, object=KG.Protein)))
        domain_count = len(list(graph.subjects(predicate=None, object=KG.DomainAnnotation)))
        function_count = len(list(graph.subjects(predicate=None, object=KG.FunctionalAnnotation)))
        
        # Verify counts match expected
        assert genome_count == result['genomes_processed']
        assert protein_count == result['proteins_annotated'] 
        assert domain_count == result['pfam_domains']
        assert function_count == result['kofam_functions']
        
        # Check relationships exist
        from src.build_kg.rdf_builder import PROTEIN, GENE
        
        # Every protein should have an encoding gene
        proteins = list(graph.subjects(predicate=None, object=KG.Protein))
        for protein_uri in proteins:
            gene_relations = list(graph.objects(subject=protein_uri, predicate=KG.encodedBy))
            assert len(gene_relations) == 1
        
        # Domains should link to proteins
        if domain_count > 0:
            domains = list(graph.subjects(predicate=None, object=KG.DomainAnnotation))
            for domain_uri in domains:
                protein_relations = list(graph.objects(subject=domain_uri, predicate=KG.belongsToProtein))
                assert len(protein_relations) == 1

    def create_minimal_test_data(self, temp_dir):
        """Create minimal test data for unit testing."""
        stage03_dir = temp_dir / "stage03_test"
        stage04_dir = temp_dir / "stage04_test" 
        
        # Create stage03 structure
        stage03_dir.mkdir()
        genomes_dir = stage03_dir / "genomes"
        genomes_dir.mkdir()
        
        # Create manifest
        manifest = {
            "version": "0.1.0",
            "timestamp": "2025-06-20T10:00:00Z",
            "genomes": [
                {
                    "genome_id": "test_genome",
                    "execution_status": "success",
                    "output_files": ["test_genome.faa"]
                }
            ]
        }
        with open(stage03_dir / "processing_manifest.json", 'w') as f:
            json.dump(manifest, f)
            
        # Create protein file
        genome_dir = genomes_dir / "test_genome"
        genome_dir.mkdir()
        protein_file = genome_dir / "test_genome.faa"
        with open(protein_file, 'w') as f:
            f.write(">test_protein_001 hypothetical protein\nMKTLRQVKRS\n")
            f.write(">test_protein_002 kinase\nMATKLRQVKRSLPQ\n")
        
        # Create stage04 structure
        stage04_dir.mkdir()
        pfam_dir = stage04_dir / "pfam_results"
        kofam_dir = stage04_dir / "kofam_results"
        pfam_dir.mkdir()
        kofam_dir.mkdir()
        
        # Create PFAM results
        import pandas as pd
        pfam_data = pd.DataFrame({
            'sequence_id': ['test_protein_001', 'test_protein_002'],
            'hmm_name': ['PF00001', 'PF00002'],
            'env_from': [5, 10],
            'env_to': [25, 30],
            'bitscore': [100.5, 85.3],
            'evalue': [1e-20, 1e-15],
            'dom_bitscore': [98.2, 83.1]
        })
        pfam_data.to_csv(pfam_dir / "PFAM_hits_df.tsv", sep='\t', index=False)
        
        # Create KOFAM results  
        kofam_data = pd.DataFrame({
            'sequence_id': ['test_protein_001', 'test_protein_002'],
            'hmm_name': ['K00001', 'K00002'],
            'bitscore': [200.5, 150.3],
            'evalue': [1e-30, 1e-25]
        })
        kofam_data.to_csv(kofam_dir / "KOFAM_hits_df.tsv", sep='\t', index=False)
        
        return stage03_dir, stage04_dir

    def test_minimal_knowledge_graph_build(self, temp_dir):
        """Test building knowledge graph with minimal synthetic data."""
        stage03_dir, stage04_dir = self.create_minimal_test_data(temp_dir)
        output_dir = temp_dir / "minimal_kg_output"
        
        result = build_knowledge_graph_from_pipeline(stage03_dir, stage04_dir, output_dir)
        
        # Verify basic results
        assert result['genomes_processed'] == 1
        assert result['proteins_annotated'] == 2
        assert result['pfam_domains'] == 2
        assert result['kofam_functions'] == 2  # Best hit per protein
        assert result['total_triples'] > 20  # Should have substantial content
        
        # Verify files exist
        kg_file = Path(result['output_files']['knowledge_graph'])
        assert kg_file.exists()
        
        stats_file = output_dir / "build_statistics.json"
        assert stats_file.exists()
        
        # Load and verify statistics match
        with open(stats_file) as f:
            saved_stats = json.load(f)
            assert saved_stats == result

    def test_error_handling_invalid_input(self, temp_dir):
        """Test error handling with invalid input directories."""
        stage03_dir = temp_dir / "nonexistent_stage03"
        stage04_dir = temp_dir / "nonexistent_stage04"
        output_dir = temp_dir / "error_test_output"
        
        with pytest.raises(FileNotFoundError):
            build_knowledge_graph_from_pipeline(stage03_dir, stage04_dir, output_dir)

    def test_knowledge_graph_serialization_formats(self, temp_dir):
        """Test knowledge graph can be serialized in different formats."""
        stage03_dir, stage04_dir = self.create_minimal_test_data(temp_dir)
        output_dir = temp_dir / "serialization_test"
        
        # Build knowledge graph
        result = build_knowledge_graph_from_pipeline(stage03_dir, stage04_dir, output_dir)
        
        # Test loading with rdflib and converting to other formats
        from rdflib import Graph
        graph = Graph()
        kg_file = Path(result['output_files']['knowledge_graph'])
        graph.parse(kg_file, format='turtle')
        
        # Test serializing to different formats
        formats_to_test = ['xml', 'n3', 'nt']
        
        for fmt in formats_to_test:
            output_file = output_dir / f"test_kg.{fmt}"
            serialized = graph.serialize(format=fmt)
            
            with open(output_file, 'w') as f:
                f.write(serialized)
            
            assert output_file.exists()
            assert output_file.stat().st_size > 100  # Should have content
            
            # Test that we can reload it
            test_graph = Graph()
            test_graph.parse(output_file, format=fmt)
            assert len(test_graph) == len(graph)  # Same number of triples