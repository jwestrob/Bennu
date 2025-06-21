#!/usr/bin/env python3
"""
Tests for RDF builder module.
"""

import pytest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock

import rdflib
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from src.build_kg.rdf_builder import (
    GenomeKGBuilder,
    build_knowledge_graph_from_pipeline,
    KG, GENOME, GENE, PROTEIN, PFAM, KO, PROV
)


class TestGenomeKGBuilder:
    """Test GenomeKGBuilder class."""

    def test_init(self):
        """Test builder initialization."""
        builder = GenomeKGBuilder()
        
        assert isinstance(builder.graph, Graph)
        
        # Check namespace bindings
        namespaces = dict(builder.graph.namespaces())
        assert namespaces['kg'] == KG
        assert namespaces['genome'] == GENOME
        assert namespaces['protein'] == PROTEIN
        
        # Check ontology definitions are added
        assert (KG.Genome, RDF.type, RDFS.Class) in builder.graph
        assert (KG.Protein, RDF.type, RDFS.Class) in builder.graph

    def test_add_genome_entity_basic(self):
        """Test adding basic genome entity."""
        builder = GenomeKGBuilder()
        
        genome_data = {
            'genome_id': 'test_genome_001',
        }
        
        genome_uri = builder.add_genome_entity(genome_data)
        
        # Check return value
        assert genome_uri == GENOME['test_genome_001']
        
        # Check triples were added
        assert (genome_uri, RDF.type, KG.Genome) in builder.graph
        assert (genome_uri, KG.genomeId, Literal('test_genome_001')) in builder.graph

    def test_add_genome_entity_with_quality_metrics(self):
        """Test adding genome entity with quality metrics."""
        builder = GenomeKGBuilder()
        
        genome_data = {
            'genome_id': 'test_genome_001',
            'quality_metrics': {
                'total_length': 4500000,
                'n50': 150000,
                'gc_content': 42.5,
                'num_contigs': 15
            }
        }
        
        genome_uri = builder.add_genome_entity(genome_data)
        metrics_uri = GENOME['test_genome_001/quality']
        
        # Check quality metrics triples
        assert (genome_uri, KG.hasQualityMetrics, metrics_uri) in builder.graph
        assert (metrics_uri, RDF.type, KG.QualityMetrics) in builder.graph
        assert (metrics_uri, KG.quast_total_length, Literal(4500000, datatype=XSD.integer)) in builder.graph
        assert (metrics_uri, KG.quast_n50, Literal(150000, datatype=XSD.integer)) in builder.graph
        assert (metrics_uri, KG.quast_gc_content, Literal(42.5, datatype=XSD.float)) in builder.graph

    def test_add_gene_protein_entities(self):
        """Test adding gene and protein entities."""
        builder = GenomeKGBuilder()
        
        # Add genome first
        genome_data = {'genome_id': 'test_genome'}
        genome_uri = builder.add_genome_entity(genome_data)
        
        gene_data = [
            {
                'gene_id': 'gene_001',
                'contig': 'contig_1',
                'start': 100,
                'end': 500,
                'strand': '+',
                'protein_sequence': 'MKTLPQVKRS'
            },
            {
                'gene_id': 'gene_002',
                'contig': 'contig_1',
                'start': 600,
                'end': 900,
                'strand': '-'
            }
        ]
        
        protein_uris = builder.add_gene_protein_entities(gene_data, genome_uri)
        
        # Check return value
        assert len(protein_uris) == 2
        assert 'gene_001' in protein_uris
        assert 'gene_002' in protein_uris
        
        # Check gene triples
        gene_uri_1 = GENE['gene_001']
        assert (gene_uri_1, RDF.type, KG.Gene) in builder.graph
        assert (gene_uri_1, KG.belongsToGenome, genome_uri) in builder.graph
        assert (gene_uri_1, KG.geneId, Literal('gene_001')) in builder.graph
        assert (gene_uri_1, KG.hasLocation, Literal('contig_1:100-500')) in builder.graph
        assert (gene_uri_1, KG.strand, Literal('+')) in builder.graph
        
        # Check protein triples
        protein_uri_1 = PROTEIN['gene_001']
        assert (protein_uri_1, RDF.type, KG.Protein) in builder.graph
        assert (protein_uri_1, KG.encodedBy, gene_uri_1) in builder.graph
        assert (protein_uri_1, KG.proteinId, Literal('gene_001')) in builder.graph
        assert (protein_uri_1, KG.sequence, Literal('MKTLPQVKRS')) in builder.graph
        assert (protein_uri_1, KG.length, Literal(10, datatype=XSD.integer)) in builder.graph

    def test_add_pfam_domains(self):
        """Test adding PFAM domain annotations."""
        builder = GenomeKGBuilder()
        
        # Set up protein URIs
        protein_uris = {
            'protein_001': PROTEIN['protein_001'],
            'protein_002': PROTEIN['protein_002']
        }
        
        domains = [
            {
                'domain_id': 'protein_001/domain/PF00001/10-50',
                'protein_id': 'protein_001',
                'pfam_id': 'PF00001',
                'start_pos': 10,
                'end_pos': 50,
                'bitscore': 120.5,
                'evalue': 1e-25
            },
            {
                'domain_id': 'protein_002/domain/PF00002/5-35',
                'protein_id': 'protein_002',
                'pfam_id': 'PF00002',
                'start_pos': 5,
                'end_pos': 35,
                'bitscore': 85.3,
                'evalue': 1e-15
            }
        ]
        
        builder.add_pfam_domains(domains, protein_uris)
        
        # Check domain triples
        domain_uri_1 = PROTEIN['protein_001/domain/PF00001/10-50']
        pfam_uri_1 = PFAM['PF00001']
        protein_uri_1 = PROTEIN['protein_001']
        
        assert (domain_uri_1, RDF.type, KG.ProteinDomain) in builder.graph
        assert (domain_uri_1, KG.belongsToProtein, protein_uri_1) in builder.graph
        assert (domain_uri_1, KG.domainFamily, pfam_uri_1) in builder.graph
        assert (domain_uri_1, KG.domainStart, Literal(10, datatype=XSD.integer)) in builder.graph
        assert (domain_uri_1, KG.domainEnd, Literal(50, datatype=XSD.integer)) in builder.graph
        assert (domain_uri_1, KG.bitscore, Literal(120.5, datatype=XSD.float)) in builder.graph
        assert (domain_uri_1, KG.evalue, Literal(1e-25, datatype=XSD.double)) in builder.graph
        
        # Check PFAM family triples
        assert (pfam_uri_1, RDF.type, KG.ProteinFamily) in builder.graph
        assert (pfam_uri_1, KG.pfamAccession, Literal('PF00001')) in builder.graph
        
        # Check protein-domain link
        assert (protein_uri_1, KG.hasDomain, domain_uri_1) in builder.graph

    def test_add_pfam_domains_unknown_protein(self):
        """Test adding PFAM domains with unknown protein ID."""
        builder = GenomeKGBuilder()
        
        protein_uris = {'protein_001': PROTEIN['protein_001']}
        
        domains = [
            {
                'domain_id': 'unknown_protein/domain/PF00001/10-50',
                'protein_id': 'unknown_protein',
                'pfam_id': 'PF00001',
                'start_pos': 10,
                'end_pos': 50,
                'bitscore': 120.5,
                'evalue': 1e-25
            }
        ]
        
        # Should not raise exception, just log warning
        builder.add_pfam_domains(domains, protein_uris)
        
        # Should not add any domain triples for unknown protein
        domain_uri = PROTEIN['unknown_protein/domain/PF00001/10-50']
        assert (domain_uri, RDF.type, KG.ProteinDomain) not in builder.graph

    def test_add_kofam_functions(self):
        """Test adding KOFAM functional annotations."""
        builder = GenomeKGBuilder()
        
        protein_uris = {
            'protein_001': PROTEIN['protein_001'],
            'protein_002': PROTEIN['protein_002']
        }
        
        functions = [
            {
                'annotation_id': 'protein_001/function/K00001',
                'protein_id': 'protein_001',
                'ko_id': 'K00001',
                'bitscore': 250.3,
                'evalue': 1e-35,
                'confidence': 'high'
            },
            {
                'annotation_id': 'protein_002/function/K00002',
                'protein_id': 'protein_002',
                'ko_id': 'K00002',
                'bitscore': 180.1,
                'evalue': 1e-20,
                'confidence': 'medium'
            }
        ]
        
        builder.add_kofam_functions(functions, protein_uris)
        
        # Check function annotation triples
        annotation_uri_1 = PROTEIN['protein_001/function/K00001']
        ko_uri_1 = KO['K00001']
        protein_uri_1 = PROTEIN['protein_001']
        
        assert (annotation_uri_1, RDF.type, KG.FunctionalAnnotation) in builder.graph
        assert (annotation_uri_1, KG.annotatesProtein, protein_uri_1) in builder.graph
        assert (annotation_uri_1, KG.assignedFunction, ko_uri_1) in builder.graph
        assert (annotation_uri_1, KG.confidence, Literal('high')) in builder.graph
        assert (annotation_uri_1, KG.bitscore, Literal(250.3, datatype=XSD.float)) in builder.graph
        assert (annotation_uri_1, KG.evalue, Literal(1e-35, datatype=XSD.double)) in builder.graph
        
        # Check KEGG Ortholog triples
        assert (ko_uri_1, RDF.type, KG.KEGGOrtholog) in builder.graph
        assert (ko_uri_1, KG.koId, Literal('K00001')) in builder.graph
        
        # Check protein-function link
        assert (protein_uri_1, KG.hasFunction, ko_uri_1) in builder.graph

    def test_add_provenance(self):
        """Test adding provenance information."""
        builder = GenomeKGBuilder()
        
        pipeline_data = {
            'version': '1.0.0',
            'astra_databases': ['PFAM', 'KOFAM']
        }
        
        builder.add_provenance(pipeline_data)
        
        kg_uri = URIRef("http://genome-kg.org/this-kg")
        
        assert (kg_uri, RDF.type, PROV.Entity) in builder.graph
        assert (kg_uri, PROV.wasGeneratedBy, Literal("genome-kg-pipeline")) in builder.graph
        assert (kg_uri, KG.pipelineVersion, Literal('1.0.0')) in builder.graph
        assert (kg_uri, KG.usedDatabase, Literal('PFAM')) in builder.graph
        assert (kg_uri, KG.usedDatabase, Literal('KOFAM')) in builder.graph

    def test_save_graph(self, temp_dir):
        """Test saving knowledge graph to file."""
        builder = GenomeKGBuilder()
        
        # Add some test data
        genome_data = {'genome_id': 'test_genome'}
        builder.add_genome_entity(genome_data)
        
        output_file = temp_dir / "test_kg.ttl"
        
        result = builder.save_graph(output_file, format='turtle')
        
        # Check file was created
        assert output_file.exists()
        
        # Check return value
        assert 'output_file' in result
        assert 'format' in result
        assert 'triple_count' in result
        assert 'timestamp' in result
        assert result['format'] == 'turtle'
        assert result['triple_count'] > 0
        
        # Check file content is valid Turtle
        with open(output_file, 'r') as f:
            content = f.read()
            assert '@prefix' in content
            assert 'kg:Genome' in content

    def test_save_graph_handles_serialization_error(self, temp_dir):
        """Test save_graph error handling."""
        builder = GenomeKGBuilder()
        
        # Try to save to invalid path
        invalid_path = temp_dir / "nonexistent_dir" / "test.ttl"
        
        with pytest.raises(Exception):
            builder.save_graph(invalid_path)


class TestBuildKnowledgeGraphFromPipeline:
    """Test full pipeline knowledge graph building."""

    def create_mock_prodigal_manifest(self, temp_dir):
        """Create mock prodigal processing manifest."""
        manifest = {
            "version": "0.1.0",
            "timestamp": "2025-06-20T10:00:00Z",
            "genomes": [
                {
                    "genome_id": "genome_A",
                    "execution_status": "success",
                    "output_files": ["genome_A.faa"]
                },
                {
                    "genome_id": "genome_B", 
                    "execution_status": "success",
                    "output_files": ["genome_B.faa"]
                },
                {
                    "genome_id": "genome_C",
                    "execution_status": "failed",
                    "error": "Prodigal failed"
                }
            ]
        }
        
        manifest_file = temp_dir / "processing_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f)
        
        return manifest

    def create_mock_protein_files(self, stage03_dir):
        """Create mock protein FASTA files."""
        genomes_dir = stage03_dir / "genomes"
        genomes_dir.mkdir(parents=True)
        
        # Genome A proteins
        genome_a_dir = genomes_dir / "genome_A"
        genome_a_dir.mkdir()
        protein_file_a = genome_a_dir / "genome_A.faa"
        with open(protein_file_a, 'w') as f:
            f.write(">genome_A_001 hypothetical protein\nMKTLRQVKRS\n")
            f.write(">genome_A_002 DNA polymerase\nMATKLRQVKRSLPQ\n")
        
        # Genome B proteins
        genome_b_dir = genomes_dir / "genome_B"
        genome_b_dir.mkdir()
        protein_file_b = genome_b_dir / "genome_B.faa"
        with open(protein_file_b, 'w') as f:
            f.write(">genome_B_001 ribosomal protein\nMATTKLRQVK\n")

    @patch('src.build_kg.rdf_builder.process_astra_results')
    def test_build_knowledge_graph_from_pipeline_complete(self, mock_process_astra, temp_dir):
        """Test complete knowledge graph building from pipeline."""
        # Set up directories
        stage03_dir = temp_dir / "stage03"
        stage04_dir = temp_dir / "stage04"
        output_dir = temp_dir / "output"
        
        stage03_dir.mkdir()
        stage04_dir.mkdir()
        
        # Create mock manifest and protein files
        self.create_mock_prodigal_manifest(stage03_dir)
        self.create_mock_protein_files(stage03_dir)
        
        # Mock astra results
        mock_astra_results = {
            'pfam_domains': [
                {
                    'domain_id': 'genome_A_001/domain/PF00001/5-25',
                    'protein_id': 'genome_A_001',
                    'pfam_id': 'PF00001',
                    'start_pos': 5,
                    'end_pos': 25,
                    'bitscore': 100.5,
                    'evalue': 1e-20
                }
            ],
            'kofam_functions': [
                {
                    'annotation_id': 'genome_A_002/function/K00001',
                    'protein_id': 'genome_A_002',
                    'ko_id': 'K00001',
                    'bitscore': 200.3,
                    'evalue': 1e-30,
                    'confidence': 'high'
                }
            ]
        }
        mock_process_astra.return_value = mock_astra_results
        
        # Run the function
        result = build_knowledge_graph_from_pipeline(stage03_dir, stage04_dir, output_dir)
        
        # Check results
        assert 'total_triples' in result
        assert 'genomes_processed' in result
        assert 'proteins_annotated' in result
        assert 'pfam_domains' in result
        assert 'kofam_functions' in result
        assert 'output_files' in result
        
        assert result['genomes_processed'] == 2  # Only successful genomes
        assert result['proteins_annotated'] == 3  # Total proteins from A and B
        assert result['pfam_domains'] == 1
        assert result['kofam_functions'] == 1
        
        # Check output files exist
        kg_file = output_dir / "knowledge_graph.ttl"
        stats_file = output_dir / "build_statistics.json"
        
        assert kg_file.exists()
        assert stats_file.exists()
        
        # Check statistics file
        with open(stats_file) as f:
            saved_stats = json.load(f)
            assert saved_stats == result

    @patch('src.build_kg.rdf_builder.process_astra_results')
    def test_build_knowledge_graph_no_proteins(self, mock_process_astra, temp_dir):
        """Test building knowledge graph when protein files are missing."""
        stage03_dir = temp_dir / "stage03"
        stage04_dir = temp_dir / "stage04"
        output_dir = temp_dir / "output"
        
        stage03_dir.mkdir()
        stage04_dir.mkdir()
        
        # Create manifest but no protein files
        self.create_mock_prodigal_manifest(stage03_dir)
        # Skip creating protein files
        
        mock_astra_results = {
            'pfam_domains': [],
            'kofam_functions': []
        }
        mock_process_astra.return_value = mock_astra_results
        
        result = build_knowledge_graph_from_pipeline(stage03_dir, stage04_dir, output_dir)
        
        # Should still work but with no proteins
        assert result['proteins_annotated'] == 0
        assert result['genomes_processed'] == 2  # Genomes were still processed

    def test_build_knowledge_graph_missing_manifest(self, temp_dir):
        """Test error handling when prodigal manifest is missing."""
        stage03_dir = temp_dir / "stage03"
        stage04_dir = temp_dir / "stage04"
        output_dir = temp_dir / "output"
        
        stage03_dir.mkdir()
        stage04_dir.mkdir()
        
        # Don't create manifest file
        
        with pytest.raises(FileNotFoundError):
            build_knowledge_graph_from_pipeline(stage03_dir, stage04_dir, output_dir)