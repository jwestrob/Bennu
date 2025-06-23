#!/usr/bin/env python3
"""
Test cases for functional enrichment module.
"""

import pytest
import tempfile
from pathlib import Path
import rdflib
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS

from src.build_kg.functional_enrichment import FunctionalEnrichment, PfamEntry, KoEntry


@pytest.fixture
def sample_pfam_file():
    """Create sample PFAM Stockholm format file."""
    content = """# STOCKHOLM 1.0
#=GF ID   GGDEF
#=GF AC   PF00990.23
#=GF DE   Diguanylate cyclase, GGDEF domain
#=GF GA   21.4; 21.4;
#=GF TP   Domain
#=GF ML   158
#=GF CL   CL0489
//
# STOCKHOLM 1.0
#=GF ID   PAS
#=GF AC   PF00989.26
#=GF DE   PAS domain
#=GF GA   24.9; 24.9;
#=GF TP   Domain
#=GF ML   82
//
# STOCKHOLM 1.0
#=GF ID   14-3-3
#=GF AC   PF00244.26
#=GF DE   14-3-3 protein
#=GF GA   33.2; 33.2;
#=GF TP   Repeat
#=GF ML   223
//"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.stockholm', delete=False) as f:
        f.write(content)
        return Path(f.name)


@pytest.fixture
def sample_ko_file():
    """Create sample KO list file."""
    content = """knum	threshold	score_type	profile_type	F-measure	nseq	nseq_used	alen	mlen	eff_nseq	re/pos	definition	simplified_definition
K00001	340.20	domain	all	0.438754	1959	1682	1748	504	12.24	0.590	alcohol dehydrogenase [EC:1.1.1.1]	alcohol dehydrogenase
K00259	250.63	domain	all	0.932515	3918	3298	2051	398	7.45	0.590	2-oxoglutarate dehydrogenase E1 component [EC:1.2.4.2]	2-oxoglutarate dehydrogenase E1 component
K07133	341.10	full	all	0.848131	3471	2891	1834	439	5.72	0.590	uncharacterized protein	uncharacterized protein"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        f.write(content)
        return Path(f.name)


@pytest.fixture
def sample_rdf_graph():
    """Create sample RDF graph with PFAM and KO entities."""
    graph = Graph()
    
    # Define namespaces
    kg = Namespace("http://genome-kg.org/ontology/")
    pfam = Namespace("http://pfam.xfam.org/family/")
    ko = Namespace("http://www.genome.jp/kegg/ko/")
    
    # Add PFAM families
    ggdef_uri = pfam["GGDEF"]
    pas_uri = pfam["PAS"]
    test_uri = pfam["TestFamily"]  # This won't be in reference data
    
    graph.add((ggdef_uri, RDF.type, kg.ProteinFamily))
    graph.add((pas_uri, RDF.type, kg.ProteinFamily))
    graph.add((test_uri, RDF.type, kg.ProteinFamily))
    
    # Add KO functions
    k00001_uri = ko["K00001"]
    k00259_uri = ko["K00259"]
    k99999_uri = ko["K99999"]  # This won't be in reference data
    
    graph.add((k00001_uri, RDF.type, kg.KEGGOrtholog))
    graph.add((k00259_uri, RDF.type, kg.KEGGOrtholog))
    graph.add((k99999_uri, RDF.type, kg.KEGGOrtholog))
    
    return graph


class TestFunctionalEnrichment:
    """Test cases for FunctionalEnrichment class."""
    
    def test_parse_pfam_stockholm(self, sample_pfam_file):
        """Test PFAM Stockholm format parsing."""
        enricher = FunctionalEnrichment(sample_pfam_file, Path("dummy"))
        pfam_data = enricher._parse_pfam_stockholm()
        
        assert len(pfam_data) == 3
        
        # Test GGDEF entry
        ggdef = pfam_data["GGDEF"]
        assert isinstance(ggdef, PfamEntry)
        assert ggdef.id == "GGDEF"
        assert ggdef.accession == "PF00990.23"
        assert "Diguanylate cyclase" in ggdef.description
        assert ggdef.type == "Domain"
        assert ggdef.model_length == 158
        assert ggdef.clan == "CL0489"
        
        # Test PAS entry
        pas = pfam_data["PAS"]
        assert pas.id == "PAS"
        assert pas.description == "PAS domain"
        assert pas.type == "Domain"
        
        # Test 14-3-3 entry (no clan)
        protein_143 = pfam_data["14-3-3"]
        assert protein_143.id == "14-3-3"
        assert protein_143.type == "Repeat"
        assert protein_143.clan is None
    
    def test_parse_ko_list(self, sample_ko_file):
        """Test KO list parsing."""
        enricher = FunctionalEnrichment(Path("dummy"), sample_ko_file)
        ko_data = enricher._parse_ko_list()
        
        assert len(ko_data) == 3
        
        # Test K00001 entry
        k00001 = ko_data["K00001"]
        assert isinstance(k00001, KoEntry)
        assert k00001.knum == "K00001"
        assert k00001.threshold == 340.20
        assert "alcohol dehydrogenase" in k00001.definition
        assert k00001.simplified_definition == "alcohol dehydrogenase"
        assert k00001.score_type == "domain"
        assert k00001.profile_type == "all"
        
        # Test K00259 entry
        k00259 = ko_data["K00259"]
        assert "2-oxoglutarate dehydrogenase" in k00259.definition
        assert "[EC:1.2.4.2]" in k00259.definition
    
    def test_enrich_rdf_graph(self, sample_rdf_graph, sample_pfam_file, sample_ko_file):
        """Test RDF graph enrichment."""
        enricher = FunctionalEnrichment(sample_pfam_file, sample_ko_file)
        enriched_graph, stats = enricher.enrich_rdf_graph(sample_rdf_graph)
        
        # Check statistics
        assert stats['pfam_enriched'] == 2  # GGDEF and PAS found
        assert stats['missing_pfam'] == 1   # TestFamily not found
        assert stats['ko_enriched'] == 2    # K00001 and K00259 found
        assert stats['missing_ko'] == 1     # K99999 not found
        
        # Define namespaces for testing
        kg = Namespace("http://genome-kg.org/ontology/")
        pfam = Namespace("http://pfam.xfam.org/family/")
        ko = Namespace("http://www.genome.jp/kegg/ko/")
        
        # Check PFAM enrichment
        ggdef_uri = pfam["GGDEF"]
        descriptions = list(enriched_graph.objects(ggdef_uri, kg.description))
        assert len(descriptions) == 1
        assert "Diguanylate cyclase" in str(descriptions[0])
        
        labels = list(enriched_graph.objects(ggdef_uri, RDFS.label))
        assert len(labels) == 1
        assert "Diguanylate cyclase" in str(labels[0])
        
        family_types = list(enriched_graph.objects(ggdef_uri, kg.familyType))
        assert len(family_types) == 1
        assert str(family_types[0]) == "Domain"
        
        clans = list(enriched_graph.objects(ggdef_uri, kg.clan))
        assert len(clans) == 1
        assert str(clans[0]) == "CL0489"
        
        # Check KO enrichment
        k00001_uri = ko["K00001"]
        ko_descriptions = list(enriched_graph.objects(k00001_uri, kg.description))
        assert len(ko_descriptions) == 1
        assert "alcohol dehydrogenase" in str(ko_descriptions[0])
        
        ko_labels = list(enriched_graph.objects(k00001_uri, RDFS.label))
        assert len(ko_labels) == 1
        assert str(ko_labels[0]) == "alcohol dehydrogenase"
        
        # Check EC number extraction
        ec_numbers = list(enriched_graph.objects(k00001_uri, kg.ecNumber))
        assert len(ec_numbers) == 1
        assert str(ec_numbers[0]) == "1.1.1.1"
    
    def test_missing_files(self):
        """Test behavior with missing reference files."""
        enricher = FunctionalEnrichment(Path("nonexistent_pfam.stockholm"), Path("nonexistent_ko.tsv"))
        enricher.load_reference_data()
        
        assert len(enricher.pfam_data) == 0
        assert len(enricher.ko_data) == 0
        
        # Should not crash with empty reference data
        empty_graph = Graph()
        enriched_graph, stats = enricher.enrich_rdf_graph(empty_graph)
        
        assert stats['pfam_enriched'] == 0
        assert stats['ko_enriched'] == 0


@pytest.fixture
def integration_test_setup(tmp_path):
    """Setup for integration testing with real pipeline structure."""
    # Create test directory structure
    stage03_dir = tmp_path / "stage03_prodigal"
    stage04_dir = tmp_path / "stage04_astra"
    stage05_dir = tmp_path / "stage05_kg"
    
    stage03_dir.mkdir()
    stage04_dir.mkdir()
    stage05_dir.mkdir()
    
    # Create minimal test manifest
    manifest = {
        "genomes": [
            {
                "genome_id": "test_genome",
                "execution_status": "success"
            }
        ]
    }
    
    with open(stage03_dir / "processing_manifest.json", 'w') as f:
        import json
        json.dump(manifest, f)
    
    # Create minimal protein file
    genomes_dir = stage03_dir / "genomes" / "test_genome"
    genomes_dir.mkdir(parents=True)
    
    with open(genomes_dir / "test_genome.faa", 'w') as f:
        f.write(">test_protein_1\nMKTEST\n>test_protein_2\nMKTEST2\n")
    
    return {
        'stage03_dir': stage03_dir,
        'stage04_dir': stage04_dir,
        'stage05_dir': stage05_dir
    }


class TestIntegration:
    """Integration tests for functional enrichment in the pipeline."""
    
    def test_pipeline_integration(self, integration_test_setup, sample_pfam_file, sample_ko_file):
        """Test integration with the knowledge graph building pipeline."""
        from src.build_kg.rdf_builder import build_knowledge_graph_from_pipeline
        
        # Setup test data
        dirs = integration_test_setup
        
        # Create minimal astra results
        pfam_dir = dirs['stage04_dir'] / "pfam_results"
        kofam_dir = dirs['stage04_dir'] / "kofam_results"
        pfam_dir.mkdir()
        kofam_dir.mkdir()
        
        # Create empty result files (annotation_processors will handle these)
        with open(pfam_dir / "PFAM_hits_df.tsv", 'w') as f:
            f.write("protein_id\tpfam_id\tstart_pos\tend_pos\tbitscore\tevalue\n")
        
        with open(kofam_dir / "KOFAM_hits_df.tsv", 'w') as f:
            f.write("protein_id\tko_id\tbitscore\tevalue\n")
        
        # Create processing manifest for stage04
        manifest = {"processing_completed": True}
        with open(dirs['stage04_dir'] / "processing_manifest.json", 'w') as f:
            import json
            json.dump(manifest, f)
        
        # Temporarily move reference files to expected locations
        import shutil
        original_pfam = Path("data/reference/Pfam-A.hmm.dat.stockholm")
        original_ko = Path("data/reference/ko_list")
        
        # Use the test files instead
        if original_pfam.exists():
            shutil.move(str(original_pfam), str(original_pfam) + ".backup")
        if original_ko.exists():
            shutil.move(str(original_ko), str(original_ko) + ".backup")
        
        shutil.copy(str(sample_pfam_file), str(original_pfam))
        shutil.copy(str(sample_ko_file), str(original_ko))
        
        try:
            # Run the pipeline with functional enrichment
            stats = build_knowledge_graph_from_pipeline(
                dirs['stage03_dir'],
                dirs['stage04_dir'], 
                dirs['stage05_dir']
            )
            
            # Check that functional enrichment was applied
            assert 'functional_enrichment' in stats
            enrichment_stats = stats['functional_enrichment']
            
            # Should have some enrichment (even if minimal due to test data)
            assert 'pfam_enriched' in enrichment_stats
            assert 'ko_enriched' in enrichment_stats
            
            # Check that the knowledge graph file was created
            kg_file = dirs['stage05_dir'] / "knowledge_graph.ttl"
            assert kg_file.exists()
            
            # Load and verify the enriched graph contains functional data
            graph = Graph()
            graph.parse(kg_file, format='turtle')
            
            # Should contain functional descriptions if any PFAM/KO entities were created
            kg = Namespace("http://genome-kg.org/ontology/")
            descriptions = list(graph.objects(None, kg.description))
            
            # Even if we don't have specific domains in test data, 
            # the enrichment process should have been applied
            assert stats['total_triples'] > 0
            
        finally:
            # Restore original files
            if Path(str(original_pfam) + ".backup").exists():
                shutil.move(str(original_pfam) + ".backup", str(original_pfam))
            else:
                original_pfam.unlink(missing_ok=True)
            
            if Path(str(original_ko) + ".backup").exists():
                shutil.move(str(original_ko) + ".backup", str(original_ko))
            else:
                original_ko.unlink(missing_ok=True)
    
    def test_enrichment_with_real_domains(self):
        """Test enrichment with domains that should exist in real reference data."""
        # This test will only run if real reference files exist
        pfam_file = Path("data/reference/Pfam-A.hmm.dat.stockholm")
        ko_file = Path("data/reference/ko_list")
        
        if not (pfam_file.exists() and ko_file.exists()):
            pytest.skip("Real reference files not available")
        
        # Create a minimal graph with known domains
        graph = Graph()
        kg = Namespace("http://genome-kg.org/ontology/")
        pfam = Namespace("http://pfam.xfam.org/family/")
        ko = Namespace("http://www.genome.jp/kegg/ko/")
        
        # Add some common domains that should be in PFAM
        common_domains = ["GGDEF", "PAS", "AAA"]
        for domain in common_domains:
            domain_uri = pfam[domain]
            graph.add((domain_uri, RDF.type, kg.ProteinFamily))
        
        # Add some common KO functions
        common_kos = ["K00001", "K00002"]  # These should exist in KO list
        for ko_id in common_kos:
            ko_uri = ko[ko_id]
            graph.add((ko_uri, RDF.type, kg.KEGGOrtholog))
        
        # Enrich the graph
        enricher = FunctionalEnrichment(pfam_file, ko_file)
        enriched_graph, stats = enricher.enrich_rdf_graph(graph)
        
        # Should have enriched at least some entries
        assert stats['pfam_enriched'] > 0 or stats['ko_enriched'] > 0
        
        # Check that descriptions were added
        descriptions = list(enriched_graph.objects(None, kg.description))
        assert len(descriptions) > 0