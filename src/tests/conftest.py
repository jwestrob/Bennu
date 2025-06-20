"""
Pytest configuration and shared fixtures for genome-kg tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import json


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def dummy_fasta_content():
    """Sample FASTA content for testing."""
    return """>contig_1
ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
CTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
>contig_2
GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
"""


@pytest.fixture
def dummy_genome_files(temp_dir):
    """Create dummy genome FASTA files for testing."""
    genome_files = []
    genomes = {
        "genome_A.fna": """>contig_1_A
ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
CTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
>contig_2_A
GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
""",
        "genome_B.fasta": """>contig_1_B
TTGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
CTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
>contig_2_B
GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
TTGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
""",
        "genome_C.fa": """>contig_1_C
AAGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
CTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
CTAGCTAGCTAGCTAGCTAGCTAGCTAG
>contig_2_C
GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
>contig_3_C
AAGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
"""
    }
    
    input_dir = temp_dir / "input"
    input_dir.mkdir()
    
    for filename, content in genomes.items():
        genome_file = input_dir / filename
        with open(genome_file, 'w') as f:
            f.write(content)
        genome_files.append(genome_file)
    
    return input_dir, genome_files


@pytest.fixture
def mock_quast_output():
    """Mock QUAST output data."""
    return {
        "genome_A": {
            "total_length": 240,
            "n50": 120,
            "num_contigs": 2,
            "gc_content": 50.0
        },
        "genome_B": {
            "total_length": 240,
            "n50": 120,
            "num_contigs": 2,
            "gc_content": 48.0
        },
        "genome_C": {
            "total_length": 183,
            "n50": 87,
            "num_contigs": 3,
            "gc_content": 52.0
        }
    }


@pytest.fixture
def mock_checkm_output():
    """Mock CheckM output data."""
    return {
        "genome_A": {
            "completeness": 95.2,
            "contamination": 1.1,
            "strain_heterogeneity": 0.0
        },
        "genome_B": {
            "completeness": 92.8,
            "contamination": 0.5,
            "strain_heterogeneity": 0.0
        },
        "genome_C": {
            "completeness": 85.3,
            "contamination": 2.1,
            "strain_heterogeneity": 10.0
        }
    }


@pytest.fixture
def mock_gtdb_output():
    """Mock GTDB-Tk taxonomic output."""
    return {
        "genome_A": {
            "classification": "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia coli",
            "fastani_reference": "GCF_000005825.2",
            "fastani_reference_radius": 95.0,
            "fastani_taxonomy": "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia coli",
            "fastani_ani": 99.8,
            "classification_method": "ANI"
        },
        "genome_B": {
            "classification": "d__Bacteria;p__Firmicutes;c__Bacilli;o__Bacillales;f__Bacillaceae;g__Bacillus;s__Bacillus subtilis",
            "fastani_reference": "GCF_000009045.1",
            "fastani_reference_radius": 95.0,
            "fastani_taxonomy": "d__Bacteria;p__Firmicutes;c__Bacilli;o__Bacillales;f__Bacillaceae;g__Bacillus;s__Bacillus subtilis",
            "fastani_ani": 98.5,
            "classification_method": "ANI"
        },
        "genome_C": {
            "classification": "d__Bacteria;p__Actinobacteriota;c__Actinobacteria;o__Mycobacteriales;f__Mycobacteriaceae;g__Mycobacterium;s__",
            "fastani_reference": "",
            "fastani_reference_radius": 95.0,
            "fastani_taxonomy": "",
            "fastani_ani": 0.0,
            "classification_method": "phylogenetic_placement"
        }
    }


@pytest.fixture
def mock_prodigal_output():
    """Mock Prodigal gene prediction output."""
    return {
        "genome_A": {
            "genes": [
                {
                    "gene_id": "genome_A_001",
                    "contig": "contig_1_A",
                    "start": 1,
                    "end": 60,
                    "strand": "+",
                    "product": "hypothetical protein"
                },
                {
                    "gene_id": "genome_A_002",
                    "contig": "contig_2_A",
                    "start": 5,
                    "end": 65,
                    "strand": "-",
                    "product": "DNA polymerase"
                }
            ]
        }
    }


@pytest.fixture
def expected_pipeline_manifest():
    """Expected pipeline processing manifest format."""
    return {
        "version": "0.1.0",
        "timestamp": "2025-06-18T21:35:00Z",
        "input_dir": "/path/to/input",
        "genomes": [
            {
                "filename": "genome_A.fna",
                "genome_id": "genome_A",
                "file_size": 240,
                "sequence_count": 2,
                "format_valid": True,
                "checksum": "abc123"
            },
            {
                "filename": "genome_B.fasta",
                "genome_id": "genome_B", 
                "file_size": 240,
                "sequence_count": 2,
                "format_valid": True,
                "checksum": "def456"
            },
            {
                "filename": "genome_C.fa",
                "genome_id": "genome_C",
                "file_size": 183,
                "sequence_count": 3,
                "format_valid": True,
                "checksum": "ghi789"
            }
        ]
    }


# CLI testing helpers
@pytest.fixture
def cli_runner():
    """Typer CLI test runner."""
    from typer.testing import CliRunner
    return CliRunner()


# Mock external tool execution
@pytest.fixture
def mock_subprocess_run(monkeypatch):
    """Mock subprocess.run for external tool execution."""
    import subprocess
    
    def mock_run(*args, **kwargs):
        # Return success by default
        result = subprocess.CompletedProcess(
            args=args[0] if args else [],
            returncode=0,
            stdout="Mock output",
            stderr=""
        )
        return result
    
    monkeypatch.setattr(subprocess, "run", mock_run)
    return mock_run
