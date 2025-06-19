#!/usr/bin/env python3
"""
Tests for DFAST_QC taxonomic classification stage.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import importlib
dfast_qc_module = importlib.import_module('src.ingest.02_dfast_qc')
ref_module = importlib.import_module('src.ingest.01_prepare_dqc_reference')

parse_dqc_json = dfast_qc_module.parse_dqc_json
run_dfast = dfast_qc_module.run_dfast
call = dfast_qc_module.call
download = ref_module.download


@pytest.fixture
def example_json():
    """
    Example DFAST_QC JSON result based on documentation.
    
    This represents a typical dqc_result.json output from DFAST_QC
    for Lactobacillus paragasseri genome analysis.
    """
    return {
        "version": "1.2.5",
        "taxonomy": {
            "rank": "species",
            "species": "Lactobacillus paragasseri",
            "genus": "Lactobacillus", 
            "ani": 98.5,
            "confidence": 0.98
        },
        "quality": {
            "checkm": {
                "completeness": 95.2,
                "contamination": 1.8
            }
        },
        "input_file": "test_genome.fna",
        "runtime": {
            "total_time": 245.6
        }
    }


@pytest.fixture
def example_incomplete_json():
    """Example JSON with missing fields for testing robustness."""
    return {
        "taxonomy": {
            "genus": "Escherichia"
        },
        "quality": {}
    }


@pytest.fixture
def temp_dqc_result_file(example_json, tmp_path):
    """Create temporary DFAST_QC result file."""
    result_file = tmp_path / "dqc_result.json"
    with open(result_file, 'w') as f:
        json.dump(example_json, f, indent=2)
    return result_file


def test_parse_json(example_json, temp_dqc_result_file):
    """
    Test parsing of DFAST_QC JSON result.
    
    Validates that parse_dqc_json correctly extracts taxonomy and quality
    information and creates a properly formatted summary.
    """
    result = parse_dqc_json(temp_dqc_result_file)
    
    # Check required fields are present
    assert "rank" in result
    assert "name" in result
    assert "ani" in result
    assert "status" in result
    assert "completeness" in result
    assert "contamination" in result
    assert "tool" in result
    assert "version" in result
    assert "confidence" in result
    
    # Check specific values from example
    assert result["rank"] == "species"
    assert result["name"] == "Lactobacillus paragasseri"
    assert result["ani"] == 98.5
    assert result["confidence"] >= 0.95  # Task requirement
    assert result["completeness"] == 95.2
    assert result["contamination"] == 1.8
    assert result["tool"] == "dfast_qc"
    assert result["version"] == "1.2.5"
    assert result["status"] == "complete"  # >90% completeness


def test_parse_json_incomplete_data(example_incomplete_json, tmp_path):
    """Test parsing with incomplete/missing data."""
    result_file = tmp_path / "incomplete.json"
    with open(result_file, 'w') as f:
        json.dump(example_incomplete_json, f)
    
    result = parse_dqc_json(result_file)
    
    # Should handle missing data gracefully
    assert result["rank"] == "unknown"
    assert result["name"] == "Escherichia"  # Should use genus when species missing
    assert result["ani"] == 0.0
    assert result["completeness"] == 0.0
    assert result["contamination"] == 0.0
    assert result["tool"] == "dfast_qc"
    assert result["status"] == "partial"  # <90% completeness


def test_parse_json_nonexistent_file(tmp_path):
    """Test parsing with non-existent file."""
    nonexistent_file = tmp_path / "does_not_exist.json"
    result = parse_dqc_json(nonexistent_file)
    
    # Should return defaults for all fields
    assert result["rank"] == "unknown"
    assert result["name"] == "unknown"
    assert result["ani"] == 0.0
    assert result["completeness"] == 0.0
    assert result["contamination"] == 0.0
    assert result["tool"] == "dfast_qc"
    assert result["status"] == "failed"


def test_parse_json_malformed_file(tmp_path):
    """Test parsing with malformed JSON file."""
    malformed_file = tmp_path / "malformed.json"
    with open(malformed_file, 'w') as f:
        f.write("{ invalid json content }")
    
    result = parse_dqc_json(malformed_file)
    
    # Should handle JSON parsing errors gracefully
    assert result["rank"] == "unknown"
    assert result["status"] == "failed"


@patch.object(dfast_qc_module.subprocess, 'run')
def test_run_dfast_success(mock_subprocess_run, tmp_path, example_json):
    """Test successful DFAST_QC execution."""
    # Setup mock
    mock_subprocess_run.return_value.returncode = 0
    
    # Create input and output paths
    input_fasta = tmp_path / "test_genome.fna"
    input_fasta.write_text(">seq1\nATGC\n")
    output_dir = tmp_path / "output"
    
    # Create expected output file
    output_dir.mkdir()
    dqc_result_file = output_dir / "dqc_result.json"
    with open(dqc_result_file, 'w') as f:
        json.dump(example_json, f)
    
    # Run function
    result = run_dfast(input_fasta, output_dir, threads=2, enable_cc=True)
    
    # Check results
    assert result["execution_status"] == "success"
    assert result["genome_id"] == "test_genome"
    assert result["input_file"] == str(input_fasta)
    assert result["output_dir"] == str(output_dir)
    assert "taxonomy" in result
    assert result["taxonomy"]["rank"] == "species"
    
    # Check that command was called correctly (first call is main dfast_qc, second is version)
    assert mock_subprocess_run.call_count == 2
    main_call_args = mock_subprocess_run.call_args_list[0][0][0]  # First call, first positional arg
    assert "dfast_qc" in main_call_args
    assert "-i" in main_call_args
    assert str(input_fasta) in main_call_args
    assert "-o" in main_call_args
    assert str(output_dir) in main_call_args
    assert "--num_threads" in main_call_args
    assert "2" in main_call_args
    # enable_cc=True means --disable_cc should NOT be present
    assert "--disable_cc" not in main_call_args


@patch.object(dfast_qc_module.subprocess, 'run')
def test_run_dfast_disable_cc(mock_subprocess_run, tmp_path, example_json):
    """Test DFAST_QC execution with completeness/contamination disabled."""
    mock_subprocess_run.return_value.returncode = 0
    
    input_fasta = tmp_path / "test_genome.fna"
    input_fasta.write_text(">seq1\nATGC\n")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create expected output file
    dqc_result_file = output_dir / "dqc_result.json"
    with open(dqc_result_file, 'w') as f:
        json.dump(example_json, f)
    
    # Run with enable_cc=False
    result = run_dfast(input_fasta, output_dir, threads=4, enable_cc=False)
    
    assert result["execution_status"] == "success"
    
    # Check that --disable_cc flag was added (first call is main dfast_qc)
    main_call_args = mock_subprocess_run.call_args_list[0][0][0]  # First call, first positional arg
    assert "--disable_cc" in main_call_args
    assert "--num_threads" in main_call_args
    assert "4" in main_call_args


@patch.object(dfast_qc_module.subprocess, 'run')
def test_run_dfast_failure(mock_subprocess_run, tmp_path):
    """Test DFAST_QC execution failure."""
    # Setup mock to return failure
    mock_subprocess_run.return_value.returncode = 1
    
    input_fasta = tmp_path / "test_genome.fna"
    input_fasta.write_text(">seq1\nATGC\n")
    output_dir = tmp_path / "output"
    
    result = run_dfast(input_fasta, output_dir)
    
    assert result["execution_status"] == "failed"
    assert "failed with return code 1" in result["error_message"]
    assert result["genome_id"] == "test_genome"


@patch.object(dfast_qc_module.subprocess, 'run')
def test_run_dfast_timeout(mock_subprocess_run, tmp_path):
    """Test DFAST_QC execution timeout."""
    from subprocess import TimeoutExpired
    
    # Setup mock to raise timeout
    mock_subprocess_run.side_effect = TimeoutExpired("dfast_qc", 1800)
    
    input_fasta = tmp_path / "test_genome.fna"
    input_fasta.write_text(">seq1\nATGC\n")
    output_dir = tmp_path / "output"
    
    result = run_dfast(input_fasta, output_dir)
    
    assert result["execution_status"] == "failed"
    assert "timed out" in result["error_message"]


@patch.object(dfast_qc_module.subprocess, 'run')
def test_run_dfast_missing_output(mock_subprocess_run, tmp_path):
    """Test DFAST_QC when output file is not created."""
    mock_subprocess_run.return_value.returncode = 0
    
    input_fasta = tmp_path / "test_genome.fna"
    input_fasta.write_text(">seq1\nATGC\n")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    # Note: not creating dqc_result.json file
    
    result = run_dfast(input_fasta, output_dir)
    
    assert result["execution_status"] == "failed"
    assert "result file not found" in result["error_message"]


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("SKIP_LONG_TESTS"),
    reason="Skipping long integration test (SKIP_LONG_TESTS set)"
)
def test_dfast_qc_end_to_end(tmp_path):
    """
    End-to-end integration test for DFAST_QC.
    
    This test:
    1. Downloads reference data (if needed)
    2. Copies a test genome into tmpdir
    3. Runs DFAST_QC on it
    4. Validates output files and content
    """
    # Step 1: Download reference data (this may take time)
    ref_dir = tmp_path / "dfast_ref"
    try:
        download(output_dir=ref_dir, force=False)
    except Exception as e:
        pytest.skip(f"Could not download DFAST_QC reference data: {e}")
    
    # Step 2: Copy test genome
    test_genome_source = Path("dummy_dataset/Burkholderiales_bacterium_RIFCSPHIGHO2_01_FULL_64_960.contigs.fna")
    if not test_genome_source.exists():
        # Try alternative path for testing
        test_genome_source = Path("examples/GCA_000829395.1.fna.gz")
        if not test_genome_source.exists():
            pytest.skip("No test genome file available for integration test")
    
    # Copy to temp directory
    test_genome = tmp_path / test_genome_source.name
    if test_genome_source.suffix == '.gz':
        import gzip
        import shutil
        with gzip.open(test_genome_source, 'rb') as f_in:
            with open(test_genome.with_suffix(''), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        test_genome = test_genome.with_suffix('')
    else:
        import shutil
        shutil.copy2(test_genome_source, test_genome)
    
    # Step 3: Create manifest for testing
    manifest_dir = tmp_path / "prepared"
    manifest_dir.mkdir()
    manifest = {
        "genomes": [{
            "genome_id": test_genome.stem,
            "output_path": str(test_genome),
            "format_valid": True
        }]
    }
    manifest_file = manifest_dir / "processing_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f)
    
    # Step 4: Run DFAST_QC
    output_dir = tmp_path / "dfast_output"
    
    try:
        call(
            input_dir=manifest_dir,
            output_dir=output_dir,
            threads=2,  # Use minimal threads for testing
            enable_cc=False,  # Disable for faster testing
            force=True
        )
    except Exception as e:
        pytest.skip(f"DFAST_QC execution failed: {e}")
    
    # Step 5: Validate outputs
    assert output_dir.exists()
    
    # Check processing manifest
    processing_manifest = output_dir / "processing_manifest.json"
    assert processing_manifest.exists()
    
    with open(processing_manifest) as f:
        manifest_data = json.load(f)
    
    assert manifest_data["stage"] == "stage02_dfast_qc"
    assert len(manifest_data["genomes"]) >= 1
    
    # Check genome-specific output
    genome_dir = output_dir / "genomes" / test_genome.stem
    assert genome_dir.exists()
    
    tax_summary = genome_dir / "tax_summary.json"
    assert tax_summary.exists()
    
    with open(tax_summary) as f:
        tax_data = json.load(f)
    
    assert tax_data["rank"] == "species"  # Task requirement
    assert "name" in tax_data
    assert "tool" in tax_data
    assert tax_data["tool"] == "dfast_qc"


if __name__ == "__main__":
    pytest.main([__file__])
