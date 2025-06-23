#!/usr/bin/env python3
"""
Tests for Stage 4: Astra Functional Annotation
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

import importlib
# Import module with numeric prefix using importlib
astra_module = importlib.import_module('src.ingest.04_astra_scan')
run_astra_scan = astra_module.run_astra_scan
run_single_astra_scan = astra_module.run_single_astra_scan


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def mock_astra_results():
    """Mock astra search results."""
    return pd.DataFrame({
        'sequence_id': ['protein_1', 'protein_2', 'protein_1'],
        'hmm_name': ['PF00001', 'PF00002', 'PF00003'],
        'bitscore': [100.5, 85.2, 75.8],
        'evalue': [1e-30, 1e-25, 1e-20],
        'c_evalue': [1e-33, 1e-28, 1e-23],
        'i_evalue': [1e-30, 1e-25, 1e-20],
        'env_from': [1, 10, 50],
        'env_to': [100, 150, 120],
        'dom_bitscore': [99.8, 84.9, 75.1]
    })


@pytest.fixture
def mock_prodigal_output(temp_dir):
    """Create mock prodigal output structure."""
    # Create prodigal-like directory structure
    prodigal_dir = temp_dir / "stage03_prodigal"
    prodigal_dir.mkdir()
    
    genomes_dir = prodigal_dir / "genomes"
    genomes_dir.mkdir()
    
    # Create symlink directory
    symlink_dir = genomes_dir / "all_protein_symlinks"
    symlink_dir.mkdir()
    
    # Create dummy protein files
    protein_files = ["genome_A.faa", "genome_B.faa"]
    for protein_file in protein_files:
        faa_file = symlink_dir / protein_file
        with open(faa_file, 'w') as f:
            f.write(">protein_1\nMKLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL\n")
            f.write(">protein_2\nMKVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV\n")
    
    # Create processing manifest
    manifest = {
        "version": "0.1.0",
        "stage": "stage03_prodigal",
        "genomes": [
            {"genome_id": "genome_A", "execution_status": "success"},
            {"genome_id": "genome_B", "execution_status": "success"}
        ]
    }
    
    manifest_file = prodigal_dir / "processing_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f)
    
    return prodigal_dir


@pytest.mark.unit
def test_run_single_astra_scan_success(temp_dir, mock_astra_results):
    """Test successful single astra scan."""
    # Setup
    protein_dir = temp_dir / "proteins"
    protein_dir.mkdir()
    output_dir = temp_dir / "output"
    output_dir.mkdir()
    
    # Create dummy protein file
    protein_file = protein_dir / "test.faa"
    with open(protein_file, 'w') as f:
        f.write(">protein_1\nMKLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL\n")
    
    # Mock subprocess.run to simulate successful astra execution
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        
        # Create expected results file
        db_output_dir = output_dir / "pfam_results"
        db_output_dir.mkdir()
        results_file = db_output_dir / "PFAM_hits_df.tsv"
        mock_astra_results.to_csv(results_file, sep='\t', index=False)
        
        # Execute function
        result = run_single_astra_scan(
            database="PFAM",
            protein_symlink_dir=protein_dir,
            output_dir=output_dir,
            threads=4,
            use_cutoffs=True
        )
        
        # Verify results
        assert result["execution_status"] == "success"
        assert result["database"] == "PFAM"
        assert result["total_hits"] == 3
        assert result["unique_proteins"] == 2
        assert result["unique_domains"] == 3
        assert result["execution_time_seconds"] >= 0  # Allow zero for mocked execution
        
        # Verify astra command was called correctly
        mock_run.assert_called_once()
        called_args = mock_run.call_args[0][0]
        assert "astra" in called_args
        assert "search" in called_args
        assert "--cut_ga" in called_args


@pytest.mark.unit
def test_run_single_astra_scan_failure(temp_dir):
    """Test astra scan failure handling."""
    protein_dir = temp_dir / "proteins"
    protein_dir.mkdir()
    output_dir = temp_dir / "output"
    output_dir.mkdir()
    
    # Mock subprocess.run to simulate failure
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(
            returncode=1, 
            stderr="Astra search failed",
            stdout=""
        )
        
        result = run_single_astra_scan(
            database="PFAM",
            protein_symlink_dir=protein_dir,
            output_dir=output_dir,
            threads=4,
            use_cutoffs=True
        )
        
        assert result["execution_status"] == "failed"
        assert "Astra search failed" in result["error_message"]


@pytest.mark.unit
def test_run_astra_scan_validation_errors(temp_dir):
    """Test input validation in run_astra_scan."""
    from typer.testing import CliRunner
    import typer
    
    runner = CliRunner()
    app = typer.Typer()
    app.command()(run_astra_scan)
    
    # Test missing input directory
    result = runner.invoke(app, [
        "--input-dir", str(temp_dir / "nonexistent"),
        "--output-dir", str(temp_dir / "output")
    ])
    assert result.exit_code == 1
    assert "does not exist" in result.stdout


@pytest.mark.integration
def test_run_astra_scan_full_workflow(mock_prodigal_output, temp_dir, mock_astra_results):
    """Test full astra scan workflow with mocked astra execution."""
    output_dir = temp_dir / "astra_output"
    
    # Mock subprocess.run for astra execution
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        
        # Mock results files creation
        def create_results_file(*args, **kwargs):
            # Extract output directory from command
            cmd_args = args[0]
            outdir_idx = cmd_args.index('--outdir') + 1
            db_outdir = Path(cmd_args[outdir_idx])
            
            # Get database name
            db_idx = cmd_args.index('--installed_hmms') + 1
            database = cmd_args[db_idx]
            
            # Create results file
            results_file = db_outdir / f"{database}_hits_df.tsv"
            results_file.parent.mkdir(parents=True, exist_ok=True)
            mock_astra_results.to_csv(results_file, sep='\t', index=False)
            
            return MagicMock(returncode=0, stderr="", stdout="")
        
        mock_run.side_effect = create_results_file
        
        # Execute function
        from typer.testing import CliRunner
        import typer
        
        runner = CliRunner()
        app = typer.Typer()
        app.command()(run_astra_scan)
        
        result = runner.invoke(app, [
            "--input-dir", str(mock_prodigal_output),
            "--output-dir", str(output_dir),
            "--databases", "PFAM",
            "--threads", "4",
            "--force"
        ])
        
        # Verify execution
        assert result.exit_code == 0
        assert "✓ Stage 4 completed successfully!" in result.stdout
        
        # Verify output files
        assert output_dir.exists()
        assert (output_dir / "processing_manifest.json").exists()
        assert (output_dir / "summary_stats.json").exists()
        
        # Verify manifest content
        with open(output_dir / "processing_manifest.json") as f:
            manifest = json.load(f)
        
        assert manifest["stage"] == "stage04_astra"
        assert manifest["summary"]["successful_databases"] == 1
        assert manifest["summary"]["total_hits"] == 3


@pytest.mark.unit
def test_astra_command_building(temp_dir):
    """Test that astra commands are built correctly."""
    # Already imported at module level
    
    test_input = temp_dir / "test_input"
    test_output = temp_dir / "test_output"
    test_input.mkdir()
    test_output.mkdir()
    
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="test", stdout="")
        
        # Test PFAM with cutoffs
        run_single_astra_scan("PFAM", test_input, test_output, 4, True)
        called_args = mock_run.call_args[0][0]
        assert "--cut_ga" in called_args
        
        # Test database without cutoffs
        run_single_astra_scan("CUSTOM_DB", test_input, test_output, 4, True)
        called_args = mock_run.call_args[0][0] 
        assert "--cut_ga" not in called_args