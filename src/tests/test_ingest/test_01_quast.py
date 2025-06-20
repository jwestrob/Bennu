#!/usr/bin/env python3
"""
Tests for Stage 1: QUAST Quality Assessment
"""

import json
import pytest
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import tempfile
import shutil

import importlib
run_quast_module = importlib.import_module('src.ingest.01_run_quast')
run_quast = run_quast_module.run_quast
run_quast_single = run_quast_module.run_quast_single
parse_quast_report = run_quast_module.parse_quast_report
validate_quast_outputs = run_quast_module.validate_quast_outputs
process_genomes_parallel = run_quast_module.process_genomes_parallel
generate_summary_stats = run_quast_module.generate_summary_stats


class TestParseQuastReport:
    """Test QUAST report parsing functionality."""
    
    def test_parse_quast_report_valid_file(self, tmp_path):
        """Test parsing of valid QUAST report file."""
        report_file = tmp_path / "report.tsv"
        report_content = """Assembly	test_genome
# contigs	45
Total length	1234567
Largest contig	123456
N50	12345
N75	5678
GC (%)	42.5
# N's	100
# N's per 100 kbp	8.1
# contigs (>= 1000 bp)	40
# contigs (>= 5000 bp)	25
# contigs (>= 10000 bp)	15
L50	12
L75	25
"""
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        stats = parse_quast_report(report_file)
        
        assert stats["contigs"] == 45
        assert stats["total_length"] == 1234567
        assert stats["largest_contig"] == 123456
        assert stats["n50"] == 12345
        assert stats["n75"] == 5678
        assert stats["gc_content"] == 0.425
        assert stats["n_count"] == 100
        assert stats["n_per_100_kbp"] == 8.1
        assert stats["contigs_1000bp"] == 40
        assert stats["contigs_5000bp"] == 25
        assert stats["contigs_10000bp"] == 15
        assert stats["l50"] == 12
        assert stats["l75"] == 25
    
    def test_parse_quast_report_missing_file(self, tmp_path):
        """Test parsing when report file doesn't exist."""
        missing_file = tmp_path / "missing_report.tsv"
        
        stats = parse_quast_report(missing_file)
        
        # Should return default stats dict
        assert stats["contigs"] == 0
        assert stats["total_length"] == 0
        assert stats["n50"] == 0
        assert stats["gc_content"] == 0.0
    
    def test_parse_quast_report_empty_file(self, tmp_path):
        """Test parsing of empty report file."""
        report_file = tmp_path / "empty_report.tsv"
        report_file.touch()
        
        stats = parse_quast_report(report_file)
        
        # Should return default stats dict
        assert stats["contigs"] == 0
        assert stats["total_length"] == 0
    
    def test_parse_quast_report_malformed_data(self, tmp_path):
        """Test parsing with malformed data."""
        report_file = tmp_path / "malformed_report.tsv"
        report_content = """# contigs	not_a_number
Total length	1,234,567
Invalid line without tab
GC (%)	invalid_percent
"""
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        stats = parse_quast_report(report_file)
        
        # Should handle malformed data gracefully
        assert stats["contigs"] == 0  # Could not parse "not_a_number"
        assert stats["total_length"] == 1234567  # Should handle commas
        assert stats["gc_content"] == 0.0  # Could not parse "invalid_percent"
    
    def test_parse_quast_report_partial_data(self, tmp_path):
        """Test parsing with only some metrics present."""
        report_file = tmp_path / "partial_report.tsv"
        report_content = """# contigs	30
N50	8500
GC (%)	38.2
"""
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        stats = parse_quast_report(report_file)
        
        assert stats["contigs"] == 30
        assert stats["n50"] == 8500
        assert stats["gc_content"] == 0.382
        # Other metrics should remain at defaults
        assert stats["total_length"] == 0
        assert stats["n75"] == 0


class TestValidateQuastOutputs:
    """Test QUAST output validation functionality."""
    
    def test_validate_quast_outputs_complete(self, tmp_path):
        """Test validation with all expected outputs present."""
        genome_id = "test_genome"
        genome_dir = tmp_path / genome_id
        genome_dir.mkdir()
        
        # Create expected files
        (genome_dir / "report.tsv").write_text("# contigs\t10\n")
        basic_stats_dir = genome_dir / "basic_stats"
        basic_stats_dir.mkdir()
        (basic_stats_dir / "stats.txt").write_text("test data")
        
        validation = validate_quast_outputs(genome_dir, genome_id)
        
        assert validation["output_files_exist"] is True
        assert validation["report_valid"] is True
        assert validation["has_plots"] is True
        assert len(validation["missing_files"]) == 0
        assert validation["report_path"] is not None
    
    def test_validate_quast_outputs_missing_report(self, tmp_path):
        """Test validation with missing report file."""
        genome_id = "test_genome"
        genome_dir = tmp_path / genome_id
        genome_dir.mkdir()
        
        validation = validate_quast_outputs(genome_dir, genome_id)
        
        assert validation["output_files_exist"] is False
        assert validation["report_valid"] is False
        assert len(validation["missing_files"]) == 1
        assert str(genome_dir / "report.tsv") in validation["missing_files"]
    
    def test_validate_quast_outputs_empty_report(self, tmp_path):
        """Test validation with empty report file."""
        genome_id = "test_genome"
        genome_dir = tmp_path / genome_id
        genome_dir.mkdir()
        
        # Create empty report file
        (genome_dir / "report.tsv").touch()
        
        validation = validate_quast_outputs(genome_dir, genome_id)
        
        assert validation["output_files_exist"] is True
        assert validation["report_valid"] is False
    
    def test_validate_quast_outputs_no_plots(self, tmp_path):
        """Test validation with no plots directory."""
        genome_id = "test_genome"
        genome_dir = tmp_path / genome_id
        genome_dir.mkdir()
        
        # Create report but no plots
        (genome_dir / "report.tsv").write_text("# contigs\t10\n")
        
        validation = validate_quast_outputs(genome_dir, genome_id)
        
        assert validation["output_files_exist"] is True
        assert validation["report_valid"] is True
        assert validation["has_plots"] is False


class TestRunQuastSingle:
    """Test single genome QUAST execution."""
    
    @patch.object(run_quast_module, 'subprocess')
    def test_run_quast_single_success(self, mock_subprocess, tmp_path):
        """Test successful QUAST execution on single genome."""
        # Setup
        output_dir = tmp_path / "output"
        genome_info = {
            "genome_id": "test_genome",
            "output_path": str(tmp_path / "test_genome.fna")
        }
        
        # Create input file
        input_file = Path(genome_info["output_path"])
        input_file.write_text(">contig1\nATCG\n")
        
        # Create expected output files
        genome_output_dir = output_dir / "genomes" / "test_genome"
        genome_output_dir.mkdir(parents=True)
        
        report_file = genome_output_dir / "report.tsv"
        report_file.write_text("# contigs\t10\nN50\t5000\n")
        
        # Mock successful subprocess calls
        mock_main_result = Mock()
        mock_main_result.returncode = 0
        
        mock_version_result = Mock()
        mock_version_result.returncode = 0
        mock_version_result.stdout = "QUAST v5.0.2"
        
        mock_subprocess.run.side_effect = [
            mock_main_result,  # Main QUAST command
            mock_version_result  # Version command
        ]
        
        result = run_quast_single(genome_info, output_dir)
        
        assert result["execution_status"] == "success"
        assert result["genome_id"] == "test_genome"
        assert result["statistics"]["contigs"] == 10
        assert result["statistics"]["n50"] == 5000
        assert result["error_message"] is None
        assert result["execution_time_seconds"] >= 0
    
    @patch.object(run_quast_module, 'subprocess')
    def test_run_quast_single_failure(self, mock_subprocess, tmp_path):
        """Test QUAST execution failure."""
        output_dir = tmp_path / "output"
        genome_info = {
            "genome_id": "test_genome",
            "output_path": str(tmp_path / "test_genome.fna")
        }
        
        # Create input file
        input_file = Path(genome_info["output_path"])
        input_file.write_text(">contig1\nATCG\n")
        
        # Mock failed subprocess
        mock_result = Mock()
        mock_result.returncode = 1
        mock_subprocess.run.return_value = mock_result
        
        result = run_quast_single(genome_info, output_dir)
        
        assert result["execution_status"] == "failed"
        assert result["error_message"] == "QUAST failed with return code 1"
    
    @patch.object(run_quast_module, 'subprocess')
    def test_run_quast_single_timeout(self, mock_subprocess, tmp_path):
        """Test QUAST execution timeout."""
        output_dir = tmp_path / "output"
        genome_info = {
            "genome_id": "test_genome",
            "output_path": str(tmp_path / "test_genome.fna")
        }
        
        # Create input file
        input_file = Path(genome_info["output_path"])
        input_file.write_text(">contig1\nATCG\n")
        
        # Mock timeout - need to preserve the real TimeoutExpired class
        import subprocess as real_subprocess
        mock_subprocess.TimeoutExpired = real_subprocess.TimeoutExpired
        mock_subprocess.run.side_effect = real_subprocess.TimeoutExpired("quast.py", 600)
        
        result = run_quast_single(genome_info, output_dir)
        
        assert result["execution_status"] == "failed"
        assert "timed out" in result["error_message"]
    
    @patch.object(run_quast_module, 'subprocess')
    def test_run_quast_single_with_reference(self, mock_subprocess, tmp_path):
        """Test QUAST execution with reference genome."""
        output_dir = tmp_path / "output"
        reference_file = tmp_path / "reference.fna"
        reference_file.write_text(">ref\nATCGATCG\n")
        
        genome_info = {
            "genome_id": "test_genome",
            "output_path": str(tmp_path / "test_genome.fna")
        }
        
        # Create input file
        input_file = Path(genome_info["output_path"])
        input_file.write_text(">contig1\nATCG\n")
        
        # Create expected output files
        genome_output_dir = output_dir / "genomes" / "test_genome"
        genome_output_dir.mkdir(parents=True)
        (genome_output_dir / "report.tsv").write_text("# contigs\t5\n")
        
        # Mock successful subprocess calls
        mock_main_result = Mock()
        mock_main_result.returncode = 0
        
        mock_version_result = Mock()
        mock_version_result.returncode = 0
        mock_version_result.stdout = "QUAST v5.0.2"
        
        mock_subprocess.run.side_effect = [
            mock_main_result,  # Main QUAST command
            mock_version_result  # Version command
        ]
        
        result = run_quast_single(
            genome_info, 
            output_dir, 
            reference_genome=reference_file
        )
        
        # Verify reference was included in command
        call_args = mock_subprocess.run.call_args_list[0][0][0]
        assert "--reference" in call_args
        assert str(reference_file) in call_args


class MockProcessPoolExecutor:
    """Mock ProcessPoolExecutor that runs tasks synchronously to preserve mock behavior."""
    
    def __init__(self, max_workers=None):
        self.max_workers = max_workers
        self._futures = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def submit(self, fn, *args, **kwargs):
        """Submit a task and return a mock future with the result."""
        future = Mock()
        try:
            result = fn(*args, **kwargs)
            future.result.return_value = result
        except Exception as e:
            future.result.side_effect = e
        
        self._futures.append(future)
        return future


def mock_as_completed(futures):
    """Mock version of as_completed that returns futures immediately."""
    return futures


class TestProcessGenomesParallel:
    """Test parallel genome processing functionality."""
    
    @patch.object(run_quast_module, 'ProcessPoolExecutor', MockProcessPoolExecutor)
    @patch.object(run_quast_module, 'as_completed', mock_as_completed)
    @patch.object(run_quast_module, 'run_quast_single')
    def test_process_genomes_parallel_success(self, mock_run_single, tmp_path):
        """Test successful parallel processing of multiple genomes."""
        output_dir = tmp_path / "output"
        
        genomes = [
            {"genome_id": "genome1", "output_path": "/path/to/genome1.fna"},
            {"genome_id": "genome2", "output_path": "/path/to/genome2.fna"},
            {"genome_id": "genome3", "output_path": "/path/to/genome3.fna"}
        ]
        
        # Mock successful results
        mock_run_single.side_effect = [
            {"genome_id": "genome1", "execution_status": "success", "execution_time_seconds": 1.0},
            {"genome_id": "genome2", "execution_status": "success", "execution_time_seconds": 1.5},
            {"genome_id": "genome3", "execution_status": "success", "execution_time_seconds": 2.0}
        ]
        
        results = process_genomes_parallel(genomes, output_dir, max_workers=2)
        
        assert len(results) == 3
        assert all(r["execution_status"] == "success" for r in results)
        assert mock_run_single.call_count == 3
    
    @patch.object(run_quast_module, 'ProcessPoolExecutor', MockProcessPoolExecutor)
    @patch.object(run_quast_module, 'as_completed', mock_as_completed)
    @patch.object(run_quast_module, 'run_quast_single')
    def test_process_genomes_parallel_mixed_results(self, mock_run_single, tmp_path):
        """Test parallel processing with mixed success/failure results."""
        output_dir = tmp_path / "output"
        
        genomes = [
            {"genome_id": "genome1", "output_path": "/path/to/genome1.fna"},
            {"genome_id": "genome2", "output_path": "/path/to/genome2.fna"}
        ]
        
        # Mock mixed results
        mock_run_single.side_effect = [
            {"genome_id": "genome1", "execution_status": "success", "execution_time_seconds": 1.0},
            {"genome_id": "genome2", "execution_status": "failed", "error_message": "Test failure", "execution_time_seconds": 0.5}
        ]
        
        results = process_genomes_parallel(genomes, output_dir, max_workers=1)
        
        assert len(results) == 2
        assert results[0]["execution_status"] == "success"
        assert results[1]["execution_status"] == "failed"
    
    @patch.object(run_quast_module, 'run_quast_single')
    def test_process_genomes_parallel_exception_handling(self, mock_run_single, tmp_path):
        """Test parallel processing with exception handling."""
        output_dir = tmp_path / "output"
        
        genomes = [
            {"genome_id": "genome1", "output_path": "/path/to/genome1.fna"}
        ]
        
        # Mock exception
        mock_run_single.side_effect = Exception("Test exception")
        
        results = process_genomes_parallel(genomes, output_dir, max_workers=1)
        
        assert len(results) == 1
        assert results[0]["execution_status"] == "failed"
        assert "Execution error" in results[0]["error_message"]


class TestGenerateSummaryStats:
    """Test summary statistics generation."""
    
    def test_generate_summary_stats_all_success(self):
        """Test summary generation with all successful results."""
        results = [
            {
                "execution_status": "success",
                "execution_time_seconds": 1.5,
                "statistics": {"contigs": 10, "total_length": 1000, "n50": 100, "gc_content": 0.4}
            },
            {
                "execution_status": "success", 
                "execution_time_seconds": 2.0,
                "statistics": {"contigs": 20, "total_length": 2000, "n50": 200, "gc_content": 0.5}
            }
        ]
        
        summary = generate_summary_stats(results)
        
        assert summary["total_genomes"] == 2
        assert summary["successful"] == 2
        assert summary["failed"] == 0
        assert summary["success_rate"] == 1.0
        assert summary["total_contigs"] == 30
        assert summary["total_assembly_length"] == 3000
        assert summary["mean_n50"] == 150
        assert summary["mean_gc_content"] == 0.45
        assert summary["mean_execution_time_seconds"] == 1.75
    
    def test_generate_summary_stats_mixed_results(self):
        """Test summary generation with mixed success/failure results."""
        results = [
            {
                "execution_status": "success",
                "execution_time_seconds": 1.0,
                "statistics": {"contigs": 15, "total_length": 1500, "n50": 150, "gc_content": 0.42}
            },
            {
                "execution_status": "failed",
                "execution_time_seconds": 0.5,
                "error_message": "Test error"
            }
        ]
        
        summary = generate_summary_stats(results)
        
        assert summary["total_genomes"] == 2
        assert summary["successful"] == 1
        assert summary["failed"] == 1
        assert summary["success_rate"] == 0.5
        assert summary["total_contigs"] == 15
        assert summary["total_assembly_length"] == 1500
    
    def test_generate_summary_stats_empty_results(self):
        """Test summary generation with empty results."""
        results = []
        
        summary = generate_summary_stats(results)
        
        assert summary["total_genomes"] == 0
        assert summary["successful"] == 0
        assert summary["failed"] == 0
        assert summary["success_rate"] == 0.0
    
    def test_generate_summary_stats_no_valid_metrics(self):
        """Test summary generation when no genomes have valid metrics."""
        results = [
            {
                "execution_status": "success",
                "execution_time_seconds": 1.0,
                "statistics": {"contigs": 0, "total_length": 0, "n50": 0, "gc_content": 0}
            }
        ]
        
        summary = generate_summary_stats(results)
        
        assert summary["mean_n50"] == 0
        assert summary["mean_gc_content"] == 0


class TestRunQuastIntegration:
    """Integration tests for the main run_quast function."""
    
    def test_run_quast_missing_input_directory(self, tmp_path):
        """Test run_quast with missing input directory."""
        missing_dir = tmp_path / "missing"
        output_dir = tmp_path / "output"
        
        import typer
        with pytest.raises(typer.Exit):
            run_quast(input_dir=missing_dir, output_dir=output_dir)
    
    def test_run_quast_missing_manifest(self, tmp_path):
        """Test run_quast with missing manifest file."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        
        import typer
        with pytest.raises(typer.Exit):
            run_quast(input_dir=input_dir, output_dir=output_dir)
    
    def test_run_quast_invalid_manifest(self, tmp_path):
        """Test run_quast with invalid manifest file."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        
        # Create invalid manifest
        manifest_file = input_dir / "processing_manifest.json"
        manifest_file.write_text("invalid json")
        
        import typer
        with pytest.raises(typer.Exit):
            run_quast(input_dir=input_dir, output_dir=output_dir)
    
    def test_run_quast_no_valid_genomes(self, tmp_path):
        """Test run_quast with no valid genomes in manifest."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        
        # Create manifest with no valid genomes
        manifest = {
            "genomes": [
                {"genome_id": "invalid", "format_valid": False}
            ]
        }
        
        manifest_file = input_dir / "processing_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f)
        
        import typer
        with pytest.raises(typer.Exit):
            run_quast(input_dir=input_dir, output_dir=output_dir)
    
    def test_run_quast_missing_reference(self, tmp_path):
        """Test run_quast with missing reference genome."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        missing_ref = tmp_path / "missing_ref.fna"
        
        # Create valid manifest
        manifest = {
            "genomes": [
                {"genome_id": "test", "format_valid": True, "output_path": str(tmp_path / "test.fna")}
            ]
        }
        
        manifest_file = input_dir / "processing_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f)
        
        import typer
        with pytest.raises(typer.Exit):
            run_quast(
                input_dir=input_dir, 
                output_dir=output_dir,
                reference_genome=missing_ref
            )
    
    @patch.object(run_quast_module, 'process_genomes_parallel')
    def test_run_quast_output_directory_exists_no_force(self, mock_process, tmp_path):
        """Test run_quast when output directory exists without force flag."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()  # Create existing output directory
        
        # Create valid manifest
        manifest = {
            "genomes": [
                {"genome_id": "test", "format_valid": True, "output_path": str(tmp_path / "test.fna")}
            ]
        }
        
        manifest_file = input_dir / "processing_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f)
        
        import typer
        with pytest.raises(typer.Exit):
            run_quast(
                input_dir=input_dir, 
                output_dir=output_dir, 
                min_contig_length=500,
                threads_per_genome=1,
                max_workers=1,
                reference_genome=None,
                force=False
            )
    
    @patch.object(run_quast_module, 'process_genomes_parallel')
    @patch.object(run_quast_module, 'shutil')
    def test_run_quast_output_directory_exists_with_force(self, mock_shutil, mock_process, tmp_path):
        """Test run_quast when output directory exists with force flag."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()  # Create existing output directory
        
        # Create valid manifest
        manifest = {
            "genomes": [
                {"genome_id": "test", "format_valid": True, "output_path": str(tmp_path / "test.fna")}
            ]
        }
        
        manifest_file = input_dir / "processing_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f)
        
        # Mock successful processing
        mock_process.return_value = [
            {"genome_id": "test", "execution_status": "success", "execution_time_seconds": 1.0, "statistics": {}}
        ]
        
        run_quast(
            input_dir=input_dir, 
            output_dir=output_dir, 
            min_contig_length=500,
            threads_per_genome=1,
            max_workers=1,
            reference_genome=None,
            force=True
        )
        
        # Verify rmtree was called to remove existing directory
        mock_shutil.rmtree.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
