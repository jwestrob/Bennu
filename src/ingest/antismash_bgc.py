#!/usr/bin/env python3
"""
Stage 5a: AntiSMASH Biosynthetic Gene Cluster Detection
Analyze genome assemblies for biosynthetic gene clusters using AntiSMASH.
"""

import logging
import json
import subprocess
import time
import shutil
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

console = Console()
logger = logging.getLogger(__name__)


def parse_antismash_genbank(gbk_file: Path) -> Dict[str, Any]:
    """
    Parse AntiSMASH GenBank output to extract BGC information.
    
    Args:
        gbk_file: Path to AntiSMASH GenBank output file
        
    Returns:
        Dict containing BGC data and gene annotations
    """
    bgc_data = {
        "file": str(gbk_file),
        "clusters": [],
        "genes": [],
        "parsing_errors": []
    }
    
    try:
        for record in SeqIO.parse(gbk_file, "genbank"):
            # Extract cluster information from record annotations
            if "structured_comment" in record.annotations:
                structured_comment = record.annotations["structured_comment"]
                if "antiSMASH-Data" in structured_comment:
                    antismash_data = structured_comment["antiSMASH-Data"]
                    
                    # Parse cluster information
                    for key, value in antismash_data.items():
                        if key.startswith("Cluster_"):
                            cluster_info = {
                                "cluster_id": key,
                                "description": value,
                                "record_id": record.id,
                                "record_length": len(record.seq)
                            }
                            bgc_data["clusters"].append(cluster_info)
            
            # Extract gene features
            for feature in record.features:
                if feature.type in ["CDS", "gene"]:
                    gene_info = {
                        "record_id": record.id,
                        "feature_type": feature.type,
                        "start": int(feature.location.start),
                        "end": int(feature.location.end),
                        "strand": feature.location.strand,
                        "qualifiers": dict(feature.qualifiers)
                    }
                    
                    # Extract protein ID if available
                    if "protein_id" in feature.qualifiers:
                        gene_info["protein_id"] = feature.qualifiers["protein_id"][0]
                    
                    # Extract gene product/function
                    if "product" in feature.qualifiers:
                        gene_info["product"] = feature.qualifiers["product"][0]
                    
                    # Extract BGC-specific annotations
                    if "gene_kind" in feature.qualifiers:
                        gene_info["gene_kind"] = feature.qualifiers["gene_kind"][0]
                    
                    if "sec_met_domain" in feature.qualifiers:
                        gene_info["sec_met_domains"] = feature.qualifiers["sec_met_domain"]
                    
                    bgc_data["genes"].append(gene_info)
                
                elif feature.type == "cluster":
                    # Extract cluster boundary information
                    cluster_info = {
                        "type": "cluster",
                        "record_id": record.id,
                        "start": int(feature.location.start),
                        "end": int(feature.location.end),
                        "qualifiers": dict(feature.qualifiers)
                    }
                    
                    # Extract cluster type and product
                    if "product" in feature.qualifiers:
                        cluster_info["product"] = feature.qualifiers["product"][0]
                    
                    if "cluster_number" in feature.qualifiers:
                        cluster_info["cluster_number"] = feature.qualifiers["cluster_number"][0]
                    
                    bgc_data["clusters"].append(cluster_info)
    
    except Exception as e:
        bgc_data["parsing_errors"].append(f"Error parsing {gbk_file}: {str(e)}")
        logger.error(f"Failed to parse AntiSMASH output {gbk_file}: {e}")
    
    return bgc_data


def run_single_antismash_analysis(genome_file: Path, output_dir: Path, 
                                 threads: int = 4) -> Dict[str, Any]:
    """
    Run AntiSMASH analysis for a single genome.
    
    Args:
        genome_file: Path to genome FASTA file
        output_dir: Output directory for AntiSMASH results
        threads: Number of threads to use
        
    Returns:
        Dict containing execution results and BGC statistics
    """
    start_time = time.time()
    genome_name = genome_file.stem
    
    # Create genome-specific output directory
    genome_output_dir = output_dir / "genomes" / genome_name
    genome_output_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        "genome": genome_name,
        "genome_file": str(genome_file),
        "execution_status": "failed",
        "execution_time_seconds": 0.0,
        "error_message": None,
        "output_dir": str(genome_output_dir),
        "gbk_files": [],
        "total_clusters": 0,
        "total_genes": 0,
        "cluster_types": [],
        "bgc_data": None
    }
    
    try:
        # Check if AntiSMASH wrapper is available
        antismash_cmd = os.path.expanduser("~/bin/run_antismash")
        if not os.path.exists(antismash_cmd):
            result["error_message"] = f"AntiSMASH wrapper not found: {antismash_cmd}"
            return result
        
        # Build AntiSMASH command
        cmd = [
            antismash_cmd,
            str(genome_file),
            str(genome_output_dir),
            "--cpus", str(threads),
            "--genefinding-tool", "prodigal",
            "--output-basename", genome_name,
            "--logfile", str(genome_output_dir / "antismash.log")
        ]
        
        console.print(f"Running AntiSMASH for {genome_name}...")
        console.print(f"Command: {' '.join(cmd)}")
        
        # Execute AntiSMASH
        process_result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if process_result.returncode != 0:
            result["error_message"] = f"AntiSMASH failed: {process_result.stderr}"
            return result
        
        # Find and parse GenBank output files
        gbk_files = list(genome_output_dir.glob("*.gbk"))
        if not gbk_files:
            result["error_message"] = f"No GenBank output files found in {genome_output_dir}"
            return result
        
        result["gbk_files"] = [str(f) for f in gbk_files]
        
        # Parse BGC data from all GenBank files
        all_bgc_data = {
            "clusters": [],
            "genes": [],
            "parsing_errors": []
        }
        
        for gbk_file in gbk_files:
            bgc_data = parse_antismash_genbank(gbk_file)
            all_bgc_data["clusters"].extend(bgc_data["clusters"])
            all_bgc_data["genes"].extend(bgc_data["genes"])
            all_bgc_data["parsing_errors"].extend(bgc_data["parsing_errors"])
        
        result["bgc_data"] = all_bgc_data
        result["total_clusters"] = len(all_bgc_data["clusters"])
        result["total_genes"] = len(all_bgc_data["genes"])
        
        # Extract cluster types
        cluster_types = set()
        for cluster in all_bgc_data["clusters"]:
            if "product" in cluster:
                cluster_types.add(cluster["product"])
        result["cluster_types"] = list(cluster_types)
        
        result["execution_status"] = "success"
        
    except subprocess.TimeoutExpired:
        result["error_message"] = f"AntiSMASH analysis for {genome_name} timed out (>1 hour)"
    except Exception as e:
        result["error_message"] = f"Unexpected error: {str(e)}"
    
    result["execution_time_seconds"] = round(time.time() - start_time, 2)
    return result


def run_antismash_analysis(
    input_dir: Path = typer.Option(
        Path("data/stage00_prepared"),
        "--input-dir", "-i",
        help="Directory containing prepared genome assemblies"
    ),
    output_dir: Path = typer.Option(
        Path("data/stage05a_antismash"),
        "--output-dir", "-o",
        help="Output directory for AntiSMASH results"
    ),
    threads: int = typer.Option(
        4,
        "--threads", "-t",
        help="Number of threads to use per genome"
    ),
    max_parallel: int = typer.Option(
        2,
        "--max-parallel", "-p",
        help="Maximum number of genomes to process in parallel"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing output directory"
    )
) -> None:
    """
    Run AntiSMASH biosynthetic gene cluster detection on genome assemblies.
    
    Analyzes genome assemblies for biosynthetic gene clusters using AntiSMASH
    in Docker containers. Processes multiple genomes in parallel with configurable
    resource limits.
    """
    console.print("[bold blue]Stage 5a: AntiSMASH BGC Detection[/bold blue]")
    
    # Validate inputs
    if not input_dir.exists():
        console.print(f"[red]Error: Input directory does not exist: {input_dir}[/red]")
        raise typer.Exit(1)
    
    # Check for input manifest
    manifest_file = input_dir / "processing_manifest.json"
    if not manifest_file.exists():
        console.print(f"[red]Error: Input manifest not found: {manifest_file}[/red]")
        raise typer.Exit(1)
    
    # Load input manifest
    try:
        with open(manifest_file, 'r') as f:
            input_manifest = json.load(f)
    except Exception as e:
        console.print(f"[red]Error: Failed to load input manifest: {e}[/red]")
        raise typer.Exit(1)
    
    # Find genome files
    genome_files = []
    for ext in ["*.fna", "*.fasta", "*.fa"]:
        genome_files.extend(input_dir.glob(ext))
    
    if not genome_files:
        console.print(f"[red]Error: No genome files found in {input_dir}[/red]")
        raise typer.Exit(1)
    
    console.print(f"Found {len(genome_files)} genome files to process")
    console.print(f"Threads per genome: {threads}")
    console.print(f"Max parallel genomes: {max_parallel}")
    
    # Handle output directory
    if output_dir.exists():
        if not force:
            console.print(f"[red]Error: Output directory already exists: {output_dir}[/red]")
            console.print("[yellow]Use --force to overwrite[/yellow]")
            raise typer.Exit(1)
        else:
            console.print(f"[yellow]Removing existing output directory: {output_dir}[/yellow]")
            shutil.rmtree(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "genomes").mkdir(exist_ok=True)
    
    # Process genomes in parallel
    console.print("\\n[bold yellow]Running AntiSMASH analyses...[/bold yellow]")
    
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        # Submit all jobs
        future_to_genome = {
            executor.submit(
                run_single_antismash_analysis,
                genome_file,
                output_dir,
                threads
            ): genome_file for genome_file in genome_files
        }
        
        # Process completed jobs
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing genomes...", total=len(genome_files))
            
            for future in as_completed(future_to_genome):
                genome_file = future_to_genome[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Show result
                    status = "✓" if result["execution_status"] == "success" else "✗"
                    clusters = result.get("total_clusters", 0)
                    time_taken = result["execution_time_seconds"]
                    console.print(f"{status} {result['genome']}: {clusters} clusters in {time_taken:.1f}s")
                    
                    if result["execution_status"] == "failed":
                        console.print(f"[red]  Error: {result.get('error_message', 'Unknown error')}[/red]")
                    
                except Exception as e:
                    error_result = {
                        "genome": genome_file.stem,
                        "genome_file": str(genome_file),
                        "execution_status": "failed",
                        "error_message": f"Execution error: {str(e)}",
                        "execution_time_seconds": 0.0
                    }
                    results.append(error_result)
                    console.print(f"✗ {genome_file.stem}: Execution failed - {str(e)}")
                
                progress.advance(task)
    
    total_time = time.time() - start_time
    
    # Generate summary statistics
    successful_results = [r for r in results if r["execution_status"] == "success"]
    failed_results = [r for r in results if r["execution_status"] == "failed"]
    
    total_clusters = sum(r.get("total_clusters", 0) for r in successful_results)
    total_genes = sum(r.get("total_genes", 0) for r in successful_results)
    
    # Collect all cluster types
    all_cluster_types = set()
    for result in successful_results:
        all_cluster_types.update(result.get("cluster_types", []))
    
    summary_stats = {
        "total_genomes": len(genome_files),
        "successful_genomes": len(successful_results),
        "failed_genomes": len(failed_results),
        "total_clusters": total_clusters,
        "total_genes": total_genes,
        "unique_cluster_types": list(all_cluster_types),
        "total_execution_time_seconds": round(total_time, 2)
    }
    
    # Create processing manifest
    manifest = {
        "version": "0.1.0",
        "stage": "stage05a_antismash",
        "timestamp": datetime.now().isoformat(),
        "input_manifest": str(manifest_file.absolute()),
        "execution_parameters": {
            "threads": threads,
            "max_parallel": max_parallel,
            "input_dir": str(input_dir)
        },
        "summary": summary_stats,
        "genome_results": results
    }
    
    # Save manifest
    output_manifest = output_dir / "processing_manifest.json"
    with open(output_manifest, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Save summary statistics
    summary_file = output_dir / "summary_stats.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Save combined BGC data
    combined_bgc_data = {
        "clusters": [],
        "genes": [],
        "parsing_errors": []
    }
    
    for result in successful_results:
        if result.get("bgc_data"):
            combined_bgc_data["clusters"].extend(result["bgc_data"]["clusters"])
            combined_bgc_data["genes"].extend(result["bgc_data"]["genes"])
            combined_bgc_data["parsing_errors"].extend(result["bgc_data"]["parsing_errors"])
    
    bgc_data_file = output_dir / "combined_bgc_data.json"
    with open(bgc_data_file, 'w') as f:
        json.dump(combined_bgc_data, f, indent=2)
    
    # Display results
    console.print("\\n[bold green]Stage 5a Results Summary[/bold green]")
    
    summary_table = Table()
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="magenta")
    
    summary_table.add_row("Input directory", str(input_dir))
    summary_table.add_row("Output directory", str(output_dir))
    summary_table.add_row("Total genomes", str(summary_stats["total_genomes"]))
    summary_table.add_row("Successful genomes", str(summary_stats["successful_genomes"]))
    summary_table.add_row("Failed genomes", str(summary_stats["failed_genomes"]))
    summary_table.add_row("Total BGCs detected", f"{summary_stats['total_clusters']:,}")
    summary_table.add_row("Total genes in BGCs", f"{summary_stats['total_genes']:,}")
    summary_table.add_row("Unique cluster types", str(len(summary_stats["unique_cluster_types"])))
    summary_table.add_row("Execution time", f"{summary_stats['total_execution_time_seconds']:.1f} seconds")
    
    console.print(summary_table)
    
    # Show cluster types found
    if all_cluster_types:
        console.print("\\n[bold cyan]BGC Types Detected:[/bold cyan]")
        for cluster_type in sorted(all_cluster_types):
            console.print(f"• {cluster_type}")
    
    # Show failed genomes if any
    if failed_results:
        console.print("\\n[bold red]Failed Genomes:[/bold red]")
        for result in failed_results:
            console.print(f"[red]• {result['genome']}: {result.get('error_message', 'Unknown error')}[/red]")
    
    # Success message
    if successful_results:
        console.print(f"\\n[bold green]✓ Stage 5a completed successfully![/bold green]")
        console.print(f"BGC detection results available in: {output_dir}/genomes/*/")
        console.print(f"Combined BGC data saved to: {bgc_data_file}")
        
    logger.info(f"Stage 5a completed: {len(successful_results)} successful, {len(failed_results)} failed")


def main():
    """Entry point for standalone execution."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Stage 5a: AntiSMASH BGC Detection")
    typer.run(run_antismash_analysis)


if __name__ == "__main__":
    main()