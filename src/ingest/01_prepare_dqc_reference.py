#!/usr/bin/env python3
"""
DFAST_QC Reference Data Helper
Download and manage DFAST_QC reference databases.
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


def download(
    output_dir: Path = typer.Option(
        Path("data/dfast_qc_reference"),
        "--output-dir", "-o",
        help="Directory to store DFAST_QC reference data"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force re-download even if data already exists"
    )
) -> None:
    """
    Download DFAST_QC reference databases.
    
    This wraps the DFAST_QC reference manager to download required databases
    for taxonomic classification and genome quality assessment.
    """
    console.print("[bold blue]DFAST_QC Reference Data Download[/bold blue]")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    done_file = output_dir / ".done"
    if done_file.exists() and not force:
        console.print(f"[green]Reference data already downloaded in {output_dir}[/green]")
        console.print("[yellow]Use --force to re-download[/yellow]")
        return
    
    # Remove existing done file if forcing
    if force and done_file.exists():
        done_file.unlink()
        console.print("[yellow]Forcing re-download of reference data[/yellow]")
    
    console.print(f"Downloading reference data to: {output_dir}")
    
    try:
        start_time = time.time()
        
        # Run dqc_ref_manager.py download
        console.print("[bold yellow]Running dqc_ref_manager.py download...[/bold yellow]")
        
        with console.status("[bold green]Downloading reference databases..."):
            result = subprocess.run(
                ["dqc_ref_manager.py", "download"],
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout for download
            )
        
        if result.returncode != 0:
            console.print(f"[red]Error: dqc_ref_manager.py failed[/red]")
            console.print(f"[red]stdout: {result.stdout}[/red]")
            console.print(f"[red]stderr: {result.stderr}[/red]")
            raise typer.Exit(1)
        
        # Write success marker
        with open(done_file, 'w') as f:
            f.write(f"Reference data downloaded successfully\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Directory: {output_dir.absolute()}\n")
        
        elapsed_time = time.time() - start_time
        console.print(f"[green]✓ Reference data downloaded successfully![/green]")
        console.print(f"[dim]Download completed in {elapsed_time:.1f} seconds[/dim]")
        console.print(f"[dim]Output directory: {output_dir.absolute()}[/dim]")
        
        # Display some basic info about downloaded data
        if result.stdout:
            console.print(f"[dim]Download log:[/dim]")
            console.print(result.stdout)
            
    except subprocess.TimeoutExpired:
        console.print("[red]Error: Download timed out (>1 hour)[/red]")
        raise typer.Exit(1)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error: dqc_ref_manager.py failed with code {e.returncode}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: Unexpected error during download: {e}[/red]")
        logger.exception("Download error details:")
        raise typer.Exit(1)


def main():
    """Entry point for standalone execution."""
    logging.basicConfig(level=logging.INFO)
    typer.run(download)


if __name__ == "__main__":
    main()
