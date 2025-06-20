#!/usr/bin/env python3
"""
Genome-to-LLM Knowledge Graph CLI
Main command-line interface for the genomic processing pipeline.
"""

import logging
from pathlib import Path
from typing import List, Optional
import sys

import typer
from rich.console import Console
from rich.logging import RichHandler

# Import pipeline stages
import importlib
prepare_inputs_module = importlib.import_module('src.ingest.00_prepare_inputs')
run_quast_module = importlib.import_module('src.ingest.01_run_quast')
run_dfast_qc_module = importlib.import_module('src.ingest.02_dfast_qc')
run_prodigal_module = importlib.import_module('src.ingest.03_prodigal')
run_astra_scan_module = importlib.import_module('src.ingest.04_astra_scan')

prepare_inputs = prepare_inputs_module.prepare_inputs
run_quast = run_quast_module.run_quast
run_dfast_qc = run_dfast_qc_module.call
run_prodigal = run_prodigal_module.run_prodigal
run_astra_scan = run_astra_scan_module.run_astra_scan

# Import LLM components
from .llm.qa_chain import create_qa_chain

app = typer.Typer(
    name="genome-kg",
    help="Genome-to-LLM Knowledge Graph Pipeline",
    add_completion=False
)
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)]
)
logger = logging.getLogger("genome-kg")


@app.command()
def build(
    input_dir: Path = typer.Option(
        Path("data/raw"),
        "--input-dir", "-i",
        help="Input directory containing genome assemblies"
    ),
    output_dir: Path = typer.Option(
        Path("data"),
        "--output-dir", "-o", 
        help="Output directory for pipeline results"
    ),
    from_stage: int = typer.Option(
        0,
        "--from-stage", "-f",
        help="Resume pipeline from specific stage (0-4)"
    ),
    to_stage: int = typer.Option(
        4,
        "--to-stage", "-t",
        help="Stop pipeline at specific stage (0-4)"
    ),
    threads: int = typer.Option(
        4,
        "--threads", "-j",
        help="Number of threads to use"
    ),
    skip_tax: bool = typer.Option(
        False,
        "--skip-tax",
        help="Skip taxonomic classification with DFAST_QC"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing outputs"
    )
) -> None:
    """
    Build knowledge graph from genome assemblies.
    
    Runs the complete genomic processing pipeline through multiple stages:
    0. Input preparation and validation
    1. Quality assessment with QUAST
    2. Taxonomic classification with DFAST_QC (ANI+CheckM)  
    3. Gene prediction with Prodigal
    4. Functional annotation with Astra/PyHMMer
    """
    console.print("[bold blue]Genome-to-LLM Knowledge Graph Pipeline[/bold blue]")
    console.print(f"Input directory: {input_dir}")
    console.print(f"Output directory: {output_dir}")
    console.print(f"Running stages {from_stage} to {to_stage}")
    console.print(f"Threads: {threads}")
    
    # Validate input directory
    if not input_dir.exists():
        console.print(f"[red]Error: Input directory does not exist: {input_dir}[/red]")
        raise typer.Exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pipeline stages configuration
    stages = {
        0: {
            "name": "Input Preparation",
            "function": lambda: prepare_inputs(
                input_dir=input_dir,
                output_dir=output_dir / "stage00_prepared"
            )
        },
        1: {
            "name": "QUAST Quality Assessment", 
            "function": lambda: run_quast(
                input_dir=output_dir / "stage00_prepared",
                output_dir=output_dir / "stage01_quast",
                max_workers=min(threads, 4),  # Limit parallel workers
                threads_per_genome=1,
                force=force
            )
        },
        2: {
            "name": "DFAST_QC Taxonomy",
            "function": lambda: run_dfast_qc(
                input_dir=output_dir / "stage00_prepared",
                output_dir=output_dir / "stage02_dfast_qc",
                threads=threads,
                enable_cc=False,
                force=force
            )
        },
        3: {
            "name": "Prodigal Gene Prediction",
            "function": lambda: run_prodigal(
                input_dir=output_dir / "stage00_prepared",
                output_dir=output_dir / "stage03_prodigal",
                max_workers=threads,
                force=force
            )
        },
        4: {
            "name": "Astra Functional Annotation",
            "function": lambda: run_astra_scan(
                input_dir=output_dir / "stage03_prodigal",
                output_dir=output_dir / "stage04_astra",
                threads=threads
            )
        }
    }
    
    # Execute pipeline stages
    try:
        for stage_num in range(from_stage, to_stage + 1):
            if stage_num not in stages:
                console.print(f"[yellow]Warning: Unknown stage {stage_num}[/yellow]")
                continue
            
            stage = stages[stage_num]
            console.print(f"\n[bold green]Stage {stage_num}: {stage['name']}[/bold green]")
            
            # TODO: Add stage output checking and skip logic
            stage_output_dir = output_dir / f"stage{stage_num:02d}_{stage['name'].lower().replace(' ', '_').replace('/', '_')}"
            if not force and stage_output_dir.exists():
                console.print(f"[yellow]Stage {stage_num} output exists, skipping (use --force to overwrite)[/yellow]")
                continue
            
            # Execute stage
            try:
                # Handle skip logic for taxonomy stage
                if stage_num == 2 and skip_tax:
                    console.print(f"[yellow]Skipping Stage {stage_num} (--skip-tax flag)[/yellow]")
                    continue
                    
                stage["function"]()
                console.print(f"[green]✓ Stage {stage_num} completed[/green]")
            except Exception as e:
                console.print(f"[red]✗ Stage {stage_num} failed: {e}[/red]")
                if not typer.confirm("Continue with remaining stages?"):
                    raise typer.Exit(1)
        
        console.print("\n[bold green]Pipeline completed successfully![/bold green]")
        
        # TODO: Generate summary report
        # TODO: Build knowledge graph from results
        # TODO: Create FAISS indices
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Pipeline failed: {e}[/red]")
        logger.exception("Pipeline error details:")
        raise typer.Exit(1)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask about genomic data"),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for answer (JSON format)"
    ),
    faiss_index_dir: Path = typer.Option(
        Path("data/kg/faiss"),
        "--faiss-index",
        help="Directory containing FAISS indices"
    ),
    neo4j_uri: str = typer.Option(
        "bolt://localhost:7687",
        "--neo4j-uri",
        help="Neo4j connection URI"
    ),
    neo4j_username: str = typer.Option(
        "neo4j",
        "--neo4j-user",
        help="Neo4j username"
    ),
    neo4j_password: str = typer.Option(
        "password",
        "--neo4j-password",
        help="Neo4j password"
    ),
    batch_file: Optional[Path] = typer.Option(
        None,
        "--batch", "-b",
        help="File containing multiple questions (one per line)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed reasoning and sources"
    )
) -> None:
    """
    Ask natural language questions about genomic data.
    
    Uses the knowledge graph and LLM components to answer questions about
    genome assemblies, genes, proteins, taxonomy, and functional annotations.
    """
    console.print("[bold blue]Genomic Question Answering[/bold blue]")
    
    # Handle batch processing
    if batch_file:
        if not batch_file.exists():
            console.print(f"[red]Error: Batch file does not exist: {batch_file}[/red]")
            raise typer.Exit(1)
        
        questions = []
        with open(batch_file) as f:
            questions = [line.strip() for line in f if line.strip()]
        
        console.print(f"Processing {len(questions)} questions from {batch_file}")
        
        # TODO: Implement batch processing
        console.print("[yellow]Batch processing not yet implemented[/yellow]")
        return
    
    # Single question processing
    console.print(f"Question: {question}")
    
    try:
        # TODO: Initialize QA chain with proper configuration
        qa_chain = create_qa_chain(
            faiss_index_dir=faiss_index_dir,
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password
        )
        
        # Generate answer
        with console.status("[bold green]Thinking..."):
            result = qa_chain.answer_question(question)
        
        # Display answer
        console.print(f"\n[bold green]Answer:[/bold green] {result['answer']}")
        console.print(f"[dim]Confidence: {result['confidence']:.2f}[/dim]")
        
        if verbose and result.get('reasoning'):
            console.print(f"\n[bold yellow]Reasoning:[/bold yellow] {result['reasoning']}")
        
        if verbose and result.get('sources'):
            console.print(f"\n[bold yellow]Sources:[/bold yellow]")
            for i, source in enumerate(result['sources'][:3], 1):
                console.print(f"  {i}. {source}")
        
        # Save to file if requested
        if output_file:
            import json
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            console.print(f"\n[dim]Answer saved to {output_file}[/dim]")
        
        qa_chain.close()
        
    except Exception as e:
        console.print(f"[red]Error answering question: {e}[/red]")
        logger.exception("QA error details:")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    console.print("Genome-to-LLM Knowledge Graph Pipeline v0.1.0")


if __name__ == "__main__":
    app()
