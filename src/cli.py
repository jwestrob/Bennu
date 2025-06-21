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
build_kg_module = importlib.import_module('src.build_kg.rdf_builder')
esm2_embeddings_module = importlib.import_module('src.ingest.06_esm2_embeddings')

prepare_inputs = prepare_inputs_module.prepare_inputs
run_quast = run_quast_module.run_quast
run_dfast_qc = run_dfast_qc_module.call
run_prodigal = run_prodigal_module.run_prodigal
run_astra_scan = run_astra_scan_module.run_astra_scan
build_knowledge_graph = build_kg_module.build_knowledge_graph_from_pipeline
run_esm2_embeddings = esm2_embeddings_module.run_esm2_embeddings

# Import LLM components
from .llm.cli import ask_question
from .llm.config import LLMConfig

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
        help="Resume pipeline from specific stage (0-6)"
    ),
    to_stage: int = typer.Option(
        6,
        "--to-stage", "-t",
        help="Stop pipeline at specific stage (0-6)"
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
    5. Knowledge graph construction with RDF triples
    6. ESM2 protein embeddings for semantic search
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
                threads=threads,
                databases=["PFAM", "KOFAM"],
                force=force
            )
        },
        5: {
            "name": "Knowledge Graph Construction",
            "function": lambda: build_knowledge_graph(
                stage03_dir=output_dir / "stage03_prodigal",
                stage04_dir=output_dir / "stage04_astra",
                output_dir=output_dir / "stage05_kg"
            )
        },
        6: {
            "name": "ESM2 Protein Embeddings",
            "function": lambda: run_esm2_embeddings(
                stage03_dir=output_dir / "stage03_prodigal",
                output_dir=output_dir / "stage06_esm2",
                batch_size=max(1, threads // 2),  # Adjust batch size based on available threads
                force=force
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
    config_file: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Configuration file path"
    ),
    neo4j_password: Optional[str] = typer.Option(
        None,
        "--neo4j-password",
        help="Neo4j password (overrides config)"
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
    import asyncio
    import json
    
    console.print("[bold blue]🧬 Genomic Question Answering[/bold blue]")
    console.print(f"[dim]Question: {question}[/dim]")
    
    try:
        # Load configuration
        if config_file and config_file.exists():
            config = LLMConfig.from_file(config_file)
        else:
            config = LLMConfig.from_env()
        
        # Override Neo4j password if provided
        if neo4j_password:
            config.database.neo4j_password = neo4j_password
        
        # Validate configuration
        status = config.validate_configuration()
        if not all(status.values()):
            console.print("[red]⚠️  Configuration issues detected:[/red]")
            for component, ok in status.items():
                icon = "✅" if ok else "❌"
                console.print(f"  {icon} {component}")
            
            if not status.get('llm_configured', False):
                console.print("\n[yellow]💡 Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable[/yellow]")
            
            raise typer.Exit(1)
        
        # Process question
        with console.status("[bold green]🤔 Processing question..."):
            result = asyncio.run(ask_question(question, config))
        
        # Display answer
        console.print(f"\n[bold green]🤖 Answer:[/bold green]")
        console.print(result['answer'])
        
        # Show confidence and metadata
        confidence_color = {
            "high": "green",
            "medium": "yellow", 
            "low": "red"
        }.get(result.get('confidence', 'unknown'), "white")
        
        console.print(f"\n[bold]Confidence:[/bold] [{confidence_color}]{result.get('confidence', 'unknown')}[/{confidence_color}]")
        
        if result.get('citations'):
            console.print(f"[bold]Sources:[/bold] {result['citations']}")
        
        if verbose and 'query_metadata' in result:
            metadata = result['query_metadata']
            console.print(f"\n[dim]Query details:[/dim]")
            console.print(f"[dim]  • Type: {metadata.get('query_type', 'unknown')}[/dim]")
            console.print(f"[dim]  • Strategy: {metadata.get('search_strategy', 'unknown')}[/dim]")
            console.print(f"[dim]  • Context items: {metadata.get('context_items', 0)}[/dim]")
            console.print(f"[dim]  • Retrieval time: {metadata.get('retrieval_time', 0):.2f}s[/dim]")
        
        # Save to file if requested
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            console.print(f"\n[dim]💾 Answer saved to {output_file}[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error answering question: {e}[/red]")
        if verbose:
            logger.exception("QA error details:")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    console.print("Genome-to-LLM Knowledge Graph Pipeline v0.1.0")


if __name__ == "__main__":
    app()
