#!/usr/bin/env python3
"""
CLI interface for genomic question answering.
Designed for both interactive use and containerized deployment.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
import json

from .config import LLMConfig
from .rag_system import GenomicRAG, EXAMPLE_GENOMIC_QUESTIONS

console = Console()
app = typer.Typer(help="Genomic Knowledge Graph Question Answering")


def setup_logging(verbose: bool = False):
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question about genomic data"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    output_format: str = typer.Option("text", "--format", "-f", help="Output format: text, json"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    save_result: Optional[Path] = typer.Option(None, "--save", "-s", help="Save result to file")
):
    """Ask a question about the genomic knowledge graph."""
    setup_logging(verbose)
    
    try:
        # Load configuration
        if config_file and config_file.exists():
            config = LLMConfig.from_file(config_file)
        else:
            config = LLMConfig.from_env()
        
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
        response = asyncio.run(_process_question(question, config))
        
        # Output result
        if output_format == "json":
            output = json.dumps(response, indent=2)
            console.print(output)
        else:
            _display_answer(response)
        
        # Save if requested
        if save_result:
            with open(save_result, 'w') as f:
                json.dump(response, f, indent=2)
            console.print(f"\n💾 Result saved to: {save_result}")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def interactive(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """Start interactive question-answering session."""
    setup_logging(verbose)
    
    try:
        # Load configuration
        if config_file and config_file.exists():
            config = LLMConfig.from_file(config_file)
        else:
            config = LLMConfig.from_env()
        
        # Start interactive session
        asyncio.run(_interactive_session(config))
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Session ended by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def demo(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    num_questions: int = typer.Option(3, "--questions", "-n", help="Number of demo questions to run"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """Run demo with example genomic questions."""
    setup_logging(verbose)
    
    try:
        # Load configuration
        if config_file and config_file.exists():
            config = LLMConfig.from_file(config_file)
        else:
            config = LLMConfig.from_env()
        
        # Run demo
        asyncio.run(_run_demo(config, num_questions))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def health(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """Check system health and configuration."""
    setup_logging(verbose)
    
    try:
        # Load configuration
        if config_file and config_file.exists():
            config = LLMConfig.from_file(config_file)
        else:
            config = LLMConfig.from_env()
        
        # Run health check
        asyncio.run(_health_check(config))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def config(
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Save configuration to file"),
    show_example: bool = typer.Option(False, "--example", help="Show example configuration")
):
    """Show or create configuration files."""
    if show_example:
        config = LLMConfig()
        console.print("[bold]Example Configuration:[/bold]")
        console.print(Syntax(config.json(indent=2), "json"))
        return
    
    # Load current config
    config = LLMConfig.from_env()
    
    if output_file:
        config.to_file(output_file)
        console.print(f"✅ Configuration saved to: {output_file}")
    else:
        console.print("[bold]Current Configuration:[/bold]")
        console.print(Syntax(config.json(indent=2), "json"))


async def _process_question(question: str, config: LLMConfig) -> dict:
    """Process a single question."""
    rag = GenomicRAG(config)
    
    try:
        response = await rag.ask(question)
        return response
    finally:
        rag.close()


async def _interactive_session(config: LLMConfig):
    """Run interactive question-answering session."""
    console.print(Panel.fit(
        "[bold green]🧬 Genomic Knowledge Graph Q&A[/bold green]\n"
        "Ask questions about proteins, genomes, and functional annotations.\n"
        "Type 'help' for examples, 'quit' to exit.",
        title="Interactive Session"
    ))
    
    rag = GenomicRAG(config)
    
    try:
        # Health check
        health = rag.health_check()
        if not all(health.values()):
            console.print("[red]⚠️  Some components are not healthy:[/red]")
            for component, ok in health.items():
                icon = "✅" if ok else "❌"
                console.print(f"  {icon} {component}")
            console.print()
        
        while True:
            question = console.input("\n[bold cyan]🔬 Your question:[/bold cyan] ")
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            elif question.lower() in ['help', 'h']:
                _show_help()
                continue
            elif question.lower() in ['examples', 'ex']:
                _show_examples()
                continue
            elif not question.strip():
                continue
            
            # Process question
            console.print("\n[dim]Processing...[/dim]")
            response = await rag.ask(question)
            _display_answer(response)
    
    finally:
        rag.close()
        console.print("\n[green]👋 Session ended[/green]")


async def _run_demo(config: LLMConfig, num_questions: int):
    """Run demo with example questions."""
    console.print(Panel.fit(
        "[bold green]🧬 Genomic RAG System Demo[/bold green]\n"
        f"Running {num_questions} example questions...",
        title="Demo Mode"
    ))
    
    rag = GenomicRAG(config)
    
    try:
        # Health check
        health = rag.health_check()
        console.print(f"[bold]System Health:[/bold] {health}")
        
        if not all(health.values()):
            console.print("[red]⚠️  Some components are not healthy. Demo may fail.[/red]")
        
        # Run demo questions
        questions = EXAMPLE_GENOMIC_QUESTIONS[:num_questions]
        
        for i, question in enumerate(questions, 1):
            console.print(f"\n[bold cyan]Demo Question {i}/{len(questions)}:[/bold cyan]")
            console.print(f"[dim]{question}[/dim]")
            
            response = await rag.ask(question)
            _display_answer(response, show_metadata=True)
            
            console.print("-" * 60)
    
    finally:
        rag.close()


async def _health_check(config: LLMConfig):
    """Perform comprehensive health check."""
    console.print("[bold]🏥 System Health Check[/bold]")
    console.print("="*50)
    
    # Configuration validation
    console.print("\n[bold]1. Configuration Validation[/bold]")
    status = config.validate_configuration()
    
    table = Table(title="Configuration Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Details", style="dim")
    
    for component, ok in status.items():
        icon = "✅ OK" if ok else "❌ FAIL"
        details = ""
        if component == "lancedb_configured" and not ok:
            details = f"Path: {config.database.lancedb_path}"
        table.add_row(component, icon, details)
    
    console.print(table)
    
    # Component health check
    if all(status.values()):
        console.print("\n[bold]2. Component Health Check[/bold]")
        
        rag = GenomicRAG(config)
        
        try:
            health = rag.health_check()
            
            for component, ok in health.items():
                icon = "✅" if ok else "❌"
                console.print(f"  {icon} {component}")
        
        finally:
            rag.close()
    else:
        console.print("\n[yellow]⚠️  Skipping component health check due to configuration issues[/yellow]")


def _display_answer(response: dict, show_metadata: bool = False):
    """Display formatted answer."""
    console.print(f"\n[bold green]🤖 Answer:[/bold green]")
    console.print(Panel(response['answer'], title="Response"))
    
    # Show confidence and citations
    confidence_color = {
        "high": "green",
        "medium": "yellow", 
        "low": "red"
    }.get(response.get('confidence', 'unknown'), "white")
    
    console.print(f"[bold]Confidence:[/bold] [{confidence_color}]{response.get('confidence', 'unknown')}[/{confidence_color}]")
    
    if response.get('citations'):
        console.print(f"[bold]Sources:[/bold] {response['citations']}")
    
    if show_metadata and 'query_metadata' in response:
        metadata = response['query_metadata']
        console.print(f"[dim]Query type: {metadata.get('query_type', 'unknown')} | "
                     f"Retrieval time: {metadata.get('retrieval_time', 0):.2f}s | "
                     f"Context items: {metadata.get('context_items', 0)}[/dim]")


def _show_help():
    """Show help information."""
    console.print(Panel(
        "[bold]Available Commands:[/bold]\n"
        "• help, h - Show this help\n"
        "• examples, ex - Show example questions\n"
        "• quit, exit, q - Exit session\n\n"
        "[bold]Question Types:[/bold]\n"
        "• Genome information: 'How many proteins in genome X?'\n"
        "• Protein similarity: 'Find proteins similar to Y'\n"
        "• Functional analysis: 'What KEGG functions in genome Z?'\n"
        "• General queries: 'What's in the database?'",
        title="Help"
    ))


def _show_examples():
    """Show example questions."""
    console.print("[bold]📝 Example Questions:[/bold]")
    for i, question in enumerate(EXAMPLE_GENOMIC_QUESTIONS[:7], 1):
        console.print(f"  {i}. {question}")


# Function for external use (e.g., from main CLI)
async def ask_question(question: str, config: Optional[LLMConfig] = None) -> dict:
    """
    Ask a question programmatically.
    
    Args:
        question: Natural language question
        config: Optional configuration (will use environment if not provided)
        
    Returns:
        Response dictionary
    """
    if config is None:
        config = LLMConfig.from_env()
    
    return await _process_question(question, config)


if __name__ == "__main__":
    app()