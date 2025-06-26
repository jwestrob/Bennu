#!/usr/bin/env python3
"""
TaskRepairAgent Demo: Showcase error detection and repair capabilities.

This demo shows how the TaskRepairAgent transforms crashes into helpful user messages.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.llm.task_repair_agent import TaskRepairAgent
from src.llm.repair_types import RepairStrategy
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def demo_comment_query_repair():
    """Demo: DSPy comment query repair"""
    console.print("\n[bold blue]🔧 Demo 1: Comment Query Repair[/bold blue]")
    
    agent = TaskRepairAgent()
    
    # This is the actual problematic query we discovered
    comment_query = "/* No valid query can be constructed: label `FakeNode` is not part of the graph schema */"
    error = Exception("Neo.ClientError.Statement.SyntaxError: Invalid input...")
    
    console.print(Panel(
        f"[red]Original Query:[/red]\n{comment_query}\n\n[red]Error:[/red]\n{str(error)[:100]}...",
        title="❌ Before TaskRepairAgent"
    ))
    
    # Apply repair
    result = agent.detect_and_repair(comment_query, error)
    
    if result.success:
        console.print(Panel(
            f"[green]Repair Strategy:[/green] {result.repair_strategy_used}\n\n"
            f"[green]User Message:[/green]\n{result.user_message}\n\n"
            f"[green]Suggestions:[/green] {', '.join(result.suggested_alternatives)}\n\n"
            f"[green]Confidence:[/green] {result.confidence}",
            title="✅ After TaskRepairAgent"
        ))
    else:
        console.print("[red]Repair failed![/red]")


def demo_relationship_repair():
    """Demo: Invalid relationship repair"""
    console.print("\n[bold blue]🔧 Demo 2: Relationship Repair[/bold blue]")
    
    agent = TaskRepairAgent()
    
    # Query with invalid relationship
    invalid_query = "MATCH (p:Protein)-[:NONEXISTENT_RELATIONSHIP]->(d:Domain) RETURN p.id LIMIT 5"
    error = Exception("Invalid relationship type")
    
    console.print(Panel(
        f"[red]Original Query:[/red]\n{invalid_query}\n\n[red]Error:[/red]\n{str(error)}",
        title="❌ Before TaskRepairAgent"
    ))
    
    # Apply repair
    result = agent.detect_and_repair(invalid_query, error)
    
    if result.success:
        console.print(Panel(
            f"[green]Repair Strategy:[/green] {result.repair_strategy_used}\n\n"
            f"[green]Repaired Query:[/green]\n{result.repaired_query}\n\n"
            f"[green]User Message:[/green]\n{result.user_message}\n\n"
            f"[green]Confidence:[/green] {result.confidence}",
            title="✅ After TaskRepairAgent"
        ))
    else:
        console.print("[red]Repair failed![/red]")


def demo_entity_suggestions():
    """Demo: Entity suggestion system"""
    console.print("\n[bold blue]🔧 Demo 3: Entity Suggestions[/bold blue]")
    
    from src.llm.error_patterns import EntitySuggester
    
    test_cases = [
        ("FakeNode", ["Protein", "Gene", "Domain", "KEGGOrtholog"]),
        ("Protien", ["Protein", "Gene", "Domain"]),  # Typo
        ("Genom", ["Gene", "Protein", "Domain"]),    # Partial match
    ]
    
    table = Table(title="Entity Suggestion Examples")
    table.add_column("Invalid Entity", style="red")
    table.add_column("Suggestions", style="green")
    table.add_column("Strategy", style="blue")
    
    for invalid_entity, valid_entities in test_cases:
        suggestions = EntitySuggester.suggest_alternatives(invalid_entity, valid_entities, max_suggestions=3)
        strategy = "Fuzzy Match" if suggestions != valid_entities[:3] else "Default List"
        table.add_row(invalid_entity, ", ".join(suggestions), strategy)
    
    console.print(table)


def demo_error_pattern_detection():
    """Demo: Error pattern detection"""
    console.print("\n[bold blue]🔧 Demo 4: Error Pattern Detection[/bold blue]")
    
    from src.llm.error_patterns import ErrorPatternRegistry
    
    registry = ErrorPatternRegistry()
    
    test_queries = [
        ("/* No valid query can be constructed: label `FakeNode` is not part of the graph schema */", "Comment Query"),
        ("MATCH (p:Protein)-[:NONEXISTENT_RELATIONSHIP]->(d:Domain)", "Invalid Relationship"),
        ("MATCH (fake:FakeNode) RETURN fake", "Invalid Node Label"),
        ("SELECT * FROM proteins", "General Syntax Error")
    ]
    
    table = Table(title="Error Pattern Detection")
    table.add_column("Query/Error", style="red", max_width=40)
    table.add_column("Detected Patterns", style="green")
    table.add_column("Repair Strategy", style="blue")
    
    for query, description in test_queries:
        patterns = registry.find_matching_patterns(query, "")
        if patterns:
            pattern_types = [p.pattern_type for p in patterns]
            strategies = [str(p.repair_strategy.value) for p in patterns]
            table.add_row(
                f"{description}\n{query[:30]}...",
                ", ".join(pattern_types),
                ", ".join(strategies)
            )
        else:
            table.add_row(f"{description}\n{query[:30]}...", "None", "Fallback")
    
    console.print(table)


def demo_schema_summary():
    """Demo: Schema information"""
    console.print("\n[bold blue]🔧 Demo 5: Schema Summary[/bold blue]")
    
    agent = TaskRepairAgent()
    summary = agent.get_schema_summary()
    
    console.print(Panel(
        f"[green]Node Labels:[/green] {', '.join(summary['node_labels'])}\n\n"
        f"[green]Relationship Types:[/green] {', '.join(summary['relationship_types'])}\n\n"
        f"[green]Error Patterns:[/green] {summary['total_patterns']}\n\n"
        f"[green]Repair Strategies:[/green] {len(summary['available_strategies'])}",
        title="📊 Genomic Database Schema"
    ))


def main():
    """Run all TaskRepairAgent demos"""
    console.print(Panel(
        "[bold green]TaskRepairAgent Demo[/bold green]\n\n"
        "This demo showcases how the TaskRepairAgent transforms crashes into helpful user messages.\n"
        "The agent detects error patterns and provides intelligent repair suggestions.",
        title="🧬 Genomic AI Platform - TaskRepairAgent"
    ))
    
    try:
        demo_comment_query_repair()
        demo_relationship_repair()
        demo_entity_suggestions()
        demo_error_pattern_detection()
        demo_schema_summary()
        
        console.print("\n[bold green]✅ All demos completed successfully![/bold green]")
        console.print("\n[dim]The TaskRepairAgent is now integrated into the genomic RAG system.[/dim]")
        
    except Exception as e:
        console.print(f"\n[red]❌ Demo failed: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


if __name__ == "__main__":
    main()