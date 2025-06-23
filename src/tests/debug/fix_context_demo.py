#!/usr/bin/env python3
"""
Quick demo showing what the LLM should be receiving for ribosomal protein data.
"""

import json
from rich.console import Console
from rich.panel import Panel

console = Console()

# Load the actual Neo4j data from our debug output
with open('rag_context_debug.json', 'r') as f:
    debug_data = json.load(f)

neo4j_data = debug_data['neo4j_raw_data']

def format_context_properly(neo4j_data):
    """Show what the LLM context should look like with proper formatting."""
    
    formatted_parts = []
    formatted_parts.append("RIBOSOMAL PROTEIN ANALYSIS:")
    formatted_parts.append(f"Found {len(neo4j_data)} ribosomal proteins across 4 genomes.\n")
    
    # Group by KO ID to show diversity
    ko_groups = {}
    for item in neo4j_data:
        ko_id = item['ko.id']
        if ko_id not in ko_groups:
            ko_groups[ko_id] = []
        ko_groups[ko_id].append(item)
    
    formatted_parts.append(f"PROTEIN FAMILIES IDENTIFIED ({len(ko_groups)} types):")
    for ko_id, proteins in ko_groups.items():
        description = proteins[0]['ko.description']
        formatted_parts.append(f"\n• {ko_id}: {description}")
        formatted_parts.append(f"  Found in {len(proteins)} copies across genomes")
        
        # Show example coordinates
        example = proteins[0]
        start = example['g.startCoordinate']
        end = example['g.endCoordinate']
        protein_id = example['p.id']
        
        # Extract genome info
        if 'Acidovorax' in protein_id:
            genome = 'Acidovorax'
        elif 'Gammaproteobacteria' in protein_id:
            genome = 'Gammaproteobacteria'
        elif 'OD1' in protein_id:
            genome = 'OD1'
        elif 'PLM0' in protein_id:
            genome = 'PLM0_60'
        else:
            genome = 'Unknown'
            
        formatted_parts.append(f"  Example: {genome} genome, position {start}-{end} bp")
    
    formatted_parts.append("\nGENOME DISTRIBUTION:")
    genome_counts = {}
    for item in neo4j_data:
        protein_id = item['p.id']
        if 'Acidovorax' in protein_id:
            genome = 'Acidovorax'
        elif 'Gammaproteobacteria' in protein_id:
            genome = 'Gammaproteobacteria' 
        elif 'OD1' in protein_id:
            genome = 'OD1'
        elif 'PLM0' in protein_id:
            genome = 'PLM0_60'
        else:
            genome = 'Unknown'
        
        genome_counts[genome] = genome_counts.get(genome, 0) + 1
    
    for genome, count in genome_counts.items():
        formatted_parts.append(f"• {genome}: {count} ribosomal proteins")
    
    formatted_parts.append("\nFUNCTIONAL BREAKDOWN:")
    subunit_counts = {'Large subunit (50S)': 0, 'Small subunit (30S)': 0, 'Associated proteins': 0}
    
    for item in neo4j_data:
        desc = item['ko.description'].lower()
        if 'large subunit' in desc:
            subunit_counts['Large subunit (50S)'] += 1
        elif 'small subunit' in desc:
            subunit_counts['Small subunit (30S)'] += 1
        else:
            subunit_counts['Associated proteins'] += 1
    
    for category, count in subunit_counts.items():
        if count > 0:
            formatted_parts.append(f"• {category}: {count} proteins")
    
    return "\n".join(formatted_parts)

# Show what the LLM should be getting
proper_context = format_context_properly(neo4j_data)

console.print(Panel(
    proper_context,
    title="What the LLM SHOULD be receiving",
    border_style="green"
))

console.print(f"\n[green]✅ Proper context length: {len(proper_context)} characters[/green]")
console.print(f"[red]❌ Current context length: {len(debug_data['formatted_context'])} characters[/red]")

console.print(f"\n[yellow]💡 This demonstrates why the LLM is giving generic responses instead of data-specific answers![/yellow]")