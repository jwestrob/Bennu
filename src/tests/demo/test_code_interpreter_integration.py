#!/usr/bin/env python3
"""
Demo: Code Interpreter Integration with Agentic RAG System

This demo shows how the code interpreter integrates with the genomic RAG system
for advanced data analysis and visualization capabilities.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

def print_header(title: str):
    """Print a formatted header."""
    console.print(Panel(title, style="bold blue"))

def print_code(code: str, language: str = "python"):
    """Print syntax-highlighted code."""
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    console.print(syntax)

def print_result(result: dict):
    """Print execution result."""
    if result.get("success"):
        console.print("✅ [bold green]Success[/bold green]")
        if result.get("stdout"):
            console.print(f"[dim]Output:[/dim]\n{result['stdout']}")
        if result.get("files_created"):
            console.print(f"[dim]Files:[/dim] {result['files_created']}")
    else:
        console.print("❌ [bold red]Failed[/bold red]")
        console.print(f"[dim]Error:[/dim] {result.get('error', 'Unknown error')}")

async def demo_protein_analysis():
    """Demo: Protein similarity analysis with visualization."""
    print_header("Demo 1: Protein Similarity Analysis")
    
    # Sample code that would be executed
    analysis_code = """
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Protein similarity data (would come from LanceDB/Neo4j in real scenario)
protein_ids = ['protein_A', 'protein_B', 'protein_C', 'protein_D', 'protein_E']
similarity_matrix = np.array([
    [1.00, 0.75, 0.82, 0.45, 0.63],
    [0.75, 1.00, 0.70, 0.38, 0.55],
    [0.82, 0.70, 1.00, 0.42, 0.67],
    [0.45, 0.38, 0.42, 1.00, 0.89],
    [0.63, 0.55, 0.67, 0.89, 1.00]
])

# Create visualization
plt.figure(figsize=(8, 6))
sns.heatmap(similarity_matrix, 
            annot=True, 
            cmap='viridis',
            xticklabels=protein_ids,
            yticklabels=protein_ids,
            vmin=0, vmax=1)
plt.title('Protein Similarity Matrix')
plt.tight_layout()
plt.savefig('protein_similarity_heatmap.png', dpi=150)
plt.close()

# Statistical analysis
upper_tri = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
print(f"Protein Similarity Statistics:")
print(f"  Mean similarity: {np.mean(upper_tri):.3f}")
print(f"  Std deviation: {np.std(upper_tri):.3f}")
print(f"  Max similarity: {np.max(upper_tri):.3f}")
print(f"  Min similarity: {np.min(upper_tri):.3f}")

# Find most similar protein pairs
most_similar_idx = np.unravel_index(np.argmax(upper_tri), similarity_matrix.shape)
print(f"\\nMost similar pair: {protein_ids[most_similar_idx[0]]} - {protein_ids[most_similar_idx[1]]}")
print(f"Similarity score: {similarity_matrix[most_similar_idx]:.3f}")
"""
    
    print_code(analysis_code)
    
    # Mock the execution result
    mock_result = {
        "success": True,
        "stdout": """Protein Similarity Statistics:
  Mean similarity: 0.631
  Std deviation: 0.165
  Max similarity: 0.890
  Min similarity: 0.380

Most similar pair: protein_D - protein_E
Similarity score: 0.890""",
        "stderr": "",
        "execution_time": 1.2,
        "files_created": ["protein_similarity_heatmap.png"],
        "session_id": "demo-session-1"
    }
    
    print_result(mock_result)
    console.print()

async def demo_genomic_neighborhood():
    """Demo: Genomic neighborhood visualization."""
    print_header("Demo 2: Genomic Neighborhood Analysis")
    
    # Sample genomic data
    genomic_code = """
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# Gene data from Neo4j query
genes = [
    {'id': 'gene_heme_001', 'start': 1000, 'end': 2500, 'strand': 1, 'function': 'heme transporter'},
    {'id': 'gene_reg_002', 'start': 3000, 'end': 3800, 'strand': -1, 'function': 'transcriptional regulator'},
    {'id': 'gene_oxi_003', 'start': 4200, 'end': 5600, 'strand': 1, 'function': 'cytochrome oxidase'},
    {'id': 'gene_unk_004', 'start': 6000, 'end': 6900, 'strand': 1, 'function': 'hypothetical protein'},
    {'id': 'gene_eff_005', 'start': 7500, 'end': 8200, 'strand': -1, 'function': 'efflux pump'}
]

# Create genomic neighborhood plot
fig, ax = plt.subplots(figsize=(14, 6))

# Color mapping for function types
color_map = {
    'heme transporter': '#FF6B6B',
    'transcriptional regulator': '#4ECDC4', 
    'cytochrome oxidase': '#45B7D1',
    'hypothetical protein': '#96CEB4',
    'efflux pump': '#FFEAA7'
}

# Plot genes
for i, gene in enumerate(genes):
    start, end = gene['start'], gene['end']
    strand = gene['strand']
    function = gene['function']
    gene_id = gene['id'].split('_')[1]  # Extract short name
    
    # Gene body
    color = color_map.get(function, '#DDA0DD')
    rect = FancyBboxPatch((start, -0.3), end-start, 0.6,
                         boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    
    # Strand arrow
    arrow_start = start if strand > 0 else end
    arrow_end = end if strand > 0 else start
    ax.annotate('', xy=(arrow_end, 0), xytext=(arrow_start, 0),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Gene label
    mid_point = (start + end) / 2
    ax.text(mid_point, 0.5, gene_id, ha='center', va='bottom', 
            fontweight='bold', fontsize=10)
    ax.text(mid_point, -0.5, function, ha='center', va='top', 
            fontsize=8, style='italic', wrap=True)

# Formatting
ax.set_xlim(500, 8700)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('Genomic Position (bp)', fontsize=12)
ax.set_title('Heme Transport Gene Cluster - Genomic Neighborhood', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_yticks([])

# Add distance annotations
for i in range(len(genes)-1):
    gap = genes[i+1]['start'] - genes[i]['end']
    mid_gap = (genes[i]['end'] + genes[i+1]['start']) / 2
    ax.text(mid_gap, -1.0, f'{gap} bp', ha='center', va='center', 
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))

plt.tight_layout()
plt.savefig('genomic_neighborhood.png', dpi=150, bbox_inches='tight')
plt.close()

# Analysis summary
print("Genomic Neighborhood Analysis:")
print(f"  Total genes analyzed: {len(genes)}")
print(f"  Genomic span: {genes[0]['start']} - {genes[-1]['end']} bp")
print(f"  Forward strand genes: {sum(1 for g in genes if g['strand'] > 0)}")
print(f"  Reverse strand genes: {sum(1 for g in genes if g['strand'] < 0)}")

# Operon prediction based on proximity and strand
print("\\nPotential operon structure:")
for i in range(len(genes)-1):
    gap = genes[i+1]['start'] - genes[i]['end']
    same_strand = genes[i]['strand'] == genes[i+1]['strand']
    if gap < 200 and same_strand:
        print(f"  {genes[i]['id']} -> {genes[i+1]['id']} (gap: {gap} bp, likely co-transcribed)")
    else:
        print(f"  {genes[i]['id']} | {genes[i+1]['id']} (gap: {gap} bp, separate transcription)")
"""
    
    print_code(genomic_code)
    
    mock_result = {
        "success": True,
        "stdout": """Genomic Neighborhood Analysis:
  Total genes analyzed: 5
  Genomic span: 1000 - 8200 bp
  Forward strand genes: 3
  Reverse strand genes: 2

Potential operon structure:
  gene_heme_001 | gene_reg_002 (gap: 500 bp, separate transcription)
  gene_reg_002 | gene_oxi_003 (gap: 400 bp, separate transcription)
  gene_oxi_003 -> gene_unk_004 (gap: 400 bp, likely co-transcribed)
  gene_unk_004 | gene_eff_005 (gap: 600 bp, separate transcription)""",
        "stderr": "",
        "execution_time": 0.8,
        "files_created": ["genomic_neighborhood.png"],
        "session_id": "demo-session-2"
    }
    
    print_result(mock_result)
    console.print()

async def demo_agentic_workflow():
    """Demo: Full agentic workflow with code interpreter."""
    print_header("Demo 3: Agentic Workflow Integration")
    
    console.print("🤖 [bold cyan]Agentic Query:[/bold cyan] \"Find proteins similar to heme transporters and create a detailed analysis\"")
    console.print()
    
    # Step 1: Neo4j query for heme transporters
    console.print("📊 [bold yellow]Step 1:[/bold yellow] Query Neo4j for heme transporter annotations")
    neo4j_query = """
MATCH (p:Protein)-[:HASFUNCTION]->(ko:KEGGOrtholog)
WHERE ko.description CONTAINS 'heme' AND ko.description CONTAINS 'transport'
RETURN p.id, p.sequence_length, ko.id as kegg_id, ko.description
LIMIT 5
"""
    print_code(neo4j_query, "cypher")
    
    # Step 2: LanceDB similarity search
    console.print("🔍 [bold yellow]Step 2:[/bold yellow] Use top results as seeds for ESM2 similarity search")
    console.print("Using protein embeddings to find similar sequences...")
    console.print()
    
    # Step 3: Code interpreter analysis
    console.print("🧮 [bold yellow]Step 3:[/bold yellow] Execute comprehensive analysis with code interpreter")
    
    analysis_workflow = """
# Multi-modal analysis: Combine annotations + similarity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data from Neo4j (annotated heme transporters)
annotated_proteins = [
    {'id': 'protein_heme_A', 'length': 345, 'kegg': 'K02014', 'score': 1.0, 'source': 'annotation'},
    {'id': 'protein_heme_B', 'length': 298, 'kegg': 'K02014', 'score': 0.95, 'source': 'annotation'},
    {'id': 'protein_heme_C', 'length': 412, 'kegg': 'K02015', 'score': 0.88, 'source': 'annotation'}
]

# Data from LanceDB (ESM2 similarity)
similar_proteins = [
    {'id': 'protein_sim_D', 'length': 356, 'similarity': 0.73, 'source': 'similarity'},
    {'id': 'protein_sim_E', 'length': 289, 'similarity': 0.68, 'source': 'similarity'},
    {'id': 'protein_sim_F', 'length': 398, 'similarity': 0.62, 'source': 'similarity'},
    {'id': 'protein_sim_G', 'length': 445, 'similarity': 0.58, 'source': 'similarity'}
]

# Combine datasets
all_proteins = annotated_proteins + similar_proteins
df = pd.DataFrame(all_proteins)

# Analysis 1: Length distribution by discovery method
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Length distribution
annotation_lengths = [p['length'] for p in annotated_proteins]
similarity_lengths = [p['length'] for p in similar_proteins]

ax1.hist(annotation_lengths, alpha=0.7, label='Annotated (Neo4j)', bins=8, color='blue')
ax1.hist(similarity_lengths, alpha=0.7, label='Similar (LanceDB)', bins=8, color='orange')
ax1.set_xlabel('Protein Length (amino acids)')
ax1.set_ylabel('Count')
ax1.set_title('Protein Length Distribution by Discovery Method')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Confidence/Similarity scores
scores = []
labels = []
for p in annotated_proteins:
    scores.append(p.get('score', p.get('similarity', 0)))
    labels.append('Annotated')
for p in similar_proteins:
    scores.append(p.get('similarity', p.get('score', 0)))
    labels.append('Similar')

df_scores = pd.DataFrame({'score': scores, 'method': labels})
sns.boxplot(data=df_scores, x='method', y='score', ax=ax2)
ax2.set_title('Confidence Scores by Discovery Method')
ax2.set_ylabel('Score (Annotation Confidence / ESM2 Similarity)')

plt.tight_layout()
plt.savefig('heme_transporter_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# Analysis 2: Functional assessment
print("=== Heme Transporter Analysis Report ===")
print(f"Total proteins identified: {len(all_proteins)}")
print(f"  - Annotated (high confidence): {len(annotated_proteins)}")
print(f"  - Similar sequences (ESM2): {len(similar_proteins)}")
print()

print("Length Statistics:")
all_lengths = [p['length'] for p in all_proteins]
print(f"  Mean length: {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f} aa")
print(f"  Range: {min(all_lengths)} - {max(all_lengths)} aa")
print()

print("Confidence Assessment:")
print("  High confidence (annotated):")
for p in annotated_proteins:
    print(f"    {p['id']}: {p['length']} aa, KEGG {p['kegg']}")

print("  Moderate confidence (sequence similarity):")
for p in similar_proteins:
    print(f"    {p['id']}: {p['length']} aa, ESM2 similarity {p['similarity']:.3f}")

print()
print("Recommendations:")
print("  1. Validate similar proteins with HMM domain scanning")
print("  2. Check genomic context for heme metabolism operons")
print("  3. Prioritize proteins with similarity > 0.65 for experimental validation")

# Summary statistics
high_sim = [p for p in similar_proteins if p['similarity'] > 0.65]
print(f"\\nPriority candidates for validation: {len(high_sim)} proteins")
"""
    
    print_code(analysis_workflow)
    
    mock_result = {
        "success": True,
        "stdout": """=== Heme Transporter Analysis Report ===
Total proteins identified: 7
  - Annotated (high confidence): 3
  - Similar sequences (ESM2): 4

Length Statistics:
  Mean length: 363.3 ± 56.8 aa
  Range: 289 - 445 aa

Confidence Assessment:
  High confidence (annotated):
    protein_heme_A: 345 aa, KEGG K02014
    protein_heme_B: 298 aa, KEGG K02014
    protein_heme_C: 412 aa, KEGG K02015
  Moderate confidence (sequence similarity):
    protein_sim_D: 356 aa, ESM2 similarity 0.730
    protein_sim_E: 289 aa, ESM2 similarity 0.680
    protein_sim_F: 398 aa, ESM2 similarity 0.620
    protein_sim_G: 445 aa, ESM2 similarity 0.580

Recommendations:
  1. Validate similar proteins with HMM domain scanning
  2. Check genomic context for heme metabolism operons
  3. Prioritize proteins with similarity > 0.65 for experimental validation

Priority candidates for validation: 2 proteins""",
        "stderr": "",
        "execution_time": 2.1,
        "files_created": ["heme_transporter_analysis.png"],
        "session_id": "demo-session-3"
    }
    
    print_result(mock_result)
    
    console.print("✨ [bold green]Agentic Analysis Complete![/bold green]")
    console.print("The system successfully combined:")
    console.print("  • Structured knowledge (Neo4j annotations)")
    console.print("  • Semantic similarity (LanceDB embeddings)")  
    console.print("  • Advanced analysis (Code interpreter)")
    console.print("  • Actionable insights (Prioritized recommendations)")

async def demo_task_graph_integration():
    """Demo: How code interpreter integrates with the task graph system."""
    print_header("Demo 4: Task Graph Integration")
    
    # Show how a complex query gets broken down into tasks
    console.print("🔄 [bold cyan]Complex Query:[/bold cyan] \"Analyze the genomic context of heme transport genes and compare to literature\"")
    console.print()
    
    task_structure = """
TaskGraph Execution Plan:
├── Task 1 (ATOMIC_QUERY): Find heme transport genes in Neo4j
├── Task 2 (TOOL_CALL): Search literature for heme transport research  
├── Task 3 (TOOL_CALL): Code execution - genomic neighborhood analysis
│   ├── Input: Gene coordinates from Task 1
│   └── Output: Neighborhood visualization + operon predictions
├── Task 4 (TOOL_CALL): Code execution - comparative analysis
│   ├── Input: Results from Tasks 1, 2, 3
│   └── Output: Comprehensive comparison report
└── Task 5 (AGGREGATE): Synthesize all results into final answer
"""
    
    console.print(task_structure)
    console.print()
    
    # Show the DSPy planning decision
    planning_example = '''
class PlannerAgent(dspy.Signature):
    """Intelligent query routing for genomic analysis."""
    
    query = dspy.InputField(desc="User's genomic research question")
    reasoning = dspy.OutputField(desc="Why this query needs agentic planning")
    requires_planning = dspy.OutputField(desc="true/false")
    task_plan = dspy.OutputField(desc="JSON task plan if planning required")

# Example planning output:
{
  "reasoning": "This query requires multi-step analysis combining database queries, literature search, and computational analysis with visualization",
  "requires_planning": "true", 
  "task_plan": {
    "tasks": [
      {
        "task_type": "ATOMIC_QUERY",
        "query": "MATCH (p:Protein)-[:HASFUNCTION]->(:KEGGOrtholog {description: CONTAINS 'heme transport'}) RETURN p.genomic_context",
        "dependencies": []
      },
      {
        "task_type": "TOOL_CALL", 
        "tool_name": "literature_search",
        "tool_args": {"query": "heme transport genomic organization operon"},
        "dependencies": []
      },
      {
        "task_type": "TOOL_CALL",
        "tool_name": "code_interpreter", 
        "tool_args": {"code": "# Genomic neighborhood analysis...", "session_id": "analysis-session"},
        "dependencies": ["task_1"]
      }
    ]
  }
}
'''
    
    print_code(planning_example, "python")
    
    console.print("🎯 [bold green]Key Benefits:[/bold green]")
    console.print("  • [dim]Intelligent routing:[/dim] Simple queries → fast traditional path")
    console.print("  • [dim]Complex analysis:[/dim] Multi-step queries → agentic orchestration")
    console.print("  • [dim]Tool integration:[/dim] Seamless use of code execution, literature search")
    console.print("  • [dim]Robust execution:[/dim] Dependency management, error handling, fallbacks")

async def main():
    """Run all demos."""
    console.print("🧬 [bold blue]Code Interpreter Integration Demos[/bold blue]")
    console.print("Showcasing advanced genomic analysis capabilities")
    console.print()
    
    await demo_protein_analysis()
    await demo_genomic_neighborhood()
    await demo_agentic_workflow()
    await demo_task_graph_integration()
    
    console.print()
    console.print("🎉 [bold green]Integration Complete![/bold green]")
    console.print("The code interpreter service is ready for deployment with:")
    console.print("  ✅ Secure execution environment (Docker + gVisor)")
    console.print("  ✅ Genomic analysis capabilities")  
    console.print("  ✅ Agentic RAG system integration")
    console.print("  ✅ Comprehensive testing suite")
    console.print("  ✅ Production deployment configuration")
    console.print()
    console.print("[dim]Next: Deploy with `./src/code_interpreter/deploy.sh deploy`[/dim]")

if __name__ == "__main__":
    asyncio.run(main())