#!/usr/bin/env python3
"""
Code Interpreter Client

Client interface for communicating with the secure code interpreter service.
Used by the RAG system to execute Python code for data analysis and visualization.
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, Optional, List
import httpx
from pathlib import Path

logger = logging.getLogger(__name__)


class CodeInterpreterClient:
    """Client for the secure code interpreter service."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the code interpreter client.
        
        Args:
            base_url: Base URL of the code interpreter service
        """
        self.base_url = base_url.rstrip('/')
        self.session_id = str(uuid.uuid4())
        
    async def execute_code(
        self, 
        code: str, 
        timeout: int = 30,
        enable_networking: bool = False
    ) -> Dict[str, Any]:
        """
        Execute Python code in the secure environment.
        
        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds
            enable_networking: Enable network access (security risk)
            
        Returns:
            Dict containing execution results
        """
        request_data = {
            "session_id": self.session_id,
            "code": code,
            "timeout": timeout,
            "enable_networking": enable_networking
        }
        
        try:
            async with httpx.AsyncClient(timeout=timeout + 10) as client:
                response = await client.post(
                    f"{self.base_url}/execute",
                    json=request_data
                )
                response.raise_for_status()
                return response.json()
                
        except httpx.TimeoutException:
            return {
                "success": False,
                "error": f"Request timed out after {timeout + 10} seconds",
                "stdout": "",
                "stderr": "",
                "execution_time": timeout + 10
            }
        except httpx.RequestError as e:
            return {
                "success": False,
                "error": f"Failed to connect to code interpreter service: {str(e)}",
                "stdout": "",
                "stderr": "",
                "execution_time": 0
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "stdout": "",
                "stderr": "",
                "execution_time": 0
            }
    
    async def reset_session(self) -> bool:
        """
        Reset the current session.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/sessions/{self.session_id}/reset"
                )
                response.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Failed to reset session: {e}")
            return False
    
    async def health_check(self) -> bool:
        """
        Check if the code interpreter service is healthy.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/health")
                response.raise_for_status()
                return response.json().get("status") == "healthy"
        except Exception:
            return False
    
    def new_session(self) -> str:
        """
        Start a new session.
        
        Returns:
            New session ID
        """
        self.session_id = str(uuid.uuid4())
        return self.session_id


class GenomicCodeInterpreter:
    """High-level interface for genomic data analysis with code execution."""
    
    def __init__(self, client: CodeInterpreterClient):
        """
        Initialize with a code interpreter client.
        
        Args:
            client: CodeInterpreterClient instance
        """
        self.client = client
        self.genomic_data_available = False
    
    async def setup_genomic_environment(self, data_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        Set up the Python environment with genomic data access.
        
        Args:
            data_paths: Dictionary mapping data types to file paths
            
        Returns:
            Execution result
        """
        setup_code = f"""
# Set up genomic data analysis environment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Configure matplotlib for non-interactive use
plt.ioff()
plt.style.use('default')

# Data paths
data_paths = {data_paths}

print("Genomic analysis environment ready!")
print(f"Available data: {{list(data_paths.keys())}}")
"""
        
        result = await self.client.execute_code(setup_code)
        if result.get("success"):
            self.genomic_data_available = True
        
        return result
    
    async def analyze_protein_similarities(self, protein_ids: List[str]) -> Dict[str, Any]:
        """
        Analyze protein similarity data.
        
        Args:
            protein_ids: List of protein IDs to analyze
            
        Returns:
            Analysis results with visualization
        """
        if not self.genomic_data_available:
            return {
                "success": False,
                "error": "Genomic environment not set up. Call setup_genomic_environment first."
            }
        
        analysis_code = f"""
# Protein similarity analysis
protein_ids = {protein_ids}

print(f"Analyzing {{len(protein_ids)}} proteins:")
for i, protein_id in enumerate(protein_ids[:5], 1):
    print(f"  {{i}}. {{protein_id}}")

# Create example similarity matrix (would be real data in production)
import numpy as np
n_proteins = len(protein_ids)
similarity_matrix = np.random.rand(n_proteins, n_proteins)
np.fill_diagonal(similarity_matrix, 1.0)

# Make symmetric
similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2

# Create visualization
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, 
            annot=True if n_proteins <= 10 else False,
            cmap='viridis',
            xticklabels=[p.split('_')[-1] for p in protein_ids],
            yticklabels=[p.split('_')[-1] for p in protein_ids])
plt.title('Protein Similarity Matrix')
plt.tight_layout()
plt.savefig('protein_similarity_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# Summary statistics
print(f"\\nSimilarity Statistics:")
print(f"  Mean similarity: {{np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]):.3f}}")
print(f"  Max similarity: {{np.max(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]):.3f}}")
print(f"  Min similarity: {{np.min(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]):.3f}}")

print("\\nVisualization saved as: protein_similarity_heatmap.png")
"""
        
        return await self.client.execute_code(analysis_code)
    
    async def plot_genomic_neighborhood(self, gene_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a genomic neighborhood visualization.
        
        Args:
            gene_data: List of gene dictionaries with position and annotation data
            
        Returns:
            Plotting results
        """
        plotting_code = f"""
# Genomic neighborhood visualization
gene_data = {gene_data}

# Extract positions and annotations
positions = []
names = []
strands = []
functions = []

for gene in gene_data:
    start = gene.get('start', 0)
    end = gene.get('end', 0)
    positions.append((start + end) / 2)  # Midpoint
    names.append(gene.get('id', 'Unknown').split('_')[-1])
    strands.append(gene.get('strand', 1))
    functions.append(gene.get('function', 'Unknown')[:30])

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot genes as arrows
for i, (pos, name, strand, func) in enumerate(zip(positions, names, strands, functions)):
    color = 'blue' if strand > 0 else 'red'
    direction = '→' if strand > 0 else '←'
    
    # Plot gene as rectangle
    ax.barh(0, 1000, left=pos-500, height=0.3, color=color, alpha=0.7)
    
    # Add gene name
    ax.text(pos, 0.2, f"{{direction}}{{name}}", ha='center', va='bottom', rotation=45, fontsize=8)
    
    # Add function annotation
    ax.text(pos, -0.2, func, ha='center', va='top', rotation=45, fontsize=6, style='italic')

ax.set_ylim(-1, 1)
ax.set_xlabel('Genomic Position (bp)')
ax.set_title('Genomic Neighborhood Analysis')
ax.grid(True, alpha=0.3)

# Remove y-axis as it's not meaningful
ax.set_yticks([])

plt.tight_layout()
plt.savefig('genomic_neighborhood.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Genomic neighborhood plot created with {{len(gene_data)}} genes")
print("Visualization saved as: genomic_neighborhood.png")

# Summary
print(f"\\nNeighborhood Summary:")
print(f"  Total genes: {{len(gene_data)}}")
print(f"  Forward strand: {{sum(1 for s in strands if s > 0)}}")
print(f"  Reverse strand: {{sum(1 for s in strands if s < 0)}}")
if positions:
    print(f"  Genomic span: {{min(positions):.0f}} - {{max(positions):.0f}} bp")
"""
        
        return await self.client.execute_code(plotting_code)
    
    async def fetch_protein_sequences(self, protein_ids: List[str]) -> Dict[str, str]:
        """
        Fetch protein sequences from the sequence database.
        
        Args:
            protein_ids: List of protein identifiers
            
        Returns:
            Dict mapping protein_id to sequence
        """
        fetch_code = f"""
# Fetch protein sequences from sequence database
import sys
sys.path.append('/app')

try:
    from sequence_service import fetch_sequences
    import asyncio
    
    protein_ids = {protein_ids}
    
    # Fetch sequences
    sequences = asyncio.run(fetch_sequences(protein_ids))
    
    print(f"Successfully fetched {{len(sequences)}} sequences out of {{len(protein_ids)}} requested")
    
    # Store in global variable for use in subsequent code
    globals()['fetched_sequences'] = sequences
    
    # Show sample
    for i, (protein_id, sequence) in enumerate(list(sequences.items())[:3]):
        print(f"  {{protein_id}}: {{sequence[:50]}}...{{sequence[-10:]}}")
        if i == 2 and len(sequences) > 3:
            print(f"  ... and {{len(sequences) - 3}} more")
            
except Exception as e:
    print(f"Error fetching sequences: {{e}}")
    globals()['fetched_sequences'] = {{}}
"""
        
        return await self.client.execute_code(fetch_code)
    
    async def analyze_amino_acid_composition_with_sequences(self, protein_groups: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Analyze amino acid composition for protein groups with automatic sequence fetching.
        
        Args:
            protein_groups: Dict mapping group name to list of protein IDs
            
        Returns:
            Analysis results with visualization
        """
        analysis_code = f"""
# Amino acid composition analysis with automatic sequence fetching
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
sys.path.append('/app')

try:
    from sequence_service import calculate_aa_composition
    import asyncio
    
    protein_groups = {protein_groups}
    
    print("=== Amino Acid Composition Analysis ===")
    
    # Fetch sequences and calculate compositions for each group
    all_compositions = {{}}
    group_data = {{}}
    
    for group_name, protein_ids in protein_groups.items():
        print(f"\\nProcessing group: {{group_name}} ({{len(protein_ids)}} proteins)")
        
        # Calculate amino acid compositions
        compositions = asyncio.run(calculate_aa_composition(protein_ids))
        all_compositions[group_name] = compositions
        
        if compositions:
            # Convert to DataFrame for analysis
            df = pd.DataFrame.from_dict(compositions, orient='index')
            group_data[group_name] = df
            
            print(f"  Mean composition calculated for {{len(compositions)}} sequences")
        else:
            print(f"  No sequences found for group {{group_name}}")
    
    if len(group_data) < 2:
        print("Error: Need at least 2 groups with sequences for comparison")
    else:
        # Statistical comparison
        print("\\n=== Statistical Comparison ===")
        
        group_names = list(group_data.keys())
        group1_name, group2_name = group_names[0], group_names[1]
        group1_data = group_data[group1_name]
        group2_data = group_data[group2_name]
        
        # Amino acids to analyze
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                      'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        
        significant_differences = []
        
        for aa in amino_acids:
            if aa in group1_data.columns and aa in group2_data.columns:
                # Mann-Whitney U test (non-parametric)
                stat, p_value = stats.mannwhitneyu(
                    group1_data[aa].dropna(), 
                    group2_data[aa].dropna(),
                    alternative='two-sided'
                )
                
                mean1 = group1_data[aa].mean()
                mean2 = group2_data[aa].mean()
                
                if p_value < 0.05:
                    significant_differences.append((aa, p_value, mean1, mean2))
                    print(f"  {{aa}}: p={{p_value:.3f}} ({{group1_name}}: {{mean1:.3f}}, {{group2_name}}: {{mean2:.3f}})")
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Mean composition comparison
        mean_compositions = pd.DataFrame({{
            group1_name: group1_data.mean(),
            group2_name: group2_data.mean()
        }})
        
        mean_compositions.plot(kind='bar', ax=ax1, alpha=0.7)
        ax1.set_title('Mean Amino Acid Composition Comparison')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Significant differences heatmap
        if significant_differences:
            sig_data = pd.DataFrame(significant_differences, 
                                  columns=['AA', 'p_value', f'{{group1_name}}_mean', f'{{group2_name}}_mean'])
            sig_data = sig_data.set_index('AA')
            
            # Create heatmap of mean differences
            diff_data = sig_data[[f'{{group1_name}}_mean', f'{{group2_name}}_mean']]
            sns.heatmap(diff_data.T, annot=True, cmap='RdBu_r', center=0, ax=ax2)
            ax2.set_title('Significant Differences (p < 0.05)')
        else:
            ax2.text(0.5, 0.5, 'No significant differences\\nfound (p < 0.05)', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Statistical Significance')
        
        # 3. Distribution comparison for most significant AA
        if significant_differences:
            most_sig_aa = min(significant_differences, key=lambda x: x[1])[0]
            
            ax3.hist(group1_data[most_sig_aa].dropna(), alpha=0.7, 
                    label=f'{{group1_name}} (n={{len(group1_data)}})', bins=15)
            ax3.hist(group2_data[most_sig_aa].dropna(), alpha=0.7, 
                    label=f'{{group2_name}} (n={{len(group2_data)}})', bins=15)
            ax3.set_title(f'Distribution: {{most_sig_aa}} (most significant)')
            ax3.set_xlabel('Frequency')
            ax3.set_ylabel('Count')
            ax3.legend()
        
        # 4. Summary statistics table
        ax4.axis('tight')
        ax4.axis('off')
        
        summary_stats = []
        for group_name, df in group_data.items():
            summary_stats.append([
                group_name,
                len(df),
                f"{{df.mean().mean():.3f}}",
                f"{{df.std().mean():.3f}}",
                len(significant_differences) if group_name == group1_name else "-"
            ])
        
        table = ax4.table(cellText=summary_stats,
                         colLabels=['Group', 'N', 'Mean AA Freq', 'Std AA Freq', 'Sig. Diff.'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax4.set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.savefig('amino_acid_composition_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Summary
        print(f"\\n=== Analysis Summary ===")
        print(f"Groups compared: {{', '.join(group_names)}}")
        print(f"Total proteins analyzed: {{sum(len(df) for df in group_data.values())}}")
        print(f"Significant differences (p < 0.05): {{len(significant_differences)}}")
        
        if significant_differences:
            print("\\nMost significant amino acid differences:")
            for aa, p_val, mean1, mean2 in sorted(significant_differences, key=lambda x: x[1])[:5]:
                fold_change = mean2 / mean1 if mean1 > 0 else float('inf')
                print(f"  {{aa}}: {{fold_change:.2f}}x enrichment, p={{p_val:.2e}}")
        
        print("\\nVisualization saved as: amino_acid_composition_analysis.png")

except Exception as e:
    print(f"Error in amino acid composition analysis: {{e}}")
    import traceback
    traceback.print_exc()
"""
        
        return await self.client.execute_code(analysis_code)
    
    async def calculate_hydrophobicity_comparison(self, protein_groups: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Calculate and compare hydrophobicity profiles between protein groups.
        
        Args:
            protein_groups: Dict mapping group name to list of protein IDs
            
        Returns:
            Hydrophobicity analysis results
        """
        hydrophobicity_code = f"""
# Hydrophobicity profile comparison
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
sys.path.append('/app')

try:
    from sequence_service import calculate_hydrophobicity
    import asyncio
    
    protein_groups = {protein_groups}
    
    print("=== Hydrophobicity Profile Analysis ===")
    
    # Calculate hydrophobicity for each group
    group_hydrophobicity = {{}}
    
    for group_name, protein_ids in protein_groups.items():
        print(f"\\nProcessing group: {{group_name}} ({{len(protein_ids)}} proteins)")
        
        hydrophobicity_scores = asyncio.run(calculate_hydrophobicity(protein_ids))
        
        if hydrophobicity_scores:
            group_hydrophobicity[group_name] = list(hydrophobicity_scores.values())
            mean_score = np.mean(list(hydrophobicity_scores.values()))
            std_score = np.std(list(hydrophobicity_scores.values()))
            print(f"  Mean hydrophobicity: {{mean_score:.3f}} ± {{std_score:.3f}}")
        else:
            print(f"  No sequences found for group {{group_name}}")
            group_hydrophobicity[group_name] = []
    
    # Statistical comparison
    if len(group_hydrophobicity) >= 2:
        group_names = list(group_hydrophobicity.keys())
        
        print("\\n=== Statistical Comparison ===")
        
        # Pairwise comparisons
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                group1, group2 = group_names[i], group_names[j]
                scores1, scores2 = group_hydrophobicity[group1], group_hydrophobicity[group2]
                
                if len(scores1) > 0 and len(scores2) > 0:
                    # Welch's t-test (assumes unequal variances)
                    t_stat, p_value = stats.ttest_ind(scores1, scores2, equal_var=False)
                    
                    # Mann-Whitney U test (non-parametric alternative)
                    u_stat, u_p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(scores1) - 1) * np.var(scores1, ddof=1) + 
                                        (len(scores2) - 1) * np.var(scores2, ddof=1)) / 
                                       (len(scores1) + len(scores2) - 2))
                    cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std if pooled_std > 0 else 0
                    
                    print(f"\\n{{group1}} vs {{group2}}:")
                    print(f"  t-test: t={{t_stat:.3f}}, p={{p_value:.3f}}")
                    print(f"  Mann-Whitney U: U={{u_stat:.1f}}, p={{u_p_value:.3f}}")
                    print(f"  Effect size (Cohen's d): {{cohens_d:.3f}}")
                    
                    if p_value < 0.05:
                        direction = "more hydrophobic" if np.mean(scores1) > np.mean(scores2) else "more hydrophilic"
                        print(f"  → {{group1}} is significantly {{direction}} than {{group2}}")
        
        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Box plot comparison
        box_data = []
        box_labels = []
        for group_name, scores in group_hydrophobicity.items():
            if scores:
                box_data.append(scores)
                box_labels.append(f'{{group_name}}\\n(n={{len(scores)}})')
        
        ax1.boxplot(box_data, labels=box_labels)
        ax1.set_title('Hydrophobicity Distribution by Group')
        ax1.set_ylabel('Kyte-Doolittle Hydrophobicity Score')
        ax1.grid(True, alpha=0.3)
        
        # 2. Histogram overlay
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for i, (group_name, scores) in enumerate(group_hydrophobicity.items()):
            if scores:
                ax2.hist(scores, alpha=0.6, label=f'{{group_name}} (n={{len(scores)}})', 
                        color=colors[i % len(colors)], bins=20)
        
        ax2.set_title('Hydrophobicity Score Distribution')
        ax2.set_xlabel('Kyte-Doolittle Score')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Mean comparison with error bars
        means = [np.mean(scores) if scores else 0 for scores in group_hydrophobicity.values()]
        stds = [np.std(scores) if scores else 0 for scores in group_hydrophobicity.values()]
        
        bars = ax3.bar(range(len(group_names)), means, yerr=stds, 
                      capsize=5, alpha=0.7, color=colors[:len(group_names)])
        ax3.set_title('Mean Hydrophobicity by Group')
        ax3.set_ylabel('Mean Kyte-Doolittle Score')
        ax3.set_xticks(range(len(group_names)))
        ax3.set_xticklabels(group_names, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.1,
                    f'{{mean:.2f}}', ha='center', va='bottom')
        
        # 4. Summary statistics
        ax4.axis('tight')
        ax4.axis('off')
        
        summary_data = []
        for group_name, scores in group_hydrophobicity.items():
            if scores:
                summary_data.append([
                    group_name,
                    len(scores),
                    f"{{np.mean(scores):.3f}}",
                    f"{{np.std(scores):.3f}}",
                    f"{{np.min(scores):.3f}}",
                    f"{{np.max(scores):.3f}}"
                ])
        
        table = ax4.table(cellText=summary_data,
                         colLabels=['Group', 'N', 'Mean', 'Std', 'Min', 'Max'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        ax4.set_title('Hydrophobicity Statistics Summary')
        
        plt.tight_layout()
        plt.savefig('hydrophobicity_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("\\nVisualization saved as: hydrophobicity_comparison.png")
        print("\\n=== Biological Interpretation ===")
        
        # Interpret results
        for group_name, scores in group_hydrophobicity.items():
            if scores:
                mean_score = np.mean(scores)
                if mean_score > 0.5:
                    print(f"{{group_name}}: Hydrophobic proteins (score={{mean_score:.2f}}) - likely membrane-associated")
                elif mean_score < -0.5:
                    print(f"{{group_name}}: Hydrophilic proteins (score={{mean_score:.2f}}) - likely cytoplasmic/extracellular")
                else:
                    print(f"{{group_name}}: Mixed hydrophobicity (score={{mean_score:.2f}}) - diverse subcellular locations")

except Exception as e:
    print(f"Error in hydrophobicity analysis: {{e}}")
    import traceback
    traceback.print_exc()
"""
        
        return await self.client.execute_code(hydrophobicity_code)

    async def calculate_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform statistical analysis on genomic data.
        
        Args:
            data: Dictionary containing data to analyze
            
        Returns:
            Statistical analysis results
        """
        stats_code = f"""
# Statistical analysis
import numpy as np
from scipy import stats

data = {data}

print("Statistical Analysis Results:")
print("=" * 40)

for key, values in data.items():
    if isinstance(values, (list, tuple)) and len(values) > 1:
        arr = np.array(values)
        
        print(f"\\n{{key.upper()}}:")
        print(f"  Count: {{len(arr)}}")
        print(f"  Mean: {{np.mean(arr):.3f}}")
        print(f"  Std: {{np.std(arr):.3f}}")
        print(f"  Min: {{np.min(arr):.3f}}")
        print(f"  Max: {{np.max(arr):.3f}}")
        print(f"  Median: {{np.median(arr):.3f}}")
        
        # Test for normality if enough data points
        if len(arr) >= 8:
            stat, p_value = stats.shapiro(arr)
            print(f"  Normality test p-value: {{p_value:.3f}}")

print("\\nAnalysis complete!")
"""
        
        return await self.client.execute_code(stats_code)


# Tool function for integration with the RAG system
async def code_interpreter_tool(
    code: str,
    session_id: Optional[str] = None,
    timeout: int = 30,
    service_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    Execute Python code using the secure code interpreter service.
    
    This function is designed to be used as a tool in the agentic RAG system.
    
    Args:
        code: Python code to execute
        session_id: Optional session ID (will create new if not provided)
        timeout: Execution timeout in seconds
        service_url: URL of the code interpreter service
        
    Returns:
        Dictionary containing execution results
    """
    client = CodeInterpreterClient(service_url)
    
    if session_id:
        client.session_id = session_id
    
    # Check if service is available
    if not await client.health_check():
        return {
            "success": False,
            "error": "Code interpreter service is not available",
            "stdout": "",
            "stderr": "",
            "execution_time": 0
        }
    
    # Execute the code
    result = await client.execute_code(code, timeout=timeout)
    
    # Add session info for future calls
    result["session_id"] = client.session_id
    
    return result