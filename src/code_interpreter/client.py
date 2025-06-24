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