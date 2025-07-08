#!/usr/bin/env python3
"""
Code interpreter enhancement for genomic data analysis.
Provides intelligent routing and data access strategies.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from .data_scaling import ScalingRouter
from .utils import safe_log_data

logger = logging.getLogger(__name__)

class CodeEnhancer:
    """Enhances code interpreter with genomic data access and analysis capabilities."""
    
    def __init__(self):
        self.scaling_router = ScalingRouter()
        self.csv_base_path = Path("data/stage07_kg/csv")
    
    async def enhance_code_interpreter(self, task_results: List[Dict], original_code: str = "") -> str:
        """
        Main enhancement logic with intelligent routing.
        
        Args:
            task_results: Results from previous tasks
            original_code: Original user code to enhance
            
        Returns:
            Enhanced code string with data access and analysis setup
        """
        logger.info("🔧 Enhancing code interpreter with intelligent routing")
        
        try:
            # Extract protein IDs and data info from task results
            analysis_info = self._analyze_task_results(task_results)
            
            # Check if CSV files are available for direct access
            csv_available = self._check_csv_availability()
            
            if csv_available and analysis_info["total_proteins"] > 100:
                logger.info("📁 Using file-based enhancement for large dataset")
                return self._enhance_code_interpreter_with_files(analysis_info, original_code)
            elif analysis_info["protein_ids"]:
                logger.info("🧬 Using protein-based enhancement")
                return self._enhance_code_interpreter_with_proteins(analysis_info, original_code)
            else:
                logger.info("📊 Using generic enhancement")
                return self._enhance_code_interpreter_generic(original_code)
                
        except Exception as e:
            logger.error(f"❌ Code interpreter enhancement failed: {e}")
            return f"# Code interpreter enhancement failed: {str(e)}\\n\\n{original_code}"
    
    def _analyze_task_results(self, task_results: List[Dict]) -> Dict[str, Any]:
        """Extract useful information from task results."""
        protein_ids = []
        total_proteins = 0
        data_types = set()
        
        for task_result in task_results:
            if isinstance(task_result, dict):
                # Extract protein IDs from various result formats
                protein_ids.extend(self._extract_protein_ids_from_result(task_result))
                
                # Detect data types
                if 'cazyme' in str(task_result).lower():
                    data_types.add('cazyme')
                if 'bgc' in str(task_result).lower():
                    data_types.add('bgc')
                if 'protein' in str(task_result).lower():
                    data_types.add('protein')
                
                # Get counts
                if 'total_proteins' in task_result:
                    total_proteins = max(total_proteins, task_result['total_proteins'])
                elif 'discovery_count' in task_result:
                    total_proteins = max(total_proteins, task_result['discovery_count'])
        
        # Remove duplicates and limit for performance
        unique_protein_ids = list(set(protein_ids))
        
        return {
            "protein_ids": unique_protein_ids,
            "total_proteins": total_proteins or len(unique_protein_ids),
            "data_types": list(data_types),
            "analysis_scope": "large" if total_proteins > 1000 else "medium" if total_proteins > 100 else "small"
        }
    
    def _extract_protein_ids_from_result(self, result: Dict[str, Any]) -> List[str]:
        """Extract protein IDs from a task result."""
        protein_ids = []
        
        # Handle different result formats
        if 'protein_ids' in result:
            protein_ids.extend(result['protein_ids'])
        
        if 'discovered_proteins' in result:
            proteins = result['discovered_proteins']
            if isinstance(proteins, list):
                for protein in proteins:
                    if isinstance(protein, dict) and 'protein_id' in protein:
                        protein_ids.append(protein['protein_id'])
                    elif isinstance(protein, dict) and 'id' in protein:
                        protein_ids.append(protein['id'])
        
        if 'enriched_proteins' in result:
            proteins = result['enriched_proteins']
            if isinstance(proteins, list):
                for protein in proteins:
                    if isinstance(protein, dict) and 'protein_id' in protein:
                        protein_ids.append(protein['protein_id'])
                    elif isinstance(protein, dict) and 'id' in protein:
                        protein_ids.append(protein['id'])
        
        # Handle tool results from comprehensive discovery
        if 'tool_result' in result:
            tool_result = result['tool_result']
            if isinstance(tool_result, str):
                # Parse tool result if it's a string
                try:
                    # Look for protein IDs in the text
                    import re
                    protein_pattern = r'protein[_:]([A-Za-z0-9_]+)'
                    matches = re.findall(protein_pattern, tool_result)
                    protein_ids.extend([f"protein:{match}" for match in matches])
                except Exception:
                    pass
        
        return protein_ids
    
    def _check_csv_availability(self) -> bool:
        """Check if CSV files are available for direct access."""
        csv_files = [
            "proteins.csv",
            "cazyme_annotations.csv", 
            "bgcs.csv",
            "kegg_orthologs.csv"
        ]
        
        available_files = []
        for csv_file in csv_files:
            file_path = self.csv_base_path / csv_file
            if file_path.exists():
                available_files.append(csv_file)
        
        logger.debug(f"📁 Available CSV files: {available_files}")
        return len(available_files) > 0
    
    def _enhance_code_interpreter_with_files(self, analysis_info: Dict[str, Any], original_code: str) -> str:
        """Enhanced code for direct CSV file access."""
        total_count = analysis_info["total_proteins"]
        data_types = analysis_info.get("data_types", [])
        
        enhanced_code = f"""
# Genomic Data Analysis Setup - CSV File Access
# Dataset: {total_count} proteins, Data types: {data_types}

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up file paths
csv_base_path = Path('/app/data/stage07_kg/csv')
datasets = {{}}

print("🗂️  Loading genomic datasets from CSV files...")

# Load available datasets
csv_files = {{
    'proteins': 'proteins.csv',
    'cazyme_annotations': 'cazyme_annotations.csv',
    'bgcs': 'bgcs.csv', 
    'kegg_orthologs': 'kegg_orthologs.csv',
    'pfam_domains': 'pfam_domains.csv'
}}

for dataset_name, filename in csv_files.items():
    file_path = csv_base_path / filename
    if file_path.exists():
        try:
            df = pd.read_csv(file_path)
            datasets[dataset_name] = df
            print(f"✅ Loaded {{dataset_name}}: {{len(df)}} rows, {{len(df.columns)}} columns")
        except Exception as e:
            print(f"❌ Failed to load {{dataset_name}}: {{e}}")
    else:
        print(f"❌ {{dataset_name}} not found at {{file_path}}")

if not datasets:
    print("❌ No CSV datasets loaded - file paths may need adjustment")
else:
    print(f"✅ Successfully loaded {{len(datasets)}} datasets")
    
    # Show dataset overview
    for name, df in datasets.items():
        print(f"\\n📊 {{name.upper()}} DATASET:")
        print(f"  Rows: {{len(df)}}")
        print(f"  Columns: {{list(df.columns)}}")
        if 'genome_id' in df.columns:
            print(f"  Genomes: {{df['genome_id'].nunique()}}")

# Helper functions for large dataset analysis
def get_distribution_by_genome(dataset_name, groupby_col='genome_id', count_col=None):
    \"\"\"Get distribution of annotations by genome.\"\"\"
    if dataset_name not in datasets:
        print(f"Dataset {{dataset_name}} not available")
        return None
    
    df = datasets[dataset_name]
    if groupby_col not in df.columns:
        print(f"Column {{groupby_col}} not found in {{dataset_name}}")
        return None
    
    if count_col:
        return df.groupby(groupby_col)[count_col].count().sort_values(ascending=False)
    else:
        return df[groupby_col].value_counts()

def get_functional_summary(dataset_name, function_col='function_description'):
    \"\"\"Get summary of functional categories.\"\"\"
    if dataset_name not in datasets:
        return None
    
    df = datasets[dataset_name]
    if function_col not in df.columns:
        return None
    
    return df[function_col].value_counts().head(20)

# Original user code:
{original_code}
"""
        return enhanced_code
    
    def _enhance_code_interpreter_with_proteins(self, analysis_info: Dict[str, Any], original_code: str) -> str:
        """Enhanced code for protein-specific analysis."""
        protein_ids = analysis_info["protein_ids"]
        total_count = analysis_info["total_proteins"]
        
        # Choose appropriate strategy based on dataset size
        strategy = self.scaling_router.choose_strategy(total_count)
        
        # Limit proteins for code interpreter performance
        max_proteins_for_code = self.scaling_router.get_protein_limit_for_code(total_count)
        limited_protein_ids = protein_ids[:max_proteins_for_code]
        
        enhanced_code = strategy.create_code_enhancement(limited_protein_ids, total_count)
        
        # Add original user code
        enhanced_code += f"\\n\\n# Original user code:\\n{original_code}"
        
        return enhanced_code
    
    def _enhance_code_interpreter_generic(self, original_code: str) -> str:
        """Generic enhancement when no specific data is available."""
        enhanced_code = f"""
# Genomic Analysis Setup - Generic Enhancement
# No specific protein data available, setting up general analysis environment

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

print("🧬 Generic genomic analysis environment ready")
print("💡 Use specific queries to get protein data for detailed analysis")

# Helper functions
def analyze_data(data):
    \"\"\"Generic data analysis helper.\"\"\"
    if isinstance(data, pd.DataFrame):
        return {{
            'rows': len(data),
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict()
        }}
    elif isinstance(data, list):
        return {{
            'length': len(data),
            'type': 'list',
            'sample': data[:5] if data else []
        }}
    else:
        return {{'type': type(data).__name__, 'value': str(data)[:100]}}

# Original user code:
{original_code}
"""
        return enhanced_code

class SequenceDatabaseConnector:
    """Connects code interpreter to sequence databases for amino acid analysis."""
    
    def __init__(self, sequence_db_path: Optional[str] = None):
        self.sequence_db_path = sequence_db_path or "data/stage08_esm2/protein_sequences.db"
    
    def generate_sequence_access_code(self, protein_ids: List[str]) -> str:
        """Generate code for accessing protein sequences."""
        sequence_code = f"""
# Protein Sequence Database Access
# Connect to sequence database for amino acid composition analysis

import sqlite3
from collections import Counter

# Database connection
sequence_db_path = "{self.sequence_db_path}"
protein_ids_for_analysis = {protein_ids[:50]}  # Limit for performance

def get_protein_sequence(protein_id):
    \"\"\"Get amino acid sequence for a protein.\"\"\"
    try:
        conn = sqlite3.connect(sequence_db_path)
        cursor = conn.cursor()
        
        # Remove protein: prefix if present
        clean_id = protein_id.replace('protein:', '')
        
        cursor.execute("SELECT sequence FROM protein_sequences WHERE protein_id = ?", (clean_id,))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    except Exception as e:
        print(f"Error getting sequence for {{protein_id}}: {{e}}")
        return None

def analyze_amino_acid_composition(protein_ids):
    \"\"\"Analyze amino acid composition for multiple proteins.\"\"\"
    compositions = {{}}
    
    for protein_id in protein_ids[:10]:  # Limit for demo
        sequence = get_protein_sequence(protein_id)
        if sequence:
            composition = Counter(sequence)
            compositions[protein_id] = composition
            print(f"✅ {{protein_id}}: {{len(sequence)}} amino acids")
        else:
            print(f"❌ No sequence found for {{protein_id}}")
    
    return compositions

# Test sequence access
print("🧬 Testing protein sequence access...")
if protein_ids_for_analysis:
    test_compositions = analyze_amino_acid_composition(protein_ids_for_analysis)
    print(f"📊 Retrieved sequences for {{len(test_compositions)}} proteins")
else:
    print("❌ No protein IDs available for sequence analysis")
"""
        return sequence_code

def extract_protein_ids_from_task_results(task_results: List[Dict]) -> List[str]:
    """Extract protein IDs from task results for code interpreter enhancement."""
    enhancer = CodeEnhancer()
    analysis_info = enhancer._analyze_task_results(task_results)
    return analysis_info["protein_ids"]