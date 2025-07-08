#!/usr/bin/env python3
"""
Data scaling strategies for handling datasets of different sizes.
Implements tiered approach: small (≤100), medium (100-1000), large (>1000).
"""

import logging
from typing import Dict, Any, List
from abc import ABC, abstractmethod
import re

logger = logging.getLogger(__name__)

class DataScalingStrategy(ABC):
    """Base class for data scaling approaches."""
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy name for logging."""
        pass
    
    @abstractmethod
    def create_code_enhancement(self, protein_ids: List[str], total_count: int) -> str:
        """Create code interpreter enhancement for this strategy."""
        pass

class SmallDatasetStrategy(DataScalingStrategy):
    """Strategy for ≤100 proteins - full context with enrichment."""
    
    def get_strategy_name(self) -> str:
        return "small_dataset"
    
    def create_code_enhancement(self, protein_ids: List[str], total_count: int) -> str:
        """Enhanced code for small datasets with full protein context."""
        enhanced_code = f"""
# Small Dataset Analysis Setup ({total_count} proteins)
# Full protein IDs and sequences available for detailed analysis

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Available protein IDs for analysis
protein_ids = {protein_ids}
protein_count = {total_count}

print(f"✅ Small dataset analysis ready: {{protein_count}} proteins")
print(f"📊 First 5 protein IDs: {{protein_ids[:5]}}")

# Helper functions for small dataset analysis
def analyze_protein_composition():
    \"\"\"Analyze amino acid composition of proteins.\"\"\"
    print("🧬 Analyzing protein composition...")
    # This would connect to sequence database for composition analysis
    return "Composition analysis ready"

def create_protein_summary():
    \"\"\"Create summary of protein characteristics.\"\"\"
    summary = {{
        'total_proteins': protein_count,
        'analysis_type': 'detailed_individual_analysis',
        'available_data': ['sequences', 'annotations', 'genomic_context']
    }}
    return summary

# Ready for detailed analysis
analysis_summary = create_protein_summary()
print(f"📋 Analysis summary: {{analysis_summary}}")
"""
        return enhanced_code

class MediumDatasetStrategy(DataScalingStrategy):
    """Strategy for 100-1000 proteins - batch processing with sampling."""
    
    def get_strategy_name(self) -> str:
        return "medium_dataset"
    
    def create_code_enhancement(self, protein_ids: List[str], total_count: int) -> str:
        """Enhanced code for medium datasets with batch processing."""
        # For medium datasets, pass first 100 proteins for detailed analysis
        sample_ids = protein_ids[:100]
        enhanced_code = f"""
# Medium Dataset Analysis Setup ({total_count} proteins)
# Batch processing with representative sampling

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import islice

# Full dataset info
total_protein_count = {total_count}
sample_protein_ids = {sample_ids}
sample_size = len(sample_protein_ids)

print(f"✅ Medium dataset analysis ready: {{total_protein_count}} total proteins")
print(f"📊 Analyzing sample of {{sample_size}} proteins for detailed analysis")
print(f"🎯 Sample IDs (first 5): {{sample_protein_ids[:5]}}")

# Helper functions for medium dataset analysis
def analyze_sample_batch(batch_size=20):
    \"\"\"Process proteins in batches for efficiency.\"\"\"
    print(f"🔄 Processing proteins in batches of {{batch_size}}...")
    batches = [sample_protein_ids[i:i+batch_size] for i in range(0, len(sample_protein_ids), batch_size)]
    return len(batches)

def create_scaling_summary():
    \"\"\"Create summary showing scaling approach.\"\"\"
    summary = {{
        'total_proteins': total_protein_count,
        'sample_size': sample_size,
        'analysis_type': 'representative_sampling',
        'scaling_strategy': 'medium_dataset_batch_processing'
    }}
    return summary

# Process in batches
batch_count = analyze_sample_batch()
scaling_summary = create_scaling_summary()
print(f"📋 Scaling summary: {{scaling_summary}}")
print(f"🔄 Ready to process {{batch_count}} batches")
"""
        return enhanced_code

class LargeDatasetStrategy(DataScalingStrategy):
    """Strategy for >1000 proteins - aggregation queries only."""
    
    def get_strategy_name(self) -> str:
        return "large_dataset"
    
    def create_code_enhancement(self, protein_ids: List[str], total_count: int) -> str:
        """Enhanced code for large datasets with aggregation focus."""
        enhanced_code = f"""
# Large Dataset Analysis Setup ({total_count} proteins)
# Statistical aggregation approach - no individual protein processing

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset statistics
total_protein_count = {total_count}

print(f"✅ Large dataset analysis ready: {{total_protein_count}} proteins")
print("📊 Using statistical aggregation approach for large dataset")
print("⚠️  Individual protein analysis not recommended for datasets this size")

# Note: For large datasets, we focus on CSV file analysis rather than individual proteins
print("🗂️  Switching to CSV file-based analysis for efficiency...")

# Check for CSV datasets (this will be populated by file enhancement logic)
if 'datasets' in locals():
    print(f"📁 Available datasets: {{list(datasets.keys()) if 'datasets' in locals() else 'None'}}")
    
    # Helper functions for large dataset aggregation
    def get_distribution_summary(dataset_name):
        \"\"\"Get high-level distribution summary.\"\"\"
        if dataset_name not in datasets:
            return None
        df = datasets[dataset_name]
        return {{
            'total_rows': len(df),
            'unique_genomes': df['genome_id'].nunique() if 'genome_id' in df.columns else 0,
            'data_types': list(df.dtypes.to_dict().keys())
        }}
    
    def create_aggregation_summary():
        \"\"\"Create statistical summary for large dataset.\"\"\"
        summary = {{
            'total_proteins': total_protein_count,
            'analysis_type': 'statistical_aggregation',
            'scaling_strategy': 'large_dataset_csv_analysis',
            'recommendation': 'Use aggregation queries and statistical summaries'
        }}
        return summary
    
    aggregation_summary = create_aggregation_summary()
    print(f"📋 Aggregation approach: {{aggregation_summary}}")
else:
    print("❌ CSV datasets not available - consider using Neo4j aggregation queries")

# Large dataset analysis ready for statistical operations
analysis_mode = "aggregation_only"
print(f"🎯 Analysis mode: {{analysis_mode}}")
"""
        return enhanced_code

class ScalingRouter:
    """Routes to appropriate scaling strategy based on data size."""
    
    def __init__(self):
        self.strategies = {
            "small_dataset": SmallDatasetStrategy(),
            "medium_dataset": MediumDatasetStrategy(), 
            "large_dataset": LargeDatasetStrategy()
        }
        
        # Configurable thresholds
        self.small_threshold = 100
        self.medium_threshold = 1000
        
        # Performance limit for code interpreter
        self.max_proteins_for_code = 500
    
    def choose_strategy(self, estimated_count: int) -> DataScalingStrategy:
        """Choose appropriate strategy based on estimated result count."""
        if estimated_count <= self.small_threshold:
            strategy_name = "small_dataset"
        elif estimated_count <= self.medium_threshold:
            strategy_name = "medium_dataset"
        else:
            strategy_name = "large_dataset"
        
        strategy = self.strategies[strategy_name]
        logger.info(f"📊 Chose {strategy.get_strategy_name()} strategy for {estimated_count} proteins")
        return strategy
    
    def get_protein_limit_for_code(self, estimated_count: int) -> int:
        """Get appropriate protein limit for code interpreter based on dataset size."""
        return min(estimated_count, self.max_proteins_for_code)

def convert_to_count_query(cypher_query: str) -> str:
    """Convert a Cypher query to count estimation format."""
    try:
        # Remove any existing LIMIT clauses
        query_without_limit = re.sub(r'\s+LIMIT\s+\d+', '', cypher_query, flags=re.IGNORECASE)
        
        # Find the RETURN clause and replace with COUNT
        if 'RETURN' in query_without_limit.upper():
            # Split at RETURN and take everything before it
            parts = re.split(r'\bRETURN\b', query_without_limit, flags=re.IGNORECASE)
            if len(parts) >= 2:
                before_return = parts[0]
                # Add count return
                count_query = f"{before_return.strip()} RETURN count(*) as estimated_count"
                logger.debug(f"🔢 Converted to count query: {count_query}")
                return count_query
        
        # Fallback: just add count to the end
        count_query = f"{query_without_limit.strip()} RETURN count(*) as estimated_count"
        logger.debug(f"🔢 Fallback count query: {count_query}")
        return count_query
        
    except Exception as e:
        logger.error(f"❌ Failed to convert to count query: {e}")
        # Return a safe default
        return "MATCH (n) RETURN count(n) as estimated_count LIMIT 1"

def convert_to_aggregated_query(cypher_query: str) -> str:
    """Convert a detailed query to aggregated format for large datasets."""
    try:
        # Remove LIMIT clauses
        query_without_limit = re.sub(r'\s+LIMIT\s+\d+', '', cypher_query, flags=re.IGNORECASE)
        
        # Look for common aggregation patterns
        if 'cazyme' in cypher_query.lower():
            # Convert CAZyme query to aggregation
            aggregated_query = """
            MATCH (p:Protein)-[:HASCAZYME]->(ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily)
            OPTIONAL MATCH (p)-[:ENCODEDBY]->(g:Gene)-[:BELONGSTOGENOME]->(genome:Genome)
            RETURN genome.genomeId AS genome_id, ca.cazymeType AS cazyme_family, 
                   count(*) AS count_per_family 
            ORDER BY genome_id, cazyme_family
            """
        elif 'bgc' in cypher_query.lower():
            # Convert BGC query to aggregation
            aggregated_query = """
            MATCH (genome:Genome)-[:HASBGC]->(bgc:Bgc)
            RETURN genome.genomeId AS genome_id, bgc.bgcProduct AS bgc_product,
                   count(*) AS bgc_count,
                   avg(bgc.averageProbability) AS avg_probability
            ORDER BY genome_id, bgc_product
            """
        else:
            # Generic aggregation by genome
            aggregated_query = query_without_limit.replace(
                'RETURN', 
                'WITH genome.genomeId AS genome_id, count(*) AS total_count RETURN genome_id, total_count ORDER BY'
            )
        
        logger.debug(f"📊 Converted to aggregated query: {aggregated_query}")
        return aggregated_query.strip()
        
    except Exception as e:
        logger.error(f"❌ Failed to convert to aggregated query: {e}")
        # Return a safe aggregation query
        return """
        MATCH (p:Protein)
        OPTIONAL MATCH (p)-[:ENCODEDBY]->(g:Gene)-[:BELONGSTOGENOME]->(genome:Genome)
        RETURN genome.genomeId AS genome_id, count(p) AS protein_count
        ORDER BY protein_count DESC
        """