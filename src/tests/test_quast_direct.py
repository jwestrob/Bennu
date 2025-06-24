#!/usr/bin/env python3
"""
Test QUAST quality metrics integration directly.
"""

import asyncio
import sys
sys.path.append('src')

from llm.config import LLMConfig
from llm.query_processor import Neo4jQueryProcessor

async def test_quast_metrics():
    config = LLMConfig.from_env()
    processor = Neo4jQueryProcessor(config)
    
    # Test direct cypher query for quality metrics
    query = """
    MATCH (genome:Genome {id: 'Candidatus_Muproteobacteria_bacterium_RIFCSPHIGHO2_01_FULL_61_200_contigs'})
    OPTIONAL MATCH (genome)-[:HASQUALITYMETRICS]->(qm:QualityMetrics)
    OPTIONAL MATCH (g:Gene)-[:BELONGSTOGENOME]->(genome)
    OPTIONAL MATCH (p:Protein)-[:ENCODEDBY]->(g)
    RETURN genome.id AS genome_id,
           qm.quast_contigs AS contig_count,
           qm.quast_total_length AS total_length_bp,
           qm.quast_largest_contig AS largest_contig_bp,
           qm.quast_n50 AS n50,
           qm.quast_n75 AS n75,
           qm.quast_gc_content AS gc_content,
           qm.quast_n_count AS ambiguous_base_count,
           qm.quast_n_per_100_kbp AS ambiguous_bases_per_100kbp,
           qm.quast_contigs_1000bp AS contigs_gt_1kbp,
           qm.quast_contigs_5000bp AS contigs_gt_5kbp,
           qm.quast_contigs_10000bp AS contigs_gt_10kbp,
           qm.quast_l50 AS l50,
           qm.quast_l75 AS l75,
           count(DISTINCT g) AS gene_count,
           count(DISTINCT p) AS protein_count
    """
    
    print("Testing QUAST quality metrics retrieval...")
    result = await processor.process_query(query, query_type="cypher")
    
    if result.results:
        print("✅ Quality metrics successfully retrieved:")
        record = result.results[0]
        
        print(f"Genome: {record['genome_id']}")
        
        # Debug: Print all values and types
        print("\nDEBUG - Raw values and types:")
        for key, value in record.items():
            print(f"  {key}: {value} (type: {type(value)})")
        
        print(f"\nAssembly Quality:")
        print(f"  - Contigs: {record['contig_count']}")
        
        # Handle potential None/string values safely
        def safe_format_int(val, default="N/A"):
            try:
                if val is None:
                    return default
                if isinstance(val, int):
                    return f"{val:,}"
                if isinstance(val, str):
                    return f"{int(val):,}"
                return default
            except (ValueError, TypeError):
                return default
            
        def safe_format_float(val, precision=2, default="N/A"):
            try:
                if val is None:
                    return default
                if isinstance(val, (int, float)):
                    return f"{val:.{precision}f}"
                if isinstance(val, str):
                    return f"{float(val):.{precision}f}"
                return default
            except (ValueError, TypeError):
                return default
            
        def safe_format_percent(val, default="N/A"):
            try:
                if val is None:
                    return default
                if isinstance(val, (int, float)):
                    return f"{val:.1%}"
                if isinstance(val, str):
                    return f"{float(val):.1%}"
                return default
            except (ValueError, TypeError):
                return default
        
        print(f"  - Total length: {safe_format_int(record['total_length_bp'])} bp")
        print(f"  - Largest contig: {safe_format_int(record['largest_contig_bp'])} bp")
        print(f"  - N50: {safe_format_int(record['n50'])} bp")
        print(f"  - N75: {safe_format_int(record['n75'])} bp")
        print(f"  - GC content: {safe_format_percent(record['gc_content'])}")
        print(f"  - Ambiguous bases: {safe_format_int(record['ambiguous_base_count'])}")
        print(f"  - Ambiguous bases per 100kbp: {safe_format_float(record['ambiguous_bases_per_100kbp'])}")
        print(f"Contig Size Distribution:")
        print(f"  - Contigs ≥1kb: {safe_format_int(record['contigs_gt_1kbp'])}")
        print(f"  - Contigs ≥5kb: {safe_format_int(record['contigs_gt_5kbp'])}")
        print(f"  - Contigs ≥10kb: {safe_format_int(record['contigs_gt_10kbp'])}")
        print(f"Scaffold Statistics:")
        print(f"  - L50: {safe_format_int(record['l50'])}")
        print(f"  - L75: {safe_format_int(record['l75'])}")
        print(f"Gene Content:")
        print(f"  - Genes: {safe_format_int(record['gene_count'])}")
        print(f"  - Proteins: {safe_format_int(record['protein_count'])}")
        
        return True
    else:
        print("❌ No quality metrics found")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_quast_metrics())
    sys.exit(0 if success else 1)