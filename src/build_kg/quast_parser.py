#!/usr/bin/env python3
"""
QUAST quality metrics parser for genome knowledge graph integration.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def parse_quast_report(report_path: Path) -> Dict[str, Any]:
    """
    Parse QUAST report.tsv file to extract quality metrics.
    
    Args:
        report_path: Path to QUAST report.tsv file
        
    Returns:
        Dictionary with parsed quality metrics
    """
    if not report_path.exists():
        logger.warning(f"QUAST report not found: {report_path}")
        return {}
    
    metrics = {}
    
    try:
        with open(report_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Skip true comment lines (lines that start with # but aren't metrics)
                if line.startswith('#') and '\t' not in line:
                    continue
                
                # Split on tab
                parts = line.split('\t')
                if len(parts) != 2:
                    continue
                
                key, value = parts
                key = key.strip()
                value = value.strip()
                
                # Parse key metrics
                if key == 'Total length':
                    metrics['total_length'] = int(value)
                elif key == 'N50':
                    metrics['n50'] = int(value)
                elif key == '# contigs':
                    metrics['num_contigs'] = int(value)
                elif key == 'GC (%)':
                    metrics['gc_content'] = float(value)
                elif key == 'Largest contig':
                    metrics['largest_contig'] = int(value)
                elif key == 'N90':
                    metrics['n90'] = int(value)
                elif key == 'L50':
                    metrics['l50'] = int(value)
                elif key == 'L90':
                    metrics['l90'] = int(value)
                elif key == 'auN':
                    metrics['aun'] = float(value)
                elif key == "# N's per 100 kbp":
                    metrics['ns_per_100kb'] = float(value)
                # Size distribution metrics
                elif key == '# contigs (>= 1000 bp)':
                    metrics['contigs_1kb_plus'] = int(value)
                elif key == '# contigs (>= 5000 bp)':
                    metrics['contigs_5kb_plus'] = int(value)
                elif key == '# contigs (>= 10000 bp)':
                    metrics['contigs_10kb_plus'] = int(value)
                elif key == '# contigs (>= 25000 bp)':
                    metrics['contigs_25kb_plus'] = int(value)
                elif key == '# contigs (>= 50000 bp)':
                    metrics['contigs_50kb_plus'] = int(value)
                # Additional size distribution metrics
                elif key == '# contigs (>= 0 bp)':
                    metrics['contigs_0bp_plus'] = int(value)
                
        logger.info(f"Parsed {len(metrics)} quality metrics from {report_path}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error parsing QUAST report {report_path}: {e}")
        return {}


def extract_genome_id_from_quast_path(quast_path: Path) -> Optional[str]:
    """
    Extract genome ID from QUAST output directory path.
    
    Args:
        quast_path: Path to QUAST genome directory
        
    Returns:
        Genome ID matching the pipeline naming convention
    """
    # Directory structure: data/stage01_quast/genomes/{genome_id}/
    if quast_path.name.endswith('_contigs'):
        return quast_path.name
    else:
        # Handle edge cases
        return f"{quast_path.name}_contigs"


def collect_all_quast_metrics(stage01_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Collect quality metrics for all genomes in QUAST stage output.
    
    Args:
        stage01_dir: Path to stage01_quast directory
        
    Returns:
        Dictionary mapping genome_id -> quality_metrics
    """
    genomes_dir = stage01_dir / "genomes"
    all_metrics = {}
    
    if not genomes_dir.exists():
        logger.warning(f"QUAST genomes directory not found: {genomes_dir}")
        return all_metrics
    
    for genome_dir in genomes_dir.iterdir():
        if not genome_dir.is_dir():
            continue
            
        genome_id = extract_genome_id_from_quast_path(genome_dir)
        if not genome_id:
            continue
            
        report_path = genome_dir / "report.tsv"
        metrics = parse_quast_report(report_path)
        
        if metrics:
            all_metrics[genome_id] = metrics
            logger.info(f"Collected metrics for genome: {genome_id}")
    
    logger.info(f"Collected QUAST metrics for {len(all_metrics)} genomes")
    return all_metrics


def validate_quality_metrics(metrics: Dict[str, Any]) -> bool:
    """
    Validate that quality metrics contain essential fields.
    
    Args:
        metrics: Quality metrics dictionary
        
    Returns:
        True if metrics are valid, False otherwise
    """
    required_fields = ['total_length', 'n50', 'num_contigs']
    
    for field in required_fields:
        if field not in metrics:
            logger.warning(f"Missing required quality metric: {field}")
            return False
        
        if not isinstance(metrics[field], (int, float)):
            logger.warning(f"Invalid quality metric type for {field}: {type(metrics[field])}")
            return False
    
    return True


def format_quality_metrics_for_rdf(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format quality metrics for RDF integration.
    
    Args:
        metrics: Raw quality metrics
        
    Returns:
        Formatted metrics ready for RDF builder
    """
    formatted = {}
    
    # Map metric names to RDF property names
    metric_mapping = {
        'total_length': 'totalLength',
        'n50': 'n50',
        'num_contigs': 'numContigs',
        'gc_content': 'gcContent',
        'largest_contig': 'largestContig',
        'n90': 'n90',
        'l50': 'l50',
        'l90': 'l90',
        'aun': 'auN',
        'ns_per_100kb': 'nsPer100kb',
        'contigs_1kb_plus': 'contigs1kbPlus',
        'contigs_5kb_plus': 'contigs5kbPlus',
        'contigs_10kb_plus': 'contigs10kbPlus',
        'contigs_25kb_plus': 'contigs25kbPlus',
        'contigs_50kb_plus': 'contigs50kbPlus'
    }
    
    for raw_key, rdf_key in metric_mapping.items():
        if raw_key in metrics:
            formatted[rdf_key] = metrics[raw_key]
    
    return formatted


if __name__ == "__main__":
    # Test the parser
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python quast_parser.py <stage01_quast_dir>")
        sys.exit(1)
    
    stage01_dir = Path(sys.argv[1])
    metrics = collect_all_quast_metrics(stage01_dir)
    
    print(f"Collected metrics for {len(metrics)} genomes:")
    for genome_id, genome_metrics in metrics.items():
        print(f"\n{genome_id}:")
        for key, value in genome_metrics.items():
            print(f"  {key}: {value}")