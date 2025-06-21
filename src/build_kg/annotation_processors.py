#!/usr/bin/env python3
"""
Annotation processors for converting HMM search results to knowledge graph entities.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AnnotationProcessor:
    """Base class for processing different annotation types."""
    
    def __init__(self, annotation_type: str, keep_multiple: bool = True):
        self.annotation_type = annotation_type
        self.keep_multiple = keep_multiple
    
    def load_hits(self, hits_file: Path) -> pd.DataFrame:
        """Load annotation hits from TSV file."""
        try:
            df = pd.read_csv(hits_file, sep='\t')
            logger.info(f"Loaded {len(df)} {self.annotation_type} hits from {hits_file}")
            return df
        except Exception as e:
            logger.error(f"Failed to load {hits_file}: {e}")
            return pd.DataFrame()
    
    def filter_all_significant(self, hits_df: pd.DataFrame, 
                             evalue_threshold: float = 1e-5) -> pd.DataFrame:
        """Keep all hits above significance threshold."""
        filtered = hits_df[hits_df['evalue'] <= evalue_threshold].copy()
        logger.info(f"Filtered {len(filtered)} significant {self.annotation_type} hits "
                   f"from {len(hits_df)} total (E-value <= {evalue_threshold})")
        return filtered
    
    def select_best_per_protein(self, hits_df: pd.DataFrame,
                              evalue_threshold: float = 1e-5) -> pd.DataFrame:
        """Select best hit per protein based on bitscore."""
        # First filter by significance
        significant = hits_df[hits_df['evalue'] <= evalue_threshold].copy()
        
        # Select best hit per protein (highest bitscore)
        best_hits = significant.loc[significant.groupby('sequence_id')['bitscore'].idxmax()]
        
        logger.info(f"Selected {len(best_hits)} best {self.annotation_type} hits "
                   f"from {len(significant)} significant hits across "
                   f"{significant['sequence_id'].nunique()} proteins")
        
        return best_hits
    
    def process_hits(self, hits_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Process annotation hits according to type-specific rules."""
        if self.keep_multiple:
            return self.filter_all_significant(hits_df, **kwargs)
        else:
            return self.select_best_per_protein(hits_df, **kwargs)


class PfamProcessor(AnnotationProcessor):
    """Processor for PFAM domain annotations - keeps all significant domains."""
    
    def __init__(self):
        super().__init__("PFAM", keep_multiple=True)
    
    def create_domain_entities(self, hits_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create protein domain entities from PFAM hits."""
        domains = []
        
        for _, hit in hits_df.iterrows():
            domain = {
                "domain_id": f"{hit['sequence_id']}/domain/{hit['hmm_name']}/{hit['env_from']}-{hit['env_to']}",
                "protein_id": hit['sequence_id'],
                "pfam_id": hit['hmm_name'],
                "start_pos": int(hit['env_from']),
                "end_pos": int(hit['env_to']),
                "bitscore": float(hit['bitscore']),
                "evalue": float(hit['evalue']),
                "dom_bitscore": float(hit.get('dom_bitscore', hit['bitscore']))
            }
            domains.append(domain)
        
        logger.info(f"Created {len(domains)} PFAM domain entities")
        return domains


class KofamProcessor(AnnotationProcessor):
    """Processor for KOFAM functional annotations - best hit per protein only."""
    
    def __init__(self):
        super().__init__("KOFAM", keep_multiple=False)
    
    def create_functional_entities(self, hits_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create functional annotation entities from KOFAM hits."""
        functions = []
        
        for _, hit in hits_df.iterrows():
            function = {
                "annotation_id": f"{hit['sequence_id']}/function/{hit['hmm_name']}",
                "protein_id": hit['sequence_id'],
                "ko_id": hit['hmm_name'],
                "bitscore": float(hit['bitscore']),
                "evalue": float(hit['evalue']),
                "confidence": "high" if hit['evalue'] <= 1e-10 else "medium"
            }
            functions.append(function)
        
        logger.info(f"Created {len(functions)} KOFAM functional entities")
        return functions


def process_astra_results(astra_output_dir: Path) -> Dict[str, Any]:
    """
    Process all astra annotation results from output directory.
    
    Args:
        astra_output_dir: Directory containing astra scan results
        
    Returns:
        Dict containing processed annotations by type
    """
    results = {
        "pfam_domains": [],
        "kofam_functions": [],
        "processing_stats": {}
    }
    
    # Process PFAM results
    pfam_results_dir = astra_output_dir / "pfam_results"
    pfam_hits_file = pfam_results_dir / "PFAM_hits_df.tsv"
    
    if pfam_hits_file.exists():
        pfam_processor = PfamProcessor()
        pfam_hits = pfam_processor.load_hits(pfam_hits_file)
        if not pfam_hits.empty:
            processed_pfam = pfam_processor.process_hits(pfam_hits)
            results["pfam_domains"] = pfam_processor.create_domain_entities(processed_pfam)
            results["processing_stats"]["pfam_total_hits"] = len(pfam_hits)
            results["processing_stats"]["pfam_significant_hits"] = len(processed_pfam)
    
    # Process KOFAM results
    kofam_results_dir = astra_output_dir / "kofam_results"
    kofam_hits_file = kofam_results_dir / "KOFAM_hits_df.tsv"
    
    if kofam_hits_file.exists():
        kofam_processor = KofamProcessor()
        kofam_hits = kofam_processor.load_hits(kofam_hits_file)
        if not kofam_hits.empty:
            processed_kofam = kofam_processor.process_hits(kofam_hits)
            results["kofam_functions"] = kofam_processor.create_functional_entities(processed_kofam)
            results["processing_stats"]["kofam_total_hits"] = len(kofam_hits)
            results["processing_stats"]["kofam_best_hits"] = len(processed_kofam)
            results["processing_stats"]["kofam_proteins_annotated"] = processed_kofam['sequence_id'].nunique()
    
    return results


def main():
    """Test the annotation processors."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python annotation_processors.py <astra_output_dir>")
        sys.exit(1)
    
    astra_dir = Path(sys.argv[1])
    
    # Test processing
    results = process_astra_results(astra_dir)
    
    print("\n=== Processing Results ===")
    print(f"PFAM domains: {len(results['pfam_domains'])}")
    print(f"KOFAM functions: {len(results['kofam_functions'])}")
    print(f"Processing stats: {results['processing_stats']}")
    
    # Show sample results
    if results['pfam_domains']:
        print(f"\nSample PFAM domain: {results['pfam_domains'][0]}")
    
    if results['kofam_functions']:
        print(f"\nSample KOFAM function: {results['kofam_functions'][0]}")


if __name__ == "__main__":
    main()