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
        # Lazy-initialized mapping of short name -> accession from reference TSV
        self._pfam_name_to_acc = None

    def _load_pfam_reference(self) -> None:
        if self._pfam_name_to_acc is not None:
            return
        from pathlib import Path
        pfam_map = {}
        ref_path = Path("data/reference/pfam_id_desc.tsv")
        if ref_path.exists():
            try:
                with ref_path.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        parts = line.rstrip("\n").split("\t")
                        if not parts or len(parts) < 2:
                            continue
                        acc = parts[0].strip()
                        short = parts[1].strip()
                        if acc and short and short not in pfam_map:
                            pfam_map[short] = acc
            except Exception:
                pfam_map = {}
        self._pfam_name_to_acc = pfam_map

    @staticmethod
    def _parse_accession(acc_raw: str) -> (str, int | None):
        """Return (PFxxxxx, version?) from a raw accession like 'PF00016.26' or 'PF00016'."""
        import re
        if not acc_raw:
            return "", None
        m = re.match(r"^(PF\d{5})(?:\.(\d+))?$", str(acc_raw).strip())
        if m:
            base = m.group(1)
            ver = int(m.group(2)) if m.group(2) else None
            return base, ver
        # Try to find embedded PFxxxxx
        m2 = re.search(r"(PF\d{5})", str(acc_raw))
        if m2:
            return m2.group(1), None
        return "", None
    
    def create_domain_entities(self, hits_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create protein domain entities from PFAM hits."""
        domains = []
        # Load mapping once if needed
        self._load_pfam_reference()

        for _, hit in hits_df.iterrows():
            short_name = str(hit.get('hmm_name', '')).strip()
            acc_raw = str(hit.get('hmm_acc', '')).strip() if 'hmm_acc' in hit else ''
            base_acc, version = self._parse_accession(acc_raw)

            # Fallback mapping from short name → accession if hmm_acc was absent
            if not base_acc and short_name and isinstance(self._pfam_name_to_acc, dict):
                mapped = self._pfam_name_to_acc.get(short_name)
                if mapped:
                    base_acc, _ = self._parse_accession(mapped)

            # As a last resort, if short_name already looks like PFxxxxx, use it
            if not base_acc:
                base_guess, _ = self._parse_accession(short_name)
                if base_guess:
                    base_acc = base_guess

            domain = {
                "domain_id": f"{hit['sequence_id']}/domain/{short_name}/{hit['env_from']}-{hit['env_to']}",
                "protein_id": hit['sequence_id'],
                # Use canonical unversioned PF accession for family ID if available; otherwise fall back to short name
                "pfam_id": base_acc or short_name,
                "pfam_name": short_name,
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
