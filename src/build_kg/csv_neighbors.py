#!/usr/bin/env python3
"""
Build adjacency relationships (NEXT) from gene node CSV for Neo4j bulk import.

Reads genes.csv with columns: 'id:ID', 'contig', 'startCoordinate', 'endCoordinate', 'strand'.
Outputs next_relationships.csv with columns: ':START_ID', ':END_ID', 'contig', 'delta:long', 'same_strand:boolean'.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Any
from rich.console import Console
from rich.progress import Progress

console = Console()


def build_next_csv(genes_csv: Path, out_csv: Path) -> int:
    if not genes_csv.exists():
        raise FileNotFoundError(f"Genes CSV not found: {genes_csv}")

    console.print(f"Reading genes from {genes_csv}")
    contigs: Dict[str, List[Dict[str, Any]]] = {}

    with genes_csv.open(newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        required = {'id:ID', 'contig', 'startCoordinate', 'endCoordinate'}
        missing = required - set(r.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in genes.csv: {sorted(missing)}")
        for row in r:
            contig = row.get('contig')
            if not contig:
                continue
            try:
                row['_start'] = int(row.get('startCoordinate') or 0)
                row['_end'] = int(row.get('endCoordinate') or row['_start'])
            except Exception:
                # Skip bad rows
                continue
            contigs.setdefault(contig, []).append(row)

    console.print(f"Building NEXT edges for {len(contigs)} contigs")

    total = 0
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([':START_ID', ':END_ID', 'contig', 'delta:long', 'same_strand:boolean'])

        with Progress(console=console) as progress:
            task = progress.add_task("Writing NEXT relationships...", total=len(contigs))
            for contig, genes in contigs.items():
                genes.sort(key=lambda g: g['_start'])
                for i in range(len(genes) - 1):
                    a = genes[i]
                    b = genes[i + 1]
                    a_id = a['id:ID']
                    b_id = b['id:ID']
                    delta = int(b['_start']) - int(a['_end'])
                    same = False
                    if 'strand' in a and 'strand' in b:
                        sa = str(a['strand'])
                        sb = str(b['strand'])
                        same = sa == sb and sa != '' and sb != ''
                    w.writerow([a_id, b_id, contig, delta, str(same).lower()])
                    total += 1
                progress.advance(task)

    console.print(f"[green]✓ Wrote {total:,} NEXT relationships to {out_csv}[/green]")
    return total


def main(argv: List[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Build NEXT relationships from genes.csv for bulk import")
    p.add_argument('--csv-dir', type=Path, default=Path('data/stage07_kg/csv'), help='Directory containing genes.csv')
    p.add_argument('--genes-file', type=Path, default=None, help='Path to genes.csv (overrides --csv-dir)')
    p.add_argument('--out-file', type=Path, default=None, help='Output next_relationships.csv (default under csv-dir)')
    args = p.parse_args(argv)

    genes_csv = args.genes_file or (args.csv_dir / 'genes.csv')
    out_csv = args.out_file or (args.csv_dir / 'next_relationships.csv')
    build_next_csv(genes_csv, out_csv)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

