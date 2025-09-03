#!/usr/bin/env python3
"""
Post-load tuning for Neo4j: constraints/indexes and genomic NEXT edges.

This module avoids neo4j-admin and runs safely against an existing database.
It is optimized to stream per contig and batch relationship creation.
"""

from typing import List, Dict, Any
import os
from rich.console import Console
from rich.progress import Progress

console = Console()


def _get_driver():
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise RuntimeError("neo4j Python driver is required. pip install neo4j") from e
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    return GraphDatabase.driver(uri, auth=(user, password))


def create_constraints_and_indexes() -> None:
    statements = [
        # Uniqueness constraints on core IDs
        "CREATE CONSTRAINT genome_id IF NOT EXISTS FOR (g:Genome) REQUIRE g.id IS UNIQUE",
        "CREATE CONSTRAINT genome_genomeId IF NOT EXISTS FOR (g:Genome) REQUIRE g.genomeId IS UNIQUE",
        "CREATE CONSTRAINT gene_id IF NOT EXISTS FOR (g:Gene) REQUIRE g.id IS UNIQUE",
        "CREATE CONSTRAINT protein_id IF NOT EXISTS FOR (p:Protein) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT domain_id IF NOT EXISTS FOR (d:Domain) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT domain_annotation_id IF NOT EXISTS FOR (da:DomainAnnotation) REQUIRE da.id IS UNIQUE",
        "CREATE CONSTRAINT kegg_id IF NOT EXISTS FOR (k:KEGGOrtholog) REQUIRE k.id IS UNIQUE",
        "CREATE CONSTRAINT pathway_id IF NOT EXISTS FOR (pw:Pathway) REQUIRE pw.id IS UNIQUE",
        "CREATE CONSTRAINT bgc_id IF NOT EXISTS FOR (b:Bgc) REQUIRE b.id IS UNIQUE",
        # Composite index for spatial gene scans
        "CREATE INDEX gene_contig_coords IF NOT EXISTS FOR (g:Gene) ON (g.contig, g.startCoordinate, g.endCoordinate)",
        # Helpful single-property indexes
        "CREATE INDEX protein_name IF NOT EXISTS FOR (p:Protein) ON (p.name)",
        "CREATE INDEX domain_name IF NOT EXISTS FOR (d:Domain) ON (d.name)",
        "CREATE INDEX domain_pfamAccession IF NOT EXISTS FOR (d:Domain) ON (d.pfamAccession)",
        "CREATE INDEX kegg_desc IF NOT EXISTS FOR (k:KEGGOrtholog) ON (k.description)",
        # Full-text indexes
        "CREATE FULLTEXT INDEX proteinText IF NOT EXISTS FOR (p:Protein) ON EACH [p.name, p.description]",
        "CREATE FULLTEXT INDEX domainText IF NOT EXISTS FOR (d:Domain) ON EACH [d.id, d.name, d.description]",
        "CREATE FULLTEXT INDEX keggText IF NOT EXISTS FOR (k:KEGGOrtholog) ON EACH [k.id, k.description]",
        "CREATE FULLTEXT INDEX pathwayText IF NOT EXISTS FOR (pw:Pathway) ON EACH [pw.id, pw.name, pw.description]",
    ]

    driver = _get_driver()
    with driver.session() as session:
        for stmt in statements:
            session.run(stmt)
    driver.close()


def precompute_next_edges(batch_size: int = 2000, contig_limit: int | None = None) -> int:
    """Create genomic NEXT edges efficiently by streaming per contig.

    Returns total number of pairs processed.
    """
    list_contigs = (
        "MATCH (g:Gene) WHERE g.contig IS NOT NULL RETURN DISTINCT g.contig AS contig ORDER BY contig"
    )
    genes_for_contig = (
        "MATCH (g:Gene {contig: $contig}) "
        "WHERE g.startCoordinate IS NOT NULL AND g.endCoordinate IS NOT NULL "
        "RETURN g.id AS id, g.startCoordinate AS start, g.endCoordinate AS end, g.strand AS strand "
        "ORDER BY g.startCoordinate"
    )
    create_next_batch = (
        "UNWIND $pairs AS row "
        "MATCH (a:Gene {id: row.a}) MATCH (b:Gene {id: row.b}) "
        "MERGE (a)-[r:NEXT]->(b) "
        "SET r.contig = $contig, r.delta = row.delta, r.same_strand = row.same_strand"
    )

    total_pairs = 0
    driver = _get_driver()
    with driver.session() as session:
        contigs = [rec["contig"] for rec in session.run(list_contigs)]
        if contig_limit:
            contigs = contigs[:contig_limit]
        console.print(f"Linking NEXT edges for {len(contigs)} contigs (batch_size={batch_size})")

        with Progress(console=console) as progress:
            task = progress.add_task("Creating NEXT edges...", total=len(contigs))
            for contig in contigs:
                records = list(session.run(genes_for_contig, contig=contig))
                n = len(records)
                if n < 2:
                    progress.advance(task)
                    continue
                pairs: List[Dict[str, Any]] = []
                for i in range(n - 1):
                    a = records[i]
                    b = records[i + 1]
                    a_end = int(a["end"]) if a["end"] is not None else 0
                    b_start = int(b["start"]) if b["start"] is not None else a_end
                    same_strand = (
                        str(a["strand"]) == str(b["strand"]) if a["strand"] is not None and b["strand"] is not None else False
                    )
                    pairs.append({
                        "a": a["id"],
                        "b": b["id"],
                        "delta": int(b_start) - int(a_end),
                        "same_strand": bool(same_strand),
                    })
                total_pairs += len(pairs)
                for j in range(0, len(pairs), batch_size):
                    chunk = pairs[j:j + batch_size]
                    session.run(create_next_batch, pairs=chunk, contig=contig)
                progress.advance(task)
    driver.close()
    console.print(f"[green]✓ Created/merged ~{total_pairs:,} NEXT edges[/green]")
    return total_pairs


def main(argv: List[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Post-load Neo4j tuning: indexes + NEXT edges")
    parser.add_argument("--create-indexes", action="store_true", help="Create constraints and indexes")
    parser.add_argument("--neighbors-only", action="store_true", help="Only precompute NEXT edges")
    parser.add_argument("--batch-size", type=int, default=2000, help="Batch size for NEXT relationship creation")
    parser.add_argument("--contig-limit", type=int, default=None, help="Limit number of contigs (for quick tests)")
    args = parser.parse_args(argv)

    if not args.neighbors_only:
        console.print("Creating constraints and indexes...")
        create_constraints_and_indexes()

    console.print("Precomputing NEXT edges...")
    precompute_next_edges(batch_size=args.batch_size, contig_limit=args.contig_limit)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
