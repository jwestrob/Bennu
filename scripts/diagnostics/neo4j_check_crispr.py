#!/usr/bin/env python3
"""
Quick CRISPR diagnostics against the augmented Neo4j schema.

Usage:
  python scripts/diagnostics/neo4j_check_crispr.py --limit 10

Environment:
  NEO4J_URI (default bolt://localhost:7687)
  NEO4J_USER / NEO4J_PASSWORD (optional; omit for no-auth docker path)
"""

from __future__ import annotations

import os
import argparse
from typing import Any


def main() -> int:
    try:
        from neo4j import GraphDatabase
    except Exception as e:
        print(f"neo4j driver not available: {e}")
        return 1

    parser = argparse.ArgumentParser(description="CRISPR diagnostics")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    auth = (user, password) if (user and password) else None

    driver = GraphDatabase.driver(uri, auth=auth)
    with driver.session() as s:
        # Global counts
        c = s.run("MATCH (ca:CrisprArray) RETURN count(ca) AS c").single()
        arrays = int(c["c"]) if c else 0
        e = s.run("MATCH ()-[r:NEXT]->() WHERE coalesce(r.crisprBetween,false)=true RETURN count(r) AS c").single()
        next_cross = int(e["c"]) if e else 0
        print(f"CRISPR arrays: {arrays}")
        print(f"NEXT edges crossing arrays: {next_cross}")

        # Per genome top counts
        print("\nArrays per genome (top 10):")
        q = (
            "MATCH (g:Genome)<-[:BELONGSTOGENOME]-(ca:CrisprArray) "
            "RETURN g.id AS genome_id, count(ca) AS arrays ORDER BY arrays DESC, genome_id LIMIT 10"
        )
        for r in s.run(q):
            print(f"  {r['genome_id']}: {r['arrays']}")

        # Example flank edges
        print("\nExample flanks (left/right)")
        qf = (
            "MATCH (gn:Gene)-[f:FLANKS_CRISPR]->(ca:CrisprArray) "
            "RETURN gn.id AS gene_id, ca.id AS crispr_id, f.side AS side, toInteger(f.distanceBp) AS dist "
            "ORDER BY dist ASC LIMIT $limit"
        )
        for r in s.run(qf, limit=args.limit):
            print(f"  {r['side']:<5} gene={r['gene_id']} ca={r['crispr_id']} dist={r['dist']}")

        # Example NEXT edges with crisprBetween
        print("\nNEXT edges crossing arrays (examples)")
        qn = (
            "MATCH (a:Gene)-[r:NEXT]->(b:Gene) WHERE coalesce(r.crisprBetween,false)=true "
            "RETURN a.id AS A, b.id AS B, toInteger(r.crisprCountBetween) AS cnt, r.contig AS contig, toInteger(r.delta) AS delta "
            "ORDER BY cnt DESC, contig, A LIMIT $limit"
        )
        for r in s.run(qn, limit=args.limit):
            print(f"  {r['A']} -> {r['B']} cnt={r['cnt']} contig={r['contig']} delta={r['delta']}")

    driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

