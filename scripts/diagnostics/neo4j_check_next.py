#!/usr/bin/env python3
"""
Neo4j adjacency ([:NEXT]) diagnostics and neighborhood sanity checks.

Runs a small battery of Cypher queries to verify:
- [:NEXT] relationships exist globally
- k-step adjacency neighbors from PF00016/ PF00101 seeds
- Flanking (±5 by contig order) neighbors

Prints per-seed summaries and sample neighbor annotations (protein_id, PFAM/KO descriptions).

Usage:
  python scripts/diagnostics/neo4j_check_next.py \
      --uri bolt://localhost:7687 --user neo4j --password your_new_password \
      --k 5 --limit 6

Environment variables (fallbacks): NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import List, Dict, Any

from neo4j import GraphDatabase
import random


class _MockSession:
    def __init__(self, seed_count: int = 5):
        self._seed_count = max(1, seed_count)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query: str, **params):
        class _R:
            def __init__(self, rows):
                self._rows = rows
            def single(self):
                return self._rows[0] if self._rows else None
            def __iter__(self):
                return iter(self._rows)
        q = " ".join((query or "").split()).lower()
        # Global NEXT count
        if "match ()-[:next]->() return count(*) as c" in q:
            return _R([{"c": 188_911}])
        # PF00016 seed query: return N mock proteins
        if "where tolower(coalesce(d.pfamaccession, d.id)) starts with tolower($pf)" in q and "return distinct p.id as pid" in q:
            cap = int(params.get("cap", 5))
            rows = [{"pid": f"protein:MOCK_CONTIG_{i}_1"} for i in range(1, cap + 1)]
            return _R(rows)
        # NEXT degree per seed
        if "optional match (g)-[:next]-() return count(*) as deg" in q:
            # Two seeds with degree=1, others 0 (single-gene contigs)
            pid = params.get("pid")
            deg = 1 if (str(pid).endswith("_1") and any(str(pid).startswith(f"protein:MOCK_CONTIG_{i}") for i in (1, 2))) else 0
            return _R([{"deg": deg}])
        # Genes per contig
        if "match (g:gene {contig: seed.contig}) return count(g) as n" in q:
            pid = params.get("pid")
            n = 2 if any(str(pid).startswith(f"protein:MOCK_CONTIG_{i}") for i in (1, 2)) else 1
            return _R([{"n": n}])
        # Adjacency neighbors (k-step)
        if "match (p:protein {id:$pid})-[:encodedby]->(g:gene) call (g) { match pth=(g)-[:next*.." in q:
            pid = params.get("pid")
            rows = []
            if any(str(pid).startswith(f"protein:MOCK_CONTIG_{i}") for i in (1, 2)):
                rows = [{
                    "gene_id": f"gene:{str(pid).replace('protein:','')}_2",
                    "contig": str(pid).split(":")[1].rsplit("_", 1)[0],
                    "start": 1000,
                    "end": 1800,
                    "strand": "+",
                    "protein_id": f"{str(pid).rsplit('_',1)[0]}_2",
                    "pfams": ["PF00016: RuBisCO large", "PF00126: HTH LysR family"],
                    "kos": ["LysR family transcriptional regulator"],
                }]
            return _R(rows)
        # Flanking neighbors (±n)
        if "with seed, g order by tointeger(g.startcoordinate)" in q and "range(-$flank_n, $flank_n)" in q:
            pid = params.get("pid")
            rows = []
            if any(str(pid).startswith(f"protein:MOCK_CONTIG_{i}") for i in (1, 2)):
                rows = [{
                    "gene_id": f"gene:{str(pid).replace('protein:','')}_2",
                    "contig": str(pid).split(":")[1].rsplit("_", 1)[0],
                    "start": 1000,
                    "end": 1800,
                    "strand": "+",
                    "protein_id": f"{str(pid).rsplit('_',1)[0]}_2",
                    "pfams": ["PF00016: RuBisCO large", "PF00126: HTH LysR family"],
                    "kos": ["LysR family transcriptional regulator"],
                }]
            return _R(rows)
        # Default empty
        return _R([])


class _MockDriver:
    def session(self):
        return _MockSession()
    def close(self):
        pass


def _connect(uri: str, user: str | None, password: str | None, no_auth: bool = False):
    auth = None if no_auth or not (user and password) else (user, password)
    return GraphDatabase.driver(uri, auth=auth)


def _fetch_pf_seeds(session, pf_token: str, cap: int) -> List[str]:
    cy = (
        "MATCH (d:Domain) "
        "WHERE toLower(coalesce(d.pfamAccession, d.id)) STARTS WITH toLower($pf) "
        "WITH d MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d) "
        "RETURN DISTINCT p.id AS pid LIMIT $cap"
    )
    return [r["pid"] for r in session.run(cy, pf=pf_token, cap=int(cap))]


def _next_count(session) -> int:
    return session.run("MATCH ()-[:NEXT]->() RETURN count(*) AS c").single()["c"]


def _deg_next(session, pid: str) -> int:
    cy = (
        "MATCH (p:Protein {id:$pid})-[:ENCODEDBY]->(g:Gene) "
        "OPTIONAL MATCH (g)-[:NEXT]-() RETURN count(*) AS deg"
    )
    return session.run(cy, pid=pid).single()["deg"]

def _deg_prop(session, pid: str) -> int:
    cy = (
        "MATCH (p:Protein {id:$pid})-[:ENCODEDBY]->(g:Gene) RETURN toInteger(coalesce(g.nextDegree,-1)) AS d"
    )
    return session.run(cy, pid=pid).single()["d"]


def _genes_on_contig(session, pid: str) -> int:
    cy = (
        "MATCH (p:Protein {id:$pid})-[:ENCODEDBY]->(seed:Gene) "
        "MATCH (g:Gene {contig: seed.contig}) RETURN count(g) AS n"
    )
    return session.run(cy, pid=pid).single()["n"]


def _adjacency_neighbors(session, pid: str, k: int, cap: int) -> List[Dict[str, Any]]:
    # Build query with fixed-hop length (Cypher does not accept parameterized pattern lengths)
    cy = (
        "MATCH (p:Protein {id:$pid})-[:ENCODEDBY]->(g:Gene) "
        f"CALL (g) {{ MATCH pth=(g)-[:NEXT*..{int(k)}]-(ng:Gene) RETURN DISTINCT ng }} "
        "OPTIONAL MATCH (np:Protein)-[:ENCODEDBY]->(ng) "
        "OPTIONAL MATCH (np)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain) "
        "OPTIONAL MATCH (np)-[:HASFUNCTION]->(ko:KEGGOrtholog) "
        "WITH ng, np, "
        "collect(DISTINCT CASE "
        "  WHEN coalesce(d.pfamAccession, d.id) IS NOT NULL AND coalesce(d.name, d.description) IS NOT NULL AND coalesce(d.name, d.description) <> '' "
        "    THEN coalesce(d.pfamAccession, d.id) + ': ' + coalesce(d.name, d.description) "
        "  WHEN coalesce(d.pfamAccession, d.id) IS NOT NULL "
        "    THEN coalesce(d.pfamAccession, d.id) "
        "  ELSE coalesce(d.name, d.description) "
        "END) AS pfams, "
        "collect(DISTINCT ko.description) AS kos "
        "RETURN ng.id AS gene_id, ng.contig AS contig, toInteger(ng.startCoordinate) AS start, "
        "toInteger(ng.endCoordinate) AS end, ng.strand AS strand, np.id AS protein_id, pfams, kos "
        "ORDER BY start LIMIT $cap"
    )
    return [dict(r) for r in session.run(cy, pid=pid, cap=int(cap))]


def _flanking_neighbors(session, pid: str, flank_n: int, cap: int) -> List[Dict[str, Any]]:
    cy = (
        "MATCH (p:Protein {id:$pid})-[:ENCODEDBY]->(seed:Gene) "
        "MATCH (g:Gene {contig: seed.contig}) "
        "WITH seed, g ORDER BY toInteger(g.startCoordinate) "
        "WITH seed, collect(g) AS gs "
        "WITH seed, gs, [i IN range(0, size(gs)-1) WHERE gs[i].id = seed.id][0] AS idx "
        "WITH seed, gs, idx, range(-$flank_n, $flank_n) AS offsets "
        "UNWIND offsets AS off "
        "WITH seed, gs, idx, off WHERE off <> 0 "
        "WITH gs[(idx + off)] AS ng WHERE (idx + off) >= 0 AND (idx + off) < size(gs) "
        "OPTIONAL MATCH (np:Protein)-[:ENCODEDBY]->(ng) "
        "OPTIONAL MATCH (np)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain) "
        "OPTIONAL MATCH (np)-[:HASFUNCTION]->(ko:KEGGOrtholog) "
        "WITH ng, np, "
        "collect(DISTINCT CASE "
        "  WHEN coalesce(d.pfamAccession, d.id) IS NOT NULL AND coalesce(d.name, d.description) IS NOT NULL AND coalesce(d.name, d.description) <> '' "
        "    THEN coalesce(d.pfamAccession, d.id) + ': ' + coalesce(d.name, d.description) "
        "  WHEN coalesce(d.pfamAccession, d.id) IS NOT NULL "
        "    THEN coalesce(d.pfamAccession, d.id) "
        "  ELSE coalesce(d.name, d.description) "
        "END) AS pfams, "
        "collect(DISTINCT ko.description) AS kos "
        "RETURN ng.id AS gene_id, ng.contig AS contig, toInteger(ng.startCoordinate) AS start, "
        "toInteger(ng.endCoordinate) AS end, ng.strand AS strand, np.id AS protein_id, pfams, kos "
        "ORDER BY start LIMIT $cap"
    )
    return [dict(r) for r in session.run(cy, pid=pid, flank_n=int(flank_n), cap=int(cap))]


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Neo4j [:NEXT] diagnostics and neighborhood checks")
    ap.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    ap.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "your_new_password"))
    ap.add_argument("--k", type=int, default=5, help="Adjacency k for NEXT*..k test")
    ap.add_argument("--flank_n", type=int, default=5, help="Flanking neighbors on each side")
    ap.add_argument("--limit", type=int, default=6, help="Number of PF00016 seeds to test")
    ap.add_argument("--mock", action="store_true", help="Run in mock mode (no DB required)")
    ap.add_argument("--allow_mock_on_failure", action="store_true", help="If connection fails, fall back to mock mode")
    ap.add_argument("--no-auth", action="store_true", help="Connect without authentication (e.g., docker NEO4J_AUTH=none)")
    args = ap.parse_args(argv)

    driver = None
    if args.mock:
        print("[MOCK] Running in mock mode — no DB connection will be attempted.")
        driver = _MockDriver()
    else:
        print(f"Connecting to Neo4j at {args.uri} as {args.user} (no_auth={args.no_auth})")
        try:
            driver = _connect(args.uri, args.user, args.password, no_auth=args.no_auth)
            # Quick ping
            with driver.session() as _s:
                _s.run("RETURN 1 AS ok").single()
        except Exception as e:
            if args.allow_mock_on_failure:
                print(f"[WARN] Connection failed: {e}\n[MOCK] Falling back to mock mode.")
                driver = _MockDriver()
            else:
                print(f"Error connecting to Neo4j: {e}")
                return 2

    with driver.session() as session:
        try:
            nxt = _next_count(session)
            print(f"NEXT edges (global): {nxt:,}")
        except Exception as e:
            print(f"Error counting NEXT edges: {e}")

        # Degree histogram
        try:
            rows = session.run("MATCH (g:Gene) RETURN toInteger(coalesce(g.nextDegree,0)) AS deg, count(*) AS n ORDER BY deg")
            hist = [(r['deg'], r['n']) for r in rows]
            if hist:
                print("Degree histogram (nextDegree):", ", ".join([f"{d}:{n}" for d,n in hist]))
        except Exception as e:
            print(f"Degree histogram unavailable: {e}")

        seeds = _fetch_pf_seeds(session, "pf00016", args.limit)
        print(f"PF00016 seeds found: {len(seeds)} -> {seeds}")

        for pid in seeds:
            print("\n==== Seed:", pid)
            deg = _deg_next(session, pid)
            degp = _deg_prop(session, pid)
            on_contig = _genes_on_contig(session, pid)
            print(f"NEXT degree={deg} (prop={degp}) | genes_on_contig={on_contig}")

            adj = _adjacency_neighbors(session, pid, args.k, 50)
            print(f"Adjacency (k={args.k}) neighbors: {len(adj)}")
            for r in adj[:3]:
                print("  adj:", {k: r[k] for k in ("protein_id","pfams","kos","contig","start","end")})

            flk = _flanking_neighbors(session, pid, args.flank_n, 50)
            print(f"Flanking (±{args.flank_n}) neighbors: {len(flk)}")
            for r in flk[:3]:
                print("  flk:", {k: r[k] for k in ("protein_id","pfams","kos","contig","start","end")})

    driver.close()
    print("\nDiagnostics complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
