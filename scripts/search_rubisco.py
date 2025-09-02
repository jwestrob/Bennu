#!/usr/bin/env python3
"""
Ad-hoc RubisCO search utility.

Connects to Neo4j at bolt://localhost:7687 (neo4j/your_new_password) and searches for
RubisCO by PFAM accessions (PF00016/ PF00101), short names, and description. Also checks
KEGG KOs rbcL (K01601), rbcS (K01602), and PRK (K00855).

If the Python neo4j driver is unavailable, falls back to cypher-shell.
"""

import json
import os
import subprocess
import sys
from typing import Any, Dict, List

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_new_password")


def _print_header(title: str) -> None:
    print("\n" + title)
    print("=" * len(title))


def run_with_driver() -> None:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def q(query: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        with driver.session() as session:
            res = session.run(query, params or {})
            return [dict(r) for r in res]

    # 1) Inspect Domain nodes that mention RubisCO anywhere
    _print_header("Domain nodes that mention RubisCO (id/accession/description contains)")
    query_domains = (
        "MATCH (d:Domain)\n"
        "WHERE toLower(d.id) CONTAINS $q\n"
        "   OR toLower(coalesce(d.pfamAccession,'')) CONTAINS $q\n"
        "   OR toLower(coalesce(d.description,'')) CONTAINS $q\n"
        "RETURN d.id AS id, d.pfamAccession AS pfamAccession, d.description AS description\n"
        "ORDER BY id LIMIT 50"
    )
    rows = q(query_domains, {"q": "rubisco"})
    print(json.dumps(rows, indent=2, ensure_ascii=False))

    # 2) Protein counts linked to PF00016 / PF00101 (prefix match to handle versioning)
    _print_header("Protein counts for PF00016*/PF00101* (prefix match on accession/id)")
    query_pf_counts = (
        "CALL {\n"
        "  WITH 'pf00016' AS acc\n"
        "  MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)\n"
        "  WHERE toLower(coalesce(d.pfamAccession,'')) STARTS WITH acc OR toLower(d.id) STARTS WITH acc\n"
        "  RETURN 'PF00016' AS family, count(DISTINCT p) AS proteins\n"
        "}\n"
        "CALL {\n"
        "  WITH 'pf00101' AS acc\n"
        "  MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)\n"
        "  WHERE toLower(coalesce(d.pfamAccession,'')) STARTS WITH acc OR toLower(d.id) STARTS WITH acc\n"
        "  RETURN 'PF00101' AS family, count(DISTINCT p) AS proteins\n"
        "}\n"
        "RETURN family, proteins"
    )
    rows = q(query_pf_counts)
    print(json.dumps(rows, indent=2, ensure_ascii=False))

    # 3) Sample proteins for PF00016 prefix
    _print_header("Sample proteins with PF00016* (prefix)")
    query_pf_samples = (
        "MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)\n"
        "WHERE toLower(coalesce(d.pfamAccession,'')) STARTS WITH $acc OR toLower(d.id) STARTS WITH $acc\n"
        "RETURN p.id AS protein_id, d.pfamAccession AS pfamAccession, d.id AS domain_id\n"
        "LIMIT 20"
    )
    rows = q(query_pf_samples, {"acc": "pf00016"})
    print(json.dumps(rows, indent=2, ensure_ascii=False))

    # 4) KEGG: rbcL/rbcS/PRK
    _print_header("KEGG KOs rbcL/rbcS/PRK counts (K01601,K01602,K00855)")
    query_ko_counts = (
        "MATCH (p:Protein)-[:HASFUNCTION]->(ko:KEGGOrtholog)\n"
        "WHERE ko.id IN ['K01601','K01602','K00855']\n"
        "RETURN ko.id AS ko_id, count(DISTINCT p) AS proteins ORDER BY ko_id"
    )
    rows = q(query_ko_counts)
    print(json.dumps(rows, indent=2, ensure_ascii=False))

    driver.close()


def run_with_cyphershell() -> None:
    shell = os.getenv("CYPHER_SHELL", "cypher-shell")

    def call(query: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        args = [shell, "-u", NEO4J_USER, "-p", NEO4J_PASSWORD, "-a", NEO4J_URI.replace("bolt://", "bolt://")]
        # Newer cypher-shell supports --format json
        args += ["--format", "json"]
        # We don't have param binding on CLI easily; embed safe literals for this debug use
        if params and "q" in params:
            qval = str(params["q"]).lower().replace("'", "\'")
            query = query.replace("$q", f"'{qval}'")
        if params and "acc" in params:
            aval = str(params["acc"]).lower().replace("'", "\'")
            query = query.replace("$acc", f"'{aval}'")
        cp = subprocess.run(args, input=query.encode("utf-8"), capture_output=True)
        if cp.returncode != 0:
            sys.stderr.write(cp.stderr.decode("utf-8", errors="ignore"))
            return []
        try:
            data = json.loads(cp.stdout.decode("utf-8"))
            # cypher-shell json format returns an object with 'columns' and 'data'
            if isinstance(data, dict) and "data" in data:
                cols = data.get("columns") or []
                out = []
                for row in data.get("data", []):
                    # row: {row: [...], meta: [...]}
                    values = row.get("row", [])
                    out.append({cols[i]: values[i] for i in range(min(len(cols), len(values)))})
                return out
            return []
        except Exception:
            # Fallback: print raw
            print(cp.stdout.decode("utf-8", errors="ignore"))
            return []

    _print_header("Domain nodes that mention RubisCO (id/accession/description contains)")
    q_domains = (
        "MATCH (d:Domain)\n"
        "WHERE toLower(d.id) CONTAINS $q\n"
        "   OR toLower(coalesce(d.pfamAccession,'')) CONTAINS $q\n"
        "   OR toLower(coalesce(d.description,'')) CONTAINS $q\n"
        "RETURN d.id AS id, d.pfamAccession AS pfamAccession, d.description AS description\n"
        "ORDER BY id LIMIT 50"
    )
    print(json.dumps(call(q_domains, {"q": "rubisco"}), indent=2, ensure_ascii=False))

    _print_header("Protein counts for PF00016*/PF00101* (prefix match on accession/id)")
    q_pf_counts = (
        "CALL { WITH 'pf00016' AS acc MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)\n"
        "  WHERE toLower(coalesce(d.pfamAccession,'')) STARTS WITH acc OR toLower(d.id) STARTS WITH acc\n"
        "  RETURN 'PF00016' AS family, count(DISTINCT p) AS proteins }\n"
        "CALL { WITH 'pf00101' AS acc MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)\n"
        "  WHERE toLower(coalesce(d.pfamAccession,'')) STARTS WITH acc OR toLower(d.id) STARTS WITH acc\n"
        "  RETURN 'PF00101' AS family, count(DISTINCT p) AS proteins }\n"
        "RETURN family, proteins"
    )
    print(json.dumps(call(q_pf_counts), indent=2, ensure_ascii=False))

    _print_header("Sample proteins with PF00016* (prefix)")
    q_pf_samples = (
        "MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)\n"
        "WHERE toLower(coalesce(d.pfamAccession,'')) STARTS WITH $acc OR toLower(d.id) STARTS WITH $acc\n"
        "RETURN p.id AS protein_id, d.pfamAccession AS pfamAccession, d.id AS domain_id\n"
        "LIMIT 20"
    )
    print(json.dumps(call(q_pf_samples, {"acc": "pf00016"}), indent=2, ensure_ascii=False))

    _print_header("KEGG KOs rbcL/rbcS/PRK counts (K01601,K01602,K00855)")
    q_ko_counts = (
        "MATCH (p:Protein)-[:HASFUNCTION]->(ko:KEGGOrtholog)\n"
        "WHERE ko.id IN ['K01601','K01602','K00855']\n"
        "RETURN ko.id AS ko_id, count(DISTINCT p) AS proteins ORDER BY ko_id"
    )
    print(json.dumps(call(q_ko_counts), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        from neo4j import GraphDatabase  # noqa: F401
        run_with_driver()
    except Exception:
        run_with_cyphershell()

