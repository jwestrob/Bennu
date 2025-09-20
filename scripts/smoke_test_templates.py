#!/usr/bin/env python3
"""
Lightweight smoke-test for Cypher templates.

- Compiles every template in the registry to catch slot/schema drift.
- Executes a subset with safe default parameters when possible.

Usage:
  python scripts/smoke_test_templates.py --run
  python scripts/smoke_test_templates.py            # compile only
"""
import os
import sys
import argparse
from neo4j import GraphDatabase

sys.path.append('src')
from llm.kg.cypher_templates.registry import SPECS, compile_query  # type: ignore


def get_driver():
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'your_new_password')
    return GraphDatabase.driver(uri, auth=(user, password))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='Execute queries when possible')
    args = parser.parse_args(argv)

    to_run = {
        'pfam_search': {"q": "integrase", "limit": 5},
        'kofam_search': {"q": "integrase", "limit": 5},
        'proteins_with_pfam': {"pfam": "PF00589", "limit": 5},
        'proteins_with_pfams': {"pfams": ["PF00589"], "limit": 5},
        'proteins_with_kos': {"kos": ["K04026"], "limit": 5},
        'count_by_label': {"label": "Protein"},
    }

    print(f"Templates in registry: {len(SPECS)}")
    failures = 0
    driver = get_driver() if args.run else None
    for name in SPECS.keys():
        try:
            slots = to_run.get(name, {})
            cypher, params = compile_query(name, slots)
            print(f"✓ Compiled: {name} params={params}")
            if args.run and slots:
                with driver.session() as s:
                    recs = list(s.run(cypher, params))
                print(f"  → Ran: {name} rows={len(recs)}")
        except Exception as e:
            failures += 1
            print(f"✗ Failed: {name} error={e}")
    if driver:
        driver.close()
    print(f"Done. Failures={failures}")
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())

