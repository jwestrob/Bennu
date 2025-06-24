#!/usr/bin/env python3
"""Example script to import CSV data into a remote Neo4j instance."""

from pathlib import Path
import argparse

from src.build_kg.neo4j_bulk_loader import Neo4jBulkLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote Neo4j bulk import")
    parser.add_argument("--csv-dir", type=Path, required=True, help="CSV directory")
    parser.add_argument("--host", required=True, help="Neo4j host")
    parser.add_argument("--port", type=int, default=7687, help="Bolt port")
    parser.add_argument("--user", default="neo4j", help="Username")
    parser.add_argument("--password", required=True, help="Password")
    parser.add_argument("--database", default="neo4j", help="Database name")
    args = parser.parse_args()

    loader = Neo4jBulkLoader(
        csv_dir=args.csv_dir,
        neo4j_home=None,
        database_name=args.database,
        host=args.host,
        port=args.port,
        username=args.user,
        password=args.password,
    )
    loader.bulk_import()


if __name__ == "__main__":
    main()
