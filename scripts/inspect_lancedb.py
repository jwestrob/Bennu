#!/usr/bin/env python3
"""Inspect LanceDB protein embeddings: count rows, print sample IDs."""

import argparse
import sys

try:
    import lancedb
except ModuleNotFoundError:
    print("lancedb not installed; activate genome-kg env", file=sys.stderr)
    sys.exit(1)

from src.llm.config import LLMConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect protein embeddings table")
    parser.add_argument("--limit", type=int, default=5, help="Number of rows to preview")
    args = parser.parse_args()

    cfg = LLMConfig()
    db = lancedb.connect(cfg.database.lancedb_path)
    table = db.open_table("protein_embeddings")
    total = len(table)
    print(f"Total embeddings: {total}")
    if total == 0:
        return
    df = table.to_lance().to_table(columns=["protein_id", "genome_id"]).to_pandas(limit=args.limit)
    print(df)


if __name__ == "__main__":
    main()
