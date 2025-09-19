#!/usr/bin/env python3
"""Quick diagnostic: check whether a protein has an embedding in LanceDB."""

import argparse
import asyncio
import sys

try:
    from src.llm.config import LLMConfig
    from src.llm.query_processor import LanceDBQueryProcessor
except ModuleNotFoundError as exc:
    print(f"Import failed: {exc}. Make sure PYTHONPATH includes the repo root.", file=sys.stderr)
    sys.exit(1)


async def _lookup(protein_id: str) -> None:
    cfg = LLMConfig()
    ldb = LanceDBQueryProcessor(cfg)
    pid = protein_id.split(':', 1)[-1]
    rows = await ldb._lookup_protein(pid)
    if not rows:
        print(f"No embedding found for {protein_id}")
    else:
        for row in rows:
            print(f"protein_id: {row.get('protein_id')} | genome_id: {row.get('genome_id')} | length: {row.get('sequence_length')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check LanceDB for a protein embedding")
    parser.add_argument(
        "protein_id",
        nargs="?",
        default="protein:RIFCSPHIGHO2_01_FULL_Acidovorax_64_960_rifcsphigho2_01_scaffold_11_111",
        help="Protein identifier to query (default: rubisco large subunit)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(_lookup(args.protein_id))
    except Exception as exc:  # pragma: no cover
        print(f"Lookup failed: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
