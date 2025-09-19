#!/usr/bin/env python3
"""Fetch PFAM/KO annotations for given proteins via template."""

import argparse
import asyncio
import sys

try:
    from src.llm.config import LLMConfig
    from src.llm.query_processor import Neo4jQueryProcessor
except ModuleNotFoundError as exc:
    print(f"Import failed: {exc}", file=sys.stderr)
    sys.exit(1)


async def _fetch(ids):
    cfg = LLMConfig()
    neo = Neo4jQueryProcessor(cfg)
    res = await neo.execute_named_template('protein_annotations_by_ids', {'protein_ids': ids})
    print(res.results)
    neo.close()


def main():
    parser = argparse.ArgumentParser(description='Fetch protein annotations')
    parser.add_argument('protein_ids', nargs='+')
    args = parser.parse_args()
    asyncio.run(_fetch(args.protein_ids))


if __name__ == '__main__':
    main()
