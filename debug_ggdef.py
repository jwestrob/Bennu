#!/usr/bin/env python3
"""Debug GGDEF domain query specifically."""

import asyncio
from src.llm.rag_system import GenomicRAG
from src.llm.config import LLMConfig


class DebugGGDEF(GenomicRAG):
    """Debug version that shows query generation."""
    
    async def _retrieve_context(self, query_type: str, retrieval_plan):
        print("=== DEBUGGING GGDEF QUERY ===")
        print(f"Query type: {query_type}")
        print(f"Neo4j query: {retrieval_plan.neo4j_query}")
        print(f"Protein search: {retrieval_plan.protein_search}")
        print(f"Search strategy: {retrieval_plan.search_strategy}")
        
        # Call parent and show results
        context = await super()._retrieve_context(query_type, retrieval_plan)
        
        print(f"\nQuery Results:")
        print(f"  Structured items: {len(context.structured_data)}")
        for i, item in enumerate(context.structured_data):
            print(f"    [{i}] {item}")
        
        return context


async def test_ggdef():
    config = LLMConfig()
    rag = DebugGGDEF(config)
    
    try:
        await rag.ask("What can you tell me about proteins that contain GGDEF domains in our dataset?")
    finally:
        pass  # No close method


if __name__ == "__main__":
    asyncio.run(test_ggdef())