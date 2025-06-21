#!/usr/bin/env python3
"""
Debug script to see exactly what data is passed to the LLM system.
"""

import asyncio
import json
from src.llm.rag_system import GenomicRAG
from src.llm.config import LLMConfig


class DebugGenomicRAG(GenomicRAG):
    """Debug version that logs all data passed to LLM."""
    
    def _format_context(self, context):
        """Override to debug what context is formatted."""
        formatted = super()._format_context(context)
        
        print("=" * 80)
        print("DEBUGGING: Raw context data being passed to LLM")
        print("=" * 80)
        
        print(f"\nStructured data ({len(context.structured_data)} items):")
        for i, item in enumerate(context.structured_data):
            print(f"  [{i}] {json.dumps(item, indent=2)}")
        
        print(f"\nSemantic data ({len(context.semantic_data)} items):")
        for i, item in enumerate(context.semantic_data):
            print(f"  [{i}] {json.dumps(item, indent=2)}")
        
        print(f"\nMetadata: {json.dumps(context.metadata, indent=2)}")
        
        print(f"\nFormatted context string being sent to LLM:")
        print("-" * 40)
        print(formatted)
        print("-" * 40)
        
        return formatted
    
    async def _retrieve_context(self, query_type: str, retrieval_plan):
        """Override to debug context retrieval."""
        print("=" * 80)
        print("DEBUGGING: Context retrieval")
        print("=" * 80)
        
        print(f"Query type: {query_type}")
        print(f"Retrieval plan:")
        if hasattr(retrieval_plan, 'neo4j_query'):
            print(f"  Neo4j query: {retrieval_plan.neo4j_query}")
        if hasattr(retrieval_plan, 'protein_search'):
            print(f"  Protein search: {retrieval_plan.protein_search}")
        if hasattr(retrieval_plan, 'search_strategy'):
            print(f"  Search strategy: {retrieval_plan.search_strategy}")
        
        # Call parent method
        context = await super()._retrieve_context(query_type, retrieval_plan)
        
        print(f"\nRetrieved context summary:")
        print(f"  Structured items: {len(context.structured_data)}")
        print(f"  Semantic items: {len(context.semantic_data)}")
        print(f"  Query time: {context.query_time:.3f}s")
        
        return context


async def debug_protein_query():
    """Debug a specific protein query."""
    config = LLMConfig()
    rag_system = DebugGenomicRAG(config)
    
    try:
        # Test the exact protein that should have enriched data
        question = "What is the function of KEGG ortholog K20469?"
        
        print(f"DEBUGGING QUESTION: {question}")
        print("=" * 80)
        
        response = await rag_system.ask(question)
        
        print("=" * 80)
        print("FINAL RESPONSE:")
        print("=" * 80)
        print(json.dumps(response, indent=2))
        
    finally:
        if hasattr(rag_system, 'close') and callable(rag_system.close):
            await rag_system.close()


async def debug_neo4j_direct():
    """Test Neo4j queries directly."""
    from src.llm.query_processor import Neo4jQueryProcessor
    from src.llm.config import LLMConfig
    
    config = LLMConfig()
    processor = Neo4jQueryProcessor(config)
    
    print("=" * 80)
    print("DEBUGGING: Direct Neo4j queries")
    print("=" * 80)
    
    # Test KEGG function query
    query1 = """
    MATCH (ko:KEGGOrtholog {id: "K20469"})
    RETURN ko.id as ko_id, ko.description as description, ko.simplifiedDescription as simplified
    """
    
    print(f"Query 1: {query1}")
    result1 = await processor.process_query(query1, query_type="cypher")
    print(f"Result 1: {json.dumps(result1.results, indent=2)}")
    
    # Test protein with function query
    query2 = """
    MATCH (p:Protein {id: "PLM0_60_b1_sep16_scaffold_10001_curated_6"})
    OPTIONAL MATCH (p)-[:hasFunction]->(ko:KEGGOrtholog)
    RETURN p.id as protein_id, ko.id as ko_id, ko.description as ko_description
    """
    
    print(f"\nQuery 2: {query2}")
    result2 = await processor.process_query(query2, query_type="cypher")
    print(f"Result 2: {json.dumps(result2.results, indent=2)}")
    
    if hasattr(processor, 'close') and callable(processor.close):
        await processor.close()


if __name__ == "__main__":
    print("Choose debugging mode:")
    print("1. Debug full LLM pipeline")
    print("2. Debug Neo4j queries only")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(debug_protein_query())
    elif choice == "2":
        asyncio.run(debug_neo4j_direct())
    else:
        print("Invalid choice")