#!/usr/bin/env python3
"""
Test script to verify DSPy is generating correct CAZyme queries after our fixes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import dspy
    from src.llm.rag_system.dspy_signatures import GenomicQuery, NEO4J_SCHEMA
    from src.llm.rag_system import ContextRetrieverV2
    
    print("✅ Successfully imported DSPy and updated signatures")
    
    # Test the GenomicQuery signature directly
    print("\n🧪 Testing GenomicQuery signature:")
    query_gen = dspy.ChainOfThought(GenomicQuery)
    
    test_question = "Tell me about CAZymes in the database"
    test_context = NEO4J_SCHEMA[:1000]  # Truncate for testing
    
    try:
        result = query_gen(question=test_question, context=test_context)
        print(f"Generated query: {result.query}")
        
        # Check if the query uses correct node labels
        if "CAZymeAnnotation" in result.query and "CAZymeFamily" in result.query:
            print("🎉 SUCCESS: Query uses correct node labels!")
        elif "Cazymeannotation" in result.query or "Cazymefamily" in result.query:
            print("❌ FAILED: Query still uses old incorrect node labels")
        else:
            print("⚠️ PARTIAL: Query doesn't contain CAZyme patterns")
            
    except Exception as e:
        print(f"❌ Error testing GenomicQuery: {e}")
    
    # Test the ContextRetrieverV2 signature
    print("\n🧪 Testing ContextRetrieverV2 signature:")
    retriever = dspy.ChainOfThought(ContextRetrieverV2)
    
    try:
        result = retriever(
            db_schema=NEO4J_SCHEMA[:1000],
            question=test_question, 
            query_type="general"
        )
        print(f"Generated strategy: {result.search_strategy}")
        if hasattr(result, 'neo4j_query'):
            print(f"Generated query: {result.neo4j_query}")
            
            # Check if the query uses correct node labels
            if "CAZymeAnnotation" in result.neo4j_query and "CAZymeFamily" in result.neo4j_query:
                print("🎉 SUCCESS: ContextRetrieverV2 uses correct node labels!")
            elif "Cazymeannotation" in result.neo4j_query or "Cazymefamily" in result.neo4j_query:
                print("❌ FAILED: ContextRetrieverV2 still uses old incorrect node labels")
            else:
                print("⚠️ PARTIAL: ContextRetrieverV2 query doesn't contain CAZyme patterns")
        else:
            print("ℹ️ No neo4j_query field in result")
            
    except Exception as e:
        print(f"❌ Error testing ContextRetrieverV2: {e}")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("DSPy or dependencies not available for testing")
except Exception as e:
    print(f"❌ Unexpected error: {e}")

print("\n✅ DSPy signature test completed")