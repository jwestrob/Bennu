#!/usr/bin/env python3
"""
Debug script to test tool result cache retrieval functionality.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm.rag_system.memory.tool_result_cache import ToolResultCache

def test_cache_retrieval():
    """Test if the tool result cache can retrieve detailed data."""
    
    session_dir = "/Users/jacob/Documents/Sandbox/microbial_claude_matter/data/session_notes/27eadb8d-08d2-4dc1-a0bb-188fc2270900"
    
    print(f"🔍 Testing tool result cache with session: {session_dir}")
    
    # Initialize cache
    cache = ToolResultCache(session_dir)
    
    # Test retrieval of a whole_genome_reader result
    result_id = "wgr_a8ac9463_182439"
    print(f"📖 Attempting to retrieve: {result_id}")
    
    result = cache.retrieve_tool_result(result_id)
    
    if result is None:
        print("❌ Failed to retrieve result")
        return False
    
    print("✅ Successfully retrieved result!")
    
    # Check if we have the detailed loci data
    if 'tool_result' in result and 'interesting_loci' in result['tool_result']:
        loci_count = len(result['tool_result']['interesting_loci'])
        print(f"🧬 Found {loci_count} interesting loci in cached result")
        
        # Show first locus details
        if loci_count > 0:
            first_locus = result['tool_result']['interesting_loci'][0]
            print(f"📍 First locus: {first_locus[:100]}...")
            
        return True
    else:
        print("⚠️ Result exists but missing interesting_loci data")
        print(f"🔍 Result keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
        return False

if __name__ == "__main__":
    test_cache_retrieval()