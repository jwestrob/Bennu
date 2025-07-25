#!/usr/bin/env python3
"""
Debug script to examine the exact structure of cached tool results.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm.rag_system.memory.tool_result_cache import ToolResultCache

def examine_cache_structure():
    """Examine the detailed structure of cached results."""
    
    session_dir = "/Users/jacob/Documents/Sandbox/microbial_claude_matter/data/session_notes/27eadb8d-08d2-4dc1-a0bb-188fc2270900"
    
    print(f"🔍 Examining cached tool result structure")
    
    # Initialize cache
    cache = ToolResultCache(session_dir)
    
    # Test retrieval of a whole_genome_reader result
    result_id = "wgr_a8ac9463_182439"
    result = cache.retrieve_tool_result(result_id)
    
    if result is None:
        print("❌ Failed to retrieve result")
        return
    
    print("✅ Retrieved result successfully!")
    print(f"📋 Top-level keys: {list(result.keys())}")
    
    # Check interesting_loci
    if 'interesting_loci' in result:
        loci = result['interesting_loci']
        print(f"🧬 interesting_loci type: {type(loci)}")
        print(f"🧬 interesting_loci count: {len(loci) if isinstance(loci, list) else 'not a list'}")
        
        if isinstance(loci, list) and len(loci) > 0:
            first_locus = loci[0]
            print(f"📍 First locus type: {type(first_locus)}")
            print(f"📍 First locus preview: {str(first_locus)[:200]}...")
            
    # Check analysis_summary
    if 'analysis_summary' in result:
        summary = result['analysis_summary']
        print(f"📊 Analysis summary keys: {list(summary.keys()) if isinstance(summary, dict) else 'not a dict'}")
        if isinstance(summary, dict) and 'interesting_loci_count' in summary:
            print(f"📊 Reported loci count: {summary['interesting_loci_count']}")

if __name__ == "__main__":
    examine_cache_structure()