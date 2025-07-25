#!/usr/bin/env python3
"""
Debug script to specifically test why whole_genome_reader results aren't expanding.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm.rag_system.memory.tool_result_cache import ToolResultCache
from llm.rag_system.memory.progressive_synthesizer import ProgressiveSynthesizer
from llm.rag_system.memory.note_keeper import NoteKeeper

def debug_wgr_expansion():
    """Debug why whole_genome_reader results aren't being expanded."""
    
    session_dir = "/Users/jacob/Documents/Sandbox/microbial_claude_matter/data/session_notes/27eadb8d-08d2-4dc1-a0bb-188fc2270900"
    
    print(f"🔍 Debugging WGR expansion for session: {session_dir}")
    
    # Initialize components
    note_keeper = NoteKeeper(session_dir)
    synthesizer = ProgressiveSynthesizer(note_keeper)
    cache = ToolResultCache(session_dir)
    
    # Get a specific note with WGR reference
    task_notes = note_keeper.get_all_task_notes()
    
    wgr_note = None
    for note in task_notes:
        if hasattr(note, 'quantitative_data') and note.quantitative_data:
            if 'tool_result_ref' in note.quantitative_data and note.quantitative_data['tool_result_ref'].startswith('wgr_'):
                wgr_note = note
                break
    
    if not wgr_note:
        print("❌ No WGR note found with tool_result_ref")
        return
    
    ref_id = wgr_note.quantitative_data['tool_result_ref']
    print(f"🔗 Testing WGR reference: {ref_id}")
    
    # Test direct cache retrieval
    print("🔄 Testing direct cache retrieval...")
    cached_result = cache.retrieve_tool_result(ref_id)
    
    if cached_result is None:
        print(f"❌ Failed to retrieve {ref_id} from cache")
        return
    
    print(f"✅ Cache retrieval successful, keys: {list(cached_result.keys())}")
    
    # Test synthesizer expansion logic
    print("🔄 Testing synthesizer expansion logic...")
    
    # Manually test the expansion process
    quantitative_data = dict(wgr_note.quantitative_data)
    
    if 'tool_result_ref' in quantitative_data and synthesizer.tool_cache:
        result_id = quantitative_data['tool_result_ref']
        print(f"🔗 Expanding tool result reference: {result_id}")
        
        # Load the referenced tool result
        tool_result = synthesizer.tool_cache.retrieve_tool_result(result_id)
        
        if tool_result:
            # Add expanded tool result to quantitative data
            quantitative_data['expanded_tool_result'] = tool_result
            print(f"✅ Successfully expanded tool result for {wgr_note.task_id}")
            print(f"🧬 Expanded result has {len(tool_result.get('interesting_loci', []))} loci")
        else:
            print(f"❌ Failed to load tool result reference: {result_id}")
    else:
        print(f"❌ Missing tool_result_ref or tool_cache")
        print(f"    - has tool_result_ref: {'tool_result_ref' in quantitative_data}")
        print(f"    - has tool_cache: {synthesizer.tool_cache is not None}")

if __name__ == "__main__":
    debug_wgr_expansion()