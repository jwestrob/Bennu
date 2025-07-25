#!/usr/bin/env python3
"""
Debug script to test if synthesis expansion is working properly.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm.rag_system.memory.tool_result_cache import ToolResultCache
from llm.rag_system.memory.progressive_synthesizer import ProgressiveSynthesizer
from llm.rag_system.memory.note_keeper import NoteKeeper

def test_synthesis_expansion():
    """Test if ProgressiveSynthesizer is properly expanding tool result references."""
    
    session_dir = "/Users/jacob/Documents/Sandbox/microbial_claude_matter/data/session_notes/27eadb8d-08d2-4dc1-a0bb-188fc2270900"
    
    print(f"🔍 Testing synthesis expansion for session: {session_dir}")
    
    # Initialize components
    note_keeper = NoteKeeper(session_dir)
    synthesizer = ProgressiveSynthesizer(note_keeper)
    
    # Load task notes from the session
    task_notes = note_keeper.get_all_task_notes()
    print(f"📝 Loaded {len(task_notes)} task notes")
    
    # Check if any notes have tool_result_ref
    notes_with_refs = []
    for note in task_notes:
        if hasattr(note, 'quantitative_data') and note.quantitative_data:
            if 'tool_result_ref' in note.quantitative_data:
                notes_with_refs.append(note)
                print(f"🔗 Found tool_result_ref in {note.task_id}: {note.quantitative_data['tool_result_ref']}")
    
    print(f"📊 {len(notes_with_refs)} notes have tool result references")
    
    # Test the _prepare_unified_data method directly
    print("🔄 Testing _prepare_unified_data expansion...")
    unified_data = synthesizer._prepare_unified_data(None, task_notes)
    
    # Check if any unified data has expanded_tool_result
    expanded_count = 0
    for item in unified_data:
        if 'quantitative_data' in item and 'expanded_tool_result' in item['quantitative_data']:
            expanded_count += 1
            expanded_result = item['quantitative_data']['expanded_tool_result']
            print(f"✅ Found expanded tool result in {item['task_id']}")
            
            # Check if it has the detailed loci data
            if 'interesting_loci' in expanded_result:
                loci_count = len(expanded_result['interesting_loci'])
                print(f"🧬 Expanded result contains {loci_count} interesting loci")
    
    print(f"📊 {expanded_count} items have expanded tool results")
    
    if expanded_count == 0:
        print("❌ No tool results were expanded - this is the problem!")
    else:
        print("✅ Tool result expansion is working")

if __name__ == "__main__":
    test_synthesis_expansion()