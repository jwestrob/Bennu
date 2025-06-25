#!/usr/bin/env python3
"""
Debug Sequence Database Lookup

Check if the sequence database actually contains the protein IDs we're looking for.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from build_kg.sequence_db import SequenceDatabase

def debug_sequence_database():
    """Debug sequence database contents and lookups."""
    print("🔍 Debugging Sequence Database")
    print("=" * 50)
    
    # Check if database exists
    db_path = Path("data/sequences.db")
    if not db_path.exists():
        print(f"❌ Database not found at: {db_path.absolute()}")
        
        # Look for it in other common locations
        alt_paths = [
            Path("sequences.db"),
            Path("../data/sequences.db"),
            Path("../../data/sequences.db"),
            Path("data/stage05_kg/sequences.db")
        ]
        
        for alt_path in alt_paths:
            if alt_path.exists():
                print(f"🔍 Found database at: {alt_path.absolute()}")
                db_path = alt_path
                break
        else:
            print("❌ No sequence database found in any common location")
            return
    
    print(f"✅ Database found at: {db_path.absolute()}")
    
    # Initialize database
    try:
        db = SequenceDatabase(db_path, read_only=True)
        print("✅ Database connection successful")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return
    
    # Get database statistics
    try:
        stats = db.get_statistics()
        print(f"\n📊 Database Statistics:")
        print(f"  Total sequences: {stats.get('total_sequences', 'Unknown')}")
        print(f"  Unique genomes: {stats.get('unique_genomes', 'Unknown')}")
        if 'sequences_by_genome' in stats:
            print(f"  Genomes: {list(stats['sequences_by_genome'].keys())[:5]}...")
    except Exception as e:
        print(f"❌ Failed to get database statistics: {e}")
    
    # Test specific protein IDs from our transport protein query
    test_protein_ids = [
        'RIFCSPHIGHO2_01_FULL_Acidovorax_64_960_rifcsphigho2_01_scaffold_3030_5',
        'RIFCSPLOWO2_01_FULL_OD1_41_220_rifcsplowo2_01_scaffold_2623_17'
    ]
    
    print(f"\n🧪 Testing specific protein ID lookups:")
    for protein_id in test_protein_ids:
        print(f"\nTesting: {protein_id}")
        
        # Check if protein exists
        try:
            exists = db.protein_exists(protein_id)
            print(f"  Exists: {exists}")
        except Exception as e:
            print(f"  Error checking existence: {e}")
        
        # Try to get sequence
        try:
            sequence = db.get_sequence(protein_id)
            if sequence:
                print(f"  Sequence length: {len(sequence)} aa")
                print(f"  First 50 chars: {sequence[:50]}...")
            else:
                print(f"  Sequence: None")
        except Exception as e:
            print(f"  Error getting sequence: {e}")
    
    # Try batch lookup
    print(f"\n🔄 Testing batch lookup:")
    try:
        sequences = db.get_sequences(test_protein_ids)
        print(f"  Batch result: {len(sequences)} sequences found")
        for protein_id, sequence in sequences.items():
            print(f"    {protein_id}: {len(sequence)} aa")
    except Exception as e:
        print(f"  Batch lookup error: {e}")
    
    # Sample some actual protein IDs from database to see what format they use
    print(f"\n📋 Sample protein IDs in database:")
    try:
        # Get a few sample sequences to see the ID format
        sample_sequences = db.get_sequences([])  # This should return empty dict, but let's see
        
        # Alternative: try to get some sequences by examining the database directly
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT protein_id FROM sequences LIMIT 10")
            sample_ids = [row[0] for row in cursor.fetchall()]
            
        for i, sample_id in enumerate(sample_ids):
            print(f"  {i+1}. {sample_id}")
            
    except Exception as e:
        print(f"  Error getting sample IDs: {e}")

if __name__ == "__main__":
    debug_sequence_database()