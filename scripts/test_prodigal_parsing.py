#!/usr/bin/env python3
"""
Test the new prodigal header parsing functionality.
"""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.build_kg.rdf_builder import parse_prodigal_header

def test_prodigal_parsing():
    """Test prodigal header parsing with real data."""
    
    # Test cases from actual prodigal output
    test_headers = [
        ">RIFCSPHIGHO2_01_FULL_Acidovorax_64_960_rifcsphigho2_01_scaffold_15917_1 # 76 # 171 # -1 # ID=1_1;partial=00;start_type=ATG;rbs_motif=AGGAG;rbs_spacer=5-10bp;gc_cont=0.573",
        ">RIFCSPHIGHO2_01_FULL_Acidovorax_64_960_rifcsphigho2_01_scaffold_15917_2 # 261 # 569 # 1 # ID=1_2;partial=00;start_type=ATG;rbs_motif=AGGAG;rbs_spacer=5-10bp;gc_cont=0.621",
        ">simple_protein_id",  # Test fallback for simple headers
    ]
    
    print("🧪 Testing Prodigal Header Parsing")
    print("=" * 60)
    
    for i, header in enumerate(test_headers, 1):
        print(f"\nTest {i}: {header[:80]}{'...' if len(header) > 80 else ''}")
        
        try:
            result = parse_prodigal_header(header)
            print("✅ Parsed successfully:")
            for key, value in result.items():
                print(f"  {key}: {value}")
            
            # Validate expected fields
            if 'start' in result and 'end' in result:
                expected_nt_length = abs(result['end'] - result['start']) + 1
                expected_aa_length = expected_nt_length // 3
                
                print(f"✅ Validation:")
                print(f"  Expected NT length: {expected_nt_length} == Actual: {result.get('length_nt', 'missing')}")
                print(f"  Expected AA length: {expected_aa_length} == Actual: {result.get('length_aa', 'missing')}")
                
        except Exception as e:
            print(f"❌ Parsing failed: {e}")
        
        print("-" * 40)

if __name__ == "__main__":
    test_prodigal_parsing()