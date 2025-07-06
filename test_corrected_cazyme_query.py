#!/usr/bin/env python3
"""
Test the corrected CAZyme queries to verify they return all 1,845 CAZymes.
"""

import sys
import subprocess

def test_corrected_queries():
    """Test the corrected CAZyme query patterns."""
    
    print("🧪 Testing Corrected CAZyme Queries\n")
    
    # Simple counting query first
    simple_query = """
    How many CAZyme annotations are in the database?
    """
    
    print("=== Test 1: Simple CAZyme count ===")
    print(f"Query: {simple_query}")
    
    try:
        # Try to run the query using the ask command
        result = subprocess.run([
            'python', '-m', 'src.cli', 'ask', simple_query
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Query executed successfully!")
            print("Output:")
            print(result.stdout)
            
            # Check if we see 1845 or similar number
            if "1845" in result.stdout or "1,845" in result.stdout:
                print("🎉 SUCCESS: Found the expected 1,845 CAZymes!")
            elif any(str(i) in result.stdout for i in range(1800, 1900)):
                print("✅ GOOD: Found a large number of CAZymes (likely correct)")
            else:
                print("⚠️ WARNING: Didn't see the expected count")
        else:
            print("❌ Query failed")
            print("Error:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("❌ Query timed out")
    except Exception as e:
        print(f"❌ Error running query: {e}")
    
    print()
    
    # More specific query
    specific_query = """
    Tell me about glycoside hydrolases in the dataset
    """
    
    print("=== Test 2: Specific CAZyme family query ===")
    print(f"Query: {specific_query}")
    
    try:
        result = subprocess.run([
            'python', '-m', 'src.cli', 'ask', specific_query
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Query executed successfully!")
            print("Output:")
            print(result.stdout[:1000] + "..." if len(result.stdout) > 1000 else result.stdout)
            
            # Check if we see reasonable GH family information
            if "GH" in result.stdout and any(term in result.stdout.lower() for term in ["glycoside", "hydrolase", "substrate"]):
                print("🎉 SUCCESS: Found detailed GH family information!")
            else:
                print("⚠️ WARNING: Didn't see expected GH family details")
        else:
            print("❌ Query failed")
            print("Error:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("❌ Query timed out")
    except Exception as e:
        print(f"❌ Error running query: {e}")

if __name__ == "__main__":
    test_corrected_queries()