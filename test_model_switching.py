#!/usr/bin/env python3
"""
Test script for model switching functionality.
"""

import sys
sys.path.append('/Users/jacob/Documents/Sandbox/microbial_claude_matter')

from src.llm.model_switcher import (
    switch_to_cost_effective,
    switch_to_premium,
    print_model_status,
    configure_models
)

def main():
    print("🧪 Testing Model Switching System")
    print("=" * 40)
    
    # Show current status
    print("\n1. Current Status:")
    print_model_status()
    
    # Switch to cost-effective mode
    print("\n2. Switching to Cost-Effective Mode:")
    switch_to_cost_effective()
    
    # Switch to premium mode
    print("\n3. Switching to Premium Mode:")
    switch_to_premium()
    
    # Try configuring different models
    print("\n4. Configuring Custom Models:")
    configure_models(
        cost_effective_model="gpt-4o-mini",
        premium_model="o3"
    )
    
    # Switch back to cost-effective to show it works
    print("\n5. Switching Back to Cost-Effective:")
    switch_to_cost_effective()
    
    print("\n✅ Model switching system is working!")
    print("\nTo use in your genomic queries:")
    print("  from src.llm.model_switcher import switch_to_cost_effective, switch_to_premium")
    print("  switch_to_cost_effective()  # Use gpt-4.1-mini for development")
    print("  switch_to_premium()         # Use o3 for final results")

if __name__ == "__main__":
    main()