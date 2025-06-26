#!/usr/bin/env python3
"""
Test script for the new intelligent annotation discovery workflow
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from llm.annotation_tools import annotation_explorer, functional_classifier, annotation_selector

async def test_annotation_workflow():
    """Test the complete annotation discovery workflow"""
    
    print("🧬 Testing Intelligent Annotation Discovery Workflow")
    print("=" * 60)
    
    # Step 1: Annotation Space Exploration
    print("\n📊 Step 1: Exploring Annotation Space...")
    exploration_result = await annotation_explorer(
        annotation_types=["KEGG", "PFAM"],
        functional_category="transport",
        max_annotations=100  # Limit for testing
    )
    
    if exploration_result["success"]:
        print(f"✅ Found {exploration_result['total_annotations']} annotations")
        for ann_type, annotations in exploration_result["annotation_catalog"].items():
            print(f"  📋 {ann_type}: {len(annotations)} annotations")
            # Show a few examples
            for i, ann in enumerate(annotations[:3]):
                print(f"    {i+1}. {ann['id']}: {ann['description']}")
    else:
        print(f"❌ Exploration failed: {exploration_result['error']}")
        return
    
    # Step 2: Functional Classification  
    print("\n🧠 Step 2: Classifying by Transport Mechanism...")
    classification_result = await functional_classifier(
        annotation_catalog=exploration_result["annotation_catalog"],
        functional_category="transport",
        user_preferences="diverse transport proteins",
        exclude_categories=["energy_metabolism"]
    )
    
    if classification_result["success"]:
        print(f"✅ Classified {classification_result['total_classified']} annotations")
        for category, ann_list in classification_result["classification"].items():
            print(f"  📊 {category}: {len(ann_list)} annotations")
            if ann_list:
                print(f"    Examples: {ann_list[:3]}")
    else:
        print(f"❌ Classification failed: {classification_result['error']}")
        return
    
    # Step 3: Annotation Selection
    print("\n🎯 Step 3: Selecting Diverse Examples...")
    selection_result = await annotation_selector(
        classified_annotations=classification_result["classification"],
        functional_category="transport",
        user_preferences="diverse examples",
        selection_count=3,
        prioritize_diversity=True
    )
    
    if selection_result["success"]:
        print(f"✅ Selected {selection_result['selection_count']} transport proteins")
        print(f"  📋 Selected annotations: {selection_result['selected_annotations']}")
        print(f"  💭 Rationale: {selection_result['selection_rationale']}")
    else:
        print(f"❌ Selection failed: {selection_result['error']}")
        return
    
    print("\n🎉 Annotation workflow test completed successfully!")
    print("📝 Next: Integrate with sequence viewer to complete the pipeline")

if __name__ == "__main__":
    asyncio.run(test_annotation_workflow())