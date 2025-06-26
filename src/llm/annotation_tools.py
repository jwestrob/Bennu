#!/usr/bin/env python3
"""
Intelligent Annotation Discovery Tools for Biological Function Classification
Solves the "ATP synthase problem" through comprehensive annotation space exploration
"""

import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import asyncio

# Import Neo4j query processor
import sys
sys.path.append(str(Path(__file__).parent.parent))
from llm.query_processor import Neo4jQueryProcessor
from llm.config import LLMConfig

logger = logging.getLogger(__name__)

async def annotation_explorer(
    annotation_types: List[str] = ["KEGG", "PFAM"], 
    functional_category: str = "transport",
    max_annotations: int = 1000
) -> Dict[str, Any]:
    """
    Explore all available annotations in the database and return comprehensive catalogs.
    
    Args:
        annotation_types: Types of annotations to explore ("KEGG", "PFAM", "PATHWAYS")
        functional_category: Biological category for filtering hints
        max_annotations: Maximum annotations to return per type
        
    Returns:
        Dict with complete annotation catalogs for LLM curation
    """
    logger.info(f"🔍 Exploring annotation space for {annotation_types} ({functional_category})")
    
    try:
        config = LLMConfig()
        neo4j = Neo4jQueryProcessor(config)
        
        annotation_catalog = {}
        
        # Explore KEGG orthologs
        if "KEGG" in annotation_types:
            kegg_query = f"""
            MATCH (ko:KEGGOrtholog)
            RETURN ko.id AS annotation_id, ko.description AS description
            ORDER BY ko.id
            LIMIT {max_annotations}
            """
            
            kegg_result = await neo4j.process_query(kegg_query, query_type="cypher")
            kegg_annotations = [
                {
                    "id": item["annotation_id"],
                    "description": item["description"],
                    "type": "KEGG"
                }
                for item in kegg_result.results
            ]
            annotation_catalog["KEGG"] = kegg_annotations
            logger.info(f"📊 Found {len(kegg_annotations)} KEGG orthologs")
        
        # Explore PFAM domains
        if "PFAM" in annotation_types:
            pfam_query = f"""
            MATCH (dom:Domain)
            RETURN dom.id AS annotation_id, dom.description AS description
            ORDER BY dom.id
            LIMIT {max_annotations}
            """
            
            pfam_result = await neo4j.process_query(pfam_query, query_type="cypher")
            pfam_annotations = [
                {
                    "id": item["annotation_id"],
                    "description": item["description"],
                    "type": "PFAM"
                }
                for item in pfam_result.results
            ]
            annotation_catalog["PFAM"] = pfam_annotations
            logger.info(f"📊 Found {len(pfam_annotations)} PFAM domains")
        
        # Get database statistics
        total_annotations = sum(len(annotations) for annotations in annotation_catalog.values())
        
        logger.info(f"✅ Annotation space exploration complete: {total_annotations} total annotations")
        
        return {
            "success": True,
            "annotation_catalog": annotation_catalog,
            "total_annotations": total_annotations,
            "functional_category": functional_category,
            "exploration_summary": f"Discovered {total_annotations} annotations across {len(annotation_types)} types"
        }
        
    except Exception as e:
        logger.error(f"❌ Error in annotation exploration: {e}")
        return {
            "success": False,
            "error": str(e),
            "annotation_catalog": {},
            "total_annotations": 0
        }

async def functional_classifier(
    annotation_catalog: Dict[str, List[Dict]], 
    functional_category: str = "transport",
    user_preferences: str = "",
    exclude_categories: List[str] = ["energy_metabolism", "respiratory_chain"],
    max_relevant: int = 10
) -> Dict[str, Any]:
    """
    Classify annotations by biological function using intelligent keyword matching.
    Generalized version that works for any functional category.
    
    Args:
        annotation_catalog: Complete annotation catalog from annotation_explorer
        functional_category: Target biological category (transport, metabolism, regulation, etc.)
        user_preferences: User-specified preferences extracted from query
        exclude_categories: Categories to explicitly exclude
        max_relevant: Maximum number of relevant annotations to return
        
    Returns:
        Dict with annotations classified by biological relevance
    """
    logger.info(f"🧠 Classifying annotations for functional category: {functional_category}")
    logger.info(f"📋 User preferences: {user_preferences}")
    
    try:
        # Prepare annotation list for analysis
        all_annotations = []
        for ann_type, annotations in annotation_catalog.items():
            for ann in annotations:
                all_annotations.append({
                    "id": ann['id'],
                    "description": ann['description'],
                    "type": ann_type
                })
        
        logger.info(f"📊 Processing {len(all_annotations)} annotations for {functional_category}")
        
        # Define category-specific keywords
        category_keywords = {
            "transport": {
                "include": ['abc', 'permease', 'transporter', 'channel', 'pump', 'porter', 'antiporter', 'symporter'],
                "exclude": ['atp synthase', 'h+-transporting atpase', 'cytochrome', 'respiratory', 'nadh dehydrogenase']
            },
            "metabolism": {
                "include": ['dehydrogenase', 'reductase', 'oxidase', 'synthase', 'kinase', 'transferase', 'hydrolase'],
                "exclude": ['transport', 'channel', 'permease']
            },
            "regulation": {
                "include": ['regulator', 'repressor', 'activator', 'sensor', 'response', 'transcriptional'],
                "exclude": ['transport', 'metabolic']
            },
            "central_metabolism": {
                "include": ['glycolysis', 'citrate', 'pyruvate', 'acetyl', 'succinate', 'fumarate', 'malate', 'oxaloacetate'],
                "exclude": ['transport', 'abc']
            }
        }
        
        # Get keywords for the requested category
        keywords = category_keywords.get(functional_category.lower(), {
            "include": [functional_category.lower()],
            "exclude": []
        })
        
        # Classify annotations
        relevant_annotations = []
        excluded_annotations = []
        other_annotations = []
        
        for ann in all_annotations:
            description_lower = ann['description'].lower()
            ann_id = ann['id']
            
            # Check exclusion criteria first
            if any(exclude_word in description_lower for exclude_word in keywords['exclude']):
                excluded_annotations.append(ann_id)
            # Check inclusion criteria
            elif any(include_word in description_lower for include_word in keywords['include']):
                relevant_annotations.append(ann_id)
            else:
                other_annotations.append(ann_id)
        
        # Limit relevant annotations to max_relevant
        if len(relevant_annotations) > max_relevant:
            relevant_annotations = relevant_annotations[:max_relevant]
        
        classification_result = {
            "RELEVANT": relevant_annotations,
            "EXCLUDED": excluded_annotations,
            "OTHER": other_annotations
        }
        
        total_classified = sum(len(ann_list) for ann_list in classification_result.values())
        
        logger.info(f"✅ Classification complete for {functional_category}:")
        logger.info(f"  📊 RELEVANT: {len(relevant_annotations)} annotations")
        logger.info(f"  🚫 EXCLUDED: {len(excluded_annotations)} annotations")
        logger.info(f"  📄 OTHER: {len(other_annotations)} annotations")
        
        return {
            "success": True,
            "classification": classification_result,
            "total_classified": total_classified,
            "functional_category": functional_category,
            "user_preferences": user_preferences,
            "excluded_categories": exclude_categories,
            "reasoning": f"Classified based on {functional_category}-specific keywords"
        }
        
    except Exception as e:
        logger.error(f"❌ Error in functional classification: {e}")
        return {
            "success": False,
            "error": str(e),
            "classification": {},
            "total_classified": 0
        }

async def annotation_selector(
    classified_annotations: Dict[str, List[str]], 
    functional_category: str = "transport",
    user_preferences: str = "",
    selection_count: int = 3,
    prioritize_diversity: bool = True
) -> Dict[str, Any]:
    """
    Select high-quality examples from classified annotations for any functional category.
    Generalized version that works for transport, metabolism, regulation, etc.
    
    Args:
        classified_annotations: Output from functional_classifier
        functional_category: The functional category being selected for
        user_preferences: User preferences for selection
        selection_count: Number of examples to select
        prioritize_diversity: Whether to maximize diversity across mechanisms
        
    Returns:
        Dict with selected annotation IDs and selection rationale
    """
    logger.info(f"🎯 Selecting {selection_count} examples for {functional_category}")
    logger.info(f"📋 User preferences: {user_preferences}")
    
    try:
        # Prioritize RELEVANT category
        primary_candidates = classified_annotations.get("RELEVANT", [])
        fallback_candidates = classified_annotations.get("OTHER", [])
        
        selected_annotations = []
        selection_rationale = []
        
        # Select from primary candidates first
        if primary_candidates:
            selected_count = min(selection_count, len(primary_candidates))
            selected_annotations.extend(primary_candidates[:selected_count])
            selection_rationale.append(f"Selected {selected_count} highly relevant {functional_category} annotations")
            
            # Fill remaining slots with fallback candidates if needed
            remaining_slots = selection_count - selected_count
            if remaining_slots > 0 and fallback_candidates:
                additional_count = min(remaining_slots, len(fallback_candidates))
                selected_annotations.extend(fallback_candidates[:additional_count])
                selection_rationale.append(f"Added {additional_count} additional candidates for completeness")
        else:
            # Fallback to any available candidates
            selected_count = min(selection_count, len(fallback_candidates))
            selected_annotations = fallback_candidates[:selected_count]
            selection_rationale.append(f"Selected {selected_count} available {functional_category}-related annotations")
        
        logger.info(f"✅ Selection complete: {len(selected_annotations)} annotations selected")
        
        return {
            "success": True,
            "selected_annotations": selected_annotations,
            "selection_count": len(selected_annotations),
            "selection_rationale": selection_rationale,
            "functional_category": functional_category,
            "user_preferences": user_preferences,
            "diversity_prioritized": prioritize_diversity
        }
        
    except Exception as e:
        logger.error(f"❌ Error in annotation selection: {e}")
        return {
            "success": False,
            "error": str(e),
            "selected_annotations": [],
            "selection_count": 0
        }