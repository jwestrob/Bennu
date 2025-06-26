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

async def transport_classifier(
    annotation_catalog: Dict[str, List[Dict]], 
    user_preferences: str = "",
    exclude_categories: List[str] = ["energy_metabolism", "respiratory_chain"]
) -> Dict[str, Any]:
    """
    Classify annotations by biological transport mechanism using LLM intelligence.
    
    Args:
        annotation_catalog: Complete annotation catalog from annotation_explorer
        user_preferences: User-specified preferences extracted from query
        exclude_categories: Categories to explicitly exclude
        
    Returns:
        Dict with annotations classified by biological function
    """
    logger.info(f"🧠 Classifying annotations by transport mechanism")
    logger.info(f"📋 User preferences: {user_preferences}")
    
    try:
        # Prepare annotation list for LLM analysis
        all_annotations = []
        for ann_type, annotations in annotation_catalog.items():
            for ann in annotations:
                all_annotations.append(f"{ann['id']}: {ann['description']} ({ann_type})")
        
        logger.info(f"📊 Processing {len(all_annotations)} annotations for classification")
        
        # For now, implement basic keyword-based classification
        # TODO: Replace with actual LLM-powered classification
        substrate_transport = []
        energy_metabolism = []
        ion_channels = []
        membrane_structural = []
        other = []
        
        for ann_line in all_annotations:
            ann_id = ann_line.split(':')[0]
            description = ann_line.lower()
            
            # Basic classification logic (to be replaced with LLM)
            if any(keyword in description for keyword in ['abc', 'permease', 'transporter', 'channel']) and \
               not any(keyword in description for keyword in ['atp synthase', 'h+-transporting atpase', 'cytochrome']):
                substrate_transport.append(ann_id)
            elif any(keyword in description for keyword in ['atp synthase', 'atpase', 'cytochrome', 'respiratory']):
                energy_metabolism.append(ann_id)
            elif any(keyword in description for keyword in ['channel', 'pore', 'ion']):
                ion_channels.append(ann_id)
            elif any(keyword in description for keyword in ['membrane', 'porin']):
                membrane_structural.append(ann_id)
            else:
                other.append(ann_id)
        
        classification_result = {
            "SUBSTRATE_TRANSPORT": substrate_transport,
            "ENERGY_METABOLISM": energy_metabolism,
            "ION_CHANNELS": ion_channels,
            "MEMBRANE_STRUCTURAL": membrane_structural,
            "OTHER": other
        }
        
        # Apply exclusions
        if exclude_categories:
            for category in exclude_categories:
                if category.upper() in classification_result:
                    excluded_count = len(classification_result[category.upper()])
                    classification_result[category.upper()] = []
                    logger.info(f"🚫 Excluded {excluded_count} annotations from {category}")
        
        total_classified = sum(len(ann_list) for ann_list in classification_result.values())
        
        logger.info(f"✅ Classification complete:")
        for category, ann_list in classification_result.items():
            logger.info(f"  📊 {category}: {len(ann_list)} annotations")
        
        return {
            "success": True,
            "classification": classification_result,
            "total_classified": total_classified,
            "user_preferences": user_preferences,
            "excluded_categories": exclude_categories,
            "reasoning": "Classified based on functional keywords and biological categories"
        }
        
    except Exception as e:
        logger.error(f"❌ Error in transport classification: {e}")
        return {
            "success": False,
            "error": str(e),
            "classification": {},
            "total_classified": 0
        }

async def transport_selector(
    classified_annotations: Dict[str, List[str]], 
    user_preferences: str = "",
    selection_count: int = 3,
    prioritize_diversity: bool = True
) -> Dict[str, Any]:
    """
    Select diverse, high-quality transport protein examples from classified annotations.
    
    Args:
        classified_annotations: Output from transport_classifier
        user_preferences: User preferences for selection
        selection_count: Number of examples to select
        prioritize_diversity: Whether to maximize diversity across mechanisms
        
    Returns:
        Dict with selected annotation IDs and selection rationale
    """
    logger.info(f"🎯 Selecting {selection_count} diverse transport examples")
    logger.info(f"📋 User preferences: {user_preferences}")
    
    try:
        # Prioritize SUBSTRATE_TRANSPORT category
        primary_candidates = classified_annotations.get("SUBSTRATE_TRANSPORT", [])
        secondary_candidates = classified_annotations.get("ION_CHANNELS", [])
        
        selected_annotations = []
        selection_rationale = []
        
        # Simple selection logic (to be enhanced)
        if primary_candidates:
            # Take up to selection_count from primary candidates
            selected_count = min(selection_count, len(primary_candidates))
            selected_annotations.extend(primary_candidates[:selected_count])
            selection_rationale.append(f"Selected {selected_count} substrate transporters as primary examples")
            
            # Fill remaining slots with ion channels if available
            remaining_slots = selection_count - selected_count
            if remaining_slots > 0 and secondary_candidates:
                additional_count = min(remaining_slots, len(secondary_candidates))
                selected_annotations.extend(secondary_candidates[:additional_count])
                selection_rationale.append(f"Added {additional_count} ion channels for mechanism diversity")
        else:
            # Fallback to any available transport-related annotations
            all_candidates = []
            for category, ann_list in classified_annotations.items():
                if category not in ["ENERGY_METABOLISM", "OTHER"]:
                    all_candidates.extend(ann_list)
            
            selected_count = min(selection_count, len(all_candidates))
            selected_annotations = all_candidates[:selected_count]
            selection_rationale.append(f"Selected {selected_count} available transport-related annotations")
        
        logger.info(f"✅ Selection complete: {len(selected_annotations)} annotations selected")
        
        return {
            "success": True,
            "selected_annotations": selected_annotations,
            "selection_count": len(selected_annotations),
            "selection_rationale": selection_rationale,
            "user_preferences": user_preferences,
            "diversity_prioritized": prioritize_diversity
        }
        
    except Exception as e:
        logger.error(f"❌ Error in transport selection: {e}")
        return {
            "success": False,
            "error": str(e),
            "selected_annotations": [],
            "selection_count": 0
        }