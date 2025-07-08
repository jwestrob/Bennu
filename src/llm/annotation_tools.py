#!/usr/bin/env python3
"""
Intelligent Annotation Discovery Tools for Biological Function Classification
Enhanced with dynamic KEGG pathway-based protein discovery
"""

import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import asyncio
from datetime import datetime

# Import Neo4j query processor and pathway tools
import sys
sys.path.append(str(Path(__file__).parent.parent))
from llm.query_processor import Neo4jQueryProcessor
from llm.config import LLMConfig
from llm.pathway_tools import pathway_based_protein_discovery, pathway_classifier

logger = logging.getLogger(__name__)

async def enrich_proteins_with_context(
    protein_ids: List[str],
    max_proteins: int = 50
) -> Dict[str, Any]:
    """
    Enrich a list of protein IDs with comprehensive genomic context.
    
    This function takes basic protein IDs and adds:
    - Gene coordinates and strand information
    - PFAM domain annotations with descriptions
    - KEGG functional annotations
    - Genomic neighborhood analysis (5kb window)
    - Domain scores and positions
    
    Args:
        protein_ids: List of protein IDs to enrich
        max_proteins: Maximum number of proteins to process
        
    Returns:
        Dict with enriched protein data including full genomic context
    """
    logger.info(f"🧬 Enriching {len(protein_ids)} proteins with comprehensive genomic context")
    
    try:
        config = LLMConfig()
        neo4j = Neo4jQueryProcessor(config)
        
        enriched_proteins = []
        
        # Process proteins in batches to avoid overwhelming the system
        batch_size = 10
        for i in range(0, min(len(protein_ids), max_proteins), batch_size):
            batch = protein_ids[i:i+batch_size]
            logger.info(f"📊 Processing batch {i//batch_size + 1}: {len(batch)} proteins")
            
            for protein_id in batch:
                try:
                    # Use the comprehensive protein_info query
                    result = await neo4j.process_query(
                        protein_id,
                        query_type="protein_info"
                    )
                    
                    if result.results:
                        protein_data = result.results[0]
                        
                        # Add enrichment metadata
                        protein_data['enrichment_source'] = 'comprehensive_protein_info'
                        protein_data['enrichment_timestamp'] = str(datetime.now())
                        
                        # Calculate neighbor summary statistics
                        neighbors = protein_data.get('detailed_neighbors', [])
                        if neighbors:
                            protein_data['neighbor_count'] = len([n for n in neighbors if n.get('protein_id')])
                            protein_data['functional_neighbors'] = len([n for n in neighbors if n.get('kegg_desc')])
                            protein_data['domain_neighbors'] = len([n for n in neighbors if n.get('pfam_ids')])
                        
                        enriched_proteins.append(protein_data)
                        logger.debug(f"✅ Enriched {protein_id} with {len(neighbors)} neighbors")
                    else:
                        logger.warning(f"⚠️ No enrichment data found for {protein_id}")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to enrich {protein_id}: {e}")
                    continue
        
        logger.info(f"🎯 Successfully enriched {len(enriched_proteins)} proteins with comprehensive context")
        
        return {
            "success": True,
            "enriched_proteins": enriched_proteins,
            "enrichment_count": len(enriched_proteins),
            "original_count": len(protein_ids),
            "enrichment_rate": len(enriched_proteins) / len(protein_ids) if protein_ids else 0,
            "context_types": [
                "gene_coordinates",
                "pfam_domains", 
                "kegg_functions",
                "genomic_neighborhood",
                "domain_scores",
                "neighbor_analysis"
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Protein enrichment failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "enriched_proteins": [],
            "enrichment_count": 0
        }

async def comprehensive_protein_discovery(
    functional_category: str = "central_metabolism",
    max_proteins: int = 100,
    include_enrichment: bool = True,
    batch_size: int = 25
) -> Dict[str, Any]:
    """
    Comprehensive protein discovery with automatic enrichment.
    
    This function combines:
    1. Functional annotation discovery (KEGG + PFAM)
    2. Protein finding via Neo4j queries
    3. Automatic enrichment with genomic context
    
    Args:
        functional_category: Biological category to search for
        max_proteins: Maximum proteins to discover
        include_enrichment: Whether to add comprehensive genomic context
        batch_size: Batch size for processing
        
    Returns:
        Dict with discovered proteins and rich genomic context
    """
    logger.info(f"🔬 Starting comprehensive protein discovery for {functional_category}")
    
    try:
        config = LLMConfig()
        neo4j = Neo4jQueryProcessor(config)
        
        # Step 1: Get annotation catalog
        logger.info("📚 Step 1: Exploring annotation catalog")
        annotation_result = await annotation_explorer(
            annotation_types=["KEGG", "PFAM"],
            functional_category=functional_category,
            max_annotations=1000
        )
        
        if not annotation_result.get("success"):
            return {
                "success": False,
                "error": "Failed to get annotation catalog",
                "discovered_proteins": [],
                "discovery_count": 0
            }
        
        # Step 2: Classify relevant annotations
        logger.info("🎯 Step 2: Classifying functional annotations")
        classification_result = await functional_classifier(
            annotation_catalog=annotation_result["annotation_catalog"],
            functional_category=functional_category,
            max_relevant=50  # Increased for comprehensive discovery
        )
        
        if not classification_result.get("success"):
            return {
                "success": False,
                "error": "Failed to classify annotations",
                "discovered_proteins": [],
                "discovery_count": 0
            }
        
        # Step 3: Find proteins using classified annotations
        logger.info("🔍 Step 3: Discovering proteins from classified annotations")
        relevant_kos = classification_result["classification"].get("RELEVANT", [])
        
        if not relevant_kos:
            return {
                "success": False,
                "error": "No relevant KEGG orthologs found",
                "discovered_proteins": [],
                "discovery_count": 0
            }
        
        discovered_proteins = []
        
        # Process KO IDs in batches to avoid query length limits
        for i in range(0, len(relevant_kos), batch_size):
            batch_kos = relevant_kos[i:i+batch_size]
            ko_list = "', '".join(batch_kos)
            
            logger.info(f"📊 Processing KO batch {i//batch_size + 1}: {len(batch_kos)} orthologs")
            
            # Query for proteins with these KEGG functions
            query = f"""
            MATCH (ko:KEGGOrtholog)
            WHERE ko.id IN ['{ko_list}']
            MATCH (p:Protein)-[:HASFUNCTION]->(ko)
            OPTIONAL MATCH (p)-[:ENCODEDBY]->(g:Gene)-[:BELONGSTOGENOME]->(genome:Genome)
            RETURN DISTINCT p.id AS protein_id, ko.id AS kegg_id, ko.description AS kegg_description,
                   g.id AS gene_id, genome.id AS genome_id
            LIMIT {max_proteins}
            """
            
            result = await neo4j.process_query(query, query_type="cypher")
            batch_proteins = result.results
            
            discovered_proteins.extend(batch_proteins)
            logger.info(f"✅ Found {len(batch_proteins)} proteins in this batch")
            
            if len(discovered_proteins) >= max_proteins:
                break
        
        # Limit to max_proteins
        discovered_proteins = discovered_proteins[:max_proteins]
        protein_ids = [p["protein_id"] for p in discovered_proteins]
        
        logger.info(f"🎯 Discovered {len(discovered_proteins)} proteins for {functional_category}")
        
        # Step 4: Enrich with comprehensive context (if requested)
        enriched_proteins = discovered_proteins
        if include_enrichment and protein_ids:
            logger.info("🧬 Step 4: Enriching proteins with comprehensive genomic context")
            enrichment_result = await enrich_proteins_with_context(
                protein_ids=protein_ids,
                max_proteins=max_proteins
            )
            
            if enrichment_result.get("success"):
                enriched_proteins = enrichment_result["enriched_proteins"]
                logger.info(f"✅ Successfully enriched {len(enriched_proteins)} proteins")
            else:
                logger.warning(f"⚠️ Enrichment failed: {enrichment_result.get('error')}")
        
        return {
            "success": True,
            "discovered_proteins": enriched_proteins,
            "discovery_count": len(enriched_proteins),
            "functional_category": functional_category,
            "relevant_kos_count": len(relevant_kos),
            "enrichment_included": include_enrichment,
            "discovery_summary": {
                "total_proteins": len(discovered_proteins),
                "enriched_proteins": len(enriched_proteins) if include_enrichment else 0,
                "relevant_annotations": len(relevant_kos),
                "functional_category": functional_category
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Comprehensive protein discovery failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "discovered_proteins": [],
            "discovery_count": 0
        }

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
            "annotation_types": annotation_types,
            "functional_category": functional_category
        }
        
    except Exception as e:
        logger.error(f"❌ Error in annotation exploration: {e}")
        return {
            "success": False,
            "error": f"Annotation exploration failed: {str(e)}",
            "annotation_catalog": {},
            "total_annotations": 0
        }

# ===== OPERON ASSESSMENT TOOLS =====

import dspy

class OperonAssessment(dspy.Signature):
    """
    Assess whether a target protein occurs within a functionally coherent operon.
    
    Consider:
    - Functional relationships between neighboring genes
    - Genomic organization (strand, spacing, gene order)
    - Biological plausibility of co-regulation
    - Annotation confidence and domain architecture
    
    OPERON CRITERIA:
    - Same strand orientation (usually required)
    - Close proximity (<200bp typical, up to 500bp possible)
    - Functionally related processes (metabolic pathway, transport system, etc.)
    - Consistent annotation quality
    
    EXAMPLES OF OPERONS:
    - ABC transporter: permease + ATPase + binding protein
    - TCA cycle: succinate dehydrogenase subunits A,B,C,D
    - Amino acid biosynthesis: enzyme cascade for pathway
    - Iron uptake: receptor + permease + ATPase + binding protein
    
    EXAMPLES OF NON-OPERONS:
    - Mixed functions: transporter + ribosomal protein + DNA repair
    - Opposite strands with unrelated functions
    - Large gaps (>1000bp) between genes
    - Single genes with no functional neighbors
    
    ASSESSMENT GUIDELINES:
    - High confidence (0.8-1.0): Clear functional theme, same strand, <200bp spacing
    - Medium confidence (0.5-0.8): Related functions, some spacing/strand issues
    - Low confidence (0.2-0.5): Weak functional relationships, mixed organization
    - No operon (0.0-0.2): Unrelated functions, poor genomic organization
    """
    
    target_protein = dspy.InputField(desc="Target protein with genomic coordinates, function, and domains")
    genomic_neighborhood = dspy.InputField(desc="List of neighboring proteins with distances, strands, and functional annotations")
    target_function_category = dspy.InputField(desc="Functional category being searched for (e.g., central_metabolism)")
    
    is_operonic = dspy.OutputField(desc="Boolean: Does target protein occur in a functionally coherent operon?")
    confidence = dspy.OutputField(desc="Confidence score 0.0-1.0 for operon assessment")
    reasoning = dspy.OutputField(desc="Biological reasoning for operon assessment decision")
    operon_partners = dspy.OutputField(desc="List of protein IDs that appear to be co-operonic with target")
    functional_theme = dspy.OutputField(desc="Functional theme of the operon (e.g., 'TCA cycle enzymes', 'iron transport system')")

async def assess_operon_context(
    protein_data: Dict[str, Any], 
    genomic_neighborhood: List[Dict[str, Any]], 
    target_function: str = "central_metabolism"
) -> Dict[str, Any]:
    """
    Use LLM to assess whether target protein occurs in a functionally coherent operon
    
    Args:
        protein_data: Target protein with coordinates, function, domains
        genomic_neighborhood: List of neighboring proteins with distances, strands, functions
        target_function: Functional category we're searching for
    
    Returns:
        {
            "is_operonic": bool,
            "confidence": float,
            "reasoning": str,
            "operon_partners": List[str],
            "functional_theme": str
        }
    """
    logger.info(f"🧬 Assessing operon context for protein: {protein_data.get('id', 'unknown')}")
    
    try:
        # Initialize DSPy model for operon assessment
        config = LLMConfig()
        operon_assessor = dspy.Predict(OperonAssessment)
        
        # Format protein data for assessment
        protein_summary = f"""
        Protein ID: {protein_data.get('id', 'unknown')}
        Function: {protein_data.get('function', 'unknown')}
        KEGG: {protein_data.get('ko_description', 'none')}
        PFAM Domains: {protein_data.get('pfam_accessions', [])}
        Coordinates: {protein_data.get('start_coordinate', 'unknown')}-{protein_data.get('end_coordinate', 'unknown')}
        Strand: {protein_data.get('strand', 'unknown')}
        """
        
        # Format neighborhood data
        neighborhood_summary = []
        for neighbor in genomic_neighborhood[:10]:  # Limit to closest 10 neighbors
            neighbor_info = f"""
            - ID: {neighbor.get('id', 'unknown')}
            - Distance: {neighbor.get('distance', 'unknown')}bp {neighbor.get('direction', 'unknown')}
            - Strand: {neighbor.get('strand_info', 'unknown')}
            - Function: {neighbor.get('function', 'unknown')[:100]}
            - Operon Status: {neighbor.get('operon_status', 'unknown')}
            """
            neighborhood_summary.append(neighbor_info)
        
        neighborhood_text = "\n".join(neighborhood_summary)
        
        # Perform LLM assessment
        result = operon_assessor(
            target_protein=protein_summary,
            genomic_neighborhood=neighborhood_text,
            target_function_category=target_function
        )
        
        # Parse and validate results
        is_operonic = str(result.is_operonic).lower() in ['true', 'yes', '1']
        
        try:
            confidence = float(result.confidence)
            confidence = max(0.0, min(1.0, confidence))  # Clamp to 0-1 range
        except (ValueError, TypeError):
            confidence = 0.5  # Default confidence
        
        # Parse operon partners
        partners = []
        if hasattr(result, 'operon_partners') and result.operon_partners:
            partners_text = str(result.operon_partners)
            # Extract protein IDs from the response
            for neighbor in genomic_neighborhood:
                neighbor_id = neighbor.get('id', '')
                if neighbor_id and neighbor_id in partners_text:
                    partners.append(neighbor_id)
        
        assessment_result = {
            "is_operonic": is_operonic,
            "confidence": confidence,
            "reasoning": str(result.reasoning) if hasattr(result, 'reasoning') else "No reasoning provided",
            "operon_partners": partners,
            "functional_theme": str(result.functional_theme) if hasattr(result, 'functional_theme') else "Unknown theme",
            "target_protein_id": protein_data.get('id', 'unknown')
        }
        
        logger.info(f"✅ Operon assessment complete: {is_operonic} (confidence: {confidence:.2f})")
        return assessment_result
        
    except Exception as e:
        logger.error(f"❌ Operon assessment failed: {str(e)}")
        return {
            "is_operonic": False,
            "confidence": 0.0,
            "reasoning": f"Assessment failed: {str(e)}",
            "operon_partners": [],
            "functional_theme": "Assessment failed",
            "target_protein_id": protein_data.get('id', 'unknown')
        }

async def prospect_operonic_proteins(
    functional_category: str = "central_metabolism", 
    min_examples: int = 2,
    max_candidates: int = 50
) -> Dict[str, Any]:
    """
    Prospect for proteins in operons using LLM assessment with iterative search
    
    Args:
        functional_category: Target functional category
        min_examples: Minimum number of operonic proteins to find
        max_candidates: Maximum number of candidates to assess
    
    Returns:
        {
            "success": bool,
            "operonic_proteins": List[Dict],
            "search_summary": Dict,
            "total_assessed": int,
            "total_operonic": int
        }
    """
    logger.info(f"🔍 Prospecting for operonic proteins in category: {functional_category}")
    
    try:
        # Import sequence viewer for genomic neighborhoods
        from .rag_system import sequence_viewer
        
        # Initialize query processor
        config = LLMConfig()
        query_processor = Neo4jQueryProcessor(config)
        
        candidates_found = []
        total_assessed = 0
        
        # Define search terms for central metabolism
        if functional_category == "central_metabolism":
            search_terms = [
                "glycol", "pyruvate", "citrate", "succinate", "malate", "fumarate", 
                "acetyl", "oxaloacetate", "isocitrate", "alpha-ketoglutarate",
                "phosphoenolpyruvate", "glucose", "fructose", "lactate"
            ]
        else:
            # Generic search terms for other categories
            search_terms = [functional_category, "enzyme", "protein"]
        
        for search_term in search_terms:
            if len(candidates_found) >= min_examples:
                break
                
            logger.info(f"🔍 Searching with term: {search_term}")
            
            # Query for proteins with this functional term
            query = f"""
            MATCH (ko:KEGGOrtholog) 
            WHERE toLower(ko.description) CONTAINS '{search_term.lower()}'
            MATCH (p:Protein)-[:HASFUNCTION]->(ko)
            OPTIONAL MATCH (p)-[:ENCODEDBY]->(g:Gene)
            OPTIONAL MATCH (p)-[:HASDOMAIN]->(da:DomainAnnotation)-[:DOMAINFAMILY]->(dom:Domain)
            RETURN p.id AS protein_id, ko.id AS ko_id, ko.description AS ko_description,
                   g.startCoordinate AS start_coordinate, g.endCoordinate AS end_coordinate, g.strand,
                   collect(DISTINCT dom.id) AS pfam_accessions
            """
            
            result = await query_processor.process_query(query, query_type="cypher")
            proteins = result.results
            
            for protein in proteins:
                if total_assessed >= max_candidates:
                    break
                    
                if len(candidates_found) >= min_examples:
                    break
                
                total_assessed += 1
                
                # Get genomic neighborhood via sequence_viewer
                try:
                    neighborhood_result = await sequence_viewer(
                        protein_ids=[protein['protein_id']], 
                        analysis_context="operon assessment"
                    )
                    
                    # Extract neighborhood data from sequence_viewer result
                    neighborhood = []
                    if neighborhood_result and 'neighborhood_data' in neighborhood_result:
                        neighborhood = neighborhood_result['neighborhood_data']
                    
                    # LLM-powered operon assessment
                    assessment = await assess_operon_context(
                        protein_data=protein,
                        genomic_neighborhood=neighborhood,
                        target_function=functional_category
                    )
                    
                    if assessment["is_operonic"] and assessment["confidence"] > 0.6:
                        candidates_found.append({
                            "protein": protein,
                            "assessment": assessment,
                            "search_term": search_term,
                            "neighborhood_size": len(neighborhood)
                        })
                        
                        logger.info(f"✅ Found operonic protein: {protein['protein_id']} (confidence: {assessment['confidence']:.2f})")
                
                except Exception as e:
                    logger.warning(f"⚠️ Failed to assess protein {protein.get('protein_id', 'unknown')}: {str(e)}")
                    continue
        
        success = len(candidates_found) >= min_examples
        
        result = {
            "success": success,
            "operonic_proteins": candidates_found,
            "search_summary": {
                "functional_category": functional_category,
                "search_terms_used": search_terms[:len(candidates_found)+1],
                "min_examples_requested": min_examples,
                "examples_found": len(candidates_found)
            },
            "total_assessed": total_assessed,
            "total_operonic": len(candidates_found)
        }
        
        if success:
            logger.info(f"🎉 Operon prospecting successful: {len(candidates_found)} operonic proteins found")
        else:
            logger.warning(f"⚠️ Operon prospecting incomplete: only {len(candidates_found)}/{min_examples} examples found")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Operon prospecting failed: {str(e)}")
        return {
            "success": False,
            "operonic_proteins": [],
            "search_summary": {"error": str(e)},
            "total_assessed": 0,
            "total_operonic": 0
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
            # Handle None descriptions safely
            description = ann.get('description', '') or ''
            description_lower = description.lower()
            ann_id = ann['id']
            
            # Skip annotations with empty descriptions
            if not description_lower.strip():
                other_annotations.append(ann_id)
                continue
            
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

async def pathway_based_annotation_selector(
    query: str,
    functional_category: str = "metabolism",
    max_pathways: int = 3,
    max_proteins_per_pathway: int = 5,
    selection_criteria: str = "",
    **kwargs
) -> Dict[str, Any]:
    """
    Select protein annotations based on KEGG pathway analysis instead of hardcoded examples.
    This replaces the hardcoded transporter examples with dynamic pathway-based discovery.
    
    Args:
        query: User query to analyze for pathway relevance
        functional_category: Biological category for context
        max_pathways: Maximum number of relevant pathways to analyze
        max_proteins_per_pathway: Maximum proteins to return per pathway
        
    Returns:
        Dict with pathway-based protein annotations for similarity search
    """
    logger.info(f"🔍 Pathway-based annotation selection for: {query}")
    
    try:
        # Step 1: Classify query to find relevant pathways
        pathway_classification = await pathway_classifier(query, max_pathways)
        
        if not pathway_classification["success"] or not pathway_classification["relevant_pathways"]:
            logger.warning(f"⚠️ No relevant pathways found for query: {query}")
            return {
                "success": False,
                "error": f"No relevant KEGG pathways found for query: {query}",
                "selected_proteins": [],
                "pathways_used": []
            }
        
        # Step 2: Extract search terms and find proteins in relevant pathways
        search_terms = pathway_classification["extracted_terms"]
        protein_discovery = await pathway_based_protein_discovery(
            search_terms, 
            functional_category, 
            max_pathways, 
            max_proteins_per_pathway
        )
        
        if not protein_discovery["success"]:
            return {
                "success": False,
                "error": protein_discovery.get("error", "Protein discovery failed"),
                "selected_proteins": [],
                "pathways_used": []
            }
        
        # Step 3: Format results for similarity search
        selected_proteins = []
        for protein in protein_discovery["proteins_found"]:
            selected_proteins.append({
                "protein_id": protein["protein_id"],
                "ko_id": protein["ko_id"],
                "ko_description": protein["ko_description"],
                "pathway_id": protein["pathway_id"],
                "pathway_relevance": protein["pathway_relevance"]
            })
        
        logger.info(f"✅ Pathway-based selection complete: {len(selected_proteins)} proteins from {protein_discovery['pathways_analyzed']} pathways")
        
        return {
            "success": True,
            "original_query": query,
            "functional_category": functional_category,
            "selected_proteins": selected_proteins,
            "pathways_used": protein_discovery["pathway_details"],
            "total_proteins": len(selected_proteins),
            "selection_method": "kegg_pathway_based",
            "search_terms_used": search_terms
        }
        
    except Exception as e:
        logger.error(f"❌ Error in pathway-based annotation selection: {e}")
        return {
            "success": False,
            "error": str(e),
            "selected_proteins": [],
            "pathways_used": []
        }