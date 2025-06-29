#!/usr/bin/env python3
"""
Dynamic KEGG Pathway-Based Protein Discovery System
Replaces hardcoded examples with intelligent pathway-based search
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
import asyncio
from collections import defaultdict

# Import Neo4j query processor
import sys
sys.path.append(str(Path(__file__).parent.parent))
from llm.query_processor import Neo4jQueryProcessor
from llm.config import LLMConfig

logger = logging.getLogger(__name__)

class KEGGPathwayMapper:
    """Maps KEGG pathways to KO orthologs and finds relevant proteins in our database"""
    
    def __init__(self, reference_dir: str = "data/reference"):
        self.reference_dir = Path(reference_dir)
        self.ko_pathway_map = {}  # KO -> [pathways]
        self.pathway_ko_map = {}  # pathway -> [KOs]
        self.ko_descriptions = {}  # KO -> description
        self._load_reference_data()
    
    def _load_reference_data(self):
        """Load KEGG pathway and KO description data"""
        try:
            # Load KO pathway mappings
            ko_pathway_file = self.reference_dir / "ko_pathway.list"
            if ko_pathway_file.exists():
                with open(ko_pathway_file, 'r') as f:
                    for line in f:
                        if line.strip() and '\t' in line:
                            ko_id, pathway_id = line.strip().split('\t')
                            ko_clean = ko_id.replace('ko:', '')
                            pathway_clean = pathway_id.replace('path:', '')
                            
                            if ko_clean not in self.ko_pathway_map:
                                self.ko_pathway_map[ko_clean] = []
                            self.ko_pathway_map[ko_clean].append(pathway_clean)
                            
                            if pathway_clean not in self.pathway_ko_map:
                                self.pathway_ko_map[pathway_clean] = []
                            self.pathway_ko_map[pathway_clean].append(ko_clean)
            
            # Load KO descriptions
            ko_list_file = self.reference_dir / "ko_list"
            if ko_list_file.exists():
                with open(ko_list_file, 'r') as f:
                    next(f)  # Skip header
                    for line in f:
                        if line.strip() and '\t' in line:
                            parts = line.strip().split('\t')
                            if len(parts) >= 12:
                                ko_id = parts[0]
                                description = parts[11]  # definition column
                                self.ko_descriptions[ko_id] = description
            
            logger.info(f"✅ Loaded {len(self.ko_pathway_map)} KO->pathway mappings")
            logger.info(f"✅ Loaded {len(self.pathway_ko_map)} pathway->KO mappings") 
            logger.info(f"✅ Loaded {len(self.ko_descriptions)} KO descriptions")
            
        except Exception as e:
            logger.error(f"❌ Error loading KEGG reference data: {e}")
    
    def find_relevant_pathways(self, query_terms: List[str], max_pathways: int = 5) -> List[Dict[str, Any]]:
        """Find KEGG pathways most relevant to query terms"""
        pathway_scores = defaultdict(int)
        pathway_matches = defaultdict(list)
        
        # Score pathways based on KO description matches
        for ko_id, description in self.ko_descriptions.items():
            description_lower = description.lower()
            
            for term in query_terms:
                term_lower = term.lower()
                if term_lower in description_lower:
                    # Get pathways for this KO
                    pathways = self.ko_pathway_map.get(ko_id, [])
                    for pathway in pathways:
                        pathway_scores[pathway] += 1
                        pathway_matches[pathway].append({
                            'ko_id': ko_id,
                            'description': description,
                            'matched_term': term
                        })
        
        # Sort pathways by relevance score
        sorted_pathways = sorted(pathway_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top pathways with details
        results = []
        for pathway_id, score in sorted_pathways[:max_pathways]:
            results.append({
                'pathway_id': pathway_id,
                'relevance_score': score,
                'matched_kos': len(pathway_matches[pathway_id]),
                'ko_matches': pathway_matches[pathway_id][:10],  # Top 10 matches
                'total_kos_in_pathway': len(self.pathway_ko_map.get(pathway_id, []))
            })
        
        return results
    
    def get_pathway_kos(self, pathway_id: str) -> List[str]:
        """Get all KO orthologs in a specific pathway"""
        return self.pathway_ko_map.get(pathway_id, [])
    
    def get_ko_description(self, ko_id: str) -> str:
        """Get description for a KO ortholog"""
        return self.ko_descriptions.get(ko_id, f"Unknown KO: {ko_id}")

async def pathway_based_protein_discovery(
    query_terms: List[str],
    functional_category: str = "metabolism",
    max_pathways: int = 3,
    max_proteins_per_pathway: int = 10
) -> Dict[str, Any]:
    """
    Discover proteins based on KEGG pathway analysis instead of hardcoded examples
    
    Args:
        query_terms: Terms to search for in pathway descriptions
        functional_category: Biological category for context
        max_pathways: Maximum number of relevant pathways to analyze
        max_proteins_per_pathway: Maximum proteins to return per pathway
        
    Returns:
        Dict with pathway-based protein discovery results
    """
    logger.info(f"🔍 Starting pathway-based discovery for: {query_terms}")
    
    try:
        # Initialize pathway mapper
        mapper = KEGGPathwayMapper()
        
        # Find relevant pathways
        relevant_pathways = mapper.find_relevant_pathways(query_terms, max_pathways)
        
        if not relevant_pathways:
            logger.warning(f"⚠️ No relevant pathways found for terms: {query_terms}")
            return {
                "success": False,
                "error": f"No KEGG pathways found matching terms: {', '.join(query_terms)}",
                "pathways_analyzed": 0,
                "proteins_found": []
            }
        
        # Query database for proteins in these pathways
        config = LLMConfig()
        neo4j = Neo4jQueryProcessor(config)
        
        all_proteins = []
        pathway_details = []
        
        for pathway_info in relevant_pathways:
            pathway_id = pathway_info['pathway_id']
            pathway_kos = mapper.get_pathway_kos(pathway_id)
            
            if not pathway_kos:
                continue
            
            # Query for proteins with these KO annotations
            ko_list_str = "', '".join(pathway_kos)
            protein_query = f"""
            MATCH (p:Protein)-[:HASFUNCTION]->(ko:KEGGOrtholog)
            WHERE ko.id IN ['{ko_list_str}']
            RETURN p.id AS protein_id, ko.id AS ko_id, ko.description AS ko_description
            ORDER BY ko.id, p.id
            LIMIT {max_proteins_per_pathway}
            """
            
            try:
                result = await neo4j.process_query(protein_query, query_type="cypher")
                pathway_proteins = []
                
                for item in result.results:
                    pathway_proteins.append({
                        'protein_id': item['protein_id'],
                        'ko_id': item['ko_id'],
                        'ko_description': item['ko_description'],
                        'pathway_id': pathway_id,
                        'pathway_relevance': pathway_info['relevance_score']
                    })
                
                all_proteins.extend(pathway_proteins)
                
                pathway_details.append({
                    'pathway_id': pathway_id,
                    'relevance_score': pathway_info['relevance_score'],
                    'proteins_found': len(pathway_proteins),
                    'total_kos_in_pathway': pathway_info['total_kos_in_pathway'],
                    'matched_kos': pathway_info['matched_kos']
                })
                
                logger.info(f"📊 Pathway {pathway_id}: {len(pathway_proteins)} proteins found")
                
            except Exception as e:
                logger.error(f"❌ Error querying pathway {pathway_id}: {e}")
                continue
        
        logger.info(f"✅ Pathway-based discovery complete: {len(all_proteins)} proteins across {len(pathway_details)} pathways")
        
        return {
            "success": True,
            "query_terms": query_terms,
            "functional_category": functional_category,
            "pathways_analyzed": len(pathway_details),
            "pathway_details": pathway_details,
            "proteins_found": all_proteins,
            "total_proteins": len(all_proteins),
            "discovery_method": "kegg_pathway_based"
        }
        
    except Exception as e:
        logger.error(f"❌ Error in pathway-based protein discovery: {e}")
        return {
            "success": False,
            "error": str(e),
            "pathways_analyzed": 0,
            "proteins_found": []
        }

async def pathway_classifier(
    query: str,
    max_pathways: int = 5
) -> Dict[str, Any]:
    """
    Classify a query and find the most relevant KEGG pathways
    
    Args:
        query: User query to analyze
        max_pathways: Maximum pathways to return
        
    Returns:
        Dict with pathway classification results
    """
    logger.info(f"🧠 Classifying query for pathway relevance: {query}")
    
    try:
        # Extract key terms from query
        query_lower = query.lower()
        
        # Common biological terms to search for
        search_terms = []
        
        # Extract specific terms
        biological_keywords = [
            'transport', 'metabolism', 'synthesis', 'degradation', 'pathway',
            'glycolysis', 'respiration', 'photosynthesis', 'amino acid', 'fatty acid',
            'nucleotide', 'carbohydrate', 'lipid', 'protein', 'enzyme',
            'kinase', 'dehydrogenase', 'synthase', 'reductase', 'oxidase',
            'transferase', 'hydrolase', 'lyase', 'isomerase', 'ligase'
        ]
        
        for keyword in biological_keywords:
            if keyword in query_lower:
                search_terms.append(keyword)
        
        # Also add any quoted terms or specific mentions
        words = query_lower.split()
        for word in words:
            if len(word) > 3 and word.isalpha():
                search_terms.append(word)
        
        # Remove duplicates
        search_terms = list(set(search_terms))
        
        if not search_terms:
            search_terms = ['metabolism']  # Default fallback
        
        logger.info(f"📋 Extracted search terms: {search_terms}")
        
        # Find relevant pathways
        mapper = KEGGPathwayMapper()
        relevant_pathways = mapper.find_relevant_pathways(search_terms, max_pathways)
        
        return {
            "success": True,
            "original_query": query,
            "extracted_terms": search_terms,
            "relevant_pathways": relevant_pathways,
            "classification_method": "kegg_pathway_analysis"
        }
        
    except Exception as e:
        logger.error(f"❌ Error in pathway classification: {e}")
        return {
            "success": False,
            "error": str(e),
            "relevant_pathways": []
        }