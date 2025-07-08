#!/usr/bin/env python3
"""
Genome scoping system for ensuring queries are properly filtered by genome context.
Addresses the issue where queries return data from all genomes instead of being scoped.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GenomeScope:
    """Genome scoping information for queries."""
    genome_id: Optional[str]
    scope_type: str  # "single", "multiple", "all", "unspecified"
    genome_pattern: Optional[str]
    confidence: float
    reasoning: str

class GenomeScopeDetector:
    """
    Detects and extracts genome scoping information from natural language queries.
    Ensures queries are properly scoped to specific genomes when context is available.
    """
    
    def __init__(self):
        self.genome_patterns = [
            # PRIORITY ORDER: Most specific patterns first to avoid false matches
            
            # Specific genome ID patterns (most specific first)
            r"(PLM0_[A-Za-z0-9_\-\.]+)",  # PLM0_60_b1_sep16_Maxbin2_047_curated
            r"(GCF_[A-Za-z0-9_\-\.]+)",  # NCBI RefSeq
            r"(GCA_[A-Za-z0-9_\-\.]+)",  # NCBI GenBank
            r"(OD1_[A-Za-z0-9_\-\.]+)",  # OD1 genomes
            r"(RIFCSPHIGHO2_[A-Za-z0-9_\-\.]+)",  # RIFCSPHIGHO2 genomes
            r"(Acidovorax_[A-Za-z0-9_\-\.]+)",  # Acidovorax genomes
            
            # Explicit genome mentions with long IDs (at least 10 chars to avoid "genome id")
            r"in\s+genome\s+([A-Za-z0-9_\-\.]{10,})",
            r"from\s+genome\s+([A-Za-z0-9_\-\.]{10,})",
            r"([A-Za-z0-9_\-\.]{10,})\s+\(genome",  # "PLM0_... (genome id)"
            r"of\s+([A-Za-z0-9_\-\.]{10,})\s+\(genome",  # "of PLM0_... (genome id)"
            
            # Organism name patterns (binomial nomenclature)
            r"in\s+([A-Z][a-z]+\s+[a-z]+)",  # "in Escherichia coli"
            r"from\s+([A-Z][a-z]+\s+[a-z]+)",  # "from Bacillus subtilis"
        ]
        
        self.multi_genome_patterns = [
            r"across\s+genomes?",
            r"between\s+genomes?",
            r"all\s+genomes?",
            r"multiple\s+genomes?",
            r"compare\s+genomes?",
            r"genomes?\s+in\s+dataset"
        ]
        
        # Known genome IDs - EXACT MATCHES ONLY for stability
        self.known_genome_ids = [
            "PLM0_60_b1_sep16_Maxbin2_047_curated_contigs",
            "PLM0_60_b1_sep16_Maxbin2_047_curated.contigs",  # Alternative format
            "OD1_41_01_41_220_01",
            "Acidovorax_64", 
            "RIFCSPHIGHO2_01_FULL"
        ]
    
    def detect_genome_scope(self, question: str) -> GenomeScope:
        """
        Detect genome scoping from natural language question.
        
        Args:
            question: User's natural language question
            
        Returns:
            GenomeScope with detected scoping information
        """
        logger.info(f"🔍 Detecting genome scope in: {question[:50]}...")
        
        question_lower = question.lower()
        
        # Check for multi-genome patterns first
        for pattern in self.multi_genome_patterns:
            if re.search(pattern, question_lower):
                return GenomeScope(
                    genome_id=None,
                    scope_type="multiple",
                    genome_pattern=pattern,
                    confidence=0.9,
                    reasoning=f"Multi-genome pattern detected: {pattern}"
                )
        
        # Check for specific genome mentions
        for pattern in self.genome_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                genome_candidate = match.group(1)
                
                # Validate against known genome IDs
                matched_genome = self._match_known_genome(genome_candidate)
                
                if matched_genome:
                    return GenomeScope(
                        genome_id=matched_genome,
                        scope_type="single",
                        genome_pattern=pattern,
                        confidence=0.95,
                        reasoning=f"Specific genome identified: {matched_genome}"
                    )
                else:
                    return GenomeScope(
                        genome_id=genome_candidate,
                        scope_type="single",
                        genome_pattern=pattern,
                        confidence=0.7,
                        reasoning=f"Potential genome identifier: {genome_candidate}"
                    )
        
        # Default: unspecified scope
        return GenomeScope(
            genome_id=None,
            scope_type="unspecified",
            genome_pattern=None,
            confidence=0.5,
            reasoning="No explicit genome scope detected"
        )
    
    def _match_known_genome(self, candidate: str) -> Optional[str]:
        """Match candidate against known genome IDs - WITH SUFFIX MAPPING."""
        candidate_clean = candidate.strip()
        
        # First try exact matches
        for genome_id in self.known_genome_ids:
            if candidate_clean == genome_id:
                logger.info(f"✅ Exact genome match: {candidate_clean} → {genome_id}")
                return genome_id
        
        # Try case-insensitive exact matches
        candidate_lower = candidate_clean.lower()
        for genome_id in self.known_genome_ids:
            if candidate_lower == genome_id.lower():
                logger.info(f"✅ Case-insensitive genome match: {candidate_clean} → {genome_id}")
                return genome_id
        
        # Try with common format variations (. vs _)
        candidate_normalized = candidate_clean.replace('.', '_')
        for genome_id in self.known_genome_ids:
            genome_normalized = genome_id.replace('.', '_')
            if candidate_normalized.lower() == genome_normalized.lower():
                logger.info(f"✅ Format-normalized genome match: {candidate_clean} → {genome_id}")
                return genome_id
        
        # CRITICAL: Try prefix matching with common suffixes
        # User provides: PLM0_60_b1_sep16_Maxbin2_047_curated
        # Database has: PLM0_60_b1_sep16_Maxbin2_047_curated_contigs
        common_suffixes = ['_contigs', '.contigs', '_scaffolds', '.scaffolds', '_final', '.final']
        
        for genome_id in self.known_genome_ids:
            for suffix in common_suffixes:
                if genome_id.endswith(suffix):
                    # Remove suffix and compare
                    genome_base = genome_id[:-len(suffix)]
                    if candidate_clean.lower() == genome_base.lower():
                        logger.info(f"✅ Prefix genome match: {candidate_clean} → {genome_id} (added {suffix})")
                        return genome_id
        
        # Also try the reverse - if user provides with suffix, try without
        for suffix in common_suffixes:
            if candidate_clean.endswith(suffix):
                candidate_base = candidate_clean[:-len(suffix)]
                for genome_id in self.known_genome_ids:
                    if candidate_base.lower() == genome_id.lower():
                        logger.info(f"✅ Suffix removal match: {candidate_clean} → {genome_id} (removed {suffix})")
                        return genome_id
        
        # No match found - be conservative
        logger.warning(f"❌ No genome match for: {candidate_clean}")
        logger.warning(f"📋 Available genomes: {', '.join(self.known_genome_ids)}")
        return None
    
# Fuzzy matching removed for stability - using conservative exact matching only

class QueryScopeEnforcer:
    """
    Enforces genome scoping in generated Cypher queries.
    Ensures queries are properly filtered to specific genomes when context is available.
    """
    
    def __init__(self):
        self.scope_detector = GenomeScopeDetector()
    
    def enforce_genome_scope(self, question: str, cypher_query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Enforce genome scoping in Cypher queries.
        
        Args:
            question: Original user question
            cypher_query: Generated Cypher query
            
        Returns:
            Tuple of (modified_query, scope_metadata)
        """
        logger.info("🎯 Enforcing genome scope in query")
        
        # Detect genome scope from question
        scope = self.scope_detector.detect_genome_scope(question)
        
        metadata = {
            "genome_scope": scope,
            "original_query": cypher_query,
            "scope_applied": False,
            "scope_reasoning": scope.reasoning
        }
        
        # Apply scoping based on detected scope
        if scope.scope_type == "single" and scope.genome_id:
            modified_query = self._apply_single_genome_scope(cypher_query, scope.genome_id)
            metadata["scope_applied"] = True
            metadata["scope_genome"] = scope.genome_id
            logger.info(f"✅ Applied single genome scope: {scope.genome_id}")
            return modified_query, metadata
        
        elif scope.scope_type == "multiple":
            # For multi-genome queries, ensure we're explicitly querying across genomes
            modified_query = self._apply_multi_genome_scope(cypher_query)
            metadata["scope_applied"] = True
            metadata["scope_type"] = "multiple"
            logger.info("✅ Applied multi-genome scope")
            return modified_query, metadata
        
        elif scope.scope_type == "unspecified":
            # For unspecified scope, add a warning and potentially limit results
            modified_query = self._apply_default_scope(cypher_query)
            metadata["scope_applied"] = True
            metadata["scope_type"] = "limited"
            logger.info("⚠️ Applied default limiting scope")
            return modified_query, metadata
        
        # No scope changes needed
        logger.info("ℹ️ No genome scope changes applied")
        return cypher_query, metadata
    
    def _apply_single_genome_scope(self, cypher_query: str, genome_id: str) -> str:
        """Apply single genome scoping to Cypher query - REPLACE EXISTING CONSTRAINTS."""
        import re
        
        # CRITICAL: Replace existing genome constraints instead of skipping
        # Common patterns to replace:
        # 1. {genomeId:'value'} -> {id:'correct_value'}
        # 2. {id:'wrong_value'} -> {id:'correct_value'}
        
        # Pattern 1: Replace genomeId with id and update value
        pattern1 = r"\{genomeId\s*:\s*['\"]([^'\"]+)['\"]\}"
        match1 = re.search(pattern1, cypher_query, re.IGNORECASE)
        if match1:
            old_constraint = match1.group(0)
            new_constraint = f"{{id:'{genome_id}'}}"
            modified_query = cypher_query.replace(old_constraint, new_constraint)
            logger.info(f"🔧 Replaced genomeId constraint: {old_constraint} → {new_constraint}")
            return modified_query
        
        # Pattern 2: Replace existing id constraint with correct value
        pattern2 = r"\{id\s*:\s*['\"]([^'\"]+)['\"]\}"
        match2 = re.search(pattern2, cypher_query, re.IGNORECASE)
        if match2:
            old_value = match2.group(1)
            if old_value.lower() != genome_id.lower():
                old_constraint = match2.group(0)
                new_constraint = f"{{id:'{genome_id}'}}"
                modified_query = cypher_query.replace(old_constraint, new_constraint)
                logger.info(f"🔧 Updated genome id constraint: {old_value} → {genome_id}")
                return modified_query
            else:
                logger.info(f"Query already has correct genome id: {genome_id}")
                return cypher_query
        
        # Find the best place to inject genome filtering
        query_upper = cypher_query.upper()
        
        # Strategy 1: If query has BELONGSTOGENOME relationship, add genome filter
        if "BELONGSTOGENOME" in query_upper:
            # Look for pattern like (gene)-[:BELONGSTOGENOME]->(genome)
            pattern = r"(\([\w]*\))-\[:BELONGSTOGENOME\]->\((\w*):?Genome\)"
            match = re.search(pattern, cypher_query, re.IGNORECASE)
            
            if match:
                # Add genome ID filter using the 'id' field (not genomeId)
                genome_var = match.group(2) if match.group(2) else "genome"
                filter_clause = f"{genome_var}.id = '{genome_id}'"
                
                # Insert the filter in the WHERE clause or create one
                if "WHERE" in query_upper:
                    # Add to existing WHERE clause
                    where_pos = cypher_query.upper().find("WHERE")
                    insert_pos = cypher_query.find(" ", where_pos + 5)  # After "WHERE "
                    modified_query = (
                        cypher_query[:insert_pos] + 
                        f" {filter_clause} AND " + 
                        cypher_query[insert_pos:].lstrip()
                    )
                else:
                    # Add new WHERE clause before RETURN
                    return_pos = query_upper.find("RETURN")
                    if return_pos != -1:
                        modified_query = (
                            cypher_query[:return_pos] + 
                            f"WHERE {filter_clause} " + 
                            cypher_query[return_pos:]
                        )
                    else:
                        modified_query = cypher_query + f" WHERE {filter_clause}"
                
                return modified_query
        
        # Strategy 2: If query starts with MATCH (genome:Genome), add ID filter
        if re.search(r"MATCH\s+\([^)]*:Genome", cypher_query, re.IGNORECASE):
            # Extract genome variable name
            genome_match = re.search(r"MATCH\s+\((\w*):Genome", cypher_query, re.IGNORECASE)
            if genome_match:
                genome_var = genome_match.group(1)
                filter_clause = f"{genome_var}.id = '{genome_id}'"
                
                # Add filter after the MATCH clause
                if "WHERE" in query_upper:
                    where_pos = cypher_query.upper().find("WHERE")
                    insert_pos = cypher_query.find(" ", where_pos + 5)
                    modified_query = (
                        cypher_query[:insert_pos] + 
                        f" {filter_clause} AND " + 
                        cypher_query[insert_pos:].lstrip()
                    )
                else:
                    return_pos = query_upper.find("RETURN")
                    if return_pos != -1:
                        modified_query = (
                            cypher_query[:return_pos] + 
                            f"WHERE {filter_clause} " + 
                            cypher_query[return_pos:]
                        )
                    else:
                        modified_query = cypher_query + f" WHERE {filter_clause}"
                
                return modified_query
        
        # Strategy 3: Add genome filtering at the beginning
        # This is more aggressive but ensures genome scoping
        genome_filter = f"MATCH (genome:Genome {{id: '{genome_id}'}}) "
        
        if cypher_query.upper().startswith("MATCH"):
            # Insert genome filter before existing MATCH
            modified_query = genome_filter + cypher_query
        else:
            # Prepend genome filter
            modified_query = genome_filter + cypher_query
        
        return modified_query
    
    def _apply_multi_genome_scope(self, cypher_query: str) -> str:
        """Apply multi-genome scoping to Cypher query."""
        # For multi-genome queries, ensure we're grouping by genome
        if "GROUP BY" not in cypher_query.upper():
            # Add genome grouping if not present
            if "RETURN" in cypher_query.upper():
                return_pos = cypher_query.upper().find("RETURN")
                return_clause = cypher_query[return_pos:]
                
                # Check if genome is already in RETURN clause
                if "genome" not in return_clause.lower():
                    # Add genome to RETURN clause
                    return_match = re.search(r"RETURN\s+(.+)", return_clause, re.IGNORECASE)
                    if return_match:
                        existing_return = return_match.group(1)
                        modified_return = f"RETURN genome.id as genome_id, {existing_return}"
                        modified_query = cypher_query[:return_pos] + modified_return
                        return modified_query
        
        return cypher_query
    
    def _apply_default_scope(self, cypher_query: str) -> str:
        """Apply default scoping to prevent overly broad queries."""
        # Add a reasonable LIMIT if not present
        if "LIMIT" not in cypher_query.upper():
            modified_query = cypher_query + " LIMIT 100"
            return modified_query
        
        return cypher_query