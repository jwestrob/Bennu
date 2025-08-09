#!/usr/bin/env python3
"""
Detector registry for resolving biological phrases to concrete KG identifiers.
Uses only schema-validated properties and parameterized queries.
"""

import logging
import re
from typing import Dict, List, Set, Any
from pydantic import BaseModel

from .schema_map import SchemaMap

logger = logging.getLogger(__name__)


class DetectorResult(BaseModel):
    """Result of detector resolution."""
    ko_ids: List[str]
    pfam_ids: List[str] 
    resolution_notes: str


class DetectorRegistry:
    """
    Resolve plain biological concepts to detector sets (KEGGOrtholog IDs, PFAM Domain IDs).
    Uses only actual database properties and parameterized queries.
    """
    
    def __init__(self, graph_client, schema_map: SchemaMap):
        self.graph_client = graph_client
        self.schema_map = schema_map
        
        # Validate required labels exist
        self.schema_map.assert_label("KEGGOrtholog")
        self.schema_map.assert_label("Domain")
        
        # Validate required properties exist 
        self.schema_map.assert_property("KEGGOrtholog", "description")
        self.schema_map.assert_property("KEGGOrtholog", "id")
        self.schema_map.assert_property("Domain", "description") 
        self.schema_map.assert_property("Domain", "id")
        
        logger.info("🔍 DetectorRegistry initialized with schema-locked queries")
    
    async def resolve(self, phrase: str, k: int = 20, candidate_cap: int = 10) -> DetectorResult:
        """
        Resolve biological phrase to concrete KG identifiers.
        
        Args:
            phrase: Plain biological phrase (e.g., "rubisco", "transport protein")
            k: Overall result limit (not used in resolution, only passed through)
            candidate_cap: Max candidates per detector type
            
        Returns:
            DetectorResult with resolved KO and PFAM IDs
        """
        try:
            # Normalize phrase
            normalized_phrase = self._normalize_phrase(phrase)
            if not normalized_phrase:
                return DetectorResult(
                    ko_ids=[],
                    pfam_ids=[],
                    resolution_notes=f"Phrase '{phrase}' normalized to empty string"
                )
            
            logger.debug(f"🔍 Resolving phrase: '{phrase}' -> '{normalized_phrase}'")
            
            # Check for direct accession patterns
            direct_ko_ids = self._extract_direct_ko_ids(phrase)
            direct_pfam_ids = self._extract_direct_pfam_ids(phrase)
            
            # Text-based resolution
            ko_ids = await self._resolve_kegg_orthologs(normalized_phrase, candidate_cap)
            pfam_ids = await self._resolve_pfam_domains(normalized_phrase, candidate_cap)
            
            # Combine direct and text-based results
            all_ko_ids = list(set(direct_ko_ids + ko_ids))
            all_pfam_ids = list(set(direct_pfam_ids + pfam_ids))
            
            # Build resolution notes
            notes_parts = []
            if direct_ko_ids:
                notes_parts.append(f"Direct KO accessions: {direct_ko_ids}")
            if direct_pfam_ids:
                notes_parts.append(f"Direct PFAM accessions: {direct_pfam_ids}")
            if ko_ids:
                notes_parts.append(f"Text-matched KOs: {len(ko_ids)} found")
            if pfam_ids:
                notes_parts.append(f"Text-matched PFAMs: {len(pfam_ids)} found")
            
            if not notes_parts:
                notes_parts.append(f"No detectors found for '{phrase}'")
            
            resolution_notes = "; ".join(notes_parts)
            
            result = DetectorResult(
                ko_ids=all_ko_ids,
                pfam_ids=all_pfam_ids,
                resolution_notes=resolution_notes
            )
            
            logger.info(f"📊 Resolved '{phrase}': {len(all_ko_ids)} KOs, {len(all_pfam_ids)} PFAMs")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error resolving phrase '{phrase}': {e}")
            return DetectorResult(
                ko_ids=[],
                pfam_ids=[],
                resolution_notes=f"Resolution failed: {e}"
            )
    
    def _normalize_phrase(self, phrase: str) -> str:
        """
        Normalize phrase for biological entity matching.
        Remove numerals, pronouns, and trivial suffixes.
        """
        if not phrase:
            return ""
        
        # Convert to lowercase
        normalized = phrase.lower().strip()
        
        # Remove common stop words and numerals
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "from", "what", "where", "when", "how", "why", "which", "that", "this", "these", "those",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "can", "must", "find", "show", "tell",
            "me", "about", "all", "any", "some", "many", "much", "more", "most", "analysis", "analyze",
            "present", "contain", "contains", "gene", "genes", "protein", "proteins"
        }
        
        # Split and filter words
        words = []
        for word in re.split(r'[\\s,;.!?()]+', normalized):
            # Skip empty words
            if not word:
                continue
            # Skip numerals
            if word.isdigit():
                continue
            # Skip pronouns and stop words
            if word in stop_words:
                continue
            # Skip very short words
            if len(word) < 3:
                continue
            
            # Clean word (remove non-alphanumeric except hyphens and underscores)
            clean_word = re.sub(r'[^a-z0-9_-]', '', word)
            if clean_word:
                words.append(clean_word)
        
        # Join words back
        result = " ".join(words)
        logger.debug(f"🧹 Normalized '{phrase}' -> '{result}'")
        return result
    
    def _extract_direct_ko_ids(self, phrase: str) -> List[str]:
        """Extract direct KO accessions from phrase (K##### pattern)."""
        ko_pattern = r'\\bK\\d{5}\\b'
        matches = re.findall(ko_pattern, phrase, re.IGNORECASE)
        return [match.upper() for match in matches]
    
    def _extract_direct_pfam_ids(self, phrase: str) -> List[str]:
        """Extract direct PFAM accessions from phrase (PF##### or PF####.# pattern).""" 
        pfam_pattern = r'\\bPF\\d{5}(?:\\.\\d+)?\\b'
        matches = re.findall(pfam_pattern, phrase, re.IGNORECASE)
        return [match.upper() for match in matches]
    
    async def _resolve_kegg_orthologs(self, phrase: str, candidate_cap: int) -> List[str]:
        """
        Resolve phrase to KEGGOrtholog IDs using text matching on description.
        Uses only schema-validated properties.
        """
        if not phrase:
            return []
        
        # Build parameterized query using only validated properties
        cypher = """
        MATCH (ko:KEGGOrtholog)
        WHERE toLower(ko.description) CONTAINS $phrase_lc
        RETURN ko.id AS id, ko.description AS description
        LIMIT $candidate_cap
        """
        
        params = {
            "phrase_lc": phrase.lower(),
            "candidate_cap": candidate_cap
        }
        
        try:
            logger.debug(f"🔍 KO query: {cypher}")
            logger.debug(f"🔍 KO params: {params}")
            
            # Use direct database access with parameters
            with self.graph_client.driver.session() as session:
                result = session.run(cypher, **params)
                result_records = [dict(record) for record in result]
            
            ko_ids = []
            for record in result_records:
                ko_id = record.get("id")
                description = record.get("description", "")
                if ko_id:
                    ko_ids.append(ko_id)
                    logger.debug(f"🎯 Found KO: {ko_id} - {description}")
            
            logger.info(f"📊 KO resolution: '{phrase}' -> {len(ko_ids)} matches")
            return ko_ids
            
        except Exception as e:
            logger.error(f"❌ KO resolution failed for '{phrase}': {e}")
            return []
    
    async def _resolve_pfam_domains(self, phrase: str, candidate_cap: int) -> List[str]:
        """
        Resolve phrase to PFAM Domain IDs using text matching on description.
        Uses only schema-validated properties.
        """
        if not phrase:
            return []
        
        # Build parameterized query using only validated properties
        cypher = """
        MATCH (dom:Domain)
        WHERE toLower(dom.description) CONTAINS $phrase_lc
        RETURN dom.id AS id, dom.description AS description
        LIMIT $candidate_cap
        """
        
        params = {
            "phrase_lc": phrase.lower(), 
            "candidate_cap": candidate_cap
        }
        
        try:
            logger.debug(f"🔍 PFAM query: {cypher}")
            logger.debug(f"🔍 PFAM params: {params}")
            
            # Use direct database access with parameters
            with self.graph_client.driver.session() as session:
                result = session.run(cypher, **params)
                result_records = [dict(record) for record in result]
            
            pfam_ids = []
            for record in result_records:
                pfam_id = record.get("id")
                description = record.get("description", "")
                if pfam_id:
                    pfam_ids.append(pfam_id)
                    logger.debug(f"🎯 Found PFAM: {pfam_id} - {description}")
            
            logger.info(f"📊 PFAM resolution: '{phrase}' -> {len(pfam_ids)} matches")
            return pfam_ids
            
        except Exception as e:
            logger.error(f"❌ PFAM resolution failed for '{phrase}': {e}")
            return []