"""
Explicit Genome Selection System for accurate genome filtering.

This module provides explicit genome lookup and selection capabilities to replace
pattern-matching approaches that fail to correctly identify target genomes.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

@dataclass
class GenomeMatch:
    """Represents a genome match with confidence scoring."""
    genome_id: str
    match_score: float
    match_reason: str
    original_query: str
    
@dataclass
class GenomeSelectionResult:
    """Result of genome selection process."""
    success: bool
    selected_genome: Optional[str] = None
    match_score: float = 0.0
    match_reason: str = ""
    available_genomes: List[str] = None
    error_message: str = ""
    suggestions: List[str] = None

class GenomeSelector:
    """
    Explicit genome selection system that looks up available genomes
    and performs intelligent matching to user requests.
    """
    
    def __init__(self, neo4j_processor):
        """Initialize with Neo4j processor for genome queries."""
        self.neo4j_processor = neo4j_processor
        self._cached_genomes = None
        self._cache_timestamp = None
        
        # Keywords that indicate specific genome targeting
        self.genome_targeting_keywords = [
            'for the', 'in the', 'from the', 'within the', 'of the',
            'annotations for', 'proteins in', 'genes in', 'domains in',
            'functions in', 'bgcs in', 'cazymes in',
            # Additional patterns for agentic tasks
            'chosen genome', 'selected genome', 'target genome'
        ]
    
    async def get_available_genomes(self, force_refresh: bool = False) -> List[str]:
        """
        Get list of all available genomes from the database.
        
        Args:
            force_refresh: Force refresh of cached genome list
            
        Returns:
            List of genome IDs available in the database
        """
        import time
        
        # Use cache if available and recent (5 minutes)
        if (not force_refresh and self._cached_genomes and 
            self._cache_timestamp and (time.time() - self._cache_timestamp) < 300):
            return self._cached_genomes
        
        try:
            # Query all genome IDs from Neo4j
            cypher = "MATCH (g:Genome) RETURN g.genomeId as genome_id ORDER BY g.genomeId"
            result = await self.neo4j_processor._execute_cypher(cypher)
            
            genome_ids = [record['genome_id'] for record in result if record.get('genome_id')]
            
            # Cache the results
            self._cached_genomes = genome_ids
            self._cache_timestamp = time.time()
            
            logger.info(f"📊 Retrieved {len(genome_ids)} available genomes from database")
            return genome_ids
            
        except Exception as e:
            logger.error(f"Failed to retrieve available genomes: {e}")
            return []
    
    def extract_genome_request(self, query: str) -> Optional[str]:
        """
        Extract genome name/identifier from user query.
        
        Args:
            query: User query text
            
        Returns:
            Extracted genome name or None if not found
        """
        query_lower = query.lower().strip()
        
        # PRIORITY 1: Look for exact genome ID patterns (MAG identifiers, etc.)
        # Pattern for IDs like PLM0_60_b1_sep16_Maxbin2_047_curated_contigs
        exact_genome_patterns = [
            r'\b([A-Z0-9]+_[a-z0-9_]+_[A-Za-z0-9_]+_[a-z]+_?[a-z]*)\b',  # PLM0_60_b1_sep16_Maxbin2_047_curated_contigs
            r'\b([A-Z]+[0-9]+_[a-z0-9_]+)\b',  # Simpler MAG patterns
            r'\b(candidatus_[A-Za-z_]+_[A-Z0-9_]+_[a-z]+)\b',  # Candidatus with MAG ID
        ]
        
        for pattern in exact_genome_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                
                # Return the full match - likely an exact genome ID
                if len(match) > 10:  # Genome IDs are typically long
                    logger.info(f"🎯 Found exact genome ID pattern: {match}")
                    return match
        
        # PRIORITY 2: Look for genome targeting patterns
        for keyword in self.genome_targeting_keywords:
            if keyword in query_lower:
                # Extract text after the keyword
                parts = query_lower.split(keyword, 1)
                if len(parts) > 1:
                    after_keyword = parts[1].strip()
                    
                    # First check if it's a long genome ID (like PLM0_...)
                    first_word = after_keyword.split()[0] if after_keyword.split() else ""
                    if len(first_word) > 15 and '_' in first_word:  # Likely a genome ID
                        logger.info(f"🎯 Found potential genome ID after keyword: {first_word}")
                        return first_word
                    
                    # Extract the first meaningful word/phrase after keyword
                    # Look for organism names, MAG identifiers, etc.
                    words = after_keyword.split()
                    if words:
                        # Take first word that looks like an organism name
                        candidate = words[0]
                        
                        # Clean up common suffixes
                        candidate = re.sub(r'(genome|mag|bacterium|bacteria)$', '', candidate).strip()
                        
                        if len(candidate) >= 3:  # Minimum length for valid name
                            return candidate
        
        # Look for explicit organism mentions without keywords
        organism_patterns = [
            r'\b([A-Z][a-z]+bacteria|[A-Z][a-z]+coccus|[A-Z][a-z]+archaea)\b',
            r'\b(candidatus[_\s]+[A-Za-z]+)\b',
            r'\b([A-Za-z]+ales|[A-Za-z]+aceae)\b',
            r'\b([A-Z][a-z]{3,})\b'  # Capitalized words 4+ chars
        ]
        
        for pattern in organism_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                
                match = match.lower().strip()
                if len(match) >= 3 and not self._is_generic_term(match):
                    return match
        
        return None
    
    def _is_generic_term(self, term: str) -> bool:
        """Check if term is too generic to be a specific genome identifier."""
        generic_terms = {
            'genome', 'mag', 'assembly', 'bacterium', 'bacteria', 'archaea',
            'protein', 'gene', 'domain', 'function', 'annotation', 'data',
            'result', 'analysis', 'comparison', 'study', 'sample', 'sequence'
        }
        return term.lower() in generic_terms
    
    async def select_genome(self, query: str) -> GenomeSelectionResult:
        """
        Select the best matching genome for a user query.
        
        Args:
            query: User query text
            
        Returns:
            GenomeSelectionResult with selected genome and metadata
        """
        # Extract genome request from query
        genome_request = self.extract_genome_request(query)
        
        if not genome_request:
            return GenomeSelectionResult(
                success=False,
                error_message="No specific genome mentioned in query",
                suggestions=["Try specifying a genome name or organism"]
            )
        
        # Get available genomes
        available_genomes = await self.get_available_genomes()
        
        if not available_genomes:
            return GenomeSelectionResult(
                success=False,
                error_message="No genomes available in database",
                available_genomes=[]
            )
        
        # Find best matching genome
        matches = self._find_matching_genomes(genome_request, available_genomes)
        
        if not matches:
            return GenomeSelectionResult(
                success=False,
                error_message=f"No genome matches found for '{genome_request}'",
                available_genomes=available_genomes,
                suggestions=self._suggest_similar_genomes(genome_request, available_genomes)
            )
        
        # Take the best match
        best_match = matches[0]
        
        # Require minimum confidence threshold (lowered for better matching)
        min_confidence = 0.3  # More permissive for complex genome names
        if best_match.match_score < min_confidence:
            return GenomeSelectionResult(
                success=False,
                error_message=f"Best match for '{genome_request}' has low confidence ({best_match.match_score:.2f})",
                available_genomes=available_genomes,
                suggestions=[m.genome_id for m in matches[:3]]
            )
        
        logger.info(f"🧬 Selected genome: {best_match.genome_id} (score: {best_match.match_score:.2f}, reason: {best_match.match_reason})")
        
        return GenomeSelectionResult(
            success=True,
            selected_genome=best_match.genome_id,
            match_score=best_match.match_score,
            match_reason=best_match.match_reason,
            available_genomes=available_genomes
        )
    
    def _find_matching_genomes(self, request: str, available_genomes: List[str]) -> List[GenomeMatch]:
        """
        Find genomes matching the request using multiple strategies.
        
        Args:
            request: Requested genome name/identifier
            available_genomes: List of available genome IDs
            
        Returns:
            List of GenomeMatch objects sorted by confidence
        """
        matches = []
        request_lower = request.lower()
        
        for genome_id in available_genomes:
            genome_lower = genome_id.lower()
            
            # Strategy 1: Exact substring match
            if request_lower in genome_lower:
                score = len(request_lower) / len(genome_lower)  # Longer matches get higher scores
                
                # Extra bonus for organism names that match well
                if len(request_lower) > 5:  # Substantial organism name
                    score += 0.4  # Higher bonus for longer organism names
                else:
                    score += 0.3  # Standard bonus
                    
                matches.append(GenomeMatch(
                    genome_id=genome_id,
                    match_score=score,
                    match_reason="exact_substring",
                    original_query=request
                ))
                continue
            
            # Strategy 2: Check organism aliases
            for canonical, aliases in self.organism_aliases.items():
                if request_lower in aliases or canonical == request_lower:
                    for alias in aliases:
                        if alias in genome_lower:
                            score = len(alias) / len(genome_lower)
                            matches.append(GenomeMatch(
                                genome_id=genome_id,
                                match_score=score + 0.25,  # Bonus for alias match
                                match_reason=f"alias_match_{canonical}",
                                original_query=request
                            ))
                            break
            
            # Strategy 3: Fuzzy string matching
            similarity = SequenceMatcher(None, request_lower, genome_lower).ratio()
            if similarity > 0.4:  # Minimum similarity threshold
                matches.append(GenomeMatch(
                    genome_id=genome_id,
                    match_score=similarity,
                    match_reason="fuzzy_match",
                    original_query=request
                ))
            
            # Strategy 4: Token-based matching (split on underscores/dots)
            request_tokens = set(re.split(r'[_\.\-\s]+', request_lower))
            genome_tokens = set(re.split(r'[_\.\-\s]+', genome_lower))
            
            if request_tokens and genome_tokens:
                overlap = len(request_tokens.intersection(genome_tokens))
                if overlap > 0:
                    token_score = overlap / len(request_tokens.union(genome_tokens))
                    if token_score > 0.3:
                        matches.append(GenomeMatch(
                            genome_id=genome_id,
                            match_score=token_score,
                            match_reason="token_match",
                            original_query=request
                        ))
        
        # Remove duplicates and sort by score
        unique_matches = {}
        for match in matches:
            if match.genome_id not in unique_matches or match.match_score > unique_matches[match.genome_id].match_score:
                unique_matches[match.genome_id] = match
        
        sorted_matches = sorted(unique_matches.values(), key=lambda x: x.match_score, reverse=True)
        
        logger.debug(f"Found {len(sorted_matches)} genome matches for '{request}'")
        for match in sorted_matches[:3]:  # Log top 3
            logger.debug(f"  {match.genome_id}: {match.match_score:.2f} ({match.match_reason})")
        
        return sorted_matches
    
    def _suggest_similar_genomes(self, request: str, available_genomes: List[str], limit: int = 5) -> List[str]:
        """
        Suggest similar genomes when no good match is found.
        
        Args:
            request: Original request
            available_genomes: Available genome IDs
            limit: Maximum number of suggestions
            
        Returns:
            List of suggested genome IDs
        """
        suggestions = []
        request_lower = request.lower()
        
        # Find genomes with partial matches
        for genome_id in available_genomes:
            genome_lower = genome_id.lower()
            
            # Check for any common tokens
            request_tokens = set(re.split(r'[_\.\-\s]+', request_lower))
            genome_tokens = set(re.split(r'[_\.\-\s]+', genome_lower))
            
            if request_tokens.intersection(genome_tokens):
                suggestions.append(genome_id)
            elif any(token in genome_lower for token in request_tokens if len(token) > 2):
                suggestions.append(genome_id)
        
        # If no partial matches, suggest some examples
        if not suggestions:
            suggestions = available_genomes[:limit]
        
        return suggestions[:limit]
    
    def should_use_genome_selection(self, query: str) -> bool:
        """
        Determine if query requires explicit genome selection.
        
        Args:
            query: User query text
            
        Returns:
            True if genome selection should be used
        """
        query_lower = query.lower()
        
        # PRIORITY: If query contains specific genome IDs, always use genome selection
        if self.extract_genome_request(query):
            extracted = self.extract_genome_request(query)
            if extracted and len(extracted) > 10:  # Likely a specific genome ID
                logger.info(f"🎯 Detected specific genome ID '{extracted}' - forcing genome selection")
                return True
        
        # Don't use for comparative queries (more precise patterns)
        comparative_keywords = [
            'compare', 'comparison', 'across genomes', 'between genomes',
            'all genomes', 'which genome', 'what genomes', 'for each genome', 'across all',
            'most genes', 'least genes', 'highest count', 'lowest count',
            'compare across', 'distribution across', 'among genomes'
        ]
        
        # But allow if there's also a specific genome mentioned
        has_comparative = any(keyword in query_lower for keyword in comparative_keywords)
        if has_comparative:
            # Check if this is actually a complex query with both specific and comparative elements
            targeting_found = any(keyword in query_lower for keyword in self.genome_targeting_keywords)
            if targeting_found:
                logger.info("🤔 Query has both comparative and targeting keywords - allowing genome selection")
                return True
            return False
        
        # Don't use for pure listing queries  
        listing_keywords = [
            'list genomes', 'show genomes', 'how many genomes',
            'genomes in the database'
        ]
        
        # Be more specific about listing exclusions
        if any(keyword in query_lower for keyword in listing_keywords):
            # But allow if there's also specific targeting
            targeting_found = any(keyword in query_lower for keyword in self.genome_targeting_keywords)
            if targeting_found:
                logger.info("🤔 Query mentions genome listing but also has targeting - allowing genome selection")
                return True
            return False
        
        # SPECIAL CASE: "available genomes" in context of analysis (not pure listing)
        if 'available genomes' in query_lower:
            # Allow if the query is asking to analyze something, not just list
            analysis_keywords = ['analyze', 'analysis', 'novel', 'loci', 'interesting', 'annotations for']
            if any(kw in query_lower for kw in analysis_keywords):
                logger.info("🎯 'available genomes' in analysis context - allowing genome selection")
                return True
            return False
        
        # Use if any genome targeting keywords are present
        return any(keyword in query_lower for keyword in self.genome_targeting_keywords)

def test_genome_selector():
    """Test function for GenomeSelector."""
    from ..query_processor import Neo4jQueryProcessor
    from ..config import LLMConfig
    
    # Create test setup
    config = LLMConfig()
    neo4j_processor = Neo4jQueryProcessor(config)
    selector = GenomeSelector(neo4j_processor)
    
    # Test queries
    test_queries = [
        "Find annotations for the Nomurabacteria MAG",
        "What proteins are in Burkholderiales genome?",
        "Show me BGCs from Acidovorax",
        "Compare all genomes",  # Should not trigger selection
        "List available genomes"  # Should not trigger selection
    ]
    
    print("=== Genome Selection Test ===")
    
    async def run_tests():
        try:
            # Test availability check
            genomes = await selector.get_available_genomes()
            print(f"Available genomes: {len(genomes)}")
            if genomes:
                print(f"Sample genomes: {genomes[:3]}")
            
            # Test selection for each query
            for query in test_queries:
                print(f"\nQuery: {query}")
                
                should_select = selector.should_use_genome_selection(query)
                print(f"Should use selection: {should_select}")
                
                if should_select:
                    result = await selector.select_genome(query)
                    print(f"Selection result: {result}")
                
        except Exception as e:
            print(f"Test failed: {e}")
        finally:
            neo4j_processor.close()
    
    import asyncio
    asyncio.run(run_tests())

if __name__ == "__main__":
    test_genome_selector()