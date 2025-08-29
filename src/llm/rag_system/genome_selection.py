"""
Unified Genome Selection System

Combines LLM-based intent analysis, pattern matching, and query scoping
into a single, robust genome selection and filtering system.

Replaces:
- genome_selector.py (explicit genome lookup)
- genome_scoping.py (query modification)  
- llm_genome_selector.py (LLM-based analysis)
"""

import logging
import re
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GenomeSelectionResult:
    """Unified result from genome selection analysis."""
    success: bool
    intent: str  # "specific", "comparative", "global", "ambiguous", "error"
    selected_genome: Optional[str] = None
    target_genomes: List[str] = None
    match_score: float = 0.0
    match_reason: str = ""
    reasoning: str = ""
    confidence: float = 0.0
    available_genomes: List[str] = None
    error_message: str = ""
    suggestions: List[str] = None
    scope_applied: bool = False
    

@dataclass
class GenomeScope:
    """Genome scoping information for query modification."""
    genome_id: Optional[str]
    scope_type: str  # "single", "multiple", "all", "unspecified"
    genome_pattern: Optional[str]
    confidence: float
    reasoning: str


class UnifiedGenomeSelector:
    """
    Unified genome selection system combining LLM analysis, pattern matching,
    and query scoping into a single robust interface.
    
    Features:
    - LLM-based intent classification (primary)
    - Regex pattern fallback (when LLM unavailable) 
    - Database lookup with fuzzy matching
    - Intelligent query scoping and Cypher modification
    - Unified caching system
    """
    
    def __init__(self, neo4j_processor, model="gpt-4.1-mini"):
        """
        Initialize unified genome selector.
        
        Args:
            neo4j_processor: Neo4j processor for database queries
            model: LLM model for analysis (default: gpt-4.1-mini)
        """
        self.neo4j_processor = neo4j_processor
        self.model = model
        
        # Unified caching system
        self._cached_genomes = None
        self._cache_timestamp = None
        self._cache_ttl = 300  # 5 minutes
        
        # Initialize model allocator for LLM analysis
        from .memory.model_allocation import get_model_allocator
        self.model_allocator = get_model_allocator()
        
        # Organism aliases for fuzzy matching
        self.organism_aliases = {
            'nomurabacteria': ['candidatus_nomurabacteria', 'plm0', 'rifcsphigho2'],
            'acidovorax': ['acidovorax'],
            'burkholderiales': ['burkholderiales_ord', 'burkholderiales'],
            'od1': ['od1', 'candidate_division_od1'],
            'candidatus': ['candidatus']
        }
        
        # Keywords that indicate specific genome targeting
        self.genome_targeting_keywords = [
            'for the', 'in the', 'from the', 'within the', 'of the',
            'annotations for', 'proteins in', 'genes in', 'domains in',
            'functions in', 'bgcs in', 'cazymes in',
            'chosen genome', 'selected genome', 'target genome'
        ]
        
        # Genome ID patterns (most specific first)
        self.genome_patterns = [
            r"(PLM0_[A-Za-z0-9_\-\.]+)",  # PLM0_60_b1_sep16_Maxbin2_047_curated
            r"(GCF_[A-Za-z0-9_\-\.]+)",  # NCBI RefSeq
            r"(GCA_[A-Za-z0-9_\-\.]+)",  # NCBI GenBank
            r"(OD1_[A-Za-z0-9_\-\.]+)",  # OD1 genomes
            r"(RIFCSPHIGHO2_[A-Za-z0-9_\-\.]+)",  # RIFCSPHIGHO2 genomes
            r"(Acidovorax_[A-Za-z0-9_\-\.]+)",  # Acidovorax genomes
            r"in\s+genome\s+([A-Za-z0-9_\-\.]{10,})",
            r"from\s+genome\s+([A-Za-z0-9_\-\.]{10,})",
            r"([A-Za-z0-9_\-\.]{10,})\s+\(genome",
            r"of\s+([A-Za-z0-9_\-\.]{10,})\s+\(genome",
            r"in\s+([A-Z][a-z]+\s+[a-z]+)",  # "in Escherichia coli"
            r"from\s+([A-Z][a-z]+\s+[a-z]+)",  # "from Bacillus subtilis"
        ]
        
        # Multi-genome patterns
        self.multi_genome_patterns = [
            r"across\s+genomes?",
            r"between\s+genomes?", 
            r"all\s+genomes?",
            r"multiple\s+genomes?",
            r"compare\s+genomes?",
            r"genomes?\s+in\s+dataset"
        ]
        
        # Known genome IDs (will be updated from database)
        self.known_genome_ids = []
        
        logger.info("🧬 Unified genome selector initialized")
    
    async def get_available_genomes(self, force_refresh: bool = False) -> List[str]:
        """
        Get list of all available genomes with unified caching.
        
        Args:
            force_refresh: Force refresh of cached genome list
            
        Returns:
            List of genome IDs available in the database
        """
        # Check cache validity
        if (not force_refresh and self._cached_genomes and 
            self._cache_timestamp and (time.time() - self._cache_timestamp) < self._cache_ttl):
            return self._cached_genomes
        
        try:
            # Query genomes using both possible field names
            cypher = """
            MATCH (g:Genome) 
            RETURN COALESCE(g.id, g.genomeId) as genome_id 
            ORDER BY genome_id
            """
            result = await self.neo4j_processor._execute_cypher(cypher)
            
            genome_ids = [record['genome_id'] for record in result if record.get('genome_id')]
            
            # Update unified cache
            self._cached_genomes = genome_ids
            self._cache_timestamp = time.time()
            self.known_genome_ids = genome_ids  # Update known IDs
            
            logger.info(f"📊 Retrieved {len(genome_ids)} available genomes from database")
            return genome_ids
            
        except Exception as e:
            logger.error(f"Failed to retrieve available genomes: {e}")
            return []
    
    def should_use_genome_selection(self, query: str) -> bool:
        """
        Determine if query requires genome selection analysis.
        
        Args:
            query: User query text
            
        Returns:
            True if genome selection analysis should be performed
        """
        query_lower = query.lower()
        
        # Skip for obvious global/comparative queries
        obvious_global_patterns = [
            'read through everything', 'analyze everything', 'scan everything',
            'across all genomes', 'all genomes', 'every genome', 'compare all',
            'global analysis', 'pan-genome', 'dataset-wide', 'compare metabolic',
            'analyze all genomes'
        ]
        
        if any(pattern in query_lower for pattern in obvious_global_patterns):
            logger.info("🌐 Obvious global analysis pattern detected - skipping genome selection")
            return False
        
        # Skip for obvious listing queries
        obvious_listing_patterns = [
            'list genomes', 'show genomes', 'how many genomes',
            'what genomes are available', 'genomes in the database'
        ]
        
        if any(pattern in query_lower for pattern in obvious_listing_patterns):
            logger.info("📝 Obvious listing query detected - skipping genome selection")
            return False
        
        # Check for specific genome mentions or targeting keywords
        if any(keyword in query_lower for keyword in self.genome_targeting_keywords):
            return True
            
        # Check for genome ID patterns
        for pattern in self.genome_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                logger.info("🎯 Genome ID pattern detected - using genome selection")
                return True
        
        return False
    
    async def analyze_genome_intent(self, query: str) -> GenomeSelectionResult:
        """
        Analyze genome selection intent using LLM (primary) or patterns (fallback).
        
        Args:
            query: User query to analyze
            
        Returns:
            GenomeSelectionResult with intent classification and target genomes
        """
        logger.info(f"🧠 Analyzing genome intent: {query[:50]}...")
        
        # Get available genomes
        available_genomes = await self.get_available_genomes()
        
        if not available_genomes:
            return GenomeSelectionResult(
                success=False,
                intent="error",
                error_message="No genomes available in database",
                available_genomes=[]
            )
        
        # Try LLM analysis first
        if DSPY_AVAILABLE:
            try:
                result = await self._analyze_with_llm(query, available_genomes)
                logger.info(f"🧠 LLM analysis: intent={result.intent}, genomes={len(result.target_genomes or [])}")
                return result
            except Exception as e:
                logger.warning(f"LLM genome analysis failed: {e}, falling back to patterns")
        
        # Fallback to pattern-based analysis
        return self._analyze_with_patterns(query, available_genomes)
    
    async def _analyze_with_llm(self, query: str, available_genomes: List[str]) -> GenomeSelectionResult:
        """Analyze using LLM structured prompting."""
        
        # Format available genomes for prompt
        genomes_text = "\n".join([f"- {genome}" for genome in available_genomes])
        
        # Call DSPy signature using model allocation
        def analyze_call(module):
            return module(
                query=query,
                available_genomes=genomes_text
            )
        
        from .dspy_signatures import GenomeSelectionSignature
        response = self.model_allocator.create_context_managed_call(
            task_name="biological_interpretation",  # COMPLEX = gpt-5 for biological reasoning
            signature_class=GenomeSelectionSignature,
            module_call_func=analyze_call,
            query=query,
            task_context="Genome selection and biological intent analysis"
        )
        
        # Parse response with fallback handling
        if response:
            intent = getattr(response, 'intent', 'ambiguous')
            target_genomes_str = getattr(response, 'target_genomes', '')
            reasoning = getattr(response, 'reasoning', 'No reasoning provided')
            confidence = float(getattr(response, 'confidence', 0.5))
        else:
            # Fallback if model allocation failed
            logger.warning("LLM analysis failed, using conservative fallback")
            intent = 'global'
            target_genomes_str = ''
            reasoning = 'LLM analysis failed - defaulting to global analysis'
            confidence = 0.7
        
        # Parse and validate target genomes
        target_genomes = self._parse_target_genomes(target_genomes_str, available_genomes)
        
        # For single genome intent, try to find the best match
        selected_genome = None
        match_score = 0.0
        match_reason = ""
        
        if intent == "specific" and target_genomes:
            selected_genome = target_genomes[0]
            match_score = confidence
            match_reason = "LLM selection"
        elif intent == "specific" and not target_genomes:
            # LLM said specific but no valid genomes - try pattern matching
            pattern_result = self._analyze_with_patterns(query, available_genomes)
            if pattern_result.success and pattern_result.selected_genome:
                selected_genome = pattern_result.selected_genome
                target_genomes = [selected_genome]
                match_score = pattern_result.match_score
                match_reason = f"Pattern fallback: {pattern_result.match_reason}"
        
        return GenomeSelectionResult(
            success=True,
            intent=intent,
            selected_genome=selected_genome,
            target_genomes=target_genomes or [],
            match_score=match_score,
            match_reason=match_reason,
            reasoning=reasoning,
            confidence=confidence,
            available_genomes=available_genomes
        )
    
    def _analyze_with_patterns(self, query: str, available_genomes: List[str]) -> GenomeSelectionResult:
        """Analyze using pattern matching (fallback when LLM unavailable)."""
        
        query_lower = query.lower()
        
        # Check for multi-genome patterns first
        for pattern in self.multi_genome_patterns:
            if re.search(pattern, query_lower):
                return GenomeSelectionResult(
                    success=True,
                    intent="comparative",
                    target_genomes=[],
                    reasoning=f"Multi-genome pattern detected: {pattern}",
                    confidence=0.8,
                    available_genomes=available_genomes
                )
        
        # Extract genome request from query
        genome_request = self._extract_genome_request(query)
        
        if not genome_request:
            return GenomeSelectionResult(
                success=True,
                intent="global",
                target_genomes=[],
                reasoning="No specific genome pattern detected",
                confidence=0.7,
                available_genomes=available_genomes
            )
        
        # Find matching genomes
        matches = self._find_matching_genomes(genome_request, available_genomes)
        
        if not matches:
            return GenomeSelectionResult(
                success=False,
                intent="ambiguous",
                error_message=f"No genome matches found for '{genome_request}'",
                suggestions=self._suggest_similar_genomes(genome_request, available_genomes),
                available_genomes=available_genomes
            )
        
        # Take best match
        best_match = matches[0]
        
        return GenomeSelectionResult(
            success=True,
            intent="specific",
            selected_genome=best_match['genome_id'],
            target_genomes=[best_match['genome_id']],
            match_score=best_match['match_score'],
            match_reason=best_match['match_reason'],
            reasoning=f"Pattern-based selection: {best_match['match_reason']}",
            confidence=best_match['match_score'],
            available_genomes=available_genomes
        )
    
    def _extract_genome_request(self, query: str) -> Optional[str]:
        """Extract genome name/identifier from user query using patterns."""
        
        # Check for exact genome ID patterns first
        for pattern in self.genome_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                candidate = match.group(1) if match.groups() else match.group(0)
                if len(candidate) > 3:  # Minimum viable length
                    logger.info(f"🎯 Found genome pattern: {candidate}")
                    return candidate
        
        # Look for targeting keywords
        query_lower = query.lower()
        for keyword in self.genome_targeting_keywords:
            if keyword in query_lower:
                parts = query_lower.split(keyword, 1)
                if len(parts) > 1:
                    after_keyword = parts[1].strip()
                    first_word = after_keyword.split()[0] if after_keyword.split() else ""
                    
                    # Clean up common suffixes
                    candidate = re.sub(r'(genome|mag|bacterium|bacteria)$', '', first_word).strip()
                    
                    if len(candidate) >= 3 and not self._is_generic_term(candidate):
                        logger.info(f"🎯 Found genome after keyword '{keyword}': {candidate}")
                        return candidate
        
        return None
    
    def _find_matching_genomes(self, request: str, available_genomes: List[str]) -> List[Dict[str, Any]]:
        """Find genomes matching the request using multiple strategies."""
        
        matches = []
        request_lower = request.lower()
        
        for genome_id in available_genomes:
            genome_lower = genome_id.lower()
            
            # Strategy 1: Exact substring match
            if request_lower in genome_lower:
                score = len(request_lower) / len(genome_lower)
                if len(request_lower) > 5:
                    score += 0.4  # Bonus for longer matches
                else:
                    score += 0.3
                    
                matches.append({
                    'genome_id': genome_id,
                    'match_score': score,
                    'match_reason': 'exact_substring'
                })
                continue
            
            # Strategy 2: Check organism aliases
            for canonical, aliases in self.organism_aliases.items():
                if request_lower in aliases or canonical == request_lower:
                    for alias in aliases:
                        if alias in genome_lower:
                            score = len(alias) / len(genome_lower) + 0.25
                            matches.append({
                                'genome_id': genome_id,
                                'match_score': score,
                                'match_reason': f'alias_match_{canonical}'
                            })
                            break
            
            # Strategy 3: Fuzzy string matching
            similarity = SequenceMatcher(None, request_lower, genome_lower).ratio()
            if similarity > 0.4:
                matches.append({
                    'genome_id': genome_id,
                    'match_score': similarity,
                    'match_reason': 'fuzzy_match'
                })
            
            # Strategy 4: Token-based matching
            request_tokens = set(re.split(r'[_\.\-\s]+', request_lower))
            genome_tokens = set(re.split(r'[_\.\-\s]+', genome_lower))
            
            if request_tokens and genome_tokens:
                overlap = len(request_tokens.intersection(genome_tokens))
                if overlap > 0:
                    token_score = overlap / len(request_tokens.union(genome_tokens))
                    if token_score > 0.3:
                        matches.append({
                            'genome_id': genome_id,
                            'match_score': token_score,
                            'match_reason': 'token_match'
                        })
        
        # Remove duplicates and sort by score
        unique_matches = {}
        for match in matches:
            genome_id = match['genome_id']
            if genome_id not in unique_matches or match['match_score'] > unique_matches[genome_id]['match_score']:
                unique_matches[genome_id] = match
        
        sorted_matches = sorted(unique_matches.values(), key=lambda x: x['match_score'], reverse=True)
        
        logger.debug(f"Found {len(sorted_matches)} genome matches for '{request}'")
        for match in sorted_matches[:3]:
            logger.debug(f"  {match['genome_id']}: {match['match_score']:.2f} ({match['match_reason']})")
        
        return sorted_matches
    
    def _parse_target_genomes(self, target_genomes_str: str, available_genomes: List[str]) -> List[str]:
        """Parse target genomes string and validate against available genomes."""
        
        if not target_genomes_str or target_genomes_str.strip() == "":
            return []
        
        # Split by commas and clean up
        candidates = [g.strip() for g in target_genomes_str.split(',') if g.strip()]
        
        # Validate against available genomes with fuzzy matching
        valid_genomes = []
        for candidate in candidates:
            # Try exact match first
            if candidate in available_genomes:
                valid_genomes.append(candidate)
                continue
                
            # Try fuzzy matching
            matches = self._find_matching_genomes(candidate, available_genomes)
            if matches and matches[0]['match_score'] > 0.5:
                valid_genomes.append(matches[0]['genome_id'])
        
        return valid_genomes
    
    def _suggest_similar_genomes(self, request: str, available_genomes: List[str], limit: int = 5) -> List[str]:
        """Suggest similar genomes when no good match is found."""
        
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
    
    def _is_generic_term(self, term: str) -> bool:
        """Check if term is too generic to be a specific genome identifier."""
        generic_terms = {
            'genome', 'mag', 'assembly', 'bacterium', 'bacteria', 'archaea',
            'protein', 'gene', 'domain', 'function', 'annotation', 'data',
            'result', 'analysis', 'comparison', 'study', 'sample', 'sequence'
        }
        return term.lower() in generic_terms
    
    def enforce_genome_scope(self, question: str, cypher_query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Enforce genome scoping in Cypher queries with simplified logic.
        
        Args:
            question: Original user question
            cypher_query: Generated Cypher query
            
        Returns:
            Tuple of (modified_query, scope_metadata)
        """
        logger.info("🎯 Enforcing genome scope in query")
        
        # Quick scope detection using patterns (for speed)
        scope = self._detect_genome_scope_simple(question)
        
        metadata = {
            "genome_scope": scope,
            "original_query": cypher_query,
            "scope_applied": False,
            "scope_reasoning": scope.reasoning
        }
        
        # Apply scoping based on detected scope
        if scope.scope_type == "single" and scope.genome_id:
            modified_query = self._apply_simple_genome_scope(cypher_query, scope.genome_id)
            metadata["scope_applied"] = True
            metadata["scope_genome"] = scope.genome_id
            logger.info(f"✅ Applied single genome scope: {scope.genome_id}")
            return modified_query, metadata
        
        elif scope.scope_type == "multiple":
            # For multi-genome queries, add genome grouping
            modified_query = self._ensure_genome_grouping(cypher_query)
            metadata["scope_applied"] = True
            metadata["scope_type"] = "multiple"
            logger.info("✅ Applied multi-genome scope")
            return modified_query, metadata
        
        # No scope changes needed
        logger.info("ℹ️ No genome scope changes applied")
        return cypher_query, metadata
    
    def _detect_genome_scope_simple(self, question: str) -> GenomeScope:
        """Fast genome scope detection using patterns only."""
        
        question_lower = question.lower()
        
        # Check for multi-genome patterns
        for pattern in self.multi_genome_patterns:
            if re.search(pattern, question_lower):
                return GenomeScope(
                    genome_id=None,
                    scope_type="multiple",
                    genome_pattern=pattern,
                    confidence=0.9,
                    reasoning=f"Multi-genome pattern: {pattern}"
                )
        
        # Check for specific genome mentions
        for pattern in self.genome_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                genome_candidate = match.group(1) if match.groups() else match.group(0)
                
                # Quick validation against known genomes
                matched_genome = self._quick_genome_match(genome_candidate)
                
                return GenomeScope(
                    genome_id=matched_genome or genome_candidate,
                    scope_type="single",
                    genome_pattern=pattern,
                    confidence=0.9 if matched_genome else 0.6,
                    reasoning=f"Specific genome detected: {matched_genome or genome_candidate}"
                )
        
        # Default: unspecified scope
        return GenomeScope(
            genome_id=None,
            scope_type="unspecified",
            genome_pattern=None,
            confidence=0.5,
            reasoning="No explicit genome scope detected"
        )
    
    def _quick_genome_match(self, candidate: str) -> Optional[str]:
        """Quick genome matching against known IDs."""
        
        if not self.known_genome_ids:
            return None
            
        candidate_clean = candidate.strip()
        
        # Try exact matches (case insensitive)
        for genome_id in self.known_genome_ids:
            if candidate_clean.lower() == genome_id.lower():
                return genome_id
        
        # Try prefix matching with common suffixes
        common_suffixes = ['_contigs', '.contigs', '_scaffolds', '.scaffolds']
        
        for genome_id in self.known_genome_ids:
            for suffix in common_suffixes:
                if genome_id.endswith(suffix):
                    genome_base = genome_id[:-len(suffix)]
                    if candidate_clean.lower() == genome_base.lower():
                        return genome_id
        
        return None
    
    def _apply_simple_genome_scope(self, cypher_query: str, genome_id: str) -> str:
        """Apply simple genome scoping with minimal query modification."""
        
        # Strategy 1: Replace existing genome constraints
        # Pattern: {genomeId:'value'} or {id:'value'}
        genome_constraint_pattern = r"\{(?:genomeId|id)\s*:\s*['\"]([^'\"]+)['\"]\}"
        match = re.search(genome_constraint_pattern, cypher_query, re.IGNORECASE)
        
        if match:
            old_constraint = match.group(0)
            new_constraint = f"{{id:'{genome_id}'}}"
            modified_query = cypher_query.replace(old_constraint, new_constraint)
            logger.info(f"🔧 Replaced genome constraint: {old_constraint} → {new_constraint}")
            return modified_query
        
        # Strategy 2: Add genome filter to existing WHERE clause
        if "WHERE" in cypher_query.upper():
            where_pos = cypher_query.upper().find("WHERE")
            insert_pos = cypher_query.find(" ", where_pos + 5)
            
            # Find genome variable in the query
            genome_var = self._find_genome_variable(cypher_query)
            if genome_var:
                filter_clause = f"{genome_var}.id = '{genome_id}'"
                modified_query = (
                    cypher_query[:insert_pos] + 
                    f" {filter_clause} AND " + 
                    cypher_query[insert_pos:].lstrip()
                )
                return modified_query
        
        # Strategy 3: Add simple genome constraint at start
        genome_filter = f"MATCH (genome:Genome {{id: '{genome_id}'}}) "
        
        if cypher_query.upper().startswith("MATCH"):
            modified_query = genome_filter + cypher_query
        else:
            modified_query = genome_filter + cypher_query
        
        return modified_query
    
    def _find_genome_variable(self, cypher_query: str) -> Optional[str]:
        """Find genome variable name in Cypher query."""
        
        # Look for patterns like (g:Genome) or (genome:Genome)
        genome_pattern = r"\((\w+):Genome\)"
        match = re.search(genome_pattern, cypher_query, re.IGNORECASE)
        
        if match:
            return match.group(1)
        
        # Look for BELONGSTOGENOME relationships
        belong_pattern = r"-\[:BELONGSTOGENOME\]->\((\w+)"
        match = re.search(belong_pattern, cypher_query, re.IGNORECASE)
        
        if match:
            return match.group(1)
        
        return "genome"  # Default fallback
    
    def _ensure_genome_grouping(self, cypher_query: str) -> str:
        """Ensure multi-genome queries include genome grouping."""
        
        # Add genome to RETURN clause if not present
        if "RETURN" in cypher_query.upper() and "genome" not in cypher_query.lower():
            return_pos = cypher_query.upper().find("RETURN")
            return_clause = cypher_query[return_pos:]
            
            return_match = re.search(r"RETURN\s+(.+)", return_clause, re.IGNORECASE)
            if return_match:
                existing_return = return_match.group(1)
                modified_return = f"RETURN genome.id as genome_id, {existing_return}"
                modified_query = cypher_query[:return_pos] + modified_return
                return modified_query
        
        return cypher_query


# DSPy Signature imported from dspy_signatures.py


# Global instance for easy access
_unified_selector = None

def get_genome_selector() -> UnifiedGenomeSelector:
    """Get the global unified genome selector instance."""
    global _unified_selector
    if _unified_selector is None:
        # Initialize with placeholder - will be set properly by calling code
        _unified_selector = UnifiedGenomeSelector(None)
    return _unified_selector

def set_genome_selector(neo4j_processor) -> UnifiedGenomeSelector:
    """Set the global genome selector with proper Neo4j processor."""
    global _unified_selector
    _unified_selector = UnifiedGenomeSelector(neo4j_processor)
    return _unified_selector


# Test function
async def test_unified_genome_selector():
    """Test the unified genome selector with various query types."""
    from ..query_processor import Neo4jQueryProcessor
    from ..config import LLMConfig
    
    config = LLMConfig()
    neo4j_processor = Neo4jQueryProcessor(config)
    selector = UnifiedGenomeSelector(neo4j_processor)
    
    test_queries = [
        "Find proteins in the Nomurabacteria genome",  # Should be specific
        "Compare metabolic capabilities across all genomes",  # Should be comparative
        "read through everything directly and see what you can find",  # Should be global
        "Show me BGCs from PLM0_60_b1_sep16_Maxbin2_047_curated",  # Should be specific
        "What transport proteins are there?",  # Should be global
        "List available genomes",  # Should skip selection
    ]
    
    print("=== Unified Genome Selector Test ===")
    
    try:
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            
            should_analyze = selector.should_use_genome_selection(query)
            print(f"   Should analyze: {should_analyze}")
            
            if should_analyze:
                result = await selector.analyze_genome_intent(query)
                print(f"   Intent: {result.intent}")
                print(f"   Selected: {result.selected_genome}")
                print(f"   Targets: {result.target_genomes}")
                print(f"   Reasoning: {result.reasoning}")
                print(f"   Confidence: {result.confidence:.2f}")
            else:
                print(f"   Skipped analysis (obvious pattern)")
                
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        neo4j_processor.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_unified_genome_selector())
