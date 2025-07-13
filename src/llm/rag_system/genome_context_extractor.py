"""
Genome Context Extractor for identifying specific genome/organism mentions in task descriptions.

Extracts genome names, organism identifiers, and strain references from natural language
to enable proper genome filtering in database queries.
"""

import re
import logging
from typing import Optional, List, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GenomeContext:
    """Context about genome-specific filtering requirements."""
    genome_name: str
    confidence: float
    extraction_method: str
    original_text: str

class GenomeContextExtractor:
    """Extracts genome/organism context from task descriptions and user queries."""
    
    def __init__(self):
        # Common patterns for genome/organism mentions
        self.genome_patterns = [
            # Direct genome references
            r'(?:genome|MAG|assembly|strain|isolate)\s+(?:of\s+)?([A-Za-z][A-Za-z0-9_\-\.]+)',
            r'([A-Za-z][A-Za-z0-9_\-\.]+)\s+(?:genome|MAG|assembly|strain|isolate)',
            
            # "the [organism] genome" pattern  
            r'the\s+([A-Za-z][A-Za-z0-9_\-\.]+)\s+(?:genome|MAG|assembly)',
            
            # "annotations for [organism]" pattern
            r'annotations?\s+for\s+(?:the\s+)?([A-Za-z][A-Za-z0-9_\-\.]+)',
            
            # "in [organism]" pattern
            r'in\s+(?:the\s+)?([A-Za-z][A-Za-z0-9_\-\.]+)(?:\s+genome|\s+MAG|\s+assembly)?',
            
            # "[organism] proteins/genes" pattern
            r'([A-Za-z][A-Za-z0-9_\-\.]+)\s+(?:proteins?|genes?|loci|sequences?)',
            
            # Scientific naming patterns
            r'(?:Candidatus\s+)?([A-Z][a-z]+(?:bacteria|coccus|bacter|archaea)(?:[a-z]*)?)',
            
            # Taxonomic group references
            r'([A-Za-z]+ales|[A-Za-z]+aceae|[A-Za-z]+idae)\s+(?:genome|MAG|bacterium)',
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.genome_patterns]
        
        # Common exclusions (generic terms that aren't specific genomes)
        self.exclusions = {
            'genome', 'mag', 'assembly', 'strain', 'isolate', 'bacterium', 'bacteria',
            'protein', 'gene', 'sequence', 'annotation', 'locus', 'loci', 'species',
            'organism', 'microbe', 'prokaryote', 'eukaryote', 'archaea', 'data',
            'dataset', 'database', 'all', 'each', 'every', 'multiple', 'various',
            'across', 'between', 'among', 'within', 'through', 'what', 'which', 'how',
            'that', 'this', 'these', 'those', 'many', 'most', 'some', 'few', 'several',
            'other', 'another', 'different', 'same', 'similar', 'compare', 'comparison'
        }
    
    def extract_genome_context(self, text: str) -> Optional[GenomeContext]:
        """
        Extract genome context from text.
        
        Args:
            text: Task description or user query
            
        Returns:
            GenomeContext if specific genome mentioned, None otherwise
        """
        if not text:
            return None
            
        text = text.strip()
        logger.debug(f"Extracting genome context from: {text}")
        
        # Try each pattern in order of specificity
        for i, pattern in enumerate(self.compiled_patterns):
            matches = pattern.findall(text)
            
            for match in matches:
                # Clean and validate the match
                candidate = self._clean_candidate(match)
                
                if self._is_valid_genome_name(candidate):
                    confidence = self._calculate_confidence(candidate, text, i)
                    
                    return GenomeContext(
                        genome_name=candidate.lower(),
                        confidence=confidence,
                        extraction_method=f"pattern_{i}",
                        original_text=text
                    )
        
        logger.debug("No specific genome context found")
        return None
    
    def _clean_candidate(self, candidate: str) -> str:
        """Clean and normalize extracted candidate."""
        if isinstance(candidate, tuple):
            candidate = candidate[0] if candidate else ""
            
        # Remove common prefixes/suffixes
        candidate = re.sub(r'^(the|a|an)\s+', '', candidate, flags=re.IGNORECASE)
        candidate = re.sub(r'\s+(genome|mag|assembly|strain)$', '', candidate, flags=re.IGNORECASE)
        
        # Normalize whitespace and special characters
        candidate = re.sub(r'\s+', '_', candidate.strip())
        candidate = re.sub(r'[^\w\-\.]', '_', candidate)
        
        return candidate
    
    def _is_valid_genome_name(self, candidate: str) -> bool:
        """Check if candidate is a valid genome name."""
        if not candidate or len(candidate) < 3:
            return False
            
        # Check against exclusions
        if candidate.lower() in self.exclusions:
            return False
            
        # Must start with letter
        if not candidate[0].isalpha():
            return False
            
        # Should have some length and complexity
        if len(candidate) < 4 and not any(c.isupper() for c in candidate):
            return False
            
        return True
    
    def _calculate_confidence(self, candidate: str, text: str, pattern_index: int) -> float:
        """Calculate confidence score for extracted genome name."""
        confidence = 0.5  # Base confidence
        
        # Higher confidence for earlier (more specific) patterns
        confidence += (len(self.compiled_patterns) - pattern_index) * 0.05
        
        # Boost for scientific naming patterns
        if re.match(r'[A-Z][a-z]+[a-z]+', candidate):
            confidence += 0.2
            
        # Boost for common genome name patterns
        if any(suffix in candidate.lower() for suffix in ['bacteria', 'bacterium', 'coccus']):
            confidence += 0.15
            
        # Boost for explicit genome references in context
        if any(term in text.lower() for term in ['genome', 'mag', 'assembly']):
            confidence += 0.1
            
        # Cap at 1.0
        return min(confidence, 1.0)
    
    def extract_multiple_genomes(self, text: str) -> List[GenomeContext]:
        """Extract multiple genome contexts from text."""
        contexts = []
        
        # Look for "and", "or", comma-separated genome lists
        # For now, just return single best match
        context = self.extract_genome_context(text)
        if context:
            contexts.append(context)
            
        return contexts
    
    def should_filter_by_genome(self, text: str) -> bool:
        """Check if text requires genome-specific filtering."""
        text_lower = text.lower()
        
        # Don't filter for comparative queries (stronger detection)
        comparative_keywords = [
            'compare', 'comparison', 'across genomes', 'between genomes',
            'all genomes', 'which genome', 'what genomes', 'most', 'least', 
            'highest', 'lowest', 'for each genome', 'across all',
            'compare across', 'distribution across', 'among genomes'
        ]
        
        if any(keyword in text_lower for keyword in comparative_keywords):
            return False
        
        # Don't filter for listing/overview queries
        listing_keywords = [
            'what genomes', 'list genomes', 'show genomes', 'genomes are',
            'genomes in the database', 'how many genomes'
        ]
        
        if any(keyword in text_lower for keyword in listing_keywords):
            return False
            
        # Check if specific genome is mentioned
        context = self.extract_genome_context(text)
        return context is not None and context.confidence > 0.7  # Higher confidence threshold