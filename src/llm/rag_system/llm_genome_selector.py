"""
LLM-based Genome Selection System

Replaces brittle keyword-matching with intelligent natural language understanding
using GPT-4.1-mini to determine genome selection intent and target genomes.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class GenomeSelectionResult:
    """Result of LLM-based genome selection analysis."""
    success: bool
    intent: str  # "specific", "comparative", "global", "ambiguous"
    target_genomes: List[str]
    reasoning: str
    confidence: float
    available_genomes: List[str] = None
    error_message: str = ""
    suggestions: List[str] = None

class LLMGenomeSelector:
    """
    Intelligent genome selection using LLM natural language understanding.
    
    Uses GPT-4.1-mini to analyze user queries and determine:
    1. Whether the query targets specific genome(s) or all genomes
    2. Which specific genomes from the available list (if any)
    3. Confidence and reasoning for the decision
    """
    
    def __init__(self, neo4j_processor, model="gpt-4.1-mini"):
        """
        Initialize LLM-based genome selector.
        
        Args:
            neo4j_processor: Neo4j processor for querying available genomes
            model: LLM model to use for analysis (default: gpt-4.1-mini for cost efficiency)
        """
        self.neo4j_processor = neo4j_processor
        self.model = model
        self._cached_genomes = None
        self._cache_timestamp = None
        
        # Initialize model allocator for intelligent model selection
        from .memory.model_allocation import get_model_allocator
        self.model_allocator = get_model_allocator()
        
        # DSPy modules are instantiated on-demand via model allocation
        # No need for persistent instances
    
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
            
            # Update cache
            self._cached_genomes = genome_ids
            self._cache_timestamp = time.time()
            
            logger.info(f"📊 Retrieved {len(genome_ids)} available genomes from database")
            return genome_ids
            
        except Exception as e:
            logger.error(f"Failed to retrieve available genomes: {e}")
            return []
    
    def should_use_genome_selection(self, query: str) -> bool:
        """
        Determine if query requires genome selection analysis.
        
        This is a lightweight pre-filter to avoid unnecessary LLM calls.
        Most queries will go through LLM analysis, but some obvious cases can be filtered.
        
        Args:
            query: User query text
            
        Returns:
            True if genome selection analysis should be performed
        """
        query_lower = query.lower()
        
        # Skip LLM analysis for obvious global/comparative queries
        obvious_global_patterns = [
            'read through everything', 'analyze everything', 'scan everything',
            'across all genomes', 'all genomes', 'every genome', 'compare all',
            'global analysis', 'pan-genome', 'dataset-wide'
        ]
        
        if any(pattern in query_lower for pattern in obvious_global_patterns):
            logger.info(f"🌐 Obvious global analysis pattern detected - skipping LLM analysis")
            return False
        
        # Skip for obvious listing queries
        obvious_listing_patterns = [
            'list genomes', 'show genomes', 'how many genomes', 
            'what genomes are available', 'genomes in the database'
        ]
        
        if any(pattern in query_lower for pattern in obvious_listing_patterns):
            logger.info(f"📝 Obvious listing query detected - skipping LLM analysis") 
            return False
        
        # For everything else, use LLM analysis
        return True
    
    async def analyze_genome_intent(self, query: str) -> GenomeSelectionResult:
        """
        Use LLM to analyze genome selection intent.
        
        Args:
            query: User query to analyze
            
        Returns:
            GenomeSelectionResult with intent classification and target genomes
        """
        # Get available genomes
        available_genomes = await self.get_available_genomes()
        
        if not available_genomes:
            return GenomeSelectionResult(
                success=False,
                intent="error",
                target_genomes=[],
                reasoning="No genomes available in database",
                confidence=0.0,
                error_message="Database contains no genomes"
            )
        
        # Use DSPy if available, otherwise fallback to direct LLM call
        if self.genome_analyzer and DSPY_AVAILABLE:
            try:
                result = await self._analyze_with_dspy(query, available_genomes)
                logger.info(f"🧠 LLM genome analysis: intent={result.intent}, genomes={len(result.target_genomes)}")
                return result
            except Exception as e:
                logger.warning(f"DSPy genome analysis failed: {e}, falling back to direct LLM")
        
        # Fallback to direct LLM call
        return await self._analyze_with_direct_llm(query, available_genomes)
    
    async def _analyze_with_dspy(self, query: str, available_genomes: List[str]) -> GenomeSelectionResult:
        """Analyze using DSPy structured prompting."""
        
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
            task_name="biological_interpretation",  # COMPLEX = o3 for biological reasoning
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
            logger.warning("Genome analysis model allocation failed, using conservative fallback")
            intent = 'global'
            target_genomes_str = ''
            reasoning = 'Model allocation failed - defaulting to global analysis across all genomes'
            confidence = 0.7
        
        # Parse target genomes list
        target_genomes = self._parse_target_genomes(target_genomes_str, available_genomes)
        
        return GenomeSelectionResult(
            success=True,
            intent=intent,
            target_genomes=target_genomes,
            reasoning=reasoning,
            confidence=confidence,
            available_genomes=available_genomes
        )
    
    async def _analyze_with_direct_llm(self, query: str, available_genomes: List[str]) -> GenomeSelectionResult:
        """Analyze using direct LLM call (fallback when DSPy unavailable)."""
        
        # Format prompt
        genomes_list = "\n".join([f"- {genome}" for genome in available_genomes])
        
        prompt = f"""You are a genomics expert analyzing user queries to determine genome selection intent.

Available genomes in the database:
{genomes_list}

User query: "{query}"

Analyze this query and determine:

1. **Intent Classification:**
   - "specific": Query targets one or more specific genomes
   - "comparative": Query wants to compare across multiple/all genomes  
   - "global": Query wants to analyze all genomes without comparison
   - "ambiguous": Unclear intent, recommend clarification

2. **Target Genomes:**
   - If specific: List the exact genome IDs from available genomes (comma-separated)
   - If comparative/global: Leave empty (analyze all)
   - If ambiguous: Leave empty and suggest clarification

3. **Reasoning:**
   - Explain your decision in 1-2 sentences
   - Note any organism names or genome references you found

4. **Confidence:**
   - Score from 0.0 to 1.0 for your analysis confidence

Return your analysis as JSON:
{{
  "intent": "specific|comparative|global|ambiguous",
  "target_genomes": "comma-separated genome IDs or empty",
  "reasoning": "explanation of decision",
  "confidence": 0.95
}}"""

        try:
            # Import LiteLLM for direct call
            import litellm
            
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            result_data = json.loads(content)
            
            intent = result_data.get('intent', 'ambiguous')
            target_genomes_str = result_data.get('target_genomes', '')
            reasoning = result_data.get('reasoning', 'No reasoning provided')
            confidence = float(result_data.get('confidence', 0.5))
            
            # Parse target genomes
            target_genomes = self._parse_target_genomes(target_genomes_str, available_genomes)
            
            return GenomeSelectionResult(
                success=True,
                intent=intent,
                target_genomes=target_genomes,
                reasoning=reasoning,
                confidence=confidence,
                available_genomes=available_genomes
            )
            
        except Exception as e:
            logger.error(f"Direct LLM genome analysis failed: {e}")
            return GenomeSelectionResult(
                success=False,
                intent="error",
                target_genomes=[],
                reasoning=f"LLM analysis failed: {str(e)}",
                confidence=0.0,
                available_genomes=available_genomes,
                error_message=str(e)
            )
    
    def _parse_target_genomes(self, target_genomes_str: str, available_genomes: List[str]) -> List[str]:
        """Parse target genomes string and validate against available genomes."""
        if not target_genomes_str or target_genomes_str.strip() == "":
            return []
        
        # Split by commas and clean up
        candidates = [g.strip() for g in target_genomes_str.split(',') if g.strip()]
        
        # Validate against available genomes
        valid_genomes = []
        for candidate in candidates:
            # Exact match
            if candidate in available_genomes:
                valid_genomes.append(candidate)
            else:
                # Try to find close match
                for available in available_genomes:
                    if candidate.lower() in available.lower() or available.lower() in candidate.lower():
                        valid_genomes.append(available)
                        break
        
        return valid_genomes

# DSPy Signature for structured genome selection
if DSPY_AVAILABLE:
    class GenomeSelectionSignature(dspy.Signature):
        """
        Analyze user query to determine genome selection intent and target genomes.
        
        Use natural language understanding to classify whether the user wants to:
        1. Analyze specific genome(s) - return their exact IDs
        2. Compare across genomes - return empty list (comparative analysis)
        3. Analyze all genomes globally - return empty list (global analysis)
        4. Ambiguous intent - return empty list and request clarification
        
        Be conservative: if unclear, classify as "global" rather than guessing specific genomes.
        """
        
        query = dspy.InputField(desc="User's natural language query about genomic data")
        available_genomes = dspy.InputField(desc="List of available genome IDs in the database")
        
        intent = dspy.OutputField(desc="Intent classification: 'specific', 'comparative', 'global', or 'ambiguous'")
        target_genomes = dspy.OutputField(desc="Comma-separated exact genome IDs if intent='specific', otherwise empty")
        reasoning = dspy.OutputField(desc="1-2 sentence explanation of the classification decision")
        confidence = dspy.OutputField(desc="Confidence score from 0.0 to 1.0 for the analysis")

# Test function
async def test_llm_genome_selector():
    """Test the LLM genome selector with various query types."""
    from ..query_processor import Neo4jQueryProcessor
    from ..config import LLMConfig
    
    config = LLMConfig()
    neo4j_processor = Neo4jQueryProcessor(config)
    selector = LLMGenomeSelector(neo4j_processor)
    
    test_queries = [
        "Find proteins in the Nomurabacteria genome",  # Should be specific
        "Compare metabolic capabilities across all genomes",  # Should be comparative  
        "read through everything directly and see what you can find",  # Should be global
        "Show me BGCs from Burkholderiales and Nomurabacteria",  # Should be specific (2 genomes)
        "What transport proteins are there?",  # Should be global
        "In the context of prophage discovery: analyze oxidation-reduction proteins",  # Should be global
    ]
    
    print("=== LLM Genome Selector Test ===")
    
    try:
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            
            should_analyze = selector.should_use_genome_selection(query)
            print(f"   Should analyze: {should_analyze}")
            
            if should_analyze:
                result = await selector.analyze_genome_intent(query)
                print(f"   Intent: {result.intent}")
                print(f"   Target genomes: {result.target_genomes}")
                print(f"   Reasoning: {result.reasoning}")
                print(f"   Confidence: {result.confidence:.2f}")
            else:
                print(f"   Skipped LLM analysis (obvious pattern)")
                
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        neo4j_processor.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_llm_genome_selector())