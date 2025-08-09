"""
SchemaResolver for dynamic biological target resolution.

DEPRECATED: This class is now a thin compatibility shim that delegates
to the new schema-locked DetectorRegistry. Use DetectorRegistry directly
for new code. This will be removed in a future version.

Legacy compatibility wrapper only.
"""

import logging
import warnings
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass

from .schema_map import SchemaMap, SchemaEnforcement
from .detector_registry import DetectorRegistry

logger = logging.getLogger(__name__)


@dataclass
class ResolvedTarget:
    """A biological entity resolved from the knowledge graph."""
    kg_id: str  # Stable KG identifier
    entity_type: str  # protein, pathway, domain, etc.
    name: str  # Primary name
    synonyms: List[str]  # Alternative names
    properties: Dict[str, Any]  # Additional KG properties


class SchemaResolver:
    """
    DEPRECATED: Compatibility shim that delegates to DetectorRegistry.
    
    This class maintains backward compatibility while the codebase
    transitions to the new schema-locked detector pipeline.
    
    Use DetectorRegistry directly for new code.
    """
    
    def __init__(self, graph_client, settings=None):
        """
        Initialize resolver with knowledge graph access.
        
        Args:
            graph_client: Neo4j query processor or compatible interface
            settings: Optional Settings instance for configuration
        """
        warnings.warn(
            "SchemaResolver is deprecated. Use DetectorRegistry directly.",
            DeprecationWarning,
            stacklevel=2
        )
        
        self.graph_client = graph_client
        self.settings = settings
        
        # Initialize new schema-locked components
        self.schema_map = None
        self.detector_registry = None
        
        logger.warning("⚠️  SchemaResolver is deprecated, using compatibility shim")
    
    async def resolve_targets_from_query(self, query: str) -> Dict[str, List[ResolvedTarget]]:
        """
        DEPRECATED: Legacy compatibility method that delegates to DetectorRegistry.
        
        Args:
            query: User's natural language query
            
        Returns:
            Dict mapping entity types to lists of resolved targets (legacy format)
        """
        try:
            # Initialize components if needed
            await self._ensure_components_initialized()
            
            # Use new detector registry to resolve biological concepts
            # For backward compatibility, we'll extract potential concepts from the query
            concepts = self._extract_legacy_concepts(query)
            
            all_targets = {"proteins": [], "pathways": [], "domains": [], "organisms": [], "functions": []}
            
            for concept in concepts:
                detector_result = await self.detector_registry.resolve(concept)
                
                # Convert DetectorResult to legacy ResolvedTarget format
                for ko_id in detector_result.ko_ids:
                    target = ResolvedTarget(
                        kg_id=ko_id,
                        entity_type="function",
                        name=ko_id,  # Simplified for compatibility
                        synonyms=[],
                        properties={"detector_source": "ko"}
                    )
                    all_targets["functions"].append(target)
                
                for pfam_id in detector_result.pfam_ids:
                    target = ResolvedTarget(
                        kg_id=pfam_id,
                        entity_type="domain",
                        name=pfam_id,  # Simplified for compatibility
                        synonyms=[],
                        properties={"detector_source": "pfam"}
                    )
                    all_targets["domains"].append(target)
            
            # Remove duplicates
            for entity_type in all_targets:
                all_targets[entity_type] = self._deduplicate_targets(all_targets[entity_type])
            
            total_targets = sum(len(targets) for targets in all_targets.values())
            logger.info(f"📊 Legacy compatibility: {total_targets} targets via DetectorRegistry")
            
            return all_targets
            
        except Exception as e:
            logger.error(f"❌ Error in legacy compatibility layer: {e}")
            return {"proteins": [], "pathways": [], "domains": [], "organisms": [], "functions": []}
    
    def has_anchor_entities(self, targets: Dict[str, List[ResolvedTarget]]) -> bool:
        """
        DEPRECATED: Legacy compatibility method.
        
        Args:
            targets: Resolved targets from resolve_targets_from_query
            
        Returns:
            True if anchor entities exist for spatial/contextual analysis
        """
        try:
            # Count anchoring entity types that enable spatial analysis
            anchor_types = ["proteins", "domains", "functions"]
            anchor_count = 0
            
            for entity_type in anchor_types:
                if entity_type in targets and targets[entity_type]:
                    anchor_count += len(targets[entity_type])
            
            has_anchors = anchor_count > 0
            logger.debug(f"🎯 Legacy anchor assessment: {anchor_count} entities, sufficient={has_anchors}")
            return has_anchors
            
        except Exception as e:
            logger.warning(f"Error in legacy anchor assessment: {e}")
            return False
    
    async def _ensure_components_initialized(self):
        """Initialize schema-locked components for compatibility layer."""
        if self.schema_map is None:
            self.schema_map = SchemaMap.from_bulk_loader(enforcement=SchemaEnforcement.WARN)
            await self.schema_map.verify_against_db(self.graph_client)
            self.detector_registry = DetectorRegistry(self.graph_client, self.schema_map)
    
    def _extract_legacy_concepts(self, query: str) -> List[str]:
        """
        Simple concept extraction for legacy compatibility.
        
        Args:
            query: Raw user query
            
        Returns:
            List of potential biological concepts
        """
        try:
            # Simple word extraction for compatibility
            words = []
            for word in query.lower().split():
                # Keep words that might be biological terms
                if len(word) > 3 and word not in {
                    'what', 'where', 'when', 'how', 'why', 'which', 'that', 'this', 'these', 'those',
                    'find', 'show', 'tell', 'about', 'proteins', 'genes', 'analysis'
                }:
                    # Clean word
                    clean_word = re.sub(r'[^a-z0-9_-]', '', word)
                    if clean_word:
                        words.append(clean_word)
            
            return words[:5]  # Limit to first 5 potential concepts
            
        except Exception as e:
            logger.warning(f"Error extracting legacy concepts: {e}")
            return []
    
    def _deduplicate_targets(self, targets: List[ResolvedTarget]) -> List[ResolvedTarget]:
        """Remove duplicate targets based on KG ID."""
        seen_ids = set()
        unique_targets = []
        
        for target in targets:
            if target.kg_id not in seen_ids:
                unique_targets.append(target)
                seen_ids.add(target.kg_id)
        
        return unique_targets