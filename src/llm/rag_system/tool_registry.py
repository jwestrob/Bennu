"""
Tool Registry for dynamic agent planning.

Defines metadata and eligibility requirements for all available tools
without instance-specific biology or URIs. Provides clean separation
between tool capabilities and execution logic.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel

from .models import Guard

logger = logging.getLogger(__name__)


@dataclass
class ToolDescriptor:
    """Metadata descriptor for an available tool."""
    name: str
    cost_tag: str  # "cheap", "moderate", "expensive"
    description: str
    input_model: Optional[BaseModel] = None  # Future: Pydantic input validation
    output_model: Optional[BaseModel] = None  # Future: Pydantic output validation
    eligibility: List[Guard] = None  # Default eligibility requirements
    capabilities: List[str] = None  # Tool capabilities for dependency resolution
    
    def __post_init__(self):
        if self.eligibility is None:
            self.eligibility = []
        if self.capabilities is None:
            self.capabilities = []


class ToolRegistry:
    """
    Registry of available tools with metadata and eligibility requirements.
    
    Provides clean separation between tool definitions and execution,
    enabling dynamic planning without hard-coded tool knowledge.
    """
    
    def __init__(self):
        """Initialize tool registry with available tool definitions."""
        self._tools: Dict[str, ToolDescriptor] = {}
        self._register_default_tools()
        logger.info(f"🔧 ToolRegistry initialized with {len(self._tools)} tools")
    
    def _register_default_tools(self):
        """Register default tool set with metadata and eligibility."""
        
        # Database query - cheapest, always eligible
        self.register_tool(ToolDescriptor(
            name="database_query",
            cost_tag="cheap",
            description="Query Neo4j knowledge graph for biological entities",
            eligibility=[],  # Always eligible
            capabilities=["kg_search", "entity_lookup", "relationship_analysis"]
        ))
        
        # Vector search - moderate cost, semantic similarity
        self.register_tool(ToolDescriptor(
            name="vector_search", 
            cost_tag="moderate",
            description="Semantic similarity search using protein embeddings",
            eligibility=[],  # Always eligible for semantic queries
            capabilities=["semantic_search", "similarity_analysis", "embedding_lookup"]
        ))
        
        # Code interpreter - moderate cost, quantitative analysis
        self.register_tool(ToolDescriptor(
            name="code_interpreter",
            cost_tag="moderate", 
            description="Statistical analysis and quantitative assessment",
            eligibility=[],  # Always eligible for analysis tasks
            capabilities=["statistical_analysis", "quantification", "data_processing"]
        ))
        
        # Literature search - moderate cost, external validation
        self.register_tool(ToolDescriptor(
            name="literature_search",
            cost_tag="moderate",
            description="PubMed literature search for biological validation",
            eligibility=[],  # Always eligible for validation
            capabilities=["literature_validation", "external_evidence", "citation_lookup"]
        ))
        
        # Whole genome reader - EXPENSIVE with strict eligibility
        self.register_tool(ToolDescriptor(
            name="whole_genome_reader",
            cost_tag="expensive",
            description="Comprehensive spatial genomic analysis and operon detection",
            eligibility=[
                Guard(name="requires_anchor", args={}),
                Guard(name="requires_inconclusive", args={}),
                Guard(name="cheap_first", args={})
            ],
            capabilities=["spatial_analysis", "genomic_context", "operon_detection", "gene_clustering"]
        ))
    
    def register_tool(self, tool_descriptor: ToolDescriptor):
        """
        Register a new tool in the registry.
        
        Args:
            tool_descriptor: ToolDescriptor with metadata and eligibility
        """
        self._tools[tool_descriptor.name] = tool_descriptor
        logger.debug(f"🔧 Registered tool: {tool_descriptor.name} ({tool_descriptor.cost_tag})")
    
    def get_tool(self, name: str) -> Optional[ToolDescriptor]:
        """
        Get tool descriptor by name.
        
        Args:
            name: Tool name
            
        Returns:
            ToolDescriptor or None if not found
        """
        return self._tools.get(name)
    
    def list_tools(self, cost_filter: Optional[str] = None) -> List[ToolDescriptor]:
        """
        List available tools, optionally filtered by cost.
        
        Args:
            cost_filter: Optional cost filter ("cheap", "moderate", "expensive")
            
        Returns:
            List of matching ToolDescriptor objects
        """
        tools = list(self._tools.values())
        
        if cost_filter:
            tools = [tool for tool in tools if tool.cost_tag == cost_filter]
        
        return sorted(tools, key=lambda t: self._cost_priority(t.cost_tag))
    
    def get_tools_by_capability(self, capability: str) -> List[ToolDescriptor]:
        """
        Get tools that provide a specific capability.
        
        Args:
            capability: Capability name (e.g., "semantic_search", "spatial_analysis")
            
        Returns:
            List of tools providing the capability
        """
        matching_tools = []
        
        for tool in self._tools.values():
            if capability in tool.capabilities:
                matching_tools.append(tool)
        
        return sorted(matching_tools, key=lambda t: self._cost_priority(t.cost_tag))
    
    def get_eligible_tools(self, context: Dict[str, Any]) -> List[ToolDescriptor]:
        """
        Get tools eligible for execution given current context.
        
        NOTE: This method provides tool metadata only. Actual guard evaluation
        is performed by PolicyEngine.evaluate_guard() during execution.
        
        Args:
            context: Execution context for eligibility assessment
            
        Returns:
            List of potentially eligible tools (guards not yet evaluated)
        """
        # Return all tools - PolicyEngine will evaluate guards during execution
        # This separation keeps registry logic simple and pure
        return list(self._tools.values())
    
    def _cost_priority(self, cost_tag: str) -> int:
        """Get numeric priority for cost ordering (lower = cheaper)."""
        cost_order = {"cheap": 1, "moderate": 2, "expensive": 3}
        return cost_order.get(cost_tag, 999)
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary of registered tools for debugging."""
        summary = {
            "total_tools": len(self._tools),
            "by_cost": {},
            "by_capability": {}
        }
        
        # Count by cost tier
        for tool in self._tools.values():
            cost = tool.cost_tag
            summary["by_cost"][cost] = summary["by_cost"].get(cost, 0) + 1
        
        # Count by capability
        for tool in self._tools.values():
            for capability in tool.capabilities:
                summary["by_capability"][capability] = summary["by_capability"].get(capability, 0) + 1
        
        return summary


# Global tool registry instance 
_tool_registry = None


def get_tool_registry() -> ToolRegistry:
    """Get global tool registry instance."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry