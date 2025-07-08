"""
Data structures for TaskRepairAgent error detection and repair.
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class RepairStrategy(Enum):
    """Available repair strategies"""
    COMMENT_QUERY_EXPLANATION = "comment_query_explanation"
    INVALID_ENTITY_SUGGESTION = "invalid_entity_suggestion"
    RELATIONSHIP_MAPPING = "relationship_mapping"
    SCHEMA_VALIDATION = "schema_validation"
    PARAMETER_SUBSTITUTION = "parameter_substitution"
    SYNTAX_ERROR = "syntax_error"
    FALLBACK_MESSAGE = "fallback_message"


@dataclass
class RepairResult:
    """Result of a repair attempt"""
    success: bool
    repaired_query: Optional[str] = None
    user_message: Optional[str] = None
    suggested_alternatives: List[str] = None
    confidence: float = 0.0
    repair_strategy_used: Optional[RepairStrategy] = None
    original_error: Optional[str] = None
    
    def __post_init__(self):
        if self.suggested_alternatives is None:
            self.suggested_alternatives = []


@dataclass
class ErrorPattern:
    """Definition of an error pattern and its repair strategy"""
    pattern_type: str
    regex_pattern: str
    repair_strategy: RepairStrategy
    confidence_threshold: float
    description: str = ""
    
    def matches(self, text: str) -> bool:
        """Check if this pattern matches the given text"""
        import re
        return bool(re.search(self.regex_pattern, text, re.IGNORECASE | re.DOTALL))


@dataclass
class SchemaInfo:
    """Information about the Neo4j database schema"""
    node_labels: List[str]
    relationship_types: List[str]
    node_properties: dict  # {label: [properties]}
    
    @classmethod
    def default_genomic_schema(cls):
        """Default schema for genomic database"""
        return cls(
            node_labels=["Protein", "Gene", "Domain", "KEGGOrtholog", "DomainAnnotation"],
            relationship_types=["ENCODEDBY", "HASDOMAIN", "DOMAINFAMILY", "HASFUNCTION"],
            node_properties={
                "Protein": ["id", "sequence", "length"],
                "Gene": ["startCoordinate", "endCoordinate", "strand"],
                "Domain": ["id", "description", "accession"],
                "KEGGOrtholog": ["id", "description"],
                "DomainAnnotation": ["evalue", "score"]
            }
        )