"""
Registry of known error patterns and repair strategies for TaskRepairAgent.
"""

import re
from typing import List, Dict
from .repair_types import ErrorPattern, RepairStrategy


class ErrorPatternRegistry:
    """Registry of known error patterns and their repair strategies"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> List[ErrorPattern]:
        """Initialize the default set of error patterns"""
        return [
            # DSPy Comment Query Pattern
            ErrorPattern(
                pattern_type="comment_query",
                regex_pattern=r"/\*.*?(?:label|entity).*?(?:`(\w+)`|'(\w+)').*?(?:not part of|doesn't exist).*?schema.*?\*/",
                repair_strategy=RepairStrategy.COMMENT_QUERY_EXPLANATION,
                confidence_threshold=0.9,
                description="DSPy generates explanatory comments instead of valid Cypher"
            ),
            
            # Invalid Node Label Pattern
            ErrorPattern(
                pattern_type="invalid_node_label",
                regex_pattern=r"Variable `(\w+)` not defined|Invalid input.*?expected.*?MATCH",
                repair_strategy=RepairStrategy.INVALID_ENTITY_SUGGESTION,
                confidence_threshold=0.8,
                description="Query references non-existent node labels"
            ),
            
            # Invalid Relationship Pattern  
            ErrorPattern(
                pattern_type="invalid_relationship",
                regex_pattern=r"NONEXISTENT_RELATIONSHIP|CONNECTS_TO|LINKS_TO",
                repair_strategy=RepairStrategy.RELATIONSHIP_MAPPING,
                confidence_threshold=0.85,
                description="Query uses non-existent relationship types"
            ),
            
            # Neo4j Syntax Error Pattern
            ErrorPattern(
                pattern_type="neo4j_syntax_error",
                regex_pattern=r"Neo\.ClientError\.Statement\.SyntaxError",
                repair_strategy=RepairStrategy.SCHEMA_VALIDATION,
                confidence_threshold=0.7,
                description="General Neo4j syntax errors"
            )
        ]
    
    def find_matching_patterns(self, query: str, error_message: str) -> List[ErrorPattern]:
        """Find all patterns that match the given query and error"""
        matching_patterns = []
        
        # Check query text
        for pattern in self.patterns:
            if pattern.matches(query):
                matching_patterns.append(pattern)
        
        # Check error message
        for pattern in self.patterns:
            if pattern.matches(error_message):
                matching_patterns.append(pattern)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_patterns = []
        for pattern in matching_patterns:
            if pattern.pattern_type not in seen:
                unique_patterns.append(pattern)
                seen.add(pattern.pattern_type)
        
        return unique_patterns
    
    def get_pattern_by_type(self, pattern_type: str) -> ErrorPattern:
        """Get a specific pattern by type"""
        for pattern in self.patterns:
            if pattern.pattern_type == pattern_type:
                return pattern
        raise ValueError(f"Pattern type '{pattern_type}' not found")


class RelationshipMapper:
    """Maps invalid relationship names to valid ones"""
    
    RELATIONSHIP_MAPPINGS = {
        "NONEXISTENT_RELATIONSHIP": "HASDOMAIN",
        "CONNECTS_TO": "ENCODEDBY", 
        "LINKS_TO": "HASDOMAIN",
        "ASSOCIATED_WITH": "HASFUNCTION",
        "BELONGS_TO": "ENCODEDBY",
        "CONTAINS": "HASDOMAIN"
    }
    
    @classmethod
    def map_relationship(cls, invalid_relationship: str) -> str:
        """Map an invalid relationship to a valid one"""
        return cls.RELATIONSHIP_MAPPINGS.get(invalid_relationship.upper(), "HASDOMAIN")
    
    @classmethod
    def get_valid_relationships(cls) -> List[str]:
        """Get list of all valid relationship types"""
        return ["ENCODEDBY", "HASDOMAIN", "DOMAINFAMILY", "HASFUNCTION"]


class EntitySuggester:
    """Suggests valid entity names for invalid ones"""
    
    @staticmethod
    def suggest_alternatives(invalid_entity: str, valid_entities: List[str], max_suggestions: int = 3) -> List[str]:
        """Suggest valid alternatives using fuzzy string matching"""
        from difflib import get_close_matches
        
        # Try fuzzy matching first
        suggestions = get_close_matches(
            invalid_entity, 
            valid_entities, 
            n=max_suggestions, 
            cutoff=0.3
        )
        
        # If no good matches, return most common entities
        if not suggestions:
            suggestions = valid_entities[:max_suggestions]
        
        return suggestions
    
    @staticmethod
    def extract_entity_from_comment(comment_query: str) -> str:
        """Extract entity name from DSPy comment query"""
        # Look for patterns like `EntityName` or 'EntityName'
        patterns = [
            r"label `(\w+)`",
            r"label '(\w+)'", 
            r"entity `(\w+)`",
            r"entity '(\w+)'",
            r"`(\w+)` is not",
            r"'(\w+)' is not"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, comment_query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "Unknown"