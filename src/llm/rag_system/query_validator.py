"""
Query Validator for ensuring genome filtering is applied when required.

Validates Cypher queries to ensure they include proper genome filtering
when specific genomes are mentioned in the user query.
"""

import re
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of query validation."""
    is_valid: bool
    error_message: Optional[str]
    suggested_fix: Optional[str]
    modified_query: Optional[str]

class QueryValidator:
    """Validates Cypher queries for proper genome filtering."""
    
    def __init__(self):
        # Patterns to identify genome filtering requirements (updated for exact matching)
        self.genome_filter_patterns = [
            r'WHERE\s+genome\.genomeId\s*=\s*[\'"][^\'\"]+[\'"]',  # Exact match pattern (new primary)
            r'WHERE\s+g\.genomeId\s*=\s*[\'"][^\'\"]+[\'"]',  # Exact match with alias
            r'WHERE\s+toLower\(genome\.genomeId\)\s+CONTAINS\s+[\'"][^\'\"]+[\'"]',  # Legacy pattern
            r'WHERE\s+genome\.genomeId\s+CONTAINS\s+[\'"][^\'\"]+[\'"]',  # Legacy pattern
            r'WHERE\s+toLower\(g\.genomeId\)\s+CONTAINS\s+[\'"][^\'\"]+[\'"]',  # Legacy pattern
        ]
        
        # Patterns to identify genome node presence
        self.genome_node_patterns = [
            r'\(\s*genome\s*:\s*Genome\s*\)',
            r'\(\s*g\s*:\s*Genome\s*\)',
        ]
        
        # Patterns for relationships to genome
        self.genome_relationship_patterns = [
            r'-\[:BELONGSTOGENOME\]->\(genome:Genome\)',
            r'-\[:BELONGSTOGENOME\]->\(g:Genome\)',
        ]
    
    def validate_genome_filtering(self, 
                                query: str, 
                                genome_filter_required: bool, 
                                target_genome: str = "") -> ValidationResult:
        """
        Validate that query includes proper genome filtering when required.
        
        Args:
            query: Cypher query to validate
            genome_filter_required: Whether genome filtering is required
            target_genome: Target genome name (if filtering required)
            
        Returns:
            ValidationResult with validation status and suggestions
        """
        query_normalized = self._normalize_query(query)
        
        # If no genome filtering required, query is valid
        if not genome_filter_required:
            return ValidationResult(
                is_valid=True,
                error_message=None,
                suggested_fix=None,
                modified_query=None
            )
        
        # Check if query has genome filtering
        has_genome_filter = self._has_genome_filtering(query_normalized)
        
        if has_genome_filter:
            # Verify the filter target matches expected genome exactly (for explicit genome selection)
            filter_target = self._extract_filter_target(query_normalized)
            if filter_target and target_genome:
                # For explicit genome selection, we require exact match
                if filter_target == target_genome:
                    return ValidationResult(
                        is_valid=True,
                        error_message=None,
                        suggested_fix=None,
                        modified_query=None
                    )
                else:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Query filters for '{filter_target}' but target genome is '{target_genome}'",
                        suggested_fix=f"Update filter to use exact genome ID: '{target_genome}'",
                        modified_query=self._fix_filter_target(query, target_genome)
                    )
            else:
                return ValidationResult(
                    is_valid=True,
                    error_message=None,
                    suggested_fix=None,
                    modified_query=None
                )
        
        # Query is missing required genome filtering
        if not self._has_genome_node(query_normalized):
            return ValidationResult(
                is_valid=False,
                error_message=f"Query requires genome filtering for '{target_genome}' but has no genome node",
                suggested_fix="Add genome node and filtering to query structure",
                modified_query=self._add_genome_filtering(query, target_genome)
            )
        
        # Has genome node but missing filter
        return ValidationResult(
            is_valid=False,
            error_message=f"Query has genome node but missing WHERE clause for '{target_genome}'",
            suggested_fix=f"Add WHERE genome.genomeId = '{target_genome}'",
            modified_query=self._add_where_clause(query, target_genome)
        )
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for pattern matching."""
        # Remove extra whitespace and newlines
        normalized = re.sub(r'\s+', ' ', query.strip())
        return normalized
    
    def _has_genome_filtering(self, query: str) -> bool:
        """Check if query has genome filtering WHERE clause."""
        for pattern in self.genome_filter_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _has_genome_node(self, query: str) -> bool:
        """Check if query includes genome node."""
        for pattern in self.genome_node_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _extract_filter_target(self, query: str) -> Optional[str]:
        """Extract the target genome from filtering clause."""
        for pattern in self.genome_filter_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # Extract the quoted value
                quoted_match = re.search(r'[\'"]([^\'"]+)[\'"]', match.group())
                if quoted_match:
                    return quoted_match.group(1)
        return None
    
    def _fix_filter_target(self, query: str, target_genome: str) -> str:
        """Fix the filter target in an existing WHERE clause."""
        # Replace the quoted value in existing filter with exact genome ID
        def replace_target(match):
            # For exact matching, use the target genome ID as-is
            return re.sub(r'[\'"][^\'"]+[\'"]', f"'{target_genome}'", match.group())
        
        for pattern in self.genome_filter_patterns:
            query = re.sub(pattern, replace_target, query, flags=re.IGNORECASE)
        
        return query
    
    def _add_genome_filtering(self, query: str, target_genome: str) -> str:
        """Add complete genome filtering to query missing genome node."""
        # This is complex - for now, return suggestion rather than automatic fix
        logger.warning(f"Cannot automatically add genome filtering to query without genome node: {query}")
        return query
    
    def _add_where_clause(self, query: str, target_genome: str) -> str:
        """Add WHERE clause to query that has genome node but no filtering."""
        # Look for RETURN statement and add WHERE before it
        where_clause = f"WHERE genome.genomeId = '{target_genome}'"
        
        # Try to insert before RETURN
        if re.search(r'\bRETURN\b', query, re.IGNORECASE):
            return re.sub(
                r'\bRETURN\b', 
                f"{where_clause} RETURN", 
                query, 
                flags=re.IGNORECASE,
                count=1
            )
        
        # Try to insert before ORDER BY
        if re.search(r'\bORDER\s+BY\b', query, re.IGNORECASE):
            return re.sub(
                r'\bORDER\s+BY\b', 
                f"{where_clause} ORDER BY", 
                query, 
                flags=re.IGNORECASE,
                count=1
            )
        
        # Try to insert before LIMIT
        if re.search(r'\bLIMIT\b', query, re.IGNORECASE):
            return re.sub(
                r'\bLIMIT\b', 
                f"{where_clause} LIMIT", 
                query, 
                flags=re.IGNORECASE,
                count=1
            )
        
        # Fallback: append to end
        return f"{query.rstrip()} {where_clause}"
    
    def should_validate_for_genome(self, query: str) -> bool:
        """Check if query should be validated for genome filtering."""
        # Skip validation for certain query types
        skip_patterns = [
            r'MATCH\s+\(genome:Genome\)\s+RETURN\s+genome\.genomeId',  # Genome listing queries
            r'COUNT\s*\(',  # Count queries
            r'SHOW\s+',     # Show commands
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False
        
        return True