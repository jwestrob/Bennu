#!/usr/bin/env python3
"""
Schema map for enforcing Neo4j database schema contracts.
Loads from bulk-loader schema validator as canonical source of truth.
"""

import logging
from typing import Dict, Set, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel
from pathlib import Path

logger = logging.getLogger(__name__)


class SchemaEnforcement(str, Enum):
    """Schema enforcement modes."""
    STRICT = "strict"  # Fail on any schema violations
    WARN = "warn"      # Warn but continue on violations


@dataclass
class SchemaDriftReport:
    """Report of schema drift between bulk-loader and live database."""
    missing_labels: Set[str]
    missing_relationships: Set[str]
    missing_properties: Dict[str, Set[str]]  # label -> missing props
    extra_labels: Set[str]
    extra_relationships: Set[str] 
    warnings: List[str]
    errors: List[str]
    
    @property
    def has_violations(self) -> bool:
        """Check if there are any schema violations."""
        return bool(
            self.missing_labels or 
            self.missing_relationships or 
            self.missing_properties or
            self.errors
        )


class SchemaMap(BaseModel):
    """
    Canonical schema definition loaded from bulk-loader validator.
    Enforces only labels, properties, and relationships that actually exist.
    """
    # Label -> Set of properties
    labels: Dict[str, Set[str]]
    
    # Set of (src_label, rel_type, dst_label) tuples
    edges: Set[Tuple[str, str, str]]
    
    # Enforcement mode
    enforcement: SchemaEnforcement = SchemaEnforcement.STRICT
    
    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True
    
    @classmethod
    def from_bulk_loader(cls, enforcement: SchemaEnforcement = SchemaEnforcement.STRICT) -> "SchemaMap":
        """
        Build SchemaMap from bulk-loader CSV schema validator.
        This is the canonical source of truth.
        """
        # Import here to avoid circular imports
        from src.build_kg.csv_schema_validator import CSVSchemaValidator
        
        # Create a temporary validator to extract schema
        csv_dir = Path("data/stage07_kg/csv")  # Standard location
        validator = CSVSchemaValidator(csv_dir)
        
        # Extract labels and properties from node schemas
        labels = {}
        for node_schema in validator.node_schemas:
            # Get properties from required headers, filtering out Neo4j-specific ones
            properties = set()
            for header in node_schema.required_headers:
                # Remove Neo4j import annotations like :ID, :START_ID, :END_ID
                clean_prop = header.replace(":ID", "").replace(":START_ID", "").replace(":END_ID", "")
                if clean_prop and clean_prop != "id":  # Skip empty and generic id
                    properties.add(clean_prop)
            
            # Add optional headers if present
            if node_schema.optional_headers:
                for header in node_schema.optional_headers:
                    clean_prop = header.replace(":ID", "").replace(":START_ID", "").replace(":END_ID", "")
                    if clean_prop and clean_prop != "id":
                        properties.add(clean_prop)
            
            # Always include 'id' as it's the primary identifier
            properties.add("id")
            labels[node_schema.label] = properties
        
        # Extract edges from relationship schemas
        edges = set()
        for rel_schema in validator.relationship_schemas:
            # Map relationship prefixes to actual labels
            src_labels = cls._resolve_prefixes_to_labels(rel_schema.start_node_prefixes, labels)
            dst_labels = cls._resolve_prefixes_to_labels(rel_schema.end_node_prefixes, labels)
            
            # Create edges for all valid combinations
            for src_label in src_labels:
                for dst_label in dst_labels:
                    edges.add((src_label, rel_schema.relationship_type, dst_label))
        
        logger.info(f"🗂️  Loaded schema: {len(labels)} labels, {len(edges)} edges")
        logger.debug(f"Labels: {list(labels.keys())}")
        logger.debug(f"Relationships: {[f'{s}-[{r}]->{d}' for s,r,d in sorted(edges)]}")
        
        return cls(
            labels=labels,
            edges=edges,
            enforcement=enforcement
        )
    
    @staticmethod
    def _resolve_prefixes_to_labels(prefixes: List[str], labels: Dict[str, Set[str]]) -> List[str]:
        """
        Resolve node prefixes from CSV schema to actual node labels.
        
        The CSV schema uses prefixes like 'protein', 'gene' but the actual
        labels are 'Protein', 'Gene'. Handle empty prefixes (direct references).
        """
        if not prefixes or prefixes == [""]:
            # Empty prefix means direct reference - could be any label
            # For now, return all labels (conservative approach)
            return list(labels.keys())
        
        resolved = []
        for prefix in prefixes:
            if not prefix:  # Empty string
                continue
                
            # Find matching label (case-insensitive)
            for label in labels.keys():
                if label.lower() == prefix.lower():
                    resolved.append(label)
                    break
            else:
                logger.warning(f"Could not resolve prefix '{prefix}' to any label")
        
        return resolved if resolved else list(labels.keys())
    
    async def verify_against_db(self, graph_client) -> SchemaDriftReport:
        """
        Verify schema against live database.
        Check labels, relationship types, and sample properties.
        """
        report = SchemaDriftReport(
            missing_labels=set(),
            missing_relationships=set(), 
            missing_properties={},
            extra_labels=set(),
            extra_relationships=set(),
            warnings=[],
            errors=[]
        )
        
        try:
            # Check labels exist by sampling nodes (avoid CALL statements)
            db_labels = set()
            for expected_label in self.labels:
                try:
                    sample_query = f"MATCH (n:{expected_label}) RETURN labels(n)[0] as label LIMIT 1"
                    sample_result = await graph_client.process_query(sample_query, query_type="cypher")
                    if sample_result.results:
                        db_labels.add(sample_result.results[0].get('label', ''))
                except Exception:
                    # Label doesn't exist or query failed
                    pass
            
            for expected_label in self.labels:
                if expected_label not in db_labels:
                    report.missing_labels.add(expected_label)
                    report.errors.append(f"Missing label: {expected_label}")
            
            # Check relationship types by sampling (avoid CALL statements)
            db_rels = set()
            expected_rels = set(rel_type for _, rel_type, _ in self.edges)
            for expected_rel in expected_rels:
                try:
                    rel_query = f"MATCH ()-[r:{expected_rel}]-() RETURN type(r) as relType LIMIT 1"
                    rel_result = await graph_client.process_query(rel_query, query_type="cypher")
                    if rel_result.results:
                        db_rels.add(rel_result.results[0].get('relType', ''))
                except Exception:
                    # Relationship doesn't exist or query failed
                    pass
            
            expected_rels = set(rel_type for _, rel_type, _ in self.edges)
            for expected_rel in expected_rels:
                if expected_rel not in db_rels:
                    report.missing_relationships.add(expected_rel)
                    report.errors.append(f"Missing relationship: {expected_rel}")
            
            # Sample property checks (light probe to avoid expensive queries)
            for label, expected_props in self.labels.items():
                if label in db_labels:  # Only check if label exists
                    try:
                        # Sample one node to check properties
                        sample_query = f"MATCH (n:{label}) WITH n LIMIT 1 RETURN keys(n) as properties"
                        sample_result = await graph_client.process_query(sample_query, query_type="cypher")
                        
                        if sample_result.results:
                            db_props = set(sample_result.results[0].get('properties', []))
                            missing_props = expected_props - db_props
                            
                            if missing_props:
                                report.missing_properties[label] = missing_props
                                report.warnings.append(
                                    f"Label {label} missing properties: {missing_props} (sampled)"
                                )
                        else:
                            report.warnings.append(f"No {label} nodes found for property sampling")
                            
                    except Exception as e:
                        report.warnings.append(f"Could not sample properties for {label}: {e}")
            
            # Log drift status
            if report.has_violations:
                if self.enforcement == SchemaEnforcement.STRICT:
                    logger.error(f"❌ Schema drift detected: {len(report.errors)} errors, {len(report.warnings)} warnings")
                else:
                    logger.warning(f"⚠️  Schema drift detected: {len(report.errors)} errors, {len(report.warnings)} warnings")
            else:
                logger.info("✅ Schema verification passed")
                
        except Exception as e:
            report.errors.append(f"Schema verification failed: {e}")
            logger.error(f"❌ Schema verification error: {e}")
        
        return report
    
    def assert_label(self, label: str) -> None:
        """Assert that a label exists in the schema."""
        if label not in self.labels:
            error = f"Unknown label: {label}. Available: {list(self.labels.keys())}"
            if self.enforcement == SchemaEnforcement.STRICT:
                raise ValueError(error)
            else:
                logger.warning(error)
    
    def assert_property(self, label: str, prop: str) -> None:
        """Assert that a property exists for a label."""
        self.assert_label(label)
        
        if label in self.labels and prop not in self.labels[label]:
            error = f"Unknown property {prop} for label {label}. Available: {self.labels[label]}"
            if self.enforcement == SchemaEnforcement.STRICT:
                raise ValueError(error) 
            else:
                logger.warning(error)
    
    def assert_edge(self, src_label: str, rel_type: str, dst_label: str) -> None:
        """Assert that an edge exists in the schema."""
        edge = (src_label, rel_type, dst_label)
        if edge not in self.edges:
            error = f"Unknown edge: {src_label}-[{rel_type}]->{dst_label}"
            if self.enforcement == SchemaEnforcement.STRICT:
                raise ValueError(error)
            else:
                logger.warning(error)
    
    def get_properties(self, label: str) -> Set[str]:
        """Get all properties for a label."""
        self.assert_label(label)
        return self.labels.get(label, set())
    
    def get_relationships_for_label(self, label: str) -> List[Tuple[str, str]]:
        """Get all relationships (type, target_label) for a source label."""
        return [(rel_type, dst_label) for src, rel_type, dst_label in self.edges if src == label]
    
    def has_label(self, label: str) -> bool:
        """Check if label exists in schema."""
        return label in self.labels
    
    def has_property(self, label: str, prop: str) -> bool:
        """Check if property exists for label."""
        return label in self.labels and prop in self.labels[label]
    
    def has_edge(self, src_label: str, rel_type: str, dst_label: str) -> bool:
        """Check if edge exists in schema."""
        return (src_label, rel_type, dst_label) in self.edges