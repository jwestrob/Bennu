"""
Tests for TaskRepairAgent functionality.
"""

import pytest
from unittest.mock import Mock, patch

from src.llm.task_repair_agent import TaskRepairAgent
from src.llm.repair_types import RepairResult, RepairStrategy, SchemaInfo
from src.llm.error_patterns import ErrorPatternRegistry


class TestTaskRepairAgent:
    """Test TaskRepairAgent error detection and repair capabilities"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.agent = TaskRepairAgent()
    
    def test_initialization(self):
        """Test TaskRepairAgent initializes correctly"""
        assert self.agent.schema_info is not None
        assert self.agent.error_registry is not None
        assert len(self.agent.schema_info.node_labels) > 0
    
    def test_comment_query_repair(self):
        """Test repair of DSPy comment queries"""
        comment_query = "/* No valid query can be constructed: label `FakeNode` is not part of the graph schema */"
        error = Exception("Neo4j syntax error")
        
        result = self.agent.detect_and_repair(comment_query, error)
        
        assert result.success is True
        assert result.repair_strategy_used == RepairStrategy.COMMENT_QUERY_EXPLANATION
        assert "FakeNode" in result.user_message
        assert "doesn't exist" in result.user_message
        assert len(result.suggested_alternatives) > 0
        assert result.confidence > 0.8
    
    def test_invalid_relationship_repair(self):
        """Test repair of invalid relationship queries"""
        invalid_query = "MATCH (p:Protein)-[:NONEXISTENT_RELATIONSHIP]->(d:Domain) RETURN p"
        error = Exception("Invalid relationship")
        
        result = self.agent.detect_and_repair(invalid_query, error)
        
        assert result.success is True
        assert result.repair_strategy_used == RepairStrategy.RELATIONSHIP_MAPPING
        assert result.repaired_query is not None
        assert "HASDOMAIN" in result.repaired_query
        assert "NONEXISTENT_RELATIONSHIP" not in result.repaired_query
    
    def test_entity_suggestion(self):
        """Test entity name suggestions"""
        from src.llm.error_patterns import EntitySuggester
        
        suggestions = EntitySuggester.suggest_alternatives(
            "FakeNode", 
            ["Protein", "Gene", "Domain"], 
            max_suggestions=2
        )
        
        assert len(suggestions) <= 2
        assert all(suggestion in ["Protein", "Gene", "Domain"] for suggestion in suggestions)
    
    def test_extract_entity_from_comment(self):
        """Test entity extraction from comment queries"""
        from src.llm.error_patterns import EntitySuggester
        
        comment = "/* No valid query can be constructed: label `FakeNode` is not part of the graph schema */"
        entity = EntitySuggester.extract_entity_from_comment(comment)
        
        assert entity == "FakeNode"
    
    def test_fallback_result(self):
        """Test fallback when no patterns match"""
        unknown_query = "SELECT * FROM unknown_table"
        unknown_error = Exception("Unknown database error")
        
        result = self.agent.detect_and_repair(unknown_query, unknown_error)
        
        # Should still provide a helpful message even if repair fails
        assert result.user_message is not None
        assert "genomic database" in result.user_message.lower()
        assert len(result.suggested_alternatives) > 0
    
    def test_schema_info_default(self):
        """Test default genomic schema information"""
        schema = SchemaInfo.default_genomic_schema()
        
        expected_labels = ["Protein", "Gene", "Domain", "KEGGOrtholog", "DomainAnnotation"]
        expected_relationships = ["ENCODEDBY", "HASDOMAIN", "DOMAINFAMILY", "HASFUNCTION"]
        
        assert all(label in schema.node_labels for label in expected_labels)
        assert all(rel in schema.relationship_types for rel in expected_relationships)
        assert "Protein" in schema.node_properties
        assert "id" in schema.node_properties["Protein"]
    
    def test_error_pattern_matching(self):
        """Test error pattern detection"""
        registry = ErrorPatternRegistry()
        
        comment_query = "/* No valid query can be constructed: label `FakeNode` is not part of the graph schema */"
        patterns = registry.find_matching_patterns(comment_query, "")
        
        assert len(patterns) > 0
        assert any(p.pattern_type == "comment_query" for p in patterns)
    
    def test_relationship_mapping(self):
        """Test relationship mapping functionality"""
        from src.llm.error_patterns import RelationshipMapper
        
        mapped = RelationshipMapper.map_relationship("NONEXISTENT_RELATIONSHIP")
        assert mapped == "HASDOMAIN"
        
        mapped = RelationshipMapper.map_relationship("CONNECTS_TO")
        assert mapped == "ENCODEDBY"
        
        valid_rels = RelationshipMapper.get_valid_relationships()
        assert "HASDOMAIN" in valid_rels
        assert "ENCODEDBY" in valid_rels
    
    def test_get_schema_summary(self):
        """Test schema summary functionality"""
        summary = self.agent.get_schema_summary()
        
        assert "node_labels" in summary
        assert "relationship_types" in summary
        assert "total_patterns" in summary
        assert "available_strategies" in summary
        assert summary["total_patterns"] > 0


class TestErrorPatternRegistry:
    """Test error pattern registry functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.registry = ErrorPatternRegistry()
    
    def test_pattern_initialization(self):
        """Test that patterns are initialized correctly"""
        assert len(self.registry.patterns) > 0
        
        pattern_types = [p.pattern_type for p in self.registry.patterns]
        expected_types = ["comment_query", "invalid_node_label", "invalid_relationship", "neo4j_syntax_error"]
        
        for expected_type in expected_types:
            assert expected_type in pattern_types
    
    def test_get_pattern_by_type(self):
        """Test retrieving patterns by type"""
        comment_pattern = self.registry.get_pattern_by_type("comment_query")
        assert comment_pattern.repair_strategy == RepairStrategy.COMMENT_QUERY_EXPLANATION
        
        with pytest.raises(ValueError):
            self.registry.get_pattern_by_type("nonexistent_pattern")


@pytest.mark.integration
class TestTaskRepairAgentIntegration:
    """Integration tests for TaskRepairAgent with real error scenarios"""
    
    def test_real_comment_query_scenario(self):
        """Test with actual DSPy-generated comment query"""
        agent = TaskRepairAgent()
        
        # This is the actual query we saw in testing
        real_comment = "/* No valid query can be constructed: label `FakeNode` is not part of the graph schema */"
        real_error = Exception(
            "Neo.ClientError.Statement.SyntaxError: Invalid input '': expected 'ALTER', 'ORDER BY', 'CALL'..."
        )
        
        result = agent.detect_and_repair(real_comment, real_error)
        
        assert result.success is True
        assert "FakeNode" in result.user_message
        assert "doesn't exist" in result.user_message
        assert len(result.suggested_alternatives) > 0
        assert any(alt in ["Protein", "Gene", "Domain"] for alt in result.suggested_alternatives)
        assert result.confidence > 0.8
    
    def test_multiple_error_patterns(self):
        """Test handling multiple error patterns in one query"""
        agent = TaskRepairAgent()
        
        # Query with both invalid entity and invalid relationship
        complex_query = "MATCH (fake:FakeNode)-[:NONEXISTENT_RELATIONSHIP]->(d:Domain) RETURN fake"
        error = Exception("Multiple issues")
        
        result = agent.detect_and_repair(complex_query, error)
        
        # Should handle at least one of the issues
        assert result.success is True or result.user_message is not None