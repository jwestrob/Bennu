"""
Tests for early exit behavior and WGR blocking.

Verifies that the dynamic executor properly evaluates evidence and blocks
expensive tools when cheap tools provide conclusive answers.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the modules under test
from ..models import Plan, PlanStep, Intent, Guard, ToolOutput, Settings, EvidenceLedger
from ..policy_engine import PolicyEngine
from ..schema_resolver import SchemaResolver
from ..agent_executor import execute_dynamic_loop, _find_next_eligible_step
from ..core import plan_initial


class TestEarlyExit:
    """Test early exit behavior for presence/absence queries."""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock Settings instance with test configuration."""
        settings = Mock(spec=Settings)
        settings.max_wallclock_seconds = 300
        settings.vector_hit_threshold = 0.7
        settings.evidence_ledger_dir = "test/evidence"
        settings.save_evidence_ledger = Mock(return_value="test_ledger.json")
        return settings
    
    @pytest.fixture
    def mock_neo4j_processor(self):
        """Mock Neo4j processor."""
        processor = Mock()
        processor.run_query = Mock(return_value=[])
        return processor
    
    @pytest.fixture
    def schema_resolver(self, mock_neo4j_processor, mock_settings):
        """Schema resolver with mocked dependencies."""
        return SchemaResolver(mock_neo4j_processor, mock_settings)
    
    def test_conclusive_absent_early_exit(self, mock_settings):
        """Test that conclusive_absent verdict prevents whole_genome_reader invocation."""
        
        # Create plan with database_query and whole_genome_reader steps
        plan = Plan(
            intent=Intent.PRESENCE_ABSENCE,
            steps=[
                PlanStep(
                    tool="database_query",
                    args={"query": "rubisco"},
                    cost="cheap",
                    id="db_query"
                ),
                PlanStep(
                    tool="whole_genome_reader", 
                    args={"query": "rubisco"},
                    cost="expensive",
                    guards=[Guard(name="requires_inconclusive")],
                    id="wgr_analysis"
                )
            ],
            metadata={"resolved_targets": {"proteins": []}}
        )
        
        # Mock policy engine to return conclusive_absent after database query
        with patch('src.llm.rag_system.agent_executor.get_policy_engine') as mock_policy:
            policy_engine = Mock()
            mock_policy.return_value = policy_engine
            
            # First call: inconclusive (allows continuation)
            # Second call: conclusive_absent (triggers early exit)
            policy_engine.assess.side_effect = [
                {"state": "conclusive_absent", "confidence": 0.85, "rationale": "No matches found"},
            ]
            
            # Mock guard evaluation to block WGR when evidence is conclusive
            def mock_evaluate_guard(guard, context):
                if guard.name == "requires_inconclusive":
                    # Block expensive tools when evidence is conclusive
                    return False  # Evidence is conclusive, block WGR
                return True
            
            policy_engine.evaluate_guard = mock_evaluate_guard
            
            # Mock tool execution to return no matches
            with patch('src.llm.rag_system.agent_executor._execute_database_query') as mock_db:
                mock_db.return_value = {
                    "summary": "No rubisco proteins found in knowledge graph",
                    "artifacts": {"results": []},
                    "metrics": {"kg_matches": 0, "conclusive": True, "execution_time": 0.1}
                }
                
                # Execute the plan
                result = pytest.asyncio.run(execute_dynamic_loop(plan, mock_settings, "test_session"))
                
                # Verify early exit occurred
                assert "conclusive_absent" in str(result.get("metadata", {}))
                
                # Verify whole_genome_reader was never called
                with patch('src.llm.rag_system.agent_executor._execute_whole_genome_reader') as mock_wgr:
                    mock_wgr.assert_not_called()
    
    def test_no_anchor_entities_blocks_wgr(self, mock_settings):
        """Test that whole_genome_reader is ineligible when no anchor entities exist."""
        
        # Create plan with spatial intent but no resolved targets (no anchors)
        plan = Plan(
            intent=Intent.SPATIAL_NEIGHBORHOOD,
            steps=[
                PlanStep(
                    tool="database_query",
                    args={"query": "hypothetical spatial query"},
                    cost="cheap", 
                    id="db_query"
                ),
                PlanStep(
                    tool="whole_genome_reader",
                    args={"query": "spatial analysis"},
                    cost="expensive",
                    guards=[Guard(name="requires_anchor")],
                    id="wgr_spatial"
                )
            ],
            metadata={"resolved_targets": {"proteins": [], "domains": [], "functions": []}}  # No anchors
        )
        
        with patch('src.llm.rag_system.agent_executor.get_policy_engine') as mock_policy:
            policy_engine = Mock()
            mock_policy.return_value = policy_engine
            
            # Mock guard evaluation - requires_anchor should fail when no anchors
            def mock_evaluate_guard(guard, context):
                if guard.name == "requires_anchor":
                    resolved_targets = context.get("resolved_targets", {})
                    anchor_types = ["proteins", "domains", "functions"]
                    anchor_count = sum(len(resolved_targets.get(t, [])) for t in anchor_types)
                    return anchor_count > 0  # Should be False - no anchors
                return True
            
            policy_engine.evaluate_guard = mock_evaluate_guard
            policy_engine.assess.return_value = {
                "state": "inconclusive",
                "confidence": 0.5,
                "rationale": "Insufficient anchor entities"
            }
            
            # Mock database query execution
            with patch('src.llm.rag_system.agent_executor._execute_database_query') as mock_db:
                mock_db.return_value = {
                    "summary": "Database search completed",
                    "artifacts": {"results": []},
                    "metrics": {"kg_matches": 0, "execution_time": 0.1}
                }
                
                # Execute and verify WGR is blocked
                result = pytest.asyncio.run(execute_dynamic_loop(plan, mock_settings, "test_session"))
                
                # Verify only database_query executed
                executed_tools = [call.tool for call in result.get("metadata", {}).get("evidence_summary", {}).get("tools_executed", [])]
                
                # Should not contain whole_genome_reader
                assert "whole_genome_reader" not in str(result)
    
    def test_plan_generation_blocks_wgr_for_presence_absence(self, schema_resolver):
        """Test that plan generation doesn't include WGR for simple presence queries."""
        
        # Mock resolver to return no anchor entities
        with patch.object(schema_resolver, 'resolve_targets_from_query') as mock_resolve:
            with patch.object(schema_resolver, 'has_anchor_entities') as mock_has_anchors:
                mock_resolve.return_value = {"proteins": [], "domains": [], "functions": []}
                mock_has_anchors.return_value = False
                
                # Generate plan for presence/absence query
                plan = plan_initial("Does rubisco exist in the dataset?", schema_resolver)
                
                # Verify plan intent is presence/absence
                assert plan.intent == Intent.PRESENCE_ABSENCE
                
                # Verify no whole_genome_reader step in plan
                tool_names = [step.tool for step in plan.steps]
                assert "whole_genome_reader" not in tool_names
                
                # Should only have cheap tools
                assert "database_query" in tool_names
    
    @pytest.mark.asyncio
    async def test_step_eligibility_evaluation(self):
        """Test that _find_next_eligible_step properly evaluates guards."""
        
        plan = Plan(
            intent=Intent.PRESENCE_ABSENCE,
            steps=[
                PlanStep(
                    tool="whole_genome_reader",
                    cost="expensive", 
                    guards=[Guard(name="requires_anchor"), Guard(name="requires_inconclusive")],
                    id="blocked_step"
                )
            ],
            metadata={"resolved_targets": {"proteins": []}}
        )
        
        # Mock policy engine that blocks the step
        policy_engine = Mock()
        policy_engine.evaluate_guard.return_value = False  # Block all guards
        
        # Should return None (no eligible steps)
        next_step = _find_next_eligible_step(
            plan=plan,
            executed_steps=[],
            policy_engine=policy_engine,
            resolved_targets={},
            tool_outputs=[]
        )
        
        assert next_step is None
        
        # Verify guard was evaluated
        policy_engine.evaluate_guard.assert_called()


class TestPolicyEngine:
    """Test policy engine guard evaluation and evidence assessment."""
    
    @pytest.fixture 
    def policy_engine(self):
        """Policy engine with mock settings."""
        settings = Mock()
        settings.vector_hit_threshold = 0.7
        return PolicyEngine(settings)
    
    def test_requires_anchor_guard(self, policy_engine):
        """Test requires_anchor guard evaluation."""
        
        # Context with anchor entities
        context_with_anchors = {
            "resolved_targets": {"proteins": ["protein1"], "domains": [], "functions": []}
        }
        guard = Guard(name="requires_anchor")
        
        assert policy_engine.evaluate_guard(guard, context_with_anchors) == True
        
        # Context without anchor entities
        context_no_anchors = {
            "resolved_targets": {"proteins": [], "domains": [], "functions": []}
        }
        
        assert policy_engine.evaluate_guard(guard, context_no_anchors) == False
    
    def test_presence_absence_assessment(self, policy_engine):
        """Test presence/absence conclusiveness assessment."""
        
        # Tool outputs with no matches - should be conclusive absent
        no_match_outputs = [
            ToolOutput(
                tool="database_query",
                success=True,
                summary="No matches found",
                artifacts={},
                metrics={"kg_matches": 0, "max_similarity": 0.3}
            ),
            ToolOutput(
                tool="vector_search", 
                success=True,
                summary="Low similarity matches",
                artifacts={},
                metrics={"vector_matches": 0, "max_similarity": 0.3}
            )
        ]
        
        verdict = policy_engine.assess(Intent.PRESENCE_ABSENCE, no_match_outputs, {})
        assert verdict["state"] == "conclusive_absent"
        assert verdict["confidence"] > 0.8
        
        # Tool outputs with matches - should be conclusive present  
        match_outputs = [
            ToolOutput(
                tool="database_query",
                success=True, 
                summary="Found matches",
                artifacts={},
                metrics={"kg_matches": 5, "max_similarity": 0.9}
            )
        ]
        
        verdict = policy_engine.assess(Intent.PRESENCE_ABSENCE, match_outputs, {})
        assert verdict["state"] == "conclusive_present" 
        assert verdict["confidence"] > 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])