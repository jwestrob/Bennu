"""
Comprehensive tests for the integrated agentic RAG system.
Tests both traditional and agentic execution paths.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from src.llm.rag_system import (
    GenomicRAG, TaskGraph, Task, TaskType, TaskStatus, 
    AVAILABLE_TOOLS, literature_search, GenomicContext
)
from src.llm.config import LLMConfig

class TestAgenticComponents:
    """Test the core agentic components within rag_system.py."""
    
    def test_task_graph_basic_operations(self):
        """Test basic task graph operations."""
        graph = TaskGraph()
        
        task = Task(
            task_id="test_task",
            task_type=TaskType.ATOMIC_QUERY,
            query="MATCH (p:Protein) RETURN p"
        )
        
        task_id = graph.add_task(task)
        assert task_id == "test_task"
        assert len(graph.tasks) == 1
        
        # Task should be ready (no dependencies)
        ready_tasks = graph.get_ready_tasks()
        assert len(ready_tasks) == 1
        assert ready_tasks[0].task_id == "test_task"
        
        # Mark complete
        graph.mark_task_status("test_task", TaskStatus.COMPLETED, result="test_result")
        completed_task = graph.get_task("test_task")
        assert completed_task.status == TaskStatus.COMPLETED
        assert completed_task.result == "test_result"
        
        # Graph should be complete
        assert graph.is_complete() is True
    
    def test_task_dependencies(self):
        """Test task dependency resolution."""
        graph = TaskGraph()
        
        task1 = Task(task_id="task1", query="Query 1")
        task2 = Task(task_id="task2", query="Query 2", dependencies={"task1"})
        
        graph.add_task(task1)
        graph.add_task(task2)
        
        # Only task1 should be ready initially
        ready_tasks = graph.get_ready_tasks()
        assert len(ready_tasks) == 1
        assert ready_tasks[0].task_id == "task1"
        
        # Complete task1
        graph.mark_task_status("task1", TaskStatus.COMPLETED)
        
        # Now task2 should be ready
        ready_tasks = graph.get_ready_tasks()
        assert len(ready_tasks) == 1
        assert ready_tasks[0].task_id == "task2"
    
    def test_task_failure_handling(self):
        """Test task failure and skipping behavior."""
        graph = TaskGraph()
        
        task1 = Task(task_id="task1", query="Query 1")
        task2 = Task(task_id="task2", query="Query 2", dependencies={"task1"})
        
        graph.add_task(task1)
        graph.add_task(task2)
        
        # Fail task1
        graph.mark_task_status("task1", TaskStatus.FAILED, error="Test failure")
        
        # Mark skipped tasks
        graph.mark_skipped_tasks()
        
        # task2 should be skipped
        task2_instance = graph.get_task("task2")
        assert task2_instance.status == TaskStatus.SKIPPED
        
        # Graph should be complete
        assert graph.is_complete() is True
    
    @patch('src.llm.rag_system.ENTREZ_AVAILABLE', True)
    @patch('src.llm.rag_system.Entrez')
    def test_literature_search_tool(self, mock_entrez):
        """Test the literature search tool."""
        # Mock Entrez responses
        mock_search_handle = Mock()
        mock_entrez.esearch.return_value = mock_search_handle
        
        mock_fetch_handle = Mock()
        mock_entrez.efetch.return_value = mock_fetch_handle
        
        mock_entrez.read.side_effect = [
            {'IdList': ['12345'], 'Count': '1'},  # Search results
            {'PubmedArticle': [{  # Fetch results
                'MedlineCitation': {
                    'Article': {
                        'ArticleTitle': 'Test Article About Proteins',
                        'Abstract': {'AbstractText': ['This is a test abstract about protein function.']}
                    },
                    'PMID': {'content': '12345'}
                }
            }]}
        ]
        
        result = literature_search("protein function", "test@example.com")
        
        assert isinstance(result, str)
        assert "protein function" in result
        assert "12345" in result
        assert "Test Article About Proteins" in result
    
    def test_available_tools_manifest(self):
        """Test the tool manifest structure."""
        assert "literature_search" in AVAILABLE_TOOLS
        assert callable(AVAILABLE_TOOLS["literature_search"])
        
        # Test tool can be called via manifest
        tool_func = AVAILABLE_TOOLS["literature_search"]
        with patch('src.llm.rag_system.ENTREZ_AVAILABLE', True), \
             patch('src.llm.rag_system.Entrez') as mock_entrez:
            
            mock_search_handle = Mock()
            mock_entrez.esearch.return_value = mock_search_handle
            mock_entrez.read.return_value = {'IdList': [], 'Count': '0'}
            
            result = tool_func(query="test", email="test@example.com")
            assert isinstance(result, str)

class TestGenomicRAGAgentic:
    """Test the enhanced GenomicRAG with agentic capabilities."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = Mock(spec=LLMConfig)
        config.llm_provider = "openai"
        config.llm_model = "gpt-4"
        config.max_results_per_query = 10
        config.get_api_key.return_value = "test_key"
        return config
    
    @patch('src.llm.rag_system.dspy')
    @patch('src.llm.rag_system.Neo4jQueryProcessor')
    @patch('src.llm.rag_system.LanceDBQueryProcessor')
    @patch('src.llm.rag_system.HybridQueryProcessor')
    def test_genomic_rag_initialization(self, mock_hybrid, mock_lancedb, mock_neo4j, mock_dspy, mock_config):
        """Test GenomicRAG initialization with agentic components."""
        # Mock the processors
        mock_neo4j.return_value = Mock()
        mock_lancedb.return_value = Mock()
        mock_hybrid.return_value = Mock()
        
        # Mock DSPy settings
        mock_dspy.settings.configure = Mock()
        mock_dspy.ChainOfThought = Mock()
        mock_dspy.LM = Mock()
        
        rag = GenomicRAG(mock_config)
        
        # Verify agentic components are initialized
        assert hasattr(rag, 'planner')
        assert hasattr(rag, 'classifier')
        assert hasattr(rag, 'retriever')
        assert hasattr(rag, 'answerer')
        
        # Verify processors are initialized
        assert hasattr(rag, 'neo4j_processor')
        assert hasattr(rag, 'lancedb_processor')
        assert hasattr(rag, 'hybrid_processor')
    
    @patch('src.llm.rag_system.dspy')
    @patch('src.llm.rag_system.Neo4jQueryProcessor')
    @patch('src.llm.rag_system.LanceDBQueryProcessor')
    @patch('src.llm.rag_system.HybridQueryProcessor')
    @pytest.mark.asyncio
    async def test_traditional_query_path(self, mock_hybrid, mock_lancedb, mock_neo4j, mock_dspy, mock_config):
        """Test the traditional (non-agentic) query path."""
        # Setup mocks
        mock_neo4j.return_value = Mock()
        mock_lancedb.return_value = Mock()
        mock_hybrid.return_value = Mock()
        
        mock_dspy.settings.configure = Mock()
        mock_dspy.ChainOfThought = Mock()
        mock_dspy.LM = Mock()
        
        # Mock DSPy responses
        mock_planner = Mock()
        mock_planner.return_value = Mock(
            requires_planning=False,
            reasoning="Simple query that can be answered directly"
        )
        
        mock_classifier = Mock()
        mock_classifier.return_value = Mock(
            query_type="structural",
            reasoning="Direct protein query"
        )
        
        mock_retriever = Mock()
        mock_retriever.return_value = Mock(
            search_strategy="Neo4j only",
            neo4j_query="MATCH (p:Protein) RETURN p"
        )
        
        mock_answerer = Mock()
        mock_answerer.return_value = Mock(
            answer="Test answer",
            confidence="high",
            citations="Test citations"
        )
        
        mock_dspy.ChainOfThought.side_effect = [
            mock_classifier, mock_retriever, mock_answerer, mock_planner
        ]
        
        rag = GenomicRAG(mock_config)
        rag.planner = mock_planner
        rag.classifier = mock_classifier
        rag.retriever = mock_retriever
        rag.answerer = mock_answerer
        
        # Mock _retrieve_context
        rag._retrieve_context = AsyncMock(return_value=GenomicContext(
            structured_data=[{"protein_id": "test_protein"}],
            semantic_data=[],
            metadata={},
            query_time=0.1
        ))
        
        response = await rag.ask("How many proteins are there?")
        
        assert response["question"] == "How many proteins are there?"
        assert response["answer"] == "Test answer"
        assert response["confidence"] == "high"
        assert response["query_metadata"]["execution_mode"] == "traditional"
    
    @patch('src.llm.rag_system.dspy')
    @patch('src.llm.rag_system.Neo4jQueryProcessor')
    @patch('src.llm.rag_system.LanceDBQueryProcessor')
    @patch('src.llm.rag_system.HybridQueryProcessor')
    @patch('src.llm.rag_system.ENTREZ_AVAILABLE', True)
    @patch('src.llm.rag_system.Entrez')
    @pytest.mark.asyncio
    async def test_agentic_query_path(self, mock_entrez, mock_hybrid, mock_lancedb, mock_neo4j, mock_dspy, mock_config):
        """Test the agentic multi-step query path."""
        # Setup mocks
        mock_neo4j.return_value = Mock()
        mock_lancedb.return_value = Mock()
        mock_hybrid.return_value = Mock()
        
        mock_dspy.settings.configure = Mock()
        mock_dspy.ChainOfThought = Mock()
        mock_dspy.LM = Mock()
        
        # Mock Entrez for literature search
        mock_search_handle = Mock()
        mock_entrez.esearch.return_value = mock_search_handle
        mock_entrez.read.side_effect = [
            {'IdList': ['12345'], 'Count': '1'},
            {'PubmedArticle': [{'MedlineCitation': {
                'Article': {'ArticleTitle': 'Test Article', 'Abstract': {'AbstractText': ['Test abstract']}},
                'PMID': {'content': '12345'}
            }}]}
        ]
        
        # Mock DSPy responses for agentic planning
        mock_planner = Mock()
        task_plan = {
            "tasks": [
                {
                    "id": "query_local",
                    "type": "atomic_query",
                    "query": "Find proteins in local database",
                    "dependencies": []
                },
                {
                    "id": "search_literature",
                    "type": "tool_call",
                    "tool_name": "literature_search",
                    "tool_args": {"query": "protein function"},
                    "dependencies": ["query_local"]
                },
                {
                    "id": "combine_results",
                    "type": "aggregate",
                    "dependencies": ["query_local", "search_literature"]
                }
            ]
        }
        
        mock_planner.return_value = Mock(
            requires_planning=True,
            task_plan=json.dumps(task_plan),
            reasoning="Complex query requiring literature search"
        )
        
        mock_classifier = Mock()
        mock_classifier.return_value = Mock(
            query_type="general",
            reasoning="Database query"
        )
        
        mock_retriever = Mock()
        mock_retriever.return_value = Mock(
            search_strategy="Neo4j",
            neo4j_query="MATCH (p:Protein) RETURN p"
        )
        
        mock_answerer = Mock()
        mock_answerer.return_value = Mock(
            answer="Combined analysis from database and literature",
            confidence="high",
            citations="Local database + PubMed"
        )
        
        mock_dspy.ChainOfThought.side_effect = [
            mock_classifier, mock_retriever, mock_answerer, mock_planner
        ]
        
        rag = GenomicRAG(mock_config)
        rag.planner = mock_planner
        rag.classifier = mock_classifier
        rag.retriever = mock_retriever
        rag.answerer = mock_answerer
        
        # Mock _retrieve_context
        rag._retrieve_context = AsyncMock(return_value=GenomicContext(
            structured_data=[{"protein_id": "test_protein"}],
            semantic_data=[],
            metadata={},
            query_time=0.1
        ))
        
        response = await rag.ask("What do we know about protein functions from both our database and recent literature?")
        
        assert response["question"] == "What do we know about protein functions from both our database and recent literature?"
        assert response["answer"] == "Combined analysis from database and literature"
        assert response["query_metadata"]["execution_mode"] == "agentic"
        assert response["query_metadata"]["tasks_completed"] == 3
        assert response["query_metadata"]["tasks_failed"] == 0
    
    @patch('src.llm.rag_system.dspy')
    @patch('src.llm.rag_system.Neo4jQueryProcessor')
    @patch('src.llm.rag_system.LanceDBQueryProcessor')
    @patch('src.llm.rag_system.HybridQueryProcessor')
    @pytest.mark.asyncio
    async def test_agentic_fallback_on_error(self, mock_hybrid, mock_lancedb, mock_neo4j, mock_dspy, mock_config):
        """Test that agentic planning falls back to traditional mode on errors."""
        # Setup mocks
        mock_neo4j.return_value = Mock()
        mock_lancedb.return_value = Mock()
        mock_hybrid.return_value = Mock()
        
        mock_dspy.settings.configure = Mock()
        mock_dspy.ChainOfThought = Mock()
        mock_dspy.LM = Mock()
        
        # Mock planner that causes an error in agentic mode
        mock_planner = Mock()
        mock_planner.return_value = Mock(
            requires_planning=True,
            task_plan='{"tasks": [{"id": "task1", "type": "invalid_type"}]}',  # Plan with invalid task type
            reasoning="Planning error test"
        )
        
        # Mock traditional path components
        mock_classifier = Mock()
        mock_classifier.return_value = Mock(
            query_type="general",
            reasoning="Fallback query"
        )
        
        mock_retriever = Mock()
        mock_retriever.return_value = Mock(
            search_strategy="Neo4j",
            neo4j_query="MATCH (p:Protein) RETURN p"
        )
        
        mock_answerer = Mock()
        mock_answerer.return_value = Mock(
            answer="Fallback answer",
            confidence="medium",
            citations="Local database"
        )
        
        mock_dspy.ChainOfThought.side_effect = [
            mock_classifier, mock_retriever, mock_answerer, mock_planner
        ]
        
        rag = GenomicRAG(mock_config)
        rag.planner = mock_planner
        rag.classifier = mock_classifier
        rag.retriever = mock_retriever
        rag.answerer = mock_answerer
        
        # Mock _retrieve_context
        rag._retrieve_context = AsyncMock(return_value=GenomicContext(
            structured_data=[{"protein_id": "test_protein"}],
            semantic_data=[],
            metadata={},
            query_time=0.1
        ))
        
        response = await rag.ask("Test query that should fail in agentic mode")
        
        # Should fall back to traditional mode
        assert response["answer"] == "Fallback answer"
        assert response["query_metadata"]["execution_mode"] == "traditional"

class TestAgenticIntegration:
    """Integration tests for the complete agentic system."""
    
    def test_task_execution_interface(self):
        """Test the interface between task graph and tool execution."""
        # Create a tool call task
        task = Task(
            task_type=TaskType.TOOL_CALL,
            tool_name="literature_search",
            tool_args={"query": "CRISPR proteins", "email": "test@example.com"}
        )
        
        # Verify tool is available
        assert task.tool_name in AVAILABLE_TOOLS
        tool_function = AVAILABLE_TOOLS[task.tool_name]
        
        # Test execution
        with patch('src.llm.rag_system.ENTREZ_AVAILABLE', True), \
             patch('src.llm.rag_system.Entrez') as mock_entrez:
            
            mock_search_handle = Mock()
            mock_entrez.esearch.return_value = mock_search_handle
            mock_entrez.read.side_effect = [
                {'IdList': ['99999'], 'Count': '1'},
                {'PubmedArticle': [{
                    'MedlineCitation': {
                        'Article': {
                            'ArticleTitle': 'CRISPR-Cas9 protein engineering',
                            'Abstract': {'AbstractText': ['Study of CRISPR proteins']}
                        },
                        'PMID': {'content': '99999'}
                    }
                }]}
            ]
            
            result = tool_function(**task.tool_args)
            
            assert isinstance(result, str)
            assert "CRISPR proteins" in result
            assert "99999" in result
    
    def test_multi_step_workflow_design(self):
        """Test a realistic multi-step workflow design."""
        graph = TaskGraph()
        
        # Step 1: Query local database
        task1 = Task(
            task_id="local_query",
            task_type=TaskType.ATOMIC_QUERY,
            query="Find proteins with heme domains"
        )
        
        # Step 2: Literature search based on findings
        task2 = Task(
            task_id="literature_search",
            task_type=TaskType.TOOL_CALL,
            tool_name="literature_search",
            tool_args={"query": "heme transport proteins", "email": "test@example.com"},
            dependencies={"local_query"}
        )
        
        # Step 3: Synthesize results
        task3 = Task(
            task_id="synthesis",
            task_type=TaskType.AGGREGATE,
            dependencies={"local_query", "literature_search"}
        )
        
        # Add tasks
        graph.add_task(task1)
        graph.add_task(task2)
        graph.add_task(task3)
        
        # Verify execution order
        ready_tasks = graph.get_ready_tasks()
        assert len(ready_tasks) == 1
        assert ready_tasks[0].task_id == "local_query"
        
        # Complete first task
        graph.mark_task_status("local_query", TaskStatus.COMPLETED, result={"proteins": ["heme_1", "heme_2"]})
        
        # Literature search should be ready
        ready_tasks = graph.get_ready_tasks()
        assert len(ready_tasks) == 1
        assert ready_tasks[0].task_id == "literature_search"
        
        # Complete literature search
        graph.mark_task_status("literature_search", TaskStatus.COMPLETED, result={"papers": ["paper_1", "paper_2"]})
        
        # Synthesis should be ready
        ready_tasks = graph.get_ready_tasks()
        assert len(ready_tasks) == 1
        assert ready_tasks[0].task_id == "synthesis"
        
        # Complete synthesis
        graph.mark_task_status("synthesis", TaskStatus.COMPLETED, result={"combined_analysis": "result"})
        
        # All tasks complete
        assert graph.is_complete()
        summary = graph.get_summary()
        assert summary["completed"] == 3
        assert summary["failed"] == 0
    
    def test_backward_compatibility(self):
        """Test that existing code still works with enhanced system."""
        # This ensures our integration doesn't break existing functionality
        
        # These imports should work without issues
        from src.llm.rag_system import GenomicRAG, GenomicContext
        from src.llm.config import LLMConfig
        
        # Core classes should be available
        assert GenomicRAG is not None
        assert GenomicContext is not None
        
        # Agentic components should also be available
        from src.llm.rag_system import TaskGraph, Task, TaskType, AVAILABLE_TOOLS
        
        assert TaskGraph is not None
        assert Task is not None
        assert TaskType is not None
        assert AVAILABLE_TOOLS is not None
        assert "literature_search" in AVAILABLE_TOOLS