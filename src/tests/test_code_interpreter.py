#!/usr/bin/env python3
"""
Tests for Code Interpreter functionality.

Tests the secure code execution service, client interface, and integration
with the agentic RAG system.
"""

import asyncio
import pytest
import json
import uuid
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

# Test the client interface - handle missing dependencies gracefully
try:
    from src.code_interpreter.client import (
        CodeInterpreterClient,
        GenomicCodeInterpreter,
        code_interpreter_tool
    )
    CODE_INTERPRETER_AVAILABLE = True
except ImportError as e:
    CODE_INTERPRETER_AVAILABLE = False
    pytestmark = pytest.mark.skip(f"Code interpreter dependencies not available: {e}")


class TestCodeInterpreterClient:
    """Test the CodeInterpreterClient class."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return CodeInterpreterClient("http://localhost:8000")
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, client):
        """Test client initializes correctly."""
        assert client.base_url == "http://localhost:8000"
        assert isinstance(client.session_id, str)
        assert len(client.session_id) > 0
    
    @pytest.mark.asyncio
    async def test_new_session(self, client):
        """Test creating a new session."""
        original_session = client.session_id
        new_session = client.new_session()
        
        assert new_session != original_session
        assert client.session_id == new_session
        assert isinstance(new_session, str)
    
    @pytest.mark.asyncio
    async def test_execute_code_success(self, client):
        """Test successful code execution."""
        # Mock the HTTP client
        mock_response = Mock()
        mock_response.json.return_value = {
            "success": True,
            "stdout": "Hello, World!\n",
            "stderr": "",
            "execution_time": 0.05,
            "files_created": [],
            "session_id": client.session_id
        }
        mock_response.raise_for_status.return_value = None
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            result = await client.execute_code("print('Hello, World!')")
            
            assert result["success"] is True
            assert "Hello, World!" in result["stdout"]
            assert result["stderr"] == ""
            assert result["execution_time"] > 0
    
    @pytest.mark.asyncio
    async def test_execute_code_timeout(self, client):
        """Test code execution timeout handling."""
        with patch('httpx.AsyncClient') as mock_client:
            # Simulate timeout
            mock_client.return_value.__aenter__.return_value.post.side_effect = asyncio.TimeoutError()
            
            result = await client.execute_code("import time; time.sleep(100)", timeout=1)
            
            assert result["success"] is False
            assert "timed out" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_execute_code_connection_error(self, client):
        """Test handling of connection errors."""
        with patch('httpx.AsyncClient') as mock_client:
            # Simulate connection error
            import httpx
            mock_client.return_value.__aenter__.return_value.post.side_effect = httpx.ConnectError("Connection failed")
            
            result = await client.execute_code("print('test')")
            
            assert result["success"] is False
            assert "connect" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, client):
        """Test health check when service is healthy."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "healthy"}
        mock_response.raise_for_status.return_value = None
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            result = await client.health_check()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, client):
        """Test health check when service is unavailable."""
        with patch('httpx.AsyncClient') as mock_client:
            import httpx
            mock_client.return_value.__aenter__.return_value.get.side_effect = httpx.ConnectError("Connection failed")
            
            result = await client.health_check()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_reset_session(self, client):
        """Test session reset functionality."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            result = await client.reset_session()
            assert result is True


class TestGenomicCodeInterpreter:
    """Test the GenomicCodeInterpreter class."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock CodeInterpreterClient."""
        client = Mock(spec=CodeInterpreterClient)
        client.execute_code = AsyncMock()
        return client
    
    @pytest.fixture
    def interpreter(self, mock_client):
        """Create a GenomicCodeInterpreter with mock client."""
        return GenomicCodeInterpreter(mock_client)
    
    @pytest.mark.asyncio
    async def test_setup_genomic_environment(self, interpreter, mock_client):
        """Test setting up the genomic environment."""
        mock_client.execute_code.return_value = {
            "success": True,
            "stdout": "Genomic analysis environment ready!\n",
            "stderr": "",
            "execution_time": 0.1
        }
        
        data_paths = {
            "proteins": "/path/to/proteins.fasta",
            "annotations": "/path/to/annotations.gff"
        }
        
        result = await interpreter.setup_genomic_environment(data_paths)
        
        assert result["success"] is True
        assert interpreter.genomic_data_available is True
        mock_client.execute_code.assert_called_once()
        
        # Check that the setup code includes the data paths
        call_args = mock_client.execute_code.call_args[0][0]
        assert "data_paths" in call_args
        assert str(data_paths) in call_args
    
    @pytest.mark.asyncio
    async def test_analyze_protein_similarities(self, interpreter, mock_client):
        """Test protein similarity analysis."""
        interpreter.genomic_data_available = True
        
        mock_client.execute_code.return_value = {
            "success": True,
            "stdout": "Analyzing 3 proteins:\nVisualization saved as: protein_similarity_heatmap.png\n",
            "stderr": "",
            "execution_time": 0.5,
            "files_created": ["protein_similarity_heatmap.png"]
        }
        
        protein_ids = ["protein1", "protein2", "protein3"]
        result = await interpreter.analyze_protein_similarities(protein_ids)
        
        assert result["success"] is True
        assert "protein_similarity_heatmap.png" in result.get("files_created", [])
        mock_client.execute_code.assert_called_once()
        
        # Check that the analysis code includes the protein IDs
        call_args = mock_client.execute_code.call_args[0][0]
        assert str(protein_ids) in call_args
    
    @pytest.mark.asyncio
    async def test_analyze_without_setup(self, interpreter, mock_client):
        """Test that analysis fails without environment setup."""
        # Don't set genomic_data_available to True
        
        result = await interpreter.analyze_protein_similarities(["protein1"])
        
        assert result["success"] is False
        assert "not set up" in result["error"]
        mock_client.execute_code.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_plot_genomic_neighborhood(self, interpreter, mock_client):
        """Test genomic neighborhood plotting."""
        mock_client.execute_code.return_value = {
            "success": True,
            "stdout": "Genomic neighborhood plot created with 4 genes\n",
            "stderr": "",
            "execution_time": 0.3,
            "files_created": ["genomic_neighborhood.png"]
        }
        
        gene_data = [
            {"id": "gene1", "start": 1000, "end": 2000, "strand": 1, "function": "transport"},
            {"id": "gene2", "start": 2500, "end": 3500, "strand": -1, "function": "regulation"},
            {"id": "gene3", "start": 4000, "end": 5000, "strand": 1, "function": "metabolism"},
            {"id": "gene4", "start": 5500, "end": 6500, "strand": 1, "function": "unknown"}
        ]
        
        result = await interpreter.plot_genomic_neighborhood(gene_data)
        
        assert result["success"] is True
        assert "genomic_neighborhood.png" in result.get("files_created", [])
        mock_client.execute_code.assert_called_once()
        
        # Check that the plotting code includes the gene data
        call_args = mock_client.execute_code.call_args[0][0]
        assert str(gene_data) in call_args
    
    @pytest.mark.asyncio
    async def test_calculate_statistics(self, interpreter, mock_client):
        """Test statistical calculations."""
        mock_client.execute_code.return_value = {
            "success": True,
            "stdout": "Statistical Analysis Results:\nSIMILARITIES:\n  Count: 5\n  Mean: 0.750\n",
            "stderr": "",
            "execution_time": 0.2
        }
        
        data = {
            "similarities": [0.8, 0.7, 0.9, 0.6, 0.75],
            "lengths": [300, 250, 400, 350, 275]
        }
        
        result = await interpreter.calculate_statistics(data)
        
        assert result["success"] is True
        assert "Mean: 0.750" in result["stdout"]
        mock_client.execute_code.assert_called_once()
        
        # Check that the stats code includes the data
        call_args = mock_client.execute_code.call_args[0][0]
        assert str(data) in call_args


class TestCodeInterpreterTool:
    """Test the code_interpreter_tool function."""
    
    @pytest.mark.asyncio
    async def test_tool_function_success(self):
        """Test successful code execution via tool function."""
        mock_result = {
            "success": True,
            "stdout": "Result: 42\n",
            "stderr": "",
            "execution_time": 0.1,
            "session_id": "test-session"
        }
        
        with patch('src.code_interpreter.client.code_interpreter_tool', new_callable=AsyncMock) as mock_tool:
            mock_tool.return_value = mock_result
            
            result = await code_interpreter_tool(
                code="print('Result:', 6 * 7)",
                session_id="test-session",
                timeout=30
            )
            
            assert result["success"] is True
            assert result["stdout"] == "Result: 42\n"
            assert result["session_id"] == "test-session"
            
            mock_tool.assert_called_once_with(
                code="print('Result:', 6 * 7)",
                session_id="test-session",
                timeout=30
            )
    
    @pytest.mark.asyncio
    async def test_tool_function_service_unavailable(self):
        """Test handling when code interpreter service is unavailable."""
        with patch('src.code_interpreter.client.code_interpreter_tool', new_callable=AsyncMock) as mock_tool:
            mock_tool.return_value = {
                "success": False,
                "error": "Code interpreter service is not available",
                "stdout": "",
                "stderr": "",
                "execution_time": 0
            }
            
            result = await code_interpreter_tool(code="print('test')")
            
            assert result["success"] is False
            assert "not available" in result["error"]


class TestCodeInterpreterIntegration:
    """Test integration with the agentic RAG system."""
    
    @pytest.mark.asyncio
    async def test_available_tools_contains_code_interpreter(self):
        """Test that code interpreter is available in AVAILABLE_TOOLS."""
        from src.llm.rag_system import AVAILABLE_TOOLS
        
        assert "code_interpreter" in AVAILABLE_TOOLS
        assert callable(AVAILABLE_TOOLS["code_interpreter"])
    
    @pytest.mark.asyncio
    async def test_task_execution_with_code_interpreter(self):
        """Test executing a code interpreter task in the task graph."""
        from src.llm.rag_system import Task, TaskType, TaskGraph
        
        # Create a code interpreter task
        task = Task(
            task_type=TaskType.TOOL_CALL,
            tool_name="code_interpreter",
            tool_args={
                "code": "import math\nresult = math.sqrt(16)\nprint(f'Square root of 16 is {result}')",
                "timeout": 10
            }
        )
        
        # Create task graph
        graph = TaskGraph()
        task_id = graph.add_task(task)
        
        # Verify task was added correctly
        assert task_id in graph.tasks
        assert graph.tasks[task_id].tool_name == "code_interpreter"
        assert "math.sqrt" in graph.tasks[task_id].tool_args["code"]
    
    def test_task_type_enum_coverage(self):
        """Test that existing TaskType enum covers code execution needs."""
        from src.llm.rag_system import TaskType
        
        # Code execution should use TOOL_CALL task type
        assert TaskType.TOOL_CALL in TaskType
        assert TaskType.ATOMIC_QUERY in TaskType
        assert TaskType.AGGREGATE in TaskType


class TestSecurityFeatures:
    """Test security-related functionality."""
    
    @pytest.mark.asyncio
    async def test_code_execution_timeout_enforcement(self):
        """Test that timeout is properly enforced."""
        client = CodeInterpreterClient()
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post.side_effect = asyncio.TimeoutError()
            
            result = await client.execute_code(
                "import time; time.sleep(100)",  # Long-running code
                timeout=1  # Short timeout
            )
            
            assert result["success"] is False
            assert "timed out" in result["error"].lower()
    
    def test_secure_globals_restrictions(self):
        """Test that the secure environment restricts dangerous operations."""
        # This would be tested in an integration test with actual service
        # For now, we can verify the structure
        if not CODE_INTERPRETER_AVAILABLE:
            pytest.skip("Code interpreter not available")
            
        try:
            from src.code_interpreter.service import SessionManager
            
            manager = SessionManager()
            session = manager.get_or_create_session("test-session")
            
            # Check that secure globals are created
            assert "globals" in session
            assert isinstance(session["globals"], dict)
            
            # Verify safe modules are available
            globals_dict = session["globals"]
            assert "json" in globals_dict
            
            # Verify restricted access to dangerous modules
            # (Full testing would require running the actual service)
        except ImportError:
            pytest.skip("Service dependencies not available")
    
    @pytest.mark.asyncio
    async def test_session_isolation(self):
        """Test that sessions are properly isolated."""
        if not CODE_INTERPRETER_AVAILABLE:
            pytest.skip("Code interpreter not available")
            
        try:
            manager_class = __import__('src.code_interpreter.service', fromlist=['SessionManager']).SessionManager
            manager = manager_class()
            
            # Create two different sessions
            session1 = manager.get_or_create_session("session1")
            session2 = manager.get_or_create_session("session2")
            
            # Verify they are different
            assert session1 is not session2
            assert session1["globals"] is not session2["globals"]
            assert session1["locals"] is not session2["locals"]
        except ImportError:
            pytest.skip("Service dependencies not available")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])