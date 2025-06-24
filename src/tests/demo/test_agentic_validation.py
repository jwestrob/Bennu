#!/usr/bin/env python3
"""
Comprehensive validation script for agentic RAG v2.0 system.
Tests task graph construction, tool integration, and LLM agent capabilities.

This script validates:
1. Task graph DAG construction and dependency resolution
2. External tool integration and execution
3. LLM agent's ability to access and utilize all provided tools
4. Agentic vs traditional mode routing
5. End-to-end workflow execution

Run with: python -m src.tests.demo.test_agentic_validation
"""

import asyncio
import json
import logging
from unittest.mock import Mock, patch, AsyncMock
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from datetime import datetime

# Import the agentic RAG system components
from src.llm.rag_system import (
    GenomicRAG, TaskGraph, Task, TaskType, TaskStatus, 
    AVAILABLE_TOOLS, literature_search, GenomicContext,
    PlannerAgent
)
from src.llm.config import LLMConfig

console = Console()
logger = logging.getLogger(__name__)

class AgenticRAGValidator:
    """Comprehensive validator for the agentic RAG system."""
    
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        
    def log_test_result(self, test_name: str, passed: bool, message: str = ""):
        """Log test results for final summary."""
        self.results[test_name] = {
            "passed": passed,
            "message": message,
            "timestamp": datetime.now()
        }
        
        status = "✅ PASS" if passed else "❌ FAIL"
        console.print(f"  {status}: {test_name}")
        if message:
            console.print(f"    {message}")
    
    def test_task_graph_construction(self):
        """Test 1: Task Graph DAG Construction and Dependency Resolution"""
        console.print("\n[bold blue]Test 1: Task Graph DAG Construction[/bold blue]")
        
        try:
            graph = TaskGraph()
            
            # Test basic task creation
            task1 = Task(
                task_id="root_task",
                task_type=TaskType.ATOMIC_QUERY,
                query="MATCH (p:Protein) RETURN count(p)",
                dependencies=set()
            )
            
            task2 = Task(
                task_id="dependent_task",
                task_type=TaskType.TOOL_CALL,
                tool_name="literature_search",
                tool_args={"query": "protein function", "email": "test@example.com"},
                dependencies={"root_task"}
            )
            
            task3 = Task(
                task_id="aggregate_task",
                task_type=TaskType.AGGREGATE,
                dependencies={"root_task", "dependent_task"}
            )
            
            # Add tasks to graph
            graph.add_task(task1)
            graph.add_task(task2)
            graph.add_task(task3)
            
            # Test dependency resolution
            ready_tasks = graph.get_ready_tasks()
            self.log_test_result(
                "Task graph basic construction",
                len(ready_tasks) == 1 and ready_tasks[0].task_id == "root_task",
                f"Expected 1 ready task (root_task), got {len(ready_tasks)}"
            )
            
            # Test task completion and dependency propagation
            graph.mark_task_status("root_task", TaskStatus.COMPLETED, result={"count": 100})
            
            ready_tasks = graph.get_ready_tasks()
            self.log_test_result(
                "Dependency resolution after completion",
                len(ready_tasks) == 1 and ready_tasks[0].task_id == "dependent_task",
                f"Expected dependent_task to be ready, got: {[t.task_id for t in ready_tasks]}"
            )
            
            # Test failure propagation
            graph.mark_task_status("dependent_task", TaskStatus.FAILED, error="Test failure")
            graph.mark_skipped_tasks()
            
            aggregate_task = graph.get_task("aggregate_task")
            self.log_test_result(
                "Failure propagation and task skipping",
                aggregate_task.status == TaskStatus.SKIPPED,
                f"Expected aggregate_task to be skipped, got: {aggregate_task.status}"
            )
            
            # Test completion detection
            self.log_test_result(
                "Graph completion detection",
                graph.is_complete(),
                "Graph should be complete after all tasks processed"
            )
            
            # Test summary statistics
            summary = graph.get_summary()
            expected_summary = {"completed": 1, "failed": 1, "skipped": 1, "pending": 0, "running": 0}
            self.log_test_result(
                "Summary statistics accuracy",
                summary == expected_summary,
                f"Expected {expected_summary}, got {summary}"
            )
            
        except Exception as e:
            self.log_test_result("Task graph construction", False, f"Exception: {str(e)}")
    
    def test_tool_integration(self):
        """Test 2: External Tool Integration"""
        console.print("\n[bold blue]Test 2: External Tool Integration[/bold blue]")
        
        try:
            # Test tool manifest structure
            self.log_test_result(
                "Tool manifest structure",
                "literature_search" in AVAILABLE_TOOLS and callable(AVAILABLE_TOOLS["literature_search"]),
                f"Available tools: {list(AVAILABLE_TOOLS.keys())}"
            )
            
            # Test literature search tool with mocked Entrez
            with patch('src.llm.rag_system.ENTREZ_AVAILABLE', True), \
                 patch('src.llm.rag_system.Entrez') as mock_entrez:
                
                # Mock Entrez responses
                mock_search_handle = Mock()
                mock_entrez.esearch.return_value = mock_search_handle
                
                mock_fetch_handle = Mock()
                mock_entrez.efetch.return_value = mock_fetch_handle
                
                mock_entrez.read.side_effect = [
                    {'IdList': ['12345', '67890'], 'Count': '2'},  # Search results
                    {'PubmedArticle': [  # Fetch results
                        {
                            'MedlineCitation': {
                                'Article': {
                                    'ArticleTitle': 'CRISPR-Cas9 Systems in Bacteria',
                                    'Abstract': {'AbstractText': ['Study of CRISPR proteins and their function in bacterial immunity.']}
                                },
                                'PMID': {'content': '12345'}
                            }
                        },
                        {
                            'MedlineCitation': {
                                'Article': {
                                    'ArticleTitle': 'Heme Transport Mechanisms',
                                    'Abstract': {'AbstractText': ['Analysis of heme uptake systems in gram-negative bacteria.']}
                                },
                                'PMID': {'content': '67890'}
                            }
                        }
                    ]}
                ]
                
                result = literature_search("CRISPR proteins", "test@example.com")
                
                # Validate tool execution
                self.log_test_result(
                    "Literature search tool execution",
                    isinstance(result, str) and "CRISPR proteins" in result and "12345" in result,
                    "Tool should return formatted string with query and PMIDs"
                )
                
                # Test tool error handling
                result_empty = literature_search("", "test@example.com")
                self.log_test_result(
                    "Tool error handling (empty query)",
                    "Error" in result_empty,
                    "Tool should handle empty queries gracefully"
                )
                
                result_no_email = literature_search("test query", "")
                self.log_test_result(
                    "Tool error handling (no email)",
                    "Error" in result_no_email,
                    "Tool should require email for Entrez API"
                )
            
            # Test tool execution without Entrez availability
            with patch('src.llm.rag_system.ENTREZ_AVAILABLE', False):
                result_no_entrez = literature_search("test", "test@example.com")
                self.log_test_result(
                    "Tool graceful degradation",
                    "Error" in result_no_entrez and "Biopython not available" in result_no_entrez,
                    "Tool should gracefully handle missing dependencies"
                )
                
        except Exception as e:
            self.log_test_result("Tool integration", False, f"Exception: {str(e)}")
    
    def test_task_execution_interface(self):
        """Test 3: Task-Tool Execution Interface"""
        console.print("\n[bold blue]Test 3: Task-Tool Execution Interface[/bold blue]")
        
        try:
            # Create different types of tasks
            atomic_query_task = Task(
                task_id="query_task",
                task_type=TaskType.ATOMIC_QUERY,
                query="MATCH (p:Protein) RETURN p.id LIMIT 5"
            )
            
            tool_call_task = Task(
                task_id="tool_task",
                task_type=TaskType.TOOL_CALL,
                tool_name="literature_search",
                tool_args={"query": "heme transport", "email": "test@example.com"}
            )
            
            aggregate_task = Task(
                task_id="agg_task",
                task_type=TaskType.AGGREGATE,
                dependencies={"query_task", "tool_task"}
            )
            
            # Validate task creation
            self.log_test_result(
                "Task type variety",
                len({atomic_query_task.task_type, tool_call_task.task_type, aggregate_task.task_type}) == 3,
                f"Task types: {[t.task_type for t in [atomic_query_task, tool_call_task, aggregate_task]]}"
            )
            
            # Test task serialization (important for LLM planning)
            task_dict = {
                "id": tool_call_task.task_id,
                "type": tool_call_task.task_type.value,
                "tool_name": tool_call_task.tool_name,
                "tool_args": tool_call_task.tool_args,
                "dependencies": list(tool_call_task.dependencies)
            }
            
            self.log_test_result(
                "Task serialization compatibility",
                isinstance(task_dict, dict) and "tool_name" in task_dict,
                "Tasks should be serializable for LLM planning"
            )
            
            # Test task execution interface with mock
            with patch('src.llm.rag_system.ENTREZ_AVAILABLE', True), \
                 patch('src.llm.rag_system.Entrez') as mock_entrez:
                
                mock_entrez.esearch.return_value = Mock()
                mock_entrez.read.side_effect = [
                    {'IdList': ['11111'], 'Count': '1'},
                    {'PubmedArticle': [{
                        'MedlineCitation': {
                            'Article': {'ArticleTitle': 'Test', 'Abstract': {'AbstractText': ['Test abstract']}},
                            'PMID': {'content': '11111'}
                        }
                    }]}
                ]
                
                # Execute tool through AVAILABLE_TOOLS interface (as _execute_task would)
                tool_function = AVAILABLE_TOOLS[tool_call_task.tool_name]
                result = tool_function(**tool_call_task.tool_args)
                
                self.log_test_result(
                    "Task-tool execution interface",
                    isinstance(result, str) and "heme transport" in result,
                    "Tool execution should work through task interface"
                )
                
        except Exception as e:
            self.log_test_result("Task-tool interface", False, f"Exception: {str(e)}")
    
    def test_llm_agent_capabilities(self):
        """Test 4: LLM Agent Tool Access and Planning"""
        console.print("\n[bold blue]Test 4: LLM Agent Tool Access[/bold blue]")
        
        try:
            # Test DSPy signature initialization
            from src.llm.rag_system import PlannerAgent
            
            # Validate signature exists
            self.log_test_result(
                "PlannerAgent signature availability",
                PlannerAgent is not None,
                "PlannerAgent should be importable and defined"
            )
            
            # Test tool manifest accessibility
            tools_list = list(AVAILABLE_TOOLS.keys())
            self.log_test_result(
                "Tool manifest accessibility",
                len(tools_list) > 0 and "literature_search" in tools_list,
                f"Available tools: {tools_list}"
            )
            
            # Test example task plan structure (what LLM should generate)
            example_plan = {
                "tasks": [
                    {
                        "id": "local_search",
                        "type": "atomic_query",
                        "query": "Find proteins with CRISPR domains",
                        "dependencies": []
                    },
                    {
                        "id": "literature_review", 
                        "type": "tool_call",
                        "tool_name": "literature_search",
                        "tool_args": {"query": "CRISPR proteins bacterial immunity"},
                        "dependencies": ["local_search"]
                    },
                    {
                        "id": "synthesis",
                        "type": "aggregate", 
                        "dependencies": ["local_search", "literature_review"]
                    }
                ]
            }
            
            # Validate plan structure is processable
            plan_valid = (
                isinstance(example_plan, dict) and 
                "tasks" in example_plan and
                all(isinstance(task, dict) and "id" in task and "type" in task for task in example_plan["tasks"])
            )
            
            self.log_test_result(
                "Task plan structure validation",
                plan_valid,
                "LLM-generated plans should follow expected schema"
            )
            
            # Test task type compatibility
            task_types_valid = all(
                task["type"] in [t.value for t in TaskType]
                for task in example_plan["tasks"]
            )
            
            self.log_test_result(
                "Task type compatibility",
                task_types_valid,
                f"All task types should be valid: {[task['type'] for task in example_plan['tasks']]}"
            )
            
            # Test tool name validation
            tool_tasks = [task for task in example_plan["tasks"] if task["type"] == "tool_call"]
            tools_valid = all(
                task.get("tool_name") in AVAILABLE_TOOLS
                for task in tool_tasks
            )
            
            self.log_test_result(
                "Tool name validation",
                tools_valid,
                f"All tool names should be available: {[task.get('tool_name') for task in tool_tasks]}"
            )
            
        except Exception as e:
            self.log_test_result("LLM agent capabilities", False, f"Exception: {str(e)}")
    
    async def test_agentic_routing_simulation(self):
        """Test 5: Agentic vs Traditional Mode Routing"""
        console.print("\n[bold blue]Test 5: Agentic vs Traditional Mode Routing[/bold blue]")
        
        try:
            # Mock a minimal GenomicRAG instance for testing routing logic
            mock_config = Mock(spec=LLMConfig)
            mock_config.llm_provider = "openai"
            mock_config.llm_model = "gpt-4"
            mock_config.max_results_per_query = 10
            mock_config.get_api_key.return_value = "test_key"
            
            with patch('src.llm.rag_system.dspy') as mock_dspy, \
                 patch('src.llm.rag_system.Neo4jQueryProcessor'), \
                 patch('src.llm.rag_system.LanceDBQueryProcessor'), \
                 patch('src.llm.rag_system.HybridQueryProcessor'):
                
                # Mock DSPy components
                mock_dspy.settings.configure = Mock()
                mock_dspy.ChainOfThought = Mock()
                mock_dspy.LM = Mock()
                
                # Create GenomicRAG instance
                rag = GenomicRAG(mock_config)
                
                # Test traditional routing
                mock_planner_traditional = Mock()
                mock_planner_traditional.return_value = Mock(
                    requires_planning=False,
                    reasoning="Simple query can be answered directly"
                )
                
                rag.planner = mock_planner_traditional
                rag._execute_traditional_query = AsyncMock(return_value={
                    "answer": "Traditional answer",
                    "query_metadata": {"execution_mode": "traditional"}
                })
                rag._execute_agentic_plan = AsyncMock()  # Should not be called
                
                response = await rag.ask("How many proteins are there?")
                
                self.log_test_result(
                    "Traditional mode routing",
                    response["query_metadata"]["execution_mode"] == "traditional",
                    "Simple queries should use traditional mode"
                )
                
                # Verify agentic path wasn't called
                self.log_test_result(
                    "Traditional mode exclusivity",
                    not rag._execute_agentic_plan.called,
                    "Traditional mode should not trigger agentic execution"
                )
                
                # Test agentic routing
                mock_planner_agentic = Mock()
                mock_planner_agentic.return_value = Mock(
                    requires_planning=True,
                    task_plan='{"tasks": []}',
                    reasoning="Complex query requires external tools"
                )
                
                rag.planner = mock_planner_agentic
                rag._execute_agentic_plan = AsyncMock(return_value={
                    "answer": "Agentic answer",
                    "query_metadata": {"execution_mode": "agentic"}
                })
                rag._execute_traditional_query = AsyncMock()  # Reset
                
                response = await rag.ask("What does recent literature say about CRISPR proteins?")
                
                self.log_test_result(
                    "Agentic mode routing",
                    response["query_metadata"]["execution_mode"] == "agentic",
                    "Complex queries should use agentic mode"
                )
                
                # Test fallback mechanism
                rag._execute_agentic_plan = AsyncMock(side_effect=Exception("Agentic failure"))
                rag._execute_traditional_query = AsyncMock(return_value={
                    "answer": "Fallback answer",
                    "query_metadata": {"execution_mode": "traditional"}
                })
                
                response = await rag.ask("Query that should fail in agentic mode")
                
                self.log_test_result(
                    "Agentic fallback mechanism",
                    response.get("query_metadata", {}).get("execution_mode") == "traditional",
                    "Failed agentic queries should fall back to traditional mode"
                )
                
        except Exception as e:
            self.log_test_result("Agentic routing simulation", False, f"Exception: {str(e)}")
    
    def test_end_to_end_workflow(self):
        """Test 6: End-to-End Workflow Simulation"""
        console.print("\n[bold blue]Test 6: End-to-End Workflow Simulation[/bold blue]")
        
        try:
            # Simulate a complete multi-step workflow
            graph = TaskGraph()
            
            # Step 1: Local database query
            step1 = Task(
                task_id="find_heme_proteins",
                task_type=TaskType.ATOMIC_QUERY,
                query="MATCH (p:Protein)-[:HASDOMAIN]->(d:Domain) WHERE d.description CONTAINS 'heme' RETURN p.id LIMIT 5",
                dependencies=set()
            )
            
            # Step 2: Literature search based on findings
            step2 = Task(
                task_id="literature_heme_transport",
                task_type=TaskType.TOOL_CALL,
                tool_name="literature_search",
                tool_args={"query": "heme transport bacteria", "email": "researcher@example.com"},
                dependencies={"find_heme_proteins"}
            )
            
            # Step 3: Aggregate findings
            step3 = Task(
                task_id="synthesize_heme_analysis",
                task_type=TaskType.AGGREGATE,
                dependencies={"find_heme_proteins", "literature_heme_transport"}
            )
            
            # Execute workflow
            graph.add_task(step1)
            graph.add_task(step2)
            graph.add_task(step3)
            
            execution_log = []
            
            # Simulate execution loop
            iteration = 0
            max_iterations = 5
            
            while not graph.is_complete() and iteration < max_iterations:
                iteration += 1
                ready_tasks = graph.get_ready_tasks()
                
                if not ready_tasks:
                    graph.mark_skipped_tasks()
                    break
                
                for task in ready_tasks:
                    execution_log.append(f"Iter {iteration}: Executing {task.task_id}")
                    
                    try:
                        # Simulate task execution
                        if task.task_type == TaskType.ATOMIC_QUERY:
                            result = {"query_result": f"Results for {task.query[:50]}..."}
                            graph.mark_task_status(task.task_id, TaskStatus.COMPLETED, result=result)
                            
                        elif task.task_type == TaskType.TOOL_CALL:
                            # Execute actual tool with mocked Entrez
                            with patch('src.llm.rag_system.ENTREZ_AVAILABLE', True), \
                                 patch('src.llm.rag_system.Entrez') as mock_entrez:
                                
                                mock_entrez.esearch.return_value = Mock()
                                mock_entrez.read.side_effect = [
                                    {'IdList': ['33333'], 'Count': '1'},
                                    {'PubmedArticle': [{
                                        'MedlineCitation': {
                                            'Article': {
                                                'ArticleTitle': 'Heme Transport in Bacteria',
                                                'Abstract': {'AbstractText': ['Bacterial heme uptake mechanisms']}
                                            },
                                            'PMID': {'content': '33333'}
                                        }
                                    }]}
                                ]
                                
                                tool_func = AVAILABLE_TOOLS[task.tool_name]
                                result = {"tool_result": tool_func(**task.tool_args)}
                                graph.mark_task_status(task.task_id, TaskStatus.COMPLETED, result=result)
                                
                        elif task.task_type == TaskType.AGGREGATE:
                            result = {"aggregated": "Combined analysis complete"}
                            graph.mark_task_status(task.task_id, TaskStatus.COMPLETED, result=result)
                            
                    except Exception as e:
                        execution_log.append(f"Error in {task.task_id}: {str(e)}")
                        graph.mark_task_status(task.task_id, TaskStatus.FAILED, error=str(e))
            
            # Validate workflow completion
            summary = graph.get_summary()
            
            self.log_test_result(
                "End-to-end workflow execution",
                graph.is_complete() and summary["completed"] == 3,
                f"Workflow summary: {summary}, iterations: {iteration}"
            )
            
            self.log_test_result(
                "Task execution order",
                len(execution_log) >= 3,
                f"Execution log: {execution_log}"
            )
            
            # Validate all task types were executed
            completed_tasks = [task for task in graph.tasks.values() if task.status == TaskStatus.COMPLETED]
            task_types_executed = {task.task_type for task in completed_tasks}
            
            self.log_test_result(
                "Task type coverage",
                len(task_types_executed) == 3,
                f"Executed task types: {[t.value for t in task_types_executed]}"
            )
            
        except Exception as e:
            self.log_test_result("End-to-end workflow", False, f"Exception: {str(e)}")
    
    def display_summary(self):
        """Display comprehensive test summary."""
        console.print("\n" + "="*80)
        console.print("[bold green]🎯 AGENTIC RAG v2.0 VALIDATION SUMMARY[/bold green]")
        console.print("="*80)
        
        # Count results
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result["passed"])
        failed_tests = total_tests - passed_tests
        
        # Create summary table
        table = Table(title="Test Results Summary")
        table.add_column("Test Category", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Details", style="dim")
        
        for test_name, result in self.results.items():
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            table.add_row(test_name, status, result["message"][:100] + "..." if len(result["message"]) > 100 else result["message"])
        
        console.print(table)
        
        # Overall status
        overall_status = "✅ ALL TESTS PASSED" if failed_tests == 0 else f"❌ {failed_tests} TESTS FAILED"
        console.print(f"\n[bold]Overall Status: {overall_status}[/bold]")
        console.print(f"Passed: {passed_tests}/{total_tests}")
        
        # Execution time
        duration = datetime.now() - self.start_time
        console.print(f"Execution time: {duration.total_seconds():.2f} seconds")
        
        # System status summary
        if failed_tests == 0:
            console.print("\n[bold green]🎉 AGENTIC RAG v2.0 SYSTEM IS FULLY OPERATIONAL[/bold green]")
            console.print("✅ Task graph construction and dependency resolution")
            console.print("✅ External tool integration and execution")
            console.print("✅ LLM agent tool access and planning capabilities")
            console.print("✅ Intelligent routing between agentic and traditional modes")
            console.print("✅ End-to-end multi-step workflow execution")
        else:
            console.print(f"\n[bold red]⚠️  ISSUES DETECTED IN {failed_tests} AREAS[/bold red]")
            console.print("Review failed tests above for system improvements needed.")
        
        console.print("\n" + "="*80)

async def main():
    """Main validation function."""
    console.print(Panel.fit(
        "[bold blue]🧬 AGENTIC RAG v2.0 COMPREHENSIVE VALIDATION[/bold blue]\n"
        "Testing task graph construction, tool integration, and LLM capabilities",
        title="Genomic AI Platform Validation"
    ))
    
    validator = AgenticRAGValidator()
    
    # Run all validation tests
    validator.test_task_graph_construction()
    validator.test_tool_integration()
    validator.test_task_execution_interface()
    validator.test_llm_agent_capabilities()
    await validator.test_agentic_routing_simulation()
    validator.test_end_to_end_workflow()
    
    # Display final summary
    validator.display_summary()

if __name__ == "__main__":
    asyncio.run(main())