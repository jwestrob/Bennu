#!/usr/bin/env python3
"""
Debug script to capture and display the exact context data presented to the LLM during RAG.
This helps analyze what information sources are being integrated and how they're formatted.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.llm.config import LLMConfig
from src.llm.rag_system import GenomicRAG

console = Console()


class RAGContextDebugger:
    """Captures and analyzes the context data fed to the LLM."""
    
    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig.from_env()
        self.rag = GenomicRAG(self.config)
        
    async def debug_query(self, question: str) -> Dict[str, Any]:
        """Run a query and capture all context data."""
        console.print(f"[bold blue]🔍 Debugging RAG Context for:[/bold blue] {question}")
        
        # Monkey patch the RAG system to capture context
        original_format_context = self.rag._format_context
        original_execute_task = self.rag._execute_task
        captured_context = {
            'agentic_tasks': [],
            'task_results': {},
            'is_agentic': False
        }
        
        def capture_format_context(context):
            # Call original method
            formatted_context = original_format_context(context)
            
            # Capture all the data
            captured_context.update({
                'formatted_context': formatted_context,
                'neo4j_raw_data': context.structured_data if hasattr(context, 'structured_data') else [],
                'lancedb_raw_data': context.semantic_data if hasattr(context, 'semantic_data') else [],
                'neo4j_record_count': len(context.structured_data) if hasattr(context, 'structured_data') and context.structured_data else 0,
                'lancedb_record_count': len(context.semantic_data) if hasattr(context, 'semantic_data') and context.semantic_data else 0
            })
            
            return formatted_context
        
        async def capture_execute_task(task, previous_results):
            """Capture agentic task execution data."""
            if not captured_context['is_agentic']:
                captured_context['is_agentic'] = True
                
            # Capture task info
            task_info = {
                'id': task.task_id,
                'type': task.task_type.value,
                'status': task.status.value,
                'dependencies': list(task.dependencies),
                'content': getattr(task, 'query', getattr(task, 'tool_name', 'N/A'))
            }
            
            # Add to tasks list if not already there
            if not any(t['id'] == task.task_id for t in captured_context['agentic_tasks']):
                captured_context['agentic_tasks'].append(task_info)
            
            # Call original method and capture results
            result = await original_execute_task(task, previous_results)
            captured_context['task_results'][task.task_id] = result
            
            return result
        
        # Replace methods temporarily
        self.rag._format_context = capture_format_context
        self.rag._execute_task = capture_execute_task
        
        try:
            # Run the query
            result = await self.rag.ask(question)
            
            # Add result info to captured data
            captured_context['llm_response'] = result
            
            return captured_context
            
        finally:
            # Restore original methods
            self.rag._format_context = original_format_context
            self.rag._execute_task = original_execute_task
    
    def display_context_analysis(self, context_data: Dict[str, Any]):
        """Display comprehensive analysis of the context data."""
        
        # Overview
        console.print("\n" + "="*80)
        console.print("[bold green]📊 RAG CONTEXT ANALYSIS[/bold green]")
        console.print("="*80)
        
        # Data Sources Summary
        neo4j_count = context_data.get('neo4j_record_count', 0)
        lancedb_count = context_data.get('lancedb_record_count', 0)
        is_agentic = context_data.get('is_agentic', False)
        task_count = len(context_data.get('agentic_tasks', []))
        
        summary_table = Table(title="Data Sources Summary")
        summary_table.add_column("Source", style="cyan")
        summary_table.add_column("Records/Tasks", style="magenta")
        summary_table.add_column("Status", style="green")
        
        summary_table.add_row("Query Type", "Agentic" if is_agentic else "Traditional", "🤖" if is_agentic else "📊")
        summary_table.add_row("Neo4j (Structured)", str(neo4j_count), "✅ Active" if neo4j_count > 0 else "❌ No data")
        summary_table.add_row("LanceDB (Semantic)", str(lancedb_count), "✅ Active" if lancedb_count > 0 else "❌ No data")
        if is_agentic:
            summary_table.add_row("Agentic Tasks", str(task_count), "🔄 Executed" if task_count > 0 else "❌ None")
        
        console.print(summary_table)
        
        # Agentic Task Analysis
        if is_agentic and context_data.get('agentic_tasks'):
            console.print("\n[bold cyan]🤖 AGENTIC TASK EXECUTION[/bold cyan]")
            self._analyze_agentic_tasks(context_data['agentic_tasks'], context_data.get('task_results', {}))
        
        # Neo4j Data Analysis
        if context_data.get('neo4j_raw_data'):
            console.print("\n[bold blue]🗃️  NEO4J STRUCTURED DATA[/bold blue]")
            self._analyze_neo4j_data(context_data['neo4j_raw_data'])
        
        # LanceDB Data Analysis  
        if context_data.get('lancedb_raw_data'):
            console.print("\n[bold purple]🔍 LANCEDB SEMANTIC DATA[/bold purple]")
            self._analyze_lancedb_data(context_data['lancedb_raw_data'])
        
        # Formatted Context (what LLM actually sees)
        if context_data.get('formatted_context'):
            console.print("\n[bold yellow]🤖 FORMATTED CONTEXT (LLM INPUT)[/bold yellow]")
            context_text = context_data['formatted_context']
            
            # Show length and structure
            console.print(f"[dim]Context length: {len(context_text)} characters[/dim]")
            
            # Display formatted context in a panel
            console.print(Panel(
                context_text[:2000] + "\n\n[dim]... (truncated, full context below)[/dim]" if len(context_text) > 2000 else context_text,
                title="LLM Context Input",
                border_style="yellow"
            ))
            
            # Full context for analysis
            if len(context_text) > 2000:
                console.print("\n[bold]📄 FULL CONTEXT:[/bold]")
                console.print(context_text)
        
        # LLM Response Analysis
        if context_data.get('llm_response'):
            console.print("\n[bold green]🤖 LLM RESPONSE ANALYSIS[/bold green]")
            response = context_data['llm_response']
            
            response_table = Table()
            response_table.add_column("Attribute", style="cyan")
            response_table.add_column("Value", style="white")
            
            response_table.add_row("Answer Length", f"{len(response.get('answer', ''))} characters")
            response_table.add_row("Confidence", str(response.get('confidence', 'unknown')))
            response_table.add_row("Citations", str(response.get('citations', 'none')))
            response_table.add_row("Query Type", str(response.get('query_type', 'unknown')))
            
            console.print(response_table)
            
            console.print(f"\\n[bold]LLM Answer:[/bold]\\n{response.get('answer', 'No answer')}")
    
    def _analyze_agentic_tasks(self, tasks, task_results):
        """Analyze agentic task execution."""
        if not tasks:
            console.print("[dim]No agentic tasks[/dim]")
            return
            
        # Task execution summary
        task_table = Table(title="Agentic Task Execution")
        task_table.add_column("Task ID", style="cyan")
        task_table.add_column("Type", style="magenta")
        task_table.add_column("Status", style="green")
        task_table.add_column("Dependencies", style="yellow")
        task_table.add_column("Content", style="white")
        
        for task in tasks:
            status_emoji = {
                'completed': '✅',
                'failed': '❌',
                'pending': '⏳',
                'running': '🔄'
            }.get(task['status'], '❓')
            
            deps_str = ', '.join(task['dependencies']) if task['dependencies'] else 'None'
            content_preview = str(task['content'])[:50] + '...' if len(str(task['content'])) > 50 else str(task['content'])
            
            task_table.add_row(
                task['id'],
                task['type'],
                f"{status_emoji} {task['status']}",
                deps_str,
                content_preview
            )
        
        console.print(task_table)
        
        # Task results analysis
        if task_results:
            console.print("\\n[bold]📊 Task Results Summary:[/bold]")
            for task_id, result in task_results.items():
                console.print(f"\\n[cyan]{task_id}:[/cyan]")
                if isinstance(result, dict):
                    if 'context' in result:
                        context = result['context']
                        structured_count = len(context.structured_data) if hasattr(context, 'structured_data') and context.structured_data else 0
                        semantic_count = len(context.semantic_data) if hasattr(context, 'semantic_data') and context.semantic_data else 0
                        console.print(f"  Neo4j records: {structured_count}")
                        console.print(f"  LanceDB records: {semantic_count}")
                    elif 'tool_result' in result:
                        tool_result = result['tool_result']
                        success = tool_result.get('success', False)
                        console.print(f"  Tool execution: {'✅ Success' if success else '❌ Failed'}")
                        if 'error' in tool_result:
                            console.print(f"  Error: {tool_result['error']}")
                        if 'stdout' in tool_result and tool_result['stdout']:
                            stdout_preview = tool_result['stdout'][:200] + '...' if len(tool_result['stdout']) > 200 else tool_result['stdout']
                            console.print(f"  Output: {stdout_preview}")
                    else:
                        console.print(f"  Result type: {type(result)}")
                        console.print(f"  Keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                else:
                    console.print(f"  Result: {str(result)[:100]}...")
    
    def _analyze_neo4j_data(self, neo4j_data):
        """Analyze Neo4j structured data."""
        if not neo4j_data:
            console.print("[dim]No Neo4j data[/dim]")
            return
            
        # Analyze data types and fields
        field_analysis = {}
        for record in neo4j_data:
            for key, value in record.items():
                if key not in field_analysis:
                    field_analysis[key] = {'count': 0, 'sample_values': [], 'types': set()}
                
                field_analysis[key]['count'] += 1
                field_analysis[key]['types'].add(type(value).__name__)
                
                if len(field_analysis[key]['sample_values']) < 3 and value:
                    field_analysis[key]['sample_values'].append(str(value)[:50])
        
        # Display field analysis
        neo4j_table = Table(title="Neo4j Data Structure")
        neo4j_table.add_column("Field", style="cyan")
        neo4j_table.add_column("Count", style="magenta")
        neo4j_table.add_column("Types", style="green")
        neo4j_table.add_column("Sample Values", style="white")
        
        for field, info in field_analysis.items():
            types_str = ", ".join(info['types'])
            samples_str = " | ".join(info['sample_values'])
            neo4j_table.add_row(field, str(info['count']), types_str, samples_str)
        
        console.print(neo4j_table)
        
        # Show first few complete records
        console.print("\n[dim]Sample Neo4j Records:[/dim]")
        for i, record in enumerate(neo4j_data[:2]):
            console.print(f"[dim]Record {i+1}:[/dim] {json.dumps(record, indent=2, default=str)[:300]}...")
    
    def _analyze_lancedb_data(self, lancedb_data):
        """Analyze LanceDB semantic search data."""
        if not lancedb_data:
            console.print("[dim]No LanceDB data[/dim]")
            return
            
        # Analyze similarity scores and metadata
        if lancedb_data:
            scores = [item.get('similarity_score', 0) for item in lancedb_data if isinstance(item, dict)]
            avg_score = sum(scores) / len(scores) if scores else 0
            
            lancedb_table = Table(title="LanceDB Semantic Results")
            lancedb_table.add_column("Metric", style="cyan")
            lancedb_table.add_column("Value", style="magenta")
            
            lancedb_table.add_row("Total Results", str(len(lancedb_data)))
            lancedb_table.add_row("Avg Similarity", f"{avg_score:.3f}")
            lancedb_table.add_row("Score Range", f"{min(scores):.3f} - {max(scores):.3f}" if scores else "N/A")
            
            console.print(lancedb_table)
            
            # Show sample results
            console.print("\n[dim]Sample LanceDB Results:[/dim]")
            for i, item in enumerate(lancedb_data[:2]):
                console.print(f"[dim]Result {i+1}:[/dim] {json.dumps(item, indent=2, default=str)[:300]}...")
    
    def close(self):
        """Clean up resources."""
        self.rag.close()


async def main():
    """Main debug function."""
    if len(sys.argv) < 2:
        console.print("[red]Usage: python debug_rag_context.py \"Your question here\"[/red]")
        console.print("\n[yellow]Example questions:[/yellow]")
        console.print("  • \"What ribosomal proteins are present?\"")
        console.print("  • \"Find proteins similar to transporters\"")
        console.print("  • \"What is the function of KEGG ortholog K02876?\"")
        sys.exit(1)
    
    question = sys.argv[1]
    
    debugger = RAGContextDebugger()
    
    try:
        # Debug the query
        context_data = await debugger.debug_query(question)
        
        # Display comprehensive analysis
        debugger.display_context_analysis(context_data)
        
        # Save raw data for further analysis
        output_file = Path("rag_context_debug.json")
        with open(output_file, 'w') as f:
            json.dump(context_data, f, indent=2, default=str)
        
        console.print(f"\n[green]💾 Raw debug data saved to: {output_file}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during debugging: {e}[/red]")
        import traceback
        traceback.print_exc()
    
    finally:
        debugger.close()


if __name__ == "__main__":
    asyncio.run(main())