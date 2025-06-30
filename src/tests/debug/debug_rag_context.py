#!/usr/bin/env python3
"""
Debug script to capture and display the exact context data presented to the LLM during RAG.
This helps analyze what information sources are being integrated and how they're formatted.
Uses automatic file redirection for extremely large outputs to prevent STDOUT pollution.
"""

import asyncio
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.llm.config import LLMConfig
from src.llm.rag_system import GenomicRAG
from src.utils.command_runner import CommandRunner

console = Console()


class RAGContextDebugger:
    """Captures and analyzes the context data fed to the LLM with optional file redirection for massive outputs."""
    
    def __init__(self, config: LLMConfig = None, chunk_context_size: int = 4096):
        self.config = config or LLMConfig.from_env()
        self.chunk_context_size = chunk_context_size
        self.rag = GenomicRAG(self.config, chunk_context_size=chunk_context_size)
        
        # Initialize file redirection system for truly massive outputs
        self.command_runner = CommandRunner(output_dir="data/debug_outputs", max_stdout_lines=30)
        self.debug_dir = Path("data/debug_outputs")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    async def debug_query(self, question: str) -> Dict[str, Any]:
        """Run a query and capture all context data."""
        console.print(f"[bold blue]🔍 Debugging RAG Context for:[/bold blue] {question}")
        
        # Monkey patch the RAG system to capture context
        original_format_context = self.rag._format_context
        original_execute_task = self.rag._execute_task
        original_answerer = self.rag.answerer
        captured_context = {
            'question': question,
            'timestamp': self.session_timestamp,
            'agentic_tasks': [],
            'task_results': {},
            'is_agentic': False,
            'formatted_context': None,
            'context_length': 0,
            'file_outputs': [],
            'neo4j_error': None,
            'llm_calls': []
        }
        
        def capture_format_context(context):
            # Call original method
            try:
                formatted_context = original_format_context(context)
                
                # Capture all the data
                captured_context.update({
                    'formatted_context': formatted_context,
                    'context_length': len(formatted_context),
                    'neo4j_raw_data': context.structured_data if hasattr(context, 'structured_data') else [],
                    'lancedb_raw_data': context.semantic_data if hasattr(context, 'semantic_data') else [],
                    'neo4j_record_count': len(context.structured_data) if hasattr(context, 'structured_data') and context.structured_data else 0,
                    'lancedb_record_count': len(context.semantic_data) if hasattr(context, 'semantic_data') and context.semantic_data else 0
                })
                
                # Only redirect to file if context is truly massive (>50,000 chars)
                if len(formatted_context) > 50000:  # Much higher threshold
                    context_file = self.debug_dir / f"{self.session_timestamp}_massive_context.txt"
                    with open(context_file, 'w') as f:
                        f.write(f"Question: {question}\n")
                        f.write(f"Timestamp: {self.session_timestamp}\n")
                        f.write(f"Context Length: {len(formatted_context)} characters\n")
                        f.write(f"Chunk Context Size: {self.chunk_context_size}\n")
                        f.write("=" * 80 + "\n")
                        f.write(formatted_context)
                    
                    console.print(f"📁 Massive context ({len(formatted_context):,} chars) saved to: {context_file}")
                    captured_context['context_file'] = str(context_file)
                    captured_context['file_outputs'].append(str(context_file))
                
                return formatted_context
                
            except Exception as e:
                console.print(f"[red]❌ Error in format_context: {e}[/red]")
                captured_context['format_context_error'] = str(e)
                return f"Error formatting context: {e}"
        
        def capture_answerer(*args, **kwargs):
            """Capture the answerer calls to see what context is being passed."""
            try:
                # Extract context from kwargs
                context_arg = kwargs.get('context', '')
                if context_arg and not captured_context.get('formatted_context'):
                    # This is likely an agentic query where context comes through answerer
                    captured_context['formatted_context'] = context_arg
                    captured_context['context_length'] = len(context_arg)
                    
                    # Save massive contexts to file
                    if len(context_arg) > 50000:
                        context_file = self.debug_dir / f"{self.session_timestamp}_agentic_context.txt"
                        with open(context_file, 'w') as f:
                            f.write(f"Question: {question}\n")
                            f.write(f"Timestamp: {self.session_timestamp}\n")
                            f.write(f"Context Length: {len(context_arg)} characters\n")
                            f.write(f"Chunk Context Size: {self.chunk_context_size}\n")
                            f.write("=" * 80 + "\n")
                            f.write(context_arg)
                        
                        console.print(f"📁 Massive agentic context ({len(context_arg):,} chars) saved to: {context_file}")
                        captured_context['context_file'] = str(context_file)
                        captured_context['file_outputs'].append(str(context_file))
                
                # Record the LLM call
                captured_context['llm_calls'].append({
                    'args': args,
                    'kwargs': {k: v[:500] + "..." if isinstance(v, str) and len(v) > 500 else v for k, v in kwargs.items()},
                    'context_length': len(context_arg) if context_arg else 0
                })
                
                # Call original answerer
                return original_answerer(*args, **kwargs)
                
            except Exception as e:
                console.print(f"[red]❌ Error in answerer: {e}[/red]")
                captured_context['answerer_error'] = str(e)
                return original_answerer(*args, **kwargs)
        
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
            try:
                result = await original_execute_task(task, previous_results)
                captured_context['task_results'][task.task_id] = result
                
                # Only save to file if result is truly massive (>100,000 chars)
                if result and len(str(result)) > 100000:
                    task_file = self.debug_dir / f"{self.session_timestamp}_massive_task_{task.task_id}.json"
                    with open(task_file, 'w') as f:
                        json.dump({
                            'task_id': task.task_id,
                            'task_type': task.task_type.value,
                            'result': result
                        }, f, indent=2, default=str)
                    
                    console.print(f"📁 Massive task result ({len(str(result)):,} chars) saved to: {task_file}")
                    captured_context['file_outputs'].append(str(task_file))
                
                return result
                
            except Exception as e:
                console.print(f"[red]❌ Error in task execution ({task.task_id}): {e}[/red]")
                captured_context['task_results'][task.task_id] = f"Error: {e}"
                return None
        
        # Replace methods temporarily
        self.rag._format_context = capture_format_context
        self.rag._execute_task = capture_execute_task
        self.rag.answerer = capture_answerer
        
        try:
            # Run the query
            result = await self.rag.ask(question)
            
            # Add result info to captured data
            captured_context['llm_response'] = result
            
            return captured_context
            
        except Exception as e:
            console.print(f"[red]❌ Error during query execution: {e}[/red]")
            captured_context['query_error'] = str(e)
            return captured_context
            
        finally:
            # Restore original methods
            self.rag._format_context = original_format_context
            self.rag._execute_task = original_execute_task
            self.rag.answerer = original_answerer
    
    def display_context_analysis(self, context_data: Dict[str, Any]):
        """Display comprehensive analysis of the context data - RESTORED ORIGINAL FUNCTIONALITY."""
        
        # Overview
        console.print("\n" + "="*80)
        console.print("[bold green]📊 RAG CONTEXT ANALYSIS[/bold green]")
        console.print("="*80)
        
        # Check for errors first
        if context_data.get('query_error'):
            console.print(f"[red]❌ Query Error: {context_data['query_error']}[/red]")
        if context_data.get('format_context_error'):
            console.print(f"[red]❌ Context Format Error: {context_data['format_context_error']}[/red]")
        if context_data.get('answerer_error'):
            console.print(f"[red]❌ Answerer Error: {context_data['answerer_error']}[/red]")
        if context_data.get('neo4j_error'):
            console.print(f"[red]❌ Neo4j Error: {context_data['neo4j_error']}[/red]")
        
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
        
        # RESTORED: Formatted Context (what LLM actually sees)
        if context_data.get('formatted_context'):
            console.print("\n[bold yellow]🤖 FORMATTED CONTEXT (LLM INPUT)[/bold yellow]")
            context_text = context_data['formatted_context']
            
            # Show length and structure
            console.print(f"[dim]Context length: {len(context_text)} characters[/dim]")
            
            # Display formatted context in a panel
            if len(context_text) <= 50000:  # Show directly if not massive
                console.print(Panel(
                    context_text[:2000] + "\n\n[dim]... (truncated, full context below)[/dim]" if len(context_text) > 2000 else context_text,
                    title="LLM Context Input",
                    border_style="yellow"
                ))
                
                # Full context for analysis
                if len(context_text) > 2000:
                    console.print("\n[bold]📄 FULL CONTEXT:[/bold]")
                    console.print(context_text)
            else:
                console.print(f"[yellow]📁 Massive context ({len(context_text):,} chars) saved to file[/yellow]")
        else:
            console.print("\n[bold yellow]🤖 FORMATTED CONTEXT (LLM INPUT)[/bold yellow]")
            console.print("[dim]No formatted context captured - this may be an agentic query with direct tool results[/dim]")
        
        # LLM Call Analysis
        if context_data.get('llm_calls'):
            console.print(f"\n[bold magenta]🧠 LLM CALLS ANALYSIS[/bold magenta]")
            console.print(f"[dim]Number of LLM calls: {len(context_data['llm_calls'])}[/dim]")
            for i, call in enumerate(context_data['llm_calls']):
                console.print(f"  Call {i+1}: Context length {call['context_length']} chars")
        
        # RESTORED: LLM Response Analysis
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
            
            console.print(f"\n[bold]LLM Answer:[/bold]\n{response.get('answer', 'No answer')}")
        
        # File outputs summary (only if files were created)
        file_outputs = context_data.get('file_outputs', [])
        if file_outputs:
            console.print("\n[bold yellow]📁 Large Files Created:[/bold yellow]")
            for file_path in file_outputs:
                file_name = Path(file_path).name
                file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
                console.print(f"  • {file_name} ({file_size:,} bytes)")
    
    def _analyze_agentic_tasks(self, tasks, task_results):
        """Analyze agentic task execution."""
        if not tasks:
            console.print("[dim]No agentic tasks[/dim]")
            return
            
        task_table = Table(title="Task Execution Details")
        task_table.add_column("Task ID", style="cyan")
        task_table.add_column("Type", style="magenta")
        task_table.add_column("Status", style="green")
        task_table.add_column("Dependencies", style="yellow")
        task_table.add_column("Result Size", style="blue")
        
        for task in tasks:
            task_id = task['id']
            result = task_results.get(task_id, {})
            result_size = len(str(result)) if result else 0
            
            task_table.add_row(
                task_id,
                task['type'],
                task['status'],
                ', '.join(task['dependencies']) if task['dependencies'] else "None",
                f"{result_size:,} chars" if result_size > 0 else "No result"
            )
        
        console.print(task_table)
    
    def _analyze_neo4j_data(self, data):
        """Analyze Neo4j structured data."""
        if not data:
            console.print("[dim]No Neo4j data available[/dim]")
            return
        
        console.print(f"[bold]Records found:[/bold] {len(data)}")
        
        # Show sample record structure
        if data:
            sample_record = data[0]
            console.print(f"[bold]Sample record keys:[/bold] {list(sample_record.keys())}")
            
            # Show first few records
            for i, record in enumerate(data[:min(5, len(data))]):
                console.print(f"\n[bold]Record {i+1}:[/bold]")
                for key, value in record.items():
                    console.print(f"  {key}: {value}")
    
    def _analyze_lancedb_data(self, data):
        """Analyze LanceDB semantic data."""
        if not data:
            console.print("[dim]No LanceDB data available[/dim]")
            return
        
        console.print(f"[bold]Records found:[/bold] {len(data)}")
        
        # Show sample record structure
        if data:
            sample_record = data[0]
            console.print(f"[bold]Sample record keys:[/bold] {list(sample_record.keys())}")
            
            # Show similarity scores if available
            similarities = [r.get('similarity', 0) for r in data if 'similarity' in r]
            if similarities:
                console.print(f"[bold]Similarity range:[/bold] {min(similarities):.3f} - {max(similarities):.3f}")
            
            # Show first few records
            for i, record in enumerate(data[:min(3, len(data))]):
                console.print(f"\n[bold]Record {i+1}:[/bold]")
                for key, value in record.items():
                    if key == 'similarity':
                        console.print(f"  {key}: {value:.3f}")
                    else:
                        console.print(f"  {key}: {value}")
    
    def close(self):
        """Clean up resources."""
        pass


async def main():
    """Main debug function."""
    parser = argparse.ArgumentParser(description="Debug RAG context for genomic queries")
    parser.add_argument("question", help="The question to debug")
    parser.add_argument("-c", "--chunk-context-size", type=int, default=4096,
                       help="Chunk context size for streaming (default: 4096)")
    
    try:
        args = parser.parse_args()
    except SystemExit:
        console.print("[red]Usage: python -m src.tests.debug.debug_rag_context \"Your question here\" [-c CHUNK_SIZE][/red]")
        console.print("\n[yellow]Example questions:[/yellow]")
        console.print("  • \"What ribosomal proteins are present?\"")
        console.print("  • \"Find proteins similar to transporters\"")
        console.print("  • \"What is the function of KEGG ortholog K02876?\"")
        console.print("\n[yellow]Options:[/yellow]")
        console.print("  • -c, --chunk-context-size: Set chunk size for streaming (default: 4096)")
        return
    
    question = args.question
    chunk_context_size = args.chunk_context_size
    
    console.print(f"[dim]Using chunk context size: {chunk_context_size}[/dim]")
    debugger = RAGContextDebugger(chunk_context_size=chunk_context_size)
    
    try:
        # Debug the query
        context_data = await debugger.debug_query(question)
        
        # Display comprehensive analysis
        debugger.display_context_analysis(context_data)
        
        # Save raw data for further analysis
        output_file = Path("rag_context_debug.json")
        with open(output_file, 'w') as f:
            # Clean data for JSON serialization
            clean_data = context_data.copy()
            if 'formatted_context' in clean_data and clean_data['formatted_context'] and len(clean_data['formatted_context']) > 50000:
                clean_data['formatted_context'] = f"[Massive context saved to file: {clean_data.get('context_file', 'unknown')}]"
            
            json.dump(clean_data, f, indent=2, default=str)
        
        console.print(f"\n[green]💾 Raw debug data saved to: {output_file}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during debugging: {e}[/red]")
        import traceback
        traceback.print_exc()
    
    finally:
        debugger.close()


if __name__ == "__main__":
    asyncio.run(main())
