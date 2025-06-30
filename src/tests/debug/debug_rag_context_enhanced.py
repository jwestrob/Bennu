#!/usr/bin/env python3
"""
Enhanced debug script with automatic file redirection for large outputs.
Captures and analyzes RAG context data without polluting STDOUT.
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


class EnhancedRAGContextDebugger:
    """Enhanced debugger with automatic file redirection for large outputs."""
    
    def __init__(self, config: LLMConfig = None, chunk_context_size: int = 4096):
        self.config = config or LLMConfig.from_env()
        self.chunk_context_size = chunk_context_size
        self.rag = GenomicRAG(self.config, chunk_context_size=chunk_context_size)
        self.command_runner = CommandRunner(output_dir="data/debug_outputs", max_stdout_lines=30)
        
        # Create debug output directory
        self.debug_dir = Path("data/debug_outputs")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
    async def debug_query_with_file_output(self, question: str) -> Dict[str, Any]:
        """Run a query and capture all context data with file redirection."""
        console.print(f"[bold blue]🔍 Enhanced RAG Debug for:[/bold blue] {question}")
        
        # Generate timestamp for this debug session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_prefix = f"{timestamp}_rag_debug"
        
        # Monkey patch the RAG system to capture context
        original_format_context = self.rag._format_context
        original_execute_task = self.rag._execute_task
        captured_context = {
            'question': question,
            'timestamp': timestamp,
            'agentic_tasks': [],
            'task_results': {},
            'is_agentic': False,
            'formatted_context': None,
            'context_length': 0
        }
        
        def capture_format_context(context):
            # Call original method and capture result
            formatted = original_format_context(context)
            captured_context['formatted_context'] = formatted
            captured_context['context_length'] = len(formatted)
            captured_context['neo4j_record_count'] = len(context.structured_data)
            captured_context['lancedb_record_count'] = len(context.semantic_data)
            captured_context['neo4j_raw_data'] = context.structured_data
            captured_context['lancedb_raw_data'] = context.semantic_data
            
            # Write large context to file if needed
            if len(formatted) > 5000:  # Large context threshold
                context_file = self.debug_dir / f"{session_prefix}_formatted_context.txt"
                with open(context_file, 'w') as f:
                    f.write(f"Question: {question}\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Context Length: {len(formatted)} characters\n")
                    f.write("=" * 80 + "\n")
                    f.write(formatted)
                
                console.print(f"📁 Large context ({len(formatted):,} chars) saved to: {context_file}")
                captured_context['context_file'] = str(context_file)
            
            return formatted
        
        async def capture_execute_task(task, previous_results):
            # Capture task execution
            captured_context['agentic_tasks'].append({
                'id': task.id,
                'type': task.task_type.value,
                'query': getattr(task, 'query', None),
                'tool_name': getattr(task, 'tool_name', None),
                'dependencies': task.dependencies
            })
            captured_context['is_agentic'] = True
            
            # Execute task and capture result
            result = await original_execute_task(task, previous_results)
            captured_context['task_results'][task.id] = {
                'type': type(result).__name__,
                'size': len(str(result)) if result else 0,
                'summary': str(result)[:200] + "..." if result and len(str(result)) > 200 else str(result)
            }
            
            # Write large task results to files
            if result and len(str(result)) > 2000:
                task_file = self.debug_dir / f"{session_prefix}_task_{task.id}.json"
                with open(task_file, 'w') as f:
                    json.dump({
                        'task_id': task.id,
                        'task_type': task.task_type.value,
                        'result': result
                    }, f, indent=2, default=str)
                
                console.print(f"📁 Large task result ({len(str(result)):,} chars) saved to: {task_file}")
                captured_context['task_results'][task.id]['result_file'] = str(task_file)
            
            return result
        
        # Apply monkey patches
        self.rag._format_context = capture_format_context
        self.rag._execute_task = capture_execute_task
        
        try:
            # Execute the query
            result = await self.rag.ask(question)
            captured_context['final_answer'] = result
            
            return captured_context
            
        finally:
            # Restore original methods
            self.rag._format_context = original_format_context
            self.rag._execute_task = original_execute_task
    
    def display_enhanced_analysis(self, context_data: Dict[str, Any]):
        """Display enhanced analysis with file references."""
        
        console.print("\n" + "="*80)
        console.print("[bold green]📊 ENHANCED RAG CONTEXT ANALYSIS[/bold green]")
        console.print("="*80)
        
        # Overview table
        overview_table = Table(title="Debug Session Overview")
        overview_table.add_column("Metric", style="cyan")
        overview_table.add_column("Value", style="magenta")
        overview_table.add_column("File Output", style="yellow")
        
        overview_table.add_row("Question", context_data['question'][:50] + "...", "")
        overview_table.add_row("Timestamp", context_data['timestamp'], "")
        overview_table.add_row("Query Type", "Agentic" if context_data['is_agentic'] else "Traditional", "")
        overview_table.add_row("Context Length", f"{context_data.get('context_length', 0):,} chars", 
                              context_data.get('context_file', 'In memory'))
        overview_table.add_row("Neo4j Records", str(context_data.get('neo4j_record_count', 0)), "")
        overview_table.add_row("LanceDB Records", str(context_data.get('lancedb_record_count', 0)), "")
        
        if context_data['is_agentic']:
            task_count = len(context_data.get('agentic_tasks', []))
            overview_table.add_row("Agentic Tasks", str(task_count), f"{task_count} task files")
        
        console.print(overview_table)
        
        # File outputs summary
        if context_data.get('context_file') or any('result_file' in task for task in context_data.get('task_results', {}).values()):
            console.print("\n[bold yellow]📁 Generated Files:[/bold yellow]")
            
            if context_data.get('context_file'):
                console.print(f"  • Context: {context_data['context_file']}")
            
            for task_id, task_result in context_data.get('task_results', {}).items():
                if 'result_file' in task_result:
                    console.print(f"  • Task {task_id}: {task_result['result_file']}")
        
        # Quick analysis
        console.print("\n[bold cyan]🔍 Quick Analysis:[/bold cyan]")
        
        if context_data['is_agentic']:
            console.print(f"  • Multi-step agentic workflow with {len(context_data.get('agentic_tasks', []))} tasks")
        else:
            console.print("  • Single-step traditional query")
        
        if context_data.get('neo4j_record_count', 0) > 0:
            console.print(f"  • Retrieved {context_data['neo4j_record_count']} structured records from Neo4j")
        
        if context_data.get('lancedb_record_count', 0) > 0:
            console.print(f"  • Retrieved {context_data['lancedb_record_count']} semantic records from LanceDB")
        
        context_len = context_data.get('context_length', 0)
        if context_len > 10000:
            console.print(f"  • [yellow]Large context ({context_len:,} chars) - check file output[/yellow]")
        elif context_len > 0:
            console.print(f"  • Context size: {context_len:,} characters")
    
    def close(self):
        """Clean up resources."""
        # Close any open connections
        pass


async def main():
    """Enhanced main function with file redirection."""
    parser = argparse.ArgumentParser(description="Enhanced RAG context debugger with file output")
    parser.add_argument("question", help="The question to debug")
    parser.add_argument("-c", "--chunk-context-size", type=int, default=4096,
                       help="Chunk context size for streaming (default: 4096)")
    parser.add_argument("--max-lines", type=int, default=30,
                       help="Maximum lines to show before redirecting to file")
    
    try:
        args = parser.parse_args()
    except SystemExit:
        console.print("[red]Usage: python debug_rag_context_enhanced.py \"Your question here\" [-c CHUNK_SIZE][/red]")
        console.print("\n[yellow]Example questions:[/yellow]")
        console.print("  • \"What ribosomal proteins are present?\"")
        console.print("  • \"Find proteins similar to transporters\"")
        console.print("  • \"Show me all proteins involved in central metabolism\"")
        return
    
    question = args.question
    chunk_context_size = args.chunk_context_size
    
    console.print(f"[dim]Using chunk context size: {chunk_context_size}[/dim]")
    debugger = EnhancedRAGContextDebugger(chunk_context_size=chunk_context_size)
    
    try:
        # Debug the query with file output
        context_data = await debugger.debug_query_with_file_output(question)
        
        # Display enhanced analysis
        debugger.display_enhanced_analysis(context_data)
        
        # Save comprehensive debug data
        timestamp = context_data['timestamp']
        debug_file = Path(f"data/debug_outputs/{timestamp}_complete_debug.json")
        with open(debug_file, 'w') as f:
            # Remove large data before saving to avoid huge JSON files
            clean_data = context_data.copy()
            if 'formatted_context' in clean_data and len(clean_data['formatted_context']) > 5000:
                clean_data['formatted_context'] = f"[Large context saved to file: {clean_data.get('context_file', 'unknown')}]"
            
            json.dump(clean_data, f, indent=2, default=str)
        
        console.print(f"\n[green]💾 Complete debug data saved to: {debug_file}[/green]")
        console.print(f"[green]📁 All large outputs saved to: data/debug_outputs/[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during debugging: {e}[/red]")
        import traceback
        traceback.print_exc()
    
    finally:
        debugger.close()


if __name__ == "__main__":
    asyncio.run(main())
