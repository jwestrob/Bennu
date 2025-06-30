#!/usr/bin/env python3
"""
Command runner utility that automatically redirects large outputs to files
to prevent STDOUT pollution in interactive sessions.
"""

import subprocess
import tempfile
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CommandRunner:
    """
    Utility class for running commands with automatic file redirection for large outputs.
    """
    
    def __init__(self, output_dir: str = "data/command_outputs", max_stdout_lines: int = 50):
        """
        Initialize CommandRunner.
        
        Args:
            output_dir: Directory to store large command outputs
            max_stdout_lines: Maximum lines to show in STDOUT before redirecting to file
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_stdout_lines = max_stdout_lines
    
    def run_command(self, command: str, description: str = None, 
                   force_file_output: bool = False) -> Dict[str, Any]:
        """
        Run a command with automatic file redirection for large outputs.
        
        Args:
            command: Command to execute
            description: Human-readable description of the command
            force_file_output: Always redirect to file regardless of size
            
        Returns:
            Dict containing execution results and file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_desc = (description or "command").replace(" ", "_").replace("/", "_")[:50]
        output_file = self.output_dir / f"{timestamp}_{safe_desc}.txt"
        
        try:
            # Run command and capture output
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            stdout_lines = result.stdout.split('\n') if result.stdout else []
            stderr_lines = result.stderr.split('\n') if result.stderr else []
            
            # Determine if output should be redirected to file
            total_lines = len(stdout_lines) + len(stderr_lines)
            should_redirect = force_file_output or total_lines > self.max_stdout_lines
            
            if should_redirect:
                # Write full output to file
                with open(output_file, 'w') as f:
                    f.write(f"Command: {command}\n")
                    f.write(f"Description: {description or 'N/A'}\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Return code: {result.returncode}\n")
                    f.write("=" * 80 + "\n")
                    f.write("STDOUT:\n")
                    f.write(result.stdout or "(no stdout)")
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("STDERR:\n")
                    f.write(result.stderr or "(no stderr)")
                
                # Show summary in STDOUT
                summary_lines = min(10, len(stdout_lines))
                stdout_summary = '\n'.join(stdout_lines[:summary_lines])
                if len(stdout_lines) > summary_lines:
                    stdout_summary += f"\n... ({len(stdout_lines) - summary_lines} more lines)"
                
                return {
                    "returncode": result.returncode,
                    "stdout_summary": stdout_summary,
                    "stderr": result.stderr,
                    "output_file": str(output_file),
                    "total_lines": total_lines,
                    "redirected": True,
                    "command": command
                }
            else:
                # Output is small enough to show directly
                return {
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "output_file": None,
                    "total_lines": total_lines,
                    "redirected": False,
                    "command": command
                }
                
        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out after 5 minutes: {command}"
            logger.error(error_msg)
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": error_msg,
                "output_file": None,
                "total_lines": 0,
                "redirected": False,
                "command": command,
                "error": "timeout"
            }
        except Exception as e:
            error_msg = f"Error executing command: {e}"
            logger.error(error_msg)
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": error_msg,
                "output_file": None,
                "total_lines": 0,
                "redirected": False,
                "command": command,
                "error": str(e)
            }

def run_with_file_redirect(command: str, description: str = None, 
                          max_lines: int = 50) -> Dict[str, Any]:
    """
    Convenience function to run a command with file redirection.
    
    Args:
        command: Command to execute
        description: Human-readable description
        max_lines: Maximum lines to show before redirecting
        
    Returns:
        Dict containing execution results
    """
    runner = CommandRunner(max_stdout_lines=max_lines)
    return runner.run_command(command, description)

def print_command_result(result: Dict[str, Any]):
    """
    Print command result in a user-friendly format.
    
    Args:
        result: Result dictionary from run_command
    """
    if result["redirected"]:
        print(f"✅ Command completed (output redirected to file)")
        print(f"📁 Full output saved to: {result['output_file']}")
        print(f"📊 Total output lines: {result['total_lines']}")
        print(f"🔍 First few lines:")
        print(result["stdout_summary"])
        if result["stderr"]:
            print(f"⚠️ Stderr: {result['stderr']}")
    else:
        print(f"✅ Command completed")
        if result["stdout"]:
            print(result["stdout"])
        if result["stderr"]:
            print(f"⚠️ Stderr: {result['stderr']}")
    
    if result["returncode"] != 0:
        print(f"❌ Command failed with return code: {result['returncode']}")

# Example usage for common genomic commands
def run_annotation_explorer(query: str = "Show me all proteins involved in central metabolism") -> str:
    """
    Run the annotation explorer with automatic file redirection.
    
    Returns:
        Path to output file if redirected, or direct output if small
    """
    command = f'cd /Users/jacob/Documents/Sandbox/microbial_claude_matter && source /Users/jacob/.pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh && conda activate genome-kg && python -m src.cli ask "{query}"'
    
    result = run_with_file_redirect(
        command, 
        description=f"annotation_explorer_{query[:30].replace(' ', '_')}"
    )
    
    print_command_result(result)
    
    if result["redirected"]:
        return result["output_file"]
    else:
        return result["stdout"]

if __name__ == "__main__":
    # Test the command runner
    result = run_annotation_explorer()
    print(f"Result: {result}")

def extract_answer_from_file(file_path: str) -> Dict[str, Any]:
    """
    Extract the actual answer content from a redirected command output file.
    
    Args:
        file_path: Path to the command output file
        
    Returns:
        Dict containing extracted answer data
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find the answer section (after STDOUT: and before STDERR:)
        stdout_start = content.find('STDOUT:')
        stderr_start = content.find('STDERR:')
        
        if stdout_start == -1:
            return {"error": "Could not find STDOUT section in file"}
        
        if stderr_start == -1:
            answer_section = content[stdout_start + 8:].strip()
        else:
            answer_section = content[stdout_start + 8:stderr_start].strip()
        
        # Extract metadata from the file header
        lines = content.split('\n')
        metadata = {}
        for line in lines[:10]:  # Check first 10 lines for metadata
            if line.startswith('Command:'):
                metadata['original_command'] = line[8:].strip()
            elif line.startswith('Description:'):
                metadata['description'] = line[12:].strip()
            elif line.startswith('Timestamp:'):
                metadata['timestamp'] = line[10:].strip()
            elif line.startswith('Return code:'):
                metadata['return_code'] = int(line[12:].strip())
        
        # Try to extract structured answer components
        answer_lines = answer_section.split('\n')
        
        # Look for confidence and sources at the end
        confidence = None
        sources = None
        answer_content = []
        
        for i, line in enumerate(answer_lines):
            if line.startswith('Confidence:'):
                confidence = line[11:].strip()
                # Everything after this might be sources
                remaining_lines = answer_lines[i+1:]
                for remaining_line in remaining_lines:
                    if remaining_line.startswith('Sources:'):
                        sources = remaining_line[8:].strip()
                        break
                # Everything before confidence is the main answer
                answer_content = answer_lines[:i]
                break
        
        if not answer_content:
            # No confidence found, use all content
            answer_content = answer_lines
        
        return {
            "success": True,
            "file_path": file_path,
            "metadata": metadata,
            "answer_content": '\n'.join(answer_content).strip(),
            "confidence": confidence,
            "sources": sources,
            "full_answer": answer_section,
            "line_count": len(answer_lines)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }

def load_and_summarize_results(file_path: str, max_lines: int = 100) -> str:
    """
    Load results from a file and create a concise summary for context.
    
    Args:
        file_path: Path to the command output file
        max_lines: Maximum lines to include in summary
        
    Returns:
        Formatted summary string suitable for context
    """
    result = extract_answer_from_file(file_path)
    
    if not result["success"]:
        return f"Error loading results from {file_path}: {result['error']}"
    
    summary_parts = []
    
    # Add metadata
    if result["metadata"]:
        summary_parts.append(f"Query: {result['metadata'].get('description', 'Unknown')}")
        summary_parts.append(f"Timestamp: {result['metadata'].get('timestamp', 'Unknown')}")
    
    # Add answer content (truncated if needed)
    answer_lines = result["answer_content"].split('\n')
    if len(answer_lines) > max_lines:
        truncated_answer = '\n'.join(answer_lines[:max_lines])
        summary_parts.append(f"Answer (first {max_lines} lines):")
        summary_parts.append(truncated_answer)
        summary_parts.append(f"... ({len(answer_lines) - max_lines} more lines in full file)")
    else:
        summary_parts.append("Answer:")
        summary_parts.append(result["answer_content"])
    
    # Add confidence and sources if available
    if result["confidence"]:
        summary_parts.append(f"Confidence: {result['confidence']}")
    if result["sources"]:
        summary_parts.append(f"Sources: {result['sources']}")
    
    summary_parts.append(f"Full results available in: {file_path}")
    
    return '\n'.join(summary_parts)

# Example usage for testing
if __name__ == "__main__":
    # Test the extraction functions
    test_file = "data/command_outputs/20250629_194123_annotation_explorer_Show_me_all_proteins_involved_.txt"
    
    print("=== Testing Answer Extraction ===")
    result = extract_answer_from_file(test_file)
    
    if result["success"]:
        print(f"✅ Successfully extracted answer")
        print(f"📊 Answer length: {len(result['answer_content'])} characters")
        print(f"📊 Line count: {result['line_count']}")
        print(f"🎯 Confidence: {result['confidence']}")
        print(f"📚 Sources: {result['sources'][:100]}..." if result['sources'] else "📚 Sources: None")
        
        print("\n=== Testing Summary Generation ===")
        summary = load_and_summarize_results(test_file, max_lines=20)
        print("Generated summary:")
        print("-" * 50)
        print(summary)
        print("-" * 50)
    else:
        print(f"❌ Failed to extract answer: {result['error']}")
