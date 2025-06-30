#!/usr/bin/env python3
"""
Wrapper script to run annotation explorer with automatic file redirection for large outputs.
This prevents STDOUT pollution in interactive sessions.
"""

import sys
import argparse
from src.utils.command_runner import run_with_file_redirect, print_command_result

def main():
    parser = argparse.ArgumentParser(description="Run annotation explorer with file redirection")
    parser.add_argument("query", help="Query to ask the annotation explorer")
    parser.add_argument("--max-lines", type=int, default=50, 
                       help="Maximum lines to show before redirecting to file")
    parser.add_argument("--force-file", action="store_true",
                       help="Always redirect to file regardless of output size")
    
    args = parser.parse_args()
    
    # Construct the command
    command = (
        f'cd /Users/jacob/Documents/Sandbox/microbial_claude_matter && '
        f'source /Users/jacob/.pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh && '
        f'conda activate genome-kg && '
        f'python -m src.cli ask "{args.query}"'
    )
    
    # Run with file redirection
    result = run_with_file_redirect(
        command, 
        description=f"annotation_explorer_{args.query[:30].replace(' ', '_')}",
        max_lines=args.max_lines
    )
    
    # Force file output if requested
    if args.force_file and not result["redirected"]:
        # Re-run with force_file_output=True
        from src.utils.command_runner import CommandRunner
        runner = CommandRunner(max_stdout_lines=args.max_lines)
        result = runner.run_command(command, 
                                   description=f"annotation_explorer_{args.query[:30].replace(' ', '_')}", 
                                   force_file_output=True)
    
    print_command_result(result)
    
    # Return the file path if redirected for further processing
    if result["redirected"]:
        print(f"\n📋 To examine the full results:")
        print(f"   cat '{result['output_file']}'")
        print(f"   grep 'pattern' '{result['output_file']}'")
        print(f"   tail -50 '{result['output_file']}'")
        return result["output_file"]
    else:
        return None

if __name__ == "__main__":
    main()
