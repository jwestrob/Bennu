#!/usr/bin/env python3
"""
Enhanced logging formatter for cleaner, more readable pipeline output.

Filters out irrelevant tokens and formats logs for better user experience.
"""

import logging
import re
from typing import List, Pattern
from datetime import datetime

class PipelineLogFormatter(logging.Formatter):
    """
    Custom log formatter that cleans up output for better readability.
    
    Features:
    - Filters out irrelevant debug tokens
    - Enhances important pipeline events
    - Consistent formatting with visual hierarchy
    - Time-aware logging with relative timestamps
    """
    
    # Patterns to filter out (these create noise)
    NOISE_PATTERNS = [
        r'dspy\..*?signature',
        r'openai\..*?api',
        r'urllib3\..*?debug',
        r'requests\..*?debug',
        r'httpx\..*?debug',
        r'neo4j\..*?debug',
        r'asyncio\..*?debug',
        r'concurrent\.futures',
        r'thread_started|thread_ended',
        r'calling_tool_selector|tool_selector_returned',
        r'o3_call:|o3_result:|o3_attrs:',
        r'parsed_args:|json_parse_fail:',
        r'cache_.*?_returned',
        r'\btrace:\b.*$',  # Stack traces in debug
        r'model_allocation.*?debug',
        r'token.*?count.*?debug'
    ]
    
    # Patterns for important events (highlight these)
    IMPORTANT_PATTERNS = [
        (r'PHASE:', '\033[1;96m'),  # Cyan bold for phases
        (r'TASK CREATED:', '\033[1;93m'),  # Yellow bold for task creation
        (r'EXECUTING:', '\033[1;92m'),  # Green bold for execution
        (r'COMPLETED:', '\033[1;92m'),  # Green bold for completion
        (r'FAILED:', '\033[1;91m'),  # Red bold for failures
        (r'LLM.*?SELECTION', '\033[1;95m'),  # Magenta for LLM calls
        (r'Tool Selection:', '\033[0;94m'),  # Blue for tool selection
        (r'Global analysis detected:', '\033[0;96m'),  # Cyan for global analysis
        (r'API Call Reduction:', '\033[1;32m'),  # Bright green for efficiency
    ]
    
    def __init__(self, show_timestamps: bool = True, filter_noise: bool = True):
        """
        Initialize the formatter.
        
        Args:
            show_timestamps: Whether to show timestamps in output
            filter_noise: Whether to filter out debug noise
        """
        super().__init__()
        self.show_timestamps = show_timestamps
        self.filter_noise = filter_noise
        self.start_time = datetime.now()
        
        # Compile noise patterns for efficiency
        if self.filter_noise:
            self.compiled_noise_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.NOISE_PATTERNS]
        else:
            self.compiled_noise_patterns = []
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record with enhanced readability.
        
        Args:
            record: The log record to format
            
        Returns:
            Formatted log string, or empty string if filtered out
        """
        # Get the basic message
        message = record.getMessage()
        
        # Filter out noise if enabled
        if self.filter_noise and self._should_filter_message(message):
            return ""
        
        # Format the level indicator
        level_indicator = self._get_level_indicator(record.levelno)
        
        # Format timestamp if enabled
        timestamp_str = ""
        if self.show_timestamps:
            elapsed = datetime.now() - self.start_time
            timestamp_str = f"[{elapsed.total_seconds():6.1f}s] "
        
        # Apply highlighting to important patterns
        formatted_message = self._apply_highlighting(message)
        
        # Combine components
        formatted_record = f"{timestamp_str}{level_indicator}{formatted_message}"
        
        # Add reset color code if colors were applied
        if '\033[' in formatted_record:
            formatted_record += '\033[0m'
        
        return formatted_record
    
    def _should_filter_message(self, message: str) -> bool:
        """Check if a message should be filtered out as noise."""
        # Always show ERROR and WARNING messages
        if any(level in message.upper() for level in ['ERROR', 'WARNING', 'CRITICAL']):
            return False
        
        # Filter out noise patterns
        for pattern in self.compiled_noise_patterns:
            if pattern.search(message):
                return True
        
        # Filter out very verbose debug messages
        if len(message) > 200 and 'debug' in message.lower():
            return True
        
        return False
    
    def _get_level_indicator(self, levelno: int) -> str:
        """Get a visual indicator for the log level."""
        if levelno >= logging.ERROR:
            return "🔴 "
        elif levelno >= logging.WARNING:
            return "🟡 "
        elif levelno >= logging.INFO:
            return ""  # No indicator for info
        else:
            return "🔹 "  # Small indicator for debug
    
    def _apply_highlighting(self, message: str) -> str:
        """Apply color highlighting to important patterns."""
        formatted_message = message
        
        for pattern, color_code in self.IMPORTANT_PATTERNS:
            formatted_message = re.sub(
                pattern, 
                f"{color_code}\\g<0>", 
                formatted_message, 
                flags=re.IGNORECASE
            )
        
        return formatted_message


class TaskGraphLogFilter(logging.Filter):
    """
    Advanced filter for task graph logs to show only relevant information.
    """
    
    def __init__(self, show_levels: List[str] = None):
        """
        Initialize the filter.
        
        Args:
            show_levels: List of log levels to show (DEBUG, INFO, WARNING, ERROR)
        """
        super().__init__()
        self.show_levels = show_levels or ['INFO', 'WARNING', 'ERROR']
        self.level_numbers = [getattr(logging, level) for level in self.show_levels]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Determine whether to include a log record.
        
        Args:
            record: The log record to evaluate
            
        Returns:
            True if the record should be included, False otherwise
        """
        # Always include records at or above the specified levels
        if record.levelno in self.level_numbers:
            return True
        
        # Include task-related messages even if they're at DEBUG level
        if any(keyword in record.getMessage().lower() for keyword in [
            'task created', 'executing', 'completed', 'failed',
            'phase:', 'tool selection', 'global analysis'
        ]):
            return True
        
        return False


def setup_enhanced_logging(log_level: str = "INFO", 
                         filter_noise: bool = True,
                         show_timestamps: bool = True,
                         export_to_file: bool = False):
    """
    Set up enhanced logging for the genomic RAG pipeline.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        filter_noise: Whether to filter out debug noise
        show_timestamps: Whether to show timestamps
        export_to_file: Whether to also export logs to file
    """
    # Get the root logger for the RAG system
    rag_logger = logging.getLogger('src.llm.rag_system')
    rag_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    rag_logger.handlers.clear()
    
    # Create console handler with enhanced formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Apply custom formatter and filter
    formatter = PipelineLogFormatter(
        show_timestamps=show_timestamps,
        filter_noise=filter_noise
    )
    console_handler.setFormatter(formatter)
    
    # Apply task graph filter
    task_filter = TaskGraphLogFilter()
    console_handler.addFilter(task_filter)
    
    rag_logger.addHandler(console_handler)
    
    # Optional file handler for complete logs
    if export_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(f"pipeline_log_{timestamp}.log")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        rag_logger.addHandler(file_handler)
        
        logging.info(f"📄 Complete logs being written to: pipeline_log_{timestamp}.log")
    
    # Suppress noisy third-party loggers
    for noisy_logger in ['urllib3', 'requests', 'httpx', 'openai', 'neo4j']:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    
    logging.info("✨ Enhanced logging configured for genomic RAG pipeline")
    return rag_logger


# Convenience function for quick setup
def enable_clean_logging():
    """Enable clean, readable logging for pipeline execution."""
    return setup_enhanced_logging(
        log_level="INFO",
        filter_noise=True,
        show_timestamps=True,
        export_to_file=False
    )


# Function to export task execution summary
def export_task_summary(task_graph, filename: str = None):
    """
    Export a clean summary of task execution to a file.
    
    Args:
        task_graph: TaskGraph instance with execution data
        filename: Optional filename for export
    """
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"task_execution_summary_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GENOMIC RAG PIPELINE - TASK EXECUTION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Write user query
        f.write(f"User Query: {getattr(task_graph, 'user_query', 'Unknown')}\n")
        f.write(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write task summary
        main_tasks = [t for t in task_graph.tasks.values() if t.is_main_task]
        sub_tasks = [t for t in task_graph.tasks.values() if not t.is_main_task]
        
        f.write(f"Total Tasks: {len(task_graph.tasks)}\n")
        f.write(f"  - Main Tasks: {len(main_tasks)}\n")
        f.write(f"  - Sub Tasks: {len(sub_tasks)}\n\n")
        
        # Write main task details
        f.write("MAIN TASKS EXECUTED:\n")
        f.write("-" * 40 + "\n")
        for task in main_tasks:
            status = "✓" if task.status == TaskStatus.COMPLETED else "✗" if task.status == TaskStatus.FAILED else "⧖"
            f.write(f"{status} {task.task_id}\n")
            f.write(f"   Description: {task.description}\n")
            f.write(f"   Type: {task.task_type.value}\n")
            if task.tool_name:
                f.write(f"   Tool: {task.tool_name}\n")
            f.write(f"   Selection: {task.tool_selection_source}\n")
            if task.execution_time_ms:
                f.write(f"   Time: {task.execution_time_ms:.1f}ms\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    logging.info(f"📋 Task summary exported to: {filename}")
    return filename