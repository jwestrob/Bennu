#!/usr/bin/env python3
"""
Shared utilities for genomic RAG system.
Common functions, constants, and helper classes.
"""

import logging
from typing import Any, Iterator, List, Dict
from pathlib import Path
from dataclasses import dataclass

# Token counting for chunking
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class GenomicContext:
    """Context extracted from database queries."""
    structured_data: List[Dict[str, Any]]
    semantic_data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    query_time: float
    compressed_context: str = ""  # New field for compressed context string

# Example questions for CLI interface
EXAMPLE_GENOMIC_QUESTIONS = [
    "What metabolic pathways are present in Escherichia coli?",
    "Find proteins similar to heme transporters",
    "What is the function of KEGG ortholog K20469?",
    "Show me all CAZymes in the database", 
    "Find BGCs with high terpene probability",
    "What genes are in the same operon as succinate dehydrogenase?",
    "Compare metabolic capabilities between organisms"
]

def safe_log_data(data: Any, max_length: int = 200, description: str = "data") -> str:
    """Safely log data structures with length limits to prevent console spam."""
    try:
        if isinstance(data, (dict, list)):
            data_str = str(data)
            if len(data_str) > max_length:
                if isinstance(data, dict):
                    return f"<large_dict: {len(data)} keys, {len(data_str)} chars>"
                elif isinstance(data, list):
                    return f"<large_list: {len(data)} items, {len(data_str)} chars>"
            return data_str
        else:
            data_str = str(data)
            if len(data_str) > max_length:
                return f"<large_{type(data).__name__}: {len(data_str)} chars>"
            return data_str
    except Exception as e:
        return f"<error_logging_{description}: {str(e)}>"

def setup_debug_logging():
    """Setup debug logging to redirect verbose output to log files."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create debug file handler
    debug_handler = logging.FileHandler(log_dir / "rag_debug.log")
    debug_handler.setLevel(logging.DEBUG)
    debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    debug_handler.setFormatter(debug_formatter)
    
    # Add handler to logger
    debug_logger = logging.getLogger(__name__)
    debug_logger.addHandler(debug_handler)
    debug_logger.setLevel(logging.DEBUG)

class ResultStreamer:
    """
    Chunks large iterators into manageable token-sized chunks to prevent context window overflow.
    Writes each chunk to JSONL files and returns summaries.
    """
    
    def __init__(self, chunk_context_size: int = 4096, output_dir: str = "data/rag_outputs"):
        self.chunk_context_size = chunk_context_size
        self.output_dir = Path(output_dir)
        self.session_id = None
        self.session_dir = None
        
        # Initialize token encoder
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoder = tiktoken.encoding_for_model("gpt-4")
            except Exception:
                # Fallback to cl100k_base encoding
                self.encoder = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoder = None
            logger.warning("tiktoken not available, using character-based approximation for token counting")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken or character approximation."""
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4
    
    def create_session(self) -> str:
        """Create a new session directory for streaming output."""
        import uuid
        import datetime
        
        self.session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"session_{timestamp}_{self.session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Created streaming session: {self.session_dir}")
        return str(self.session_dir)
    
    def stream_iterator(
        self, 
        iterator: Iterator[dict], 
        context_prefix: str = "",
        chunk_summary_template: str = "Processed {count} items"
    ) -> Iterator[str]:
        """
        Stream an iterator in token-sized chunks, yielding summaries.
        
        Args:
            iterator: Iterator of dict items to chunk
            context_prefix: Prefix text for each chunk
            chunk_summary_template: Template for chunk summaries
            
        Yields:
            Summary strings for each chunk
        """
        if not self.session_dir:
            self.create_session()
        
        chunk_num = 0
        current_chunk = []
        current_tokens = self.count_tokens(context_prefix)
        
        for item in iterator:
            item_text = str(item)
            item_tokens = self.count_tokens(item_text)
            
            # Check if adding this item would exceed chunk size
            if current_tokens + item_tokens > self.chunk_context_size and current_chunk:
                # Write current chunk and yield summary
                chunk_num += 1
                chunk_file = self.session_dir / f"chunk_{chunk_num:03d}.jsonl"
                
                with open(chunk_file, 'w') as f:
                    for chunk_item in current_chunk:
                        f.write(f"{chunk_item}\n")
                
                summary = chunk_summary_template.format(
                    count=len(current_chunk),
                    chunk_num=chunk_num,
                    tokens=current_tokens,
                    file=chunk_file.name
                )
                
                logger.debug(f"📦 Wrote chunk {chunk_num}: {len(current_chunk)} items, {current_tokens} tokens")
                yield summary
                
                # Reset for next chunk
                current_chunk = []
                current_tokens = self.count_tokens(context_prefix)
            
            # Add item to current chunk
            current_chunk.append(item_text)
            current_tokens += item_tokens
        
        # Handle final chunk if any items remain
        if current_chunk:
            chunk_num += 1
            chunk_file = self.session_dir / f"chunk_{chunk_num:03d}.jsonl"
            
            with open(chunk_file, 'w') as f:
                for chunk_item in current_chunk:
                    f.write(f"{chunk_item}\n")
            
            summary = chunk_summary_template.format(
                count=len(current_chunk),
                chunk_num=chunk_num,
                tokens=current_tokens,
                file=chunk_file.name
            )
            
            logger.debug(f"📦 Wrote final chunk {chunk_num}: {len(current_chunk)} items, {current_tokens} tokens")
            yield summary
    
    def get_session_summary(self) -> dict:
        """Get summary of current streaming session."""
        if not self.session_dir or not self.session_dir.exists():
            return {"error": "No active session"}
        
        chunk_files = list(self.session_dir.glob("chunk_*.jsonl"))
        total_lines = 0
        total_size = 0
        
        for chunk_file in chunk_files:
            with open(chunk_file, 'r') as f:
                total_lines += sum(1 for _ in f)
            total_size += chunk_file.stat().st_size
        
        return {
            "session_id": self.session_id,
            "session_dir": str(self.session_dir),
            "chunk_count": len(chunk_files),
            "total_items": total_lines,
            "total_size_bytes": total_size,
            "chunk_files": [f.name for f in sorted(chunk_files)]
        }