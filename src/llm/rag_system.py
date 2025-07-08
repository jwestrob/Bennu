#!/usr/bin/env python3
"""
Genomic RAG System - Modular Architecture
This file provides backward compatibility by importing from the new modular structure.

The actual implementation has been split into:
- core.py: Main GenomicRAG class
- context_processing.py: Context retrieval and formatting
- data_scaling.py: Tiered scaling strategies
- code_enhancement.py: Code interpreter enhancement
- utils.py: Shared utilities and constants
"""

# Import from the new modular structure for backward compatibility
from .rag_system.core import GenomicRAG
from .rag_system.utils import EXAMPLE_GENOMIC_QUESTIONS
from .rag_system.context_processing import ContextProcessor, ContextFormatter
from .rag_system.data_scaling import ScalingRouter, DataScalingStrategy
from .rag_system.code_enhancement import CodeEnhancer
from .rag_system.utils import ResultStreamer, safe_log_data, setup_debug_logging

# Export main classes and functions for backward compatibility
__all__ = [
    'GenomicRAG',
    'EXAMPLE_GENOMIC_QUESTIONS', 
    'ContextProcessor',
    'ContextFormatter',
    'ScalingRouter',
    'DataScalingStrategy',
    'CodeEnhancer',
    'ResultStreamer',
    'safe_log_data',
    'setup_debug_logging'
]