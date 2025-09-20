#!/usr/bin/env python3
"""
Genomic RAG System - Modular Architecture
This file provides backward compatibility by importing from the new modular structure.

<<<<<<< HEAD
=======
DEPRECATION NOTICE:
- This module is a compatibility shim. Prefer importing directly from the modular
  implementation, e.g. `from llm.rag_system.core import GenomicRAG`.

>>>>>>> feat/agent-router-typed
The actual implementation has been split into:
- core.py: Main GenomicRAG class
- context_processing.py: Context retrieval and formatting
- data_scaling.py: Tiered scaling strategies
- code_enhancement.py: Code interpreter enhancement
- utils.py: Shared utilities and constants
"""

<<<<<<< HEAD
# Import from the new modular structure for backward compatibility
from .rag_system.core import GenomicRAG
from .rag_system.utils import EXAMPLE_GENOMIC_QUESTIONS
from .rag_system.context_processing import ContextProcessor, ContextFormatter
from .rag_system.data_scaling import ScalingRouter, DataScalingStrategy
from .rag_system.code_enhancement import CodeEnhancer
from .rag_system.utils import ResultStreamer, safe_log_data, setup_debug_logging

=======
import warnings as _warnings

# Emit a deprecation warning when this shim is imported
_warnings.warn(
    "llm.rag_system is a compatibility shim; import from llm.rag_system.core, "
    "llm.rag_system.context_processing, etc.",
    DeprecationWarning,
    stacklevel=2,
)

# Import from the new modular structure for backward compatibility
from .rag_system.core import GenomicRAG
from .rag_system.utils import EXAMPLE_GENOMIC_QUESTIONS
from .rag_system.context_processing import ContextProcessor, ContextFormatter
from .rag_system.data_scaling import ScalingRouter, DataScalingStrategy
from .rag_system.code_enhancement import CodeEnhancer
from .rag_system.utils import ResultStreamer, safe_log_data, setup_debug_logging

>>>>>>> feat/agent-router-typed
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
<<<<<<< HEAD
]
=======
]
>>>>>>> feat/agent-router-typed
