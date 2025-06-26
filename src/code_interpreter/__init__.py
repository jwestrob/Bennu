"""
Code Interpreter Module

Secure Python code execution service for genomic data analysis.
Provides containerized, sandboxed execution environment with stateful sessions.
"""

from .client import CodeInterpreterClient, GenomicCodeInterpreter, code_interpreter_tool

# Conditionally import service to avoid FastAPI dependency in main environment
try:
    from .service import app as code_interpreter_service
    SERVICE_AVAILABLE = True
except ImportError:
    code_interpreter_service = None
    SERVICE_AVAILABLE = False

__all__ = [
    'CodeInterpreterClient',
    'GenomicCodeInterpreter', 
    'code_interpreter_tool'
]

if SERVICE_AVAILABLE:
    __all__.append('code_interpreter_service')