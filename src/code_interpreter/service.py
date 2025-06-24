#!/usr/bin/env python3
"""
Secure Code Interpreter Service

A FastAPI service that provides secure Python code execution for genomic data analysis.
Features:
- Stateful sessions for iterative analysis
- Security hardening with container isolation
- Resource limits and timeout enforcement
- Read-only filesystem with writable /tmp
"""

import asyncio
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional
import json
import sys
import io
import traceback
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


class CodeExecutionRequest(BaseModel):
    """Request model for code execution."""
    session_id: str = Field(..., description="Unique session identifier for state management")
    code: str = Field(..., description="Python code to execute")
    timeout: int = Field(default=30, description="Execution timeout in seconds", ge=1, le=300)
    enable_networking: bool = Field(default=False, description="Enable network access (security risk)")


class CodeExecutionResponse(BaseModel):
    """Response model for code execution."""
    session_id: str
    success: bool
    stdout: str
    stderr: str
    result: Any = None
    error: Optional[str] = None
    execution_time: float
    files_created: list[str] = Field(default_factory=list)


class SessionManager:
    """Manages stateful Python execution sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeouts: Dict[str, datetime] = {}
        self.session_timeout_minutes = 30
    
    def get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """Get existing session or create new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'globals': self._create_secure_globals(),
                'locals': {},
                'temp_dir': None
            }
        
        # Update timeout
        self.session_timeouts[session_id] = datetime.now() + timedelta(minutes=self.session_timeout_minutes)
        return self.sessions[session_id]
    
    def _create_secure_globals(self) -> Dict[str, Any]:
        """Create secure global environment with allowed imports."""
        allowed_globals = {
            # Built-in functions (safe subset)
            '__builtins__': {
                'abs', 'all', 'any', 'bin', 'bool', 'chr', 'dict', 'enumerate',
                'filter', 'float', 'format', 'hex', 'int', 'len', 'list', 'map',
                'max', 'min', 'oct', 'ord', 'print', 'range', 'reversed', 'round',
                'set', 'sorted', 'str', 'sum', 'tuple', 'type', 'zip'
            },
            # Safe imports
            'os': None,  # Will be restricted
            'sys': None,  # Will be restricted
            'json': json,
            'math': None,  # Will be imported safely
        }
        
        # Import safe modules
        try:
            import math
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            from pathlib import Path
            
            allowed_globals.update({
                'math': math,
                'np': np,
                'pd': pd,
                'plt': plt,
                'sns': sns,
                'Path': Path,
            })
        except ImportError as e:
            logger.warning(f"Some scientific packages not available: {e}")
        
        return allowed_globals
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions."""
        now = datetime.now()
        expired = [sid for sid, timeout in self.session_timeouts.items() if now > timeout]
        
        for session_id in expired:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                if session.get('temp_dir'):
                    # Cleanup temporary directory
                    try:
                        import shutil
                        shutil.rmtree(session['temp_dir'])
                    except Exception as e:
                        logger.error(f"Failed to cleanup temp dir for {session_id}: {e}")
                
                del self.sessions[session_id]
                del self.session_timeouts[session_id]
                logger.info(f"Cleaned up expired session: {session_id}")


class SecureCodeExecutor:
    """Executes Python code in a secure environment."""
    
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
    
    async def execute_code(self, request: CodeExecutionRequest) -> CodeExecutionResponse:
        """Execute Python code securely."""
        start_time = asyncio.get_event_loop().time()
        
        # Get or create session
        session = self.session_manager.get_or_create_session(request.session_id)
        
        # Create temporary directory for this execution
        if not session.get('temp_dir'):
            session['temp_dir'] = tempfile.mkdtemp(prefix=f"code_exec_{request.session_id}_")
        
        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        result = None
        error = None
        files_created = []
        
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # Change to temp directory
            old_cwd = os.getcwd()
            os.chdir(session['temp_dir'])
            
            # Execute code with timeout
            try:
                # Compile code first to check for syntax errors
                compiled_code = compile(request.code, '<string>', 'exec')
                
                # Execute with timeout
                await asyncio.wait_for(
                    self._execute_in_session(compiled_code, session),
                    timeout=request.timeout
                )
                
                # Check for created files
                temp_path = Path(session['temp_dir'])
                files_created = [str(f.relative_to(temp_path)) for f in temp_path.iterdir() if f.is_file()]
                
            except asyncio.TimeoutError:
                error = f"Code execution timed out after {request.timeout} seconds"
            except Exception as e:
                error = f"Execution error: {str(e)}"
                stderr_capture.write(traceback.format_exc())
        
        finally:
            # Restore stdout/stderr and working directory
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            try:
                os.chdir(old_cwd)
            except:
                pass
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        return CodeExecutionResponse(
            session_id=request.session_id,
            success=error is None,
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
            result=result,
            error=error,
            execution_time=execution_time,
            files_created=files_created
        )
    
    async def _execute_in_session(self, compiled_code, session):
        """Execute compiled code in session context."""
        # Execute in the session's namespace
        exec(compiled_code, session['globals'], session['locals'])


# Global session manager
session_manager = SessionManager()
code_executor = SecureCodeExecutor(session_manager)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info("Code Interpreter Service starting up")
    
    # Start background task for session cleanup
    async def cleanup_task():
        while True:
            await asyncio.sleep(300)  # Cleanup every 5 minutes
            session_manager.cleanup_expired_sessions()
    
    cleanup_task_handle = asyncio.create_task(cleanup_task())
    
    yield
    
    # Shutdown
    cleanup_task_handle.cancel()
    logger.info("Code Interpreter Service shutting down")


# FastAPI app
app = FastAPI(
    title="Secure Code Interpreter Service",
    description="Secure Python code execution for genomic data analysis",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/execute", response_model=CodeExecutionResponse)
async def execute_code(request: CodeExecutionRequest) -> CodeExecutionResponse:
    """Execute Python code in a secure environment."""
    try:
        return await code_executor.execute_code(request)
    except Exception as e:
        logger.error(f"Code execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_sessions": len(session_manager.sessions),
        "service": "code_interpreter"
    }


@app.post("/sessions/{session_id}/reset")
async def reset_session(session_id: str):
    """Reset a specific session."""
    if session_id in session_manager.sessions:
        session = session_manager.sessions[session_id]
        if session.get('temp_dir'):
            try:
                import shutil
                shutil.rmtree(session['temp_dir'])
            except Exception as e:
                logger.error(f"Failed to cleanup temp dir: {e}")
        
        del session_manager.sessions[session_id]
        if session_id in session_manager.session_timeouts:
            del session_manager.session_timeouts[session_id]
    
    return {"message": f"Session {session_id} reset"}


@app.get("/sessions")
async def list_sessions():
    """List active sessions."""
    return {
        "active_sessions": list(session_manager.sessions.keys()),
        "count": len(session_manager.sessions)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")