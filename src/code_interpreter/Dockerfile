# Secure Code Interpreter Container
# Based on security best practices for untrusted code execution

FROM python:3.11-slim

# Create non-root user
RUN groupadd -r codeexec && useradd -r -g codeexec -d /app -s /sbin/nologin codeexec

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set up application directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY service.py .

# Create necessary directories
RUN mkdir -p /tmp/code_execution && \
    chown -R codeexec:codeexec /app /tmp/code_execution

# Switch to non-root user
USER codeexec

# Expose port
EXPOSE 8000

# Security: Run as non-root, read-only filesystem
# Additional security will be enforced by container runtime flags:
# --read-only --cap-drop=ALL --security-opt=no-new-privileges --net=none

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Start the service
CMD ["python", "service.py"]