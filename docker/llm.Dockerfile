# Dockerfile for LLM Integration Service
# Designed for containerized deployment with Nextflow

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-llm.txt .
RUN pip install --no-cache-dir -r requirements-llm.txt

# Copy source code
COPY src/ ./src/
COPY *.py ./

# Set environment variables for container deployment
ENV PYTHONPATH=/app
ENV NEO4J_URI=bolt://neo4j:7687
ENV NEO4J_USER=neo4j
ENV NEO4J_PASSWORD=genomics2024
ENV LANCEDB_PATH=/data/lancedb

# Create data directory
RUN mkdir -p /data

# Expose port for potential web service
EXPOSE 8000

# Default command runs the CLI
ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["ask", "--help"]