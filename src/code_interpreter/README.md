# Secure Code Interpreter Service

A containerized, security-hardened Python code execution service designed for the genomic RAG system. Provides safe execution of data analysis and visualization code with multiple security layers.

## Features

- **Secure Execution Environment**: Containerized with dropped capabilities and read-only filesystem
- **Session Management**: Stateful sessions for iterative analysis
- **Genomic Analysis Tools**: Pre-loaded with scientific computing stack
- **Maximum Security**: Optional gVisor integration for VM-level isolation
- **Resource Limits**: Memory, CPU, and execution time constraints
- **Health Monitoring**: Built-in health checks and monitoring

## Security Architecture

### Standard Security (Docker)
- Non-root user execution (uid 1000)
- All capabilities dropped except SETUID/SETGID
- Read-only container filesystem
- Network isolation (disabled by default)
- Resource limits (512MB memory, 30s timeout)
- Secure tmpfs mounts for temporary files

### Maximum Security (gVisor)
- All standard security features +
- VM-level isolation with gVisor runtime
- Kernel syscall filtering and interception
- Enhanced container escape protection

## Quick Start

### 1. Standard Deployment

```bash
# Build and deploy with standard security
cd src/code_interpreter
./deploy.sh deploy

# Test the service
./deploy.sh test
```

### 2. Maximum Security Deployment

```bash
# Install gVisor (Ubuntu/Debian)
curl -fsSL https://gvisor.dev/archive.key | sudo gpg --dearmor -o /usr/share/keyrings/gvisor-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" | sudo tee /etc/apt/sources.list.d/gvisor.list > /dev/null
sudo apt-get update && sudo apt-get install -y runsc

# Configure Docker to use gVisor
sudo runsc install
sudo systemctl reload docker

# Deploy with maximum security
./deploy.sh deploy-max
```

## API Usage

### Execute Code

```bash
curl -X POST "http://localhost:8000/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session",
    "code": "import numpy as np\nprint(\"Hello from secure Python!\")",
    "timeout": 30
  }'
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Reset Session

```bash
curl -X POST "http://localhost:8000/sessions/my-session/reset"
```

## Integration with RAG System

The code interpreter integrates seamlessly with the agentic RAG system:

```python
from src.code_interpreter.client import code_interpreter_tool

# Execute code via the RAG system
result = await code_interpreter_tool(
    code="""
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Analyze protein similarity data
    similarities = [0.8, 0.7, 0.9, 0.6, 0.75]
    plt.hist(similarities, bins=10)
    plt.title('Protein Similarity Distribution')
    plt.savefig('similarity_hist.png')
    print(f'Mean similarity: {sum(similarities)/len(similarities):.3f}')
    """,
    session_id="analysis-session",
    timeout=30
)

print(result['stdout'])  # Analysis output
print(result['files_created'])  # ['similarity_hist.png']
```

## Genomic Analysis Examples

### Protein Similarity Analysis

```python
from src.code_interpreter.client import GenomicCodeInterpreter, CodeInterpreterClient

client = CodeInterpreterClient()
interpreter = GenomicCodeInterpreter(client)

# Set up environment
await interpreter.setup_genomic_environment({
    'proteins': '/data/proteins.fasta',
    'annotations': '/data/annotations.gff'
})

# Analyze protein similarities
result = await interpreter.analyze_protein_similarities([
    'protein_001', 'protein_002', 'protein_003'
])
```

### Genomic Neighborhood Visualization

```python
gene_data = [
    {'id': 'gene1', 'start': 1000, 'end': 2000, 'strand': 1, 'function': 'transport'},
    {'id': 'gene2', 'start': 2500, 'end': 3500, 'strand': -1, 'function': 'regulation'},
    {'id': 'gene3', 'start': 4000, 'end': 5000, 'strand': 1, 'function': 'metabolism'}
]

result = await interpreter.plot_genomic_neighborhood(gene_data)
# Creates genomic_neighborhood.png visualization
```

## Available Python Packages

The secure environment includes:

- **Data Analysis**: pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn
- **Bioinformatics**: (can be added as needed)
- **Utilities**: json, pathlib, datetime

## Management Commands

```bash
# Deployment
./deploy.sh build          # Build Docker image
./deploy.sh deploy         # Deploy standard security
./deploy.sh deploy-max     # Deploy maximum security
./deploy.sh stop           # Stop services
./deploy.sh restart        # Restart services

# Monitoring
./deploy.sh status         # Show service status
./deploy.sh logs           # Follow service logs
./deploy.sh test           # Run test execution

# Development
python -m pytest src/tests/test_code_interpreter.py  # Run tests
```

## Security Considerations

### What's Protected
- Container escape attacks (gVisor)
- Privilege escalation (capabilities dropped)
- File system access (read-only + controlled mounts)
- Network access (disabled by default)
- Resource exhaustion (memory/CPU/time limits)
- Persistent state pollution (session isolation)

### What's Not Protected
- Side-channel attacks between sessions
- Inference attacks on co-located containers
- Host kernel vulnerabilities (mitigated by gVisor)

### Security Best Practices
1. Always use gVisor for maximum security in production
2. Monitor resource usage and set appropriate limits
3. Regularly update base images and dependencies
4. Use network isolation when possible
5. Implement session timeouts for long-running analyses

## Troubleshooting

### Service Won't Start
```bash
# Check Docker daemon
sudo systemctl status docker

# Check port availability
netstat -tulpn | grep :8000

# View detailed logs
./deploy.sh logs
```

### gVisor Issues
```bash
# Verify gVisor installation
runsc --version

# Check Docker runtime configuration
docker info | grep -i runtime

# Test gVisor manually
docker run --runtime=runsc hello-world
```

### Performance Issues
```bash
# Monitor resource usage
docker stats

# Check memory limits
docker exec code-interpreter cat /sys/fs/cgroup/memory/memory.limit_in_bytes

# Increase limits in docker-compose.yml if needed
```

## Development

### Running Tests
```bash
# Run all code interpreter tests
python -m pytest src/tests/test_code_interpreter.py -v

# Run with coverage
python -m pytest src/tests/test_code_interpreter.py --cov=src.code_interpreter

# Run specific test class
python -m pytest src/tests/test_code_interpreter.py::TestSecurityFeatures -v
```

### Adding New Tools
1. Update `GenomicCodeInterpreter` class with new methods
2. Add corresponding tests
3. Update documentation

### Security Testing
```bash
# Test container isolation
docker exec code-interpreter ls /proc/1/  # Should fail or show container init

# Test capability restrictions
docker exec code-interpreter capsh --print  # Should show minimal capabilities

# Test filesystem restrictions
docker exec code-interpreter touch /test  # Should fail (read-only)
```

## Configuration

Environment variables (set in docker-compose.yml):
- `MAX_EXECUTION_TIME`: Maximum code execution time (default: 30s)
- `MAX_MEMORY_MB`: Maximum memory usage (default: 512MB)
- `MAX_OUTPUT_SIZE`: Maximum output size (default: 1MB)
- `ENABLE_NETWORKING`: Enable network access (default: false)

## License

Part of the Microbial Claude Matter genomic intelligence platform.