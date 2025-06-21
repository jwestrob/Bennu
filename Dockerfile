FROM continuumio/miniconda3:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment
COPY env/environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Install bioinformatics tools
RUN wget https://github.com/hyattpd/Prodigal/releases/download/v2.6.3/prodigal.linux \
    -O /usr/local/bin/prodigal && \
    chmod +x /usr/local/bin/prodigal

# Install QUAST
RUN pip install quast

# Install DFAST_QC 
RUN pip install dfast_qc

# Set up conda environment activation
SHELL ["conda", "run", "-n", "genome-kg", "/bin/bash", "-c"]

# Copy pipeline scripts
COPY src/ /opt/pipeline/src/
COPY bin/ /opt/pipeline/bin/

# Set working directory and PATH
WORKDIR /opt/pipeline
ENV PATH="/opt/pipeline/bin:$PATH"

# Default command
CMD ["bash"]