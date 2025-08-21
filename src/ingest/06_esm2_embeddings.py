#!/usr/bin/env python3
"""
Stage 6: ESM2 Protein Embeddings
Generate protein embeddings using ESM2 transformer model for semantic search.
"""

import logging
import json
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModel
import lancedb
from rich.console import Console
from rich.progress import Progress
from enum import Enum

console = Console()
logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Different strategies for aggregating sliding window embeddings."""
    MAX_POOL = "max_pool"          # Element-wise maximum (preserves strongest features)
    MEAN_POOL = "mean_pool"        # Element-wise average (balanced representation)
    WEIGHTED_MEAN = "weighted_mean" # Length-weighted average (longer windows weighted more)
    ATTENTION_POOL = "attention"    # Learned attention weights (future enhancement)


@dataclass
class ProteinSequence:
    """Container for protein sequence information."""
    protein_id: str
    genome_id: str
    sequence: str
    length: int
    source_file: Path


class ESM2EmbeddingGenerator:
    """Generate protein embeddings using ESM2 transformer model with sliding window support."""
    
    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D", device: Optional[str] = None,
                 window_size: int = 1024, overlap: int = 256, 
                 aggregation: AggregationStrategy = AggregationStrategy.MAX_POOL):
        """
        Initialize ESM2 model for protein embedding generation with sliding window support.
        
        Args:
            model_name: HuggingFace model identifier for ESM2 variant
            device: Device to run inference on ('cpu', 'cuda', 'mps', or None for auto)
            window_size: Size of sliding window for long sequences (amino acids)
            overlap: Overlap between consecutive windows (amino acids)
            aggregation: Strategy for combining window embeddings
        """
        self.model_name = model_name
        self.window_size = window_size
        self.overlap = overlap
        self.aggregation = aggregation
        
        # Auto-detect best device for Apple Silicon
        if device is None:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = 'mps'  # Apple Silicon GPU
            elif torch.cuda.is_available():
                self.device = 'cuda'  # NVIDIA GPU
            else:
                self.device = 'cpu'   # CPU fallback
        else:
            self.device = device
        
        console.print(f"Loading ESM2 model: {model_name}")
        console.print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        console.print(f"Embedding dimension: {self.embedding_dim}")
        console.print(f"Sliding window: {self.window_size} AA, overlap: {self.overlap} AA")
        console.print(f"Aggregation strategy: {self.aggregation.value}")
        
    def embed_sequences(self, sequences: List[ProteinSequence], 
                       batch_size: int = 8) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for a list of protein sequences.
        
        Args:
            sequences: List of ProteinSequence objects
            batch_size: Number of sequences to process in each batch
            
        Returns:
            Dict mapping protein_id to embedding vector
        """
        embeddings = {}
        
        with Progress(console=console) as progress:
            task = progress.add_task("Generating embeddings...", total=len(sequences))
            
            with torch.no_grad():
                for i in range(0, len(sequences), batch_size):
                    batch = sequences[i:i + batch_size]
                    
                    # Dynamically adjust batch size for very long sequences to manage memory
                    max_seq_len = max(len(seq.sequence) for seq in batch)
                    if max_seq_len > 2000 and len(batch) > 2:
                        # Process long sequences individually to avoid memory issues
                        console.print(f"[yellow]Long sequences detected (max: {max_seq_len} AA), processing individually[/yellow]")
                        for single_seq in batch:
                            single_embedding = self._embed_batch([single_seq])
                            embeddings.update(single_embedding)
                            progress.advance(task, 1)
                    else:
                        batch_embeddings = self._embed_batch(batch)
                        embeddings.update(batch_embeddings)
                        progress.advance(task, len(batch))
        
        return embeddings
    
    def _should_use_sliding_window(self, sequence: str, threshold: int = None) -> bool:
        """Determine if sequence needs sliding window processing."""
        if threshold is None:
            threshold = self.window_size
        # Account for special tokens and safety margin
        effective_limit = threshold - 10  # Safety margin for special tokens
        return len(sequence) > effective_limit
    
    def _embed_long_sequence_with_sliding_window(self, sequence: str, protein_id: str) -> np.ndarray:
        """
        Process long sequences using sliding window approach with overlap.
        
        Args:
            sequence: Protein sequence (amino acids)
            protein_id: Protein identifier for logging
            
        Returns:
            Aggregated embedding representing full sequence
        """
        # 1. Create overlapping windows
        windows = []
        start = 0
        while start < len(sequence):
            end = min(start + self.window_size, len(sequence))
            window = sequence[start:end]
            windows.append(window)
            
            if end == len(sequence):
                break
            start += (self.window_size - self.overlap)
        
        # Suppress sliding window logging for clean progress bar
        
        # 2. Generate embeddings for each window
        window_embeddings = []
        window_lengths = []
        
        with torch.no_grad():
            for i, window in enumerate(windows):
                try:
                    # Process individual window (guaranteed to fit in memory)
                    inputs = self.tokenizer(
                        [window], 
                        return_tensors="pt", 
                        truncation=False
                    ).to(self.device)
                    
                    outputs = self.model(**inputs)
                    window_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
                    window_embeddings.append(window_emb)
                    window_lengths.append(len(window))
                    
                    # Clean up immediately
                    if self.device in ['cuda', 'mps']:
                        del inputs, outputs
                        if self.device == 'cuda':
                            torch.cuda.empty_cache()
                        elif self.device == 'mps':
                            torch.mps.empty_cache()
                            
                except Exception as e:
                    logger.error(f"Failed to process window {i} of {protein_id}: {e}")
                    # Use zero embedding for failed window
                    window_embeddings.append(np.zeros(self.embedding_dim))
                    window_lengths.append(len(window))
        
        # 3. Aggregate window embeddings using specified strategy
        if not window_embeddings:
            logger.error(f"No valid windows processed for {protein_id}")
            return np.zeros(self.embedding_dim)
            
        if self.aggregation == AggregationStrategy.MAX_POOL:
            # Element-wise maximum preserves strongest features
            aggregated_embedding = np.maximum.reduce(window_embeddings)
        elif self.aggregation == AggregationStrategy.MEAN_POOL:
            # Simple average
            aggregated_embedding = np.mean(window_embeddings, axis=0)
        elif self.aggregation == AggregationStrategy.WEIGHTED_MEAN:
            # Weight by window length
            weights = np.array(window_lengths) / sum(window_lengths)
            aggregated_embedding = np.average(window_embeddings, axis=0, weights=weights)
        else:
            # Default to max pooling
            logger.warning(f"Unknown aggregation strategy {self.aggregation}, using max pooling")
            aggregated_embedding = np.maximum.reduce(window_embeddings)
        
        return aggregated_embedding
    
    def _embed_batch_with_smart_windowing(self, sequences: List[ProteinSequence]) -> Dict[str, np.ndarray]:
        """
        Smart batch processing that separates short and long sequences.
        
        - Short sequences: Normal batch processing
        - Long sequences: Individual sliding window processing
        """
        short_sequences = [seq for seq in sequences if not self._should_use_sliding_window(seq.sequence)]
        long_sequences = [seq for seq in sequences if self._should_use_sliding_window(seq.sequence)]
        
        embeddings = {}
        
        # Process short sequences in normal batches
        if short_sequences:
            embeddings.update(self._embed_normal_batch(short_sequences))
        
        # Process long sequences individually with sliding window
        for seq in long_sequences:
            embedding = self._embed_long_sequence_with_sliding_window(seq.sequence, seq.protein_id)
            embeddings[seq.protein_id] = embedding
        
        return embeddings
    
    def _embed_normal_batch(self, sequences: List[ProteinSequence]) -> Dict[str, np.ndarray]:
        """Generate embeddings for a batch of normal-length sequences."""
        # Prepare batch data
        protein_ids = [seq.protein_id for seq in sequences]
        sequence_texts = [seq.sequence for seq in sequences]
        
        # Log sequence length statistics for this batch
        lengths = [len(seq) for seq in sequence_texts]
        max_len = max(lengths)
        # Completely suppress normal batch logging for clean progress bar
        
        # Tokenize sequences WITHOUT truncation to preserve full biological information
        try:
            inputs = self.tokenizer(
                sequence_texts,
                return_tensors="pt",
                padding=True,
                truncation=False  # CRITICAL: Preserve full sequence lengths
            ).to(self.device)
            
            # Generate embeddings
            outputs = self.model(**inputs)
            
            # Extract per-sequence embeddings (mean pooling over sequence length)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
            embeddings = embeddings.cpu().numpy()
            
            # Clean up GPU memory
            if self.device in ['cuda', 'mps']:
                del inputs, outputs
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device == 'mps':
                    torch.mps.empty_cache()
            
            # Map protein IDs to embeddings
            batch_embeddings = {}
            for protein_id, embedding in zip(protein_ids, embeddings):
                batch_embeddings[protein_id] = embedding
                
            return batch_embeddings
            
        except Exception as e:
            logger.error(f"Error embedding normal batch (max_len={max_len}): {e}")
            # Clean up on error
            if self.device in ['cuda', 'mps']:
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device == 'mps':
                    torch.mps.empty_cache()
            
            # Return zero embeddings for failed sequences
            return {protein_id: np.zeros(self.embedding_dim) for protein_id in protein_ids}
    
    def _embed_batch(self, sequences: List[ProteinSequence]) -> Dict[str, np.ndarray]:
        """Generate embeddings for a batch of sequences using smart windowing."""
        return self._embed_batch_with_smart_windowing(sequences)


def load_protein_sequences(stage03_dir: Path) -> List[ProteinSequence]:
    """
    Load all protein sequences from prodigal output.
    
    Args:
        stage03_dir: Path to stage03_prodigal directory
        
    Returns:
        List of ProteinSequence objects
    """
    sequences = []
    genomes_dir = stage03_dir / "genomes"
    
    if not genomes_dir.exists():
        raise FileNotFoundError(f"Genomes directory not found: {genomes_dir}")
    
    console.print(f"Loading protein sequences from: {genomes_dir}")
    
    for genome_dir in genomes_dir.iterdir():
        if not genome_dir.is_dir():
            continue
            
        # Skip symlink directory
        if genome_dir.name == "all_protein_symlinks":
            continue
            
        genome_id = genome_dir.name
        
        # Find protein FASTA file
        protein_files = list(genome_dir.glob("*.faa"))
        if not protein_files:
            logger.warning(f"No protein file found for genome: {genome_id}")
            continue
            
        protein_file = protein_files[0]
        genome_sequences = load_fasta_sequences(protein_file, genome_id)
        sequences.extend(genome_sequences)
        
    console.print(f"Loaded {len(sequences)} protein sequences from {len(set(seq.genome_id for seq in sequences))} genomes")
    return sequences


def load_fasta_sequences(fasta_file: Path, genome_id: str) -> List[ProteinSequence]:
    """Load protein sequences from a FASTA file."""
    sequences = []
    
    try:
        with open(fasta_file, 'r') as f:
            current_id = None
            current_seq = []
            
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous sequence
                    if current_id and current_seq:
                        sequence = ''.join(current_seq)
                        # Remove stop codon if present
                        sequence = sequence.rstrip('*')
                        sequences.append(ProteinSequence(
                            protein_id=current_id,
                            genome_id=genome_id,
                            sequence=sequence,
                            length=len(sequence),
                            source_file=fasta_file
                        ))
                    
                    # Start new sequence
                    current_id = line[1:].split()[0]  # Take first part of header
                    current_seq = []
                else:
                    current_seq.append(line)
            
            # Save last sequence
            if current_id and current_seq:
                sequence = ''.join(current_seq)
                sequence = sequence.rstrip('*')
                sequences.append(ProteinSequence(
                    protein_id=current_id,
                    genome_id=genome_id,
                    sequence=sequence,
                    length=len(sequence),
                    source_file=fasta_file
                ))
                
    except Exception as e:
        logger.error(f"Error loading FASTA file {fasta_file}: {e}")
        
    return sequences


def create_lancedb_table(embeddings: Dict[str, np.ndarray], 
                         sequences: List[ProteinSequence],
                         db_path: Path) -> Any:
    """
    Create LanceDB table for fast similarity search.
    
    Args:
        embeddings: Dict mapping protein_id to embedding vector
        sequences: Original protein sequence data for metadata
        db_path: Path to LanceDB database
        
    Returns:
        LanceDB table object
    """
    console.print("Creating LanceDB table for similarity search...")
    
    # Create LanceDB database
    db = lancedb.connect(str(db_path))
    
    # Prepare data for LanceDB
    protein_data = []
    sequence_lookup = {seq.protein_id: seq for seq in sequences}
    
    for protein_id, embedding in embeddings.items():
        seq_info = sequence_lookup.get(protein_id)
        protein_data.append({
            "protein_id": protein_id,
            "genome_id": seq_info.genome_id if seq_info else "unknown",
            "sequence_length": seq_info.length if seq_info else 0,
            "source_file": str(seq_info.source_file) if seq_info else "",
            "vector": embedding.astype(np.float32)
        })
    
    # Create table with embeddings and metadata
    table = db.create_table("protein_embeddings", data=protein_data, mode="overwrite")
    
    console.print(f"LanceDB table created with {len(protein_data)} vectors")
    return table


def save_embeddings_and_db(embeddings: Dict[str, np.ndarray],
                          lancedb_table: Any,
                          sequences: List[ProteinSequence],
                          output_dir: Path,
                          model_name: str,
                          embedding_generator: ESM2EmbeddingGenerator) -> Dict[str, Any]:
    """
    Save embeddings, LanceDB table, and metadata to disk.
    
    Args:
        embeddings: Dict mapping protein_id to embedding vector
        lancedb_table: LanceDB table for similarity search
        sequences: Original protein sequence data
        output_dir: Output directory for files
        model_name: ESM2 model name used
        embedding_generator: ESM2 generator with sliding window config
        
    Returns:
        Dict containing saved file information
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings in HDF5 format
    embeddings_file = output_dir / "protein_embeddings.h5"
    console.print(f"Saving embeddings to: {embeddings_file}")
    
    with h5py.File(embeddings_file, 'w') as f:
        # Create datasets
        protein_ids_list = list(embeddings.keys())
        embedding_matrix = np.vstack([embeddings[pid] for pid in protein_ids_list])
        
        f.create_dataset('protein_ids', data=[pid.encode('utf-8') for pid in protein_ids_list])
        f.create_dataset('embeddings', data=embedding_matrix)
        f.attrs['model_name'] = model_name
        f.attrs['embedding_dim'] = embedding_matrix.shape[1]
        f.attrs['num_proteins'] = len(protein_ids_list)
        f.attrs['created_at'] = datetime.now().isoformat()
    
    # LanceDB table is already persisted in the database
    lancedb_dir = output_dir / "lancedb"
    console.print(f"LanceDB database saved to: {lancedb_dir}")
    
    # Create embedding manifest with metadata
    manifest = {
        "version": "0.1.0",
        "created_at": datetime.now().isoformat(),
        "model_name": model_name,
        "embedding_dim": embeddings[next(iter(embeddings))].shape[0],
        "total_proteins": len(embeddings),
        "genomes_processed": len(set(seq.genome_id for seq in sequences)),
        "output_files": {
            "embeddings": str(embeddings_file),
            "lancedb": str(lancedb_dir)
        },
        "statistics": {
            "min_protein_length": min(seq.length for seq in sequences),
            "max_protein_length": max(seq.length for seq in sequences),
            "mean_protein_length": np.mean([seq.length for seq in sequences]),
            "proteins_over_1024_aa": len([s for s in sequences if s.length > 1024]),
            "proteins_over_2000_aa": len([s for s in sequences if s.length > 2000]),
            "proteins_using_sliding_window": len([s for s in sequences if s.length > 1024]),
            "truncation_policy": "NONE - Sliding window for long sequences",
            "sliding_window_config": {
                "window_size": embedding_generator.window_size,
                "overlap": embedding_generator.overlap,
                "aggregation_strategy": embedding_generator.aggregation.value
            },
            "genome_protein_counts": {
                genome_id: len([s for s in sequences if s.genome_id == genome_id])
                for genome_id in set(seq.genome_id for seq in sequences)
            }
        }
    }
    
    manifest_file = output_dir / "embedding_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    console.print(f"Embedding manifest saved to: {manifest_file}")
    
    return {
        "embeddings_file": str(embeddings_file),
        "lancedb_dir": str(lancedb_dir),
        "manifest_file": str(manifest_file),
        "total_proteins": len(embeddings),
        "embedding_dim": manifest["embedding_dim"]
    }


def run_esm2_embeddings(stage03_dir: Path, output_dir: Path, 
                       model_name: str = "facebook/esm2_t6_8M_UR50D",
                       batch_size: Optional[int] = None,
                       window_size: int = 1024,
                       overlap: int = 256,
                       aggregation: AggregationStrategy = AggregationStrategy.MAX_POOL,
                       force: bool = False) -> Dict[str, Any]:
    """
    Generate ESM2 embeddings for all proteins in stage03 output with sliding window support.
    
    Args:
        stage03_dir: Path to stage03_prodigal directory
        output_dir: Output directory for embeddings and indices
        model_name: ESM2 model variant to use
        batch_size: Batch size for inference
        window_size: Size of sliding window for long sequences (amino acids)
        overlap: Overlap between consecutive windows (amino acids)
        aggregation: Strategy for combining window embeddings
        force: Overwrite existing outputs
        
    Returns:
        Dict containing processing results and output files
    """
    console.print("[bold blue]Stage 6: ESM2 Protein Embeddings[/bold blue]")
    
    # Check if output already exists
    if output_dir.exists() and not force:
        manifest_file = output_dir / "embedding_manifest.json"
        if manifest_file.exists():
            console.print(f"[yellow]Output directory exists: {output_dir}[/yellow]")
            console.print("[yellow]Use --force to overwrite existing results[/yellow]")
            with open(manifest_file) as f:
                existing_manifest = json.load(f)
            return existing_manifest
    
    # Create output directory
    if output_dir.exists() and force:
        console.print(f"Removing existing output directory: {output_dir}")
        import shutil
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load protein sequences
    sequences = load_protein_sequences(stage03_dir)
    if not sequences:
        raise ValueError("No protein sequences found in stage03 output")
    
    # Initialize ESM2 model with sliding window parameters
    embedding_generator = ESM2EmbeddingGenerator(
        model_name=model_name,
        window_size=window_size,
        overlap=overlap,
        aggregation=aggregation
    )
    
    # Auto-optimize batch size based on device
    if batch_size is None:
        if embedding_generator.device == 'mps':
            batch_size = 16  # Apple Silicon can handle larger batches
        elif embedding_generator.device == 'cuda':
            batch_size = 32  # NVIDIA GPU can handle even larger batches
        else:
            batch_size = 4   # Conservative for CPU
    
    console.print(f"Using batch size: {batch_size}")
    
    # Generate embeddings
    console.print(f"Generating embeddings for {len(sequences)} proteins...")
    embeddings = embedding_generator.embed_sequences(sequences, batch_size=batch_size)
    
    # Create LanceDB table for similarity search
    lancedb_path = output_dir / "lancedb"
    lancedb_table = create_lancedb_table(embeddings, sequences, lancedb_path)
    
    # Save results
    result = save_embeddings_and_db(
        embeddings=embeddings,
        lancedb_table=lancedb_table,
        sequences=sequences,
        output_dir=output_dir,
        model_name=model_name,
        embedding_generator=embedding_generator
    )
    
    console.print(f"[green]✓ ESM2 embeddings completed successfully[/green]")
    console.print(f"Generated embeddings for {result['total_proteins']} proteins")
    console.print(f"Embedding dimension: {result['embedding_dim']}")
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python 06_esm2_embeddings.py <stage03_dir> <output_dir>")
        sys.exit(1)
    
    stage03_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    logging.basicConfig(level=logging.INFO)
    
    result = run_esm2_embeddings(stage03_dir, output_dir)
    print(f"ESM2 embeddings completed: {result['total_proteins']} proteins processed")