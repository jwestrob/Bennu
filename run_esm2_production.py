#!/usr/bin/env python3
"""
Production ESM2 embedding generation with optimized settings.
"""

import sys
from pathlib import Path
import logging
import gc
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)

# Add current directory to path
sys.path.append('.')

from src.ingest import __06_esm2_embeddings as esm2_module

def run_optimized_esm2(stage03_dir: Path, output_dir: Path):
    """Run ESM2 with production-optimized settings."""
    
    # Use smaller model for faster processing
    model_name = "facebook/esm2_t6_8M_UR50D"  # 8M parameters (smallest)
    
    # Optimize for CPU and memory
    batch_size = 4  # Small batch for CPU
    
    print(f"Running ESM2 embeddings with optimized settings:")
    print(f"  Model: {model_name}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: CPU")
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    try:
        result = esm2_module.run_esm2_embeddings(
            stage03_dir=stage03_dir,
            output_dir=output_dir,
            model_name=model_name,
            batch_size=batch_size,
            force=True
        )
        
        print(f"\n✅ ESM2 embeddings completed successfully!")
        print(f"   Total proteins: {result['total_proteins']:,}")
        print(f"   Embedding dimension: {result['embedding_dim']}")
        print(f"   Output files:")
        print(f"     - Embeddings: {result['embeddings_file']}")
        print(f"     - FAISS index: {result['faiss_index_file']}")
        print(f"     - Manifest: {result['manifest_file']}")
        
        return result
        
    except Exception as e:
        print(f"❌ ESM2 embeddings failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    stage03_dir = Path("data/stage03_prodigal")
    output_dir = Path("data/stage06_esm2")
    
    if not stage03_dir.exists():
        print(f"❌ Stage 3 directory not found: {stage03_dir}")
        print("Run the pipeline through stage 3 first.")
        sys.exit(1)
    
    result = run_optimized_esm2(stage03_dir, output_dir)
    
    if result:
        print(f"\n🎉 Ready for LLM integration with {result['total_proteins']:,} protein embeddings!")
    else:
        sys.exit(1)