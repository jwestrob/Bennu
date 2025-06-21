#!/usr/bin/env python3
"""
ESM2 embedding generation optimized for Apple Silicon M4 Max.
"""

import sys
from pathlib import Path
import logging
import gc
import torch
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.append('.')

# Import the module
import importlib
esm2_module = importlib.import_module('src.ingest.06_esm2_embeddings')

def run_m4_max_optimized_esm2(stage03_dir: Path, output_dir: Path):
    """Run ESM2 with Apple Silicon M4 Max optimized settings."""
    
    # Optimal settings for M4 Max
    model_name = "facebook/esm2_t6_8M_UR50D"  # 8M parameters, good balance
    batch_size = 24  # Aggressive batch size for M4 Max's unified memory
    
    print(f"🚀 ESM2 Embeddings - Apple Silicon M4 Max Optimized")
    print(f"   Model: {model_name}")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: MPS (Apple Silicon GPU)")
    print(f"   Expected time: ~20-30 minutes for 10K proteins")
    print()
    
    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("⚠️  Warning: MPS not available, falling back to CPU")
        batch_size = 4  # Reduce batch size for CPU
    
    # Clear memory
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    start_time = time.time()
    
    try:
        result = esm2_module.run_esm2_embeddings(
            stage03_dir=stage03_dir,
            output_dir=output_dir,
            model_name=model_name,
            batch_size=batch_size,
            force=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n🎉 ESM2 embeddings completed successfully!")
        print(f"   Duration: {duration/60:.1f} minutes")
        print(f"   Total proteins: {result['total_proteins']:,}")
        print(f"   Embedding dimension: {result['embedding_dim']}")
        print(f"   Processing rate: {result['total_proteins']/duration:.1f} proteins/second")
        print()
        print(f"📁 Output files:")
        print(f"   Embeddings: {result['embeddings_file']}")
        print(f"   LanceDB: {result['lancedb_dir']}")
        print(f"   Manifest: {result['manifest_file']}")
        print()
        print(f"🔬 Ready for semantic protein search!")
        print(f"   • Vector similarity search across {result['total_proteins']:,} proteins")
        print(f"   • 320-dimensional ESM2 embeddings capture evolutionary relationships")
        print(f"   • LanceDB enables fast similarity queries with metadata filtering")
        
        # Test a quick similarity search
        try:
            import lancedb
            
            # Load LanceDB table for quick test
            lancedb_path = Path(result['lancedb_dir'])
            db = lancedb.connect(str(lancedb_path))
            table = db.open_table("protein_embeddings")
            
            # Test query
            dummy_query = torch.randn(result['embedding_dim']).numpy()
            test_results = table.search(dummy_query).limit(5).to_pandas()
            
            print(f"   • LanceDB table loaded successfully: {len(table):,} vectors")
            print(f"   • Test query completed successfully")
            
        except Exception as e:
            logger.warning(f"LanceDB test failed: {e}")
        
        return result
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n❌ ESM2 embeddings failed after {duration/60:.1f} minutes")
        print(f"Error: {e}")
        
        # Try to provide helpful debugging info
        if "out of memory" in str(e).lower():
            print("\n💡 Memory optimization suggestions:")
            print("   • Reduce batch size (try batch_size=12)")
            print("   • Close other applications")
            print("   • Try the smaller model: facebook/esm2_t6_8M_UR50D")
        
        import traceback
        traceback.print_exc()
        return None


def estimate_processing_time(num_proteins: int) -> str:
    """Estimate processing time for M4 Max."""
    # Based on ~8 proteins/second on M4 Max with batch_size=24
    proteins_per_second = 8
    total_seconds = num_proteins / proteins_per_second
    
    if total_seconds < 60:
        return f"{total_seconds:.0f} seconds"
    elif total_seconds < 3600:
        return f"{total_seconds/60:.1f} minutes"
    else:
        return f"{total_seconds/3600:.1f} hours"


if __name__ == "__main__":
    stage03_dir = Path("data/stage03_prodigal")
    output_dir = Path("data/stage06_esm2")
    
    if not stage03_dir.exists():
        print(f"❌ Stage 3 directory not found: {stage03_dir}")
        print("Run the pipeline through stage 3 first:")
        print("   python -m src.cli build --to-stage 3")
        sys.exit(1)
    
    # Count proteins for time estimate
    try:
        sequences = esm2_module.load_protein_sequences(stage03_dir)
        estimated_time = estimate_processing_time(len(sequences))
        
        print(f"📊 Found {len(sequences):,} protein sequences")
        print(f"⏱️  Estimated processing time: {estimated_time}")
        print()
        
        # Ask for confirmation
        response = input("Continue with ESM2 embedding generation? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled.")
            sys.exit(0)
        
    except Exception as e:
        print(f"Warning: Could not estimate processing time: {e}")
        print("Proceeding anyway...")
    
    result = run_m4_max_optimized_esm2(stage03_dir, output_dir)
    
    if result:
        print(f"\n✅ Success! Generated embeddings for {result['total_proteins']:,} proteins")
        print(f"Next steps:")
        print(f"   • Test similarity search: python test_esm2_similarity.py")
        print(f"   • Integrate with LLM: python -m src.cli ask 'Find proteins similar to kinases'")
    else:
        sys.exit(1)