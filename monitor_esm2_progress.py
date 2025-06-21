#!/usr/bin/env python3
"""
Monitor ESM2 embedding generation progress.
"""

import time
from pathlib import Path
import json
import h5py
from datetime import datetime, timedelta

def monitor_progress(output_dir: Path = Path("data/stage06_esm2")):
    """Monitor ESM2 embedding generation progress."""
    
    print("🔍 ESM2 Embedding Progress Monitor")
    print(f"Watching: {output_dir}")
    print("Press Ctrl+C to stop monitoring")
    print("="*50)
    
    start_time = time.time()
    last_check = None
    
    try:
        while True:
            current_time = datetime.now()
            
            # Check if output directory exists
            if not output_dir.exists():
                print(f"⏳ Waiting for output directory to be created...")
                time.sleep(5)
                continue
            
            # Check for files
            embedding_file = output_dir / "protein_embeddings.h5"
            manifest_file = output_dir / "embedding_manifest.json"
            
            status = "🔄 In Progress"
            details = []
            
            if embedding_file.exists():
                try:
                    # Try to read current progress from HDF5 file
                    with h5py.File(embedding_file, 'r') as f:
                        if 'embeddings' in f:
                            current_proteins = f['embeddings'].shape[0]
                            details.append(f"Proteins processed: {current_proteins:,}")
                            
                            if current_proteins > 0:
                                # Calculate rate if we have previous data
                                if last_check and 'proteins' in last_check:
                                    elapsed = (current_time - last_check['time']).total_seconds()
                                    if elapsed > 0:
                                        rate = (current_proteins - last_check['proteins']) / elapsed
                                        if rate > 0:
                                            remaining = (10102 - current_proteins) / rate
                                            eta = current_time + timedelta(seconds=remaining)
                                            details.append(f"Rate: {rate:.1f} proteins/sec")
                                            details.append(f"ETA: {eta.strftime('%H:%M:%S')}")
                                
                                last_check = {'time': current_time, 'proteins': current_proteins}
                except:
                    details.append("File being written...")
            else:
                details.append("Embedding file not yet created")
            
            if manifest_file.exists():
                try:
                    with open(manifest_file) as f:
                        manifest = json.load(f)
                        if 'total_proteins' in manifest:
                            status = "✅ Complete"
                            details = [
                                f"Total proteins: {manifest['total_proteins']:,}",
                                f"Embedding dim: {manifest['embedding_dim']}",
                                f"Model: {manifest['model_name']}"
                            ]
                            break
                except:
                    pass
            
            # Print status
            elapsed = time.time() - start_time
            elapsed_str = f"{elapsed/60:.1f}m" if elapsed > 60 else f"{elapsed:.1f}s"
            
            print(f"\r{current_time.strftime('%H:%M:%S')} | {status} | Elapsed: {elapsed_str} | {' | '.join(details)}", end="", flush=True)
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Monitoring stopped")
        elapsed = time.time() - start_time
        print(f"Total monitoring time: {elapsed/60:.1f} minutes")
    
    # Final status check
    if manifest_file.exists():
        print(f"\n✅ ESM2 embeddings completed!")
        print(f"Run tests with: python test_esm2_similarity.py {output_dir}")
    else:
        print(f"\n⏳ ESM2 embeddings still in progress...")
        print(f"Check back later or run: python monitor_esm2_progress.py")


if __name__ == "__main__":
    import sys
    
    output_dir = Path("data/stage06_esm2")
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    
    monitor_progress(output_dir)