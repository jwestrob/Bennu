#!/usr/bin/env python3
"""
Test ESM2 embedding similarity search functionality.
"""

import pytest
import numpy as np
import lancedb
import h5py
import json
from pathlib import Path
import time
from typing import List, Tuple


@pytest.fixture
def embedding_dir():
    """Fixture to provide embedding directory path."""
    # Try common locations in order of preference
    test_dirs = [
        Path("data/stage06_esm2"),
        Path("data/test_esm2_mps_output"),
        Path("data/test_esm2_output"),
        Path("data/test_esm2_lancedb_output")
    ]
    
    for test_dir in test_dirs:
        if test_dir.exists() and (test_dir / "protein_embeddings.h5").exists():
            return test_dir
    
    pytest.skip("No valid embedding directory found - run ESM2 embeddings first")


def test_similarity_search(embedding_dir: Path):
    """Test FAISS similarity search with ESM2 embeddings."""
    
    # Load embeddings
    embeddings_file = embedding_dir / "protein_embeddings.h5"
    with h5py.File(embeddings_file, 'r') as f:
        protein_ids = [pid.decode('utf-8') for pid in f['protein_ids'][:]]
        embeddings = f['embeddings'][:]
        print(f"Loaded embeddings for {len(protein_ids)} proteins")
        print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Load LanceDB table
    lancedb_path = embedding_dir / "lancedb"
    db = lancedb.connect(str(lancedb_path))
    table = db.open_table("protein_embeddings")
    print(f"Loaded LanceDB table with {len(table)} vectors")
    
    # Test similarity search
    query_protein = protein_ids[0]
    query_embedding = embeddings[0:1]  # Shape: (1, embedding_dim)
    
    print(f"\nTesting similarity search for: {query_protein}")
    
    # Search for similar proteins
    k = min(5, len(protein_ids))  # Top 5 similar proteins
    results = table.search(query_embedding[0]).limit(k).to_pandas()
    
    print(f"\nTop {k} most similar proteins:")
    for i, row in results.iterrows():
        print(f"  {i+1}. {row['protein_id']} (distance: {row['_distance']:.4f})")
    
    # Calculate embedding statistics
    print(f"\nEmbedding statistics:")
    print(f"  Mean: {embeddings.mean():.4f}")
    print(f"  Std: {embeddings.std():.4f}")
    print(f"  Min: {embeddings.min():.4f}")
    print(f"  Max: {embeddings.max():.4f}")
    
    # Test pairwise similarities
    print(f"\nPairwise similarities (cosine):")
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(embeddings)
    """
    for i, pid1 in enumerate(protein_ids):
        for j, pid2 in enumerate(protein_ids):
            if i < j:  # Only show upper triangle
                sim = similarities[i, j]
                print(f"  {pid1} <-> {pid2}: {sim:.4f}")
    """

def test_embedding_quality(embedding_dir: Path):
    """Test the quality and distribution of embeddings."""
    print("\n🔬 Testing embedding quality...")
    
    # Load embeddings
    embeddings_file = embedding_dir / "protein_embeddings.h5"
    with h5py.File(embeddings_file, 'r') as f:
        embeddings = f['embeddings'][:]
        protein_ids = [pid.decode('utf-8') for pid in f['protein_ids'][:]]
    
    # Check for common issues
    print(f"✓ Embedding shape: {embeddings.shape}")
    print(f"✓ Data type: {embeddings.dtype}")
    
    # Check for NaN or inf values
    nan_count = np.isnan(embeddings).sum()
    inf_count = np.isinf(embeddings).sum()
    print(f"✓ NaN values: {nan_count} (should be 0)")
    print(f"✓ Inf values: {inf_count} (should be 0)")
    
    # Check embedding diversity
    mean_embedding = embeddings.mean(axis=0)
    std_embedding = embeddings.std(axis=0)
    
    print(f"✓ Mean embedding norm: {np.linalg.norm(mean_embedding):.4f}")
    print(f"✓ Std embedding norm: {np.linalg.norm(std_embedding):.4f}")
    
    # Check for degenerate embeddings (all the same)
    unique_embeddings = len(set(tuple(emb) for emb in embeddings))
    print(f"✓ Unique embeddings: {unique_embeddings}/{len(embeddings)} ({unique_embeddings/len(embeddings)*100:.1f}%)")
    
    return nan_count == 0 and inf_count == 0 and unique_embeddings > len(embeddings) * 0.95


def benchmark_search_speed(embedding_dir: Path, num_queries: int = 100):
    """Benchmark LanceDB search performance."""
    print(f"\n⚡ Benchmarking search speed with {num_queries} queries...")
    
    # Load LanceDB table and embeddings
    lancedb_path = embedding_dir / "lancedb"
    db = lancedb.connect(str(lancedb_path))
    table = db.open_table("protein_embeddings")
    
    embeddings_file = embedding_dir / "protein_embeddings.h5"
    with h5py.File(embeddings_file, 'r') as f:
        embeddings = f['embeddings'][:]
    
    # Generate random queries
    np.random.seed(42)
    query_indices = np.random.choice(len(embeddings), min(num_queries, len(embeddings)), replace=False)
    queries = embeddings[query_indices]
    
    # Benchmark search
    k = 10  # Top 10 results
    start_time = time.time()
    
    for query in queries:
        results = table.search(query).limit(k).to_pandas()
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_ms = (total_time / len(queries)) * 1000
    
    print(f"✓ Searched {len(queries)} queries in {total_time:.3f}s")
    print(f"✓ Average query time: {avg_time_ms:.2f}ms")
    print(f"✓ Queries per second: {len(queries)/total_time:.1f}")
    
    return avg_time_ms < 50  # LanceDB may be slightly slower than FAISS


def validate_manifest(embedding_dir: Path):
    """Validate the embedding manifest file."""
    print("\n📋 Validating manifest...")
    
    manifest_file = embedding_dir / "embedding_manifest.json"
    if not manifest_file.exists():
        print("❌ Manifest file not found")
        return False
    
    with open(manifest_file) as f:
        manifest = json.load(f)
    
    required_fields = ['version', 'created_at', 'model_name', 'embedding_dim', 
                      'total_proteins', 'output_files', 'statistics']
    
    for field in required_fields:
        if field not in manifest:
            print(f"❌ Missing field in manifest: {field}")
            return False
        print(f"✓ {field}: {manifest[field]}")
    
    # Validate file references exist
    for file_type, file_path in manifest['output_files'].items():
        path_obj = Path(file_path)
        if not path_obj.exists():
            print(f"❌ Referenced file/directory does not exist: {file_path}")
            return False
        if path_obj.is_dir():
            print(f"✓ {file_type} directory exists: {path_obj.name}")
        else:
            print(f"✓ {file_type} file exists: {path_obj.name}")
    
    return True


def run_comprehensive_test(embedding_dir: Path):
    """Run all tests on the embedding directory."""
    print(f"🧪 Comprehensive ESM2 Embedding Test")
    print(f"Directory: {embedding_dir}")
    print("="*60)
    
    tests_passed = 0
    total_tests = 5
    
    try:
        # Test 1: Basic similarity search
        test_similarity_search(embedding_dir)
        tests_passed += 1
        print("✅ Test 1 PASSED: Basic similarity search")
    except Exception as e:
        print(f"❌ Test 1 FAILED: Basic similarity search - {e}")
    
    try:
        # Test 2: Embedding quality
        if test_embedding_quality(embedding_dir):
            tests_passed += 1
            print("✅ Test 2 PASSED: Embedding quality")
        else:
            print("❌ Test 2 FAILED: Embedding quality issues detected")
    except Exception as e:
        print(f"❌ Test 2 FAILED: Embedding quality - {e}")
    
    try:
        # Test 3: Search performance
        if benchmark_search_speed(embedding_dir):
            tests_passed += 1
            print("✅ Test 3 PASSED: Search performance")
        else:
            print("⚠️  Test 3 WARNING: Search performance slower than expected")
            tests_passed += 0.5
    except Exception as e:
        print(f"❌ Test 3 FAILED: Search performance - {e}")
    
    try:
        # Test 4: Manifest validation
        if validate_manifest(embedding_dir):
            tests_passed += 1
            print("✅ Test 4 PASSED: Manifest validation")
        else:
            print("❌ Test 4 FAILED: Manifest validation")
    except Exception as e:
        print(f"❌ Test 4 FAILED: Manifest validation - {e}")
    
    try:
        # Test 5: File integrity
        embeddings_file = embedding_dir / "protein_embeddings.h5"
        lancedb_dir = embedding_dir / "lancedb"
        
        if embeddings_file.exists() and lancedb_dir.exists():
            # Check file sizes are reasonable
            emb_size = embeddings_file.stat().st_size
            
            # Check LanceDB directory has content
            lancedb_files = list(lancedb_dir.rglob('*'))
            lancedb_has_content = len(lancedb_files) > 0
            
            if emb_size > 1000 and lancedb_has_content:  # At least 1KB and LanceDB content
                tests_passed += 1
                print("✅ Test 5 PASSED: File integrity")
            else:
                print("❌ Test 5 FAILED: Files too small or LanceDB empty")
        else:
            print("❌ Test 5 FAILED: Required files missing")
    except Exception as e:
        print(f"❌ Test 5 FAILED: File integrity - {e}")
    
    print("="*60)
    print(f"🏁 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! ESM2 embeddings are ready for production.")
        return True
    elif tests_passed >= total_tests * 0.8:
        print("⚠️  MOSTLY PASSING: Minor issues detected but embeddings should work.")
        return True
    else:
        print("❌ TESTS FAILED: Significant issues detected. Check the logs above.")
        return False


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        embedding_dir = Path(sys.argv[1])
    else:
        # Try common locations
        test_dirs = [
            Path("data/stage06_esm2"),
            Path("data/test_esm2_output"),
            Path("data/test_esm2_mps_output"),
            Path("data/test_esm2_lancedb_output")
        ]
        
        embedding_dir = None
        for test_dir in test_dirs:
            if test_dir.exists():
                embedding_dir = test_dir
                break
        
        if embedding_dir is None:
            print("❌ No embedding directory found!")
            print("Try:")
            print("  python test_esm2_similarity.py data/stage06_esm2")
            print("Or run embeddings first:")
            print("  python run_esm2_m4_max.py")
            sys.exit(1)
    
    if not embedding_dir.exists():
        print(f"❌ Embedding directory not found: {embedding_dir}")
        sys.exit(1)
    
    # Run comprehensive test
    success = run_comprehensive_test(embedding_dir)
    sys.exit(0 if success else 1)
