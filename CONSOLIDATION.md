# Pipeline Consolidation Plan: Direct CSV Export

**Branch**: `consolidation`  
**Target**: Eliminate RDF serialization bottleneck by generating CSV files directly from in-memory RDF graph  
**Expected Performance Gain**: 8+ minutes → ~30 seconds for knowledge graph export

---

## Current Inefficient Architecture

### **Problem: Double Serialization**
```
In-Memory RDF Graph (rdflib.Graph)
    ↓ [8+ minutes serialization]
TTL/NT File (100+ MB on disk)
    ↓ [Parse TTL back to memory]  
RDFToCSVConverter
    ↓ [Generate CSV files]
Neo4j CSV Files
    ↓ [Bulk import]
Neo4j Database
```

### **Root Cause Analysis**
- **9.5M triples** for large metagenomes (387K proteins)
- **RDF serialization** is CPU-intensive, single-threaded operation  
- **TTL files only used** by `RDFToCSVConverter` - no other consumers
- **Unnecessary disk I/O** for large intermediate files
- **Double processing** of identical data structures

---

## Proposed Efficient Architecture

### **Solution: Direct CSV Export**
```
In-Memory RDF Graph (rdflib.Graph)
    ↓ [Direct traversal, ~30 seconds]
Neo4j CSV Files
    ↓ [Bulk import]
Neo4j Database
```

### **Performance Benefits**
- **Time**: 8+ minutes → 30 seconds (16x speedup)
- **Disk Space**: Eliminate 100+ MB intermediate files
- **Memory**: No serialize/deserialize memory pressure
- **Simplicity**: Fewer pipeline steps, cleaner architecture

---

## Implementation Plan

### **Phase 1: Core Infrastructure (High Priority)**

#### **1.1: Create DirectCSVExporter Class**
**File**: `src/build_kg/direct_csv_exporter.py`

```python
class DirectCSVExporter:
    """Generate Neo4j CSV files directly from rdflib.Graph without serialization."""
    
    def __init__(self, graph: rdflib.Graph, output_dir: Path):
        self.graph = graph
        self.output_dir = output_dir
    
    def export_all(self) -> Dict[str, Any]:
        """Export all CSV files needed for Neo4j bulk import."""
        # Node CSVs
        self.export_genomes()
        self.export_proteins() 
        self.export_pfam_domains()
        self.export_kegg_functions()
        self.export_pathways()
        self.export_bgc_clusters()
        
        # Relationship CSVs
        self.export_protein_domain_relationships()
        self.export_protein_function_relationships()
        self.export_function_pathway_relationships()
        # ... etc
        
    def export_proteins(self):
        """Export protein nodes to proteins.csv"""
        with open(self.output_dir / 'proteins.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['protein_id:ID', 'sequence', 'length:int', 'genome_id', ':LABEL'])
            
            # Direct SPARQL query on in-memory graph
            for row in self.graph.query("""
                SELECT ?protein_id ?sequence ?length ?genome_id WHERE {
                    ?protein rdf:type kg:Protein .
                    ?protein kg:proteinId ?protein_id .
                    ?protein kg:sequence ?sequence .
                    ?protein kg:length ?length .
                    ?protein kg:fromGenome ?genome .
                    ?genome kg:genomeId ?genome_id .
                }
            """):
                writer.writerow([row.protein_id, row.sequence, row.length, row.genome_id, 'Protein'])
```

**Key Features**:
- Direct SPARQL queries on in-memory graph
- Streaming CSV writing (low memory footprint)
- Progress tracking for large datasets
- Full compatibility with Neo4j bulk import format

#### **1.2: Modify RDF Builder Integration**
**File**: `src/build_kg/rdf_builder.py`

**Add method**:
```python
def export_to_csv_direct(self, output_dir: Path) -> Dict[str, Any]:
    """Export RDF graph directly to CSV files without serialization."""
    from .direct_csv_exporter import DirectCSVExporter
    
    csv_output_dir = output_dir / "csv"
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    
    exporter = DirectCSVExporter(self.graph, csv_output_dir)
    return exporter.export_all()
```

**Modify main build functions**:
```python
# In build_knowledge_graph_with_extended_annotations()
if direct_csv_export:
    logger.info("Exporting directly to CSV files...")
    csv_stats = builder.export_to_csv_direct(output_dir)
    logger.info(f"Generated CSV files in {csv_stats['export_time']:.1f} seconds")
else:
    # Existing RDF serialization path
    save_stats = builder.save_graph(kg_file, format='nt')
```

#### **1.3: Add CLI Option**
**File**: `src/cli.py`

**Add parameter**:
```python
direct_csv: bool = typer.Option(
    False,
    "--direct-csv",
    help="Skip RDF serialization, export directly to CSV for faster processing"
)
```

**Update Stage 7 call**:
```python
7: {
    "name": "Knowledge Graph Construction",
    "function": lambda: build_knowledge_graph_with_extended_annotations(
        # ... existing parameters ...
        direct_csv_export=direct_csv
    )
}
```

### **Phase 2: CSV Format Compatibility (Medium Priority)**

#### **2.1: Analyze Existing CSV Schema**
**Action**: Examine `src/build_kg/rdf_to_csv_converter.py` output format

**Files to analyze**:
- Node CSV headers and data format
- Relationship CSV headers and data format  
- Neo4j property types and constraints
- Header naming conventions (`:ID`, `:LABEL`, etc.)

#### **2.2: Ensure Format Compatibility**
**Goal**: DirectCSVExporter output must be identical to RDFToCSVConverter output

**Testing Strategy**:
```python
def test_csv_format_compatibility():
    """Ensure DirectCSVExporter generates identical CSV to RDFToCSVConverter."""
    # Generate CSVs using both methods
    # Compare file-by-file for identical content
    # Validate Neo4j bulk import works identically
```

### **Phase 3: Performance Optimization (Medium Priority)**

#### **3.1: Parallel CSV Generation**
**Enhancement**: Generate multiple CSV files simultaneously

```python
def export_all_parallel(self):
    """Export CSV files in parallel using ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(self.export_proteins),
            executor.submit(self.export_pfam_domains),  
            executor.submit(self.export_kegg_functions),
            executor.submit(self.export_pathways)
        ]
        wait(futures)
```

#### **3.2: Memory-Efficient SPARQL**
**Optimization**: Use streaming SPARQL results for large queries

```python
def export_proteins_streaming(self):
    """Export proteins using streaming SPARQL to handle large datasets."""
    query = self.graph.query("""...""")  # Large result set
    
    with open(self.output_dir / 'proteins.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        batch_size = 10000
        batch = []
        for row in query:  # Stream results
            batch.append(row)
            if len(batch) >= batch_size:
                self._write_batch(writer, batch)
                batch.clear()
        
        if batch:  # Write remaining
            self._write_batch(writer, batch)
```

### **Phase 4: Integration & Validation (High Priority)**

#### **4.1: Update Pipeline Configuration**
**File**: `config/pipeline.yaml`

```yaml
knowledge_graph:
  direct_csv_export: true              # Skip RDF serialization  
  preserve_rdf_files: false            # Don't save TTL files
  csv_parallel_workers: 4              # Parallel CSV generation
  rdf_serialization_format: "nt"      # Fallback format if needed
```

#### **4.2: Backward Compatibility**
**Strategy**: Keep existing RDF path as fallback

```python
def build_knowledge_graph_with_extended_annotations(
    # ... existing params ...
    direct_csv_export: bool = True,     # New default
    preserve_rdf_files: bool = False    # Skip TTL generation
):
    if direct_csv_export:
        # New fast path
        csv_stats = builder.export_to_csv_direct(output_dir)
        
        if preserve_rdf_files:
            # Still generate RDF for debugging
            rdf_stats = builder.save_graph(kg_file, format='nt')
    else:
        # Legacy path: RDF → CSV conversion
        rdf_stats = builder.save_graph(kg_file, format='nt')
        csv_converter = RDFToCSVConverter(kg_file, output_dir / "csv")
        csv_stats = csv_converter.convert()
```

#### **4.3: Testing Strategy**

**Unit Tests**:
```python
def test_direct_csv_export():
    """Test DirectCSVExporter generates valid CSV files."""

def test_csv_neo4j_compatibility():
    """Test CSV files work with Neo4j bulk import."""

def test_performance_improvement():
    """Verify direct export is faster than RDF→CSV path."""
```

**Integration Tests**:
- End-to-end pipeline with `--direct-csv` flag
- Neo4j database loading with generated CSVs  
- Query validation: ensure identical results vs RDF path

---

## Files to Modify

### **New Files**
1. `src/build_kg/direct_csv_exporter.py` - Core CSV export functionality
2. `tests/test_direct_csv_export.py` - Comprehensive test suite

### **Modified Files**
1. `src/build_kg/rdf_builder.py` - Add direct CSV export method
2. `src/cli.py` - Add `--direct-csv` CLI option  
3. `config/pipeline.yaml` - Update default configuration
4. `CLAUDE.md` - Document new optimization

### **Files to Analyze** (No Changes)
1. `src/build_kg/rdf_to_csv_converter.py` - Reference implementation
2. `src/build_kg/neo4j_bulk_loader.py` - Ensure compatibility

---

## Migration Strategy

### **Development Approach**
1. **Branch-based development** on `consolidation` branch
2. **Incremental implementation** - each phase can be tested independently
3. **Feature flag approach** - `--direct-csv` allows A/B testing
4. **Backward compatibility** - existing RDF path preserved

### **Validation Plan**
1. **Performance benchmarking** - measure actual speedup on large datasets
2. **Data integrity** - ensure CSV outputs are identical to RDF→CSV path  
3. **Neo4j compatibility** - verify bulk import works identically
4. **Regression testing** - ensure no functionality is lost

### **Rollout Strategy**
1. **Phase 1**: Implement and test on development datasets
2. **Phase 2**: Validate with production-size metagenomes (387K proteins)  
3. **Phase 3**: Make direct CSV export the default
4. **Phase 4**: Deprecate RDF serialization (keep as optional debug feature)

---

## Expected Outcomes

### **Performance Improvements**
- **Stage 7 runtime**: 8+ minutes → ~30 seconds (16x faster)
- **Disk usage**: Eliminate 100+ MB intermediate TTL files
- **Memory efficiency**: No serialize/parse memory pressure
- **Pipeline reliability**: Fewer complex operations, less likely to fail

### **Code Quality Improvements**  
- **Simplified architecture**: Fewer intermediate steps
- **Better maintainability**: Direct data flow is easier to debug
- **Reduced complexity**: Eliminate RDF serialization edge cases
- **Modern design**: Direct data processing instead of file-based roundtrips

### **User Experience Improvements**
- **Faster pipeline completion** for large metagenomes  
- **Clearer progress reporting** with streaming CSV export
- **Lower disk space requirements** for processing
- **More reliable processing** with fewer failure points

---

## Risk Assessment & Mitigation

### **Risks**
1. **CSV format incompatibility** → Mitigation: Comprehensive format validation tests
2. **Performance assumptions wrong** → Mitigation: Benchmark against realistic datasets  
3. **Neo4j import issues** → Mitigation: Validate with identical test datasets
4. **Regression in functionality** → Mitigation: Keep RDF path as fallback

### **Rollback Plan**  
If direct CSV export causes issues:
1. **Immediate**: Use `--direct-csv=false` to revert to RDF path
2. **Short-term**: Fix issues while maintaining RDF fallback  
3. **Long-term**: Only remove RDF path after extensive validation

---

## Success Criteria

### **Performance Targets**
- [ ] **≥10x speedup** in Stage 7 execution time
- [ ] **≥90% reduction** in temporary disk usage  
- [ ] **Zero degradation** in Neo4j database quality
- [ ] **100% compatibility** with existing Neo4j bulk loading

### **Quality Targets**
- [ ] **Identical CSV output** compared to RDF→CSV path
- [ ] **All tests pass** for both direct and RDF paths
- [ ] **Zero data loss** in knowledge graph construction
- [ ] **Clean code architecture** with proper separation of concerns

---

**This consolidation plan eliminates a major pipeline bottleneck while maintaining full backward compatibility and data integrity.**