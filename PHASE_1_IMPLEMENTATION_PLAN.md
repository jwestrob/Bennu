# Phase 1 Detailed Implementation Plan

## 1. Database Integration Tasks

### 1.1 AntiSMASH Integration

#### Task: Create AntiSMASH Parser
**File**: `src/ingest/05_antismash.py`
```python
# Parse AntiSMASH GenBank output files
# Extract: BGC type, boundaries, genes, products
# Create relationships: (Gene)-[:PART_OF_BGC]->(BGC)
# Add properties: bgc_type, product_class, detection_rules
```

**Specific Implementation**:
- [ ] Parse `.gbk` files from antiSMASH output
- [ ] Extract cluster boundaries (start, end coordinates)
- [ ] Map genes within clusters to existing Gene nodes
- [ ] Create BGC nodes with properties: type, product, confidence_score
- [ ] Add relationships to substrate predictions if available

#### Task: Update RDF Builder
**File**: `src/build_kg/rdf_builder.py`
- [ ] Add BGC node type to schema
- [ ] Create `add_bgc_annotations()` method
- [ ] Define relationships: PART_OF_BGC, PRODUCES_METABOLITE
- [ ] Add BGC properties: cluster_type, predicted_products, mibig_similarity

### 1.2 dbCAN CAZyme Integration

#### Task: Create dbCAN Parser
**File**: `src/ingest/05_dbcan.py`
```python
# Parse dbCAN HMMER output
# Extract: CAZy family, E-value, domain boundaries
# Create relationships: (Protein)-[:HAS_CAZYME]->(CAZymeFamily)
```

**Specific Implementation**:
- [ ] Parse dbCAN `overview.txt` output format
- [ ] Create CAZymeFamily nodes (GH, GT, PL, CE, AA families)
- [ ] Map to existing proteins using protein IDs
- [ ] Store domain boundaries for sub-protein localization
- [ ] Add substrate predictions from CAZy database

#### Task: Extend Functional Enrichment
**File**: `src/build_kg/functional_enrichment.py`
- [ ] Add `parse_cazy_families()` method
- [ ] Download and parse CAZy family descriptions
- [ ] Link to known substrates and reaction types

### 1.3 VOG/PHROG Phage Annotations

#### Task: Create Phage Gene Parser
**File**: `src/ingest/05_phage_annotations.py`
```python
# Parse VOG/PHROG HMM search results
# Extract: Viral ortholog groups, functional categories
# Create relationships: (Protein)-[:VIRAL_ORTHOLOG]->(VOG)
```

**Specific Implementation**:
- [ ] Parse HMMER3 output for VOG/PHROG databases
- [ ] Create ViralOrtholog nodes with functional categories
- [ ] Add properties: host_range, lifestyle_prediction
- [ ] Link to prophage regions if detected
- [ ] Flag auxiliary metabolic genes (AMGs)

### 1.4 Unified Schema Extensions

#### Task: Update Neo4j Schema
**File**: `src/build_kg/schema.py`
- [ ] Add node types: BGC, CAZymeFamily, ViralOrtholog
- [ ] Define new relationships:
  - `(Gene)-[:PART_OF_BGC]->(BGC)`
  - `(Protein)-[:HAS_CAZYME]->(CAZymeFamily)`
  - `(Protein)-[:VIRAL_ORTHOLOG]->(VOG)`
  - `(BGC)-[:PRODUCES]->(Metabolite)`
- [ ] Add indexes for new node types

#### Task: Create Cross-Database Linker
**File**: `src/build_kg/cross_database_links.py`
- [ ] Link BGCs that produce CAZymes
- [ ] Find phage-encoded CAZymes (AMGs)
- [ ] Connect metabolic pathways to BGC products
- [ ] Create co-occurrence relationships

## 2. Performance Optimization Tasks

### 2.1 Semantic Cache Implementation

#### Task: Build Semantic Cache
**File**: `src/llm/semantic_cache.py`
```python
class SemanticCache:
    def __init__(self, similarity_threshold=0.95):
        self.cache_db = "data/cache/semantic_cache.lance"
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
```

**Specific Implementation**:
- [ ] Create LanceDB cache for query embeddings
- [ ] Store: query, embedding, result, timestamp
- [ ] Implement `check_cache()` method with similarity search
- [ ] Add cache invalidation for data updates
- [ ] Add metrics: hit rate, response time savings

#### Task: Integrate Cache into RAG
**File**: `src/llm/rag_system.py`
- [ ] Modify `ask()` method to check cache first
- [ ] Add cache warming for common queries
- [ ] Implement cache management commands

### 2.2 Query Router Implementation

#### Task: Build Pattern-Based Router
**File**: `src/llm/query_router.py`
```python
class QueryRouter:
    def __init__(self):
        self.patterns = {
            'simple_lookup': [...],
            'count_query': [...],
            'pathway_analysis': [...],
            'complex_reasoning': [...]
        }
```

**Specific Implementation**:
- [ ] Define regex patterns for each query type
- [ ] Create direct Neo4j query templates
- [ ] Implement `route()` method returning handler type
- [ ] Add `execute_simple_query()` for non-LLM queries
- [ ] Log routing decisions for analysis

### 2.3 Context Compression

#### Task: Smart Context Compressor
**File**: `src/llm/context_compression.py`
```python
class ContextCompressor:
    def compress(self, query, context_items, max_tokens=2000):
        # Rank by relevance, pack greedily
```

**Specific Implementation**:
- [ ] Implement relevance scoring (embedding + keyword)
- [ ] Add token counting with tiktoken
- [ ] Create summarization for overflow items
- [ ] Preserve essential properties per item type
- [ ] Add compression metrics

### 2.4 Batch Processing System

#### Task: Batch Query Processor
**File**: `src/build_kg/batch_processor.py`
```python
class BatchProcessor:
    def __init__(self, batch_size=100, max_workers=4):
        self.executor = ProcessPoolExecutor(max_workers)
```

**Specific Implementation**:
- [ ] Create queue system for large analyses
- [ ] Implement parallel Neo4j queries
- [ ] Add progress tracking with rich
- [ ] Handle partial failures gracefully
- [ ] Create batch result aggregator

## 3. Core Feature: "Find Interesting Biology" Button

### 3.1 Interesting Biology Detector

#### Task: Main Detector Class
**File**: `src/analysis/interesting_biology_detector.py`
```python
class InterestingBiologyDetector:
    def __init__(self, neo4j_conn, lancedb_conn):
        self.detectors = [
            MetabolicAnomalyDetector(),
            ViralElementDetector(),
            NovelFamilyDetector(),
            SymbiosisDetector(),
            HGTDetector()
        ]
```

**Specific Implementation**:
- [ ] Create abstract `BiologyDetector` base class
- [ ] Implement `detect_all()` method
- [ ] Score findings by importance/novelty
- [ ] Generate natural language explanations
- [ ] Create visualization for each finding type

### 3.2 Specific Detectors

#### Task: Metabolic Anomaly Detector
**File**: `src/analysis/detectors/metabolic_anomaly.py`
- [ ] Find incomplete pathways with unusual completion patterns
- [ ] Detect pathways in unexpected organisms
- [ ] Identify redundant/backup pathways
- [ ] Score by phylogenetic unusualness

#### Task: Novel Family Detector
**File**: `src/analysis/detectors/novel_family.py`
- [ ] Use HDBSCAN on embeddings to find clusters
- [ ] Identify clusters with no functional annotation
- [ ] Check genomic context for function hints
- [ ] Calculate cluster coherence score

#### Task: Symbiosis Detector
**File**: `src/analysis/detectors/symbiosis.py`
- [ ] Find complementary metabolic pathways
- [ ] Detect co-occurring nutrient dependencies
- [ ] Identify potential metabolite exchanges
- [ ] Score by pathway completion reciprocity

### 3.3 Report Generation

#### Task: Automated Report Builder
**File**: `src/reports/interesting_biology_report.py`
```python
class InterestingBiologyReport:
    def __init__(self, findings):
        self.sections = {
            'executive_summary': None,
            'key_findings': [],
            'detailed_analysis': {},
            'figures': []
        }
```

**Specific Implementation**:
- [ ] Create markdown report template
- [ ] Add figure generation for each finding type
- [ ] Implement importance-based ordering
- [ ] Generate testable hypotheses section
- [ ] Add methods section automatically

### 3.4 Visualization Components

#### Task: Finding Visualizers
**File**: `src/visualization/biology_visualizers.py`
- [ ] Create pathway incompleteness diagram
- [ ] Build viral element integration map
- [ ] Design protein family clustering plot
- [ ] Implement metabolic exchange network
- [ ] Add interactive HTML output option

## 4. Integration Points

### Task: Update CLI
**File**: `src/cli.py`
- [ ] Add `find-interesting` command
- [ ] Add `--cache` flag for cache control
- [ ] Add `--batch` flag for batch processing
- [ ] Add `--output-format` for reports

### Task: Update Main RAG System
**File**: `src/llm/rag_system.py`
- [ ] Integrate query router
- [ ] Add semantic cache checking
- [ ] Use context compression
- [ ] Add batch mode support

### Task: Create Integration Tests
**File**: `src/tests/test_integration/test_phase1_features.py`
- [ ] Test AntiSMASH → Neo4j pipeline
- [ ] Test cache hit/miss scenarios
- [ ] Test batch processing with 100+ queries
- [ ] Test interesting biology detection

## 5. Data Pipeline Updates

### Task: Update Pipeline Runner
**File**: `src/pipeline/pipeline_runner.py`
- [ ] Add stage 5a: AntiSMASH annotation
- [ ] Add stage 5b: dbCAN annotation
- [ ] Add stage 5c: Phage annotation
- [ ] Add stage 7: Interesting biology detection

### Task: Create Download Scripts
**File**: `scripts/download_databases.py`
- [ ] Download dbCAN HMMs
- [ ] Download VOG/PHROG HMMs
- [ ] Download CAZy family descriptions
- [ ] Create version tracking

## Implementation Order

1. **Week 1, Days 1-2**: Database Integration
   - Start with parsers (can work in parallel)
   - Update schema once parsers work
   - Test with small dataset

2. **Week 1, Days 3-4**: Performance Optimizations
   - Implement semantic cache first (biggest impact)
   - Add query router
   - Test performance improvements

3. **Week 1, Days 5-7**: Interesting Biology Detector
   - Start with metabolic anomalies (easiest)
   - Add novel family detection
   - Create basic report generation

## Success Criteria

- [ ] Can process metagenome with all annotations in <10 minutes
- [ ] Cache achieves >50% hit rate on common queries
- [ ] Finds at least 5 interesting patterns per metagenome
- [ ] Generates readable report with figures
- [ ] All tests passing

## Notes for Claude Code

- Use existing patterns from `src/ingest/04_astra_scan.py` for new parsers
- Follow the established logging patterns with rich
- Add progress bars for long-running operations
- Ensure all new nodes/relationships are indexed
- Test each component in isolation before integration
