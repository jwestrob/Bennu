# Whole Genome Reader Hierarchical Analysis Plan

## 🚨 **CRITICAL ARCHITECTURAL ISSUE**

**Problem**: The current system does brute-force context stuffing - dumping 4MB+ of raw genomic data into every synthesis call instead of intelligent hierarchical analysis.

**Result**: LLM gets overwhelmed with raw data dumps and cannot identify specific interesting loci, leading to generic responses that ignore user requests for "three specific loci with detailed information."

## 🎯 **ORIGINAL INTENT vs CURRENT REALITY**

### **Original Vision (Hierarchical Analysis):**
```
whole_genome_reader → Raw 4MB genomic data
    ↓
Sub-Agent 1: Chunk 1 (genes 1-1000) → "Hypothetical cluster: contig_97:12580-24990 (18 genes)"
Sub-Agent 2: Chunk 2 (genes 1001-2000) → "Prophage candidate: contig_305:3100-9870 (14 genes)"  
Sub-Agent 3: Chunk 3 (genes 2001-3000) → "No significant clusters found"
    ↓
Higher-Level Agent: "Rank and analyze top 3 loci" → Detailed biological interpretation
    ↓ 
Final Synthesis: Gets CURATED findings about specific loci, not raw dumps
```

### **Current Broken Reality (Context Stuffing):**
```
whole_genome_reader → DUMP 4MB raw data → Cache → Load ALL 4MB → Compress → Send to LLM
```

## 📋 **IMPLEMENTATION PLAN**

### **Phase 1: Chunked Sub-Agent Analysis System**

#### **1.1 Create GenomicChunkAnalyzer**
- **Location**: `src/llm/rag_system/genomic_chunk_analyzer.py`
- **Purpose**: Analyze chunks of genomic data to identify regions of interest
- **Input**: Subset of `genome_contexts` data (e.g., 500-1000 genes)
- **Output**: List of interesting loci with coordinates, gene counts, biological significance

#### **1.2 Implement Intelligent Chunking Strategy**
- **Biological chunking**: Group by contigs, not arbitrary gene counts
- **Size management**: Each chunk ~50K tokens (analyzable by single LLM call)
- **Overlap handling**: Ensure operon boundaries aren't split across chunks

#### **1.3 Create LociPrioritizer Agent**
- **Location**: `src/llm/rag_system/loci_prioritizer.py`
- **Purpose**: Rank and select top N loci from sub-agent findings
- **Input**: Candidate loci from all sub-agents
- **Output**: Prioritized list with biological justification

### **Phase 2: Modify Whole Genome Reader Integration**

#### **2.1 Update AgentExecutor Integration**
- **Change**: `_execute_whole_genome_reader()` should NOT store full data in notes
- **Instead**: Store only summary stats + trigger hierarchical analysis
- **Cache**: Full genomic data remains in tool cache for sub-agent access

#### **2.2 Create Hierarchical Analysis Orchestrator**
- **Workflow**:
  1. `whole_genome_reader` collects data → Cache only
  2. Orchestrator splits into chunks → Sub-agent analysis
  3. Sub-agents identify interesting loci → Store specific findings
  4. Prioritizer ranks loci → Store top candidates
  5. Detail analyzer examines top loci → Rich biological context

### **Phase 3: Note Storage Redesign**

#### **3.1 Structured Loci Notes**
Instead of storing massive raw data, notes should contain:
```json
{
  "key_findings": [
    "Locus 1: contig_97:12580-24990 - 18 hypothetical proteins, integrase flanking",
    "Locus 2: contig_305:3100-9870 - 14 genes, 70% hypothetical, tRNA integration site",
    "Locus 3: contig_88:42200-57600 - 22 genes, novel DUF domains, lytic modules"
  ],
  "quantitative_data": {
    "loci_analyzed": 3,
    "total_candidates_screened": 38,
    "criteria": "≥5 consecutive hypothetical proteins + integration signatures"
  }
}
```

#### **3.2 Reference-Based Detail Storage**
- **Loci details**: Store in cache with IDs like `locus_contig97_12580_24990`
- **Notes reference**: `"locus_detail_refs": ["locus_contig97_12580_24990", ...]`
- **Synthesis loads**: Only the top 3 loci details, not entire genome

### **Phase 4: Synthesis Enhancement**

#### **4.1 Loci-Focused Synthesis**
- **Input**: Curated loci findings + detailed analysis of top candidates
- **Context**: ~10K tokens of highly relevant loci data vs 4MB raw dump
- **Output**: Detailed report on specific loci as requested by user

#### **4.2 Biological Context Enrichment**
- **Gene annotation lookup**: For genes in prioritized loci
- **Comparative analysis**: Cross-loci pattern detection
- **Integration site analysis**: tRNA, direct repeats, GC content shifts

## 🛠️ **TECHNICAL SPECIFICATIONS**

### **New Components to Create:**

1. **`GenomicChunkAnalyzer`**
   - **Method**: `analyze_genomic_chunk(chunk_data, analysis_criteria)`
   - **DSPy Signature**: `GenomicRegionIdentifier`
   - **Output**: List of `InterestingLocus` objects

2. **`LociPrioritizer`**
   - **Method**: `prioritize_loci(candidate_loci, max_count=3)`
   - **DSPy Signature**: `LociRanker`
   - **Output**: Ranked list with biological justification

3. **`HierarchicalGenomeAnalyzer`** (Orchestrator)
   - **Method**: `analyze_genome_hierarchically(genome_contexts, question)`
   - **Workflow**: Chunk → Analyze → Prioritize → Detail → Synthesize

### **Data Structures:**

```python
@dataclass
class InterestingLocus:
    contig_id: str
    start: int
    end: int
    gene_count: int
    hypothetical_count: int
    significance_score: float
    biological_features: List[str]
    flanking_genes: List[str]
    
@dataclass
class LocusAnalysis:
    locus: InterestingLocus
    detailed_genes: List[GeneContext]
    functional_predictions: List[str]
    comparative_context: str
    novelty_assessment: str
```

## ⚡ **EXPECTED PERFORMANCE IMPROVEMENTS**

### **Before (Current Broken System):**
- **Context size**: 8.6M+ tokens per synthesis
- **Synthesis quality**: Generic, ignores specific loci requests
- **Processing time**: 8+ minutes with compression
- **User satisfaction**: Low - doesn't answer the actual question

### **After (Hierarchical System):**
- **Context size**: ~10K tokens of curated loci analysis
- **Synthesis quality**: Detailed reports on specific requested loci
- **Processing time**: ~2 minutes with intelligent analysis
- **User satisfaction**: High - provides exactly what was requested

## 🎯 **SUCCESS CRITERIA**

1. **Specific Loci Identification**: System identifies exact genomic coordinates of interesting regions
2. **Detailed Biological Analysis**: Rich annotation and context for each prioritized locus
3. **User Request Fulfillment**: Provides exactly what user asks for (e.g., "three loci with detailed information")
4. **Efficient Processing**: No more massive context dumps, only relevant curated data
5. **Scalable Architecture**: Can handle larger genomes without context explosion

## 📅 **IMPLEMENTATION PHASES**

- **Phase 1**: Create chunking and sub-agent analysis components *(3-4 hours)*
- **Phase 2**: Integrate with existing AgentExecutor and note system *(2-3 hours)*
- **Phase 3**: Redesign note storage for loci-focused capture *(1-2 hours)*
- **Phase 4**: Enhance synthesis for curated loci reporting *(1 hour)*

**Total Estimated Time**: 7-10 hours of development

## 🚨 **PRIORITY**: This is a critical architectural fix that transforms the system from a broken context-dumping approach to an intelligent hierarchical analysis system that actually answers user questions correctly.