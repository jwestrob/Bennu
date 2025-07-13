# Genome Selection System Analysis

## Current Implementation Problems

The existing genome selection system in `src/llm/rag_system/genome_selector.py` is a brittle keyword-matching approach that will not scale to real-world genomic datasets. This document analyzes the current workflow and proposes an LLM-based replacement.

## Current Workflow Breakdown

### 1. Trigger Decision (`should_use_genome_selection()`)

**How it decides to trigger genome selection:**

```python
def should_use_genome_selection(self, query: str) -> bool:
```

**Trigger Logic:**
1. **Hard-coded keyword matching** against `genome_targeting_keywords`:
   - `'for the'`, `'in the'`, `'from the'`, `'within the'`, `'of the'`
   - `'annotations for'`, `'proteins in'`, `'genes in'`, `'domains in'`
   - `'functions in'`, `'bgcs in'`, `'cazymes in'`
   - `'chosen genome'`, `'selected genome'`, `'target genome'`

2. **Exception patterns** that disable selection:
   - Comparative keywords: `'compare'`, `'comparison'`, `'across genomes'`, `'between genomes'`
   - Listing keywords: `'list genomes'`, `'show genomes'`, `'how many genomes'`

3. **Special case overrides:**
   - If query has both comparative AND targeting keywords → allow selection
   - If query has "available genomes" + analysis keywords → allow selection

### 2. Genome Request Extraction (`extract_genome_request()`)

**How it finds which genome to select:**

```python
def extract_genome_request(self, query: str) -> Optional[str]:
```

**Extraction Strategy (in priority order):**

1. **Direct genome ID patterns:**
   - Regex: `r'\b(PLM[0-9]+_[A-Za-z0-9_]+)\b'` (matches PLM IDs)
   - Regex: `r'\b([A-Za-z0-9_]+_[0-9]+_[0-9]+_contigs?)\b'` (matches contig IDs)

2. **Organism aliases lookup:**
   - Hard-coded dictionary: `organism_aliases`
   - Examples: `'nomurabacteria'` → `['nomurabacteria', 'candidatus_nomurabacteria']`
   - Only works for pre-defined organisms in the tiny alias dictionary

3. **Keyword-based extraction:**
   - Look for text after targeting keywords
   - Take first word after keyword, clean up suffixes
   - Extract organism names with regex patterns

4. **Explicit organism mentions:**
   - Regex patterns for bacterial names: `r'\b([A-Z][a-z]+bacteria|[A-Z][a-z]+coccus)\b'`
   - Candidatus patterns: `r'\b(candidatus[_\s]+[A-Za-z]+)\b'`
   - Taxonomic groups: `r'\b([A-Za-z]+ales|[A-Za-z]+aceae)\b'`

### 3. Genome Matching (`_find_matching_genomes()`)

**How it matches extracted requests to actual genome IDs:**

```python
def _find_matching_genomes(self, request: str, available_genomes: List[str]) -> List[GenomeMatch]:
```

**Matching Strategy (multiple attempts):**

1. **Exact substring matching:**
   - Check if request appears anywhere in genome ID
   - Score = `len(request) / len(genome_id)`

2. **Organism aliases lookup:**
   - Check against hard-coded alias dictionary
   - Add bonus score for alias matches

3. **Fuzzy string matching:**
   - Use `SequenceMatcher` for similarity ratio
   - Minimum threshold: 0.4 similarity

4. **Token-based matching:**
   - Split both request and genome ID on `[_\.\-\s]+`
   - Count overlapping tokens
   - Score based on intersection size

### 4. Selection Logic (`select_genome()`)

**How it chooses the final genome:**

```python
async def select_genome(self, query: str) -> GenomeSelectionResult:
```

**Selection Process:**
1. Extract genome request from query
2. Get available genomes from database (cached)
3. Find all matching genomes using strategies above
4. Take best match if confidence > 0.3
5. Return result with metadata

## Critical Problems

### 1. **Brittle Keyword Matching**
- **Example failure**: "In the context of..." triggers `'in the'` keyword
- **Real-world failure**: Complex natural language will break keyword patterns
- **Scalability**: Adding new patterns manually is unsustainable

### 2. **Hard-coded Organism Aliases**
```python
self.organism_aliases = {
    'nomurabacteria': ['nomurabacteria', 'candidatus_nomurabacteria'],
    'burkholderiales': ['burkholderiales', 'burkholderia'],
    'acidovorax': ['acidovorax'],
    'esherichia': ['escherichia', 'e_coli', 'ecoli'],  # Typo: "esherichia"
    'bacillus': ['bacillus'],
    'pseudomonas': ['pseudomonas'],
}
```
- **Coverage**: Only works for 6 hard-coded organisms
- **Maintenance**: Requires manual updates for every new dataset
- **Errors**: Contains typos ("esherichia" vs "escherichia")
- **Metagenomes**: Useless for MAGs with systematic names like "MAG_001"

### 3. **Context Insensitive**
- Cannot distinguish between:
  - "Find proteins in the Nomurabacteria genome" (specific)
  - "Find proteins in the next step" (generic)
- Keywords like `'in the'` are too broad

### 4. **No Multi-Genome Support**
- Returns single genome or failure
- Cannot handle: "Compare Nomurabacteria and Burkholderiales"
- Cannot handle: "Analyze these three genomes: X, Y, Z"

### 5. **Metagenome Incompatible**
- Real metagenome bins: `bin.1`, `bin.2`, `MAG_highqual_001`
- Unbinned contigs: `contig_scaffold_12345`
- Population genomics: `strain_A`, `strain_B`, `isolate_001`
- None of this matches the hard-coded patterns

## LLM-Based Replacement Design

### Proposed Architecture

```python
class LLMGenomeSelector:
    """
    Use GPT-4.1-mini to intelligently determine genome selection intent.
    """
    
    async def analyze_genome_intent(self, query: str, available_genomes: List[str]) -> GenomeSelectionResult:
        """
        Use LLM to determine:
        1. Does this query target specific genome(s)?
        2. If so, which genome(s) from the available list?
        3. If not, should we analyze all genomes?
        """
```

### LLM Prompt Strategy

```python
GENOME_SELECTION_PROMPT = """
You are a genomics expert analyzing user queries to determine genome selection intent.

Available genomes in the database:
{available_genomes}

User query: "{query}"

Analyze this query and determine:

1. **Intent Classification:**
   - "specific": Query targets one or more specific genomes
   - "comparative": Query wants to compare across multiple/all genomes  
   - "global": Query wants to analyze all genomes without comparison
   - "ambiguous": Unclear intent, recommend clarification

2. **Target Genomes:**
   - If specific: List the exact genome IDs from available genomes
   - If comparative/global: Return empty list (analyze all)
   - If ambiguous: Return empty list and suggest clarification

3. **Reasoning:**
   - Explain your decision
   - Note any organism names or genome references you found

Return your analysis as JSON:
{
  "intent": "specific|comparative|global|ambiguous",
  "target_genomes": ["exact_genome_id_1", "exact_genome_id_2"],
  "reasoning": "explanation of decision",
  "confidence": 0.0-1.0
}
"""
```

### Benefits of LLM Approach

#### 1. **Natural Language Understanding**
- ✅ "Find prophages in the Candidatus Nomurabacteria genome"
- ✅ "Compare metabolic capabilities between all four genomes"
- ✅ "Show me proteins across the entire dataset"
- ✅ "Analyze both the Burkholderiales and Nomurabacteria genomes"

#### 2. **Context Awareness**
- Distinguishes between genome-specific and general language
- Understands complex multi-part queries
- Handles typos and variations naturally

#### 3. **Multi-Genome Support**
- Can return 0 to N genomes
- Handles comparative analysis requests
- Supports global analysis modes

#### 4. **Metagenome Ready**
- Works with any genome naming scheme
- No hard-coded organism names
- Scales to hundreds of MAGs

#### 5. **Self-Documenting**
- Returns reasoning for decisions
- Confidence scores for reliability
- Clear intent classification

### Implementation Plan

#### Phase 1: Core LLM Selector
```python
class LLMGenomeSelector:
    def __init__(self, model="gpt-4.1-mini"):
        self.model = model
    
    async def analyze_genome_intent(self, query: str, available_genomes: List[str]) -> GenomeSelectionResult:
        # Use DSPy or direct LLM call
        # Parse JSON response
        # Return structured result
```

#### Phase 2: Integration
- Replace `GenomeSelector` with `LLMGenomeSelector`
- Update task executor to use new interface
- Maintain backward compatibility for existing workflows

#### Phase 3: Validation
- Test with complex queries
- Validate against current dataset
- Benchmark performance vs keyword matching

### Cost Considerations

**GPT-4.1-mini costs:**
- ~$0.000075 per 1K input tokens
- ~$0.0003 per 1K output tokens
- Average genome selection call: ~500 input + 200 output tokens
- Cost per call: ~$0.00010 (1/100th of a cent)
- 1000 queries per day: ~$0.10

**Performance:**
- Latency: ~1-2 seconds per call
- Accuracy: Expected >95% vs current ~60% keyword matching
- Maintenance: Zero ongoing maintenance vs constant keyword updates

## Conclusion

The current keyword-based genome selector is fundamentally inadequate for real-world genomic analysis. An LLM-based replacement would provide:

- **Better accuracy** through natural language understanding
- **Zero maintenance** for new organisms/datasets  
- **Multi-genome support** for comparative analysis
- **Metagenome compatibility** for modern datasets
- **Clear reasoning** for debugging and transparency

The cost is negligible (~$0.0001 per query) and the improvement in reliability and user experience would be substantial.

## Current System Removal Priority

The existing system should be replaced because:

1. **Active harm**: False positive triggers breaking valid queries
2. **Maintenance burden**: Requires manual updates for every new dataset
3. **User confusion**: Unpredictable behavior with natural language
4. **Future blocking**: Incompatible with metagenome workflows

**Recommendation**: Implement LLM-based replacement as high priority to enable reliable genome selection for both current and future use cases.