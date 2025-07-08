# Basic Query Tutorial

Learn how to ask questions about your genomic data using natural language queries.

## Getting Started

### Prerequisites

1. **Environment Setup**:
```bash
source /Users/jacob/.pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh && conda activate genome-kg
```

2. **Verify System Health**:
```bash
python -c "
from src.llm.config import LLMConfig
from src.llm.rag_system import GenomicRAG

config = LLMConfig()
rag = GenomicRAG(config)
health = rag.health_check()
print('System status:', health)
rag.close()
"
```

Expected output:
```
System status: {'neo4j': True, 'lancedb': True, 'hybrid': True, 'dspy': True}
```

## Basic Query Types

### 1. Counting and Statistics

**Simple counts** are perfect for understanding your dataset:

<details>
<summary><strong>Example: Protein Count (Click to expand)</strong></summary>

**Query**:
```bash
python -m src.cli ask "How many proteins are in the database?"
```

**Output**:
```
🧬 Processing question: How many proteins are in the database?
🤖 Agentic planning: False
💭 Planning reasoning: The user is asking for a single count of proteins—this is 
a straightforward database lookup that does not require multi-step reasoning.
📋 Using traditional query path
📊 Query type: structural
🔍 Search strategy: direct_query

🤖 Answer:
The database currently contains 10,102 distinct protein records.

Confidence: high
```

**What Happened**:
- **Query Classification**: `structural` (direct database lookup)
- **Execution Path**: `traditional` (single-step query)
- **Database Query**: Simple COUNT query to Neo4j
- **Response Time**: ~1-2 seconds

</details>

<details>
<summary><strong>Example: KEGG Ortholog Statistics (Click to expand)</strong></summary>

**Query**:
```bash
python -m src.cli ask "How many KEGG orthologs are annotated?"
```

**Output**:
```
🤖 Answer:
A total of 2,609 distinct KEGG orthologs (KO identifiers) are annotated in the current dataset.

Confidence: high
Sources: Results 1-1
```

**Related Queries You Can Try**:
- "How many genomes are in the dataset?"
- "What's the average number of proteins per genome?"
- "How many PFAM domains are annotated?"

</details>

### 2. Functional Searches

**Function-based queries** help you find proteins by their biological roles:

<details>
<summary><strong>Example: Transport Proteins (Click to expand)</strong></summary>

**Query**:
```bash
python -m src.cli ask "What transport proteins are present in the database?"
```

**Output Preview**:
```
🧬 Processing question: What transport proteins are present in the database?
📊 Query type: functional
🔍 Search strategy: direct_query

🤖 Answer:
The records you supplied already reveal a broad repertoire of membrane transporters 
in the genome collection. From the annotated KEGG Orthology (KO) terms and PFAM 
domains we can sort them into a few major functional families:

1. P–type cation-transporting ATPases  
   • Ca²⁺ P-type ATPase, type I (K01537)
   • Cu²⁺-transporting P-type ATPase (K01533)

2. ABC transporters
   • ABC transporter ATP-binding protein (K02003)
   • Heme transporter, periplasmic binding protein (K07224)
   
[...detailed analysis continues...]

Confidence: medium
Total results: 635
Query time: 0.13 seconds
```

**What Makes This Powerful**:
- **Intelligent Classification**: System recognizes "transport" as functional category
- **Cross-Database Integration**: Combines PFAM domains + KEGG orthologs
- **Biological Interpretation**: Groups results by transport mechanism
- **Quantitative Results**: 635 transport proteins identified

</details>

<details>
<summary><strong>Example: Metabolic Enzymes (Click to expand)</strong></summary>

**Query**:
```bash
python -m src.cli ask "Show me central metabolism proteins"
```

**Expected Result Type**:
- Glycolysis enzymes (K00001, K00002, etc.)
- TCA cycle components (K01902, K01903, etc.)  
- Pentose phosphate pathway enzymes
- Biological context and pathway completion analysis

**Related Functional Queries**:
- "What DNA repair proteins are present?"
- "Find cell wall biosynthesis enzymes"
- "Show me stress response proteins"

</details>

### 3. Structural Queries

**Structural queries** ask about specific entities, locations, or identifiers:

<details>
<summary><strong>Example: Specific Protein Function (Click to expand)</strong></summary>

**Query**:
```bash
python -m src.cli ask "What is the function of KEGG ortholog K20469?"
```

**Output Type**:
```
🤖 Answer:
KEGG ortholog K20469 corresponds to [specific function with detailed description].
This enzyme participates in [metabolic pathway] and is characterized by [key features].

In the current dataset, K20469 is found in [X] proteins across [Y] genomes, 
suggesting [biological interpretation].

Confidence: high
```

**Key Features**:
- **Direct Lookup**: Specific identifier queried directly
- **Functional Context**: Pathway and biological role explained
- **Dataset Context**: How this function appears in your specific genomes

</details>

<details>
<summary><strong>Example: Gene Location Queries (Click to expand)</strong></summary>

**Queries You Can Try**:
```bash
# Specific gene coordinates
python -m src.cli ask "Where is gene RIFCSPHIGHO2_01_scaffold_10_364 located?"

# Genomic neighborhoods
python -m src.cli ask "What genes surround protein PLM0_60_b1_sep16_scaffold_5_892?"

# Operon analysis
python -m src.cli ask "What genes are in the same operon as succinate dehydrogenase?"
```

**Expected Information**:
- Genomic coordinates (start, end, strand)
- Neighboring genes within 5kb
- Functional annotation of neighbors
- Operon prediction based on proximity and strand

</details>

### 4. Similarity Searches

**Semantic queries** leverage protein embeddings for similarity-based discovery:

<details>
<summary><strong>Example: Finding Similar Proteins (Click to expand)</strong></summary>

**Query**:
```bash
python -m src.cli ask "Find proteins similar to heme transporters"
```

**How This Works**:
1. **Stage 1**: System finds annotated heme transporters in Neo4j database
2. **Stage 2**: Uses those proteins as seeds for LanceDB similarity search
3. **Stage 3**: Returns proteins with high ESM2 embedding similarity (>0.7)
4. **Stage 4**: Provides biological interpretation of similarity scores

**Output Preview**:
```
🤖 Answer:
Multi-stage similarity analysis identified proteins with structural similarity to 
known heme transport systems:

**High Similarity (>0.9)**:
- protein:PLM0_scaffold_14_362: ESM2 similarity 0.934
  Function: ABC transporter periplasmic binding protein
  Genomic context: Part of 3-gene ABC transporter operon
  
**Medium Similarity (0.7-0.9)**:
- protein:RIFCS_scaffold_7_220: ESM2 similarity 0.823
  Function: Iron complex transport system substrate-binding protein
  Note: Functional convergence - different metal but similar transport mechanism

Confidence: high (based on embedding similarity + functional annotation)
```

</details>

### 5. Comparative Queries

**Comparative analysis** examines differences and similarities across genomes:

<details>
<summary><strong>Example: CAZyme Distribution (Click to expand)</strong></summary>

**Query**:
```bash
python -m src.cli ask "Show me the distribution of CAZyme types among each genome in the dataset; compare and contrast."
```

**Real Output** (from your successful test):
```
🤖 Answer:
Overview
Carbohydrate-active enzymes (CAZymes) fall into several functional classes: 
glycoside hydrolases (GH), glycosyltransferases (GT), carbohydrate-binding 
modules (CBM), carbohydrate esterases (CE), polysaccharide lyases (PL) 
and auxiliary activities (AA).

Per-genome CAZyme profiles
1. Burkholderiales_bacterium: 1,056 CAZymes
   • GH 436 (41.3%), GT 356 (33.7%), CBM 163 (15.4%), AA 56 (5.3%)
   • Interpretation: GH-rich toolkit suggests specialization for aggressive 
     breakdown of diverse plant-derived polysaccharides

2. PLM0_60_b1_sep16: 425 CAZymes  
   • GT 178 (41.9%), GH 156 (36.7%), CBM 53 (12.5%)
   • Interpretation: GT emphasis indicates investment in cell-wall/EPS biosynthesis

[...continues with detailed comparative analysis...]

Key contrasts
• Repertoire size spans >10-fold (100 vs 1,056 CAZymes)
• Functional partitioning suggests trophic hierarchy

Confidence: high
```

**Why This Query Is Sophisticated**:
- **Agentic Planning**: Recognized as multi-step analysis requirement
- **Cross-Genome Analysis**: Compares all 4 genomes systematically  
- **Quantitative Analysis**: Precise percentages and statistical comparisons
- **Biological Interpretation**: Ecological implications and trophic relationships

</details>

## Query Best Practices

### 1. Start Simple

<details>
<summary><strong>Query Progression Strategy (Click to expand)</strong></summary>

**Beginner Queries** (single concept):
```bash
"How many proteins are there?"
"What is protein X?"
"Show me transport proteins"
```

**Intermediate Queries** (two concepts):
```bash
"Compare metabolic proteins between genomes"
"Find proteins similar to transporters"
"What genes are near protein Y?"
```

**Advanced Queries** (multi-step analysis):
```bash
"Show CAZyme distribution and compare genomes"
"Find transport proteins, analyze their neighborhoods, and compare across genomes"
"Identify unique metabolic capabilities in each organism"
```

</details>

### 2. Use Specific Terms

<details>
<summary><strong>Effective Query Phrasing (Click to expand)</strong></summary>

**Good Examples**:
- ✅ "Show me glycoside hydrolases"
- ✅ "Find ABC transporter proteins"  
- ✅ "What genes are in the TCA cycle?"

**Less Effective**:
- ❌ "Show me enzymes" (too broad)
- ❌ "Find stuff related to metabolism" (vague)
- ❌ "What's this protein?" (missing identifier)

**Domain-Specific Terms That Work Well**:
- **Functional**: transport, metabolism, biosynthesis, degradation
- **Structural**: ABC transporter, P-type ATPase, glycoside hydrolase
- **Pathways**: TCA cycle, glycolysis, pentose phosphate pathway
- **Locations**: periplasmic, cytoplasmic, membrane, secreted

</details>

### 3. Leverage System Intelligence

<details>
<summary><strong>System Capabilities (Click to expand)</strong></summary>

**The system automatically handles**:
- **Synonym Recognition**: "transport proteins" = "transporters" = "permeases"
- **Abbreviation Expansion**: "TCA" → "tricarboxylic acid cycle"
- **Biological Context**: Understands enzyme classifications, pathway relationships
- **Cross-Database Integration**: Combines PFAM, KEGG, CAZyme, BGC data seamlessly

**You don't need to**:
- Specify database sources ("from PFAM" or "from KEGG")
- Use exact technical terms (can say "breakdown" instead of "hydrolysis")
- Format queries in any special way (natural language works)

</details>

## Understanding Output

### Confidence Levels

<details>
<summary><strong>Confidence Interpretation (Click to expand)</strong></summary>

**High Confidence** (90%+ of responses):
- Direct database lookups (counts, specific IDs)
- Well-annotated functional categories
- Clear similarity matches (>0.9 embedding similarity)

**Medium Confidence**:
- Complex comparative analysis  
- Moderate similarity matches (0.7-0.9)
- Inferences requiring biological interpretation

**Low Confidence** (rare):
- System errors or missing data
- Highly ambiguous queries
- Technical limitations

</details>

### Response Structure

<details>
<summary><strong>Response Components (Click to expand)</strong></summary>

**Every response includes**:
```
🧬 Processing question: [your question]
🤖 Agentic planning: [True/False - whether multi-step analysis needed]
💭 Planning reasoning: [explanation of processing approach]
📋 Using [traditional/agentic] query path
📊 Query type: [structural/semantic/hybrid/functional/comparative]
🔍 Search strategy: [direct_query/similarity_search/hybrid_search]

🤖 Answer:
[Detailed biological analysis with quantitative data]

Confidence: [high/medium/low]
Sources: [database results used]
```

**Query Metadata** (when available):
- Total results found
- Query execution time
- Database sources accessed

</details>

## Common Query Patterns

### Database Exploration

<details>
<summary><strong>Dataset Overview Queries (Click to expand)</strong></summary>

```bash
# Dataset composition
python -m src.cli ask "How many genomes, proteins, and annotations are in the database?"

# Functional coverage
python -m src.cli ask "What functional categories are most represented?"

# Quality assessment
python -m src.cli ask "What is the annotation completeness of the genomes?"

# Taxonomic distribution
python -m src.cli ask "What organisms are represented in the dataset?"
```

</details>

### Functional Analysis

<details>
<summary><strong>Pathway and Function Queries (Click to expand)</strong></summary>

```bash
# Pathway completeness
python -m src.cli ask "What metabolic pathways are complete in each genome?"

# Enzyme families
python -m src.cli ask "Show me all glycosyltransferases and their substrates"

# Biosynthetic potential
python -m src.cli ask "What secondary metabolites can these organisms produce?"

# Stress response
python -m src.cli ask "What stress response mechanisms are present?"
```

</details>

### Comparative Genomics

<details>
<summary><strong>Cross-Genome Analysis (Click to expand)</strong></summary>

```bash
# Functional differences
python -m src.cli ask "What metabolic capabilities are unique to each genome?"

# Core vs accessory functions
python -m src.cli ask "Which functions are shared across all genomes?"

# Specialization analysis
python -m src.cli ask "Which genome is most specialized for carbohydrate metabolism?"

# Synteny and organization
python -m src.cli ask "Are there conserved gene clusters across genomes?"
```

</details>

## Next Steps

- **[Complex Analysis Tutorial](complex-analysis.md)**: Multi-step workflows and advanced queries
- **[Python API Reference](../api-reference/python-api.md)**: Programmatic access
- **[Example Gallery](../examples/genomic-questions.md)**: 50+ tested queries with outputs