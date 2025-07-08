# Genomic Questions Gallery

A comprehensive collection of tested queries with expected outputs and biological insights.

## Quick Reference

| Category | Example Query | Expected Result Type |
|----------|---------------|---------------------|
| **Counting** | "How many proteins are there?" | 10,102 proteins |
| **Functional** | "Show me transport proteins" | 635 transporters with mechanisms |
| **Comparative** | "Compare CAZyme distribution" | Per-genome breakdown with analysis |
| **Similarity** | "Find proteins similar to heme transporters" | ESM2 similarity scores + context |
| **Pathways** | "What metabolic pathways are complete?" | Pathway coverage analysis |

## Categories

### 1. Database Exploration

<details>
<summary><strong>Dataset Overview (Click to expand)</strong></summary>

**Basic Counts**:
```bash
# Core entities
python -m src.cli ask "How many proteins are in the database?"
# → 10,102 distinct protein records

python -m src.cli ask "How many genomes are in the dataset?"
# → 4 microbial genomes processed

python -m src.cli ask "How many KEGG orthologs are annotated?"
# → 2,609 distinct KEGG orthologs

python -m src.cli ask "How many PFAM domains are identified?"
# → Domain coverage across protein families

python -m src.cli ask "How many CAZymes are annotated?"
# → 1,845 carbohydrate-active enzymes
```

**Coverage Statistics**:
```bash
python -m src.cli ask "What percentage of proteins have functional annotations?"
# → Annotation completeness analysis

python -m src.cli ask "Which genome has the most proteins?"
# → Per-genome protein counts with rankings

python -m src.cli ask "What is the average protein length in the dataset?"
# → Length distribution statistics

python -m src.cli ask "How many biosynthetic gene clusters are predicted?"
# → GECCO BGC detection results
```

**Quality Metrics**:
```bash
python -m src.cli ask "What is the annotation quality across genomes?"
# → Completeness and confidence metrics

python -m src.cli ask "Which proteins have the highest confidence annotations?"
# → Top-scoring functional assignments

python -m src.cli ask "What fraction of proteins are hypothetical?"
# → Unknown function analysis
```

</details>

### 2. Functional Classification

<details>
<summary><strong>Protein Function Queries (Click to expand)</strong></summary>

**Transport Systems**:
```bash
python -m src.cli ask "What transport proteins are present in the database?"
# → 635 transporters categorized by mechanism:
#   • ABC transporters (ATP-binding proteins, permeases)
#   • P-type ATPases (Ca²⁺, Cu²⁺, Na⁺/K⁺)
#   • Secondary transporters (symporters, antiporters)
#   • Ion channels and porins

python -m src.cli ask "Show me ABC transporter components"
# → ATP-binding proteins, permeases, substrate-binding proteins

python -m src.cli ask "Find iron acquisition systems"
# → Siderophore transporters, heme uptake, iron ABC systems

python -m src.cli ask "What sugar transport systems are present?"
# → PTS system components, sugar ABC transporters
```

**Metabolic Enzymes**:
```bash
python -m src.cli ask "Show me central metabolism proteins"
# → Glycolysis, TCA cycle, pentose phosphate pathway enzymes

python -m src.cli ask "What amino acid biosynthesis pathways are complete?"
# → Per-pathway completeness analysis

python -m src.cli ask "Find carbohydrate metabolism enzymes"
# → Glycoside hydrolases, glycosyltransferases, related enzymes

python -m src.cli ask "Show me energy production pathways"
# → Respiratory complexes, ATP synthase, fermentation
```

**DNA/RNA Processing**:
```bash
python -m src.cli ask "What DNA repair mechanisms are present?"
# → Mismatch repair, base excision repair, recombination

python -m src.cli ask "Show me transcription factors and regulators"
# → Transcriptional regulators, sigma factors, two-component systems

python -m src.cli ask "Find ribosomal proteins and RNA processing enzymes"
# → Translation machinery components

python -m src.cli ask "What DNA replication proteins are annotated?"
# → DNA polymerases, helicases, primase, topoisomerases
```

**Stress Response**:
```bash
python -m src.cli ask "What stress response proteins are present?"
# → Heat shock proteins, cold shock, oxidative stress

python -m src.cli ask "Find antibiotic resistance mechanisms"
# → Beta-lactamases, efflux pumps, modification enzymes

python -m src.cli ask "Show me oxidative stress defense systems"
# → Catalases, peroxidases, superoxide dismutase

python -m src.cli ask "What toxin-antitoxin systems are present?"
# → TA system components and mechanisms
```

</details>

### 3. Carbohydrate-Active Enzymes (CAZymes)

<details>
<summary><strong>CAZyme Analysis Queries (Click to expand)</strong></summary>

**Distribution Analysis**:
```bash
python -m src.cli ask "Show me the distribution of CAZyme types among each genome in the dataset; compare and contrast."
# → PROVEN OUTPUT: Detailed per-genome CAZyme profiles:
#   1. Burkholderiales: 1,056 CAZymes (GH-dominant: 41.3%)
#   2. PLM0_60_b1: 425 CAZymes (GT-dominant: 41.9%)  
#   3. Candidatus Muproteobacteria: 264 CAZymes (balanced GH/GT)
#   4. Candidatus Nomurabacteria: 100 CAZymes (smallest, balanced)
#   + Biological interpretation of trophic roles

python -m src.cli ask "Which genome has the most diverse CAZyme repertoire?"
# → Diversity analysis across CAZyme families

python -m src.cli ask "Compare glycoside hydrolase families across genomes"
# → GH family distribution and substrate specificity

python -m src.cli ask "What carbohydrate substrates can each genome process?"
# → Substrate prediction from CAZyme complement
```

**Functional CAZyme Queries**:
```bash
python -m src.cli ask "Show me cellulose degradation capabilities"
# → Cellulases (GH families), cellulose-binding modules

python -m src.cli ask "What starch processing enzymes are present?"
# → Alpha-amylases, pullulanases, debranching enzymes

python -m src.cli ask "Find chitin and peptidoglycan degradation enzymes"
# → Chitinases, lysozyme, peptidoglycan hydrolases

python -m src.cli ask "Show me pectin degradation pathways"
# → Polygalacturonases, pectin lyases, esterases

python -m src.cli ask "What glycosyltransferases are involved in cell wall biosynthesis?"
# → GT families for peptidoglycan, cellulose, other structural polysaccharides
```

**Specialized CAZyme Analysis**:
```bash
python -m src.cli ask "Which CAZyme families are unique to specific genomes?"
# → Genome-specific CAZyme repertoires

python -m src.cli ask "Show me auxiliary activity enzymes and their roles"
# → AA families for lignin modification, cellulose oxidation

python -m src.cli ask "What carbohydrate-binding modules are most common?"
# → CBM family distribution and substrate targets

python -m src.cli ask "Find CAZymes with multiple catalytic domains"
# → Multi-domain architecture analysis
```

</details>

### 4. Biosynthetic Gene Clusters (BGCs)

<details>
<summary><strong>BGC and Secondary Metabolism (Click to expand)</strong></summary>

**BGC Discovery**:
```bash
python -m src.cli ask "How many biosynthetic gene clusters are predicted?"
# → GECCO BGC detection results with confidence scores

python -m src.cli ask "What types of secondary metabolites can be produced?"
# → BGC product classification: polyketides, NRPs, terpenes, etc.

python -m src.cli ask "Show me BGCs with high confidence scores"
# → GECCO probability filtering (>0.8) with product predictions

python -m src.cli ask "Which genome has the most biosynthetic potential?"
# → Per-genome BGC counts and diversity analysis
```

**Product-Specific Queries**:
```bash
python -m src.cli ask "Find polyketide biosynthesis clusters"
# → Type I, II, III PKS systems with product predictions

python -m src.cli ask "Show me non-ribosomal peptide synthetases"
# → NRPS modules and predicted amino acid specificity

python -m src.cli ask "What terpene biosynthesis pathways are present?"
# → Terpene synthases and precursor biosynthesis

python -m src.cli ask "Find antibiotic biosynthesis clusters"
# → Known antibiotic classes and novel clusters
```

**BGC Architecture**:
```bash
python -m src.cli ask "What is the average size of biosynthetic clusters?"
# → BGC length distribution and gene content

python -m src.cli ask "Show me BGC organization and gene synteny"
# → Gene order conservation and functional clustering

python -m src.cli ask "Find BGCs with transport and resistance genes"
# → Self-protection mechanisms in biosynthetic clusters

python -m src.cli ask "What regulatory elements are associated with BGCs?"
# → Transcriptional regulators in cluster boundaries
```

</details>

### 5. Similarity and Homology

<details>
<summary><strong>Protein Similarity Searches (Click to expand)</strong></summary>

**General Similarity**:
```bash
python -m src.cli ask "Find proteins similar to heme transporters"
# → Multi-stage analysis:
#   Stage 1: Find annotated heme transporters in Neo4j
#   Stage 2: Use as seeds for LanceDB ESM2 similarity search
#   Result: Proteins with >0.7 similarity + biological interpretation

python -m src.cli ask "Show me proteins similar to DNA repair enzymes"
# → Sequence similarity to repair mechanisms

python -m src.cli ask "Find proteins with similar domain architecture to ABC transporters"
# → Domain-based similarity and functional prediction

python -m src.cli ask "What proteins are most similar to protein X?"
# → Direct similarity search for specific protein
```

**Functional Similarity**:
```bash
python -m src.cli ask "Find uncharacterized proteins similar to known enzymes"
# → Function prediction via similarity

python -m src.cli ask "Show me potential paralogs of metabolic enzymes"
# → Within-genome similarity for gene family analysis

python -m src.cli ask "Find proteins with similar catalytic domains"
# → Active site conservation analysis

python -m src.cli ask "What orphan proteins might have transport functions?"
# → Similarity-based function prediction
```

**Evolutionary Analysis**:
```bash
python -m src.cli ask "Which proteins are conserved across all genomes?"
# → Core genome analysis via similarity

python -m src.cli ask "Find recently duplicated genes"
# → High-similarity protein pairs within genomes

python -m src.cli ask "Show me horizontally transferred genes"
# → Similarity patterns suggesting HGT

python -m src.cli ask "What protein families have expanded in specific genomes?"
# → Gene family expansion analysis
```

</details>

### 6. Genomic Context and Organization

<details>
<summary><strong>Gene Neighborhood Analysis (Click to expand)</strong></summary>

**Operon Prediction**:
```bash
python -m src.cli ask "What genes are in the same operon as succinate dehydrogenase?"
# → Co-transcribed gene prediction based on proximity and strand

python -m src.cli ask "Show me ribosomal protein operons"
# → rRNA operon organization and ribosomal protein clusters

python -m src.cli ask "Find ABC transporter operons"
# → Multi-gene transport system organization

python -m src.cli ask "What genes cluster with DNA repair functions?"
# → Repair gene neighborhoods and regulatory coupling
```

**Synteny and Conservation**:
```bash
python -m src.cli ask "Are there conserved gene clusters across genomes?"
# → Synteny analysis for functional gene clustering

python -m src.cli ask "Show me genomic islands and their functions"
# → Horizontally acquired regions

python -m src.cli ask "What gene neighborhoods are unique to each genome?"
# → Genome-specific organization patterns

python -m src.cli ask "Find genes with conserved neighborhood architecture"
# → Functionally coupled gene clusters
```

**Regulatory Context**:
```bash
python -m src.cli ask "What genes are near transcriptional regulators?"
# → Regulatory neighborhoods and target prediction

python -m src.cli ask "Show me sigma factor regulons"
# → Alternative sigma factor target genes

python -m src.cli ask "Find two-component system neighborhoods"
# → Sensor-regulator-target organization

python -m src.cli ask "What genes have similar regulatory contexts?"
# → Co-regulation prediction via neighborhood similarity
```

</details>

### 7. Comparative Genomics

<details>
<summary><strong>Cross-Genome Analysis (Click to expand)</strong></summary>

**Functional Comparison**:
```bash
python -m src.cli ask "What metabolic capabilities are unique to each genome?"
# → Differential functional analysis

python -m src.cli ask "Compare nitrogen metabolism across genomes"
# → N-cycle pathway completeness comparison

python -m src.cli ask "Which genome is most specialized for plant biomass degradation?"
# → CAZyme specialization analysis

python -m src.cli ask "Show me core vs accessory functions"
# → Pan-genome functional analysis
```

**Pathway Analysis**:
```bash
python -m src.cli ask "What metabolic pathways are complete in each genome?"
# → KEGG pathway completeness matrix

python -m src.cli ask "Compare central carbon metabolism between organisms"
# → Glycolysis, TCA, PPP pathway comparison

python -m src.cli ask "Which pathways show the most variation?"
# → Variable pathway analysis

python -m src.cli ask "Find complementary metabolic capabilities"
# → Syntrophic potential analysis
```

**Ecological Interpretation**:
```bash
python -m src.cli ask "What do the functional profiles suggest about ecological roles?"
# → Niche prediction from genomic content

python -m src.cli ask "Which organisms might compete for the same resources?"
# → Resource overlap analysis

python -m src.cli ask "Show me evidence for metabolic cooperation"
# → Cross-feeding potential

python -m src.cli ask "What environmental adaptations are evident?"
# → Stress response and environmental gene analysis
```

</details>

### 8. Pathway and Systems Analysis

<details>
<summary><strong>Metabolic Network Queries (Click to expand)</strong></summary>

**Central Metabolism**:
```bash
python -m src.cli ask "What central metabolism pathways are present?"
# → Glycolysis, TCA cycle, pentose phosphate pathway analysis

python -m src.cli ask "Show me energy production mechanisms"
# → Respiratory chains, fermentation, photosynthesis

python -m src.cli ask "Find carbon fixation pathways"
# → Calvin cycle, alternative CO2 fixation routes

python -m src.cli ask "What alternative metabolic routes exist?"
# → Bypass pathways and metabolic flexibility
```

**Amino Acid Metabolism**:
```bash
python -m src.cli ask "Which amino acids can each organism synthesize?"
# → Biosynthetic pathway completeness

python -m src.cli ask "Show me amino acid degradation pathways"
# → Catabolic routes for nitrogen and carbon

python -m src.cli ask "Find branched-chain amino acid metabolism"
# → Leucine, isoleucine, valine pathways

python -m src.cli ask "What amino acid transport systems are present?"
# → Uptake mechanisms for external amino acids
```

**Cofactor and Vitamin Biosynthesis**:
```bash
python -m src.cli ask "What cofactors can each organism produce?"
# → B-vitamin, heme, quinone biosynthesis

python -m src.cli ask "Show me folate metabolism pathways"
# → One-carbon metabolism and methylation

python -m src.cli ask "Find heme biosynthesis and utilization"
# → Porphyrin biosynthesis and heme proteins

python -m src.cli ask "What metal cofactor systems are present?"
# → Iron-sulfur clusters, molybdenum cofactors
```

</details>

### 9. Specific Protein Inquiries

<details>
<summary><strong>Individual Protein Analysis (Click to expand)</strong></summary>

**Specific Functional Queries**:
```bash
python -m src.cli ask "What is the function of KEGG ortholog K20469?"
# → Specific KO functional description and pathway context

python -m src.cli ask "Show me details for protein RIFCSPHIGHO2_01_scaffold_10_364"
# → Individual protein annotation, domains, context

python -m src.cli ask "What domains are present in protein PLM0_60_scaffold_5_892?"
# → PFAM domain architecture and functional prediction

python -m src.cli ask "Find all proteins with PFAM domain PF01594"
# → Domain-specific protein search and functional analysis
```

**Protein Context**:
```bash
python -m src.cli ask "Where is gene X located and what are its neighbors?"
# → Genomic coordinates and neighborhood analysis

python -m src.cli ask "What is the operon structure around protein Y?"
# → Co-transcription prediction and regulatory context

python -m src.cli ask "Show me the evolutionary context of protein Z"
# → Homology, conservation, and phylogenetic distribution

python -m src.cli ask "What pathways involve protein W?"
# → Pathway mapping and functional network analysis
```

</details>

### 10. Advanced Multi-Step Queries

<details>
<summary><strong>Complex Analysis Workflows (Click to expand)</strong></summary>

**Integrated Analysis**:
```bash
python -m src.cli ask "Find transport proteins, analyze their genomic context, and compare across genomes"
# → Multi-step workflow:
#   1. Identify transport proteins
#   2. Analyze genomic neighborhoods  
#   3. Compare organization patterns

python -m src.cli ask "Show me CAZyme distribution, identify specializations, and predict ecological roles"
# → Functional profiling → specialization → ecological interpretation

python -m src.cli ask "Find BGCs, predict products, and analyze distribution patterns"
# → BGC discovery → product prediction → comparative analysis

python -m src.cli ask "Identify core metabolic functions and compare pathway completeness"
# → Core genome → pathway mapping → completeness analysis
```

**Hypothesis Generation**:
```bash
python -m src.cli ask "What evidence suggests symbiotic relationships between these organisms?"
# → Metabolic complementarity analysis

python -m src.cli ask "Which organism is most likely to be a primary degrader?"
# → Functional capability ranking

python -m src.cli ask "What novel enzyme activities might be present?"
# → Orphan annotation and similarity analysis

python -m src.cli ask "How do these genomes compare to their reference strains?"
# → Reference comparison and novel feature identification
```

</details>

## Performance Expectations

### Query Response Times

<details>
<summary><strong>Typical Performance Metrics (Click to expand)</strong></summary>

**Fast Queries** (<5 seconds):
- Simple counts ("How many proteins?")
- Direct lookups ("What is function of K12345?")
- Single-category searches ("Show transport proteins")

**Medium Queries** (5-15 seconds):
- Functional analysis with interpretation
- Simple comparative queries
- Similarity searches with small result sets

**Complex Queries** (15-60 seconds):
- Multi-genome comparative analysis
- Large similarity searches with genomic context
- Integrated multi-step workflows

**Note**: Times depend on:
- LLM processing (reasoning models take longer)
- Database query complexity
- Result set size and interpretation depth

</details>

### Query Success Factors

<details>
<summary><strong>What Makes Queries Work Well (Click to expand)</strong></summary>

**High Success Rate**:
- Specific biological terms (transport, metabolism, etc.)
- Clear intent (count, compare, find, show)
- Reasonable scope (single concept or logical combination)

**Medium Success Rate**:
- Complex multi-step analysis
- Ambiguous terminology
- Very broad questions

**Optimization Tips**:
- Start with simple versions of complex questions
- Use established biological terminology
- Break complex analyses into multiple queries
- Specify the type of answer you want (count, list, comparison)

</details>

## Next Steps

- **[Complex Analysis Tutorial](../tutorials/complex-analysis.md)**: Multi-step workflows
- **[API Reference](../api-reference/python-api.md)**: Programmatic access
- **[Architecture Overview](../architecture/overview.md)**: System design details