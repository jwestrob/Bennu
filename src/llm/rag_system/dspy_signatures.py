#!/usr/bin/env python3
"""
DSPy signatures for genomic RAG system.
Defines structured prompting interfaces for LLM interactions.
"""

import logging
from typing import List, Optional
import dspy

logger = logging.getLogger(__name__)

# Neo4j Database Schema
NEO4J_SCHEMA = """
CRITICAL RELATIONSHIP DIRECTION RULES:

MANDATORY: START WITH (p:Protein) - NEVER BACKWARDS!

WRONG: (da:DomainAnnotation)-[:DOMAINFAMILY]->(dom)<-[:HASDOMAIN]-(p:Protein)
CORRECT: (p:Protein)-[:HASDOMAIN]->(da:DomainAnnotation)-[:DOMAINFAMILY]->(dom:Domain)

PATTERN RULE: (p:Protein)-[:REL]->() NOT ()<-[:REL]-(p:Protein)

**ACTUAL DATABASE SCHEMA - ONLY USE THESE PROPERTIES**

**Node Labels and ONLY Available Properties:**

*   **`Genome`** - ONLY 2 properties available:
    *   `id`: (String) Unique identifier (e.g., `PLM0_60_b1_sep16_Maxbin2_047_curated.contigs`)
    *   `genomeId`: (String) Internal genome identifier
    *   **QUALITY METRICS**: Available via QualityMetrics node through HASQUALITYMETRICS relationship

*   **`Protein`**
    *   Represents a protein sequence translated from a gene.
    *   **Properties:**
        *   `id`: (String) Unique identifier for the protein (e.g., `protein:PLM0_60_b1_sep16_scaffold_10001_curated_1`).
        *   `length`: (String) The length of the amino acid sequence.
        *   `proteinId`: (String) The protein identifier without prefix.
    *   **Note**: Gene coordinates (start, end, strand) are on Gene nodes, accessible via ENCODEDBY relationship.

*   **`Gene`**
    *   Represents a protein-coding gene predicted from a genome.
    *   **Properties:**
        *   `id`: (String) Unique identifier for the gene (e.g., `gene:PLM0_60_b1_sep16_scaffold_10001_curated_1`).
        *   `geneId`: (String) The gene identifier without prefix.
        *   `startCoordinate`: (String) Start position of the gene on the contig.
        *   `endCoordinate`: (String) End position of the gene on the contig.
        *   `strand`: (String) Strand of the gene ('+1' or '-1').
        *   `gcContent`: (String) GC content of the gene.
        *   `lengthNt`: (String) Length in nucleotides.
        *   `lengthAA`: (String) Length in amino acids.
        *   `hasLocation`: (String) Location string format.

*   **`Domain`**
    *   Represents a protein domain from the PFAM database.
    *   **Properties:**
        *   `id`: (String) The PFAM accession ID (e.g., `PF00005.28`).
        *   `description`: (String) A description of the PFAM domain.

*   **`DomainAnnotation`**
    *   Represents domain annotation hits on proteins.
    *   **Properties:**
        *   `id`: (String) Unique annotation identifier.

*   **`KEGGOrtholog`**
    *   Represents a KEGG Orthology (KO) group.
    *   **Properties:**
        *   `id`: (String) The KO identifier (e.g., `K02014`).
        *   `description`: (String) A description of the KO group.

*   **`Bgc`**
    *   Represents a biosynthetic gene cluster predicted by GECCO (Gene Cluster Prediction with Conditional Random Fields).
    *   **Properties:**
        *   `id`: (String) Unique identifier (e.g., `cluster_1`, `cluster_2`).
        *   `bgcId`: (String) BGC identifier.
        *   `bgcProduct`: (String) Predicted product type (e.g., `Terpene`, `Unknown`).
        *   `contig`: (String) Source contig identifier.
        *   `startCoordinate`: (Integer) BGC start position.
        *   `endCoordinate`: (Integer) BGC end position.
        *   `lengthNt`: (Integer) BGC length in nucleotides.
        *   `proteinCount`: (Integer) Number of proteins in BGC.
        *   `averageProbability`: (Float) Average BGC probability score (0-1).
        *   `maxProbability`: (Float) Maximum BGC probability score (0-1).
        *   `alkaloidProbability`: (Float) Alkaloid product probability (0-1).
        *   `nrpProbability`: (Float) Non-ribosomal peptide probability (0-1).
        *   `polyketideProbability`: (Float) Polyketide probability (0-1).
        *   `rippProbability`: (Float) RiPP (ribosomally synthesized peptide) probability (0-1).
        *   `saccharideProbability`: (Float) Saccharide probability (0-1).
        *   `terpeneProbability`: (Float) Terpene probability (0-1).
        *   `domains`: (String) Semicolon-separated list of PFAM domains in BGC.

*   **`Cazymeannotation`**
    *   Represents a CAZyme (Carbohydrate-Active enZyme) annotation on a protein.
    *   IMPORTANT: To be used when users search for 'CAZymes' or similar, rather than searching by PFAM. Find entries where 'Cazymeannotation' has been populated.
    *   **Properties:**
        *   `id`: (String) Unique annotation identifier (e.g., `cazyme:PLM0_60_b1_sep16_scaffold_12180_curated_2_GH176_401`).
        *   `cazymeType`: (String) CAZyme family type (e.g., `GH`, `GT`, `PL`, `CE`, `AA`, `CBM`).
        *   `familyId`: (String) Specific CAZyme family ID (e.g., `GH3`, `GT2`, `CBM50`).
        *   `substrateSpecificity`: (String) Substrate prediction (e.g., `peptidoglycan (peptidoglycan)`, `xyloglucan`).
        *   `evalue`: (Float) E-value of the CAZyme annotation.
        *   `coverage`: (Float) Coverage of the CAZyme domain.
        *   `startPosition`: (Integer) Start position of CAZyme domain on protein.
        *   `endPosition`: (Integer) End position of CAZyme domain on protein.
        *   `hmmLength`: (Integer) Length of the HMM model.

*   **`Cazymefamily`**
    *   Represents a CAZyme family with functional information.
    *   **Properties:**
        *   `familyId`: (String) Family identifier (e.g., `GH3`, `GT2`, `CBM50`).
        *   `cazymeType`: (String) Family type (e.g., `GH`, `GT`, `PL`, `CE`, `AA`, `CBM`).
        *   `substrateSpecificity`: (String) Known substrates for this family.

*   **`QualityMetrics`**
    *   Represents genome quality metrics from QUAST analysis.
    *   **Properties:**
        *   `quast_totalLength`: (Integer) Total assembly length in base pairs.
        *   `quast_n50`: (Integer) N50 contig length in base pairs.
        *   `quast_numContigs`: (Integer) Number of contigs in the assembly.
        *   `quast_gcContent`: (Float) GC content percentage (0-100).
        *   `quast_largestContig`: (Integer) Size of the largest contig.
        *   `quast_n90`: (Integer) N90 contig length in base pairs.
        *   `quast_l50`: (Integer) L50 metric (number of contigs >= N50).
        *   `quast_l90`: (Integer) L90 metric (number of contigs >= N90).
        *   `quast_auN`: (Float) Area under the Nx curve.
        *   `quast_nsPer100kb`: (Float) N's per 100 kilobases.
        *   `quast_contigs1kbPlus`: (Integer) Number of contigs >= 1 kb.
        *   `quast_contigs5kbPlus`: (Integer) Number of contigs >= 5 kb.
        *   `quast_contigs10kbPlus`: (Integer) Number of contigs >= 10 kb.
        *   `quast_contigs25kbPlus`: (Integer) Number of contigs >= 25 kb.
        *   `quast_contigs50kbPlus`: (Integer) Number of contigs >= 50 kb.

**Relationships (ALL UPPERCASE):**

*   `(:Protein)-[:ENCODEDBY]->(:Gene)`: Connects a protein to the gene that encodes it.
*   `(:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(:Domain)`: Path to protein domains.
*   `(:Protein)-[:HASFUNCTION]->(:KEGGOrtholog)`: Connects a protein to its KEGG function.
*   `(:Gene)-[:BELONGSTOGENOME]->(:Genome)`: Connects a gene to the genome it belongs to.
*   `(:Genome)-[:HASBGC]->(:Bgc)`: Connects a genome to its BGCs.
*   `(:Gene)-[:PARTOFBGC]->(:Bgc)`: Connects genes that are part of a BGC.
*   `(:Protein)-[:HASCAZYME]->(:Cazymeannotation)`: Connects proteins to their CAZyme annotations.
*   `(:Cazymeannotation)-[:CAZYMEFAMILY]->(:Cazymefamily)`: Links CAZyme annotations to family information.
*   `(:Genome)-[:HASQUALITYMETRICS]->(:QualityMetrics)`: Connects genomes to their QUAST quality metrics.

**CRITICAL QUERY PATTERNS FOR TRANSPORT PROTEINS:**

**Pattern 1 - KEGG Transport Search (RECOMMENDED):**
```cypher
MATCH (ko:KEGGOrtholog) 
WHERE toLower(ko.description) CONTAINS 'transport'
MATCH (p:Protein)-[:HASFUNCTION]->(ko)
OPTIONAL MATCH (p)-[:ENCODEDBY]->(g:Gene)
OPTIONAL MATCH (p)-[:HASDOMAIN]->(da:DomainAnnotation)-[:DOMAINFAMILY]->(dom:Domain)
RETURN p.id AS protein_id, ko.id AS ko_id, ko.description AS ko_description,
       g.startCoordinate AS start_coordinate, g.endCoordinate AS end_coordinate, g.strand,
       collect(DISTINCT dom.id) AS pfam_accessions
```

**Pattern 2 - PFAM Domain Search:**
```cypher
MATCH (dom:Domain) 
WHERE toLower(dom.description) CONTAINS 'transport'
MATCH (p:Protein)-[:HASDOMAIN]->(da:DomainAnnotation)-[:DOMAINFAMILY]->(dom)
OPTIONAL MATCH (p)-[:ENCODEDBY]->(g:Gene)
OPTIONAL MATCH (p)-[:HASFUNCTION]->(ko:KEGGOrtholog)
RETURN p.id AS protein_id, ko.id AS ko_id, ko.description AS ko_description,
       g.startCoordinate AS start_coordinate, g.endCoordinate AS end_coordinate, g.strand,
       collect(DISTINCT dom.id) AS pfam_accessions
```

**Pattern 3 - BGC Search (GECCO-detected clusters):**
```cypher
MATCH (genome:Genome)-[:HASBGC]->(bgc:Bgc)
OPTIONAL MATCH (bgc)<-[:PARTOFBGC]-(gene:Gene)<-[:ENCODEDBY]-(protein:Protein)
OPTIONAL MATCH (protein)-[:HASFUNCTION]->(ko:KEGGOrtholog)
RETURN genome.genomeId, bgc.bgcId, bgc.bgcProduct, bgc.contig,
       bgc.startCoordinate, bgc.endCoordinate, bgc.lengthNt, bgc.proteinCount,
       bgc.averageProbability, bgc.maxProbability,
       bgc.alkaloidProbability, bgc.nrpProbability, bgc.polyketideProbability,
       bgc.rippProbability, bgc.saccharideProbability, bgc.terpeneProbability,
       bgc.domains,
       count(gene) as genes_in_bgc,
       collect(DISTINCT ko.koId) as ko_functions
ORDER BY bgc.maxProbability DESC, bgc.bgcId
```

CRITICAL QUERY PATTERNS FOR CAZYME ANNOTATIONS:

MANDATORY: When user mentions CAZyme, carbohydrate, glycoside, Cazymeannotation → USE THESE PATTERNS

For carbohydrate-active enzymes, glycoside hydrolases, glycosyltransferases:
- ALWAYS use `Cazymeannotation` and `Cazymefamily` nodes, NEVER Domain/PFAM search
- Filter by `cazymeType`: 'GH' (glycoside hydrolases), 'GT' (glycosyltransferases), 'PL' (polysaccharide lyases), 'CE' (carbohydrate esterases), 'AA' (auxiliary activities), 'CBM' (carbohydrate-binding modules)
- Access substrate specificity via `ca.substrateSpecificity` property
- Include quality metrics: `ca.evalue`, `ca.coverage`, `ca.startPosition`, `ca.endPosition`

Pattern 4 - CAZyme Search with Genome Info (RECOMMENDED for carbohydrate enzymes):
CRITICAL: NEVER USE [:BELONGSTO] - THE CORRECT RELATIONSHIP IS [:BELONGSTOGENOME]
ALWAYS WRITE: (g:Gene)-[:BELONGSTOGENOME]->(genome:Genome)
NEVER WRITE: (g:Gene)-[:BELONGSTO]->(genome:Genome)
```cypher
MATCH (p:Protein)-[:HASCAZYME]->(ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily)
OPTIONAL MATCH (p)-[:ENCODEDBY]->(g:Gene)
OPTIONAL MATCH (g)-[:BELONGSTOGENOME]->(genome:Genome)
OPTIONAL MATCH (p)-[:HASFUNCTION]->(ko:KEGGOrtholog)
RETURN genome.genomeId AS genome_id, p.id AS protein_id, cf.familyId AS cazyme_family, cf.cazymeType AS cazyme_type, ca.substrateSpecificity AS substrate,
       ca.evalue AS cazyme_evalue, ca.coverage AS cazyme_coverage,
       ko.id AS ko_id, ko.description AS ko_description,
       g.startCoordinate AS start_coordinate, g.endCoordinate AS end_coordinate, g.strand
```

**Pattern 5 - CAZyme Count and Distribution:**
```cypher
MATCH (ca:Cazymeannotation) 
RETURN ca.cazymeType AS cazyme_type, count(*) AS annotation_count 
ORDER BY annotation_count DESC
```

**Pattern 6 - All CAZyme families with counts:**
```cypher
MATCH (ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily)
MATCH (p:Protein)-[:HASCAZYME]->(ca)
RETURN cf.cazymeType AS family_type, cf.familyId AS family_id, cf.substrateSpecificity AS family_description,
       ca.substrateSpecificity AS substrate_specificity,
       count(p) AS protein_count
ORDER BY protein_count DESC, family_type, family_id
```

**Pattern 7 - Total CAZyme count:**
```cypher
MATCH (ca:Cazymeannotation) RETURN count(ca) AS total_cazymes
```

**CRITICAL QUERY PATTERNS FOR GENOME QUALITY METRICS:**

**Pattern 8 - Genome Quality Metrics (RECOMMENDED for assembly quality questions):**
```cypher
MATCH (genome:Genome)-[:HASQUALITYMETRICS]->(qm:QualityMetrics)
RETURN genome.genomeId AS genome_id,
       qm.quast_totalLength AS total_length,
       qm.quast_n50 AS n50,
       qm.quast_numContigs AS num_contigs,
       qm.quast_gcContent AS gc_content,
       qm.quast_largestContig AS largest_contig,
       qm.quast_n90 AS n90,
       qm.quast_l50 AS l50,
       qm.quast_l90 AS l90,
       qm.quast_auN AS auN,
       qm.quast_nsPer100kb AS ns_per_100kb,
       qm.quast_contigs1kbPlus AS contigs_1kb_plus,
       qm.quast_contigs5kbPlus AS contigs_5kb_plus,
       qm.quast_contigs10kbPlus AS contigs_10kb_plus,
       qm.quast_contigs25kbPlus AS contigs_25kb_plus,
       qm.quast_contigs50kbPlus AS contigs_50kb_plus
ORDER BY qm.quast_n50 DESC
```

**Pattern 9 - Compare Genome Quality Metrics:**
```cypher
MATCH (genome:Genome)-[:HASQUALITYMETRICS]->(qm:QualityMetrics)
RETURN genome.genomeId AS genome_id,
       qm.quast_totalLength AS total_length,
       qm.quast_n50 AS n50,
       qm.quast_numContigs AS num_contigs,
       qm.quast_gcContent AS gc_content
ORDER BY qm.quast_totalLength DESC
```

**Pattern 10 - Specific Genome Quality Metrics:**
```cypher
MATCH (genome:Genome {genomeId: "SPECIFIC_GENOME_ID"})-[:HASQUALITYMETRICS]->(qm:QualityMetrics)
RETURN genome.genomeId AS genome_id,
       qm.quast_totalLength AS total_length,
       qm.quast_n50 AS n50,
       qm.quast_numContigs AS num_contigs,
       qm.quast_gcContent AS gc_content,
       qm.quast_largestContig AS largest_contig
```
"""

class PlannerAgent(dspy.Signature):
    """
    Intelligent planning agent that determines if multi-step agentic execution is needed.

    Analyze the user's question to determine if it requires:
    1. Simple database lookup (traditional mode)
    2. Multi-step analysis with tools (agentic mode)

    Examples requiring agentic mode:
    - "Find proteins similar to X and analyze their functions" (similarity search + analysis)
    - "Compare metabolic capabilities between genomes" (multiple queries + comparison)
    - "Generate a report on CAZyme distribution" (query + code analysis + visualization)

    Examples for traditional mode:
    - "How many proteins are there?" (simple count)
    - "What is the function of protein X?" (direct lookup)
    - "Show me transport proteins" (single query)
    """

    user_query = dspy.InputField(desc="User's natural language question")
    requires_planning = dspy.OutputField(desc="Boolean: Does this query require multi-step agentic execution?")
    reasoning = dspy.OutputField(desc="Explanation of why agentic planning is or isn't needed")
    task_plan = dspy.OutputField(desc="If agentic: high-level task breakdown. If traditional: 'N/A'")

class QueryClassifier(dspy.Signature):
    """
    Classify genomic queries into categories for appropriate retrieval strategy.

    Query types:
    - structural: Specific database lookups (protein IDs, gene locations, annotations)
    - semantic: Similarity-based searches (find proteins similar to X)
    - hybrid: Combination of both (find similar proteins and their locations)
    - functional: Function-based searches (transport proteins, metabolic enzymes)
    - comparative: Cross-genome comparisons
    """

    question = dspy.InputField(desc="User's question about genomic data")
    query_type = dspy.OutputField(desc="Query classification: structural, semantic, hybrid, functional, or comparative")
    reasoning = dspy.OutputField(desc="Explanation of classification")
    key_entities = dspy.OutputField(desc="Key biological entities mentioned (proteins, genes, functions, etc.)")

class ContextRetriever(dspy.Signature):
    """
    Generate retrieval strategy for genomic queries based on classification and schema.

    CRITICAL: Generate ONLY executable Cypher queries with NO COMMENTS.

    FORBIDDEN:
    - Comments like /* comment */ or // comment
    - Multiple MATCH statements
    - Section headers or explanatory text
    - Semicolons separating queries

    REQUIRED: Start directly with MATCH, WITH, or OPTIONAL MATCH.

    CAZYME QUERY DETECTION - MANDATORY:
    When user mentions CAZyme, carbohydrate, glycoside, carbohydrate-active, dbCAN:
    - ALWAYS use Cazymeannotation and Cazymefamily nodes (NOT Domain/PFAM)
    - Use Pattern 4 from schema: (p:Protein)-[:HASCAZYME]->(ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily)
    - Include cf.familyId, cf.cazymeType, ca.substrateSpecificity for family details
    - Connect to genome via: (p)-[:ENCODEDBY]->(g:Gene)-[:BELONGSTOGENOME]->(genome:Genome)

    TRANSPORT QUERY DETECTION:
    When user mentions transport, transporter, channel, permease:
    - Use Pattern 1 from schema: KEGG-based transport search
    - Filter by: WHERE toLower(ko.description) CONTAINS 'transport'

    PROPHAGE/SPATIAL ANALYSIS DETECTION - CRITICAL:
    When user mentions prophage, phage, operons, spatial analysis, hypothetical proteins:
    - These queries require spatial genome reading tools, not database queries
    - If forced to generate a query, use basic gene/protein patterns:
      MATCH (p:Protein)-[:ENCODEDBY]->(g:Gene) RETURN p.id, g.startCoordinate, g.endCoordinate
    - NEVER generate BGC queries for prophage analysis

    COMPARATIVE QUERY RULES - NEVER USE LIMIT FOR THESE PATTERNS:
    - "Which genomes" → Show ALL genomes for comparison
    - "Compare across genomes" → Show ALL genomes
    - "Most/least among genomes" → Show ALL genomes ranked
    - "For each genome" → Show ALL genomes with counts
    - "All genomes" → Show ALL genomes
    - "Across all genomes" → Show ALL genomes
    - "Between genomes" → Show ALL genomes
    - "What genome has the most" → Show ALL genomes (let user see ranking)

    ONLY USE LIMIT 1 FOR:
    - "Show me ONE example"
    - "Just give me the top one"
    - "Only the best result"
    - "A single protein"

    EXAMPLES:
    ❌ BAD: "Which genome has most metal transporters?" → LIMIT 1
    ✅ GOOD: "Which genome has most metal transporters?" → ORDER BY count DESC (no LIMIT)

    ❌ BAD: "For each genome, how many transport proteins?" → LIMIT 1  
    ✅ GOOD: "For each genome, how many transport proteins?" → GROUP BY genome (no LIMIT)
    """

    db_schema = dspy.InputField(desc="Neo4j database schema with node types, relationships, and query patterns")
    question = dspy.InputField(desc="User's question")
    query_type = dspy.InputField(desc="Classified query type from QueryClassifier")

    search_strategy = dspy.OutputField(desc="Retrieval approach: direct_query, similarity_search, or hybrid_search")
    cypher_query = dspy.OutputField(desc="EXECUTABLE Cypher query with NO COMMENTS - must start with MATCH/WITH/OPTIONAL")
    reasoning = dspy.OutputField(desc="Explanation of retrieval strategy choice")
    expected_result_size = dspy.OutputField(desc="Estimated result size: small, medium, or large")

class RelevanceValidator(dspy.Signature):
    """
    Validate that retrieved genomic data is actually relevant to the user's original question.

    This prevents the system from analyzing irrelevant data (e.g., BGCs when user asks about phage).
    For phage/prophage queries, data about transporters, BGCs, or general metabolism is NOT relevant.
    For functional annotation queries, spatial/genomic organization data may not be relevant.

    Be strict - it's better to reject irrelevant data than to provide wrong analysis.
    """

    original_question = dspy.InputField(desc="User's original question")
    retrieved_data_summary = dspy.InputField(desc="Summary of what data was retrieved from the database")
    analysis_type = dspy.InputField(desc="Expected analysis type: spatial_genomic, functional_annotation, or comprehensive_discovery")

    is_relevant = dspy.OutputField(desc="Boolean: Is the retrieved data relevant to answering the original question?")
    relevance_score = dspy.OutputField(desc="Relevance score 0.0-1.0")
    reasoning = dspy.OutputField(desc="Explanation of why the data is or isn't relevant")
    missing_data_types = dspy.OutputField(desc="What types of data would be more relevant to the question?")

class GenomicAnswerer(dspy.Signature):
    """
    Generate comprehensive answers to genomic questions using retrieved context.

    Provides detailed biological interpretation with:
    - Scientific accuracy and proper terminology
    - Confidence assessment based on data quality
    - Relevant citations and data sources
    - Clear explanations for non-experts when appropriate

    CRITICAL: Only analyze data that is relevant to the original question.
    If the context doesn't contain relevant data, state this clearly rather than 
    providing analysis of unrelated genomic features.
    """

    question = dspy.InputField(desc="Original user question")
    context = dspy.InputField(desc="Retrieved genomic data and annotations")
    analysis_type = dspy.InputField(desc="Analysis type: spatial_genomic, functional_annotation, or comprehensive_discovery")
    answer = dspy.OutputField(desc="Comprehensive answer with biological insights, or statement that relevant data was not found")
    confidence = dspy.OutputField(desc="Confidence level: high, medium, or low")
    citations = dspy.OutputField(desc="Data sources and references used")

class GenomicSummarizer(dspy.Signature):
    """
    Summarize large genomic datasets while preserving essential biological information.

    Designed for context compression in multi-stage RAG systems where large datasets
    need to be condensed to fit within token limits while maintaining scientific accuracy.

    Focus areas:
    - Functional annotations (KEGG, PFAM domains)
    - Unique biological insights and patterns
    - Quantitative summaries (counts, distributions)
    - Key examples representative of larger datasets
    """

    genomic_data = dspy.InputField(desc="Large genomic dataset to summarize")
    target_length = dspy.InputField(desc="Target length: brief, medium, or detailed")
    focus_areas = dspy.InputField(desc="Specific biological aspects to emphasize")
    summary = dspy.OutputField(desc="Concise summary preserving essential biological information")
    key_findings = dspy.OutputField(desc="Most important biological insights from the data")
    data_statistics = dspy.OutputField(desc="Quantitative summary of the dataset")

class NotingDecision(dspy.Signature):
    """
    Decide whether task results warrant note-taking to capture ALL BIOLOGICAL INFORMATION.

    **GOAL: EXTENSIVE NOTE-TAKING FOR BIOLOGICAL DISCOVERY**
    
    **RECORD NOTES FOR ALL:**
    - Genome reading results (gene counts, coordinates, annotations, domains)
    - Spatial data (contig information, gene organization, strand distribution)
    - Discovery findings (prophage indicators, hypothetical protein clusters, unusual patterns)
    - Quantitative results (counts, percentages, scores, statistics)
    - Analysis outcomes (what was searched, what was found, what patterns emerged)
    - Annotation data (PFAM domains, KO functions, specific gene annotations)
    - Coordinate information (start/end positions, genomic locations)
    - Comparative data (differences between genomes, organisms, or regions)
    
    **BE DETAILED AND SPECIFIC:**
    - Extract specific gene IDs, coordinates, and annotations
    - Note exact counts, percentages, and measurements
    - Record domain identifiers (PF00123) and their positions
    - Document spatial patterns and genomic organization
    - Capture both positive findings AND negative results ("searched X, found Y")
    
    **ONLY SKIP FOR:**
    - Complete task failures (errors, exceptions, no data)
    - Pure meta-operations with no biological content
    
    **DECISION CRITERIA:**
    Default to taking notes. If there's ANY biological information, genomic data, 
    analysis results, or discovery content, record detailed notes about it.
    """

    task_description = dspy.InputField(desc="Description of the task that was executed")
    execution_result = dspy.InputField(desc="Results from task execution (structured data, context, etc.)")
    existing_notes = dspy.InputField(desc="Summary of notes from previous tasks in this session")
    original_user_question = dspy.InputField(desc="Original user question that started this analysis")
    task_type = dspy.InputField(desc="Type of task executed (ATOMIC_QUERY, TOOL_CALL)")
    analysis_context = dspy.InputField(desc="Type of analysis being performed (discovery, comparison, lookup, exploration)")

    should_record = dspy.OutputField(desc="Boolean: Should we record notes for this task? Default TRUE for all biological content")
    importance_score = dspy.OutputField(desc="Importance score 1-10 for this information (biological data gets 5+ minimum)")
    reasoning = dspy.OutputField(desc="Explanation of note-taking decision emphasizing biological information captured")
    observations = dspy.OutputField(desc="If recording: DETAILED observations with coordinates, genes, counts, and patterns")
    key_findings = dspy.OutputField(desc="If recording: ALL biological findings, annotations, and discoveries")
    cross_connections = dspy.OutputField(desc="If recording: connections to other tasks (task_id:connection_type:description format)")
    quantitative_data = dspy.OutputField(desc="If recording: ALL numerical data, coordinates, counts, and measurements")

class ReportPartGenerator(dspy.Signature):
    """
    Generate a specific part of a multi-part genomic report.

    This signature handles individual sections of large reports that have been
    chunked to manage token limits while maintaining scientific rigor and
    comprehensive analysis.

    Each part should:
    - Be self-contained but reference overall context
    - Maintain consistent terminology and formatting
    - Include quantitative analysis and biological insights
    - Connect to broader themes when relevant
    """

    question = dspy.InputField(desc="Original user question for context")
    data_chunk = dspy.InputField(desc="Data subset for this specific report part")
    part_context = dspy.InputField(desc="Context about this part's role in the overall report")
    previous_parts_summary = dspy.InputField(desc="Summary of previous parts to maintain consistency")
    report_type = dspy.InputField(desc="Type of report (e.g., 'crispr_systems', 'functional_comparison')")

    part_content = dspy.OutputField(desc="Detailed scientific content for this report part")
    key_findings = dspy.OutputField(desc="Key biological findings from this part")
    quantitative_summary = dspy.OutputField(desc="Quantitative metrics and statistics")
    connections_to_whole = dspy.OutputField(desc="How this part connects to overall analysis")

class ExecutiveSummaryGenerator(dspy.Signature):
    """
    Generate executive summary for multi-part genomic reports.

    Creates a concise overview that captures the essence of large-scale
    genomic analyses while highlighting key findings and providing roadmap
    for detailed sections.

    Should include:
    - Overall scope and methodology
    - Key quantitative findings
    - Major biological insights
    - Navigation guide for detailed sections
    """

    question = dspy.InputField(desc="Original user question")
    data_overview = dspy.InputField(desc="Statistical overview of the complete dataset")
    key_patterns = dspy.InputField(desc="Major patterns identified across all data")
    report_structure = dspy.InputField(desc="Structure of the multi-part report")

    executive_summary = dspy.OutputField(desc="Concise executive summary of findings")
    scope_and_methodology = dspy.OutputField(desc="Overview of analysis scope and approach")
    key_statistics = dspy.OutputField(desc="Essential quantitative findings")
    navigation_guide = dspy.OutputField(desc="Guide to navigating the detailed report parts")

class ReportSynthesisGenerator(dspy.Signature):
    """
    Generate final synthesis section for multi-part reports.

    Integrates findings from all report parts into coherent conclusions,
    identifies cross-cutting themes, and provides biological interpretation
    and recommendations for future work.

    Should provide:
    - Integration of all parts
    - Cross-cutting biological insights
    - Evolutionary and ecological implications
    - Recommendations for follow-up studies
    """

    question = dspy.InputField(desc="Original user question")
    all_parts_summary = dspy.InputField(desc="Summary of findings from all report parts")
    cross_cutting_themes = dspy.InputField(desc="Themes that emerge across multiple parts")
    quantitative_integration = dspy.InputField(desc="Integrated quantitative analysis")

    synthesis_content = dspy.OutputField(desc="Comprehensive synthesis of all findings")
    biological_implications = dspy.OutputField(desc="Broader biological and evolutionary implications")
    recommendations = dspy.OutputField(desc="Recommendations for future research or applications")
    confidence_assessment = dspy.OutputField(desc="Assessment of confidence in conclusions")

class GenomicQuery(dspy.Signature):
    """You must generate exactly one Cypher query. DO NOT generate multiple queries.

CRITICAL: If user asks for comprehensive analysis of metabolism and lifestyle, focus on the MOST IMPORTANT aspect only (like KEGG functions) in a single query.

FORBIDDEN:
- Multiple MATCH statements in sequence
- Comments like /* comment */
- Semicolons separating queries
- Section headers or numbers

REQUIRED FORMAT:
MATCH (genome:Genome {genomeId: "GENOME_ID"})<-[:BELONGSTOGENOME]-(g:Gene)<-[:ENCODEDBY]-(p:Protein)-[:HASFUNCTION]->(ko:KEGGOrtholog) RETURN ko.id, ko.description, count(p) ORDER BY count(p) DESC
"""
    question = dspy.InputField(desc="Question about genomic data")
    context = dspy.InputField(desc="Relevant genomic context and schema information")
    query = dspy.OutputField(desc="ONE Cypher query only. No comments. No multiple queries.")

class TaxonomicClassification(dspy.Signature):
    """
    Taxonomic classification and comparison signature.

    TODO: Add taxonomic reasoning capabilities
    """
    genome_features = dspy.InputField(desc="Genome quality and gene content features")
    known_taxa = dspy.InputField(desc="Known taxonomic classifications for reference")
    classification = dspy.OutputField(desc="Predicted taxonomic classification with confidence")
    reasoning = dspy.OutputField(desc="Explanation of classification reasoning")

class FunctionalAnnotation(dspy.Signature):
    """
    Functional annotation interpretation signature.

    TODO: Add domain and pathway analysis
    """
    protein_domains = dspy.InputField(desc="Protein domain hits and annotations")
    gene_context = dspy.InputField(desc="Genomic context and gene clustering")
    function_prediction = dspy.OutputField(desc="Predicted protein function")
    confidence_score = dspy.OutputField(desc="Confidence in functional prediction")

class ComparativeGenomics(dspy.Signature):
    """
    Comparative genomic analysis signature.

    TODO: Add genome comparison capabilities
    """
    query_genome = dspy.InputField(desc="Query genome features and annotations")
    reference_genomes = dspy.InputField(desc="Reference genome data for comparison")
    similarities = dspy.OutputField(desc="Identified genomic similarities and differences")
    evolutionary_insights = dspy.OutputField(desc="Evolutionary and ecological insights")

class MetabolicPathway(dspy.Signature):
    """
    Metabolic pathway analysis signature.

    TODO: Add pathway reconstruction logic
    """
    enzyme_annotations = dspy.InputField(desc="Enzyme classifications and domains")
    pathway_database = dspy.InputField(desc="Known metabolic pathway information")
    pathway_presence = dspy.OutputField(desc="Predicted metabolic pathway completeness")
    missing_enzymes = dspy.OutputField(desc="Enzymes missing from pathways")

class GenomeQuality(dspy.Signature):
    """
    Genome quality assessment signature.

    TODO: Add quality interpretation logic
    """
    assembly_metrics = dspy.InputField(desc="QUAST and CheckM quality metrics")
    quality_thresholds = dspy.InputField(desc="Quality thresholds for different analyses")
    quality_assessment = dspy.OutputField(desc="Overall genome quality assessment")
    recommendations = dspy.OutputField(desc="Recommendations for genome usage")