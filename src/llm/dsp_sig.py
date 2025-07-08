"""
DSPy Signature Definitions
Structured prompting signatures for genomic question answering.
"""

from typing import List, Optional
import logging

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    logging.warning("DSPy not available - install dsp-ml package")

logger = logging.getLogger(__name__)


if DSPY_AVAILABLE:
    
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

For comprehensive questions, choose ONE of:
1. KEGG functions (most important for metabolism)
2. CAZyme families (for carbohydrate metabolism)  
3. BGC clusters (for secondary metabolism)

Never combine multiple approaches in one response."""
        question = dspy.InputField(desc="Question about genomic data")
        context = dspy.InputField(desc="Relevant genomic context and schema information")
        query = dspy.OutputField(desc="ONE Cypher query only. No comments. No multiple queries.")

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
    *   FORBIDDEN: taxon, total_length, n50, num_contigs, completeness, contamination (DO NOT USE - THESE DO NOT EXIST)

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
    *   Represents a biosynthetic gene cluster predicted by GECCO.
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

**Relationships (ALL UPPERCASE):**

*   `(:Protein)-[:ENCODEDBY]->(:Gene)`: Connects a protein to the gene that encodes it.
*   `(:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(:Domain)`: Path to protein domains.
*   `(:Protein)-[:HASFUNCTION]->(:KEGGOrtholog)`: Connects a protein to its KEGG function.
*   `(:Gene)-[:BELONGSTOGENOME]->(:Genome)`: Connects a gene to the genome it belongs to.
*   `(:Genome)-[:HASBGC]->(:Bgc)`: Connects a genome to its BGCs.
*   `(:Gene)-[:PARTOFBGC]->(:Bgc)`: Connects genes that are part of a BGC.
*   `(:Protein)-[:HASCAZYME]->(:Cazymeannotation)`: Connects proteins to their CAZyme annotations.
*   `(:Cazymeannotation)-[:CAZYMEFAMILY]->(:Cazymefamily)`: Links CAZyme annotations to family information.

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

**Pattern 3 - BGC Search:**
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
"""
    
    
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

else:
    # Fallback classes when DSPy is not available
    
    class BaseMockSignature:
        """Base class for mock signatures when DSPy is unavailable."""
        
        def __init__(self, **kwargs):
            self.inputs = kwargs
            logger.warning("DSPy not available - using mock signature")
        
        def __call__(self, **kwargs):
            return {"answer": "DSPy not available - install dsp-ml package"}
    
    
    class GenomicQuery(BaseMockSignature):
        """Mock genomic query signature."""
        pass
    
    
    class TaxonomicClassification(BaseMockSignature):
        """Mock taxonomic classification signature."""
        pass
    
    
    class FunctionalAnnotation(BaseMockSignature):
        """Mock functional annotation signature."""
        pass
    
    
    class ComparativeGenomics(BaseMockSignature):
        """Mock comparative genomics signature."""
        pass
    
    
    class MetabolicPathway(BaseMockSignature):
        """Mock metabolic pathway signature."""
        pass
    
    
    class GenomeQuality(BaseMockSignature):
        """Mock genome quality signature."""
        pass


def create_genomic_chain():
    """
    Create a DSPy chain for genomic question answering.
    
    TODO: Implement complete chain construction
    """
    if not DSPY_AVAILABLE:
        logger.error("Cannot create chain - DSPy not available")
        return None
    
    # TODO: Configure LLM backend
    # TODO: Create chain with retrieval and reasoning
    # TODO: Add genomic domain knowledge
    
    logger.info("Genomic chain creation placeholder - not yet implemented")
    return None


def configure_llm_backend(
    provider: str = "openai",
    model: str = "gpt-3.5-turbo",
    **kwargs
):
    """
    Configure LLM backend for DSPy signatures.
    
    TODO: Add support for multiple LLM providers
    
    Args:
        provider: LLM provider (openai, anthropic, local, etc.)
        model: Model name/identifier
        **kwargs: Additional provider-specific configuration
    """
    if not DSPY_AVAILABLE:
        logger.error("Cannot configure LLM - DSPy not available")
        return
    
    # TODO: Configure DSPy LLM backend
    # if provider == "openai":
    #     lm = dspy.OpenAI(model=model, **kwargs)
    # elif provider == "anthropic":
    #     lm = dspy.Claude(model=model, **kwargs)
    # else:
    #     raise ValueError(f"Unsupported provider: {provider}")
    
    # dspy.settings.configure(lm=lm)
    
    logger.info(f"LLM configuration placeholder - {provider}:{model}")
