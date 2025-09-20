# RDF Structure Reference

Based on analysis of reference data in `knowledge_graphs/dummy_4genomes/stage05_kg/csv/`

## Node Types

### 1. Genomes
**CSV**: `genomes.csv`
**Properties**: 
- `id:ID` - Format: `{genome_name}`
- `genomeId` - Same as ID

**Example**: `EXAMPLE_GENOME_ID_contigs`

### 2. Genes  
**CSV**: `genes.csv`
**Properties**:
- `id:ID` - Format: `gene:{gene_id}`
- `endCoordinate` - Integer
- `gcContent` - Float (0-1)
- `geneId` - Gene identifier
- `hasLocation` - Format: `:{start}-{end}`
- `lengthAA` - Amino acid length
- `lengthNt` - Nucleotide length  
- `startCoordinate` - Integer
- `strand` - 1 or -1

**Example**: `gene:PLM0_60_b1_sep16_scaffold_611_curated_25`

### 3. Proteins
**CSV**: `proteins.csv` 
**Properties**:
- `id:ID` - Format: `protein:{protein_id}`
- `length` - Integer
- `proteinId` - Protein identifier (same as gene_id)

**Example**: `protein:EXAMPLE_CONTIG_ID_2623_32`

### 4. Functional Annotations
**CSV**: `functionalannotations.csv`
**Properties**:
- `id:ID` - Format: `protein:{protein_id}/function/{kegg_id}`
- `bitscore` - Float
- `confidence` - String (high/medium/low)
- `evalue` - Scientific notation float

**Example**: `protein:EXAMPLE_CONTIG_ID_1705_107/function/K03466`

### 5. Domain Annotations
**CSV**: `domainannotations.csv`
**Properties**:
- `id:ID` - Format: `protein:{protein_id}/domain/{pfam_family}/{start}-{end}`
- `bitscore` - Float
- `domainEnd` - Integer
- `domainStart` - Integer  
- `evalue` - Scientific notation float

**Example**: `protein:RIFCSPHIGHO2_01_FULL_Acidovorax_64_960_rifcsphigho2_01_scaffold_14_241/domain/Thioredoxin_9/43-134`

### 6. KEGG Orthologs
**CSV**: `keggorthologs.csv`
**Properties**:
- `id:ID` - Format: `{kegg_id}` (e.g., K06142)
- `description` - Full description
- `ecNumber` - EC number (optional)
- `koId` - Same as ID
- `label` - Short label
- `profileType` - "all" 
- `scoreType` - "domain"
- `simplifiedDescription` - Simplified description
- `threshold` - Float threshold

### 7. PFAM Domains
**CSV**: `domains.csv`
**Properties**:
- `id:ID` - Format: `{pfam_accession}` (e.g., UvrD-helicase)
- `clan` - PFAM clan (optional)
- `description` - Full description
- `familyType` - "Domain" or "Family"
- `label` - Same as description
- `modelLength` - Integer
- `pfamAccession` - Same as ID

### 8. Pathways
**CSV**: `pathways.csv`
**Properties**:
- `id:ID` - Format: `map{pathway_number}` (e.g., map04115)
- `description` - KEGG pathway description
- `label` - KEGG pathway label
- `name` - Same as label
- `pathwayNumber` - 5-digit number (e.g., 04115)
- `pathwayType` - "map"

## Relationship Types

### 1. ENCODED_BY (proteins ← genes)
**CSV**: `encodedby_relationships.csv`
**Format**: `protein:{protein_id} ← gene:{gene_id}`

### 2. BELONGS_TO_GENOME (genes ← genomes)  
**CSV**: `belongstogenome_relationships.csv`
**Format**: `gene:{gene_id} ← {genome_id}`

### 3. HAS_DOMAIN (proteins → domain annotations)
**CSV**: `hasdomain_relationships.csv` 
**Format**: `protein:{protein_id} → protein:{protein_id}/domain/{pfam}/{coords}`

### 4. DOMAIN_FAMILY (domain annotations → PFAM domains)
**CSV**: `domainfamily_relationships.csv`
**Format**: `protein:{protein_id}/domain/{pfam}/{coords} → {pfam_accession}`

### 5. HAS_FUNCTION (proteins → KEGG orthologs)
**CSV**: `hasfunction_relationships.csv`
**Format**: `protein:{protein_id} → {kegg_id}`

### 6. ANNOTATES_PROTEIN (functional annotations → proteins)
**CSV**: `annotatesprotein_relationships.csv`
**Format**: `protein:{protein_id}/function/{kegg_id} → protein:{protein_id}`

### 7. ASSIGNED_FUNCTION (functional annotations → KEGG orthologs)
**CSV**: `assignedfunction_relationships.csv`
**Format**: `protein:{protein_id}/function/{kegg_id} → {kegg_id}`

### 8. BELONGS_TO_PROTEIN (domain annotations → proteins)
**CSV**: `belongstoprotein_relationships.csv`
**Format**: `protein:{protein_id}/domain/{pfam}/{coords} → protein:{protein_id}`

### 9. HAS_PARTICIPANT (pathways → KEGG orthologs)
**CSV**: `hasparticipant_relationships.csv`
**Format**: `map{pathway_id} → {kegg_id}`

### 10. PARTICIPATES_IN (KEGG orthologs → pathways)
**CSV**: `participatesin_relationships.csv`  
**Format**: `{kegg_id} → map{pathway_id}`

### 11. HAS_QUALITY_METRICS (genomes → quality metrics)
**CSV**: `hasqualitymetrics_relationships.csv`
**Format**: `{genome_id} → quality:{genome_id}`

## Key Patterns

1. **Gene-Protein Relationship**: 1:1 mapping with identical IDs except for prefix
2. **Functional Annotations**: Composite IDs linking proteins to KEGG functions
3. **Domain Annotations**: Composite IDs with protein, PFAM family, and coordinates
4. **Dual Relationships**: Both direct protein→function and via annotation nodes
5. **Pathway Integration**: Bidirectional KEGG function ↔ pathway relationships

## RDF Namespaces

- **kg**: `http://genome-kg.org/ontology/` - Core ontology
- **protein**: `http://genome-kg.org/proteins/` - Protein entities
- **gene**: `http://genome-kg.org/genes/` - Gene entities  
- **genome**: `http://genome-kg.org/genomes/` - Genome entities
- **pfam**: `http://pfam.xfam.org/family/` - PFAM domains
- **ko**: `http://www.genome.jp/kegg/ko/` - KEGG orthologs
