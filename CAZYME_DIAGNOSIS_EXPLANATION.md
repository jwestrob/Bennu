# CAZyme Count Diagnosis and Troubleshooting

## Problem Statement
The current analysis only shows 2 CAZymes when the CSV files indicate there should be 1,846 CAZyme relationships in the database.

## Current Query Pattern (Potentially Incorrect)
```cypher
MATCH (p:Protein)-[:HASCAZYME]->(ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily) 
RETURN p.id, cf.familyId, ca.substrateSpecificity, cf.familyType 
LIMIT 20
```

## Analysis from CSV Files

### 1. Data Volume in CSV Files
- **hascazyme_relationships.csv**: 1,846 lines (1,845 relationships + header)
- **cazymeannotations.csv**: Contains CAZyme annotation nodes
- **cazymefamily_relationships.csv**: Contains family relationship mappings

### 2. CSV Structure Analysis

#### CAZyme Annotations (cazymeannotations.csv)
```
id:ID,annotationId,cazymeType,coverage,endPosition,evalue,hmmLength,startPosition,substrateSpecificity
cazyme:PLM0_60_b1_sep16_scaffold_7257_curated_2_PL1_1_724,PLM0_60_b1_sep16_scaffold_7257_curated_2_PL1_1_724,PL,0.8,0,1e-16,0,0,
```

#### Protein-CAZyme Relationships (hascazyme_relationships.csv)
```
:START_ID,:END_ID
protein:PLM0_60_b1_sep16_scaffold_31049_curated_4,cazyme:PLM0_60_b1_sep16_scaffold_31049_curated_4_GT2_586
```

#### CAZyme-Family Relationships (cazymefamily_relationships.csv)
```
:START_ID,:END_ID
cazyme:RIFCSPLOWO2_01_FULL_OD1_41_220_rifcsplowo2_01_scaffold_1705_98_GT1_284,cazyme:family_GT1
```

## Potential Issues

### 1. **Node Label Case Sensitivity**
- CSV shows `cazyme:` prefix, suggesting node labels might be:
  - `Cazyme` instead of `Cazymeannotation`
  - Case sensitivity issues with Neo4j import

### 2. **CAZyme Family Structure**
- CSV shows families as `cazyme:family_GT1`, `cazyme:family_CBM50`, etc.
- This suggests families might be stored differently than expected
- May not have separate `Cazymefamily` nodes

### 3. **Relationship Naming**
- The relationship from CAZyme to family might not be `[:CAZYMEFAMILY]`
- Could be unnamed relationship or different name

### 4. **Import Process Issues**
- Neo4j CSV import might have failed partially
- Node labels might not have been applied correctly
- Data might be imported but not properly indexed

## Diagnostic Steps

### Step 1: Run Basic Structure Query
Execute `corrected_cazyme_count.cypher` to identify:
- Actual node labels in the database
- Actual relationship types
- Count of nodes by pattern

### Step 2: Run Detailed Diagnosis
Execute `cazyme_structure_diagnosis.cypher` to:
- Find exact node structure
- Identify relationship patterns
- Sample actual data

### Step 3: Run Verification Count
Execute `verify_cazyme_count.cypher` to:
- Count total nodes and relationships
- Verify data integrity
- Check for orphaned nodes

## Expected Corrections

Based on CSV analysis, the correct query might be:
```cypher
// Option 1: If nodes are labeled as 'Cazyme'
MATCH (p:Protein)-[:HASCAZYME]->(ca:Cazyme)-[r]->(cf)
WHERE cf.id STARTS WITH "cazyme:family_"
RETURN p.id, cf.id, ca.substrateSpecificity, ca.cazymeType
LIMIT 20

// Option 2: If family relationships are different
MATCH (p:Protein)-[:HASCAZYME]->(ca)
WHERE any(label in labels(ca) WHERE toLower(label) CONTAINS "cazyme")
RETURN p.id, ca.id, ca.substrateSpecificity, ca.cazymeType
LIMIT 20
```

## Next Steps
1. Execute the diagnostic queries in order
2. Based on results, create the corrected counting query
3. Verify the total count matches the expected 1,845 relationships
4. Update the analysis code with the correct query pattern