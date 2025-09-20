# Vector Search Fix Implementation Plan

## Problem Analysis

The LanceDB vector search step is being skipped because it fails guard evaluation. The root cause is that the `vector_search` tool expects protein sequences/embeddings but only receives protein metadata from the database query results.

## Current Flow (Broken)
1. **Database Query**: Returns protein metadata (ID, KO, PFAM, coordinates) ✅
2. **Vector Search**: Expects protein sequences/embeddings ❌ → Guard fails → Step skipped

## Target Flow (Fixed)
1. **Database Query**: Returns protein metadata (ID, KO, PFAM, coordinates) ✅
2. **Vector Search**: 
   - Takes protein IDs from database results
   - Fetches corresponding embeddings from LanceDB
   - Performs similarity search against all embeddings
   - Returns similar proteins (excluding original integrases)
   - Filters results to non-integrase proteins as requested

## Implementation Strategy

### Option A: Fix Vector Search Tool (Recommended)
Modify the `vector_search` tool to:
1. **Extract protein IDs** from previous database query results
2. **Fetch embeddings** from LanceDB using those protein IDs
3. **Perform similarity search** using those embeddings as queries
4. **Filter results** to exclude integrase proteins (as per user request)
5. **Return enriched results** with both similarity scores and protein metadata

### Option B: Fix Database Query Tool
Modify database query to also fetch protein sequences, but this:
- Increases token usage massively (sequences are long)
- Doesn't solve the core vector search logic
- Less efficient

## Technical Implementation Plan

### Step 1: Locate Vector Search Tool
- File: `src/llm/rag_system/external_tools.py` or similar
- Function: `vector_search` or `_execute_vector_search`

### Step 2: Modify Vector Search Logic
```python
async def vector_search(args: Dict[str, Any], settings: Settings, preprocess_bundle=None) -> Dict[str, Any]:
    """
    Perform LanceDB similarity search using protein IDs from previous results.
    """
    # 1. Extract protein IDs from args or previous tool results
    protein_ids = extract_protein_ids_from_context(args)
    
    # 2. Connect to LanceDB
    lancedb_client = connect_to_lancedb(settings.lancedb_path)
    
    # 3. Fetch embeddings for query proteins
    query_embeddings = fetch_embeddings_by_ids(lancedb_client, protein_ids[:2])  # User wants "two integrase proteins"
    
    # 4. Perform similarity search for each query embedding
    all_results = []
    for query_embedding in query_embeddings:
        similar_proteins = lancedb_client.search(query_embedding).limit(50).to_list()
        all_results.extend(similar_proteins)
    
    # 5. Filter out integrase proteins (user wants non-integrases)
    filtered_results = filter_non_integrase_proteins(all_results)
    
    # 6. Enrich with Neo4j metadata for hypothetical proteins
    enriched_results = enrich_with_neo4j_metadata(filtered_results, settings)
    
    return {
        "summary": f"Vector similarity search found {len(enriched_results)} non-integrase proteins similar to query integrases",
        "artifacts": {
            "results": enriched_results,
            "query_protein_count": len(protein_ids),
            "similarity_threshold": 0.7  # or whatever threshold used
        }
    }
```

### Step 3: Implement Helper Functions
```python
def extract_protein_ids_from_context(args: Dict[str, Any]) -> List[str]:
    """Extract protein IDs from previous tool results or args."""
    # Check args for explicit protein IDs
    if 'protein_ids' in args:
        return args['protein_ids']
    
    # Extract from previous tool context (evidence ledger)
    # This requires access to the evidence ledger from previous steps
    pass

def filter_non_integrase_proteins(results: List[Dict]) -> List[Dict]:
    """Filter out proteins with integrase annotations."""
    non_integrases = []
    integrase_terms = ['integrase', 'recombinase', 'XerC', 'XerD', 'K03733', 'K04763', 'K14059']
    
    for result in results:
        # Check protein annotation/description
        description = result.get('description', '').lower()
        if not any(term.lower() in description for term in integrase_terms):
            non_integrases.append(result)
    
    return non_integrases

def enrich_with_neo4j_metadata(lancedb_results: List[Dict], settings: Settings) -> List[Dict]:
    """Enrich LanceDB results with Neo4j metadata for genomic context."""
    # Connect to Neo4j and fetch additional metadata for each protein
    # Including genomic neighborhood information for hypothetical proteins
    pass
```

### Step 4: Fix Guard Logic
The vector search step guard likely checks for:
- `requires_protein_data`: Protein IDs or sequences available
- `requires_lancedb`: LanceDB connection available

Update the guard to accept protein IDs from database results, not just raw sequences.

### Step 5: Integration Points
- **Input**: Protein IDs from database_query tool results
- **Processing**: LanceDB similarity search with filtering
- **Output**: Non-integrase similar proteins with genomic context
- **Next Step**: Genomic neighborhood analysis for hypothetical proteins

## Expected User Experience Improvement

### Before (Current)
```
🔧 Executing step: vector_search (cost: moderate)
✅ No more eligible steps, execution complete
```
*Vector search skipped, no similarity analysis*

### After (Fixed)
```
🔧 Executing step: vector_search (cost: moderate)
🔍 Vector search: Using 2 integrase proteins as queries
🎯 Found 45 non-integrase proteins with similarity > 0.7
📊 Top hits include hypothetical proteins on scaffolds X, Y, Z
✅ Vector search complete
```

## Testing Strategy

1. **Unit Test**: Test vector search with known protein IDs
2. **Integration Test**: Full pipeline from database query → vector search
3. **User Query Test**: Run the exact failing query to verify fix

## Files to Modify

1. **Vector Search Tool**: `src/llm/rag_system/external_tools.py` or similar
2. **Guard Logic**: Policy engine or tool registry guards
3. **Tool Integration**: Agent executor tool parameter passing

## Success Criteria

✅ Vector search step executes instead of being skipped
✅ LanceDB similarity search performed with integrase queries
✅ Non-integrase similar proteins returned (excluding integrases)
✅ Hypothetical proteins identified with genomic context
✅ Rich synthesis includes similarity search results

This fix will complete the complex integrase query end-to-end and demonstrate the full power of the schema-locked system with vector similarity search integration.