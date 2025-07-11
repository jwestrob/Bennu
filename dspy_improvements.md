# DSPy Query Generation Improvements

## Non-Hard-Coded Strategies for Better Biological Query Generation

### 1. Enhanced ContextRetriever Prompt Additions

```python
"""
BIOLOGICAL SEARCH STRATEGIES - GENERAL PRINCIPLES:

MULTI-DATABASE APPROACH:
- Always consider multiple annotation sources (KEGG, PFAM, CAZyme)
- Use HYBRID queries combining function and domain information
- Search both exact terms and related biological concepts

FLEXIBLE KEYWORD STRATEGIES:
- Break compound terms: "TonB transporter" → search "TonB" AND "transport" separately
- Include biological synonyms and related terms automatically
- Use pattern matching for protein families (e.g., ABC, TonB, Two-component)

DOMAIN-FIRST APPROACH:
- Prioritize PFAM domain searches for protein families
- Use KEGG for metabolic pathways and specific functions  
- Combine both for comprehensive coverage

PROGRESSIVE SEARCH STRATEGY:
1. Try exact term matching first
2. If no results, break into component terms
3. Search related biological concepts
4. Use domain/family-based queries as fallback

AVOID OVERLY SPECIFIC STRING MATCHING:
- Don't require exact phrase matches for compound terms
- Use CONTAINS with individual keywords rather than full phrases
- Include common biological abbreviations and variations
"""
```

### 2. Query Expansion Logic

```python
"""
QUERY EXPANSION LOGIC:

For any protein/transport query:
- Extract key biological terms (transport, kinase, synthase, etc.)
- Search each term independently in both KEGG and PFAM
- Combine results rather than requiring exact phrase matches

EXAMPLE TRANSFORMATION:
User: "TonB transporters"
Bad: WHERE description CONTAINS 'TonB transporter'  
Good: WHERE description CONTAINS 'TonB' OR description CONTAINS 'transport'
      OR domain.description CONTAINS 'TonB' OR domain.description CONTAINS 'receptor'
"""
```

### 3. Enhanced Biological Reasoning

```python
"""
BIOLOGICAL REASONING FOR QUERIES:

Think about what the user REALLY wants:
- "TonB transporters" = proteins involved in TonB-dependent transport
- Could be annotated as: TonB, outer membrane receptor, iron transport, siderophore
- Search strategy: Look for ANY of these terms, not just exact phrase

GENERAL APPROACH:
1. Identify the biological concept (transport, metabolism, regulation)
2. Consider multiple ways it might be annotated
3. Search broadly across annotation databases
4. Use OR logic for related terms, AND logic for required concepts
"""
```

### 4. Key Principles

- Compound biological terms often need to be split
- Protein families have multiple naming conventions  
- Domain annotations are often more reliable than function annotations
- Different databases use different terminology for the same concept
- Teach general biological reasoning instead of hard-coding specific cases