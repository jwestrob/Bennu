# Python API

Recommended import path:

```
from src.llm.rag_system.core import GenomicRAG
```

The legacy shim `src.llm.rag_system` remains for backward compatibility but emits a deprecation warning on import.

## Quick Example

```
from src.llm.rag_system.core import GenomicRAG

rag = GenomicRAG()
answer = rag.answer(
    question="List genes flanking proteins with PFAM PF00005 in top 3 genomes",
    options={"output_profile": "summary"}
)
print(answer.text)
```

See Agents → Planner Guidance for how operators are scheduled and validated.

