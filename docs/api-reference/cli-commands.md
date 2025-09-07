# CLI Reference

Entry point: `python -m src.cli`

## build

Builds the pipeline across stages. Heavy modules are lazily imported only when needed.

```
python -m src.cli build [OPTIONS]

Options:
  -i, --input-dir PATH         Input directory (default: data/raw)
  -o, --output-dir PATH        Output directory (default: data)
  -f, --from-stage INTEGER     Start stage (0-8; default: 0)
  -t, --to-stage INTEGER       End stage (0-8; default: 8)
  -j, --threads INTEGER        Threads per task (default: SYSTEM_JOBS or CPU cores)
      --skip-tax               Skip DFAST_QC taxonomy (Stage 2)
      --force                  Overwrite existing outputs
      --engine [docker|system] Neo4j engine for Stage 07 import (default: docker)
```

Notes:
- Stage 06 forwards `--threads` to dbCAN via `--threads` per job; DIAMOND uses the specified value.
- Stage 07 performs RDF→CSV conversion, precomputes `[:NEXT]` and gene degrees in CSVs, then runs `neo4j-admin` bulk import. No auth required on Docker engine.

## ask

Ask a natural language question over the knowledge graph with the modular RAG system.

```
python -m src.cli ask "<question>" [OPTIONS]

Options:
  -o, --output PATH            Output file for JSON answer
      --planner TEXT           Override Planner model id
      --irb TEXT               Override IRB Editor model id
      --reporter TEXT          Override Reporter model id
  -c, --config PATH            Path to config file
      --neo4j-password TEXT    Override Neo4j password
  -v, --verbose                Print detailed reasoning/metadata
```

## version

Prints the CLI version.

