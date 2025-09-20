// Fast, non-destructive load of NEXT relationships from CSV
// Requires next_relationships.csv to be placed in Neo4j's import/ directory.
// Usage:
//   cp data/stage07_kg/csv/next_relationships.csv $NEO4J_HOME/import/
//   cypher-shell -u neo4j -p $NEO4J_PASSWORD -f scripts/neo4j/load_next_from_csv.cypher

CALL {
  WITH 'file:///next_relationships.csv' AS url
  LOAD CSV WITH HEADERS FROM url AS row
  WITH row
  MATCH (a:Gene {id: row.`:START_ID`})
  MATCH (b:Gene {id: row.`:END_ID`})
  MERGE (a)-[r:NEXT]->(b)
  SET r.contig = row.contig,
      r.delta  = toInteger(row.`delta:long`),
      r.same_strand = toLower(row.`same_strand:boolean`) = 'true'
} IN TRANSACTIONS OF 10000 ROWS;
