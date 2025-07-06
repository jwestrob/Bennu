// CAZyme Structure Diagnosis Query
// This will help determine the exact node labels and relationship structure

// Part 1: Find any nodes that might be CAZyme-related
MATCH (n)
WHERE 
  any(label in labels(n) WHERE toLower(label) CONTAINS "cazyme") OR
  any(label in labels(n) WHERE toLower(label) CONTAINS "cazy") OR
  (exists(n.id) AND (n.id CONTAINS "cazyme" OR n.id CONTAINS "family_"))
RETURN 
  labels(n) as node_labels,
  n.id as node_id,
  keys(n) as properties,
  count(*) as count
ORDER BY node_labels, count DESC
LIMIT 20;

// Part 2: Find relationships involving potential CAZyme nodes
MATCH (source)-[r]->(target)
WHERE 
  any(label in labels(source) WHERE toLower(label) CONTAINS "cazyme") OR
  any(label in labels(target) WHERE toLower(label) CONTAINS "cazyme") OR
  (exists(source.id) AND source.id CONTAINS "cazyme") OR
  (exists(target.id) AND target.id CONTAINS "cazyme")
RETURN DISTINCT
  labels(source) as source_labels,
  type(r) as relationship_type,
  labels(target) as target_labels,
  count(*) as relationship_count
ORDER BY relationship_count DESC
LIMIT 20;

// Part 3: Check for nodes with family patterns
MATCH (n)
WHERE exists(n.id) AND n.id CONTAINS "family_"
RETURN 
  labels(n) as node_labels,
  n.id as node_id,
  keys(n) as properties,
  count(*) as count
ORDER BY count DESC
LIMIT 20;

// Part 4: Sample actual nodes to see their structure
MATCH (n)
WHERE any(label in labels(n) WHERE toLower(label) CONTAINS "cazyme")
RETURN 
  labels(n) as node_labels,
  n.id as node_id,
  n
LIMIT 5;

// Part 5: Check if the issue is with the relationship names
MATCH (p:Protein)-[r]->(n)
WHERE any(label in labels(n) WHERE toLower(label) CONTAINS "cazyme")
RETURN DISTINCT
  type(r) as relationship_from_protein,
  labels(n) as target_node_labels,
  count(*) as count
ORDER BY count DESC;