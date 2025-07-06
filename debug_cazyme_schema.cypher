// Debug CAZyme Schema and Relationship Issues
// This query helps identify potential schema or naming issues

// 1. Check all node labels in the database
CALL db.labels() YIELD label
WHERE label CONTAINS "azyme" OR label CONTAINS "AZYME"
RETURN label as node_label
ORDER BY label;

// 2. Check all relationship types in the database
CALL db.relationshipTypes() YIELD relationshipType
WHERE relationshipType CONTAINS "CAZYME" OR relationshipType CONTAINS "cazyme"
RETURN relationshipType as relationship_type
ORDER BY relationshipType;

// 3. Check for case sensitivity issues with CAZyme nodes
MATCH (n)
WHERE any(label in labels(n) WHERE label CONTAINS "azyme")
RETURN DISTINCT labels(n) as node_labels, count(n) as node_count
ORDER BY node_labels;

// 4. Check for case sensitivity issues with CAZyme relationships
MATCH ()-[r]->()
WHERE type(r) CONTAINS "CAZYME" OR type(r) CONTAINS "cazyme"
RETURN DISTINCT type(r) as relationship_type, count(r) as relationship_count
ORDER BY relationship_type;

// 5. Sample nodes to check actual structure
MATCH (n)
WHERE any(label in labels(n) WHERE label CONTAINS "azyme")
RETURN labels(n) as node_labels, keys(n) as node_properties, n
LIMIT 10;

// 6. Check if there are any CAZyme-related nodes with different capitalization
MATCH (n)
WHERE any(label in labels(n) WHERE 
    label =~ "(?i).*cazyme.*" OR 
    label =~ "(?i).*cazym.*" OR 
    label =~ "(?i).*cazy.*"
)
RETURN DISTINCT labels(n) as found_labels, count(n) as count
ORDER BY found_labels;