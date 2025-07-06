// Corrected Comprehensive CAZyme Count Query
// Based on CSV analysis: cazyme: prefix suggests different node structure

// 1. Check all actual node labels containing "cazyme" (case insensitive)
CALL db.labels() YIELD label
WHERE toLower(label) CONTAINS "cazyme"
RETURN "Node Label" as type, label as name, "found" as status

UNION ALL

// 2. Check all relationship types containing "cazyme" (case insensitive)
CALL db.relationshipTypes() YIELD relationshipType
WHERE toLower(relationshipType) CONTAINS "cazyme"
RETURN "Relationship Type" as type, relationshipType as name, "found" as status

UNION ALL

// 3. Try different possible node label combinations
MATCH (n)
WHERE any(label in labels(n) WHERE toLower(label) CONTAINS "cazyme")
WITH labels(n) as node_labels, count(n) as node_count
RETURN "Node Count" as type, toString(node_labels) as name, toString(node_count) as status

UNION ALL

// 4. Count nodes by common CAZyme-related patterns
MATCH (n)
WHERE any(label in labels(n) WHERE 
    label IN ["Cazyme", "CazymeAnnotation", "Cazymeannotation", "CazymeFamily", "Cazymefamily"]
)
WITH labels(n) as node_labels, count(n) as node_count
RETURN "Pattern Match" as type, toString(node_labels) as name, toString(node_count) as status

UNION ALL

// 5. Try the corrected query patterns based on CSV structure
MATCH (p:Protein)-[:HASCAZYME]->(ca)
WHERE any(label in labels(ca) WHERE toLower(label) CONTAINS "cazyme")
WITH labels(ca) as cazyme_labels, count(ca) as cazyme_count
RETURN "HASCAZYME Relations" as type, toString(cazyme_labels) as name, toString(cazyme_count) as status

UNION ALL

// 6. Check for nodes with 'cazyme' in their ID (based on CSV pattern)
MATCH (n)
WHERE n.id CONTAINS "cazyme:" OR n.id STARTS WITH "cazyme:"
WITH labels(n) as node_labels, count(n) as node_count
RETURN "ID Pattern Match" as type, toString(node_labels) as name, toString(node_count) as status

UNION ALL

// 7. Check for family nodes (based on CSV pattern showing family_GT1, etc.)
MATCH (n)
WHERE n.id CONTAINS "family_" OR n.id STARTS WITH "cazyme:family_"
WITH labels(n) as node_labels, count(n) as node_count
RETURN "Family Pattern Match" as type, toString(node_labels) as name, toString(node_count) as status

ORDER BY type, name;