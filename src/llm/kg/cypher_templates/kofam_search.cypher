// Find KEGG Orthologs matching a substring (case-insensitive)
MATCH (ko:KEGGOrtholog)
WHERE toLower(ko.id) CONTAINS toLower($q)
   OR toLower(ko.description) CONTAINS toLower($q)
RETURN ko.id AS id, ko.description AS description
LIMIT $limit;

