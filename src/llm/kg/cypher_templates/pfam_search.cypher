// Find PFAM domain families matching a substring (case-insensitive)
MATCH (d:Domain)
WHERE toLower(d.id) CONTAINS toLower($q)
   OR toLower(d.pfamAccession) CONTAINS toLower($q)
   OR toLower(d.description) CONTAINS toLower($q)
RETURN d.id AS id, d.pfamAccession AS pfam, d.name AS name, d.description AS description
LIMIT $limit;

