/* Input: $q: string, $limit: int */
// Resolve KO ids by anchor text (id/description match)
MATCH (ko:KEGGOrtholog)
WHERE toLower(ko.id) CONTAINS toLower($q)
   OR toLower(coalesce(ko.description, '')) CONTAINS toLower($q)
RETURN DISTINCT toLower(ko.id) AS ko_id
LIMIT $limit

