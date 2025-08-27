/* Input: $q: string, $limit: int */
// Resolve PFAM families by anchor text (id/accession/description match)
MATCH (d:Domain)
WHERE toLower(d.id) CONTAINS toLower($q)
   OR (d.pfamAccession IS NOT NULL AND toLower(d.pfamAccession) CONTAINS toLower($q))
   OR toLower(coalesce(d.description, '')) CONTAINS toLower($q)
RETURN DISTINCT toLower(coalesce(d.pfamAccession, d.id)) AS pfam_id,
                toLower(d.id) AS id
LIMIT $limit

