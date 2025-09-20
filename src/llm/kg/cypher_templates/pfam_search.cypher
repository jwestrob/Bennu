// Full-text search for PFAM domain families (domainText index)
CALL db.index.fulltext.queryNodes('domainText', $q) YIELD node AS d, score
RETURN d.id AS id, d.pfamAccession AS pfam, d.name AS name, d.description AS description
LIMIT $limit;
