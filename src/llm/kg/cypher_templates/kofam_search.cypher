// Full-text search for KO terms (keggText index)
CALL db.index.fulltext.queryNodes('keggText', $q) YIELD node AS ko, score
RETURN ko.id AS id, ko.description AS description
LIMIT $limit;
