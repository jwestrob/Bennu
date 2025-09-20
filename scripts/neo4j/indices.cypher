// Neo4j constraints and indexes for Bennu KG performance
// Safe to run multiple times (IF NOT EXISTS guards). Neo4j 5.x syntax.

// --- Uniqueness constraints on core IDs ---
CREATE CONSTRAINT genome_id IF NOT EXISTS FOR (g:Genome) REQUIRE g.id IS UNIQUE;
CREATE CONSTRAINT genome_genomeId IF NOT EXISTS FOR (g:Genome) REQUIRE g.genomeId IS UNIQUE;
CREATE CONSTRAINT gene_id IF NOT EXISTS FOR (g:Gene) REQUIRE g.id IS UNIQUE;
CREATE CONSTRAINT protein_id IF NOT EXISTS FOR (p:Protein) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT domain_id IF NOT EXISTS FOR (d:Domain) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT domain_annotation_id IF NOT EXISTS FOR (da:DomainAnnotation) REQUIRE da.id IS UNIQUE;
CREATE CONSTRAINT kegg_id IF NOT EXISTS FOR (k:KEGGOrtholog) REQUIRE k.id IS UNIQUE;
CREATE CONSTRAINT pathway_id IF NOT EXISTS FOR (pw:Pathway) REQUIRE pw.id IS UNIQUE;
CREATE CONSTRAINT bgc_id IF NOT EXISTS FOR (b:Bgc) REQUIRE b.id IS UNIQUE;

// --- Composite index for spatial gene scans ---
// Supports fast ORDER BY and range filtering during contig/coordinate traversals
CREATE INDEX gene_contig_coords IF NOT EXISTS FOR (g:Gene) ON (g.contig, g.startCoordinate, g.endCoordinate);

// --- Helpful single-property indexes (cheap, improves filters) ---
CREATE INDEX protein_name IF NOT EXISTS FOR (p:Protein) ON (p.name);
CREATE INDEX domain_name IF NOT EXISTS FOR (d:Domain) ON (d.name);
// Dedicated accession index used for PFxxxxx accession/prefix queries
CREATE INDEX domain_pfamAccession IF NOT EXISTS FOR (d:Domain) ON (d.pfamAccession);
CREATE INDEX kegg_desc IF NOT EXISTS FOR (k:KEGGOrtholog) ON (k.description);

// --- Full-text indexes for fuzzy search (optional) ---
// Use with CALL db.index.fulltext.queryNodes('indexName', 'query')
CREATE FULLTEXT INDEX proteinText IF NOT EXISTS FOR (p:Protein) ON EACH [p.name, p.description];
CREATE FULLTEXT INDEX domainText IF NOT EXISTS FOR (d:Domain) ON EACH [d.id, d.name, d.description];
CREATE FULLTEXT INDEX keggText IF NOT EXISTS FOR (k:KEGGOrtholog) ON EACH [k.id, k.description];
CREATE FULLTEXT INDEX pathwayText IF NOT EXISTS FOR (pw:Pathway) ON EACH [pw.id, pw.name, pw.description];

// Hints: In stubborn queries, consider USING INDEX on (g:Gene contig, startCoordinate)
// and return minimal columns first; fetch heavy fields lazily.
