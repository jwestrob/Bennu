// GECCO-predicted BGCs for a given genome (schema-tolerant, warning-light)
// Strategy:
//  - Avoid hard-typed rels that may not exist (e.g., PARTOFBGC)
//  - Use dynamic property access with parameterized key lists to avoid UnknownPropertyKey warnings
//  - Accept multiple BGC labels and rel-type spellings; filter via toLower(type(rel))

// Parameters:
//   $genome_id  : string (optional) — single genome filter
//   $genome_ids : [string] (optional) — multiple genome filter; empty or null means all
MATCH (gen:Genome)
WHERE ($genome_id IS NULL OR gen.id = $genome_id)
  AND ($genome_ids IS NULL OR size($genome_ids) = 0 OR gen.id IN $genome_ids)
CALL (gen) {
  MATCH (gen)<-[:BELONGSTOGENOME]-(gene:Gene)-[rbgc]->(bgc)
  WHERE any(l IN labels(bgc) WHERE toLower(l) IN ['bgc','biosyntheticgenecluster','bgcluster'])
    AND toLower(type(rbgc)) CONTAINS 'bgc'
  RETURN bgc
  UNION
  MATCH (gen)-[rel]->(bgc)
  WHERE any(l IN labels(bgc) WHERE toLower(l) IN ['bgc','biosyntheticgenecluster','bgcluster'])
    AND toLower(type(rel)) CONTAINS 'bgc'
  RETURN bgc
}
WITH gen, bgc,
     $id_keys        AS idks,
     $product_keys   AS prodks,
     $contig_keys    AS contks,
     $start_keys     AS startks,
     $end_keys       AS endks,
     $length_keys    AS lenks,
     $protein_keys   AS procks,
     $avg_prob_keys  AS avgks,
     $max_prob_keys  AS maxks

WITH gen, bgc,
     [k IN idks    WHERE k IN keys(bgc) AND bgc[k] IS NOT NULL][0] AS id_key,
     [k IN prodks  WHERE k IN keys(bgc) AND bgc[k] IS NOT NULL][0] AS prod_key,
     [k IN contks  WHERE k IN keys(bgc) AND bgc[k] IS NOT NULL][0] AS contig_key,
     [k IN startks WHERE k IN keys(bgc) AND bgc[k] IS NOT NULL][0] AS start_key,
     [k IN endks   WHERE k IN keys(bgc) AND bgc[k] IS NOT NULL][0] AS end_key,
     [k IN lenks   WHERE k IN keys(bgc) AND bgc[k] IS NOT NULL][0] AS len_key,
     [k IN procks  WHERE k IN keys(bgc) AND bgc[k] IS NOT NULL][0] AS protein_key,
     [k IN avgks   WHERE k IN keys(bgc) AND bgc[k] IS NOT NULL][0] AS avg_key,
     [k IN maxks   WHERE k IN keys(bgc) AND bgc[k] IS NOT NULL][0] AS max_key

WITH gen, bgc, id_key, prod_key, contig_key, start_key, end_key, len_key, protein_key, avg_key, max_key,
     CASE WHEN id_key     IS NULL THEN elementId(bgc) ELSE bgc[id_key]   END AS id_val,
     CASE WHEN prod_key   IS NULL THEN 'Unknown'     ELSE bgc[prod_key]  END AS product_val,
     CASE WHEN contig_key IS NULL THEN ''            ELSE bgc[contig_key] END AS contig_val,
     CASE WHEN start_key  IS NULL THEN null          ELSE bgc[start_key] END AS start_val,
     CASE WHEN end_key    IS NULL THEN null          ELSE bgc[end_key]   END AS end_val,
     CASE WHEN len_key    IS NULL THEN null          ELSE bgc[len_key]   END AS length_val,
     CASE WHEN protein_key IS NULL THEN null         ELSE bgc[protein_key] END AS protein_val,
     CASE WHEN avg_key    IS NULL THEN null          ELSE bgc[avg_key]   END AS avgp_val,
     CASE WHEN max_key    IS NULL THEN null          ELSE bgc[max_key]   END AS maxp_val

RETURN
  gen.id AS genome_id,
  coalesce(id_val, elementId(bgc)) AS bgc_id,
  coalesce(product_val, 'Unknown') AS bgc_product,
  coalesce(contig_val, '') AS contig,
  coalesce(toInteger(start_val), 0) AS startCoordinate,
  coalesce(toInteger(end_val), 0) AS endCoordinate,
  coalesce(toInteger(length_val), (coalesce(toInteger(end_val),0) - coalesce(toInteger(start_val),0) + 1)) AS lengthNt,
  coalesce(toInteger(protein_val), 0) AS proteinCount,
  coalesce(toFloat(avgp_val), 0.0) AS averageProbability,
  coalesce(toFloat(maxp_val), 0.0) AS maxProbability
ORDER BY maxProbability DESC, bgc_id
