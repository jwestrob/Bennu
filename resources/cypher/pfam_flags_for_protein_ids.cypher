/*
Input:
  $protein_ids        : [string]  -- stable protein IDs used in LanceDB
  $exclude_needle     : string    -- lowercase 'integrase' or similar (may be empty)
  $exclude_markers    : [string]  -- accessions/IDs such as ['pf00589', ...]
  $include_needle     : string    -- lowercase include text (may be empty)
  $include_markers    : [string]  -- accessions/IDs to require (may be empty)
Output:
  protein_id, is_marker : bool, matches_include : bool
*/
UNWIND $protein_ids AS pid
MATCH (p:Protein)
WHERE p.id = pid OR p.proteinId = pid
// Follow schema: Protein -[:HASDOMAIN]-> DomainAnnotation -[:DOMAINFAMILY]-> Domain
OPTIONAL MATCH (p)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)
WITH pid,
     collect(toLower(coalesce(d.description, ''))) AS descs,
     collect(toLower(coalesce(d.id, '')))         AS ids,
     collect(toLower(coalesce(d.pfamAccession, ''))) AS accs,
     toLower(coalesce($exclude_needle, '')) AS ex_need,
     toLower(coalesce($include_needle, '')) AS in_need,
     coalesce($exclude_markers, []) AS ex_marks,
     coalesce($include_markers, []) AS in_marks
RETURN pid AS protein_id,
       (
         (size(ex_need) > 0 AND ANY(x IN descs WHERE x CONTAINS ex_need)) OR
         (size(ex_marks) > 0 AND (ANY(x IN ids WHERE x IN ex_marks) OR ANY(x IN accs WHERE x IN ex_marks)))
       ) AS is_marker,
       (
         (size(in_need) > 0 AND ANY(x IN descs WHERE x CONTAINS in_need)) OR
         (size(in_marks) > 0 AND (ANY(x IN ids WHERE x IN in_marks) OR ANY(x IN accs WHERE x IN in_marks)))
       ) AS matches_include
