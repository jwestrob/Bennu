// Comprehensive CAZyme Count and Verification Query
// This query will help diagnose why only 2 CAZymes are showing when there should be 1,846+

// 1. Count total CAZyme annotations in database
MATCH (ca:Cazymeannotation)
WITH count(ca) as total_cazyme_annotations
RETURN "Total CAZyme Annotations" as metric, total_cazyme_annotations as count

UNION ALL

// 2. Count unique proteins with CAZyme annotations
MATCH (p:Protein)-[:HASCAZYME]->(ca:Cazymeannotation)
WITH count(DISTINCT p) as proteins_with_cazymes
RETURN "Proteins with CAZymes" as metric, proteins_with_cazymes as count

UNION ALL

// 3. Count CAZyme families in database
MATCH (cf:Cazymefamily)
WITH count(cf) as total_cazyme_families
RETURN "Total CAZyme Families" as metric, total_cazyme_families as count

UNION ALL

// 4. Count HASCAZYME relationships
MATCH (p:Protein)-[r:HASCAZYME]->(ca:Cazymeannotation)
WITH count(r) as hascazyme_relationships
RETURN "HASCAZYME Relationships" as metric, hascazyme_relationships as count

UNION ALL

// 5. Count CAZYMEFAMILY relationships
MATCH (ca:Cazymeannotation)-[r:CAZYMEFAMILY]->(cf:Cazymefamily)
WITH count(r) as cazymefamily_relationships
RETURN "CAZYMEFAMILY Relationships" as metric, cazymefamily_relationships as count

UNION ALL

// 6. Count CAZymes by family type
MATCH (ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily)
WHERE cf.familyType = "GH"
WITH count(ca) as gh_count
RETURN "GH CAZymes" as metric, gh_count as count

UNION ALL

MATCH (ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily)
WHERE cf.familyType = "GT"
WITH count(ca) as gt_count
RETURN "GT CAZymes" as metric, gt_count as count

UNION ALL

MATCH (ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily)
WHERE cf.familyType = "PL"
WITH count(ca) as pl_count
RETURN "PL CAZymes" as metric, pl_count as count

UNION ALL

MATCH (ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily)
WHERE cf.familyType = "CE"
WITH count(ca) as ce_count
RETURN "CE CAZymes" as metric, ce_count as count

UNION ALL

MATCH (ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily)
WHERE cf.familyType = "AA"
WITH count(ca) as aa_count
RETURN "AA CAZymes" as metric, aa_count as count

UNION ALL

MATCH (ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily)
WHERE cf.familyType = "CBM"
WITH count(ca) as cbm_count
RETURN "CBM CAZymes" as metric, cbm_count as count

UNION ALL

// 7. Check for orphaned CAZyme annotations (no family relationship)
MATCH (ca:Cazymeannotation)
WHERE NOT (ca)-[:CAZYMEFAMILY]->()
WITH count(ca) as orphaned_cazymes
RETURN "Orphaned CAZymes (no family)" as metric, orphaned_cazymes as count

UNION ALL

// 8. Check for orphaned CAZyme families (no annotation relationship)
MATCH (cf:Cazymefamily)
WHERE NOT ()-[:CAZYMEFAMILY]->(cf)
WITH count(cf) as orphaned_families
RETURN "Orphaned Families (no annotations)" as metric, orphaned_families as count

ORDER BY metric;