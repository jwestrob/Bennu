// Sample CAZyme Entries with Detailed Information
// This query shows examples of CAZyme entries and their relationships

// Show sample CAZyme entries with full details
MATCH (p:Protein)-[:HASCAZYME]->(ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily)
RETURN 
    p.id as protein_id,
    ca.domainStart as domain_start,
    ca.domainEnd as domain_end,
    ca.eValue as e_value,
    ca.substrateSpecificity as substrate_specificity,
    cf.familyId as family_id,
    cf.familyType as family_type,
    cf.description as family_description
ORDER BY cf.familyType, cf.familyId, p.id
LIMIT 50;

// Alternative query pattern - check if relationship names are different
MATCH (p:Protein)-[r1]->(ca:Cazymeannotation)-[r2]->(cf:Cazymefamily)
RETURN 
    type(r1) as protein_to_cazyme_relationship,
    type(r2) as cazyme_to_family_relationship,
    p.id as protein_id,
    ca.domainStart as domain_start,
    cf.familyId as family_id,
    cf.familyType as family_type
LIMIT 20;

// Check all relationship types involving CAZyme nodes
MATCH (ca:Cazymeannotation)-[r]->(node)
RETURN DISTINCT type(r) as relationship_type, labels(node) as target_node_labels
LIMIT 20;

// Check all relationship types targeting CAZyme nodes
MATCH (node)-[r]->(ca:Cazymeannotation)
RETURN DISTINCT labels(node) as source_node_labels, type(r) as relationship_type
LIMIT 20;