"""
Helper functions for comprehensive protein discovery.
These will be added to annotation_tools.py in smaller chunks.
"""

async def _discover_kegg_proteins(neo4j, kegg_annotations, batch_size, stats):
    """Discover proteins for KEGG orthologs using intelligent batching."""
    all_proteins = []
    
    # Create batches of KO IDs
    ko_ids = [ann['id'] for ann in kegg_annotations]
    
    for i in range(0, len(ko_ids), batch_size):
        batch_kos = ko_ids[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(ko_ids) + batch_size - 1) // batch_size
        
        logger.info(f"🔍 KEGG Batch {batch_num}/{total_batches}: Searching {len(batch_kos)} orthologs")
        
        try:
            # Comprehensive query for this batch - no artificial limits
            query = f"""
            MATCH (ko:KEGGOrtholog) 
            WHERE ko.id IN {batch_kos}
            MATCH (p:Protein)-[:HASFUNCTION]->(ko)
            OPTIONAL MATCH (p)-[:ENCODEDBY]->(g:Gene)
            OPTIONAL MATCH (p)-[:HASDOMAIN]->(da:DomainAnnotation)-[:DOMAINFAMILY]->(dom:Domain)
            RETURN p.id AS protein_id, 
                   ko.id AS ko_id, 
                   ko.description AS ko_description,
                   g.startCoordinate AS start_coordinate, 
                   g.endCoordinate AS end_coordinate, 
                   g.strand AS strand,
                   collect(DISTINCT dom.id) AS pfam_accessions,
                   collect(DISTINCT dom.description) AS pfam_descriptions
            ORDER BY ko.id, p.id
            """
            
            result = await neo4j.process_query(query, query_type="cypher")
            batch_proteins = result.results
            
            # Track statistics
            stats['total_batches'] += 1
            stats['proteins_per_batch'].append(len(batch_proteins))
            
            # Track KO coverage
            for protein in batch_proteins:
                ko_id = protein['ko_id']
                if ko_id not in stats['ko_coverage']:
                    stats['ko_coverage'][ko_id] = 0
                stats['ko_coverage'][ko_id] += 1
                stats['unique_proteins'].add(protein['protein_id'])
            
            all_proteins.extend(batch_proteins)
            logger.info(f"✅ KEGG Batch {batch_num}: Found {len(batch_proteins)} proteins")
            
        except Exception as e:
            logger.error(f"❌ KEGG Batch {batch_num} failed: {e}")
            stats['failed_batches'].append(f"KEGG_batch_{batch_num}")
    
    return all_proteins


async def _discover_pfam_proteins(neo4j, pfam_annotations, batch_size, stats):
    """Discover proteins for PFAM domains using intelligent batching."""
    all_proteins = []
    
    # Create batches of PFAM IDs
    pfam_ids = [ann['id'] for ann in pfam_annotations]
    
    for i in range(0, len(pfam_ids), batch_size):
        batch_pfams = pfam_ids[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(pfam_ids) + batch_size - 1) // batch_size
        
        logger.info(f"🔍 PFAM Batch {batch_num}/{total_batches}: Searching {len(batch_pfams)} domains")
        
        try:
            # Comprehensive query for PFAM domains
            query = f"""
            MATCH (dom:Domain) 
            WHERE dom.id IN {batch_pfams}
            MATCH (p:Protein)-[:HASDOMAIN]->(da:DomainAnnotation)-[:DOMAINFAMILY]->(dom)
            OPTIONAL MATCH (p)-[:ENCODEDBY]->(g:Gene)
            OPTIONAL MATCH (p)-[:HASFUNCTION]->(ko:KEGGOrtholog)
            RETURN p.id AS protein_id,
                   dom.id AS pfam_id,
                   dom.description AS pfam_description,
                   g.startCoordinate AS start_coordinate,
                   g.endCoordinate AS end_coordinate,
                   g.strand AS strand,
                   collect(DISTINCT ko.id) AS ko_ids,
                   collect(DISTINCT ko.description) AS ko_descriptions
            ORDER BY dom.id, p.id
            """
            
            result = await neo4j.process_query(query, query_type="cypher")
            batch_proteins = result.results
            
            # Track statistics
            stats['total_batches'] += 1
            stats['proteins_per_batch'].append(len(batch_proteins))
            
            # Track PFAM coverage
            for protein in batch_proteins:
                pfam_id = protein['pfam_id']
                if pfam_id not in stats['pfam_coverage']:
                    stats['pfam_coverage'][pfam_id] = 0
                stats['pfam_coverage'][pfam_id] += 1
                stats['unique_proteins'].add(protein['protein_id'])
            
            all_proteins.extend(batch_proteins)
            logger.info(f"✅ PFAM Batch {batch_num}: Found {len(batch_proteins)} proteins")
            
        except Exception as e:
            logger.error(f"❌ PFAM Batch {batch_num} failed: {e}")
            stats['failed_batches'].append(f"PFAM_batch_{batch_num}")
    
    return all_proteins


def _deduplicate_proteins(proteins):
    """Deduplicate proteins while preserving all annotation information."""
    protein_map = {}
    
    for protein in proteins:
        protein_id = protein['protein_id']
        
        if protein_id not in protein_map:
            # First occurrence - initialize
            protein_map[protein_id] = {
                'protein_id': protein_id,
                'start_coordinate': protein.get('start_coordinate'),
                'end_coordinate': protein.get('end_coordinate'),
                'strand': protein.get('strand'),
                'ko_annotations': set(),
                'pfam_annotations': set(),
                'ko_descriptions': set(),
                'pfam_descriptions': set()
            }
        
        # Accumulate annotations
        entry = protein_map[protein_id]
        
        if protein.get('ko_id'):
            entry['ko_annotations'].add(protein['ko_id'])
            if protein.get('ko_description'):
                entry['ko_descriptions'].add(protein['ko_description'])
        
        if protein.get('pfam_id'):
            entry['pfam_annotations'].add(protein['pfam_id'])
            if protein.get('pfam_description'):
                entry['pfam_descriptions'].add(protein['pfam_description'])
        
        # Handle list fields from PFAM queries
        if protein.get('ko_ids'):
            entry['ko_annotations'].update(protein['ko_ids'])
        if protein.get('ko_descriptions'):
            entry['ko_descriptions'].update(protein['ko_descriptions'])
        if protein.get('pfam_accessions'):
            entry['pfam_annotations'].update(protein['pfam_accessions'])
        if protein.get('pfam_descriptions'):
            entry['pfam_descriptions'].update(protein['pfam_descriptions'])
    
    # Convert sets back to lists for JSON serialization
    unique_proteins = []
    for protein_id, data in protein_map.items():
        unique_proteins.append({
            'protein_id': protein_id,
            'start_coordinate': data['start_coordinate'],
            'end_coordinate': data['end_coordinate'],
            'strand': data['strand'],
            'ko_annotations': list(data['ko_annotations']),
            'pfam_annotations': list(data['pfam_annotations']),
            'ko_descriptions': list(data['ko_descriptions']),
            'pfam_descriptions': list(data['pfam_descriptions']),
            'total_annotations': len(data['ko_annotations']) + len(data['pfam_annotations'])
        })
    
    return unique_proteins


def _generate_discovery_statistics(proteins, batch_stats, functional_category):
    """Generate comprehensive discovery statistics."""
    total_proteins = len(proteins)
    total_ko_annotations = len(batch_stats['ko_coverage'])
    total_pfam_annotations = len(batch_stats['pfam_coverage'])
    total_annotations = total_ko_annotations + total_pfam_annotations
    
    # Calculate coverage statistics
    ko_protein_counts = list(batch_stats['ko_coverage'].values())
    pfam_protein_counts = list(batch_stats['pfam_coverage'].values())
    
    return {
        'total_proteins_discovered': total_proteins,
        'annotations_covered': total_annotations,
        'kegg_orthologs_covered': total_ko_annotations,
        'pfam_domains_covered': total_pfam_annotations,
        'avg_proteins_per_annotation': total_proteins / total_annotations if total_annotations > 0 else 0,
        'avg_proteins_per_ko': sum(ko_protein_counts) / len(ko_protein_counts) if ko_protein_counts else 0,
        'avg_proteins_per_pfam': sum(pfam_protein_counts) / len(pfam_protein_counts) if pfam_protein_counts else 0,
        'batch_performance': {
            'total_batches_executed': batch_stats['total_batches'],
            'failed_batches': len(batch_stats['failed_batches']),
            'success_rate': (batch_stats['total_batches'] - len(batch_stats['failed_batches'])) / batch_stats['total_batches'] if batch_stats['total_batches'] > 0 else 0,
            'avg_proteins_per_batch': sum(batch_stats['proteins_per_batch']) / len(batch_stats['proteins_per_batch']) if batch_stats['proteins_per_batch'] else 0
        },
        'functional_category': functional_category
    }


def _generate_biological_insights(proteins, functional_category):
    """Generate biological insights from discovered proteins."""
    # Analyze annotation patterns
    ko_frequency = {}
    pfam_frequency = {}
    multi_functional_proteins = 0
    
    for protein in proteins:
        # Count KO annotations
        for ko in protein.get('ko_annotations', []):
            ko_frequency[ko] = ko_frequency.get(ko, 0) + 1
        
        # Count PFAM annotations
        for pfam in protein.get('pfam_annotations', []):
            pfam_frequency[pfam] = pfam_frequency.get(pfam, 0) + 1
        
        # Count multi-functional proteins
        if protein.get('total_annotations', 0) > 1:
            multi_functional_proteins += 1
    
    # Find most common annotations
    top_kos = sorted(ko_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
    top_pfams = sorted(pfam_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        'most_common_functions': {
            'kegg_orthologs': top_kos,
            'pfam_domains': top_pfams
        },
        'functional_diversity': {
            'unique_ko_functions': len(ko_frequency),
            'unique_pfam_domains': len(pfam_frequency),
            'multi_functional_proteins': multi_functional_proteins,
            'multi_functional_percentage': (multi_functional_proteins / len(proteins) * 100) if proteins else 0
        },
        'biological_interpretation': _interpret_functional_category(functional_category, len(proteins), len(ko_frequency))
    }


def _interpret_functional_category(category, protein_count, function_count):
    """Provide biological interpretation of discovery results."""
    if category.lower() == 'central_metabolism':
        if protein_count < 20:
            return f"Limited central metabolism representation ({protein_count} proteins, {function_count} functions). May indicate specialized metabolism or incomplete genome."
        elif protein_count < 50:
            return f"Moderate central metabolism coverage ({protein_count} proteins, {function_count} functions). Typical for streamlined genomes."
        else:
            return f"Comprehensive central metabolism ({protein_count} proteins, {function_count} functions). Indicates metabolically versatile organisms."
    
    elif category.lower() == 'transport':
        if protein_count < 10:
            return f"Limited transport systems ({protein_count} proteins). May indicate specialized niche or symbiotic lifestyle."
        else:
            return f"Diverse transport capabilities ({protein_count} proteins, {function_count} functions). Indicates environmental adaptability."
    
    else:
        return f"Discovered {protein_count} proteins with {function_count} distinct functions for {category}."