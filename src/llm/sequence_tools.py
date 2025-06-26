#!/usr/bin/env python3
"""
Sequence visualization tools for LLM-powered protein analysis.
Fetches and formats protein sequences optimally for biological insights.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import sequence database and Neo4j query processor
import sys
sys.path.append(str(Path(__file__).parent.parent))
from build_kg.sequence_db import SequenceDatabase
from llm.query_processor import Neo4jQueryProcessor
from llm.config import LLMConfig

logger = logging.getLogger(__name__)

async def sequence_viewer(
    protein_ids: List[str], 
    analysis_context: str = "",
    max_proteins: int = 5,
    include_metadata: bool = True,
    include_genomic_context: bool = True
) -> Dict[str, Any]:
    """
    Fetch and format protein sequences for LLM biological analysis with genomic neighborhood context.
    
    Args:
        protein_ids: List of protein IDs to display (with or without 'protein:' prefix)
        analysis_context: Previous analysis results for context
        max_proteins: Maximum number of sequences to display
        include_metadata: Whether to include organism/function metadata
        include_genomic_context: Whether to retrieve genomic neighborhood context from Neo4j
        
    Returns:
        Dict with formatted sequences, metadata, genomic context, and analysis suggestions
    """
    # Handle case where protein_ids might be passed as a string instead of list
    if isinstance(protein_ids, str):
        logger.warning(f"⚠️ protein_ids received as string: '{protein_ids}' - converting to list")
        protein_ids = [protein_ids]
    
    logger.info(f"🧬 Sequence viewer called with {len(protein_ids)} protein IDs")
    logger.debug(f"Raw protein IDs received: {protein_ids}")
    logger.debug(f"Protein IDs type: {type(protein_ids)}")
    logger.debug(f"First few protein IDs: {protein_ids[:5] if len(protein_ids) > 5 else protein_ids}")
    
    try:
        # Initialize sequence database
        db_path = Path(__file__).parent.parent.parent / "data" / "sequences.db"
        logger.debug(f"Database path: {db_path}")
        logger.debug(f"Database exists: {db_path.exists()}")
        
        db = SequenceDatabase(str(db_path), read_only=True)
        logger.info(f"✅ Sequence database initialized successfully")
        
        # Clean protein IDs (remove 'protein:' prefix if present)
        clean_ids = []
        for i, pid in enumerate(protein_ids[:max_proteins]):
            original_id = pid
            clean_id = pid.replace('protein:', '') if pid.startswith('protein:') else pid
            clean_ids.append(clean_id)
            logger.debug(f"ID {i+1}: '{original_id}' → '{clean_id}'")
        
        logger.info(f"🔍 Attempting to retrieve {len(clean_ids)} sequences")
        logger.debug(f"Clean protein IDs: {clean_ids}")
        
        # Retrieve sequences
        sequences = db.get_sequences(clean_ids)
        logger.info(f"📊 Retrieved {len(sequences)} sequences from database")
        
        # Retrieve genomic neighborhood context from Neo4j if requested
        genomic_contexts = {}
        if include_genomic_context and sequences:
            logger.info(f"🗺️ Retrieving genomic neighborhood context for {len(sequences)} proteins")
            try:
                config = LLMConfig()
                neo4j_processor = Neo4jQueryProcessor(config)
                
                for protein_id in sequences.keys():
                    # Ensure protein ID has the correct prefix for Neo4j
                    neo4j_protein_id = f"protein:{protein_id}" if not protein_id.startswith("protein:") else protein_id
                    
                    # Query Neo4j for comprehensive protein info including neighborhood
                    neo4j_result = await neo4j_processor.process_query(
                        query=neo4j_protein_id,
                        query_type="protein_info"
                    )
                    
                    if neo4j_result.results:
                        genomic_contexts[protein_id] = neo4j_result.results[0]
                        logger.debug(f"✅ Retrieved genomic context for {protein_id}")
                    else:
                        logger.debug(f"⚠️ No genomic context found for {protein_id}")
                        
                logger.info(f"📊 Retrieved genomic context for {len(genomic_contexts)} proteins")
                
            except Exception as e:
                logger.error(f"❌ Error retrieving genomic context: {e}")
                # Continue without genomic context
                pass
        
        if not sequences:
            logger.warning(f"❌ No sequences found for any of {len(clean_ids)} protein IDs")
            
            # Get database statistics for debugging
            stats = db.get_statistics()
            logger.info(f"📈 Database stats: {stats}")
            
            # Try a few sample lookups to debug
            logger.info("🔎 Testing sample database lookups...")
            for i, test_id in enumerate(clean_ids[:3]):
                single_result = db.get_sequences([test_id])
                logger.info(f"  Test {i+1}: '{test_id}' → {len(single_result)} sequences")
                
            # Get a few sample protein IDs from the database to compare
            logger.info("🗂️ Getting sample protein IDs from database for comparison:")
            try:
                # Let's see what IDs actually exist in the database
                sample_query = "SELECT protein_id FROM sequences LIMIT 5"
                # We need to access the database connection directly
                import sqlite3
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute(sample_query)
                sample_ids = cursor.fetchall()
                conn.close()
                
                logger.info(f"📋 Sample database protein IDs:")
                for i, (sample_id,) in enumerate(sample_ids):
                    logger.info(f"    DB ID {i+1}: '{sample_id}'")
                    
                # Compare format with what we're looking for
                if sample_ids and clean_ids:
                    db_sample = sample_ids[0][0]
                    our_sample = clean_ids[0]
                    logger.info(f"🔍 Format comparison:")
                    logger.info(f"    Database format: '{db_sample}' (length: {len(db_sample)})")
                    logger.info(f"    Our lookup format: '{our_sample}' (length: {len(our_sample)})")
                    logger.info(f"    Exact match: {db_sample == our_sample}")
                    
            except Exception as db_debug_error:
                logger.error(f"Error getting sample IDs: {db_debug_error}")
            
            return {
                "success": False,
                "message": f"No sequences found for {len(clean_ids)} protein IDs",
                "protein_ids_tested": clean_ids,
                "database_stats": stats,
                "sequences": {}
            }
        
        # Format sequences for LLM analysis
        formatted_output = []
        formatted_output.append("=== PROTEIN SEQUENCE ANALYSIS ===\\n")
        
        sequence_data = {}
        for i, (protein_id, sequence) in enumerate(sequences.items(), 1):
            # Extract organism from protein ID
            organism = extract_organism_from_id(protein_id)
            
            # Basic sequence properties
            length = len(sequence)
            hydrophobic_count = sum(1 for aa in sequence if aa in 'AVLIMFWYP')
            charged_count = sum(1 for aa in sequence if aa in 'RKDE')
            hydrophobic_pct = (hydrophobic_count / length * 100) if length > 0 else 0
            charged_pct = (charged_count / length * 100) if length > 0 else 0
            
            # Format individual protein
            protein_section = [
                f"Protein {i}: {protein_id}",
                f"Organism: {organism}",
                f"Length: {length} amino acids",
                f"Composition: {hydrophobic_pct:.1f}% hydrophobic, {charged_pct:.1f}% charged",
                f""
            ]
            
            # Add genomic context if available
            if protein_id in genomic_contexts:
                context = genomic_contexts[protein_id]
                protein_section.extend([
                    f"=== GENOMIC CONTEXT ===",
                    f"Gene Position: {context.get('gene_start', 'N/A')}-{context.get('gene_end', 'N/A')} ({context.get('gene_strand', 'N/A')} strand)",
                    f"Genome: {context.get('genome_id', 'N/A')}",
                    f""
                ])
                
                # Add functional annotations
                kegg_functions = context.get('kegg_functions', [])
                kegg_descriptions = context.get('kegg_descriptions', [])
                if kegg_functions and kegg_descriptions:
                    protein_section.extend([
                        f"KEGG Functions:",
                    ])
                    for j, (ko_id, ko_desc) in enumerate(zip(kegg_functions, kegg_descriptions)):
                        if ko_id and ko_desc:  # Skip empty entries
                            protein_section.append(f"  • {ko_id}: {ko_desc}")
                    protein_section.append(f"")
                
                # Add PFAM domains
                domain_descriptions = context.get('domain_descriptions', [])
                if domain_descriptions:
                    protein_section.extend([
                        f"PFAM Domains:",
                    ])
                    for domain_desc in domain_descriptions[:3]:  # Limit to 3 domains
                        if domain_desc:  # Skip empty entries
                            protein_section.append(f"  • {domain_desc}")
                    protein_section.append(f"")
                
                # Add genomic neighborhood
                neighbor_details = context.get('neighbor_details', [])
                if neighbor_details:
                    # Filter out empty neighbors and sort by distance
                    valid_neighbors = [n for n in neighbor_details if n and n.get('neighbor_id') and n.get('distance')]
                    if valid_neighbors:
                        valid_neighbors.sort(key=lambda n: n.get('distance', 999999))
                        
                        protein_section.extend([
                            f"Genomic Neighborhood (within 5kb):",
                        ])
                        for neighbor in valid_neighbors[:5]:  # Show closest 5 neighbors
                            distance = neighbor.get('distance', 'N/A')
                            direction = neighbor.get('direction', 'N/A')
                            function = neighbor.get('function', 'Unknown function')
                            if function and len(function) > 50:
                                function = function[:47] + "..."
                            protein_section.append(f"  • {distance}bp {direction}: {function}")
                        protein_section.append(f"")
            
            protein_section.extend([
                f"N-terminus (1-30): {sequence[:30]}",
                f"C-terminus (-30): {sequence[-30:] if length > 30 else sequence}",
                f"",
                f"Full Sequence:",
                sequence,
                f"",
                f"{'='*60}",
                f""
            ])
            
            formatted_output.extend(protein_section)
            
            # Store structured data
            sequence_data[protein_id] = {
                "sequence": sequence,
                "length": length,
                "organism": organism,
                "hydrophobic_percent": hydrophobic_pct,
                "charged_percent": charged_pct,
                "n_terminus": sequence[:30],
                "c_terminus": sequence[-30:] if length > 30 else sequence,
                "genomic_context": genomic_contexts.get(protein_id, {})
            }
        
        # Add analysis suggestions
        if len(sequences) > 1:
            formatted_output.extend([
                "=== ANALYSIS OPPORTUNITIES ===",
                "• Motif identification: Look for conserved sequences across proteins",
                "• Transmembrane prediction: Identify hydrophobic regions (20+ consecutive hydrophobic residues)",
                "• Signal sequences: Check N-terminus for cleavage sites",
                "• Functional domains: Identify known transporter motifs (e.g., GXXXD, antiporter signatures)",
                "• Evolutionary comparison: Compare sequence similarities and differences",
                ""
            ])
        
        formatted_sequences = "\\n".join(formatted_output)
        
        logger.info(f"✅ Sequence viewer retrieved {len(sequences)} sequences for LLM analysis")
        logger.debug(f"🧬 Retrieved sequences for: {list(sequences.keys())}")
        
        return {
            "success": True,
            "sequences_found": len(sequences),
            "sequences_requested": len(clean_ids),
            "genomic_contexts_found": len(genomic_contexts),
            "formatted_display": formatted_sequences,
            "sequence_data": sequence_data,
            "genomic_contexts": genomic_contexts,
            "analysis_context": analysis_context,
            "analysis_suggestions": [
                "Examine full sequences for conserved motifs and structural features",
                "Compare hydrophobic regions for transmembrane helix prediction",
                "Identify functional domains specific to transport proteins",
                "Analyze genomic neighborhoods for functional clustering",
                "Look for organism-specific sequence adaptations"
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Error in sequence_viewer: {e}")
        logger.exception("Full exception details:")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to retrieve or format sequences",
            "protein_ids_received": protein_ids
        }

def extract_organism_from_id(protein_id: str) -> str:
    """Extract organism name from protein ID."""
    # Handle different ID formats
    if '_FULL_' in protein_id:
        parts = protein_id.split('_FULL_')
        if len(parts) > 1:
            organism_part = parts[1].split('_')[0]
            return organism_part
    
    # Handle PLM format
    if protein_id.startswith('PLM'):
        return "Environmental_sample"
    
    # Handle Candidatus format
    if 'Candidatus' in protein_id:
        return "Candidatus_bacterium"
    
    # Default fallback
    return "Unknown_organism"

def extract_protein_ids_from_analysis(analysis_output: str) -> List[str]:
    """
    Extract protein IDs that were successfully analyzed from code interpreter output.
    
    Args:
        analysis_output: stdout/result from code interpreter execution
        
    Returns:
        List of protein IDs that were actually processed
    """
    protein_ids = []
    
    # Look for patterns like "Protein 1: RIFCSPHIGHO2_01_FULL_..."
    protein_patterns = [
        r"Protein \d+: ([^\s\n:]+)",
        r"protein_id[^\w]*([A-Za-z0-9_]+)",
        r"([A-Za-z]+_\d+_[A-Za-z0-9_]+_scaffold_[A-Za-z0-9_]+)",
    ]
    
    for pattern in protein_patterns:
        matches = re.findall(pattern, analysis_output)
        protein_ids.extend(matches)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_ids = []
    for pid in protein_ids:
        if pid not in seen:
            seen.add(pid)
            unique_ids.append(pid)
    
    return unique_ids[:5]  # Limit to first 5 found