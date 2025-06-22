#!/usr/bin/env python3
"""
DSPy-based RAG system for genomic knowledge graph.
Combines structured queries (Neo4j) with semantic search (LanceDB).
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import json
from dataclasses import dataclass

import dspy
from rich.console import Console

from .config import LLMConfig
from .query_processor import Neo4jQueryProcessor, LanceDBQueryProcessor, HybridQueryProcessor, QueryResult

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class GenomicContext:
    """Context extracted from database queries."""
    structured_data: List[Dict[str, Any]]
    semantic_data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    query_time: float


class QueryClassifier(dspy.Signature):
    """Classify the type of genomic query to determine retrieval strategy."""
    
    question = dspy.InputField(desc="User's question about genomic data")
    query_type = dspy.OutputField(desc="Type of query: 'structural', 'semantic', 'hybrid', or 'general'")
    reasoning = dspy.OutputField(desc="Brief explanation of query type classification")


class ContextRetriever(dspy.Signature):
    """Generate database queries to retrieve relevant genomic context.
    
    SCHEMA KNOWLEDGE:
    - KEGGOrtholog nodes: Use .description property for detailed function descriptions, .id for KO IDs
    - Domain nodes: Use .description property for authoritative PFAM descriptions, .pfamAccession for accessions, .id for family names  
    - DomainAnnotation nodes: Use .id containing "/domain/DOMAIN_NAME/start-end", .bitscore for confidence
    - Protein nodes: Use .id property with "protein:" prefix, linked via HASFUNCTION->KEGGOrtholog and HASDOMAIN->DomainAnnotation->DOMAINFAMILY->Domain
    - Gene nodes: Use .id, .startCoordinate, .endCoordinate, .strand, .lengthAA, .gcContent properties, linked via BELONGSTOGENOME->Genome and (p)-[:ENCODEDBY]->(g:Gene)
    - Genome nodes: Use .id property
    
    CRITICAL: Protein IDs must include "protein:" prefix. Gene-Protein relationship is (p:Protein)-[:ENCODEDBY]->(g:Gene) NOT (p)<-[:ENCODEDBY]-(g).
    
    QUERY CONSTRUCTION PRINCIPLES:
    1. ALWAYS include functional descriptions when available (Domain.description, KEGGOrtholog.description)
    2. Traverse full relationship chains: DomainAnnotation->DOMAINFAMILY->Domain for rich annotations
    3. Include PFAM accessions (Domain.pfamAccession) for authoritative references
    4. Use OPTIONAL MATCH for KEGGOrtholog to include proteins without KEGG annotations
    5. Prioritize biological insights over technical metrics like bitscores
    
    RICH QUERY EXAMPLES:
    - Domain analysis with genomic context: MATCH (p:Protein)-[:HASDOMAIN]->(d:DomainAnnotation)-[:DOMAINFAMILY]->(pf:Domain) WHERE d.id CONTAINS '/domain/GGDEF/' MATCH (p)-[:ENCODEDBY]->(g:Gene) OPTIONAL MATCH (p)-[:HASFUNCTION]->(ko:KEGGOrtholog) RETURN p.id, d.id, d.bitscore, pf.description, pf.pfamAccession, ko.id, ko.description, g.startCoordinate, g.endCoordinate, g.strand, g.lengthAA
    - Protein details with neighbors: MATCH (p:Protein {id: 'protein:FULL_PROTEIN_ID'})-[:ENCODEDBY]->(g:Gene)-[:BELONGSTOGENOME]->(genome:Genome) MATCH (g2:Gene)-[:BELONGSTOGENOME]->(genome) WHERE abs(toInteger(g.startCoordinate) - toInteger(g2.startCoordinate)) < 5000 AND g.id <> g2.id MATCH (g2)<-[:ENCODEDBY]-(p2:Protein) OPTIONAL MATCH (p2)-[:HASDOMAIN]->(d2:DomainAnnotation)-[:DOMAINFAMILY]->(pf2:Domain) RETURN p.id, g.startCoordinate, g.endCoordinate, g.strand, collect(p2.id) as neighbors, collect(pf2.id) as neighbor_families
    - Function search: MATCH (ko:KEGGOrtholog) WHERE ko.description CONTAINS 'cyclase' MATCH (p:Protein)-[:HASFUNCTION]->(ko) RETURN ko.id, ko.description, collect(p.id)
    
    IMPORTANT: All relationship names are UPPERCASE. Always include descriptions for biological context.
    """
    
    question = dspy.InputField(desc="User's question")
    query_type = dspy.InputField(desc="Classified query type")
    neo4j_query = dspy.OutputField(desc="Rich Cypher query traversing full relationship chains: Use DomainAnnotation->DOMAINFAMILY->Domain for descriptions, include OPTIONAL MATCH for KEGGOrtholog, prioritize functional annotations over technical metrics")
    protein_search = dspy.OutputField(desc="Protein ID or description for semantic search (if needed)")
    search_strategy = dspy.OutputField(desc="How to combine the results")


class GenomicAnswerer(dspy.Signature):
    """Generate comprehensive answers using genomic context with authoritative biological insights."""
    
    question = dspy.InputField(desc="Original user question")
    context = dspy.InputField(desc="Retrieved genomic data including domain descriptions, KEGG functions, and quantitative metrics")
    answer = dspy.OutputField(desc="Comprehensive biological answer prioritizing: 1) Authoritative functional descriptions from PFAM/KEGG databases, 2) Specific protein mechanisms and pathways, 3) Quantitative evidence (counts, scores, distributions), 4) Genomic context and evolutionary significance. Use technical terminology from domain descriptions.")
    confidence = dspy.OutputField(desc="Confidence level: high (with authoritative descriptions), medium (partial annotations), or low (limited data)")
    citations = dspy.OutputField(desc="Specific data sources: PFAM accessions (PF#####), KEGG orthologs (K#####), domain names, genome IDs")


class GenomicRAG(dspy.Module):
    """Main RAG system for genomic question answering."""
    
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        
        # Initialize DSPy components
        self.classifier = dspy.ChainOfThought(QueryClassifier)
        self.retriever = dspy.ChainOfThought(ContextRetriever)
        self.answerer = dspy.ChainOfThought(GenomicAnswerer)
        
        # Initialize query processors
        self.neo4j_processor = Neo4jQueryProcessor(config)
        self.lancedb_processor = LanceDBQueryProcessor(config)
        self.hybrid_processor = HybridQueryProcessor(config)
        
        # Configure DSPy LLM
        self._configure_dspy()
        
        logger.info("GenomicRAG system initialized")
    
    def _configure_dspy(self):
        """Configure DSPy with the specified LLM provider."""
        api_key = self.config.get_api_key()
        
        if not api_key:
            raise ValueError(f"No API key found for provider: {self.config.llm_provider}")
        
        if self.config.llm_provider == "openai":
            # Use LiteLLM wrapper for OpenAI
            import litellm
            lm = dspy.LM(
                model=f"openai/{self.config.llm_model}",
                api_key=api_key,
                max_tokens=self.config.max_context_length
            )
        elif self.config.llm_provider == "anthropic":
            # Use LiteLLM wrapper for Anthropic
            import litellm
            lm = dspy.LM(
                model=f"anthropic/{self.config.llm_model}",
                api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
        
        dspy.settings.configure(lm=lm)
        logger.info(f"Configured DSPy with {self.config.llm_provider}: {self.config.llm_model}")
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all components."""
        return {
            "neo4j": self.neo4j_processor.health_check(),
            "lancedb": self.lancedb_processor.health_check(),
            "hybrid": self.hybrid_processor.health_check(),
            "dspy_configured": dspy.settings.lm is not None
        }
    
    async def ask(self, question: str) -> Dict[str, Any]:
        """
        Main method to answer genomic questions.
        
        Args:
            question: Natural language question about genomic data
            
        Returns:
            Dict containing answer, confidence, sources, and metadata
        """
        try:
            console.print(f"🧬 [bold blue]Processing question:[/bold blue] {question}")
            
            # Step 1: Classify the query type
            classification = self.classifier(question=question)
            console.print(f"📊 Query type: {classification.query_type}")
            console.print(f"💭 Reasoning: {classification.reasoning}")
            
            # Step 2: Generate retrieval strategy
            retrieval_plan = self.retriever(
                question=question,
                query_type=classification.query_type
            )
            console.print(f"🔍 Search strategy: {retrieval_plan.search_strategy}")
            
            # Step 3: Execute database queries
            context = await self._retrieve_context(classification.query_type, retrieval_plan)
            
            # Step 4: Generate answer using context
            answer_result = self.answerer(
                question=question,
                context=self._format_context(context)
            )
            
            # Compile final response
            response = {
                "question": question,
                "answer": answer_result.answer,
                "confidence": answer_result.confidence,
                "citations": answer_result.citations,
                "query_metadata": {
                    "query_type": classification.query_type,
                    "search_strategy": retrieval_plan.search_strategy,
                    "context_items": len(context.structured_data) + len(context.semantic_data),
                    "retrieval_time": context.query_time
                }
            }
            
            console.print(f"✅ [green]Answer generated[/green] (confidence: {answer_result.confidence})")
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "question": question,
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "confidence": "low",
                "citations": "",
                "error": str(e)
            }
    
    async def _retrieve_context(self, query_type: str, retrieval_plan) -> GenomicContext:
        """Retrieve context based on query type and plan."""
        import time
        start_time = time.time()
        
        structured_data = []
        semantic_data = []
        metadata = {}
        
        try:
            # Execute Neo4j query if one was generated (for any query type)
            if hasattr(retrieval_plan, 'neo4j_query') and retrieval_plan.neo4j_query.strip():
                
                # Detect if this is a protein-specific query that should use enhanced protein_info
                protein_search = getattr(retrieval_plan, 'protein_search', '')
                is_protein_query = (
                    'Protein {id:' in retrieval_plan.neo4j_query or  # DSPy generated protein query
                    ('RIFCS' in protein_search or 'scaffold' in protein_search) or  # Protein ID patterns
                    len(protein_search) > 15  # Long ID suggests protein
                )
                
                if is_protein_query and protein_search.strip():
                    # Use enhanced protein_info query instead of basic cypher
                    neo4j_result = await self.neo4j_processor.process_query(
                        protein_search,
                        query_type="protein_info"
                    )
                else:
                    # Use the generated cypher query
                    neo4j_result = await self.neo4j_processor.process_query(
                        retrieval_plan.neo4j_query,
                        query_type="cypher"
                    )
                
                structured_data = neo4j_result.results
                metadata['neo4j_execution_time'] = neo4j_result.execution_time
            
            if query_type in ["semantic", "hybrid"]:
                # For protein-specific queries, get detailed protein info first
                protein_search = retrieval_plan.protein_search if hasattr(retrieval_plan, 'protein_search') else ""
                
                if "RIFCS" in protein_search or any(id_part in protein_search for id_part in ["scaffold", "contigs"]):
                    # This looks like a protein ID - get detailed info
                    protein_info_result = await self.neo4j_processor.process_query(
                        protein_search,
                        query_type="protein_info"
                    )
                    structured_data.extend(protein_info_result.results)
                    metadata['protein_info_time'] = protein_info_result.execution_time
                
                # Execute similarity search (excluding self)
                if protein_search.strip():
                    lancedb_result = await self.lancedb_processor.process_query(
                        protein_search,
                        query_type="similarity",
                        limit=max(5, self.config.max_results_per_query // 2)  # Fewer but better results
                    )
                    semantic_data = lancedb_result.results
                    metadata['lancedb_execution_time'] = lancedb_result.execution_time
            
            if query_type == "general":
                # General database overview
                neo4j_result = await self.neo4j_processor.process_query(
                    "database overview",
                    query_type="auto"
                )
                structured_data = neo4j_result.results
                metadata['neo4j_execution_time'] = neo4j_result.execution_time
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            metadata['retrieval_error'] = str(e)
        
        query_time = time.time() - start_time
        
        return GenomicContext(
            structured_data=structured_data,
            semantic_data=semantic_data,
            metadata=metadata,
            query_time=query_time
        )
    
    def _format_context(self, context: GenomicContext) -> str:
        """Format context for LLM consumption with enhanced genomic intelligence and quantitative insights."""
        formatted_parts = []
        
        def _format_protein_id(full_id: str) -> tuple[str, str]:
            """Format protein ID for display: (short_form, full_id)"""
            if not full_id or len(full_id) < 60:
                return full_id, full_id
            
            # Extract meaningful short form from prodigal format
            # Example: protein:RIFCSPHIGHO2_01_FULL_Acidovorax_64_960_rifcsphigho2_01_scaffold_14_66
            if 'scaffold_' in full_id:
                # Extract organism and scaffold info
                parts = full_id.split('_')
                if len(parts) >= 6:
                    # Find organism type and scaffold
                    organism_idx = None
                    for i, part in enumerate(parts):
                        if part in ['Acidovorax', 'Gammaproteobacteria', 'OD1', 'Muproteobacteria', 'Nomurabacteria']:
                            organism_idx = i
                            break
                    
                    if organism_idx and 'scaffold' in full_id:
                        scaffold_part = '_'.join([p for p in parts if 'scaffold' in p or (p.isdigit() and len(p) <= 3)])
                        organism = parts[organism_idx]
                        sample = parts[organism_idx + 1] if organism_idx + 1 < len(parts) else 'unknown'
                        short_form = f"{organism}_{sample}_{scaffold_part}"
                        return short_form, full_id
            
            # Fallback: show first 30 + last 20 chars
            return f"{full_id[:30]}...{full_id[-20:]}", full_id
        
        if context.structured_data:
            # Detect different query patterns
            is_domain_query = any(
                'p.id' in item and 'd.id' in item and 'd.bitscore' in item 
                for item in context.structured_data
            )
            
            is_count_query = any(
                'numberOfProteins' in item or any(key.lower().startswith('count') for key in item.keys())
                for item in context.structured_data
            )
            
            if is_count_query:
                formatted_parts.append("QUANTITATIVE ANALYSIS:")
                for item in context.structured_data:
                    for key, value in item.items():
                        if 'protein' in key.lower() and ('count' in key.lower() or 'number' in key.lower()):
                            formatted_parts.append(f"  • {key.replace('numberOfProteins', 'Total proteins')}: {value:,}")
                        elif 'domain' in key.lower() and 'count' in key.lower():
                            formatted_parts.append(f"  • {key}: {value:,}")
                        elif key.endswith('_count') or key.startswith('count'):
                            formatted_parts.append(f"  • {key.replace('_', ' ').title()}: {value:,}")
            
            if is_domain_query:
                formatted_parts.append("DOMAIN ANALYSIS:")
                
                # Extract and organize domain information
                domain_data = []
                for item in context.structured_data:
                    if 'p.id' in item and 'd.id' in item:
                        protein_id = item['p.id']
                        domain_id = item['d.id']
                        bitscore = item.get('d.bitscore', 'N/A')
                        
                        # Extract domain type and position from domain_id
                        domain_type = "Unknown"
                        position_part = "unknown"
                        if '/domain/' in domain_id:
                            parts = domain_id.split('/domain/')[1]
                            if '/' in parts:
                                domain_type = parts.split('/')[0]
                                position_part = parts.split('/')[1]
                            else:
                                domain_type = parts
                        
                        domain_data.append({
                            'protein_id': protein_id,
                            'domain_type': domain_type,
                            'position': position_part,
                            'bitscore': bitscore
                        })
                
                if domain_data:
                    # Summary statistics with domain type
                    domain_type = domain_data[0]['domain_type']
                    formatted_parts.append(f"  • Total {domain_type} domains found: {len(domain_data):,}")
                    
                    # Top scoring domains with quantitative emphasis
                    try:
                        sorted_domains = sorted(domain_data, key=lambda x: float(x['bitscore']), reverse=True)
                        top_scores = [float(d['bitscore']) for d in sorted_domains[:5]]
                        formatted_parts.append(f"  • Top confidence scores: {', '.join(f'{s:.1f}' for s in top_scores)} bits")
                        
                        # Show score distribution
                        if len(sorted_domains) > 5:
                            avg_score = sum(float(d['bitscore']) for d in sorted_domains if d['bitscore'] != 'N/A') / len([d for d in sorted_domains if d['bitscore'] != 'N/A'])
                            formatted_parts.append(f"  • Average score: {avg_score:.1f} bits across {len(sorted_domains)} domains")
                    except:
                        pass
                    
                    # Genomic distribution
                    genomes = {}
                    for d in domain_data:
                        parts = d['protein_id'].split('_')
                        if len(parts) >= 3:
                            genome = '_'.join(parts[:3])
                            genomes[genome] = genomes.get(genome, 0) + 1
                    
                    if len(genomes) > 1:
                        formatted_parts.append(f"  • Distribution across {len(genomes)} genomes:")
                        for genome, count in sorted(genomes.items(), key=lambda x: x[1], reverse=True)[:3]:
                            formatted_parts.append(f"    - {genome}: {count} domains")
                    
                    # Top scoring examples
                    formatted_parts.append(f"\n  Top {min(3, len(domain_data))} {domain_type} domains:")
                    for i, d in enumerate(sorted_domains[:3], 1):
                        short_id, full_id = _format_protein_id(d['protein_id'])
                        try:
                            score = float(d['bitscore'])
                            formatted_parts.append(f"    {i}. Protein: {short_id}")
                            formatted_parts.append(f"       Position: {d['position']} aa, Score: {score:.1f} bits")
                        except:
                            formatted_parts.append(f"    {i}. Protein: {short_id}")
                            formatted_parts.append(f"       Position: {d['position']} aa, Score: {d['bitscore']}")
            
            # Handle protein-specific information with enhanced genomic context
            formatted_parts.append("\nPROTEIN ANALYSIS:")
            unique_proteins = {}
            
            # Deduplicate proteins by ID to avoid showing same protein multiple times
            for item in context.structured_data:
                if 'protein_id' in item:
                    protein_id = item['protein_id']
                    if protein_id not in unique_proteins:
                        unique_proteins[protein_id] = item
                    else:
                        # Merge data from multiple records for same protein
                        for key, value in item.items():
                            if key not in unique_proteins[protein_id] or not unique_proteins[protein_id][key]:
                                unique_proteins[protein_id][key] = value
            
            for i, (protein_id, item) in enumerate(list(unique_proteins.items())[:2]):  # Show max 2 proteins
                short_id, full_id = _format_protein_id(protein_id)
                gene_id = item.get('gene_id', 'N/A')
                short_gene_id, full_gene_id = _format_protein_id(gene_id) if gene_id != 'N/A' else ('N/A', 'N/A')
                
                formatted_parts.append(f"\nProtein {i+1}:")
                formatted_parts.append(f"  • Protein: {short_id}")
                formatted_parts.append(f"    Full ID: {full_id}")
                formatted_parts.append(f"  • Gene: {short_gene_id}")
                if gene_id != 'N/A':
                    formatted_parts.append(f"    Full ID: {full_gene_id}")
                formatted_parts.append(f"  • Genome: {item.get('genome_id', 'N/A')}")
                
                # Enhanced genomic coordinates with quantitative context
                if 'gene_start' in item and 'gene_end' in item:
                    start = item.get('gene_start', 'N/A')
                    end = item.get('gene_end', 'N/A')
                    strand = item.get('gene_strand', 'N/A')
                    strand_symbol = "+" if str(strand) == "1" else "-" if str(strand) == "-1" else strand
                    
                    # Calculate gene length in bp
                    try:
                        gene_length_bp = abs(int(end) - int(start)) + 1
                        formatted_parts.append(f"  • Genomic Location: {start:,}-{end:,} bp (strand {strand_symbol})")
                        formatted_parts.append(f"  • Gene Length: {gene_length_bp:,} bp")
                    except:
                        formatted_parts.append(f"  • Genomic Location: {start}-{end} bp (strand {strand_symbol})")
                    
                    if 'gene_length_aa' in item:
                        formatted_parts.append(f"  • Protein Length: {item.get('gene_length_aa', 'N/A')} amino acids")
                    
                    if 'gene_gc_content' in item:
                        try:
                            gc_content = float(item['gene_gc_content']) * 100
                            formatted_parts.append(f"  • GC Content: {gc_content:.1f}%")
                        except:
                            formatted_parts.append(f"  • GC Content: {item.get('gene_gc_content', 'N/A')}")
                
                # Enhanced domain annotations with quantitative emphasis
                domain_count = item.get('domain_count', 0)
                if domain_count > 0:
                    formatted_parts.append(f"  • Domain Annotations: {domain_count} detected")
                    
                    # Show domain families with descriptions
                    if item.get('protein_families') and any(item['protein_families']):
                        families = [f for f in item['protein_families'] if f and f != 'None']
                        if families:
                            formatted_parts.append(f"    - Families: {', '.join(families[:3])}")
                            if len(families) > 3:
                                formatted_parts.append(f"      (and {len(families) - 3} more)")
                    
                    # Show domain descriptions
                    if item.get('domain_descriptions') and any(item['domain_descriptions']):
                        descriptions = [d for d in item['domain_descriptions'] if d and d != 'None']
                        if descriptions:
                            formatted_parts.append(f"    - Functions: {descriptions[0][:80]}...")
                            if len(descriptions) > 1:
                                formatted_parts.append(f"      (and {len(descriptions) - 1} more functions)")
                    
                    # Show domain scores with statistics
                    if item.get('domain_scores') and any(item['domain_scores']):
                        scores = []
                        for s in item['domain_scores']:
                            try:
                                if s and s != 'None':
                                    scores.append(float(s))
                            except:
                                pass
                        if scores:
                            top_scores = sorted(scores, reverse=True)[:3]
                            formatted_parts.append(f"    - Top Scores: {', '.join(f'{s:.1f}' for s in top_scores)} bits")
                            if len(scores) > 3:
                                avg_score = sum(scores) / len(scores)
                                formatted_parts.append(f"      (average: {avg_score:.1f} bits across {len(scores)} domains)")
                    
                    # Show domain positions
                    if item.get('domain_positions') and any(item['domain_positions']):
                        positions = [pos for pos in item['domain_positions'][:3] if pos and pos != 'None']
                        if positions:
                            formatted_parts.append(f"    - Positions: {', '.join(positions)} aa")
                
                # Enhanced KEGG functional information
                if item.get('kegg_functions') and any(item['kegg_functions']):
                    functions = [f for f in item['kegg_functions'] if f and f != 'None']
                    if functions:
                        formatted_parts.append(f"  • KEGG Functions: {', '.join(functions[:2])}")
                        
                        # Show function descriptions
                        if item.get('kegg_descriptions') and any(item['kegg_descriptions']):
                            descriptions = [d for d in item['kegg_descriptions'] if d and d != 'None']
                            if descriptions:
                                formatted_parts.append(f"    - Details: {descriptions[0][:100]}...")
                
                # Enhanced genomic neighborhood with strand and direction analysis
                if item.get('neighboring_proteins') and any(item['neighboring_proteins']):
                    neighbors = [n for n in item['neighboring_proteins'] if n and n != 'None']
                    if neighbors:
                        formatted_parts.append(f"  • Genomic Neighborhood: {len(neighbors)} proteins within 5kb")
                        
                        # Calculate distances and directions
                        if item.get('neighbor_coordinates') and any(item['neighbor_coordinates']):
                            try:
                                gene_start = int(item.get('gene_start', 0))
                                gene_strand = item.get('gene_strand', 'N/A')
                                strand_symbol = "+" if str(gene_strand) == "1" else "-" if str(gene_strand) == "-1" else "?"
                                
                                neighbor_info = []
                                for coord in item['neighbor_coordinates'][:3]:
                                    if coord and coord != 'N/A':
                                        neighbor_pos = int(coord)
                                        distance = abs(neighbor_pos - gene_start)
                                        direction = "upstream" if neighbor_pos < gene_start else "downstream"
                                        neighbor_info.append(f"{distance:,}bp {direction}")
                                
                                if neighbor_info:
                                    formatted_parts.append(f"    - Distances: {', '.join(neighbor_info)}")
                                    formatted_parts.append(f"    - Gene strand: {strand_symbol}")
                                    
                                    # Operon prediction based on proximity and strand
                                    close_neighbors = [info for info in neighbor_info if int(info.split('bp')[0].replace(',', '')) < 1000]
                                    if len(close_neighbors) >= 2:
                                        formatted_parts.append(f"    - Operon likelihood: HIGH (≥2 genes within 1kb, strand {strand_symbol})")
                                    elif len(close_neighbors) >= 1:
                                        formatted_parts.append(f"    - Operon likelihood: MODERATE (1 gene within 1kb)")
                            except:
                                formatted_parts.append(f"    - Coordinate analysis unavailable")
                        
                        # Show neighborhood functional clustering
                        if item.get('neighborhood_families') and any(item['neighborhood_families']):
                            families = [f for f in item['neighborhood_families'] if f and f != 'None']
                            if families:
                                formatted_parts.append(f"    - Neighbor Functions: {', '.join(families[:2])}")
                                if len(families) > 2:
                                    formatted_parts.append(f"      (and {len(families) - 2} more neighbor functions)")
                
        # Handle semantic similarity data with enhanced formatting
        if context.semantic_data:
            formatted_parts.append("\nFUNCTIONALLY SIMILAR PROTEINS:")
            for i, item in enumerate(context.semantic_data[:3]):  # Show top 3 similar proteins
                similarity = item.get('similarity', 0)
                protein_id = item.get('protein_id', 'Unknown')
                short_id, full_id = _format_protein_id(protein_id)
                
                formatted_parts.append(f"  {i+1}. {short_id} (similarity: {similarity:.3f})")
                formatted_parts.append(f"     Genome: {item.get('genome_id', 'Unknown')}")
                
                # Add biological interpretation of similarity scores
                if similarity > 0.95:
                    formatted_parts.append(f"     Interpretation: IDENTICAL/ORTHOLOG (>95% similarity)")
                elif similarity > 0.8:
                    formatted_parts.append(f"     Interpretation: HIGHLY SIMILAR - likely functional ortholog")
                elif similarity > 0.6:
                    formatted_parts.append(f"     Interpretation: MODERATELY SIMILAR - possible functional analog")
                elif similarity > 0.4:
                    formatted_parts.append(f"     Interpretation: WEAKLY SIMILAR - distantly related")
                
                # Show full ID if different from short ID
                if short_id != full_id:
                    formatted_parts.append(f"     Full ID: {full_id}")
        
        return "\n".join(formatted_parts) if formatted_parts else "No relevant genomic context found."
    
    def close(self):
        """Clean up resources."""
        self.neo4j_processor.close()
        # LanceDB doesn't need explicit closing


# Example questions for testing
EXAMPLE_GENOMIC_QUESTIONS = [
    "How many genomes are in the database?",
    "What proteins are found in Burkholderiales_bacterium_RIFCSPHIGHO2_01_FULL_64_960_contigs?",
    "Find proteins similar to RIFCSPHIGHO2_01_FULL_Gammaproteobacteria_61_200_rifcsphigho2_01_scaffold_29964_1",
    "What KEGG functions are associated with ATP synthase?",
    "Which genome has the most protein families?",
    "What are the most common protein domains across all genomes?",
    "Find all proteins involved in energy metabolism",
    "Compare the functional profiles of different genomes",
    "What is the average protein length in each genome?",
    "Which proteins have the most functional annotations?"
]


async def demo_rag_system(config: LLMConfig):
    """Demonstration of the RAG system with example questions."""
    console.print("[bold green]🧬 Genomic RAG System Demo[/bold green]")
    console.print("="*60)
    
    rag = GenomicRAG(config)
    
    # Health check
    health = rag.health_check()
    console.print(f"System health: {health}")
    
    if not all(health.values()):
        console.print("[red]⚠️  Some components are not healthy. Check configuration.[/red]")
        return
    
    # Test with a few example questions
    for question in EXAMPLE_GENOMIC_QUESTIONS[:3]:  # Test first 3 questions
        console.print(f"\n[bold cyan]Question:[/bold cyan] {question}")
        
        response = await rag.ask(question)
        
        console.print(f"[bold green]Answer:[/bold green] {response['answer']}")
        console.print(f"[bold yellow]Confidence:[/bold yellow] {response['confidence']}")
        console.print(f"[dim]Query time: {response['query_metadata']['retrieval_time']:.2f}s[/dim]")
        console.print("-" * 40)
    
    rag.close()
    console.print("\n[green]✅ Demo completed successfully![/green]")


if __name__ == "__main__":
    import os
    
    # For testing - requires OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        console.print("[red]Please set OPENAI_API_KEY environment variable[/red]")
        exit(1)
    
    config = LLMConfig.from_env()
    asyncio.run(demo_rag_system(config))