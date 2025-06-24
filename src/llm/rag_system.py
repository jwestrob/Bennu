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
    """Classify genomic queries to determine optimal retrieval strategy.
    
    CLASSIFICATION GUIDELINES:
    
    🔍 SEMANTIC: Use when query involves:
    - Similarity searches with specific protein IDs ("proteins similar to PLM0_scaffold_123", "find related proteins")  
    - Functional searches that need similarity expansion ("alcohol dehydrogenase proteins", "iron transporters")
    - Comparative analysis starting from known proteins ("find orthologs", "functionally equivalent")
    → Strategy: Neo4j finds starting points → LanceDB similarity expansion
    
    🏗️ STRUCTURAL: Use when query targets:
    - Specific protein/gene IDs for context only (no similarity needed)
    - Domain family names and counts (GGDEF, TPR, etc.)
    - Genomic coordinates or neighborhoods
    - Pathway/function statistics and aggregations
    → Primary: Neo4j graph traversal only
    
    🔀 HYBRID: Use when query combines:
    - Similarity + genomic context ("similar proteins and their neighborhoods")
    - Functional search + structural analysis ("iron transporters and their operons")
    - Cross-genome comparisons with genomic context
    → Strategy: LanceDB similarity → Neo4j structural context for results
    
    📊 GENERAL: Use for:
    - Database overviews and statistics
    - Broad exploratory queries without specific targets
    → Primary: Neo4j aggregation queries
    """
    
    question = dspy.InputField(desc="User's question about genomic data")
    query_type = dspy.OutputField(desc="Query type: 'semantic', 'structural', 'hybrid', or 'general'")
    reasoning = dspy.OutputField(desc="Specific reasoning for classification based on guidelines above")
    primary_database = dspy.OutputField(desc="Primary database: 'lancedb', 'neo4j', or 'both'")


class ContextRetriever(dspy.Signature):
    """Generate database queries to retrieve relevant genomic context.
    
    SCHEMA KNOWLEDGE:
    - KEGGOrtholog nodes: Use .description property for detailed function descriptions, .id for KO IDs
    - Domain nodes: Use .description property for authoritative PFAM descriptions, .pfamAccession for accessions, .id for family names  
    - DomainAnnotation nodes: Use .id containing "/domain/DOMAIN_NAME/start-end", .bitscore for confidence
    - Protein nodes: Use .id property with "protein:" prefix, linked DIRECTLY via HASFUNCTION->KEGGOrtholog and HASDOMAIN->DomainAnnotation->DOMAINFAMILY->Domain
    - Gene nodes: Use .id, .startCoordinate, .endCoordinate, .strand, .lengthAA, .gcContent properties, linked via BELONGSTOGENOME->Genome and (p)-[:ENCODEDBY]->(g:Gene)
    - Genome nodes: Use .id property, linked to QualityMetrics via HASQUALITYMETRICS relationship
    - QualityMetrics nodes: Use QUAST properties: quast_contigs, quast_total_length, quast_largest_contig, quast_n50, quast_n75, quast_gc_content, quast_n_count, quast_n_per_100_kbp, quast_contigs_1000bp, quast_contigs_5000bp, quast_contigs_10000bp, quast_l50, quast_l75
    
    CRITICAL: 
    - Protein IDs must include "protein:" prefix. 
    - Gene-Protein relationship is (p:Protein)-[:ENCODEDBY]->(g:Gene) NOT (p)<-[:ENCODEDBY]-(g).
    - KEGG relationship is DIRECT: (p:Protein)-[:HASFUNCTION]->(ko:KEGGOrtholog) - NO intermediate nodes.
    
    QUERY CONSTRUCTION PRINCIPLES:
    1. ALWAYS include functional descriptions when available (Domain.description, KEGGOrtholog.description)
    2. Traverse full relationship chains: DomainAnnotation->DOMAINFAMILY->Domain for rich annotations
    3. Include PFAM accessions (Domain.pfamAccession) for authoritative references
    4. Use OPTIONAL MATCH for KEGGOrtholog to include proteins without KEGG annotations
    5. Prioritize biological insights over technical metrics like bitscores
    6. FOR DOMAIN FAMILY SEARCHES: Use Domain.id CONTAINS 'DOMAIN_NAME' instead of Domain.description to find all variants (e.g., TPR_1, TPR_2, etc.)
    7. CRITICAL: NEVER use exists() function - use "variable.property IS NOT NULL" instead
    
    SEMANTIC SEARCH GUIDANCE:
    - Use protein_search for: similarity queries, functional descriptions, protein names without specific IDs
    - Use Neo4j only for: specific gene/protein IDs, coordinate ranges, taxonomic searches, domain family names
    - Use hybrid approach for: complex questions combining similarity and structural data
    
    QUERY EXAMPLES BY TYPE:
    
    STRUCTURAL QUERIES (Neo4j only):
    - Domain family search: MATCH (p:Protein)-[:HASDOMAIN]->(d:DomainAnnotation)-[:DOMAINFAMILY]->(dom:Domain) WHERE dom.id CONTAINS 'TPR' MATCH (p)-[:ENCODEDBY]->(g:Gene) OPTIONAL MATCH (p)-[:HASFUNCTION]->(ko:KEGGOrtholog) RETURN p.id, d.id, d.bitscore, dom.description, dom.pfamAccession, ko.id, ko.description, g.startCoordinate, g.endCoordinate, g.strand, g.lengthAA
    - Function search: MATCH (ko:KEGGOrtholog) WHERE ko.description CONTAINS 'cyclase' MATCH (p:Protein)-[:HASFUNCTION]->(ko) RETURN ko.id, ko.description, collect(p.id)
    - Ribosomal protein search: MATCH (ko:KEGGOrtholog) WHERE toLower(ko.description) CONTAINS 'ribosom' AND toLower(ko.description) CONTAINS 'l15' MATCH (p:Protein)-[:HASFUNCTION]->(ko) OPTIONAL MATCH (p)-[:ENCODEDBY]->(g:Gene) RETURN ko.id, ko.description, p.id, g.startCoordinate, g.endCoordinate LIMIT 20
    - Genome quality metrics: MATCH (genome:Genome {id: 'GENOME_ID'}) OPTIONAL MATCH (genome)-[:HASQUALITYMETRICS]->(qm:QualityMetrics) OPTIONAL MATCH (g:Gene)-[:BELONGSTOGENOME]->(genome) OPTIONAL MATCH (p:Protein)-[:ENCODEDBY]->(g) RETURN genome.id, qm.quast_contigs, qm.quast_total_length, qm.quast_n50, qm.quast_gc_content, qm.quast_largest_contig, qm.quast_l50, count(DISTINCT g) as gene_count, count(DISTINCT p) as protein_count
    
    GENOMIC CONTEXT QUERIES (Neo4j + neighbor analysis):
    - CRITICAL: For context questions, ALWAYS include neighbor analysis on same scaffold/contig only
    - Include neighbor coordinates and distances for co-transcription analysis
    - Example: MATCH (dom:Domain) WHERE dom.id CONTAINS 'GGDEF' MATCH (p:Protein)-[:HASDOMAIN]->(d:DomainAnnotation)-[:DOMAINFAMILY]->(dom) MATCH (p)-[:ENCODEDBY]->(g:Gene) WITH p, dom, g, split(g.id, '_scaffold_')[0] + '_scaffold_' + split(split(g.id, '_scaffold_')[1], '_')[0] as scaffold_id MATCH (g2:Gene) WHERE g2.id STARTS WITH scaffold_id AND abs(toInteger(g.startCoordinate) - toInteger(g2.startCoordinate)) < 5000 AND g.id <> g2.id MATCH (g2)<-[:ENCODEDBY]-(p2:Protein) OPTIONAL MATCH (p2)-[:HASFUNCTION]->(ko2:KEGGOrtholog) OPTIONAL MATCH (p2)-[:HASDOMAIN]->(d2:DomainAnnotation)-[:DOMAINFAMILY]->(dom2:Domain) RETURN p.id, dom.id, dom.description, g.startCoordinate, g.endCoordinate, g.strand, collect(DISTINCT p2.id) as neighbors, collect(DISTINCT ko2.description) as neighbor_functions, collect(DISTINCT dom2.description) as neighbor_domains
    
    PATHWAY/FUNCTIONAL QUERIES (Neo4j aggregation):
    - Pathway analysis: MATCH (ko:KEGGOrtholog)-[:PARTICIPATESIN]->(pathway:Pathway) WHERE pathway.id = 'ko00010' MATCH (p:Protein)-[:HASFUNCTION]->(ko) RETURN pathway.name, pathway.description, ko.id, ko.description, count(p) as protein_count
    - Metabolic overview: MATCH (ko:KEGGOrtholog)-[:PARTICIPATESIN]->(pathway:Pathway) MATCH (p:Protein)-[:HASFUNCTION]->(ko) WHERE pathway.pathwayType = 'ko' AND pathway.pathwayNumber IN ['00010', '00020', '00030'] RETURN pathway.name, pathway.description, count(DISTINCT ko) as ko_count, count(DISTINCT p) as protein_count
    
    CRITICAL FOR GENOMIC CONTEXT: When users ask about "genomic context", "surrounding genes", "neighborhoods", or "what's around gene X", ALWAYS retrieve neighboring genes within 5-10kb using the neighbor analysis pattern above.
    
    IMPORTANT: All relationship names are UPPERCASE. Always include descriptions for biological context.
    """
    
    question = dspy.InputField(desc="User's question")
    query_type = dspy.InputField(desc="Classified query type")
    neo4j_query = dspy.OutputField(desc="Rich Cypher query traversing full relationship chains: Use DomainAnnotation->DOMAINFAMILY->Domain for descriptions, include OPTIONAL MATCH for KEGGOrtholog, prioritize functional annotations over technical metrics. IMPORTANT: Use only simple alphanumeric characters in alias names (no unicode symbols like ≥)")
    protein_search = dspy.OutputField(desc="Protein ID or description for semantic search (if needed)")
    requires_multi_stage = dspy.OutputField(desc="Boolean: true if this query needs multi-stage processing (keyword search → similarity expansion). Use for functional similarity questions like 'find proteins similar to heme transporters'")
    seed_proteins_for_similarity = dspy.OutputField(desc="Boolean: true if Neo4j results should be used as seeds for LanceDB similarity search in stage 2")
    search_strategy = dspy.OutputField(desc="How to combine the results")


class GenomicAnswerer(dspy.Signature):
    """Generate specific, data-driven genomic analysis with concrete biological insights."""
    
    question = dspy.InputField(desc="Original user question")
    context = dspy.InputField(desc="Retrieved genomic data including domain descriptions, KEGG functions, and quantitative metrics")
    answer = dspy.OutputField(desc="Structured biological analysis that MUST: 1) Ground all statements in specific data points (coordinates, counts, IDs) and quantify relationships where possible, 2) Prioritize analysis of proximal neighbors vs distal ones except when there is an obvious functional relationship between the protein of interest and the distal neighbor, 3) Calculate and report specific distances between genes, identifying potential co-transcription based on SAME STRAND + proximity (<200bp = likely co-transcribed, 200-500bp same strand = possible operon, different strands = separate regulation), 4) Consider strand orientation and gene order for operon/cluster identification, 5) Use specific protein/domain names from the data, 6) Organize response logically: Genomic Location → Functional Context → Neighborhood Analysis → Biological Significance. FORBIDDEN: Language lacking description or analysis; avoid fluff language.")
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
                #max_tokens=self.config.max_context_length,
                temperature=1.0,
                max_tokens=100_000
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
            
            # Check for retrieval errors
            if 'retrieval_error' in context.metadata:
                raise Exception(f"Query execution failed: {context.metadata['retrieval_error']}")
            
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
                    (len(protein_search) > 15 and '_' in protein_search and not ' ' in protein_search)  # Long ID without spaces
                )
                
                if is_protein_query and protein_search.strip():
                    # Use enhanced protein_info query instead of basic cypher
                    neo4j_result = await self.neo4j_processor.process_query(
                        protein_search,
                        query_type="protein_info"
                    )
                else:
                    # Use the generated cypher query
                    print(f"🔍 DSPy Generated Neo4j Query:\n{retrieval_plan.neo4j_query}")
                    
                    # Use the original query for now - enhancement causing issues
                    enhanced_query = retrieval_plan.neo4j_query
                    
                    neo4j_result = await self.neo4j_processor.process_query(
                        enhanced_query,
                        query_type="cypher"
                    )
                
                # Check for Neo4j query errors
                if 'error' in neo4j_result.metadata:
                    raise Exception(f"Neo4j query failed: {neo4j_result.metadata['error']}")
                
                structured_data = neo4j_result.results
                metadata['neo4j_execution_time'] = neo4j_result.execution_time
            
            if query_type in ["semantic", "hybrid"]:
                # Enhanced logic for semantic and hybrid queries with multi-stage support
                protein_search = getattr(retrieval_plan, 'protein_search', '')
                functional_search = getattr(retrieval_plan, 'functional_search', '')
                primary_database = getattr(retrieval_plan, 'primary_database', 'lancedb')
                requires_multi_stage = getattr(retrieval_plan, 'requires_multi_stage', False)
                seed_proteins_for_similarity = getattr(retrieval_plan, 'seed_proteins_for_similarity', False)
                
                # Debug output for development (can be removed in production)
                logger.debug(f"Query processing: {query_type}, multi_stage: {requires_multi_stage}, seed_proteins: {seed_proteins_for_similarity}")
                
                # Check if protein_search is actually a protein ID (not just a description)
                is_actual_protein_id = (
                    protein_search and (
                        "RIFCS" in protein_search or 
                        any(id_part in protein_search for id_part in ["scaffold", "contigs"]) or
                        (len(protein_search) > 15 and "_" in protein_search and not " " in protein_search)
                    )
                )
                
                if query_type == "semantic":
                    if requires_multi_stage and seed_proteins_for_similarity:
                        # MULTI-STAGE SEMANTIC: Stage 1 already done (Neo4j), now Stage 2 (LanceDB similarity)
                        if structured_data:
                            # Extract protein IDs from Neo4j results to use as similarity seeds
                            seed_protein_ids = []
                            for item in structured_data:
                                protein_id = item.get('protein_id', '')
                                if protein_id:
                                    # Remove "protein:" prefix for LanceDB
                                    clean_id = protein_id.replace("protein:", "") if protein_id.startswith("protein:") else protein_id
                                    seed_protein_ids.append(clean_id)
                            
                            # Run similarity search for each seed protein and aggregate results
                            all_similarity_results = []
                            for seed_id in seed_protein_ids[:3]:  # Limit to top 3 seeds to avoid overload
                                try:
                                    lancedb_result = await self.lancedb_processor.process_query(
                                        seed_id,
                                        query_type="similarity",
                                        limit=3  # Fewer results per seed
                                    )
                                    # Tag results with their seed protein
                                    for result in lancedb_result.results:
                                        result['seed_protein'] = seed_id
                                    all_similarity_results.extend(lancedb_result.results)
                                except Exception as e:
                                    logger.debug(f"Similarity search failed for {seed_id}: {e}")
                            
                            # Remove duplicates and sort by similarity
                            seen_proteins = set()
                            unique_results = []
                            for result in sorted(all_similarity_results, key=lambda x: x.get('similarity', -999), reverse=True):
                                protein_id = result.get('protein_id', '')
                                if protein_id not in seen_proteins:
                                    seen_proteins.add(protein_id)
                                    unique_results.append(result)
                            
                            semantic_data = unique_results[:5]  # Top 5 overall
                            metadata['lancedb_execution_time'] = sum([r.get('execution_time', 0) for r in all_similarity_results])
                            metadata['multi_stage_seeds_used'] = len(seed_protein_ids)
                    
                    elif is_actual_protein_id:
                        # SEMANTIC with protein ID: Neo4j for context → LanceDB for similarity
                        protein_info_result = await self.neo4j_processor.process_query(
                            protein_search,
                            query_type="protein_info"
                        )
                        structured_data.extend(protein_info_result.results)
                        metadata['protein_info_time'] = protein_info_result.execution_time
                        
                        # Execute similarity search (excluding self)
                        # Remove "protein:" prefix for LanceDB search
                        clean_protein_id = protein_search.replace("protein:", "") if protein_search.startswith("protein:") else protein_search
                        
                        lancedb_result = await self.lancedb_processor.process_query(
                            clean_protein_id,
                            query_type="similarity",
                            limit=max(5, self.config.max_results_per_query // 2)
                        )
                        semantic_data = lancedb_result.results
                        metadata['lancedb_execution_time'] = lancedb_result.execution_time
                    
                    elif functional_search or (protein_search and not is_actual_protein_id):
                        # SEMANTIC with functional description: LanceDB semantic search
                        search_term = functional_search if functional_search else protein_search
                        lancedb_result = await self.lancedb_processor.process_query(
                            search_term,
                            query_type="functional_search",
                            limit=self.config.max_results_per_query
                        )
                        semantic_data = lancedb_result.results
                        metadata['lancedb_execution_time'] = lancedb_result.execution_time
                
                elif query_type == "hybrid":
                    # HYBRID: Combine both approaches based on primary_database guidance
                    if primary_database == "lancedb" and functional_search:
                        # LanceDB similarity → Neo4j context for results
                        lancedb_result = await self.lancedb_processor.process_query(
                            functional_search,
                            query_type="functional_search",
                            limit=max(3, self.config.max_results_per_query // 2)
                        )
                        semantic_data = lancedb_result.results
                        metadata['lancedb_execution_time'] = lancedb_result.execution_time
                        
                        # Get additional context for top LanceDB results from Neo4j
                        if semantic_data:
                            top_protein_ids = [item.get('protein_id') for item in semantic_data[:3]]
                            for protein_id in top_protein_ids:
                                if protein_id:
                                    try:
                                        protein_context = await self.neo4j_processor.process_query(
                                            protein_id,
                                            query_type="protein_info"
                                        )
                                        structured_data.extend(protein_context.results)
                                    except Exception as e:
                                        logger.debug(f"Could not get context for {protein_id}: {e}")
                    
                    elif is_actual_protein_id:
                        # HYBRID starting from protein ID: detailed context + similarity
                        protein_info_result = await self.neo4j_processor.process_query(
                            protein_search,
                            query_type="protein_info"
                        )
                        structured_data.extend(protein_info_result.results)
                        metadata['protein_info_time'] = protein_info_result.execution_time
                        
                        # Add similarity search
                        lancedb_result = await self.lancedb_processor.process_query(
                            protein_search,
                            query_type="similarity",
                            limit=max(3, self.config.max_results_per_query // 3)
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
        
        # Safe formatting functions to handle string values from Neo4j
        def _safe_format_int(val, default="N/A"):
            """Safely format integer values (handles strings from Neo4j)."""
            try:
                if val is None or val == "N/A":
                    return default
                if isinstance(val, int):
                    return f"{val:,}"
                if isinstance(val, str):
                    return f"{int(val):,}"
                return default
            except (ValueError, TypeError):
                return default
                
        def _safe_format_float(val, precision=2, default="N/A"):
            """Safely format float values (handles strings from Neo4j)."""
            try:
                if val is None or val == "N/A":
                    return default
                if isinstance(val, (int, float)):
                    return f"{val:.{precision}f}"
                if isinstance(val, str):
                    return f"{float(val):.{precision}f}"
                return default
            except (ValueError, TypeError):
                return default
        
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
        
        def _analyze_neighbor_functions(neighbors: list) -> dict:
            """Analyze functional themes in genomic neighborhood."""
            if not neighbors:
                return {}
            
            # Define functional categories for small model compatibility
            transport_keywords = ['transport', 'ABC', 'permease', 'receptor', 'transporter']
            metabolism_keywords = ['synthase', 'reductase', 'transferase', 'kinase', 'dehydrogenase']
            regulation_keywords = ['sigma', 'regulator', 'activator', 'repressor', 'transcriptional']
            mobile_keywords = ['transpos', 'insertion', 'IS66', 'mobile', 'recombinase']
            
            themes = {
                'transport': 0,
                'metabolism': 0, 
                'regulation': 0,
                'mobile_elements': 0
            }
            
            for neighbor in neighbors:
                descriptions = neighbor.get('pfam_description', []) + neighbor.get('kegg_descriptions', [])
                for desc in descriptions:
                    if not desc or desc == 'None':
                        continue
                    desc_lower = desc.lower()
                    
                    if any(kw in desc_lower for kw in transport_keywords):
                        themes['transport'] += 1
                    if any(kw in desc_lower for kw in metabolism_keywords):
                        themes['metabolism'] += 1
                    if any(kw in desc_lower for kw in regulation_keywords):
                        themes['regulation'] += 1
                    if any(kw in desc_lower for kw in mobile_keywords):
                        themes['mobile_elements'] += 1
            
            # Determine dominant theme
            dominant_theme = max(themes, key=themes.get) if max(themes.values()) > 0 else None
            return {
                'themes': themes,
                'dominant_theme': dominant_theme,
                'total_annotated': sum(themes.values())
            }
        
        
        def _format_neighbor_context(neighbors: list, target_gene: dict) -> list:
            """Format rich neighbor context for LLM - scaffold neighbors only."""
            if not neighbors or not target_gene.get('gene_start'):
                return []
            
            target_pos = int(target_gene['gene_start'])
            
            # Extract target scaffold ID for filtering
            target_protein_id = target_gene.get('protein_id', '')
            target_scaffold = None
            if 'scaffold_' in target_protein_id:
                # Extract scaffold number from protein ID
                parts = target_protein_id.split('scaffold_')
                if len(parts) > 1:
                    scaffold_part = parts[1].split('_')[0]  # Get just the scaffold number
                    target_scaffold = f"scaffold_{scaffold_part}"
            
            # Group neighbors by protein_id, filtering by same scaffold only
            protein_groups = {}
            for neighbor in neighbors:
                if not neighbor.get('position') or not neighbor.get('protein_id'):
                    continue
                
                protein_id = neighbor['protein_id']
                
                # Only include neighbors from same scaffold
                if target_scaffold and 'scaffold_' in protein_id:
                    neighbor_parts = protein_id.split('scaffold_')
                    if len(neighbor_parts) > 1:
                        neighbor_scaffold_part = neighbor_parts[1].split('_')[0]
                        neighbor_scaffold = f"scaffold_{neighbor_scaffold_part}"
                        if neighbor_scaffold != target_scaffold:
                            continue  # Skip neighbors from different scaffolds
                
                if protein_id not in protein_groups:
                    distance = abs(neighbor['position'] - target_pos)
                    direction = 'upstream' if neighbor['position'] < target_pos else 'downstream'
                    
                    protein_groups[protein_id] = {
                        'protein_id': protein_id,
                        'distance': distance,
                        'direction': direction,
                        'strand': '+' if str(neighbor.get('strand', '0')) == '1' else '-' if str(neighbor.get('strand', '0')) == '-1' else '?',
                        'pfam_id': [],
                        'pfam_description': [],
                        'kegg_ko': [],
                        'kegg_descriptions': []
                    }
                
                # Add domains from this record
                if neighbor.get('pfam_ids') and neighbor['pfam_ids'] != 'None':
                    protein_groups[protein_id]['pfam_id'].append(neighbor['pfam_ids'])
                if neighbor.get('pfam_desc') and neighbor['pfam_desc'] != 'None':
                    protein_groups[protein_id]['pfam_description'].append(neighbor['pfam_desc'])
                if neighbor.get('kegg_id') and neighbor['kegg_id'] != 'None':
                    protein_groups[protein_id]['kegg_ko'].append(neighbor['kegg_id'])
                if neighbor.get('kegg_desc') and neighbor['kegg_desc'] != 'None':
                    protein_groups[protein_id]['kegg_descriptions'].append(neighbor['kegg_desc'])
            
            # Convert to list and sort by distance
            processed_neighbors = list(protein_groups.values())
            processed_neighbors.sort(key=lambda x: x['distance'])
            
            return processed_neighbors[:5]  # Show more neighbors since they're now truly local
        
        if context.structured_data:
            # Detect different query patterns
            is_domain_query = any(
                'p.id' in item and 'd.id' in item and 'd.bitscore' in item 
                for item in context.structured_data
            )
            
            is_count_query = any(
                'numberOfProteins' in item or 
                any(key.lower().startswith('count') or key.lower().endswith('_count') for key in item.keys()) or
                '_domain_total_count' in item  # Detect enhanced domain queries
                for item in context.structured_data
            )
            
            if is_count_query:
                formatted_parts.append("QUANTITATIVE ANALYSIS:")
                
                # Handle enhanced domain count metadata first (show total once)
                domain_total_shown = False
                for item in context.structured_data:
                    if '_domain_total_count' in item and not domain_total_shown:
                        domain_name = item.get('_domain_name', 'Unknown')
                        total_count = item.get('_domain_total_count', 0)
                        is_sample = item.get('_is_sample', False)
                        sample_size = item.get('_sample_size', 0)
                        
                        if is_sample and total_count > sample_size:
                            formatted_parts.append(f"  • Total {domain_name} domains in dataset: {_safe_format_int(total_count)} (showing {_safe_format_int(sample_size)} representative examples)")
                        else:
                            formatted_parts.append(f"  • Total {domain_name} domains found: {_safe_format_int(total_count)}")
                        domain_total_shown = True
                        break
                
                # Then handle regular count fields and genome quality metrics
                for item in context.structured_data:
                    for key, value in item.items():
                        if key.startswith('_'):  # Skip metadata fields
                            continue
                        
                        # Handle genome quality metrics specifically
                        if key == 'total_length_bp':
                            formatted_parts.append(f"  • Total genome length: {_safe_format_int(value)} bp")
                        elif key == 'contig_count':
                            formatted_parts.append(f"  • Contig count: {_safe_format_int(value)}")
                        elif key == 'largest_contig_bp':
                            formatted_parts.append(f"  • Largest contig: {_safe_format_int(value)} bp")
                        elif key == 'n50':
                            formatted_parts.append(f"  • N50: {_safe_format_int(value)} bp")
                        elif key == 'n75':
                            formatted_parts.append(f"  • N75: {_safe_format_int(value)} bp")
                        elif key == 'gc_content':
                            gc_val = _safe_format_float(value, precision=1)
                            if gc_val != "N/A":
                                try:
                                    gc_percent = float(str(value)) * 100 if float(str(value)) <= 1.0 else float(str(value))
                                    formatted_parts.append(f"  • GC content: {gc_percent:.1f}%")
                                except:
                                    formatted_parts.append(f"  • GC content: {gc_val}%")
                        elif key == 'contigs_gt_1kbp':
                            formatted_parts.append(f"  • Contigs ≥1kb: {_safe_format_int(value)}")
                        elif key == 'contigs_gt_5kbp':
                            formatted_parts.append(f"  • Contigs ≥5kb: {_safe_format_int(value)}")
                        elif key == 'contigs_gt_10kbp':
                            formatted_parts.append(f"  • Contigs ≥10kb: {_safe_format_int(value)}")
                        
                        # Handle regular count fields
                        elif 'protein' in key.lower() and ('count' in key.lower() or 'number' in key.lower()):
                            formatted_parts.append(f"  • {key.replace('numberOfProteins', 'Total proteins')}: {_safe_format_int(value)}")
                        elif 'domain' in key.lower() and 'count' in key.lower():
                            formatted_parts.append(f"  • {key}: {_safe_format_int(value)}")
                        elif key.endswith('_count') or key.startswith('count'):
                            formatted_parts.append(f"  • {key.replace('_', ' ').title()}: {_safe_format_int(value)}")
                    break  # Only process first item for counts to avoid repetition
            
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
                    # Summary statistics with domain type and accurate counts
                    domain_type = domain_data[0]['domain_type']
                    
                    # Check if we have accurate count metadata
                    first_item = context.structured_data[0] if context.structured_data else {}
                    total_count = first_item.get('_domain_total_count', len(domain_data))
                    is_sample = first_item.get('_is_sample', False)
                    sample_size = first_item.get('_sample_size', len(domain_data))
                    
                    if is_sample and total_count > sample_size:
                        formatted_parts.append(f"  • Total {domain_type} domains in dataset: {_safe_format_int(total_count)} (showing {_safe_format_int(sample_size)} representative examples)")
                    else:
                        formatted_parts.append(f"  • Total {domain_type} domains found: {_safe_format_int(total_count)}")
                    
                    # Sort domains by position instead of score
                    sorted_domains = domain_data
                    
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
                    
                    # Example domains
                    formatted_parts.append(f"\n  Example {domain_type} domains:")
                    for i, d in enumerate(sorted_domains[:3], 1):
                        short_id, full_id = _format_protein_id(d['protein_id'])
                        formatted_parts.append(f"    {i}. Protein: {short_id}")
                        formatted_parts.append(f"       Position: {d['position']} aa")
            
            # Handle protein-specific information with enhanced genomic context
            formatted_parts.append("\nPROTEIN ANALYSIS:")
            unique_proteins = {}
            
            # Deduplicate proteins by ID to avoid showing same protein multiple times
            for item in context.structured_data:
                # Handle both legacy field names (protein_id) and Neo4j field names (p.id)
                protein_id = item.get('protein_id') or item.get('p.id')
                if protein_id:
                    if protein_id not in unique_proteins:
                        unique_proteins[protein_id] = item
                    else:
                        # Merge data from multiple records for same protein
                        for key, value in item.items():
                            if key not in unique_proteins[protein_id] or not unique_proteins[protein_id][key]:
                                unique_proteins[protein_id][key] = value
            
            for i, (protein_id, item) in enumerate(list(unique_proteins.items())[:2]):  # Show max 2 proteins
                short_id, full_id = _format_protein_id(protein_id)
                
                # Extract gene ID (protein ID without 'protein:' prefix)
                gene_id = protein_id.replace('protein:', '') if protein_id.startswith('protein:') else protein_id
                short_gene_id, full_gene_id = _format_protein_id(gene_id)
                
                # Extract genome name from protein ID structure
                # Example: protein:RIFCSPHIGHO2_01_FULL_Gammaproteobacteria_61_200_...
                genome_id = 'N/A'
                contig_id = 'N/A'
                if '_FULL_' in protein_id:
                    parts = protein_id.split('_FULL_')
                    if len(parts) > 1:
                        genome_part = parts[1].split('_')
                        if len(genome_part) > 0:
                            genome_id = genome_part[0]  # Extract first part after _FULL_
                        
                        # Extract contig ID (full protein name with last '_'-delimited field removed)
                        # Example: RIFCSPHIGHO2_01_FULL_Gammaproteobacteria_61_200_rifcsphigho2_01_scaffold_513609_2 
                        #       -> RIFCSPHIGHO2_01_FULL_Gammaproteobacteria_61_200_rifcsphigho2_01_scaffold_513609
                        if '_' in gene_id:
                            parts = gene_id.split('_')
                            if len(parts) > 1:
                                contig_id = '_'.join(parts[:-1])  # Remove last field
                
                formatted_parts.append(f"\nProtein {i+1}:")
                formatted_parts.append(f"  • Protein: {short_id}")
                formatted_parts.append(f"    Full ID: {full_id}")
                formatted_parts.append(f"  • Gene: {short_gene_id}")
                formatted_parts.append(f"    Full ID: {full_gene_id}")
                formatted_parts.append(f"  • Genome: {genome_id}")
                formatted_parts.append(f"  • Contig: {contig_id}")
                
                # Enhanced genomic coordinates with quantitative context
                # Handle both legacy field names and Neo4j field names
                start = item.get('gene_start') or item.get('g.startCoordinate', 'N/A')
                end = item.get('gene_end') or item.get('g.endCoordinate', 'N/A')
                strand = item.get('gene_strand') or item.get('g.strand', 'N/A')
                
                if start != 'N/A' and end != 'N/A':
                    strand_symbol = "+" if str(strand) == "1" else "-" if str(strand) == "-1" else strand
                    
                    # Calculate gene length in bp using safe formatting
                    try:
                        gene_length_bp = abs(int(str(end)) - int(str(start))) + 1
                        formatted_parts.append(f"  • Genomic Location: {_safe_format_int(start)}-{_safe_format_int(end)} bp (strand {strand_symbol})")
                        formatted_parts.append(f"  • Gene Length: {_safe_format_int(gene_length_bp)} bp")
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
                    
                    # Show all domain families
                    if item.get('protein_families') and any(item['protein_families']):
                        families = [f for f in item['protein_families'] if f and f != 'None']
                        if families:
                            formatted_parts.append(f"    - Families: {', '.join(families)}")
                    
                    # Show all domain descriptions  
                    if item.get('domain_descriptions') and any(item['domain_descriptions']):
                        descriptions = [d for d in item['domain_descriptions'] if d and d != 'None']
                        if descriptions:
                            formatted_parts.append(f"    - Functions: {', '.join(descriptions)}")
                    
                    
                    # Skip domain positions unless specifically relevant for analysis
                    # (Positions are technical details that clutter LLM context unless needed)
                
                # Enhanced KEGG functional information
                # Handle both legacy field names and Neo4j field names
                kegg_id = item.get('kegg_functions') or item.get('ko.id')
                kegg_desc = item.get('kegg_descriptions') or item.get('ko.description')
                
                # Convert single values to lists for consistent processing
                if kegg_id and not isinstance(kegg_id, list):
                    kegg_id = [kegg_id] if kegg_id else []
                if kegg_desc and not isinstance(kegg_desc, list):
                    kegg_desc = [kegg_desc] if kegg_desc else []
                
                if kegg_id and any(kegg_id):
                    functions = [f for f in kegg_id if f and f != 'None']
                    if functions:
                        formatted_parts.append(f"  • KEGG Functions: {', '.join(functions[:2])}")
                        
                        # Show all function descriptions
                        if kegg_desc and any(kegg_desc):
                            descriptions = [d for d in kegg_desc if d and d != 'None']
                            if descriptions:
                                formatted_parts.append(f"    - Details: {', '.join(descriptions)}")
                
                # Handle new neighbor analysis format from DSPy queries
                neighbors = item.get('neighbors', [])
                neighbor_functions = item.get('neighbor_functions', [])
                neighbor_domains = item.get('neighbor_domains', [])
                
                if neighbors and any(neighbors):
                    # Clean up neighbor data
                    clean_neighbors = [n for n in neighbors if n and n != 'None']
                    clean_functions = [f for f in neighbor_functions if f and f != 'None'] if neighbor_functions else []
                    clean_domains = [d for d in neighbor_domains if d and d != 'None'] if neighbor_domains else []
                    
                    if clean_neighbors:
                        formatted_parts.append(f"  • Genomic Neighborhood Analysis:")
                        formatted_parts.append(f"    - {len(clean_neighbors)} neighboring proteins within 10kb")
                        
                        if clean_functions:
                            # Show unique neighbor functions (avoid duplicates)
                            unique_functions = list(set(clean_functions))[:5]  # Show top 5 unique functions
                            formatted_parts.append(f"    - Neighbor Functions: {', '.join(unique_functions)}")
                        
                        if clean_domains:
                            # Show unique neighbor domains
                            unique_domains = list(set(clean_domains))[:5]  # Show top 5 unique domains
                            formatted_parts.append(f"    - Neighbor Domains: {', '.join(unique_domains)}")
                
                # Handle enhanced neighbor_details format with distances and directions
                if item.get('neighbor_details'):
                    try:
                        neighbor_details = [n for n in item['neighbor_details'] if n and n.get('neighbor_id')]
                        if neighbor_details:
                            formatted_parts.append(f"  • Detailed Genomic Neighborhood Analysis:")
                            formatted_parts.append(f"    - {len(neighbor_details)} neighboring proteins with precise positioning")
                        
                        # Sort neighbors by distance for proximity analysis
                        sorted_neighbors = sorted(neighbor_details, key=lambda x: x.get('distance', 0))
                        
                        # Analyze close neighbors (0-200bp)
                        close_neighbors = [n for n in sorted_neighbors if n.get('distance', float('inf')) < 200]
                        if close_neighbors:
                            formatted_parts.append(f"    - Close neighbors (0-200bp): {len(close_neighbors)}")
                            for neighbor in close_neighbors[:3]:  # Show top 3 closest
                                distance = neighbor.get('distance', 'unknown')
                                direction = neighbor.get('direction', 'unknown')
                                strand = neighbor.get('neighbor_strand', 'unknown')
                                function = neighbor.get('function', 'unknown function')
                                # Handle null function values
                                if function is None:
                                    function = 'unknown function'
                                target_strand = item.get('gene_strand') or item.get('g.strand')
                                same_strand = str(strand) == str(target_strand)
                                cotranscription = "likely co-transcribed" if same_strand and distance < 200 else "different regulation"
                                formatted_parts.append(f"      • {distance}bp {direction}, strand {strand} ({cotranscription}): {function[:50]}")
                        
                        # Analyze proximal neighbors (200-500bp)
                        proximal_neighbors = [n for n in sorted_neighbors if 200 <= n.get('distance', float('inf')) < 500]
                        if proximal_neighbors:
                            formatted_parts.append(f"    - Proximal neighbors (200-500bp): {len(proximal_neighbors)}")
                            for neighbor in proximal_neighbors[:2]:  # Show top 2
                                distance = neighbor.get('distance', 'unknown')
                                direction = neighbor.get('direction', 'unknown')
                                strand = neighbor.get('neighbor_strand', 'unknown')
                                function = neighbor.get('function', 'unknown function')
                                # Handle null function values
                                if function is None:
                                    function = 'unknown function'
                                target_strand = item.get('gene_strand') or item.get('g.strand')
                                same_strand = str(strand) == str(target_strand)
                                regulation = "same strand" if same_strand else "different strand"
                                formatted_parts.append(f"      • {distance}bp {direction}, strand {strand} ({regulation}): {function[:50]}")
                        
                        # Analyze distal neighbors (>500bp)
                        distal_neighbors = [n for n in sorted_neighbors if n.get('distance', float('inf')) > 500]
                        if distal_neighbors:
                            formatted_parts.append(f"    - Distal neighbors (>500bp): {len(distal_neighbors)}")
                            for neighbor in distal_neighbors[:2]:  # Show top 2
                                distance = neighbor.get('distance', 'unknown')
                                direction = neighbor.get('direction', 'unknown')
                                function = neighbor.get('function', 'unknown function')
                                # Handle null function values
                                if function is None:
                                    function = 'unknown function'
                                formatted_parts.append(f"      • {distance}bp {direction}: {function[:50]}")
                    except Exception as e:
                        print(f"Error processing neighbor_details: {e}")
                        # Fall back to basic neighbor info if available
                        if item.get('neighbors'):
                            formatted_parts.append(f"  • Basic Neighborhood Analysis:")
                            formatted_parts.append(f"    - {len(item['neighbors'])} neighboring proteins within 10kb")
                
                # Enhanced genomic neighborhood analysis using detailed_neighbors
                if item.get('detailed_neighbors'):
                    neighbors = [n for n in item['detailed_neighbors'] if n and n.get('protein_id')]
                    if neighbors:
                        # Analyze functional themes
                        target_gene = {
                            'gene_start': item.get('gene_start'),
                            'gene_strand': item.get('gene_strand'),
                            'protein_id': item.get('protein_id')
                        }
                        
                        # Filter neighbors to same scaffold first
                        target_protein_id = item.get('protein_id', '')
                        target_scaffold = None
                        if 'scaffold_' in target_protein_id:
                            parts = target_protein_id.split('scaffold_')
                            if len(parts) > 1:
                                scaffold_part = parts[1].split('_')[0]
                                target_scaffold = f"scaffold_{scaffold_part}"
                        
                        scaffold_neighbors = []
                        if target_scaffold:
                            for neighbor in neighbors:
                                if neighbor.get('protein_id') and 'scaffold_' in neighbor['protein_id']:
                                    neighbor_parts = neighbor['protein_id'].split('scaffold_')
                                    if len(neighbor_parts) > 1:
                                        neighbor_scaffold_part = neighbor_parts[1].split('_')[0]
                                        neighbor_scaffold = f"scaffold_{neighbor_scaffold_part}"
                                        if neighbor_scaffold == target_scaffold:
                                            scaffold_neighbors.append(neighbor)
                        
                        functional_analysis = _analyze_neighbor_functions(scaffold_neighbors)
                        formatted_neighbors = _format_neighbor_context(neighbors, target_gene)
                        
                        # Count upstream vs downstream
                        if item.get('gene_start'):
                            target_pos = int(item['gene_start'])
                            upstream_count = sum(1 for n in neighbors if n.get('position') and n['position'] < target_pos)
                            downstream_count = len(neighbors) - upstream_count
                            
                            formatted_parts.append(f"  • Genomic Neighborhood Analysis:")
                            formatted_parts.append(f"    - {len(neighbors)} proteins within 5kb ({upstream_count} upstream, {downstream_count} downstream)")
                            
                            # Show detailed neighbor information
                            if formatted_neighbors:
                                upstream_neighbors = [n for n in formatted_neighbors if n['direction'] == 'upstream']
                                downstream_neighbors = [n for n in formatted_neighbors if n['direction'] == 'downstream']
                                
                                if upstream_neighbors:
                                    formatted_parts.append(f"    \n    Upstream Neighbors:")
                                    for neighbor in upstream_neighbors[:2]:  # Show top 2 upstream
                                        short_id, _ = _format_protein_id(neighbor['protein_id'])
                                        formatted_parts.append(f"    • {short_id} ({_safe_format_int(neighbor['distance'])}bp upstream, strand {neighbor['strand']}):")
                                        
                                        # Show all PFAM domains
                                        pfam_domains = [d for d in neighbor.get('pfam_id', []) if d and d != 'None']
                                        if pfam_domains:
                                            formatted_parts.append(f"      - PFAM: {', '.join(pfam_domains)}")
                                        
                                        # Show all functions
                                        descriptions = neighbor.get('pfam_description', []) + neighbor.get('kegg_descriptions', [])
                                        descriptions = [d for d in descriptions if d and d != 'None']
                                        if descriptions:
                                            formatted_parts.append(f"      - Function: {', '.join(descriptions)}")
                                        
                                        # Show KEGG orthologs
                                        kegg_ko = [k for k in neighbor.get('kegg_ko', []) if k and k != 'None']
                                        if kegg_ko:
                                            formatted_parts.append(f"      - KEGG: {', '.join(kegg_ko[:2])}")
                                
                                if downstream_neighbors:
                                    formatted_parts.append(f"    \n    Downstream Neighbors:")
                                    for neighbor in downstream_neighbors[:2]:  # Show top 2 downstream
                                        short_id, _ = _format_protein_id(neighbor['protein_id'])
                                        formatted_parts.append(f"    • {short_id} ({_safe_format_int(neighbor['distance'])}bp downstream, strand {neighbor['strand']}):")
                                        
                                        # Show all PFAM domains
                                        pfam_domains = [d for d in neighbor.get('pfam_id', []) if d and d != 'None']
                                        if pfam_domains:
                                            formatted_parts.append(f"      - PFAM: {', '.join(pfam_domains)}")
                                        
                                        # Show all functions  
                                        descriptions = neighbor.get('pfam_description', []) + neighbor.get('kegg_descriptions', [])
                                        descriptions = [d for d in descriptions if d and d != 'None']
                                        if descriptions:
                                            formatted_parts.append(f"      - Function: {', '.join(descriptions)}")
                                        
                                        # Show KEGG orthologs
                                        kegg_ko = [k for k in neighbor.get('kegg_ko', []) if k and k != 'None']
                                        if kegg_ko:
                                            formatted_parts.append(f"      - KEGG: {', '.join(kegg_ko[:2])}")
                            
                            # Add biological context summary
                            formatted_parts.append(f"    \n    Biological Context:")
                            
                            if functional_analysis.get('dominant_theme'):
                                theme = functional_analysis['dominant_theme'].replace('_', ' ').title()
                                formatted_parts.append(f"    • Functional theme: {theme}")
                            
                            if functional_analysis.get('themes'):
                                themes = functional_analysis['themes']
                                active_themes = [k.replace('_', ' ').title() for k, v in themes.items() if v > 0]
                                if active_themes:
                                    formatted_parts.append(f"    • Neighborhood functions: {', '.join(active_themes[:3])}")
                        else:
                            formatted_parts.append(f"  • Genomic Neighborhood: {len(neighbors)} proteins within 5kb")
                
        # Handle general structured data that doesn't match specific patterns
        if context.structured_data and not any(['p.id' in str(item) or 'protein_id' in str(item) for item in context.structured_data]):
            formatted_parts.append("\nSTRUCTURED DATA ANALYSIS:")
            
            # Handle contig/genomic queries
            if any('contig' in str(item) for item in context.structured_data):
                formatted_parts.append("  GENOMIC CONTIGS:")
                for i, item in enumerate(context.structured_data[:10]):  # Show up to 10 contigs
                    contig_id = item.get('contig_id', 'Unknown')
                    count = item.get('ribosomal_gene_count') or item.get('gene_count') or item.get('count', 0)
                    formatted_parts.append(f"    {i+1}. {contig_id}: {count} genes")
            
            # Handle pathway queries
            elif any('pathway' in str(item) for item in context.structured_data):
                formatted_parts.append("  PATHWAY ANALYSIS:")
                for i, item in enumerate(context.structured_data[:5]):
                    pathway_name = item.get('pathway_name') or item.get('name', 'Unknown pathway')
                    count = item.get('protein_count') or item.get('ko_count') or item.get('count', 0)
                    formatted_parts.append(f"    {i+1}. {pathway_name}: {count} proteins")
            
            # Handle general aggregation queries
            else:
                formatted_parts.append("  QUERY RESULTS:")
                for i, item in enumerate(context.structured_data[:10]):
                    # Show all non-metadata fields
                    item_parts = []
                    for key, value in item.items():
                        if not key.startswith('_'):  # Skip metadata
                            item_parts.append(f"{key}: {value}")
                    if item_parts:
                        formatted_parts.append(f"    {i+1}. {', '.join(item_parts)}")

        # Handle semantic similarity data with enhanced formatting
        if context.semantic_data:
            formatted_parts.append("\nFUNCTIONALLY SIMILAR PROTEINS:")
            for i, item in enumerate(context.semantic_data[:3]):  # Show top 3 similar proteins
                similarity = item.get('similarity', 0)
                protein_id = item.get('protein_id', 'Unknown')
                short_id, full_id = _format_protein_id(protein_id)
                
                formatted_parts.append(f"  {i+1}. {short_id} (ESM2 similarity: {similarity:.3f})")
                formatted_parts.append(f"     Genome: {item.get('genome_id', 'Unknown')}")
                
                # Add biological interpretation of similarity scores with enhanced context
                if similarity > 0.95:
                    formatted_parts.append(f"     Interpretation: IDENTICAL/ORTHOLOG (>95% similarity) - functionally equivalent")
                elif similarity > 0.8:
                    formatted_parts.append(f"     Interpretation: HIGHLY SIMILAR - likely functional ortholog with conserved structure")
                elif similarity > 0.6:
                    formatted_parts.append(f"     Interpretation: MODERATELY SIMILAR - possible functional analog or domain similarity")
                elif similarity > 0.4:
                    formatted_parts.append(f"     Interpretation: WEAKLY SIMILAR - distantly related or shared domain")
                else:
                    formatted_parts.append(f"     Interpretation: LOW SIMILARITY - possible distant evolutionary relationship")
                
                # Add functional annotation comparison if available
                if item.get('pfam_domains'):
                    domains = [d for d in item['pfam_domains'] if d and d != 'None']
                    if domains:
                        formatted_parts.append(f"     Domains: {', '.join(domains[:3])}")
                
                if item.get('kegg_functions'):
                    functions = [f for f in item['kegg_functions'] if f and f != 'None']
                    if functions:
                        formatted_parts.append(f"     KEGG Functions: {', '.join(functions[:2])}")
                
                # Add sequence length comparison if available
                if item.get('protein_length'):
                    length = item['protein_length']
                    formatted_parts.append(f"     Protein Length: {length} amino acids")
                
                # Add genomic context if available
                if item.get('start_coordinate') and item.get('end_coordinate'):
                    start = item['start_coordinate']
                    end = item['end_coordinate']
                    strand = item.get('strand', '?')
                    strand_symbol = "+" if str(strand) == "1" else "-" if str(strand) == "-1" else strand
                    formatted_parts.append(f"     Genomic Location: {_safe_format_int(start)}-{_safe_format_int(end)} bp (strand {strand_symbol})")
                
                # Show full ID if different from short ID (collapsed to save space)
                if short_id != full_id:
                    formatted_parts.append(f"     [Full ID: {full_id}]")
        
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