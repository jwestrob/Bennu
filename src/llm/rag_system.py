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
    - KEGGOrtholog nodes: Use .description property for function descriptions, .id for KO IDs
    - Domain nodes: Use .description property for PFAM descriptions, .id for family names  
    - DomainAnnotation nodes: Use .id property containing "/domain/DOMAIN_NAME/start-end" patterns
    - Protein nodes: Use .id property, linked via hasFunction->KEGGOrtholog and hasDomain->DomainAnnotation->domainFamily->Domain
    - Gene nodes: Use .id property, linked via belongsToGenome->Genome
    - Genome nodes: Use .id property
    
    IMPORTANT: When users ask about "domains" (e.g., "GGDEF domains"), search DomainAnnotation.id field which contains patterns like "/domain/GGDEF/"
    
    EXAMPLE QUERIES:
    - For KEGG function: MATCH (ko:KEGGOrtholog {id: 'K12345'}) RETURN ko.id, ko.description
    - For protein function: MATCH (p:Protein {id: 'protein_id'})-[:hasFunction]->(ko:KEGGOrtholog) RETURN p.id, ko.id, ko.description
    - For domain search: MATCH (p:Protein)-[:hasDomain]->(d:DomainAnnotation) WHERE d.id CONTAINS '/domain/GGDEF/' RETURN p.id, d.id, d.bitscore
    - For family search: MATCH (pf:Domain {id: 'GGDEF'}) RETURN pf.id, pf.description
    """
    
    question = dspy.InputField(desc="User's question")
    query_type = dspy.InputField(desc="Classified query type")
    neo4j_query = dspy.OutputField(desc="Cypher query for Neo4j using CORRECT schema: For domains use DomainAnnotation.id CONTAINS '/domain/NAME/', for families use Domain.description")
    protein_search = dspy.OutputField(desc="Protein ID or description for semantic search (if needed)")
    search_strategy = dspy.OutputField(desc="How to combine the results")


class GenomicAnswerer(dspy.Signature):
    """Generate comprehensive answers using genomic context with biological insights."""
    
    question = dspy.InputField(desc="Original user question")
    context = dspy.InputField(desc="Retrieved genomic data and metadata")
    answer = dspy.OutputField(desc="Comprehensive biological answer focusing on: 1) Protein function and domains, 2) Genomic context and neighbors, 3) Evolutionary/metabolic significance. Avoid mentioning self-similarity.")
    confidence = dspy.OutputField(desc="Confidence level: high, medium, or low")
    citations = dspy.OutputField(desc="Data sources used: genome IDs, protein families, KEGG functions, etc.")


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
        """Format context for LLM consumption with biological emphasis."""
        formatted_parts = []
        
        if context.structured_data:
            formatted_parts.append("PROTEIN AND GENOMIC INFORMATION:")
            for i, item in enumerate(context.structured_data[:3]):  # Focus on key data
                # Format protein information more clearly
                if 'protein_id' in item:
                    formatted_parts.append(f"\nProtein Details:")
                    formatted_parts.append(f"  • ID: {item.get('protein_id', 'N/A')}")
                    formatted_parts.append(f"  • Gene: {item.get('gene_id', 'N/A')}")
                    formatted_parts.append(f"  • Genome: {item.get('genome_id', 'N/A')}")
                    
                    # Enhanced domain family information with descriptions
                    if item.get('protein_families') and any(item['protein_families']):
                        families = [f for f in item['protein_families'] if f]
                        if families:
                            formatted_parts.append(f"  • Domain Families: {', '.join(families[:5])}")
                    
                    # Add domain family descriptions if available
                    if item.get('domain_descriptions') and any(item['domain_descriptions']):
                        descriptions = [d for d in item['domain_descriptions'] if d and d != 'None']
                        if descriptions:
                            formatted_parts.append(f"  • Domain Functions: {', '.join(descriptions[:3])}")
                    
                    if item.get('pfam_accessions') and any(item['pfam_accessions']):
                        accessions = [a for a in item['pfam_accessions'] if a]
                        if accessions:
                            formatted_parts.append(f"  • PFAM Accessions: {', '.join(accessions[:3])}")
                    
                    # Enhanced KEGG functional information
                    if item.get('kegg_functions') and any(item['kegg_functions']):
                        functions = [f for f in item['kegg_functions'] if f]
                        if functions:
                            formatted_parts.append(f"  • KEGG Functions: {', '.join(functions[:3])}")
                    
                    # Add KEGG function descriptions if available
                    if item.get('kegg_descriptions') and any(item['kegg_descriptions']):
                        descriptions = [d for d in item['kegg_descriptions'] if d and d != 'None']
                        if descriptions:
                            formatted_parts.append(f"  • Function Details: {descriptions[0][:100]}...")
                    
                    # Enhanced domain annotation details
                    if item.get('domain_count', 0) > 0:
                        formatted_parts.append(f"  • Domain Annotations: {item['domain_count']} detected")
                        
                        # Show domain instances with scores if available
                        if item.get('domain_scores') and any(item['domain_scores']):
                            scores = [f"{s:.1f}" for s in item['domain_scores'][:3] if s]
                            formatted_parts.append(f"  • Domain Bitscores: {', '.join(scores)}")
                        
                        # Show domain positions if available
                        if item.get('domain_positions') and any(item['domain_positions']):
                            positions = [pos for pos in item['domain_positions'][:3] if pos]
                            formatted_parts.append(f"  • Domain Positions: {', '.join(positions)}")
                    
                    if item.get('neighboring_proteins') and any(item['neighboring_proteins']):
                        neighbors = [n for n in item['neighboring_proteins'] if n]
                        if neighbors:
                            formatted_parts.append(f"  • Genomic Neighbors: {len(neighbors)} nearby proteins")
                    
                    if item.get('neighborhood_families') and any(item['neighborhood_families']):
                        neighborhoods = [n for n in item['neighborhood_families'] if n]
                        if neighborhoods:
                            formatted_parts.append(f"  • Neighborhood Functions: {', '.join(neighborhoods[:3])}")
                else:
                    formatted_parts.append(f"  {i+1}. {json.dumps(item, indent=2)}")
        
        if context.semantic_data:
            formatted_parts.append("\nFUNCTIONALLY SIMILAR PROTEINS:")
            for i, item in enumerate(context.semantic_data[:4]):  # Fewer but more meaningful
                formatted_parts.append(f"  • {item.get('protein_id', 'Unknown')} (similarity: {item.get('similarity', 0):.3f})")
                formatted_parts.append(f"    from genome: {item.get('genome_id', 'Unknown')}")
        
        return "\n".join(formatted_parts) if formatted_parts else "No relevant context found."
    
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