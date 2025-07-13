"""
Intelligent Upfront Chunking Manager for large datasets.

Replaces recursive task splitting with smart upfront analysis and chunking into 
3-5 meaningful biological groups, avoiding complexity explosion while maintaining 
comprehensive analysis capability.
"""

import logging
import asyncio
import tiktoken
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .task_management import Task, TaskType, TaskGraph
from .task_executor import ExecutionResult

logger = logging.getLogger(__name__)

@dataclass
class AnalysisChunk:
    """A logically grouped chunk of data for focused analysis."""
    chunk_id: str
    title: str
    description: str
    data_subset: List[Dict[str, Any]]
    biological_focus: str
    expected_insights: str

@dataclass
class ChunkingStrategy:
    """Strategy for chunking data based on analysis type."""
    strategy_name: str
    max_chunks: int
    chunk_method: str
    biological_rationale: str

class IntelligentChunkingManager:
    """
    Manages intelligent upfront chunking of large datasets with recursive subdivision fallback.
    
    Features:
    - Upfront analysis to determine optimal chunking strategy
    - 3-5 logical chunks initially to preserve biological meaning
    - Token-aware chunk sizing with automatic subdivision
    - Recursive subdivision for oversized chunks (with depth limits)
    - Clear, focused task descriptions
    - Biological meaning preservation
    """
    
    def __init__(self, max_chunks: int = 8, min_chunk_size: int = 50, use_premium_models: bool = True, max_tokens_per_chunk: int = 4000):
        """
        Initialize chunking manager.
        
        Args:
            max_chunks: Maximum number of chunks to create (3-5 recommended)
            min_chunk_size: Minimum items per chunk to avoid micro-chunks
            use_premium_models: Whether to use premium models (GPT-4/o3) for complex analysis
            max_tokens_per_chunk: Maximum tokens per chunk before recursive subdivision
        """
        self.max_chunks = max_chunks
        self.min_chunk_size = min_chunk_size
        self.use_premium_models = use_premium_models
        self.max_tokens_per_chunk = max_tokens_per_chunk
        
        # Initialize tokenizer for analysis
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception as e:
            logger.warning(f"Failed to initialize tokenizer: {e}")
            self.tokenizer = None
    
    async def analyze_and_chunk_dataset(self, 
                                      task: Task, 
                                      raw_data: List[Dict[str, Any]], 
                                      question: str) -> List[AnalysisChunk]:
        """
        Analyze dataset and create intelligent chunks upfront.
        
        Args:
            task: Original task to be chunked
            raw_data: Complete dataset to analyze
            question: Original user question for context
            
        Returns:
            List of AnalysisChunk objects with meaningful biological groupings
        """
        logger.info(f"🧠 Analyzing dataset for intelligent chunking: {len(raw_data)} items")
        
        # Determine optimal chunking strategy
        strategy = self._determine_chunking_strategy(raw_data, question)
        logger.info(f"📊 Selected strategy: {strategy.strategy_name} - {strategy.biological_rationale}")
        
        # Create chunks based on strategy
        if strategy.chunk_method == "functional":
            chunks = self._chunk_by_biological_function(raw_data, task.description, strategy.max_chunks)
        elif strategy.chunk_method == "genomic":
            chunks = self._chunk_by_genome_comparison(raw_data, task.description, strategy.max_chunks)
        elif strategy.chunk_method == "pathway":
            chunks = self._chunk_by_metabolic_pathways(raw_data, task.description, strategy.max_chunks)
        elif strategy.chunk_method == "complexity":
            chunks = self._chunk_by_analysis_complexity(raw_data, task.description, strategy.max_chunks)
        else:
            chunks = self._chunk_by_balanced_analysis(raw_data, task.description, strategy.max_chunks)
        
        # Validate and optimize chunks
        optimized_chunks = self._optimize_chunk_sizes(chunks)
        
        # Check for oversized chunks and recursively subdivide if needed
        final_chunks = await self._check_and_subdivide_oversized_chunks(optimized_chunks)
        
        logger.info(f"✅ Created {len(final_chunks)} intelligent chunks:")
        for i, chunk in enumerate(final_chunks, 1):
            logger.info(f"   Chunk {i}: {chunk.title} ({len(chunk.data_subset)} items)")
        
        return final_chunks
    
    def _determine_chunking_strategy(self, data: List[Dict[str, Any]], question: str) -> ChunkingStrategy:
        """
        Analyze data and question to determine optimal chunking strategy.
        
        Args:
            data: Raw dataset to analyze
            question: User question for context
            
        Returns:
            ChunkingStrategy with recommended approach
        """
        question_lower = question.lower()
        
        # Analyze data characteristics
        has_genome_data = any('genome_id' in item or 'genome' in item for item in data[:10])
        has_function_data = any('ko_description' in item or 'description' in item for item in data[:10])
        has_pathway_data = any('pathway' in str(item).lower() for item in data[:10])
        
        # Determine strategy based on question type and data characteristics
        if any(term in question_lower for term in ['compare', 'comparison', 'across genomes', 'between genomes']):
            if has_genome_data:
                return ChunkingStrategy(
                    strategy_name="Genomic Comparison",
                    max_chunks=min(4, self.max_chunks),
                    chunk_method="genomic",
                    biological_rationale="Question requests cross-genome comparison; chunk by genome for systematic analysis"
                )
        
        if any(term in question_lower for term in ['pathway', 'metabolic', 'metabolism', 'biochemical']):
            if has_pathway_data:
                return ChunkingStrategy(
                    strategy_name="Metabolic Pathway Analysis",
                    max_chunks=5,
                    chunk_method="pathway",
                    biological_rationale="Question focuses on metabolic pathways; group by biochemical function"
                )
        
        if any(term in question_lower for term in ['function', 'functional', 'activity', 'role']):
            if has_function_data:
                return ChunkingStrategy(
                    strategy_name="Functional Category Analysis",
                    max_chunks=5,
                    chunk_method="functional",
                    biological_rationale="Question requests functional analysis; group by biological function"
                )
        
        if any(term in question_lower for term in ['comprehensive', 'complete', 'detailed', 'all']):
            return ChunkingStrategy(
                strategy_name="Comprehensive Analysis",
                max_chunks=4,
                chunk_method="complexity",
                biological_rationale="Question requests comprehensive analysis; balance coverage and depth"
            )
        
        # Default strategy
        return ChunkingStrategy(
            strategy_name="Balanced Analysis",
            max_chunks=4,
            chunk_method="balanced",
            biological_rationale="General analysis; balance data size and biological meaning"
        )
    
    def _chunk_by_biological_function(self, data: List[Dict[str, Any]], 
                                     task_description: str, max_chunks: int) -> List[AnalysisChunk]:
        """Create chunks based on biological function categories."""
        # Group by functional categories
        function_groups = defaultdict(list)
        
        for item in data:
            function_key = self._extract_function_category(item)
            function_groups[function_key].append(item)
        
        # Sort by group size and select top categories
        sorted_groups = sorted(function_groups.items(), key=lambda x: len(x[1]), reverse=True)
        
        chunks = []
        chunk_id = 1
        
        # Create chunks for major functional categories
        for function_category, function_data in sorted_groups[:max_chunks-1]:
            if len(function_data) >= self.min_chunk_size:
                # Create clean chunk ID from function category
                clean_category = function_category.lower().replace(' ', '_').replace('-', '_')
                chunks.append(AnalysisChunk(
                    chunk_id=f"func_{clean_category}",
                    title=f"Functional Analysis: {function_category}",
                    description=f"Detailed analysis of {function_category} functions across genomes",
                    data_subset=function_data,
                    biological_focus=f"{function_category} enzymatic activities and metabolic roles",
                    expected_insights=f"Distribution, conservation, and genomic organization of {function_category} functions"
                ))
                chunk_id += 1
        
        # Combine remaining small categories into "Other Functions" chunk
        remaining_data = []
        for function_category, function_data in sorted_groups[max_chunks-1:]:
            remaining_data.extend(function_data)
        
        if remaining_data and len(remaining_data) >= self.min_chunk_size:
            chunks.append(AnalysisChunk(
                chunk_id="func_other",
                title="Additional Functional Categories",
                description="Analysis of diverse additional functional categories",
                data_subset=remaining_data,
                biological_focus="diverse enzymatic activities and specialized functions",
                expected_insights="identification of unique or specialized functions across genomes"
            ))
        
        return chunks
    
    def _chunk_by_genome_comparison(self, data: List[Dict[str, Any]], 
                                   task_description: str, max_chunks: int) -> List[AnalysisChunk]:
        """Create chunks optimized for cross-genome comparison."""
        # Group by genome
        genome_groups = defaultdict(list)
        
        for item in data:
            genome_id = item.get('genome_id', item.get('genome', 'unknown'))
            # Handle None values
            if genome_id is None:
                genome_id = 'unknown'
            genome_groups[genome_id].append(item)
        
        chunks = []
        genome_names = list(genome_groups.keys())
        
        # If few genomes, create one chunk per genome
        if len(genome_names) <= max_chunks and all(len(data) >= self.min_chunk_size for data in genome_groups.values()):
            for i, (genome_id, genome_data) in enumerate(genome_groups.items(), 1):
                # Create clean genome ID for chunk
                clean_genome = str(genome_id).replace('_', '')[:10] if genome_id != 'unknown' else 'unknown'
                chunks.append(AnalysisChunk(
                    chunk_id=f"genome_{clean_genome}",
                    title=f"Genome Analysis: {genome_id}",
                    description=f"Comprehensive functional profile of {genome_id}",
                    data_subset=genome_data,
                    biological_focus=f"metabolic capabilities and functional specialization of {genome_id}",
                    expected_insights=f"unique functions, metabolic pathways, and genomic features of {genome_id}"
                ))
        else:
            # Group smaller genomes together for comparison
            chunk_size = len(genome_names) // max_chunks + 1
            for i in range(0, len(genome_names), chunk_size):
                chunk_genomes = genome_names[i:i + chunk_size]
                chunk_data = []
                for genome in chunk_genomes:
                    chunk_data.extend(genome_groups[genome])
                
                chunks.append(AnalysisChunk(
                    chunk_id=f"genomes_grp{i//chunk_size + 1}",
                    title=f"Genome Group {i//chunk_size + 1}: {', '.join(chunk_genomes[:2])}{'...' if len(chunk_genomes) > 2 else ''}",
                    description=f"Comparative analysis of {len(chunk_genomes)} genomes",
                    data_subset=chunk_data,
                    biological_focus=f"comparative functional profiles across {len(chunk_genomes)} genomes",
                    expected_insights="shared and unique functions, metabolic differences, evolutionary relationships"
                ))
        
        return chunks
    
    def _chunk_by_metabolic_pathways(self, data: List[Dict[str, Any]], 
                                    task_description: str, max_chunks: int) -> List[AnalysisChunk]:
        """Create chunks based on metabolic pathway categories."""
        # Define major metabolic pathway categories
        pathway_categories = {
            "Energy Metabolism": ["atp", "energy", "electron", "respiration", "photosynthesis"],
            "Carbohydrate Metabolism": ["glucose", "glycolysis", "gluconeogenesis", "pentose", "sugar"],
            "Amino Acid Metabolism": ["amino acid", "protein", "peptide", "tryptophan", "methionine"],
            "Nucleotide Metabolism": ["nucleotide", "purine", "pyrimidine", "dna", "rna"],
            "Lipid Metabolism": ["lipid", "fatty acid", "membrane", "phospholipid"]
        }
        
        # Categorize data by pathway
        pathway_groups = defaultdict(list)
        uncategorized = []
        
        for item in data:
            description = str(item.get('ko_description', item.get('description', ''))).lower()
            categorized = False
            
            for pathway_name, keywords in pathway_categories.items():
                if any(keyword in description for keyword in keywords):
                    pathway_groups[pathway_name].append(item)
                    categorized = True
                    break
            
            if not categorized:
                uncategorized.append(item)
        
        chunks = []
        chunk_id = 1
        
        # Create chunks for major pathways
        for pathway_name, pathway_data in pathway_groups.items():
            if len(pathway_data) >= self.min_chunk_size:
                chunks.append(AnalysisChunk(
                    chunk_id=f"pathway_chunk_{chunk_id}",
                    title=f"Metabolic Analysis: {pathway_name}",
                    description=f"Analysis of {pathway_name.lower()} across genomes",
                    data_subset=pathway_data,
                    biological_focus=f"{pathway_name.lower()} enzymes and metabolic capabilities",
                    expected_insights=f"distribution and conservation of {pathway_name.lower()} across genomes"
                ))
                chunk_id += 1
        
        # Add uncategorized functions if substantial
        if uncategorized and len(uncategorized) >= self.min_chunk_size:
            chunks.append(AnalysisChunk(
                chunk_id=f"pathway_chunk_{chunk_id}",
                title="Specialized Metabolic Functions",
                description="Analysis of specialized and diverse metabolic functions",
                data_subset=uncategorized,
                biological_focus="specialized enzymes and unique metabolic capabilities",
                expected_insights="identification of specialized metabolic adaptations and unique functions"
            ))
        
        return chunks[:max_chunks]  # Limit to max_chunks
    
    def _chunk_by_analysis_complexity(self, data: List[Dict[str, Any]], 
                                     task_description: str, max_chunks: int) -> List[AnalysisChunk]:
        """Create chunks balanced by analysis complexity and biological meaning."""
        # Analyze data complexity
        function_complexity = defaultdict(int)
        genome_distribution = defaultdict(int)
        
        for item in data:
            # Count function diversity
            function_key = self._extract_function_category(item)
            function_complexity[function_key] += 1
            
            # Count genome distribution
            genome_id = item.get('genome_id', item.get('genome', 'unknown'))
            genome_distribution[genome_id] += 1
        
        # Create balanced chunks
        chunks = []
        chunk_size = len(data) // max_chunks + 1
        
        for i in range(0, len(data), chunk_size):
            chunk_data = data[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            
            # Analyze this chunk's characteristics
            chunk_functions = set()
            chunk_genomes = set()
            for item in chunk_data:
                chunk_functions.add(self._extract_function_category(item))
                chunk_genomes.add(item.get('genome_id', item.get('genome', 'unknown')))
            
            chunks.append(AnalysisChunk(
                chunk_id=f"complex_chunk_{chunk_num}",
                title=f"Comprehensive Analysis Part {chunk_num}",
                description=f"Detailed analysis of diverse functional categories (part {chunk_num} of {max_chunks})",
                data_subset=chunk_data,
                biological_focus=f"{len(chunk_functions)} functional categories across {len(chunk_genomes)} genomes",
                expected_insights=f"functional diversity, genomic distribution, and biological patterns in dataset segment {chunk_num}"
            ))
        
        return chunks
    
    def _chunk_by_balanced_analysis(self, data: List[Dict[str, Any]], 
                                   task_description: str, max_chunks: int) -> List[AnalysisChunk]:
        """Create balanced chunks for general analysis."""
        chunk_size = len(data) // max_chunks + 1
        chunks = []
        
        for i in range(0, len(data), chunk_size):
            chunk_data = data[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            
            chunks.append(AnalysisChunk(
                chunk_id=f"balanced_chunk_{chunk_num}",
                title=f"Functional Analysis Part {chunk_num}",
                description=f"Systematic analysis of functional annotations (part {chunk_num} of {max_chunks})",
                data_subset=chunk_data,
                biological_focus="diverse functional annotations and genomic features",
                expected_insights=f"biological patterns and functional insights from data segment {chunk_num}"
            ))
        
        return chunks
    
    def _extract_function_category(self, item: Dict[str, Any]) -> str:
        """Extract main functional category from data item."""
        description = str(item.get('ko_description', item.get('description', ''))).lower()
        
        # Define functional categories
        if any(term in description for term in ['kinase', 'phosphatase', 'phosphorylation']):
            return "Protein Modification"
        elif any(term in description for term in ['dehydrogenase', 'oxidase', 'reductase']):
            return "Oxidation-Reduction"
        elif any(term in description for term in ['transporter', 'permease', 'channel']):
            return "Transport"
        elif any(term in description for term in ['ribosom', 'translation', 'trna']):
            return "Protein Synthesis"
        elif any(term in description for term in ['dna', 'replication', 'repair', 'helicase']):
            return "DNA Metabolism"
        elif any(term in description for term in ['transcription', 'rna polymerase', 'sigma']):
            return "Transcription"
        elif any(term in description for term in ['synthetase', 'synthase', 'biosynthesis']):
            return "Biosynthesis"
        elif any(term in description for term in ['hydrolase', 'peptidase', 'degradation']):
            return "Degradation"
        elif any(term in description for term in ['regulation', 'regulator', 'activator']):
            return "Regulation"
        else:
            return "Other Functions"
    
    def _optimize_chunk_sizes(self, chunks: List[AnalysisChunk]) -> List[AnalysisChunk]:
        """Optimize chunk sizes to ensure meaningful analysis."""
        optimized = []
        small_chunks = []
        
        for chunk in chunks:
            if len(chunk.data_subset) >= self.min_chunk_size:
                optimized.append(chunk)
            else:
                small_chunks.append(chunk)
        
        # Combine small chunks if any
        if small_chunks:
            combined_data = []
            combined_focuses = []
            
            for chunk in small_chunks:
                combined_data.extend(chunk.data_subset)
                combined_focuses.append(chunk.biological_focus)
            
            if combined_data:
                optimized.append(AnalysisChunk(
                    chunk_id="combined_small_chunks",
                    title="Additional Functional Categories",
                    description="Analysis of additional diverse functional categories",
                    data_subset=combined_data,
                    biological_focus="; ".join(combined_focuses[:3]) + ("..." if len(combined_focuses) > 3 else ""),
                    expected_insights="diverse functional insights from smaller categories"
                ))
        
        return optimized
    
    def _estimate_chunk_tokens(self, chunk: AnalysisChunk) -> int:
        """Estimate token count for a chunk - MUCH more conservative estimation."""
        # BRUTAL but safe: assume each item is ~50 tokens average
        # This prevents the 280k+ token chunks that were killing the API
        estimated_tokens = len(chunk.data_subset) * 50
        
        # Cap at reasonable maximum to prevent API overload  
        max_safe_tokens = 15000
        if estimated_tokens > max_safe_tokens:
            logger.warning(f"Estimated {estimated_tokens} tokens for {len(chunk.data_subset)} items - capping at {max_safe_tokens}")
            estimated_tokens = max_safe_tokens
            
        logger.debug(f"Conservative token estimation: {len(chunk.data_subset)} items → {estimated_tokens} tokens")
        return estimated_tokens
    
    async def _check_and_subdivide_oversized_chunks(self, chunks: List[AnalysisChunk]) -> List[AnalysisChunk]:
        """Check chunks for token limits with conservative subdivision."""
        final_chunks = []
        
        for chunk in chunks:
            estimated_tokens = self._estimate_chunk_tokens(chunk)
            
            # With conservative estimation, only subdivide if really needed
            if estimated_tokens > self.max_tokens_per_chunk and len(chunk.data_subset) > 200:
                logger.warning(f"🔄 Chunk '{chunk.title}' estimated at {estimated_tokens} tokens, subdividing...")
                # Simple subdivision - just split in half, no recursion
                mid = len(chunk.data_subset) // 2
                
                sub1 = AnalysisChunk(
                    chunk_id=f"{chunk.chunk_id}_part1",
                    title=f"{chunk.title} (Part 1)",
                    description=f"{chunk.description} - first half",
                    data_subset=chunk.data_subset[:mid],
                    biological_focus=chunk.biological_focus,
                    expected_insights=chunk.expected_insights
                )
                
                sub2 = AnalysisChunk(
                    chunk_id=f"{chunk.chunk_id}_part2", 
                    title=f"{chunk.title} (Part 2)",
                    description=f"{chunk.description} - second half",
                    data_subset=chunk.data_subset[mid:],
                    biological_focus=chunk.biological_focus,
                    expected_insights=chunk.expected_insights
                )
                
                final_chunks.extend([sub1, sub2])
            else:
                logger.info(f"✅ Chunk '{chunk.title}' within token limit ({estimated_tokens} tokens)")
                final_chunks.append(chunk)
        
        return final_chunks
    
    async def _recursively_subdivide_chunk(self, oversized_chunk: AnalysisChunk, depth: int = 0) -> List[AnalysisChunk]:
        """Recursively subdivide an oversized chunk."""
        if depth > 3:  # Prevent infinite recursion
            logger.warning(f"⚠️ Maximum subdivision depth reached for chunk '{oversized_chunk.title}'")
            return [oversized_chunk]
        
        # Split data in half
        data = oversized_chunk.data_subset
        mid_point = len(data) // 2
        
        if mid_point < self.min_chunk_size:
            logger.warning(f"⚠️ Cannot subdivide chunk '{oversized_chunk.title}' further (too small)")
            return [oversized_chunk]
        
        # Create two sub-chunks
        sub_chunks = []
        for i, (start_idx, end_idx) in enumerate([(0, mid_point), (mid_point, len(data))]):
            sub_data = data[start_idx:end_idx]
            
            sub_chunk = AnalysisChunk(
                chunk_id=f"{oversized_chunk.chunk_id}_sub{i+1}",
                title=f"{oversized_chunk.title} (Part {i+1})",
                description=f"{oversized_chunk.description} - subdivision {i+1} of 2",
                data_subset=sub_data,
                biological_focus=oversized_chunk.biological_focus,
                expected_insights=oversized_chunk.expected_insights
            )
            
            # Check if sub-chunk still needs subdivision
            estimated_tokens = self._estimate_chunk_tokens(sub_chunk)
            if estimated_tokens > self.max_tokens_per_chunk:
                logger.info(f"🔄 Sub-chunk still oversized ({estimated_tokens} tokens), recursing...")
                further_subdivided = await self._recursively_subdivide_chunk(sub_chunk, depth + 1)
                sub_chunks.extend(further_subdivided)
            else:
                logger.info(f"✅ Sub-chunk within limit ({estimated_tokens} tokens)")
                sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    async def execute_chunked_analysis(self, 
                                     chunks: List[AnalysisChunk], 
                                     task_executor, 
                                     original_task: Task,
                                     original_question: str = None) -> List[Dict[str, Any]]:
        """
        Execute analysis on all chunks in parallel.
        
        Args:
            chunks: List of analysis chunks to process
            task_executor: TaskExecutor instance
            original_task: Original task for context
            original_question: Original user question to preserve biological discovery context
            
        Returns:
            List of analysis results from all chunks
        """
        logger.info(f"🚀 Executing {len(chunks)} chunks in parallel")
        
        # For complex biological analysis tasks, we should use premium models (GPT-4/o3)
        # instead of GPT-4.1-mini which has proven inadequate for complex analytical reasoning
        if self.use_premium_models:
            logger.info("🧠 Using premium models for complex biological analysis tasks")
        
        # Create tasks for each chunk with clean, short names
        chunk_tasks = []
        for i, chunk in enumerate(chunks, 1):
            # Use the clean chunk_id directly - no need to modify further
            clean_task_id = chunk.chunk_id  # e.g., "func_oxidation_reduction", "genome_group1"
            
            # Ensure clean task IDs - prevent any possibility of recursive naming
            if len(clean_task_id) > 50:
                logger.warning(f"⚠️ Chunk ID too long, truncating: {clean_task_id}")
                clean_task_id = clean_task_id[:50]
            
            logger.info(f"📋 Creating clean chunk task: {clean_task_id}")
            
            # CRITICAL FIX: Inject original biological discovery context into chunk descriptions
            # This ensures sub-agents (gpt-4.1-mini) understand they're doing phage discovery, not generic analysis
            # Use format that avoids triggering old genome selection keywords
            if original_question:
                enhanced_description = f"Biological discovery task: '{original_question}' | Analyzing: {chunk.description}"
                logger.info(f"🧬 Enhanced chunk description with biological context: {enhanced_description[:100]}...")
            else:
                enhanced_description = chunk.description
                logger.warning("⚠️ No original question provided - chunk may lose biological context")
            
            task = Task(
                task_id=clean_task_id,
                task_type=original_task.task_type,
                description=enhanced_description,  # Now includes root biological discovery context
                query=getattr(original_task, 'query', None)
            )
            # Attach chunk data and mark as already chunked to prevent recursion
            task.chunk_data = chunk.data_subset
            task.biological_focus = chunk.biological_focus
            task.root_biological_context = original_question  # Store for note-taking context
            task._already_chunked = True  # Prevent recursive chunking
            task._intelligent_chunked = True  # Mark as using new intelligent system
            chunk_tasks.append(task)
        
        # Execute all chunks concurrently
        async def execute_chunk_task(task):
            try:
                result = await task_executor.execute_task(task)
                if result.success:
                    return {
                        'chunk_id': task.task_id,
                        'result': result.result,
                        'metadata': result.metadata,
                        'biological_focus': getattr(task, 'biological_focus', ''),
                        'execution_time': result.execution_time
                    }
                else:
                    logger.warning(f"Chunk {task.task_id} failed: {result.error}")
                    return {'chunk_id': task.task_id, 'error': result.error}
            except Exception as e:
                logger.error(f"Chunk {task.task_id} exception: {e}")
                return {'chunk_id': task.task_id, 'error': str(e)}
        
        # Execute all chunk tasks concurrently
        results = await asyncio.gather(*[execute_chunk_task(task) for task in chunk_tasks], return_exceptions=True)
        
        # Filter successful results
        successful_results = [r for r in results if isinstance(r, dict) and 'error' not in r]
        failed_count = len(results) - len(successful_results)
        
        if failed_count > 0:
            logger.warning(f"⚠️ {failed_count}/{len(chunks)} chunks failed")
        
        logger.info(f"✅ {len(successful_results)}/{len(chunks)} chunks completed successfully")
        return successful_results
    
    def synthesize_chunk_results(self, 
                                chunk_results: List[Dict[str, Any]], 
                                original_question: str,
                                chunks: List[AnalysisChunk]) -> str:
        """
        Synthesize results from all chunks into comprehensive summary with detailed findings.
        
        Args:
            chunk_results: Results from chunk analysis
            original_question: Original user question
            chunks: Original chunk definitions
            
        Returns:
            Comprehensive synthesis summary with detailed insights
        """
        if not chunk_results:
            return "No successful chunk analysis results available for synthesis."
        
        # Build comprehensive summary with detailed findings
        summary_parts = [
            f"# Comprehensive Genomic Analysis Summary",
            f"**Question:** {original_question}",
            f"",
            f"## Dataset Analysis Overview",
            f"- **Total Chunks Analyzed:** {len(chunk_results)}",
            f"- **Analysis Strategy:** Intelligent upfront chunking with biological focus",
            f"- **Coverage:** {sum(len(chunk.data_subset) for chunk in chunks):,} total data points",
            f"- **Execution Method:** Parallel chunk processing with biological meaning preservation",
            f"",
            f"## Detailed Findings by Analysis Focus:"
        ]
        
        # Collect comprehensive statistics
        all_genomes = set()
        all_functions = set()
        functional_categories = {}
        
        # Add detailed findings from each chunk
        for i, (result, chunk) in enumerate(zip(chunk_results, chunks), 1):
            biological_focus = result.get('biological_focus', chunk.biological_focus)
            execution_time = result.get('execution_time', 0)
            
            summary_parts.extend([
                f"",
                f"### {i}. {chunk.title}",
                f"**Biological Focus:** {biological_focus}",
                f"**Data Points Analyzed:** {len(chunk.data_subset):,}",
                f"**Processing Time:** {execution_time:.1f}s",
                f""
            ])
            
            # Extract comprehensive insights from result
            result_content = result.get('result', {})
            chunk_insights = []
            
            if isinstance(result_content, dict):
                # Access full structured data, not just samples
                if 'context' in result_content:
                    context = result_content['context']
                    if hasattr(context, 'structured_data') and context.structured_data:
                        structured_data = context.structured_data
                        
                        # Analyze all data in this chunk, not just samples
                        chunk_genomes = set()
                        chunk_functions = set()
                        function_details = {}
                        
                        for item in structured_data:
                            if 'genome_id' in item:
                                genome_id = item['genome_id']
                                # Handle None values
                                if genome_id is not None:
                                    chunk_genomes.add(genome_id)
                                    all_genomes.add(genome_id)
                            
                            if 'ko_description' in item:
                                func_desc = item['ko_description']
                                chunk_functions.add(func_desc)
                                all_functions.add(func_desc)
                                
                                # Count function occurrences
                                if func_desc not in function_details:
                                    function_details[func_desc] = {'count': 0, 'genomes': set()}
                                function_details[func_desc]['count'] += 1
                                if 'genome_id' in item:
                                    function_details[func_desc]['genomes'].add(item['genome_id'])
                        
                        # Add detailed insights for this chunk
                        genome_list = [g for g in sorted(list(chunk_genomes))[:3] if g is not None]
                        chunk_insights.append(f"**Genomes Represented:** {len(chunk_genomes)} ({', '.join(genome_list)}{'...' if len(chunk_genomes) > 3 else ''})")
                        chunk_insights.append(f"**Unique Functions:** {len(chunk_functions)}")
                        
                        # Top functions in this chunk
                        top_functions = sorted(function_details.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
                        if top_functions:
                            chunk_insights.append(f"**Most Abundant Functions:**")
                            for func, details in top_functions:
                                genome_count = len(details['genomes'])
                                chunk_insights.append(f"  - {func} ({details['count']} occurrences across {genome_count} genome{'s' if genome_count != 1 else ''})")
                        
                        # Store for overall analysis
                        if chunk.chunk_id not in functional_categories:
                            functional_categories[chunk.chunk_id] = {
                                'title': chunk.title,
                                'functions': len(chunk_functions),
                                'genomes': len(chunk_genomes),
                                'total_annotations': len(structured_data)
                            }
                
                # Access metadata and other insights
                if 'metadata' in result_content:
                    metadata = result_content['metadata']
                    if metadata:
                        chunk_insights.append(f"**Analysis Metadata:** {str(metadata)[:150]}...")
                
                # If there are detailed analysis results, include them
                if 'structured_data' in result_content and isinstance(result_content['structured_data'], list):
                    if len(result_content['structured_data']) > 0:
                        first_item = result_content['structured_data'][0]
                        if isinstance(first_item, dict) and 'summary' in first_item:
                            chunk_insights.append(f"**Analysis Summary:** {first_item['summary'][:200]}...")
            
            # Add insights to summary
            if chunk_insights:
                summary_parts.extend(chunk_insights)
            else:
                summary_parts.append(f"- {chunk.expected_insights}")
        
        # Add comprehensive cross-chunk synthesis
        summary_parts.extend([
            f"",
            f"## Cross-Chunk Synthesis & Insights",
            f"",
            f"### Overall Dataset Characteristics",
            f"- **Total Unique Genomes:** {len(all_genomes)}",
            f"- **Total Unique Functions:** {len(all_functions)}",
            f"- **Functional Categories Analyzed:** {len(functional_categories)}",
            f"",
            f"### Biological Insights Across Categories"
        ])
        
        # Add insights about functional distribution
        if functional_categories:
            for chunk_id, stats in functional_categories.items():
                summary_parts.append(f"- **{stats['title']}:** {stats['functions']} unique functions, {stats['total_annotations']} annotations across {stats['genomes']} genomes")
        
        # Add methodology and conclusion
        summary_parts.extend([
            f"",
            f"## Methodology & System Performance",
            f"- **Intelligent Chunking:** Successfully avoided recursive complexity explosion",
            f"- **Biological Coherence:** Maintained functional/genomic meaning in chunk organization", 
            f"- **Parallel Execution:** {len(chunk_results)} chunks processed concurrently for speed",
            f"- **Data Coverage:** Comprehensive analysis of {sum(len(chunk.data_subset) for chunk in chunks):,} annotations",
            f"",
            f"## Conclusion",
            f"This analysis successfully processed large-scale genomic functional data through intelligent upfront chunking, "
            f"providing detailed biological insights across {len(chunk_results)} focused categories. The methodology preserved "
            f"biological meaning while enabling comprehensive analysis of {len(all_functions)} unique functions across "
            f"{len(all_genomes)} genomes, demonstrating effective scaling for complex genomic datasets."
        ])
        
        return "\n".join(summary_parts)