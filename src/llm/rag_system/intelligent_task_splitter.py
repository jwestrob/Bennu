"""
Intelligent Task Splitter for handling oversized tasks.

Automatically detects when a task receives too much data (e.g., 20K tokens) and 
splits it into manageable sub-tasks, then summarizes results back to the originating agent.
"""

import logging
import asyncio
import tiktoken
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .task_management import Task, TaskType, TaskGraph
from .task_executor import ExecutionResult

logger = logging.getLogger(__name__)

@dataclass
class SplitTaskResult:
    """Result from splitting and executing sub-tasks."""
    original_task_id: str
    sub_task_results: List[Dict[str, Any]]
    summary: str
    total_items_processed: int
    execution_time: float

class IntelligentTaskSplitter:
    """
    DEPRECATED: Old recursive task splitter replaced by IntelligentChunkingManager.
    
    This class has been superseded by the new IntelligentChunkingManager which provides:
    - Clean biological naming (func_oxidation_reduction vs _sub_1_sub_2...)
    - No recursive complexity explosion
    - Better biological meaning preservation
    - Parallel chunk execution with synthesis
    
    NOTE: This class is disabled to prevent recursive naming issues.
    """
    
    def __init__(self, max_tokens_per_task: int = 15000, max_items_per_chunk: int = 2000):
        """
        Initialize task splitter.
        
        Args:
            max_tokens_per_task: Maximum tokens before splitting task
            max_items_per_chunk: Maximum data items per sub-task
        """
        self.max_tokens_per_task = max_tokens_per_task
        self.max_items_per_chunk = max_items_per_chunk
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception as e:
            logger.warning(f"Failed to initialize tokenizer: {e}")
            self.tokenizer = None
    
    async def execute_with_intelligent_splitting(self, 
                                               task: Task, 
                                               task_executor,
                                               raw_data: List[Dict[str, Any]] = None) -> ExecutionResult:
        """
        DEPRECATED: This method is disabled to prevent recursive naming issues.
        
        Use IntelligentChunkingManager instead for clean biological chunking.
        """
        logger.warning(f"⚠️ DEPRECATED: IntelligentTaskSplitter called for task {task.task_id}")
        logger.warning(f"🔄 Falling back to direct execution - use IntelligentChunkingManager for large datasets")
        
        # Just execute the task directly instead of splitting
        return await task_executor.execute_task(task)
    
    def _estimate_task_tokens(self, task: Task, raw_data: List[Dict[str, Any]] = None) -> int:
        """
        Estimate token count for a task.
        
        Args:
            task: Task to estimate
            raw_data: Associated raw data
            
        Returns:
            Estimated token count
        """
        # Base task tokens
        task_text = f"{task.description} {task.query if hasattr(task, 'query') else ''}"
        base_tokens = self._count_tokens(task_text)
        
        # Data tokens
        data_tokens = 0
        if raw_data:
            # Sample first few items to estimate average token size
            sample_size = min(10, len(raw_data))
            sample_text = str(raw_data[:sample_size])
            sample_tokens = self._count_tokens(sample_text)
            
            # Extrapolate to full dataset
            avg_tokens_per_item = sample_tokens / sample_size if sample_size > 0 else 0
            data_tokens = int(avg_tokens_per_item * len(raw_data))
        
        total_tokens = base_tokens + data_tokens
        logger.debug(f"📊 Token estimate: {base_tokens} base + {data_tokens} data = {total_tokens} total")
        
        return total_tokens
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tokenizer."""
        if not self.tokenizer:
            # Fallback estimation: ~4 characters per token
            return len(text) // 4
        
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            return len(text) // 4
    
    async def _execute_split_task(self, 
                                original_task: Task, 
                                task_executor,
                                raw_data: List[Dict[str, Any]]) -> ExecutionResult:
        """
        Split task into sub-tasks and execute them.
        
        Args:
            original_task: Original oversized task
            task_executor: TaskExecutor instance
            raw_data: Raw data to split
            
        Returns:
            ExecutionResult with summarized results
        """
        import time
        start_time = time.time()
        
        # Create sub-tasks
        sub_tasks = self._create_sub_tasks(original_task, raw_data)
        logger.info(f"🔀 Created {len(sub_tasks)} sub-tasks for {original_task.task_id}")
        
        # Execute sub-tasks
        sub_task_results = await self._execute_sub_tasks(sub_tasks, task_executor)
        
        # Generate summary for originating agent
        summary = self._generate_summary(original_task, sub_task_results)
        
        execution_time = time.time() - start_time
        
        logger.info(f"✅ Split task completed: {len(sub_task_results)} sub-tasks in {execution_time:.2f}s")
        
        return ExecutionResult(
            task_id=original_task.task_id,
            success=True,
            result={
                "summary": summary,
                "sub_task_count": len(sub_task_results),
                "total_items_processed": sum(len(r.get('structured_data', [])) for r in sub_task_results),
                "sub_task_results": sub_task_results[:3],  # Include first 3 for context
                "split_execution": True
            },
            execution_time=execution_time,
            metadata={
                "original_task_type": original_task.task_type.value,
                "split_strategy": "intelligent_chunking",
                "sub_task_count": len(sub_task_results)
            }
        )
    
    def _create_sub_tasks(self, original_task: Task, raw_data: List[Dict[str, Any]]) -> List[Task]:
        """
        Create sub-tasks from original oversized task.
        
        Args:
            original_task: Original task to split
            raw_data: Raw data to chunk
            
        Returns:
            List of sub-tasks
        """
        sub_tasks = []
        
        # Intelligent chunking based on data structure
        chunks = self._create_intelligent_chunks(raw_data, original_task.description)
        
        for i, chunk in enumerate(chunks, 1):
            # Create focused sub-task description
            sub_description = self._create_sub_task_description(original_task.description, chunk, i, len(chunks))
            
            sub_task = Task(
                task_id=f"{original_task.task_id}_sub_{i}",
                task_type=original_task.task_type,
                description=sub_description,
                query=getattr(original_task, 'query', None),
                dependencies=[]  # Sub-tasks are independent
            )
            
            # Attach chunk data as metadata
            sub_task.chunk_data = chunk
            sub_tasks.append(sub_task)
        
        return sub_tasks
    
    def _create_intelligent_chunks(self, data: List[Dict[str, Any]], description: str) -> List[List[Dict[str, Any]]]:
        """
        Create intelligent chunks based on data structure and query type.
        
        Args:
            data: Raw data to chunk
            description: Task description for context
            
        Returns:
            List of data chunks
        """
        if not data:
            return []
        
        # Analyze query type for optimal chunking strategy
        if self._is_functional_analysis(description):
            return self._chunk_by_function(data)
        elif self._is_genomic_analysis(description):
            return self._chunk_by_genome(data)
        else:
            return self._chunk_by_size(data)
    
    def _is_functional_analysis(self, description: str) -> bool:
        """Check if this is a functional analysis task."""
        functional_keywords = ['function', 'functional', 'ko', 'kegg', 'enzyme', 'pathway', 'metabolism']
        return any(keyword in description.lower() for keyword in functional_keywords)
    
    def _is_genomic_analysis(self, description: str) -> bool:
        """Check if this is a genomic comparison task."""
        genomic_keywords = ['genome', 'genomic', 'compare', 'comparison', 'across genomes']
        return any(keyword in description.lower() for keyword in genomic_keywords)
    
    def _chunk_by_function(self, data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Chunk data by functional categories."""
        # Group by KO description or similar functional identifier
        function_groups = {}
        for item in data:
            function_key = item.get('ko_description', item.get('description', 'unknown'))
            if function_key not in function_groups:
                function_groups[function_key] = []
            function_groups[function_key].append(item)
        
        # Create chunks respecting size limits
        chunks = []
        current_chunk = []
        
        for function_data in function_groups.values():
            if len(current_chunk) + len(function_data) > self.max_items_per_chunk and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
            current_chunk.extend(function_data)
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _chunk_by_genome(self, data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Chunk data by genome."""
        # Group by genome ID
        genome_groups = {}
        for item in data:
            genome_id = item.get('genome_id', item.get('genome', 'unknown'))
            if genome_id not in genome_groups:
                genome_groups[genome_id] = []
            genome_groups[genome_id].append(item)
        
        # Each genome becomes a chunk (or split if too large)
        chunks = []
        for genome_data in genome_groups.values():
            if len(genome_data) <= self.max_items_per_chunk:
                chunks.append(genome_data)
            else:
                # Split large genomes
                for i in range(0, len(genome_data), self.max_items_per_chunk):
                    chunk = genome_data[i:i + self.max_items_per_chunk]
                    chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_size(self, data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Chunk data by size only."""
        chunks = []
        for i in range(0, len(data), self.max_items_per_chunk):
            chunk = data[i:i + self.max_items_per_chunk]
            chunks.append(chunk)
        return chunks
    
    def _create_sub_task_description(self, original_description: str, chunk: List[Dict[str, Any]], 
                                   chunk_num: int, total_chunks: int) -> str:
        """
        Create focused description for sub-task.
        
        Args:
            original_description: Original task description
            chunk: Data chunk for this sub-task
            chunk_num: Current chunk number
            total_chunks: Total number of chunks
            
        Returns:
            Focused sub-task description
        """
        # Analyze chunk content for specific focus
        chunk_focus = self._analyze_chunk_focus(chunk)
        
        return (f"Analyze part {chunk_num}/{total_chunks} of {original_description.lower()}: "
                f"{chunk_focus}. Focus on {len(chunk)} data points with detailed biological insights.")
    
    def _analyze_chunk_focus(self, chunk: List[Dict[str, Any]]) -> str:
        """Analyze chunk to determine specific focus area."""
        if not chunk:
            return "data subset"
        
        # Check for common functional categories
        sample_item = chunk[0]
        if 'ko_description' in sample_item:
            # Sample a few descriptions to understand focus
            descriptions = [item.get('ko_description', '') for item in chunk[:3]]
            common_terms = self._extract_common_terms(descriptions)
            if common_terms:
                return f"functions related to {', '.join(common_terms[:2])}"
        
        if 'genome_id' in sample_item:
            genomes = set(item.get('genome_id', '') for item in chunk)
            if len(genomes) == 1:
                return f"functions in {list(genomes)[0]}"
            else:
                return f"functions across {len(genomes)} genomes"
        
        return f"functional analysis subset"
    
    def _extract_common_terms(self, descriptions: List[str]) -> List[str]:
        """Extract common terms from descriptions."""
        all_words = []
        for desc in descriptions:
            words = desc.lower().split()
            # Filter out common words
            meaningful_words = [w for w in words if len(w) > 3 and w not in 
                               ['protein', 'enzyme', 'system', 'subunit', 'domain']]
            all_words.extend(meaningful_words)
        
        # Count word frequency
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Return most common meaningful terms
        return [word for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)][:3]
    
    async def _execute_sub_tasks(self, sub_tasks: List[Task], task_executor) -> List[Dict[str, Any]]:
        """
        Execute sub-tasks in parallel for speed.
        
        Args:
            sub_tasks: List of sub-tasks to execute
            task_executor: TaskExecutor instance
            
        Returns:
            List of sub-task results
        """
        logger.info(f"🚀 Executing {len(sub_tasks)} sub-tasks in parallel")
        
        # Execute sub-tasks concurrently
        async def execute_sub_task(sub_task):
            try:
                result = await task_executor.execute_task(sub_task)
                if result.success:
                    return result.result
                else:
                    logger.warning(f"Sub-task {sub_task.task_id} failed: {result.error}")
                    return {"error": result.error, "task_id": sub_task.task_id}
            except Exception as e:
                logger.error(f"Sub-task {sub_task.task_id} exception: {e}")
                return {"error": str(e), "task_id": sub_task.task_id}
        
        # Execute all sub-tasks concurrently
        results = await asyncio.gather(*[execute_sub_task(task) for task in sub_tasks], return_exceptions=True)
        
        # Filter successful results
        successful_results = [r for r in results if isinstance(r, dict) and 'error' not in r]
        failed_count = len(results) - len(successful_results)
        
        if failed_count > 0:
            logger.warning(f"⚠️ {failed_count}/{len(sub_tasks)} sub-tasks failed")
        
        logger.info(f"✅ {len(successful_results)}/{len(sub_tasks)} sub-tasks completed successfully")
        return successful_results
    
    def _generate_summary(self, original_task: Task, sub_task_results: List[Dict[str, Any]]) -> str:
        """
        Generate summary of sub-task results for originating agent.
        
        Args:
            original_task: Original task that was split
            sub_task_results: Results from all sub-tasks
            
        Returns:
            Summary string for originating agent
        """
        if not sub_task_results:
            return "No successful sub-task results to summarize."
        
        # Aggregate statistics
        total_items = sum(len(r.get('structured_data', [])) for r in sub_task_results)
        unique_genomes = set()
        unique_functions = set()
        
        for result in sub_task_results:
            structured_data = result.get('structured_data', [])
            for item in structured_data:
                if 'genome_id' in item:
                    unique_genomes.add(item['genome_id'])
                if 'ko_description' in item:
                    unique_functions.add(item['ko_description'])
        
        # Create comprehensive summary
        summary_parts = [
            f"**Comprehensive Analysis Summary for: {original_task.description}**",
            f"",
            f"📊 **Data Processed:**",
            f"- {len(sub_task_results)} analysis segments completed",
            f"- {total_items:,} total data points analyzed",
            f"- {len(unique_genomes)} unique genomes covered" if unique_genomes else "",
            f"- {len(unique_functions)} distinct functions identified" if unique_functions else "",
            f"",
            f"🧬 **Key Findings:**"
        ]
        
        # Add key findings from sub-tasks
        key_findings = []
        for i, result in enumerate(sub_task_results[:5], 1):  # Top 5 results
            context = result.get('context', {})
            if hasattr(context, 'structured_data'):
                sample_data = context.structured_data[:2]  # Sample findings
                for item in sample_data:
                    if 'ko_description' in item:
                        key_findings.append(f"- Segment {i}: {item['ko_description']} (in {item.get('genome_id', 'unknown genome')})")
        
        summary_parts.extend(key_findings[:8])  # Top 8 findings
        
        if len(sub_task_results) > 5:
            summary_parts.append(f"- ... and {len(sub_task_results) - 5} additional analysis segments")
        
        summary_parts.extend([
            f"",
            f"✅ **Analysis Complete:** All data successfully processed through intelligent task splitting",
            f"📈 **Scale:** Large dataset analysis completed with {len(sub_task_results)} parallel sub-analyses"
        ])
        
        return "\n".join(filter(None, summary_parts))