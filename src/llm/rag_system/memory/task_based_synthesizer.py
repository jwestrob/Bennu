"""
Task-based synthesizer for handling unlimited dataset sizes.

Instead of artificially capping data chunks, creates analysis tasks that can
be properly managed by the existing task execution system.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .progressive_synthesizer import ProgressiveSynthesizer
from .note_keeper import NoteKeeper
from .note_schemas import TaskNote
from ..task_management import TaskGraph, Task, TaskType
from ..task_executor import TaskExecutor
from .parallel_task_executor import ParallelTaskExecutor, ParallelExecutionConfig

logger = logging.getLogger(__name__)

@dataclass 
class AnalysisChunk:
    """A chunk of data for analysis without artificial size limits."""
    chunk_id: str
    title: str
    data_subset: List[Dict[str, Any]]
    context: str
    analysis_focus: str

class TaskBasedSynthesizer(ProgressiveSynthesizer):
    """
    Synthesizer that uses task management instead of artificial data limits.
    
    Creates analysis tasks for data subsets and executes them through
    the task management system, allowing unlimited dataset processing.
    """
    
    def __init__(self, note_keeper: NoteKeeper, chunk_size: int = 8, 
                 max_items_per_task: int = 2000):  # Large but manageable chunks
        """
        Initialize task-based synthesizer.
        
        Args:
            note_keeper: NoteKeeper instance
            chunk_size: Number of tasks to process per synthesis chunk  
            max_items_per_task: Maximum data items per analysis task (much higher than current 120)
        """
        super().__init__(note_keeper, chunk_size)
        self.max_items_per_task = max_items_per_task  # Large chunks for detailed analysis
        
    def synthesize_unlimited_dataset(self, 
                                   task_notes: List[TaskNote],
                                   dspy_synthesizer,
                                   question: str,
                                   raw_data: List[Dict[str, Any]],
                                   rag_system) -> str:
        """
        Synthesize analysis for unlimited dataset size using task management.
        
        Args:
            task_notes: Existing task notes
            dspy_synthesizer: DSPy synthesizer module  
            question: Original user question
            raw_data: Complete dataset (no size limits)
            rag_system: RAG system instance for task execution
            
        Returns:
            Comprehensive synthesis of full dataset
        """
        logger.info(f"🔄 Starting task-based synthesis for {len(raw_data)} data points")
        
        # Log current parallel configuration
        from .parallel_config import get_parallel_config
        profile = get_parallel_config()
        logger.info(f"⚙️  Parallel config: {profile.name} ({profile.max_concurrent_tasks} concurrent, {profile.batch_size} batch size)")
        
        # Create analysis chunks without artificial size limits
        analysis_chunks = self._create_analysis_chunks(raw_data, question)
        
        # Only use standard synthesis for truly small datasets (< 500 items)
        # Always use multi-part analysis for larger datasets to provide detailed reports
        if len(raw_data) < 500 and len(analysis_chunks) == 1:
            logger.info("📄 Small dataset, using standard synthesis")
            return self._synthesize_standard(task_notes, dspy_synthesizer, question)
        
        logger.info(f"📋 Created {len(analysis_chunks)} analysis tasks")
        
        # Show parallel execution estimate
        from .parallel_config import estimate_parallel_speedup
        speedup_info = estimate_parallel_speedup(len(analysis_chunks))
        logger.info(f"📊 Estimated parallel execution: {speedup_info['parallel_time']:.1f}s ({speedup_info['speedup']:.1f}x speedup)")
        
        # Create task graph for parallel analysis  
        task_graph = self._create_analysis_task_graph(analysis_chunks, question)
        
        # Execute analysis tasks - with fallback to sequential if parallel fails
        task_executor = TaskExecutor(rag_system, self.note_keeper)
        
        # Async compatibility is now fixed, enable parallel execution for speed
        use_parallel = True  # Parallel execution now works with proper async handling
        
        if use_parallel:
            try:
                logger.info("🚀 Attempting parallel execution...")
                analysis_results = self._execute_analysis_tasks_parallel(task_graph, task_executor)
            except Exception as e:
                logger.error(f"❌ Parallel execution failed: {e}")
                logger.info("🔄 Falling back to sequential execution")
                analysis_results = self._execute_analysis_tasks_sequential(task_graph, task_executor)
        else:
            logger.info("🔄 Using sequential execution (parallel disabled)")
            analysis_results = self._execute_analysis_tasks_sequential(task_graph, task_executor)
        
        # Synthesize results from all analysis tasks
        return self._synthesize_task_results(analysis_results, question, dspy_synthesizer)
    
    def _create_analysis_chunks(self, data: List[Dict[str, Any]], 
                               question: str) -> List[AnalysisChunk]:
        """Create analysis chunks based on logical data groupings."""
        chunks = []
        
        # Group data by meaningful categories (not arbitrary size limits)
        if self._is_functional_comparison(question):
            chunks = self._chunk_by_functional_categories(data)
        elif self._is_genomic_comparison(question):
            chunks = self._chunk_by_genomes(data)
        elif self._is_pathway_analysis(question):
            chunks = self._chunk_by_pathways(data)
        else:
            # Default: chunk by logical size but much larger than current 120
            chunks = self._chunk_by_logical_size(data)
            
        return chunks
    
    def _chunk_by_functional_categories(self, data: List[Dict[str, Any]]) -> List[AnalysisChunk]:
        """Chunk by functional categories without size limits."""
        # Group by function/category
        category_groups = {}
        for item in data:
            category = item.get('ko_description', item.get('category', 'unknown'))
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(item)
        
        chunks = []
        current_chunk_data = []
        current_categories = []
        chunk_id = 1
        
        for category, category_data in category_groups.items():
            # Split if we exceed the task limit to create meaningful analysis chunks
            if (len(current_chunk_data) + len(category_data) > self.max_items_per_task and 
                current_chunk_data):
                chunks.append(AnalysisChunk(
                    chunk_id=f"func_chunk_{chunk_id}",
                    title=f"Functional Categories: {', '.join(current_categories[:3])}{'...' if len(current_categories) > 3 else ''}",
                    data_subset=current_chunk_data,
                    context=f"Analysis of {len(current_categories)} functional categories with {len(current_chunk_data)} proteins",
                    analysis_focus="functional_comparison"
                ))
                
                current_chunk_data = []
                current_categories = []
                chunk_id += 1
            
            current_chunk_data.extend(category_data)
            current_categories.append(category)
        
        # Add final chunk
        if current_chunk_data:
            chunks.append(AnalysisChunk(
                chunk_id=f"func_chunk_{chunk_id}",
                title=f"Functional Categories: {', '.join(current_categories[:3])}{'...' if len(current_categories) > 3 else ''}",
                data_subset=current_chunk_data,
                context=f"Analysis of {len(current_categories)} functional categories with {len(current_chunk_data)} proteins",
                analysis_focus="functional_comparison"
            ))
        
        return chunks
    
    def _chunk_by_genomes(self, data: List[Dict[str, Any]]) -> List[AnalysisChunk]:
        """Chunk by genomes, including all data per genome."""
        genome_groups = {}
        for item in data:
            genome_id = item.get('genome_id', item.get('genome', 'unknown'))
            if genome_id not in genome_groups:
                genome_groups[genome_id] = []
            genome_groups[genome_id].append(item)
        
        chunks = []
        for i, (genome_id, genome_data) in enumerate(genome_groups.items(), 1):
            chunks.append(AnalysisChunk(
                chunk_id=f"genome_chunk_{i}",
                title=f"Genome Analysis: {genome_id}",
                data_subset=genome_data,
                context=f"Complete functional analysis of genome {genome_id} with {len(genome_data)} annotations",
                analysis_focus="genome_analysis"
            ))
        
        return chunks
    
    def _chunk_by_logical_size(self, data: List[Dict[str, Any]]) -> List[AnalysisChunk]:
        """Create logical chunks with large but manageable size limits."""
        chunks = []
        
        # Always create meaningful analysis chunks (never put everything in one chunk)
        # This ensures we get detailed multi-part reports even for large datasets
        for i in range(0, len(data), int(self.max_items_per_task)):
            chunk_data = data[i:i + int(self.max_items_per_task)]
            chunk_id = i // int(self.max_items_per_task) + 1
            
            chunks.append(AnalysisChunk(
                chunk_id=f"data_chunk_{chunk_id}",
                title=f"Data Analysis Part {chunk_id}",
                data_subset=chunk_data,
                context=f"Analysis of {len(chunk_data)} data points (items {i+1}-{min(i+int(self.max_items_per_task), len(data))})",
                analysis_focus="comprehensive_analysis"
            ))
        
        return chunks
    
    def _create_analysis_task_graph(self, chunks: List[AnalysisChunk], 
                                   question: str) -> TaskGraph:
        """Create task graph for parallel analysis execution."""
        task_graph = TaskGraph()
        
        for chunk in chunks:
            # Create analysis task for each chunk
            analysis_query = self._generate_analysis_query(chunk, question)
            
            task = Task(
                task_id=chunk.chunk_id,
                task_type=TaskType.ATOMIC_QUERY,
                description=f"Analyze {chunk.title}",
                query=analysis_query
            )
            
            task_graph.add_task(task)
        
        return task_graph
    
    def _generate_analysis_query(self, chunk: AnalysisChunk, question: str) -> str:
        """Generate analysis query for a data chunk."""
        # Create a focused analysis query for this chunk
        base_context = f"Analyze the following {len(chunk.data_subset)} data points focusing on {chunk.analysis_focus}"
        
        if chunk.analysis_focus == "functional_comparison":
            query = f"{base_context}. Provide detailed functional comparison and biological insights for: {chunk.context}. Original question: {question}"
        elif chunk.analysis_focus == "genome_analysis":
            query = f"{base_context}. Provide comprehensive genome analysis including metabolic capabilities, specializations, and unique features. {chunk.context}. Original question: {question}"
        else:
            query = f"{base_context}. {chunk.context}. Original question: {question}"
        
        return query
    
    def _execute_analysis_tasks_parallel(self, task_graph: TaskGraph, 
                                        task_executor: TaskExecutor) -> List[Dict[str, Any]]:
        """Execute all analysis tasks in parallel for massive speed improvement."""
        # Configure parallel execution
        parallel_config = ParallelExecutionConfig(
            max_concurrent_tasks=10,  # Start with 10 concurrent API calls
            batch_size=5,  # Process 5 at a time
            timeout_per_task=120.0,  # 2 minutes per task
            retry_failed_tasks=True,
            max_retries=2
        )
        
        # Execute tasks in parallel
        parallel_executor = ParallelTaskExecutor(parallel_config)
        return parallel_executor.execute_analysis_tasks_parallel(task_graph, task_executor)
    
    def _execute_analysis_tasks_sequential(self, task_graph: TaskGraph, 
                                          task_executor: TaskExecutor) -> List[Dict[str, Any]]:
        """Execute all analysis tasks sequentially (handles async tasks properly)."""
        results = []
        
        # Execute tasks sequentially using await in async context
        async def execute_all_tasks():
            task_results = []
            for task_id in task_graph.get_executable_tasks():
                try:
                    task = task_graph.tasks[task_id]
                    logger.info(f"🔄 Executing analysis task: {task.description}")
                    
                    # Execute the async task properly with await
                    execution_result = await task_executor.execute_task(task)
                    
                    if execution_result.success:
                        task_results.append({
                            'task_id': task_id,
                            'result': execution_result.result,
                            'metadata': execution_result.metadata or {}
                        })
                        task_graph.mark_task_completed(task_id)
                    else:
                        logger.error(f"❌ Task {task_id} failed: {execution_result.error}")
                        task_graph.mark_task_failed(task_id, execution_result.error)
                        
                except Exception as e:
                    logger.error(f"❌ Error executing task {task_id}: {e}")
                    task_graph.mark_task_failed(task_id, str(e))
            
            return task_results
        
        # Check if we're in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an event loop, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, execute_all_tasks())
                results = future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            results = asyncio.run(execute_all_tasks())
        
        return results
    
    def _synthesize_task_results(self, analysis_results: List[Dict[str, Any]], 
                                question: str, dspy_synthesizer) -> str:
        """Synthesize final report from all task results."""
        logger.info(f"🔬 Synthesizing results from {len(analysis_results)} analysis tasks")
        
        # Combine all analysis results
        combined_analysis = []
        for result in analysis_results:
            combined_analysis.append(f"Task {result['task_id']}: {result['result']}")
        
        # Use DSPy to synthesize final comprehensive report
        try:
            if dspy_synthesizer:
                comprehensive_context = "\\n\\n".join(combined_analysis)
                final_result = dspy_synthesizer(
                    genomic_data=comprehensive_context,
                    target_length="comprehensive",
                    focus_areas="integration of all analyses, cross-cutting insights, comprehensive biological interpretation"
                )
                return final_result.summary
            else:
                # Fallback synthesis
                return self._create_fallback_comprehensive_synthesis(analysis_results, question)
                
        except Exception as e:
            logger.error(f"❌ Final synthesis failed: {e}")
            return self._create_fallback_comprehensive_synthesis(analysis_results, question)
    
    def _create_fallback_comprehensive_synthesis(self, analysis_results: List[Dict[str, Any]], 
                                               question: str) -> str:
        """Create fallback synthesis when DSPy fails."""
        sections = [
            f"# Comprehensive Analysis Report",
            f"*Question: {question}*",
            f"*Total analysis tasks completed: {len(analysis_results)}*",
            "",
            "## Task Results Summary:",
        ]
        
        for i, result in enumerate(analysis_results, 1):
            sections.append(f"### Task {i}: {result['task_id']}")
            sections.append(f"{result['result']}")
            sections.append("")
        
        sections.append("## Overall Conclusion:")
        sections.append(f"Analysis completed across {len(analysis_results)} comprehensive tasks covering the full dataset without artificial size limitations.")
        
        return "\\n".join(sections)
    
    def _is_functional_comparison(self, question: str) -> bool:
        """Check if this is a functional comparison question."""
        return any(term in question.lower() for term in ['functional', 'function', 'compare', 'comparison'])
    
    def _is_genomic_comparison(self, question: str) -> bool:
        """Check if this is a genomic comparison question.""" 
        return any(term in question.lower() for term in ['genome', 'genomic', 'across genomes'])
    
    def _is_pathway_analysis(self, question: str) -> bool:
        """Check if this is a pathway analysis question."""
        return any(term in question.lower() for term in ['pathway', 'metabolic', 'metabolism'])