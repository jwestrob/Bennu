"""
Task Plan Parser for converting DSPy text plans into executable Task objects.

This module bridges DSPy planning output with TaskGraph execution by parsing
natural language task descriptions into structured Task objects with proper
dependencies and execution parameters.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .task_management import Task, TaskType, TaskStatus

logger = logging.getLogger(__name__)

@dataclass
class ParsedPlan:
    """Container for parsed task plan with metadata."""
    tasks: List[Task]
    parsing_success: bool
    errors: List[str]
    original_plan: str

class TaskPlanParser:
    """
    Parser that converts DSPy natural language plans into executable Task objects.
    
    Handles:
    - Numbered step extraction from DSPy output
    - Task type identification (query vs tool call)
    - Dependency inference from step ordering
    - Natural language to executable action mapping
    """
    
    # Pattern recognition for different task types
    QUERY_PATTERNS = [
        r'\b(retrieve|query|find|search|get|match|count|show|list)\b',
        r'\b(database|genome|protein|annotation|domain)\b',
        r'\bCYPHER\b|MATCH\b|RETURN\b'
    ]
    
    TOOL_PATTERNS = {
        'code_interpreter': [
            r'\b(analyz|analysis|statistical|matrix|visualiz|plot|chart|graph)\b',
            r'\b(calculate|compute|process|transform|aggregate)\b',
            r'\b(python|pandas|numpy|matplotlib|seaborn)\b',
            r'\b(generate|create)\b.*\b(visual|plot|chart|graph)\b'
        ],
        'literature_search': [
            r'\b(literature|research|papers|publications|pubmed)\b',
            r'\b(recent|current|latest|study|studies)\b'
        ]
    }
    
    def __init__(self):
        """Initialize parser with compiled regex patterns."""
        self.query_regex = re.compile('|'.join(self.QUERY_PATTERNS), re.IGNORECASE)
        self.tool_regexes = {
            tool: re.compile('|'.join(patterns), re.IGNORECASE)
            for tool, patterns in self.TOOL_PATTERNS.items()
        }
    
    def parse_dspy_plan(self, plan_text: str) -> ParsedPlan:
        """
        Parse DSPy task plan into executable Task objects.
        
        Args:
            plan_text: Multi-line text with numbered steps from DSPy
            
        Returns:
            ParsedPlan with tasks, success status, and any errors
            
        Example:
            Input: "1. Retrieve list of all genomes.
                   2. Query DUF annotations for each genome.
                   3. Create stratified analysis."
            Output: ParsedPlan with 3 Task objects with proper dependencies
        """
        logger.info(f"Parsing DSPy plan: {len(plan_text)} characters")
        
        try:
            # Extract numbered steps
            steps = self._extract_numbered_steps(plan_text)
            if not steps:
                return ParsedPlan(
                    tasks=[],
                    parsing_success=False,
                    errors=["No numbered steps found in plan"],
                    original_plan=plan_text
                )
            
            # Convert steps to tasks
            tasks = []
            errors = []
            
            for i, (step_num, description) in enumerate(steps):
                try:
                    task = self._create_task_from_description(
                        step_num=step_num,
                        description=description,
                        previous_tasks=tasks
                    )
                    tasks.append(task)
                    logger.debug(f"Created task {step_num}: {task.task_type.value}")
                    
                except Exception as e:
                    error_msg = f"Failed to parse step {step_num}: {str(e)}"
                    errors.append(error_msg)
                    logger.warning(error_msg)
            
            success = len(tasks) > 0 and len(errors) == 0
            
            return ParsedPlan(
                tasks=tasks,
                parsing_success=success,
                errors=errors,
                original_plan=plan_text
            )
            
        except Exception as e:
            logger.error(f"Critical parsing error: {str(e)}")
            return ParsedPlan(
                tasks=[],
                parsing_success=False,
                errors=[f"Critical parsing error: {str(e)}"],
                original_plan=plan_text
            )
    
    def _extract_numbered_steps(self, plan_text: str) -> List[Tuple[int, str]]:
        """
        Extract numbered steps from DSPy plan text.
        
        Returns:
            List of (step_number, description) tuples
        """
        steps = []
        
        # Pattern for numbered steps: "1. Description" or "Step 1: Description"
        # Use MULTILINE flag to match line beginnings after whitespace
        step_pattern = re.compile(r'^\s*(?:Step\s+)?(\d+)[\.\:]?\s*(.+)', re.MULTILINE | re.IGNORECASE)
        
        for match in step_pattern.finditer(plan_text):
            step_num = int(match.group(1))
            description = match.group(2).strip()
            
            # Clean up description (remove trailing punctuation, extra whitespace)
            description = re.sub(r'\s+', ' ', description).strip()
            description = description.rstrip('.,;:')
            
            if description:  # Only add non-empty descriptions
                steps.append((step_num, description))
        
        # Sort by step number to ensure proper ordering
        steps.sort(key=lambda x: x[0])
        
        logger.debug(f"Extracted {len(steps)} numbered steps")
        return steps
    
    def _create_task_from_description(self, step_num: int, description: str, previous_tasks: List[Task]) -> Task:
        """
        Create a Task object from a natural language description.
        
        Args:
            step_num: Step number from DSPy plan
            description: Natural language task description
            previous_tasks: Previously created tasks for dependency inference
            
        Returns:
            Task object with appropriate type and parameters
        """
        # Generate unique task ID
        task_id = f"step_{step_num}_{description[:20].replace(' ', '_').lower()}"
        
        # Determine task type
        task_type, tool_name = self._classify_task_type(description)
        
        # Infer dependencies (typically depends on previous step)
        dependencies = self._infer_dependencies(step_num, description, previous_tasks)
        
        # Create task based on type
        if task_type == TaskType.ATOMIC_QUERY:
            return Task(
                task_id=task_id,
                task_type=task_type,
                description=description,
                dependencies=dependencies,
                query=self._generate_query_template(description),
                status=TaskStatus.PENDING
            )
        
        elif task_type == TaskType.TOOL_CALL:
            return Task(
                task_id=task_id,
                task_type=task_type,
                description=description,
                dependencies=dependencies,
                tool_name=tool_name,
                tool_args=self._extract_tool_args(description),
                status=TaskStatus.PENDING
            )
        
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def _classify_task_type(self, description: str) -> Tuple[TaskType, Optional[str]]:
        """
        Classify task type based on description content.
        
        Returns:
            (TaskType, tool_name) where tool_name is None for ATOMIC_QUERY
        """
        # Check for tool patterns first (more specific)
        for tool_name, regex in self.tool_regexes.items():
            if regex.search(description):
                logger.debug(f"Classified as TOOL_CALL: {tool_name}")
                return TaskType.TOOL_CALL, tool_name
        
        # Check for query patterns
        if self.query_regex.search(description):
            logger.debug("Classified as ATOMIC_QUERY")
            return TaskType.ATOMIC_QUERY, None
        
        # Default to query if unclear
        logger.debug("Defaulting to ATOMIC_QUERY (unclear classification)")
        return TaskType.ATOMIC_QUERY, None
    
    def _infer_dependencies(self, step_num: int, description: str, previous_tasks: List[Task]) -> List[str]:
        """
        Infer task dependencies based on step ordering and content.
        
        For now, we use simple sequential dependency (each step depends on previous).
        Future enhancement: parse "using results from step X" type references.
        """
        dependencies = []
        
        # Sequential dependency: depend on immediately previous task
        if previous_tasks and step_num > 1:
            dependencies.append(previous_tasks[-1].task_id)
        
        # TODO: Parse explicit references like "using data from step 2"
        # reference_pattern = re.compile(r'(?:using|from|with).*step\s+(\d+)', re.IGNORECASE)
        # for match in reference_pattern.finditer(description):
        #     ref_step = int(match.group(1))
        #     # Find task with that step number and add as dependency
        
        return dependencies
    
    def _generate_query_template(self, description: str) -> str:
        """
        Generate a query template based on task description.
        
        This is a simplified approach - the actual query will be generated
        by the DSPy ContextRetriever during execution.
        """
        # Transform "for each" patterns into comparative language
        transformed_description = self._transform_for_each_patterns(description)
        
        # For now, return the description as a query template
        # The actual Cypher generation will happen during execution
        return f"# Query for: {transformed_description}"
    
    def _transform_for_each_patterns(self, description: str) -> str:
        """
        Transform 'for each' patterns into comparative query language.
        
        This helps DSPy understand that these should be comparative queries
        showing all results, not single-item queries.
        
        Args:
            description: Original task description
            
        Returns:
            Transformed description that's clearer for DSPy
        """
        import re
        
        # Transform "for each genome" patterns
        patterns = [
            (r'\bfor\s+each\s+genome,?\s+', 'compare across all genomes to '),
            (r'\bfor\s+each\s+protein,?\s+', 'compare across all proteins to '),
            (r'\bfor\s+each\s+gene,?\s+', 'compare across all genes to '),
            (r'\bfor\s+each\s+domain,?\s+', 'compare across all domains to '),
        ]
        
        transformed = description
        for pattern, replacement in patterns:
            transformed = re.sub(pattern, replacement, transformed, flags=re.IGNORECASE)
        
        # If we transformed anything, log it
        if transformed != description:
            logger.debug(f"Transformed task description: '{description}' -> '{transformed}'")
        
        return transformed
    
    def _extract_tool_args(self, description: str) -> Dict[str, Any]:
        """
        Extract tool arguments from task description.
        
        Returns basic arguments that can be refined during execution.
        """
        args = {
            "description": description,
            "task_context": "genomic_analysis"
        }
        
        # Extract specific keywords that might be useful
        if "matrix" in description.lower():
            args["output_type"] = "matrix"
        if "visualization" in description.lower() or "plot" in description.lower():
            args["output_type"] = "visualization"
        if "statistical" in description.lower() or "analysis" in description.lower():
            args["analysis_type"] = "statistical"
        
        return args

def test_parser():
    """Test function for the TaskPlanParser."""
    parser = TaskPlanParser()
    
    # Test case 1: Simple genomic analysis plan
    test_plan_1 = """
    1. Retrieve list of all genomes in the dataset.
    2. Query DUF annotations for each genome.  
    3. Create stratified analysis matrix.
    4. Generate comparative visualizations.
    """
    
    result = parser.parse_dspy_plan(test_plan_1)
    
    print("=== Task Plan Parser Test ===")
    print(f"Original plan:\n{test_plan_1}")
    print(f"\nParsing success: {result.parsing_success}")
    print(f"Errors: {result.errors}")
    print(f"Tasks created: {len(result.tasks)}")
    
    for i, task in enumerate(result.tasks):
        print(f"\nTask {i+1}:")
        print(f"  ID: {task.task_id}")
        print(f"  Type: {task.task_type.value}")
        print(f"  Description: {task.description}")
        print(f"  Dependencies: {task.dependencies}")
        if task.tool_name:
            print(f"  Tool: {task.tool_name}")
            print(f"  Args: {task.tool_args}")
    
    return result

if __name__ == "__main__":
    test_parser()