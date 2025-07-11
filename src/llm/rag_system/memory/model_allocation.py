"""
Model allocation system for cost-optimized multi-part report generation.

Provides intelligent model selection based on task complexity while maintaining
easy switching between cost-optimized and premium model configurations.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class TaskComplexity(str, Enum):
    """Task complexity levels for model selection."""
    SIMPLE = "simple"          # Basic classification, formatting
    MEDIUM = "medium"          # Structured analysis, summarization
    COMPLEX = "complex"        # Deep reasoning, synthesis, biological interpretation


class ModelTier(str, Enum):
    """Model tier classifications."""
    NANO = "nano"              # Cheapest, fastest
    MINI = "mini"              # Balanced cost/performance
    STANDARD = "standard"      # Full capability
    PREMIUM = "premium"        # Best reasoning, most expensive


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    model_name: str
    provider: str              # 'openai', 'anthropic'
    cost_per_million: float    # Input token cost
    max_context: int           # Maximum context window
    specialties: list          # Areas of strength


class ModelAllocation:
    """
    Manages model selection based on task complexity and cost optimization.
    
    Provides easy switching between cost-optimized and premium configurations.
    """
    
    def __init__(self, use_premium_everywhere: bool = True):
        """
        Initialize model allocation system.
        
        Args:
            use_premium_everywhere: If True, uses premium models for all tasks
        """
        self.use_premium_everywhere = use_premium_everywhere
        self.fallback_enabled = True
        
        # Define available models
        self.models = {
            ModelTier.NANO: ModelConfig(
                model_name="gpt-4.1-nano",
                provider="openai",
                cost_per_million=0.50,
                max_context=1000000,
                specialties=["classification", "simple_formatting"]
            ),
            ModelTier.MINI: ModelConfig(
                model_name="gpt-4.1-mini",
                provider="openai",
                cost_per_million=0.15,
                max_context=1000000,
                specialties=["analysis", "summarization", "structured_output"]
            ),
            ModelTier.STANDARD: ModelConfig(
                model_name="gpt-4.1",
                provider="openai",
                cost_per_million=3.00,
                max_context=1000000,
                specialties=["reasoning", "coding", "complex_analysis"]
            ),
            ModelTier.PREMIUM: ModelConfig(
                model_name="o3",
                provider="openai",
                cost_per_million=15.00,
                max_context=200000,
                specialties=["deep_reasoning", "scientific_analysis", "synthesis"]
            )
        }
        
        # Define task complexity mapping
        self.task_complexity = {
            # Simple tasks - basic classification and formatting
            "report_type_classification": TaskComplexity.SIMPLE,
            "chunking_strategy_selection": TaskComplexity.SIMPLE,
            "data_formatting": TaskComplexity.SIMPLE,
            "progress_tracking": TaskComplexity.SIMPLE,
            
            # Medium tasks - structured analysis and summarization
            "executive_summary": TaskComplexity.MEDIUM,
            "report_part_generation": TaskComplexity.MEDIUM,
            "data_overview": TaskComplexity.MEDIUM,
            "pattern_identification": TaskComplexity.MEDIUM,
            
            # Complex tasks - deep reasoning and synthesis  
            "query_classification": TaskComplexity.COMPLEX,    # Biological reasoning needed
            "context_preparation": TaskComplexity.COMPLEX,     # Query generation needs domain knowledge
            "final_synthesis": TaskComplexity.COMPLEX,
            "biological_interpretation": TaskComplexity.COMPLEX,
            "cross_task_integration": TaskComplexity.COMPLEX,
            "confidence_assessment": TaskComplexity.COMPLEX,
            "scientific_validation": TaskComplexity.COMPLEX,
            "emergent_insights": TaskComplexity.COMPLEX,
            "agentic_planning": TaskComplexity.COMPLEX
        }
        
        # Define model allocation based on complexity
        self.complexity_to_tier = {
            TaskComplexity.SIMPLE: ModelTier.MINI,      # Use mini for simple tasks
            TaskComplexity.MEDIUM: ModelTier.MINI,      # Use mini for medium tasks  
            TaskComplexity.COMPLEX: ModelTier.PREMIUM   # Use o3 for complex synthesis tasks
        }
        
        logger.info(f"🎯 Model allocation initialized (premium_everywhere={use_premium_everywhere})")
    
    def get_model_for_task(self, task_name: str) -> Tuple[str, ModelConfig]:
        """
        Get the appropriate model for a given task.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Tuple of (model_name, model_config)
        """
        if self.use_premium_everywhere:
            # Override: use premium model for everything
            model_config = self.models[ModelTier.PREMIUM]
            logger.info(f"🔥 PREMIUM MODE: Using {model_config.model_name} for {task_name}")
            return model_config.model_name, model_config
        
        # Get task complexity
        complexity = self.task_complexity.get(task_name, TaskComplexity.MEDIUM)
        
        # Get appropriate model tier
        tier = self.complexity_to_tier[complexity]
        model_config = self.models[tier]
        
        # Enhanced logging for synthesis tasks
        if complexity == TaskComplexity.COMPLEX:
            logger.info(f"🧠 COMPLEX TASK: Using {model_config.model_name} for {task_name} (complexity: {complexity.value})")
        else:
            logger.debug(f"🎯 Selected {tier.value} model for {task_name} ({complexity.value}): {model_config.model_name}")
        
        return model_config.model_name, model_config
    
    def get_fallback_model(self, original_task: str) -> Tuple[str, ModelConfig]:
        """
        Get fallback model when primary model fails.
        
        Args:
            original_task: Name of the original task
            
        Returns:
            Tuple of (model_name, model_config)
        """
        if not self.fallback_enabled:
            return self.get_model_for_task(original_task)
        
        # Always fallback to premium model
        model_config = self.models[ModelTier.PREMIUM]
        logger.warning(f"🔄 Falling back to premium model for {original_task}: {model_config.model_name}")
        return model_config.model_name, model_config
    
    def switch_to_premium_mode(self):
        """Switch to premium models for all tasks."""
        self.use_premium_everywhere = True
        logger.info("🔥 Switched to premium mode - using o3 for all tasks")
    
    def switch_to_optimized_mode(self):
        """Switch to cost-optimized model selection."""
        self.use_premium_everywhere = False
        logger.info("🎯 Switched to optimized mode - using task-appropriate models")
    
    def get_cost_estimate(self, task_name: str, estimated_tokens: int) -> float:
        """
        Estimate cost for a task.
        
        Args:
            task_name: Name of the task
            estimated_tokens: Estimated token count
            
        Returns:
            Estimated cost in dollars
        """
        _, model_config = self.get_model_for_task(task_name)
        cost = (estimated_tokens / 1_000_000) * model_config.cost_per_million
        return cost
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get summary of current model allocation."""
        if self.use_premium_everywhere:
            return {
                "mode": "premium_everywhere",
                "primary_model": self.models[ModelTier.PREMIUM].model_name,
                "estimated_cost_multiplier": 100.0  # 100x more expensive than optimized
            }
        
        allocation = {}
        total_cost_weight = 0
        task_count = 0
        
        for task_name, complexity in self.task_complexity.items():
            tier = self.complexity_to_tier[complexity]
            model_config = self.models[tier]
            allocation[task_name] = {
                "model": model_config.model_name,
                "tier": tier.value,
                "complexity": complexity.value,
                "cost_per_million": model_config.cost_per_million
            }
            total_cost_weight += model_config.cost_per_million
            task_count += 1
        
        return {
            "mode": "optimized",
            "task_allocation": allocation,
            "average_cost_per_million": total_cost_weight / task_count,
            "cost_savings_vs_premium": f"{(1 - (total_cost_weight / task_count) / 15.0) * 100:.1f}%"
        }
    
    def create_context_managed_call(self, task_name: str, signature_class, module_call_func):
        """
        Execute a DSPy module call with appropriate model context.
        
        Args:
            task_name: Name of the task
            signature_class: DSPy signature class
            module_call_func: Function that takes a module and returns result
            
        Returns:
            Result from module call with appropriate model
        """
        try:
            import dspy
        except ImportError:
            logger.error("DSPy not available for model allocation")
            return None
        
        model_name, model_config = self.get_model_for_task(task_name)
        
        try:
            # Use DSPy 2.6+ LM format with provider/model
            if model_config.provider == "openai":
                model_string = f"openai/{model_name}"
                # Special handling for reasoning models
                if model_name.startswith(('o1', 'o3')):
                    lm = dspy.LM(model=model_string, temperature=1.0, max_tokens=20000)
                else:
                    lm = dspy.LM(model=model_string, temperature=0.0, max_tokens=8000)
            elif model_config.provider == "anthropic":
                model_string = f"anthropic/{model_name}"
                lm = dspy.LM(model=model_string, max_tokens=8000)
            else:
                logger.warning(f"Unknown provider {model_config.provider}, using default")
                model_string = f"openai/{model_name}"
                lm = dspy.LM(model=model_string, max_tokens=8000)
            
            # Create module and execute with context manager
            module = dspy.Predict(signature_class)
            
            logger.debug(f"🔥 Using {model_name} for {task_name} via context manager")
            with dspy.context(lm=lm):
                result = module_call_func(module)
            
            logger.debug(f"Successfully executed {task_name} with {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute {task_name} with model allocation: {e}")
            if self.fallback_enabled:
                # Try fallback with default module
                try:
                    logger.warning(f"Falling back to default model for {task_name}")
                    fallback_module = dspy.Predict(signature_class)
                    result = module_call_func(fallback_module)
                    logger.info(f"Successfully executed {task_name} with fallback")
                    return result
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed for {task_name}: {fallback_error}")
            
            return None


# Global model allocation instance  
model_allocator = ModelAllocation(use_premium_everywhere=False)


def get_model_allocator() -> ModelAllocation:
    """Get the global model allocator instance."""
    return model_allocator


def switch_to_premium_everywhere():
    """Convenience function to switch to premium models globally."""
    global model_allocator
    model_allocator.switch_to_premium_mode()


def switch_to_optimized_mode():
    """Convenience function to switch to optimized model selection globally."""
    global model_allocator
    model_allocator.switch_to_optimized_mode()