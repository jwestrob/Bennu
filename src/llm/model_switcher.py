"""
Simple model switching utilities for easy cost control.

Provides functions to switch between cost-effective and premium models
globally across the entire genomic RAG system.
"""

import logging
from typing import Dict, Any

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

from .config import LLMConfig

logger = logging.getLogger(__name__)

# Global config instance
_config = None

def get_config() -> LLMConfig:
    """Get or create the global config instance."""
    global _config
    if _config is None:
        _config = LLMConfig.from_env()
    return _config

def reconfigure_dspy(config: LLMConfig) -> None:
    """Reconfigure DSPy with the current config settings."""
    if not DSPY_AVAILABLE:
        logger.warning("DSPy not available - model switching disabled")
        return
    
    try:
        # Get API key
        api_key = config.get_api_key()
        
        if config.llm_provider == "openai" and api_key:
            import os
            os.environ['OPENAI_API_KEY'] = api_key
            
            # Use config-based model selection
            current_model = config.get_current_model()
            model_string = f"openai/{current_model}"
            
            # Special handling for OpenAI reasoning models (o1, o3)
            if current_model.startswith(('o1', 'o3')):
                lm = dspy.LM(model=model_string, temperature=1.0, max_tokens=20000)
                logger.info(f"🎯 DSPy reconfigured with reasoning model: {model_string}")
            else:
                lm = dspy.LM(model=model_string, temperature=0.0, max_tokens=2000)
                logger.info(f"🎯 DSPy reconfigured with standard model: {model_string}")
            
            dspy.settings.configure(lm=lm)
            
        elif config.llm_provider == "anthropic" and api_key:
            import os
            os.environ['ANTHROPIC_API_KEY'] = api_key
            
            current_model = config.get_current_model()
            # Map to Anthropic models if needed
            if current_model.startswith(('gpt', 'o1', 'o3')):
                anthropic_model = "claude-3-haiku-20240307" if config.model_mode == "cost_effective" else "claude-3-opus-20240229"
            else:
                anthropic_model = current_model
            
            model_string = f"anthropic/{anthropic_model}"
            lm = dspy.LM(model=model_string, max_tokens=1000)
            dspy.settings.configure(lm=lm)
            logger.info(f"🎯 DSPy reconfigured with Anthropic model: {model_string}")
            
    except Exception as e:
        logger.error(f"Failed to reconfigure DSPy: {e}")

def switch_to_cost_effective() -> None:
    """
    Switch to cost-effective model globally.
    
    Uses gpt-4.1-mini (or configured cost-effective model) for all tasks.
    Great for development, testing, and bulk processing.
    """
    config = get_config()
    config.set_cost_effective_mode()
    reconfigure_dspy(config)
    
    model_info = config.get_model_info()
    logger.info(f"💡 Switched to cost-effective mode: {model_info['current_model']}")
    
    print(f"💡 Cost-Effective Mode Active")
    print(f"   Current model: {model_info['current_model']}")
    print(f"   Mode: {model_info['mode']}")
    print(f"   Best for: Development, testing, bulk processing")

def switch_to_premium() -> None:
    """
    Switch to premium model globally.
    
    Uses o3 (or configured premium model) for all tasks.
    Best for final results, complex analysis, and publication-quality output.
    """
    config = get_config()
    config.set_premium_mode()
    reconfigure_dspy(config)
    
    model_info = config.get_model_info()
    logger.info(f"🔥 Switched to premium mode: {model_info['current_model']}")
    
    print(f"🔥 Premium Mode Active")
    print(f"   Current model: {model_info['current_model']}")
    print(f"   Mode: {model_info['mode']}")
    print(f"   Best for: Final results, complex analysis, publication quality")

def get_current_model_status() -> Dict[str, Any]:
    """
    Get current model configuration status.
    
    Returns:
        Dictionary with current model information
    """
    config = get_config()
    return config.get_model_info()

def print_model_status() -> None:
    """Print current model configuration status."""
    config = get_config()
    model_info = config.get_model_info()
    
    print(f"\n🎯 Current Model Configuration:")
    print(f"   Mode: {model_info['mode']}")
    print(f"   Current model: {model_info['current_model']}")
    print(f"   Cost-effective option: {model_info['cost_effective_model']}")
    print(f"   Premium option: {model_info['premium_model']}")
    print(f"\nTo switch:")
    print(f"   switch_to_cost_effective() - Use {model_info['cost_effective_model']}")
    print(f"   switch_to_premium() - Use {model_info['premium_model']}")

def configure_models(cost_effective_model: str = None, premium_model: str = None) -> None:
    """
    Configure the available models.
    
    Args:
        cost_effective_model: Model to use for cost-effective mode
        premium_model: Model to use for premium mode
    """
    config = get_config()
    
    if cost_effective_model:
        config.cost_effective_model = cost_effective_model
        logger.info(f"💡 Cost-effective model updated to: {cost_effective_model}")
    
    if premium_model:
        config.premium_model = premium_model
        logger.info(f"🔥 Premium model updated to: {premium_model}")
    
    # Reconfigure DSPy with current mode
    reconfigure_dspy(config)
    
    print(f"✅ Model configuration updated:")
    if cost_effective_model:
        print(f"   Cost-effective: {cost_effective_model}")
    if premium_model:
        print(f"   Premium: {premium_model}")

# Convenience aliases
cost_effective = switch_to_cost_effective
premium = switch_to_premium
status = print_model_status