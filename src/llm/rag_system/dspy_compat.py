"""
DSPy compatibility wrapper for GPT-5 models.

This module provides compatibility fixes for DSPy's OpenAI integration
to properly handle GPT-5 models that require max_completion_tokens instead of max_tokens.
"""

import logging
from typing import Any, Optional

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

logger = logging.getLogger(__name__)


def _needs_max_completion_tokens(model: str) -> bool:
    """
    Check if a model requires max_completion_tokens instead of max_tokens.
    """
    return any(model.startswith(prefix) or prefix in model for prefix in [
        "gpt-5", "o1", "o3", "o4", "o1-", "o3-", "o4-"
    ])
        
        
def create_compatible_lm(model: str, **kwargs) -> Any:
    """
    Create a compatible Language Model instance for DSPy.
    
    This function automatically handles parameter mapping for different model families.
    
    Args:
        model: Model string (e.g., "openai/gpt-5-mini-2025-08-07")
        **kwargs: Parameters for the language model
        
    Returns:
        DSPy Language Model instance
    """
    if not DSPY_AVAILABLE:
        raise ImportError("DSPy not available")
    
    # Extract max_tokens if present
    mt = kwargs.pop("max_tokens", None)
    
    # Check if this model needs max_completion_tokens
    if mt is not None and _needs_max_completion_tokens(model):
        kwargs["max_completion_tokens"] = mt
        logger.debug(f"🔄 Mapped max_tokens={mt} → max_completion_tokens={mt} for {model}")
    elif mt is not None:
        kwargs["max_tokens"] = mt
        logger.debug(f"✅ Using max_tokens={mt} for {model}")
    
    # Create regular DSPy LM with corrected parameters
    return dspy.LM(model=model, **kwargs)


def test_gpt5_compatibility():
    """Test function to verify GPT-5 compatibility."""
    if not DSPY_AVAILABLE:
        print("❌ DSPy not available")
        return False
        
    try:
        # Test creating LM without max_tokens
        lm1 = create_compatible_lm("openai/gpt-5-mini-2025-08-07", temperature=0.0)
        print("✅ GPT-5 LM created without max_tokens")
        
        # Test creating LM with max_tokens (should be converted)
        lm2 = create_compatible_lm("openai/gpt-5-mini-2025-08-07", temperature=0.0, max_tokens=100)
        print("✅ GPT-5 LM created with max_tokens → max_completion_tokens mapping")
        
        # Test with regular GPT-4 (should use max_tokens)
        lm3 = create_compatible_lm("openai/gpt-4o-mini", temperature=0.0, max_tokens=100)
        print("✅ GPT-4 LM created with max_tokens (no mapping)")
        
        # Try a simple test call
        with dspy.context(lm=lm2):
            result = dspy.Predict('question -> answer')(question='What is 2+2?')
            print(f"✅ GPT-5 test call successful: {result.answer[:50] if hasattr(result, 'answer') else str(result)[:50]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        return False


if __name__ == "__main__":
    test_gpt5_compatibility()