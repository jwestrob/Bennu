#!/usr/bin/env python3
"""
Small LM factory to construct per-step model clients based on a model id string.

Supports:
- OpenAI reasoning models (e.g., openai/gpt-5-*, o1-*): chat-style minimal config
- OpenAI non-reasoning models (e.g., openai/gpt-4.1-mini): chat style
- OpenRouter Sonnet 4: uses dspy.OpenAI with OpenRouter api_base

Behavioral notes:
- Avoid specifying any max_tokens parameter for GPT-5/o1 per project policy.
- When GPT-5 aliases include an effort suffix (gpt-5-minimal|low|medium|high), attach
  reasoning={"effort": <level>} to the LM so the API uses the desired effort.
"""

import os
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


def _resolve_alias(model_id: str) -> Optional[str]:
    """Resolve friendly aliases to provider/model ids.

    Returns a full provider/model string if alias is known; otherwise None.
    """
    s = (model_id or "").strip().lower()
    if not s:
        return None
    # Strip provider prefix if present for alias matching
    name = s.split("/", 1)[1] if "/" in s else s
    alias = {
        # GPT-5 friendly names map to the current GPT-5 release id
        "gpt-5": "openai/gpt-5-2025-08-07",
        "gpt-5-high": "openai/gpt-5-2025-08-07",
        "gpt-5-premium": "openai/gpt-5-2025-08-07",
        "gpt-5-minimal": "openai/gpt-5-2025-08-07",
        "gpt-5-mini": "openai/gpt-5-2025-08-07",
        "gpt-5-medium": "openai/gpt-5-2025-08-07",
        "gpt-5-low": "openai/gpt-5-2025-08-07",
        # Handy short-hands for 4.1-mini
        "gpt-4.1-mini": "openai/gpt-4.1-mini",
        "4.1-mini": "openai/gpt-4.1-mini",
        # Sonnet 4 aliases route to OpenRouter by default
        "claude-sonnet-4": "anthropic/claude-sonnet-4",
        "sonnet-4": "anthropic/claude-sonnet-4",
    }
    return alias.get(name)


def _normalize_model(model_id: str) -> str:
    mid = (model_id or "").strip()
    if not mid:
        return mid
    # Preserve explicit OpenRouter routing
    if mid.lower().startswith("openrouter/"):
        return mid
    # Resolve aliases first
    resolved = _resolve_alias(mid)
    if resolved:
        return resolved
    # If no provider prefix given, assume OpenAI
    if '/' not in mid:
        return f"openai/{mid}"
    return mid


def _extract_gpt5_effort(model_id: str) -> Optional[str]:
    """Extract reasoning effort from GPT-5 alias.

    Accepts: gpt-5-minimal | gpt-5-low | gpt-5-medium | gpt-5-high (with or without provider prefix).
    Returns one of: minimal, low, medium, high; otherwise None.
    """
    if not model_id:
        return None
    # Consider raw alias (without provider) for effort parsing
    alias = model_id.split("/", 1)[1] if "/" in model_id else model_id
    a = alias.strip().lower()
    if a.startswith("gpt-5-"):
        # map suffix → effort
        if a.endswith("-minimal"):
            return "minimal"
        if a.endswith("-low"):
            return "low"
        if a.endswith("-medium"):
            return "medium"
        if a.endswith("-high"):
            return "high"
    return None


def make_lm(model_id: str, step: str = "") -> Any:
    """
    Create a DSPy-compatible LM object based on a model id string.

    - For GPT-5/o1 reasoning: use dspy.LM with model_type='responses', temperature=1.0
    - For non-reasoning OpenAI (gpt-4.1 family): use dspy.LM, temperature=0.0
    - For anthropic/claude-sonnet-4 via OpenRouter: use dspy.OpenAI with api_base to OpenRouter

    Never sets max_tokens; also drops any accidental token-limit params when supported.
    """
    import dspy

    # Preserve original for effort extraction before normalization
    effort = _extract_gpt5_effort(model_id)
    model = _normalize_model(model_id)
    lower = model.lower()

    # OpenRouter provider explicit prefix: openrouter/<model>
    # Example: openrouter/claude-sonnet-4 or openrouter/anthropic/claude-sonnet-4
    if lower.startswith("openrouter/"):
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set for OpenRouter provider")
        # Convert to OpenRouter's model id: prefer vendor-prefixed form
        inner = model.split("/", 1)[1]
        if "/" not in inner:
            # Assume Anthropic when only a bare model name like 'claude-sonnet-4' is given
            routed_model = f"anthropic/{inner}"
        else:
            routed_model = inner
        # Route dspy.LM via OpenRouter by setting OpenAI-compatible env vars
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = base_url
        # Minimal LM; drop params that could introduce caps or wrong routes
        # IMPORTANT: force OpenAI provider path by prefixing model with 'openai/'
        lm = dspy.LM(
            model=f"openai/{routed_model}",
            drop_params=True,
            additional_drop_params=[
                "max_tokens",
                "max_output_tokens",
                "max_completion_tokens",
                "response_format",
            ],
        )
        try:
            for k in ("max_tokens", "max_output_tokens", "max_completion_tokens", "response_format"):
                lm.kwargs.pop(k, None)
        except Exception:
            pass
        return lm

    # Native Anthropic provider
    if lower.startswith("anthropic/"):
        # Use native Anthropic via dspy.LM, no max_tokens
        lm = dspy.LM(
            model=model,
            # Anthropic typically ignores temperature on some models; keep default
        )
        return lm

    # OpenAI reasoning models (GPT-5, o1): use chat semantics like the original Planner
    # - No model_type override
    # - No temperature
    # - No token caps (respect user's request to exclude max_tokens)
    # - Drop params that can force wrong routes or caps if injected by adapters
    if ("/gpt-5" in lower) or ("/o1" in lower):
        # Attach GPT-5 reasoning effort if requested via alias
        kwargs = dict(
            model=model,
            drop_params=True,
            additional_drop_params=[
                "response_format",
                "max_tokens",
                "max_output_tokens",
                "max_completion_tokens",
            ],
        )
        if ("/gpt-5" in lower) and effort:
            # litellm/OpenAI adapter expects 'reasoning_effort' top-level param
            # rather than a nested 'reasoning' dict; set it here so the
            # underlying client forwards it as-is.
            kwargs["reasoning_effort"] = effort
            try:
                logger.info(f"🎯 GPT-5 reasoning effort set: step='{step}' effort='{effort}' model='{model}'")
            except Exception:
                pass
        lm = dspy.LM(**kwargs)
        try:
            for k in ("response_format", "max_tokens", "max_output_tokens", "max_completion_tokens"):
                lm.kwargs.pop(k, None)
        except Exception:
            pass
        return lm

    # Default: non-reasoning OpenAI models (4.1 family etc.)
    # To avoid truncation by dspy's default (e.g., 4000), explicitly set a high max_tokens.
    # Applies to GPT-4.1 family and similar non-reasoning OpenAI models.
    max_toks = 30000 if ("/gpt-4.1" in lower) else None
    if max_toks is not None:
        lm = dspy.LM(
            model=model,
            temperature=0.0,
            max_tokens=max_toks,
        )
    else:
        lm = dspy.LM(
            model=model,
            temperature=0.0,
        )
    return lm
