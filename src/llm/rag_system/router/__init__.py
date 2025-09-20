"""
Unified, typed router entrypoint.

Two-stage routing will live here: deterministic guardrail (Stage A) then
LLM router (Stage B). This module provides a single import surface for routing.
"""

from .two_stage import TwoStageRouter, RouterDecision

def get_router() -> TwoStageRouter:
    return TwoStageRouter()

__all__ = ["TwoStageRouter", "RouterDecision", "get_router"]

