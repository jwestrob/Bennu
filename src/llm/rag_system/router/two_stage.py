import os
from typing import Any, Dict, Optional
import logging

from .llm_router import LLMRouter
from ..agent.tools.validate import validate_toolcall
from ..tracing import get_tracer
from .signatures import RouterDecision


class TwoStageRouter:
    """
    Placeholder two-stage router. Stage A (deterministic) and Stage B (LLM)
    will be implemented in subsequent tasks. For now, this provides a stable
    import surface to consolidate routing.
    """

    def __init__(self) -> None:
        self._legacy_enabled = os.getenv("AGENT_ENABLE_LEGACY_SELECTORS", "0") == "1"
        self._llm = LLMRouter()
        self._log = logging.getLogger(__name__)
        self._tracer = get_tracer()

    def route(self, question: str, context: Optional[str] = None) -> RouterDecision:
        """
        Temporary deterministic stub: routes obvious spatial queries to
        `whole_genome_reader`, else `database_query`.

        NOTE: This is intentionally deterministic and minimal; Stage B LLM
        will be added in T3.
        """
        q = (question or "").lower()
        # Trace router input
        try:
            self._tracer.emit("router.input", {"question": question, "has_context": bool(context)})
        except Exception:
            pass

        spatial_markers = ("genome", "contig", "locus", "neighborhood", "operon", "coordinates")
        if any(tok in q for tok in spatial_markers):
            # Deterministic defaults for Stage A spatial routing
            decision = RouterDecision(
                tool="whole_genome_reader",
                params={
                    "window_bp": 20000,
                    "loci_limit": 2000,
                },
                reasoning="StageA: spatial markers detected; forcing whole_genome_reader with defaults",
            )
            try:
                self._tracer.emit("router.stage_a.decision", {"route": {"tool": decision.tool, "params": decision.params}})
            except Exception:
                pass
            return decision

        # Stage B: Single LLM router with strict schema validation
        try:
            self._tracer.emit("router.stage_b.start", {})
            decision = self._llm.route(question, context)
            # Extra safety check on params before returning
            ok, errs = validate_toolcall({"tool": decision.tool, "params": decision.params})
            if not ok:
                raise ValueError("Post-check validation failed: " + "; ".join(errs))
            return decision
        except Exception as e:
            self._log.error(f"Stage B routing failed: {e}")
            # Fail closed: return minimal safe default for downstream flow
            fallback = RouterDecision(tool="database_query", params={}, reasoning="StageB failed; defaulting to database_query")
            try:
                self._tracer.emit("router.stage_b.error", {"error": str(e)})
                self._tracer.emit("router.decision", {"route": {"tool": fallback.tool, "params": fallback.params}})
            except Exception:
                pass
            return fallback
