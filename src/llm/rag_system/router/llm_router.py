import json
import logging
from typing import Any, Dict, Optional

try:
    import dspy  # type: ignore
    DSPY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    DSPY_AVAILABLE = False

from ..agent.tools.validate import validate_toolcall, make_repair_prompt
from ..agent.tools.schemas import TOOLCALL_JSON_SCHEMA
from ..memory.model_allocation import get_model_allocator
from ..tracing import get_tracer
from .signatures import ToolRoute, ToolRouteRepair  # type: ignore
from .two_stage import RouterDecision


logger = logging.getLogger(__name__)


class LLMRouter:
    """
    Single LLM-based router that emits strictly typed toolcalls.
    Validation-first: one repair attempt, then fail closed.
    """

    def __init__(self) -> None:
        if not DSPY_AVAILABLE:
            logger.warning("LLMRouter initialized without DSPy; will raise if used.")
        self.model_allocator = get_model_allocator()
        self._tracer = get_tracer()

    def _predict_route(self, question: str, context: Optional[str]) -> Dict[str, Any]:
        def predict_call(module):
            return module(question=question, context=context or "")

        result = self.model_allocator.create_context_managed_call(
            task_name="tool_routing",
            signature_class=ToolRoute,  # type: ignore
            module_call_func=predict_call,
            query=question,
            task_context="router.stage_b",
        )
        if not result:
            raise RuntimeError("LLM router returned no result")

        tool = getattr(result, "tool", None)
        params = getattr(result, "params", {})
        try:
            if isinstance(params, str):
                params = json.loads(params)
        except Exception:
            # Keep raw; validation will fail and trigger repair
            pass
        obj = {"tool": tool, "params": params}
        try:
            self._tracer.emit("router.stage_b.raw", {"route": obj})
        except Exception:
            pass
        return obj

    def _repair(self, bad_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        instruction = make_repair_prompt(bad_obj, ["schema validation failed"])  # brief; real errors included earlier

        def repair_call(module):
            return module(
                instruction=instruction,
                bad=json.dumps(bad_obj, ensure_ascii=False),
                schema=json.dumps(TOOLCALL_JSON_SCHEMA, ensure_ascii=False),
            )

        fixed = self.model_allocator.create_context_managed_call(
            task_name="tool_routing_repair",
            signature_class=ToolRouteRepair,  # type: ignore
            module_call_func=repair_call,
            query="router_repair",
            task_context="router.stage_b.repair",
        )
        if not fixed:
            return None
        text = getattr(fixed, "json", "{}")
        try:
            obj = json.loads(text)
            return obj
        except Exception:
            return None

    def route(self, question: str, context: Optional[str] = None) -> RouterDecision:
        if not DSPY_AVAILABLE:
            raise RuntimeError("Stage B LLM router requires DSPy; not available")

        # Predict
        raw = self._predict_route(question, context)

        # Validate
        ok, errs = validate_toolcall(raw)
        if not ok:
            logger.error(f"Stage B route invalid: {'; '.join(errs)}. Attempting repair.")
            try:
                self._tracer.emit("router.stage_b.invalid", {"errors": errs, "bad": raw})
            except Exception:
                pass
            # Single repair attempt
            repaired = self._repair(raw)
            if not repaired:
                raise ValueError("Stage B repair failed: could not produce valid JSON")
            ok2, errs2 = validate_toolcall(repaired)
            if not ok2:
                raise ValueError(f"Stage B repair still invalid: {'; '.join(errs2)}")
            raw = repaired
        try:
            self._tracer.emit("router.stage_b.decision", {"route": raw})
        except Exception:
            pass

        return RouterDecision(tool=raw["tool"], params=raw["params"], reasoning="StageB: schema-valid toolcall")
