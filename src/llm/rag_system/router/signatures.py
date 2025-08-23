from dataclasses import dataclass

try:
    import dspy  # type: ignore
    DSPY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    DSPY_AVAILABLE = False


@dataclass
class RouterDecision:
    tool: str
    params: dict
    reasoning: str | None = None


if DSPY_AVAILABLE:
    class ToolRoute(dspy.Signature):  # type: ignore
        question = dspy.InputField()
        context = dspy.InputField()
        # Optional hints to inform routing; callers may pass empty strings
        data_profile = dspy.InputField(desc="Dataset scale/complexity summary; may be empty")
        policy_hints = dspy.InputField(desc="Generic routing hints; may be empty")
        db_templates_catalog = dspy.InputField(desc="JSON catalog of available DB templates and slots; may be empty")
        tool_costs = dspy.InputField(desc="JSON map of tool cost tags; may be empty")
        tool = dspy.OutputField(choices=[
            "database_query",
            "whole_genome_reader",
            "neighborhood_extractor",
            "similarity_search",
            "code_interpreter",
            "literature_search",
            "synthesize",
        ])
        params = dspy.OutputField()
        # Advisory flag; not enforced by router
        approval_hint = dspy.OutputField(desc="true if the route likely requires approval; optional")

    class ToolRouteRepair(dspy.Signature):  # type: ignore
        instruction = dspy.InputField()
        bad = dspy.InputField()
        schema = dspy.InputField()
        json = dspy.OutputField(desc="Return ONLY a JSON object matching the schema")
