try:
    import dspy  # type: ignore
    DSPY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    DSPY_AVAILABLE = False


if DSPY_AVAILABLE:
    class ToolRoute(dspy.Signature):  # type: ignore
        question = dspy.InputField()
        context = dspy.InputField()
        tool = dspy.OutputField(choices=[
            "database_query",
            "whole_genome_reader",
            "similarity_search",
            "code_interpreter",
            "literature_search",
            "synthesize",
        ])
        params = dspy.OutputField()

    class ToolRouteRepair(dspy.Signature):  # type: ignore
        instruction = dspy.InputField()
        bad = dspy.InputField()
        schema = dspy.InputField()
        json = dspy.OutputField(desc="Return ONLY a JSON object matching the schema")

