from __future__ import annotations
from typing import Tuple, Optional
import json
from pydantic import ValidationError
from .models import CanonicalIntent, TaskFamily


SYSTEM_PROMPT = (
    "You convert a biological natural-language task into a STRICT Canonical JSON for a DSL compiler.\n"
    "Rules:\n"
    "- Only emit keys declared in the JSON schema below; extra keys are forbidden.\n"
    "- Choose exactly one of: FIND_LOCI_BY_MARKER | FIND_LOCI_BY_SIGNATURE | PATHWAY_COMPLETENESS.\n"
    "- 'actions' is an array of optional post-steps; allowed: LANCEDB_KNN, SUMMARIZE_NEIGHBORHOOD.\n"
    "- Do NOT explain. Output JSON only.\n\n"
    "JSON schema (informal):\n"
    "{\n"
    "  'task': 'FIND_LOCI_BY_MARKER' | 'FIND_LOCI_BY_SIGNATURE' | 'PATHWAY_COMPLETENESS',\n"
    "  'n': <int>,  // For PATHWAY_COMPLETENESS, set to 1\n"
    "  'find_by_marker': {'marker': <str>, 'flank_k': <int>} | null,\n"
    "  'find_by_signature': {'signature_name': <str>, 'flank_k': <int>} | null,\n"
    "  'genome_ids': [<str>] | null,\n"
    "  'min_completeness': <float 0..1> | null,\n"
    "  'pathways': [<str>] | null,  // Optional filter of KEGG map IDs (e.g., 'map00010'); if omitted, use all\n"
    "  'actions': [\n"
    "    {'kind': 'LANCEDB_KNN', 'top_k': <int>, 'exclude_pfam': [<str>]} |\n"
    "    {'kind': 'SUMMARIZE_NEIGHBORHOOD'}\n"
    "  ],\n"
    "  'version_tag': 'v1'\n"
    "}\n"
    "Constraints:\n"
    "- If task = FIND_LOCI_BY_MARKER, 'find_by_marker' must be present and 'find_by_signature' must be null, and vice versa.\n"
    "- If task = PATHWAY_COMPLETENESS, 'find_by_marker' and 'find_by_signature' must be null (set 'n' to 1).\n"
    "- Prefer concise canonicalization (choose reasonable flank_k if omitted in NL).\n"
)


def _write_note(note_keeper, rel_path: str, payload: object) -> None:
    try:
        if not note_keeper or not hasattr(note_keeper, "session_path"):
            return
        base = note_keeper.session_path
        dbg = base / "debug_data_flow"
        dbg.mkdir(exist_ok=True)
        p = dbg / rel_path
        if isinstance(payload, (dict, list)):
            with open(p, "w") as f:
                json.dump(payload, f, indent=2)
        else:
            with open(p, "w") as f:
                f.write(str(payload))
    except Exception:
        # Best-effort only
        pass


def canonicalize(natural_query: str, note_keeper, model_allocator=None) -> Tuple[CanonicalIntent, str]:
    """LLM canonicalizer to convert NL into CanonicalIntent; persists artifacts.

    Returns (intent, raw_json_string). Raises ValueError on validation errors.
    """
    try:
        import dspy  # type: ignore
        DSPY_AVAILABLE = True
    except Exception:
        DSPY_AVAILABLE = False

    if not DSPY_AVAILABLE:
        raise RuntimeError("Canonicalizer requires DSPy")

    # Define a minimal signature for a JSON-only return
    class CanonicalizerSignature(dspy.Signature if DSPY_AVAILABLE else object):  # type: ignore
        instruction = dspy.InputField(desc="System rules for canonicalization")  # type: ignore
        natural_request = dspy.InputField(desc="Natural-language task")  # type: ignore
        canonical_json = dspy.OutputField(desc="Return ONLY the canonical JSON")  # type: ignore

    def call(module):
        return module(instruction=SYSTEM_PROMPT, natural_request=natural_query)

    # Hard-wire GPT-5 with medium reasoning effort for canonicalization.
    # Falls back to allocator (mini) if unsupported in this environment.
    result = None
    try:
        lm = None
        try:
            # Preferred: GPT-5 (2025-08-07) with reasoning_effort=medium
            lm = dspy.LM(
                model="openai/gpt-5-2025-08-07",
                temperature=0.0,
                max_completion_tokens=8000,  # GPT-5 rejects max_tokens
                reasoning_effort="medium",   # hint for reasoning models
            )
        except TypeError:
            # dspy.LM may not accept GPT-5-specific kwargs; try minimal args
            lm = dspy.LM(model="openai/gpt-5-2025-08-07", temperature=1.0)
        module = dspy.Predict(CanonicalizerSignature)
        with dspy.context(lm=lm):
            result = call(module)
    except Exception as e:
        # Fallback to allocator path (cost-optimized) if available
        if model_allocator is None:
            raise RuntimeError(f"Canonicalizer GPT-5 path failed and no allocator provided: {e}")
        result = model_allocator.create_context_managed_call(
            task_name="query_classification",
            signature_class=CanonicalizerSignature,
            module_call_func=call,
            query=natural_query,
            task_context="canonicalize to strict JSON (fallback)",
        )
        if result is None:
            raise RuntimeError("Canonicalizer LLM call failed (fallback)")

    raw = getattr(result, "canonical_json", None) or ""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Canonicalizer produced empty output")

    # Validate into a CanonicalIntent
    try:
        intent = CanonicalIntent.model_validate_json(raw)
    except ValidationError as e:  # type: ignore
        raise ValueError(f"CanonicalizerValidationError: {e}") from e

    # Persist artefacts
    _write_note(note_keeper, "canonicalizer.json", intent.to_minimal_dict())
    _write_note(note_keeper, "canonicalizer.raw.txt", raw)
    return intent, raw
