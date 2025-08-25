from __future__ import annotations
from typing import Tuple, Optional
import json
from pydantic import ValidationError
from .models import CanonicalIntent, TaskFamily


SYSTEM_PROMPT = (
    "You convert a biological natural-language task into a STRICT Canonical JSON for a DSL compiler.\n"
    "Rules:\n"
    "- Only emit keys declared in the JSON schema below; extra keys are forbidden.\n"
    "- Choose exactly one of: FIND_LOCI_BY_MARKER or FIND_LOCI_BY_SIGNATURE for the primary task.\n"
    "- 'actions' is an array of optional post-steps; allowed: LANCEDB_KNN, SUMMARIZE_NEIGHBORHOOD.\n"
    "- Do NOT explain. Output JSON only.\n\n"
    "JSON schema (informal):\n"
    "{\n"
    "  'task': 'FIND_LOCI_BY_MARKER' | 'FIND_LOCI_BY_SIGNATURE',\n"
    "  'n': <int>,\n"
    "  'find_by_marker': {'marker': <str>, 'flank_k': <int>} | null,\n"
    "  'find_by_signature': {'signature_name': <str>, 'flank_k': <int>} | null,\n"
    "  'actions': [\n"
    "    {'kind': 'LANCEDB_KNN', 'top_k': <int>, 'exclude_pfam': [<str>]} |\n"
    "    {'kind': 'SUMMARIZE_NEIGHBORHOOD'}\n"
    "  ],\n"
    "  'version_tag': 'v1'\n"
    "}\n"
    "Constraints:\n"
    "- If task = FIND_LOCI_BY_MARKER, 'find_by_marker' must be present and 'find_by_signature' must be null, and vice versa.\n"
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

    if not DSPY_AVAILABLE or model_allocator is None:
        raise RuntimeError("Canonicalizer requires DSPy and a model allocator")

    # Define a minimal signature for a JSON-only return
    class CanonicalizerSignature(dspy.Signature if DSPY_AVAILABLE else object):  # type: ignore
        instruction = dspy.InputField(desc="System rules for canonicalization")  # type: ignore
        natural_request = dspy.InputField(desc="Natural-language task")  # type: ignore
        canonical_json = dspy.OutputField(desc="Return ONLY the canonical JSON")  # type: ignore

    def call(module):
        return module(instruction=SYSTEM_PROMPT, natural_request=natural_query)

    result = model_allocator.create_context_managed_call(
        task_name="query_classification",  # cheap tier
        signature_class=CanonicalizerSignature,
        module_call_func=call,
        query=natural_query,
        task_context="canonicalize to strict JSON",
    )
    if result is None:
        raise RuntimeError("Canonicalizer LLM call failed")

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

