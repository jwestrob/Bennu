# Regression Tests Overview

This suite exercises the typed router, template compiler, similarity interface (at a unit level), and tracing. Tests default to fast, dependency-light execution. Integration/E2E paths are intentionally avoided unless explicitly marked.

What’s covered:
- Router determinism and schema validity (Stage A/B with a stubbed LLM router)
- Cypher template compilation and slot validation (no free-form Cypher)
- Tracing JSONL emission
- Schema validator strictness (reject unknown fields)
- Router regression scaffold: JSON set exercises Stage A detection and template mapping without LLM
 - Snapshot scaffold for synthesis formatting (skips if package import constrained)

How to run (fast set):
- Single test example:
  - `pytest -q tests/regression/test_router.py::test_stage_a_routes_spatial_to_wgr`
- Run a file:
  - `pytest -q tests/regression/test_templates.py`
  - `pytest -q tests/regression/test_router_regression.py`
  - `pytest -q tests/regression/test_synthesis_snapshot.py`

Notes:
- No APOC/GDS calls are used in these tests.
- Similarity tests that would hit LanceDB or ESM2 are deferred; we validate ordering/filters via unit functions elsewhere.
- The tracer is default-on; tests use an isolated JSONL path via env override.

Environment:
- The project’s `env/environment.yml` includes `pytorch` and `transformers`. Runtime sequence embedding is guarded; if those deps are missing, related paths surface clear errors and are not exercised here.
