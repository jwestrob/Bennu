# GenomicRAG Agent Refactor Plan

Owner: Jacob West-Roberts
Branch: `feat/agent-router-typed`
Policy: Deterministic, typed, template-driven; no cardinality features in this plan.

## Executive Summary

Why: The current system has duplicate/overlapping routers, free-form tool parameters, and a nondeterministic agent loop. This causes brittle behavior, security risk (LLM-generated Cypher), and hard-to-reproduce results.

What: Replace legacy routing and free-form actions with a deterministic, typed pipeline:
- Two-stage router (A: deterministic guardrail; B: single LLM router) producing a strictly validated toolcall.
- Typed toolcalls (Pydantic + jsonschema) with hard rejection/repair for invalid fields; no free-form params.
- ActionGraph finite state machine governs execution; only enumerated transitions; bounded depth/timeouts.
- All Neo4j queries come from a curated template/slot library; no arbitrary Cypher from the model.
- Immutable `GenomeScope` propagated across all processors; tools cannot override it.
- Strong observability (Langfuse/LangSmith), a router regression set, and snapshot/property tests for determinism.
- ESM2/LanceDB hardening and batch kNN with deterministic tie-breakers.

Note: Cardinality hints/policies are intentionally excluded per directive.

## Architecture: Current → Target (Mermaid diagram)

```mermaid
flowchart TD
  A[Front Door: GenomicRAG.ask] --> B{Stage A Guardrail}
  B -- force --> T1[whole_genome_reader|database_query|similarity_search]
  B -- allow --> C[Stage B LLM Router (typed)]
  C -->|toolcall(schema-valid)| D[ActionGraph]
  D --> DB[DB Query] --> ACC[Accumulate] --> DEC{Decide}
  D --> SIM[ESM2 kNN (LanceDB)]
  D --> WGR[Whole Genome Reader]
  DEC -->|more| A2[Plan]
  DEC -->|synthesize| SYN[Synthesis Split: Evidence -> Narrative]
```

## Design Decisions & Rationale

- Two-stage routing: deterministic veto/force for obvious intents reduces LLM variance; a single LLM router centralizes policy.
- Typed toolcalls: Pydantic/jsonschema ensures strict schemas, allowing safe backend enforcement and repair prompts.
- FSM over loop: an ActionGraph with explicit states prevents oscillations and enforces limits, improving reproducibility.
- Template Cypher: named, versioned queries with slots eliminate arbitrary query generation and tighten security.
- GenomeScope: an immutable contextual envelope prevents accidental scope drift across tools and steps.
- Observability/eval: tracing + regression/snapshot tests provide accountability and reproducibility in CI.
- ESM2/LanceDB: dimension assertion and deterministic batching remove hidden nondeterminism and early breakages.

## Planned Changes (T1…Tn) with Acceptance Criteria

- [ ] T1: Quarantine & remove legacy planners/selectors
  - Step: Identify and delete/disable legacy task planner/executor and duplicate tool selectors; leave a feature flag for instant rollback.
  - Acceptance: Only one router module remains importable; CI passes.

- [ ] T2: Introduce typed toolcall schemas & validators
  - Steps:
    - Create `agent/tools/schemas.py` (Pydantic models) and `agent/tools/validate.py` (jsonschema validation); reject on unknown fields; emit repair prompts.
    - Make the router output match the schema; block execution on invalid params.
  - Acceptance: 100% of toolcalls pass schema checks; repair path tested.

- [ ] T3: Two-stage router
  - Steps:
    - Implement Stage A guardrail (pure rules); Stage B LLM router (single place).
    - Log router_input/decision/params/results into tracing.
  - Acceptance: Regression set: ≥95% correct tool selection; deterministic across seeds.

- [ ] T4: ActionGraph state machine
  - Step: Replace loop with typed FSM; encode legal transitions; enforce max_depth/timeout from policy.
  - Acceptance: No oscillation; traces show one of the enumerated paths only.

- [ ] T5: Cypher template library
  - Step: Add `kg/cypher_templates/*.cypher` with named templates and slot specs; compiler fills params; validator runs post-compile.
  - Acceptance: 0 free-form Cypher from LLM; all queries derive from templates.

- [ ] T6: GenomeScope propagation
  - Step: Define immutable `GenomeScope` object; thread through processors; forbid overrides.
  - Acceptance: All toolcalls carry a scope; unit tests enforce immutability.

- [ ] T7: Observability and evaluation
  - Step: Wire Langfuse/LangSmith; add router regression set; snapshot tests; metrics export.
  - Acceptance: CI job `router_eval` passes; dashboards show traces for each state.

- [ ] T8: ESM2/LanceDB hardening + batch kNN
  - Step: Dimension assertion; batch API with deterministic merges; filters honored.
  - Acceptance: Unit tests for dim mismatch; batch returns stable ordering; perf within budget.

## Test Plan & Benchmarks

- Property tests: fuzz router params and schema validation (Hypothesis); assert rejection/repair paths are correct and deterministic.
- Snapshot tests: freeze seeds for router outputs and synthesis text (`pytest-snapshot`).
- Regression set: 30–50 canonical prompts (spatial, similarity, KO filters, counts, “explain hits”); used to assert routing correctness.
- Performance budgets: max toolcalls per question; per-state timeouts; red-line in CI if exceeded.
- Determinism: set model seeds, stable sampling params; disallow temperature > 0 for routing.

## Risk Register & Rollback

- Router consolidation risk: behavior changes on edge prompts.
  - Mitigation: Feature flag `AGENT_ROUTER_V2=on`; maintain legacy path behind flag for one release.
- Schema strictness: initial invalid toolcalls block execution.
  - Mitigation: Implement repair prompts; fail closed with clear error surfaces and traces.
- Template coverage gaps: missing Cypher templates for rare queries.
  - Mitigation: Add fast-follow templates; no fallback to free-form Cypher.
- ESM2/LanceDB mismatch: embedding_dim drift.
  - Mitigation: Hard startup assertion; CI check; explicit migration guide.

## Design Artifacts (Schemas, Pseudocode, Paths)

Toolcall JSON Schema (no cardinality):

```json
{
  "type": "object",
  "required": ["tool", "params"],
  "additionalProperties": false,
  "properties": {
    "tool": {
      "enum": [
        "database_query",
        "whole_genome_reader",
        "similarity_search",
        "code_interpreter",
        "literature_search",
        "synthesize"
      ]
    },
    "params": {
      "oneOf": [
        {
          "type": "object",
          "required": ["template", "slots"],
          "additionalProperties": false,
          "properties": {
            "template": {
              "enum": [
                "protein_by_id",
                "proteins_with_ko",
                "neighbors_by_window",
                "pathway_membership",
                "count_by_label",
                "cazy_family"
              ]
            },
            "slots": { "type": "object" }
          }
        },
        {
          "type": "object",
          "required": ["mode", "k"],
          "additionalProperties": false,
          "properties": {
            "mode": { "enum": ["by_id", "by_sequence"] },
            "id": { "type": "string" },
            "sequence": { "type": "string" },
            "k": { "type": "integer", "minimum": 1, "maximum": 1000 },
            "filters": { "type": "object" }
          }
        },
        {
          "type": "object",
          "required": ["window_bp", "loci_limit"],
          "additionalProperties": false,
          "properties": {
            "window_bp": { "type": "integer", "minimum": 100, "maximum": 2000000 },
            "loci_limit": { "type": "integer", "minimum": 1, "maximum": 5000 }
          }
        }
      ]
    }
  }
}
```

DSPy Signatures (sketch):

```python
# agent/router/signatures.py
class ToolRoute(dspy.Signature):
    question = dspy.InputField()
    context  = dspy.InputField()
    tool     = dspy.OutputField(choices=[
        "database_query","whole_genome_reader","similarity_search",
        "code_interpreter","literature_search","synthesize"
    ])
    params   = dspy.OutputField()  # validated against JSON Schema above

# agent/plans/structured.py
@dataclass
class StructuredCypherPlan:
    template: Literal[
        "protein_by_id","proteins_with_ko","neighbors_by_window",
        "pathway_membership","count_by_label","cazy_family"
    ]
    slots: Dict[str, Any]

@dataclass
class SimilarityPlan:
    mode: Literal["by_id","by_sequence"]
    k: int
    id: Optional[str] = None
    sequence: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SpatialGenomePlan:
    window_bp: int
    loci_limit: int
```

ActionGraph (FSM) sketch:

```python
# agent/fsm/action_graph.py
class State(Enum):
    PLAN = auto()
    DB = auto()
    SIM = auto()
    GENOME = auto()
    ACCUM = auto()
    DECIDE = auto()
    SYN = auto()

LEGAL: Dict[State, Tuple[State, ...]] = {
    State.PLAN: (State.DB, State.SIM, State.GENOME),
    State.DB: (State.ACCUM,),
    State.SIM: (State.ACCUM,),
    State.GENOME: (State.ACCUM,),
    State.ACCUM: (State.DECIDE,),
    State.DECIDE: (State.PLAN, State.SYN),
    State.SYN: tuple(),
}
```

GenomeScope definition:

```python
# agent/context/scope.py
@dataclass(frozen=True)
class GenomeScope:
    genome_id: str
    contig_ids: Tuple[str, ...]
    coordinate_window: Tuple[int, int]
```

Cypher templates (examples):

```cypher
// kg/cypher_templates/protein_by_id.cypher
MATCH (p:Protein {id:$id}) RETURN p LIMIT 1;

// kg/cypher_templates/proteins_with_ko.cypher
MATCH (p:Protein)-[:HAS_KO]->(k:KO {id:$ko}) RETURN p;

// kg/cypher_templates/neighbors_by_window.cypher
MATCH (g:Gene {contig:$contig}) WHERE g.start >= $start AND g.end <= $end
RETURN g ORDER BY g.start;
```

Observability & evaluation:

- Tracing: Langfuse or LangSmith for per-step traces (router input → decision → params → result).
- Metrics: `tool_error_rate`, `schema_repair_rate`, `router_agreement(StageA/B)`, `latency_by_state`, DB round-trips.
- CI job `router_eval` runs the regression and snapshot suites.

LanceDB/ESM2:

- Startup asserts `embedding_dim` matches LanceDB table; hard-fail on mismatch.
- Batch kNN API (N×k) with deterministic tie-breakers (score → length → id); filters honored.

GDS usage policy:

- Expose selected Neo4j GDS algorithms only via curated backend wrappers; keep CALL disabled to the LLM.

## Changelog (auto-appended)

- 2025-08-21 09:48:41Z — Run started.
- 2025-08-21 09:49:52Z — Created branch `feat/agent-router-typed`.
- 2025-08-21 09:49:52Z — Committed initial plan seed.
- 2025-08-21 09:57:11Z — Added unified router skeleton (`src/llm/rag_system/router`) and feature flags: `AGENT_ENABLE_LEGACY_TASKGRAPH`, `AGENT_DISABLE_LEGACY_SELECTORS` (default off) to prepare T1 quarantine.
- 2025-08-21 09:57:11Z — Added typed toolcall schemas and validators at `src/llm/rag_system/agent/tools/{schemas.py,validate.py}` (no cardinality).

## Open Questions

- Which tracing provider to standardize on (Langfuse vs LangSmith)?
- Final module boundaries/names for router/FSM (`agent/router/two_stage.py`, `agent/fsm/action_graph.py`)?
- Scope seeding: where `GenomeScope` is set (front door vs Stage A)?
- Minimum template set for initial pass; which queries first?
