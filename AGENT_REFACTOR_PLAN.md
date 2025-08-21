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
  - Progress:
    - [x] Legacy selectors disabled by default via `AGENT_ENABLE_LEGACY_SELECTORS` (set to 1 to re-enable).
    - [x] TaskGraph exports gated off by default via `AGENT_ENABLE_LEGACY_TASKGRAPH` (set to 1 to re-enable).

- [ ] T2: Introduce typed toolcall schemas & validators
  - Steps:
    - [x] Create `agent/tools/schemas.py` (Pydantic models) and `agent/tools/validate.py` (jsonschema validation); reject on unknown fields; emit repair prompts.
    - Make the router output match the schema; block execution on invalid params.
  - Acceptance: 100% of toolcalls pass schema checks; repair path tested.

- [ ] T3: Two-stage router
  - Steps:
    - [x] Implement Stage A guardrail (pure rules); Stage B LLM router (single place).
    - Log router_input/decision/params/results into tracing.
  - Notes:
    - Stage B implemented with strict schema validation and one repair attempt. Core currently logs non-spatial router decisions; full tracing to be added.
  - Acceptance: Regression set: ≥95% correct tool selection; deterministic across seeds.

- [ ] T4: ActionGraph state machine
  - Step: Replace loop with typed FSM; encode legal transitions; enforce max_depth/timeout from policy.
  - Acceptance: No oscillation; traces show one of the enumerated paths only.
  - Progress:
    - [x] FSM added with states/transitions; minimal enforcement integrated in `UnifiedAgentExecutor`.
    - [ ] Replace loop fully with FSM runner (planned follow-up).

- [ ] T5: Cypher template library
  - Step: Add `kg/cypher_templates/*.cypher` with named templates and slot specs; compiler fills params; validator runs post-compile.
  - Acceptance: 0 free-form Cypher from LLM; all queries derive from templates.
  - Progress:
    - [x] Templates added: `protein_by_id`, `proteins_with_ko`, `neighbors_by_window`, `pathway_membership`, `cazy_family`, `count_by_label` (label enum guarded).
    - [x] Safe compiler/registry created; parameterized execution through Neo4j.
    - [x] Stage B `database_query` wired to execute templates and synthesize.
    - [x] Agent path `database_query` now strict-template only in `UnifiedAgentExecutor`.
    - [x] Traditional path strict mode added (env `AGENT_DB_TEMPLATES_ONLY=1`): maps question to templates (protein_by_id, proteins_with_ko, cazy_family, pathway_membership); bypasses free-form Cypher.
    - [x] Expanded templates: `proteins_by_genome`, `genes_on_contig`, `proteins_with_pfam`, `count_proteins_with_ko`.
    - [x] Disabled auto-query free-form path in Neo4jQueryProcessor; require templates.
    - [x] Default `limit` injected from policy engine when missing.
    - [x] Added new count templates: `count_proteins_with_pfam`, `count_proteins_in_pathway`; mapper recognizes patterns (count PFxxxxx, count proteins in mapxxxxx).
    - [x] Added adjacency compilers: `gene_neighbors_k`, `protein_neighbors_k`; mapper recognizes neighbors of gene/protein with optional k.

- [ ] T6: GenomeScope propagation
  - Step: Define immutable `GenomeScope` object; thread through processors; forbid overrides.
  - Acceptance: All toolcalls carry a scope; unit tests enforce immutability.
  - Progress:
    - [x] `GenomeScope` dataclass added. Initial propagation hooks in core (foundation for threading scope through tools).
    - [ ] Thread scope through all tool invocations and attach to metadata consistently (WGR set; DB to follow via slots or metadata).

- [ ] T7: Observability and evaluation
  - Step: Wire Langfuse/LangSmith; add router regression set; snapshot tests; metrics export.
  - Acceptance: CI job `router_eval` passes; dashboards show traces for each state.
  - Progress:
    - [x] Internal lightweight tracing added; structured router events persisted to JSONL when enabled via `AGENT_TRACING`.
    - [x] MultiTracer scaffolding for Langfuse/LangSmith via env vars (no external deps), combined with JSONL tracing.
    - [ ] Add high-level router decision event for literature/code paths.

- [ ] T8: ESM2/LanceDB hardening + batch kNN
  - Step: Dimension assertion; batch API with deterministic merges; filters honored.
  - Acceptance: Unit tests for dim mismatch; batch returns stable ordering; perf within budget.
  - Progress:
    - [x] Optional manifest dimension logging; added deterministic `execute_similarity_batch`.
    - [ ] Add dimension assert in CI harness (out of runtime path).

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
- 2025-08-21 10:08:53Z — Wired Stage A router into traditional path; validate whole_genome_reader toolcalls; removed redundant spatial branch in `core.py`.
- 2025-08-21 10:12:42Z — Implemented Stage B LLM router with strict schema validation + single repair attempt; integrated into two-stage router; core logs non-spatial router decisions.
- 2025-08-21 10:15:26Z — Added lightweight tracing (`src/llm/rag_system/tracing.py`) and instrumented Stage A/B router to emit structured events. Enable via `AGENT_TRACING=jsonl:logs/agent_traces.jsonl`.
- 2025-08-21 10:20:13Z — Quarantined legacy selectors by default (set `AGENT_ENABLE_LEGACY_SELECTORS=1` to re-enable). Gated TaskGraph exports off by default. Enabled JSONL tracing by default and instrumented pipeline start/plan.
- 2025-08-21 10:24:21Z — Added Cypher template library (`src/llm/kg/cypher_templates`) and safe compiler/registry; wired Stage B `database_query` to execute named templates via Neo4j with parameters; short-circuits to synthesis.
- 2025-08-21 10:27:56Z — Implemented Stage B `similarity_search` (by_id) with deterministic ordering and filters; by_sequence not supported at runtime.
- 2025-08-21 10:36:47Z — Added runtime ESM2 embedder wrapper (mirrors pipeline manifest) and wired by_sequence similarity with dimension assertion; surfaces clear error if dependencies are missing.
- 2025-08-21 10:55:24Z — Implemented typed FSM and minimal enforcement in `UnifiedAgentExecutor`; enforced strict template-only DB queries in agent path.
- 2025-08-21 10:59:40Z — Added FSM-governed runner behind `AGENT_FSM_STRICT=1` in `UnifiedAgentExecutor` to avoid oscillations without disrupting defaults.
- 2025-08-21 11:03:53Z — Enabled FSM runner by default (set `AGENT_FSM_STRICT=0` to disable). Added strict traditional DB template mode (`AGENT_DB_TEMPLATES_ONLY=1` default) with heuristic mapping; blocks free-form LLM Cypher.
- 2025-08-21 11:09:41Z — Added tests for FSM transitions and template mapping helper; kept tests fast and dependency-light.
- 2025-08-21 11:18:47Z — Added strict traditional-path test with mocks; verifies template execution + synthesis; skips gracefully if package import is constrained.
- 2025-08-21 11:24:13Z — Expanded template library: proteins_by_genome, genes_on_contig, proteins_with_pfam, count_proteins_with_ko; added compile tests.
- 2025-08-21 11:29:30Z — Added router regression scaffold (`tests/regression/router_regression_set.json` + test) using Stage A detection + template mapping; extended mapper for genome/contig.
- 2025-08-21 11:35:27Z — Added default limit propagation (`AGENT_DEFAULT_DB_LIMIT`, default 100) to mapper; extended PFAM mapping; expanded router regression to 10 prompts.
- 2025-08-21 11:53:17Z — Policy-aware limits for DB templates (reads policy engine before env); injected default limit in Stage B DB path; added LIMIT compile test; added snapshot scaffold for synthesis formatting.

### BLOCKER: similarity_search by_sequence

- When: 2025-08-21 10:27:56Z
- Context: Stage B router can emit `similarity_search` with `mode=by_sequence` requiring runtime embedding.
- Failure: No runtime ESM2 embedding function wired in pipeline to produce vectors from raw sequence.
- Stack: core -> lancedb_processor.execute_similarity(mode='by_sequence') → NotImplementedError.
- Proposed Fix: Add a backend embedder wrapper (HF ESM2) with fixed model + deterministic config; assert LanceDB dim matches; expose as curated tool behind policy. Wire `execute_similarity` to call embedder for `by_sequence`.

## Open Questions

- Which tracing provider to standardize on (Langfuse vs LangSmith)?
- Final module boundaries/names for router/FSM (`agent/router/two_stage.py`, `agent/fsm/action_graph.py`)?
- Scope seeding: where `GenomeScope` is set (front door vs Stage A)?
- Minimum template set for initial pass; which queries first?

- 2025-08-21 12:10:00Z — Threaded initial GenomeScope into WGR metadata; disabled free-form DB auto-query; added PFAM/pathway count templates + mapper hooks; added GDS wrapper scaffolding behind flag; documented flags in docs/AGENT_FLAGS.md.
