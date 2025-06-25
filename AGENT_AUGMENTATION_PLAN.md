<?xml version="1.0" encoding="UTF-8"?>
<implementationGuide title="Implementation Guide: Agentic RAG v2.0">
    <overview>
        Implement a next-generation agentic system for the genomic RAG pipeline. This system will feature a deliberative, DAG-based planner, advanced research and code execution tools, and a security-hardened architecture.
    </overview>
    <gitWorkflow>
        <command description="Create a feature branch for the new agentic system">git checkout -b feature/agentic_rag_v2</command>
        <command description="Push the new branch to origin">git push -u origin feature/agentic_rag_v2</command>
    </gitWorkflow>

    <phase number="1" title="Core Infrastructure - Agentic Task Graph">
        <description>This phase lays the foundation by upgrading the task graph to a DAG and defining the new agentic components.</description>
        <file path="src/llm/task_graph.py" purpose="Defines the core data structures for the Directed Acyclic Graph (DAG).">
            <code><![CDATA[
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import uuid

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # For tasks whose dependencies have failed

class TaskType(Enum):
    ATOMIC_QUERY = "atomic_query" # Query against Neo4j or LanceDB
    TOOL_CALL = "tool_call"        # A call to an external tool
    AGGREGATE = "aggregate"      # A task to synthesize results

@dataclass
class Task:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: TaskType = TaskType.ATOMIC_QUERY
    query: Optional[str] = None
    dependencies: Set[str] = field(default_factory=set)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    
    # New fields for agentic capabilities
    retry_count: int = 0
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class TaskGraph:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        
    def add_task(self, task: Task) -> str:
        self.tasks[task.task_id] = task
        return task.task_id
    
    def get_ready_tasks(self) -> List[Task]:
        ready = []
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                deps_satisfied = all(
                    self.tasks.get(dep_id, Task(status=TaskStatus.FAILED)).status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                )
                if deps_satisfied:
                    ready.append(task)
        return ready
]]></code>
        </file>
        <file path="src/llm/tool_suite.py" purpose="Defines the agent's advanced tools.">
            <code><![CDATA[
# In src/llm/tool_suite.py
from .third_party.unpaywall import Unpywall # Assuming a helper module
from Bio import Entrez
import requests

def literature_search(query: str, email: str) -> str:
    """
    Searches PubMed for biomedical literature and attempts to find free full-text PDFs.
    """
    print(f"Executing literature search for: {query}")
    Entrez.email = email
    
    # (Full implementation using Entrez to search and Unpaywall to find PDFs)
    # This is a placeholder for the actual implementation.
    
    # Returns a formatted string of abstracts and links for the LLM
    return f"Formatted search results for query: '{query}'"

def code_interpreter(session_id: str, code: str) -> Dict[str, Any]:
    """
    Sends code to the external, sandboxed Code Interpreter service for execution.
    Manages state via the session_id.
    """
    service_url = "http://localhost:8000/execute" # Local service endpoint
    
    try:
        response = requests.post(
            service_url,
            json={"session_id": session_id, "code": code},
            timeout=60 # Add a timeout for the request
        )
        response.raise_for_status()
        return response.json() # Should contain stdout, stderr, and result files
    except requests.RequestException as e:
        return {"error": f"Failed to connect to Code Interpreter service: {e}"}

# Tool manifest for the agent
AVAILABLE_TOOLS = {
    "literature_search": literature_search,
    "code_interpreter": code_interpreter,
}
]]></code>
        </file>
    </phase>

    <phase number="2" title="Agentic Planner &amp; Specialized Agents">
        <description>This phase implements the 'brains' of the operation.</description>
        <file path="src/llm/dsp_sig.py" purpose="The PlannerAgent's DSPy signature.">
            <code><![CDATA[
import dspy

class PlannerAgentSignature(dspy.Signature):
    """
    You are a master bioinformatics research assistant.
    Given a user's query, decompose it into a detailed, multi-step Directed Acyclic Graph (DAG) of tasks.
    The goal is to produce a plan that can be executed to fully answer the user's question.

    You have access to the following tools:
    - "atomic_query": Execute a Cypher or vector search query against the local knowledge graph. Use for known facts.
    - "literature_search": Search PubMed for published scientific literature. Use for finding evidence or background info.
    - "code_interpreter": Write and execute Python code for calculations, data analysis, or visualization.

    Generate a valid JSON object representing the task graph.
    - Each task must have a unique "id".
    - Use the "dependencies" field to define the execution order.
    - For tool calls, specify "tool_name" and "tool_args".
    """
    
    user_query = dspy.InputField(desc="A complex user query.")
    task_graph_json = dspy.OutputField(desc="A JSON string representing the complete task graph.")
]]></code>
        </file>
        <component name="Specialized Agents">
            <agent name="TaskRepairAgent" description="Given a failed task (query + error message), its job is to provide a corrected query." />
            <agent name="KnowledgeGapAgent" description="Given a query that returns no results, its job is to formulate a TOOL_CALL task (like a literature_search or code_interpreter call) to find the missing information." />
        </component>
    </phase>

    <phase number="3" title="Secure Code Interpreter Service (TaaS)">
        <description>A separate project: a standalone FastAPI service that runs our code interpreter securely.</description>
        <status>✅ COMPLETED - Full implementation with Docker + gVisor security</status>
        <component name="Service Architecture">
            <framework>FastAPI</framework>
            <hosting>Locally hosted for development, with a plan to migrate to a serverless cloud environment for production.</hosting>
            <stateManagement>The service will maintain stateful sessions, identified by a `session_id`, to allow for iterative data analysis where variables and imports persist across multiple tool calls.</stateManagement>
        </component>
        <component name="Gold Standard Security Model">
            <runtime tool="gVisor" detail="Containers will be executed with the `gvisor` runtime to provide strong kernel-level isolation, acting as a 'user-space kernel' to intercept syscalls." />
            <containerHardening>
                <setting name="Run as non-root user" detail="The container's Dockerfile will define and switch to an unprivileged user." />
                <setting name="Drop all capabilities" detail="Containers will be run with `--cap-drop=ALL`." />
                <setting name="No new privileges" detail="Containers will be run with `--security-opt=no-new-privileges`." />
                <setting name="Read-only filesystem" detail="The container's root filesystem will be read-only (`--read-only`), with a small, writable `/tmp` mount for output." />
            </containerHardening>
            <networkPolicy setting="Networking Disabled" detail="Networking will be disabled by default (`--net=none`) for all sandbox containers. It will only be enabled for specific tasks that require it." />
            <resourceManagement setting="Strict resource limits" detail="Containers will be run with strict limits on memory (`--memory`), CPU (`--cpus`), and a hard timeout will be enforced by the orchestrator to prevent DoS attacks." />
        </component>
    </phase>

    <phase number="4" title="Task Orchestrator">
        <description>This phase upgrades the orchestrator to handle the new agentic flow.</description>
        <file path="src/llm/task_orchestrator.py" purpose="Manages the execution of the task graph.">
            <summary>The TaskOrchestrator will be enhanced to manage the full agentic loop, including invoking specialized agents for error repair and knowledge gap filling, and calling external tools.</summary>
            <code><![CDATA[
# Conceptual changes to TaskOrchestrator._execute_task
# ...
async def _execute_task(self, task: Task) -> Any:
    """Execute single task based on type."""
    if task.task_type == TaskType.ATOMIC_QUERY:
        # Existing logic for Neo4j/LanceDB queries
        # If result is empty, this is a knowledge gap.
        # The main loop will detect this and trigger the KnowledgeGapAgent.
    
    elif task.task_type == TaskType.TOOL_CALL:
        from .tool_suite import AVAILABLE_TOOLS
        tool_function = AVAILABLE_TOOLS.get(task.tool_name)
        if not tool_function:
            raise ValueError(f"Unknown tool: {task.tool_name}")
        
        # Execute the tool with its arguments
        return tool_function(**task.tool_args)

# The main execution loop in the orchestrator will be expanded to include:
# - A try/except block around _execute_task.
# - On failure, invoke the TaskRepairAgent and retry a limited number of times.
# - If a query returns no results, invoke the KnowledgeGapAgent to generate a new TOOL_CALL task and add it to the graph.
]]></code>
        </file>
    </phase>

    <phase number="5" title="Testing &amp; Integration">
        <test type="Unit">
            <item>Update `src/tests/test_task_graph.py` to test the new `Task` fields and DAG structures.</item>
            <item>Create `src/tests/test_tool_suite.py` to test the new tools (using mocks for external APIs).</item>
            <item>Create unit tests for the security features of the `CodeInterpreterService`.</item>
        </test>
        <test type="Integration">
            <item>Create `src/tests/test_agentic_flow.py` with tests for the full loops.</item>
            <subitem>Test that a failed Cypher query invokes the `TaskRepairAgent`.</subitem>
            <subitem>Test that a query with no results invokes the `KnowledgeGapAgent` and adds a `literature_search` task.</subitem>
        </test>
    </phase>

    <phase number="6" title="Sequence Database Integration">
        <description>🚧 IN PROGRESS - Lightweight sequence access for advanced protein analysis</description>
        <rationale>
            Code interpreter testing revealed a critical gap: protein sequences are not available in Neo4j.
            Advanced analyses (amino acid composition, hydrophobicity profiles, motif searches) require 
            sequence data but adding ~3MB of sequences to Neo4j would bloat the graph database.
        </rationale>
        <component name="SQLite Sequence Database">
            <file path="src/build_kg/sequence_db.py" purpose="Database schema and operations">
                <schema>
                    CREATE TABLE sequences (
                        protein_id TEXT PRIMARY KEY,
                        sequence TEXT NOT NULL,
                        length INTEGER NOT NULL, 
                        genome_id TEXT NOT NULL,
                        source_file TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                </schema>
            </file>
            <file path="src/build_kg/sequence_db_builder.py" purpose="Build database from prodigal FASTA files">
                <input>data/stage03_prodigal/*.faa files</input>
                <output>data/sequences.db (~10MB SQLite database)</output>
                <performance>Indexed lookups &lt;1ms per protein</performance>
            </file>
        </component>
        <component name="Code Interpreter Integration">
            <file path="src/code_interpreter/sequence_service.py" purpose="Sequence lookup service">
                <method name="get_sequences" args="protein_ids: List[str]" returns="Dict[str, str]" />
                <method name="get_sequences_by_genome" args="genome_id: str" returns="Dict[str, str]" />
                <method name="search_by_pattern" args="pattern: str" returns="List[Tuple[str, str]]" />
            </file>
            <enhancement path="src/code_interpreter/client.py">
                <method name="fetch_protein_sequences" description="Auto-fetch sequences when needed for analysis" />
                <method name="analyze_amino_acid_composition" description="Enhanced with automatic sequence retrieval" />
                <method name="calculate_hydrophobicity_profiles" description="Kyte-Doolittle scale calculations" />
            </enhancement>
        </component>
        <workflow>
            <step>1. Neo4j query returns protein IDs</step>
            <step>2. Code interpreter automatically fetches sequences from SQLite</step>
            <step>3. Advanced analysis (composition, hydrophobicity, motifs)</step>
            <step>4. Statistical testing and visualization</step>
            <step>5. Results synthesis with biological context</step>
        </workflow>
        <benefits>
            <item>Lean Neo4j database (metadata only)</item>
            <item>Fast sequence access (&lt;1ms lookups)</item>
            <item>Enables advanced protein analysis</item>
            <item>Scalable to 100K+ proteins</item>
            <item>Transparent integration (auto-fetch sequences)</item>
        </benefits>
    </phase>

    <phase number="7" title="Future Work">
        <description>High-impact features planned for future iterations.</description>
        <feature name="Symbiotic KG-Researcher Loop" description="An autonomous loop where the agent uses the literature search tool to fill gaps in its own knowledge graph by finding papers, extracting structured facts, and loading them into Neo4j." />
        <feature name="User-in-the-Loop Tool" description="A tool that allows the agent to pause its execution and ask the user for clarification when it encounters ambiguity." />
        <feature name="Multi-modal Integration" description="Image analysis, document processing, and other data modalities" />
        <feature name="Distributed Computing" description="Scale code execution across multiple containers/nodes" />
    </body>
</implementationGuide>