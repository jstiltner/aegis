# Aegis

Aegis provides durable state management, verifiable commitments, and policy enforcement for autonomous AI agents. The core insight is that agent reliability requires treating commitments as first-class objects with explicit failure modes and recovery strategies—an approach grounded in the GCL (Grounded Commitment Learning) framework.

---

## The Problem

Autonomous agents face a fundamental tension: they need to operate independently while remaining accountable. Current solutions address pieces of this problem but miss the integration:

**Workflow engines (Temporal, Prefect)** handle durable execution well. State survives restarts, tasks can be retried, and there's a clear execution history. But they lack agent-specific abstractions. There's no concept of a commitment an agent makes to a user, no policy evaluation before tool invocation, no multi-agent coordination primitives. You'd need to build all of that on top.

**Agent frameworks (LangChain, AutoGPT)** provide agent abstractions—tool calling, memory, planning. But state is ephemeral. Restart the process and you lose everything. There's no checkpoint/replay, no formal verification of what the agent promised to do, no structured recovery when things go wrong.

**The gap**: Neither approach treats agent commitments as first-class objects. When an agent says "I will complete this task by 5pm," that's just a string in a conversation. There's no mechanism to verify the commitment was fulfilled, detect when it's violated, or recover gracefully.

Aegis fills this gap by combining durable execution (from workflow engines) with agent abstractions (from agent frameworks) with verifiable commitments (from GCL) with policy enforcement (from constitutional AI).

---

## Design Decisions

### Why Event Sourcing for Agent State

Agent state is mutable by nature—conversations grow, tool calls resolve, memory updates. The naive approach is to mutate state in place and periodically snapshot. This works until you need to debug why an agent made a particular decision, or replay execution from an earlier point, or branch to explore an alternative path.

Event sourcing inverts this: state is derived from an append-only log of events. Each user message, LLM response, tool invocation, and commitment update is an event. The current state is computed by replaying events from the last checkpoint.

```python
# State is immutable — all "mutations" return new instances
class AgentState(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    def with_message(self, message: Message) -> AgentState:
        """Create a new state with an additional message."""
        return self.model_copy(
            update={
                "conversation_history": (*self.conversation_history, message),
                "version": self.version + 1,
                "updated_at": datetime.now(timezone.utc),
            }
        )
```

The trade-off: storage grows with event count, and replay adds latency on restore. Aegis mitigates this with configurable checkpoint intervals—checkpoint every N transitions or every M seconds, whichever comes first. Restore replays only events since the last checkpoint.

### Commitments as First-Class Objects

Most agent systems treat commitments implicitly. The agent says "I'll search for that" and then either does or doesn't. There's no structured representation, no verification, no recovery path.

Aegis models commitments explicitly using the GCL 5-tuple: `(debtor, creditor, action, condition, deadline)`. This isn't just documentation—it enables:

1. **Verification**: Check if the commitment's success condition holds
2. **Violation detection**: Identify when deadlines pass or conditions fail
3. **Recovery**: Select and execute appropriate recovery strategies
4. **Audit**: Track commitment lifecycle for accountability

```python
# Commitments have explicit structure and lifecycle
class RuntimeCommitment(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    debtor: str      # Who made the commitment
    creditor: str    # Who receives the commitment
    action: str      # What was committed
    condition: str   # Success condition (evaluable expression)
    deadline: datetime | None
    status: CommitmentStatus  # CREATED → ACTIVE → FULFILLED/VIOLATED/CANCELLED
```

The condition field contains an evaluable expression, not just a description. `"task_complete AND error_count == 0"` can be verified against runtime context.

### Policy Enforcement at the Gateway

Where should policy checks happen? Options:

1. **At planning time**: Check before the agent decides to use a tool
2. **At invocation time**: Check when the tool is actually called
3. **Both**: Defense in depth

Aegis enforces at invocation time (option 2), with the tool gateway as the enforcement point. Rationale:

- **Principle of least authority**: The gateway sees the actual arguments, not the agent's intent. An agent might plan to "read a file" but the actual path could be `/etc/passwd`.
- **Single enforcement point**: All tool calls flow through the gateway. No way to bypass.
- **Separation of concerns**: The agent focuses on planning; the gateway focuses on safety.

```python
# Policy rules match on tool name, arguments, and context
class PolicyRule(BaseModel):
    name: str
    action: PolicyAction  # ALLOW, DENY, REQUIRE_APPROVAL
    tool_pattern: str     # Glob pattern: "file_*", "web_search"
    argument_conditions: dict[str, Any]  # {"path": {"not_contains": "/etc"}}
    context_conditions: dict[str, Any]   # {"user_role": "admin"}
```

The trade-off: enforcement at invocation can't prevent the agent from wasting tokens planning a disallowed action. In practice, this is acceptable—the cost of a rejected tool call is low compared to the security benefit.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                   AEGIS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         Interface Layer                                 │ │
│  │     REST API (FastAPI)  ·  CLI (Click)  ·  WebSocket  ·  Web UI        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      Agent Runtime Core                                 │ │
│  │                                                                         │ │
│  │   StateMachine ──────► CheckpointManager ──────► ReplayEngine          │ │
│  │        │                      │                       │                 │ │
│  │   Event-driven           Auto-save              Deterministic          │ │
│  │   transitions            + branching            reconstruction         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         Tool Gateway                                    │ │
│  │                                                                         │ │
│  │   PolicyEngine ─► AuthDelegator ─► RateLimiter ─► ToolRegistry         │ │
│  │        │                                                                │ │
│  │   Rule matching         OAuth/API keys        Per-agent limits         │ │
│  │   + deny/allow          delegation            + caching                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    Commitment & Verification                            │ │
│  │                                                                         │ │
│  │   CommitmentManager ──► ViolationDetector ──► RecoveryOrchestrator     │ │
│  │        │                      │                       │                 │ │
│  │   GCL 5-tuple            Deadline/condition      Strategy selection    │ │
│  │   lifecycle              monitoring              + execution           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         Audit & Tracing                                 │ │
│  │                                                                         │ │
│  │   AuditLogger ──────► TraceContext ──────► Storage (SQLite/Postgres)   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

Data flows top-to-bottom for requests, bottom-to-top for events. The state machine is the central coordinator—it receives events, applies transitions, triggers checkpoints, and notifies listeners.

---

## Implementation Highlights

### 1. Immutable State with Functional Updates

The hardest bug to debug in a stateful system is unexpected mutation. Aegis prevents this at the type level: all state objects use Pydantic's `frozen=True`, making them immutable after construction.

```python
class AgentState(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    conversation_history: tuple[Message, ...]  # tuple, not list
    active_commitments: tuple[str, ...]
    pending_tool_calls: tuple[ToolCall, ...]
```

Note the use of `tuple` instead of `list`. Lists are mutable; tuples are not. This prevents accidental `state.conversation_history.append(msg)`—it would raise an error.

State updates return new instances:

```python
def with_message(self, message: Message) -> AgentState:
    return self.model_copy(
        update={
            "conversation_history": (*self.conversation_history, message),
            "version": self.version + 1,
        }
    )
```

The `model_copy` creates a shallow copy with specified fields replaced. The spread operator `(*self.conversation_history, message)` creates a new tuple with the message appended.

This pattern has a cost: memory allocation for each state change. In practice, agent state is small (KB, not MB) and changes are infrequent (seconds, not milliseconds), so the overhead is negligible.

### 2. Recovery Strategy Selection

When a commitment is violated, what should happen? The answer depends on the violation type, severity, and available recovery options. Aegis uses a strategy pattern with priority-based selection.

```python
class StrategySelector:
    def select(self, violation: Violation, commitment: RuntimeCommitment) -> RecoveryStrategy | None:
        candidates = [s for s in self._strategies if s.can_handle(violation, commitment)]
        
        if not candidates:
            return None
        
        def priority(s: RecoveryStrategy) -> int:
            # High severity → escalate
            if violation.severity >= 0.9 and s.name == "escalate":
                return 100
            # Low severity transient → retry
            if violation.severity < 0.5 and s.name == "retry":
                return 80
            # Deadline exceeded → renegotiate
            if violation.type == ViolationType.DEADLINE_EXCEEDED and s.name == "renegotiate":
                return 70
            return 50
        
        candidates.sort(key=priority, reverse=True)
        return candidates[0]
```

The priority function encodes domain knowledge: high-severity violations should escalate to humans, low-severity transient failures should retry, deadline violations should attempt renegotiation. This is configurable—you can register custom strategies and adjust the priority logic.

Each strategy implements a common interface:

```python
class RecoveryStrategy(ABC):
    @abstractmethod
    def can_handle(self, violation: Violation, commitment: RuntimeCommitment) -> bool: ...
    
    @abstractmethod
    def create_plan(self, violation: Violation, commitment: RuntimeCommitment) -> RecoveryPlan: ...
    
    @abstractmethod
    async def execute(self, plan: RecoveryPlan, commitment: RuntimeCommitment) -> dict[str, Any]: ...
```

The `can_handle` check allows strategies to opt out based on context. For example, `RetryStrategy` refuses if `retry_count >= max_retries` or if `severity > 0.9`.

### 3. Constitutional AI Principle Evaluation

Policy enforcement goes beyond simple allow/deny rules. Aegis includes a constitutional AI evaluator that checks actions against principles like helpfulness, harmlessness, and honesty.

```python
HARMLESSNESS = Principle(
    name="harmlessness",
    description="Avoid actions that could cause harm",
    check_expression="not contains_harmful_content and not enables_harm",
    severity=1.0,  # Violations are critical
)

HONESTY = Principle(
    name="honesty", 
    description="Provide accurate information",
    check_expression="not contains_false_claims and sources_cited_when_applicable",
    severity=0.8,
)
```

The `check_expression` is evaluated against a context dictionary containing action metadata. This is intentionally simple—complex evaluations should use custom evaluator functions, not expression strings.

The evaluator returns a decision with any violations:

```python
class EvaluationResult(BaseModel):
    decision: PolicyDecision  # ALLOW, DENY, WARN, REQUIRE_REVIEW
    violations: list[PolicyViolation]
    warnings: list[PolicyViolation]
    evaluation_time_ms: float
```

This integrates with the tool gateway: before executing a tool, the gateway can run constitutional evaluation and block or flag concerning actions.

---

## GCL Integration

Aegis builds on the Grounded Commitment Learning framework. GCL provides the theoretical foundation; Aegis provides the runtime implementation.

The GCL 5-tuple maps directly to `RuntimeCommitment`:

| GCL Concept | Aegis Implementation |
|-------------|---------------------|
| Debtor | `commitment.debtor` (agent ID) |
| Creditor | `commitment.creditor` (user/system ID) |
| Action | `commitment.action` (string description) |
| Condition | `commitment.condition` (evaluable expression) |
| Deadline | `commitment.deadline` (datetime) |

The `GCLBridge` handles bidirectional conversion when the full GCL library is available:

```python
class GCLBridge:
    def to_gcl_commitment(self, runtime: RuntimeCommitment) -> GroundedCommitment:
        return GroundedCommitment(
            debtor=runtime.debtor,
            creditor=runtime.creditor,
            action=Predicate(runtime.action),
            condition=Predicate(runtime.condition),
            deadline=runtime.deadline,
        )
    
    def verify_with_gcl(self, commitment: RuntimeCommitment, context: dict) -> VerificationResult:
        gcl_commitment = self.to_gcl_commitment(commitment)
        return self.verification_engine.verify(gcl_commitment, context)
```

When GCL isn't installed, Aegis falls back to its own expression evaluator for condition checking. The evaluator handles basic comparisons, logical operators, and membership tests:

```python
# Supported expressions
"task_complete == True"
"progress >= 80 and errors == 0"
"status in ['done', 'complete']"
```

Unsafe expressions (function calls, imports, attribute access) are rejected.

---

## Validation

**Test coverage**: 303 tests covering state machine transitions, checkpoint integrity, policy evaluation edge cases, and multi-agent message ordering. Property-based tests via Hypothesis verify state serialization round-trips.

**Module breakdown**:
- `core/` — State, checkpoint, replay: 42 tests
- `state_machine/` — Transitions, validation: 28 tests
- `tools/` — Gateway, policy, auth: 48 tests
- `commitments/` — GCL integration, verification: 32 tests
- `audit/` — Logging, tracing: 38 tests
- `llm/` — Client, streaming, tool execution: 52 tests
- `api/` — REST endpoints, models: 37 tests
- `recovery/` — Violation detection, strategies: 26 tests

**What's not tested**: The constitutional AI evaluator uses simple expression matching, not actual content analysis. Real deployment would need integration with a content classifier. Multi-node distributed operation isn't implemented—this is single-node only.

---

## Limitations & Future Work

**Single-node only**: State is stored locally. Distributed coordination (multiple agents across nodes, shared state) is not implemented. This would require a distributed event log (Kafka, Redis Streams) and consensus for checkpoint coordination.

**No content-aware policy**: Constitutional AI principles check metadata, not content. Evaluating whether a response "contains harmful content" requires an external classifier.

**Synchronous recovery**: Recovery strategies execute synchronously. Long-running recoveries (waiting for human approval) block the agent. Async recovery with callbacks is future work.

**Limited benchmarking**: No systematic performance characterization. Checkpoint latency, message throughput, and policy evaluation overhead need measurement under realistic workloads.

---

## Project Structure

```
aegis/
├── src/aegis/
│   ├── core/              # State, checkpoint, replay, events
│   ├── state_machine/     # Transitions, validation, machine
│   ├── tools/             # Gateway, policy, auth, registry
│   ├── commitments/       # GCL models, manager, verifier
│   ├── gcl/               # GCL library bridge
│   ├── constitutional/    # Principles, evaluator, critique
│   ├── multiagent/        # Registry, messaging, protocols
│   ├── recovery/          # Detector, strategies, orchestrator
│   ├── audit/             # Logger, trace, storage, viewer
│   ├── llm/               # Client, providers, streaming
│   ├── api/               # FastAPI server, routes, models
│   └── cli/               # Command-line interface
├── tests/                 # 303 tests
├── examples/              # Demo applications
├── docs/                  # Documentation
└── web/                   # React UI
```

---

## Links

- **GitHub**: [github.com/yourusername/aegis](https://github.com/yourusername/aegis)
- **Documentation**: [/docs](./docs/)
- **API Reference**: [/docs/api-reference.md](./api-reference.md)
