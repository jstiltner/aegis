# GCL Integration Guide

This document explains how the Autonomous Agent Infrastructure integrates with the Grounded Commitment Learning (GCL) framework for verifiable agent commitments.

## Overview

GCL provides a formal framework for agent commitments based on the 5-tuple model:

```
C = (debtor, creditor, action, condition, deadline)
```

Where:
- **debtor**: The agent making the commitment
- **creditor**: The agent receiving the commitment
- **action**: What the debtor commits to do
- **condition**: When the commitment becomes active
- **deadline**: When the commitment must be fulfilled

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Runtime                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Commitment Manager                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │   Create    │  │   Verify    │  │   Fulfill   │     │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │   │
│  │         │                │                │             │   │
│  │         └────────────────┼────────────────┘             │   │
│  │                          │                              │   │
│  │                    ┌─────┴─────┐                        │   │
│  │                    │GCL Bridge │                        │   │
│  │                    └─────┬─────┘                        │   │
│  └──────────────────────────┼──────────────────────────────┘   │
│                             │                                   │
└─────────────────────────────┼───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GCL Library                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ GroundedCommit  │  │ VerificationEng │  │ CommitPortfolio │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Runtime Commitments

### Creating Commitments

```python
from aegis.commitments import RuntimeCommitment, CommitmentStatus
from datetime import datetime, timezone, timedelta

# Create a commitment
commitment = RuntimeCommitment(
    debtor="agent-123",
    creditor="user",
    action="complete_research_task",
    condition="research_data_available",
    deadline=datetime.now(timezone.utc) + timedelta(hours=2),
)

print(f"Commitment ID: {commitment.id}")
print(f"Status: {commitment.status}")  # CommitmentStatus.CREATED
```

### Commitment Lifecycle

```python
from aegis.commitments import CommitmentManager

manager = CommitmentManager()

# Create and register
commitment = manager.create_commitment(
    debtor="agent-123",
    creditor="user",
    action="deliver_report",
    condition="data_collected",
    deadline=datetime.now(timezone.utc) + timedelta(hours=1),
)

# Activate when condition is met
manager.activate_commitment(commitment.id)
print(commitment.status)  # CommitmentStatus.ACTIVE

# Verify progress
result = manager.verify_commitment(
    commitment.id,
    context={"data_collected": True, "report_ready": False},
)
print(result.status)  # VerificationStatus.IN_PROGRESS

# Fulfill when complete
manager.fulfill_commitment(commitment.id)
print(commitment.status)  # CommitmentStatus.FULFILLED
```

### Tool Commitments

Commitments can be automatically created for tool executions:

```python
from aegis.commitments import ToolCommitment

# Create a tool commitment
tool_commitment = ToolCommitment(
    tool_name="web_search",
    expected_result="search_results",
    timeout=30.0,
    agent_id="agent-123",
)

# Convert to runtime commitment
runtime_commitment = tool_commitment.to_runtime_commitment()
```

## GCL Bridge

The GCL Bridge provides bidirectional conversion between runtime commitments and GCL's GroundedCommitment format.

### Basic Usage

```python
from aegis.gcl import GCLBridge

bridge = GCLBridge()

# Convert runtime commitment to GCL format
runtime_commitment = RuntimeCommitment(
    debtor="agent-123",
    creditor="user",
    action="complete_task",
)

gcl_commitment = bridge.to_gcl_commitment(runtime_commitment)

# Convert back
restored = bridge.from_gcl_commitment(gcl_commitment)
```

### Portfolio Management

```python
# Add commitment to agent's portfolio
bridge.add_to_portfolio("agent-123", runtime_commitment)

# Get all commitments for an agent
commitments = bridge.get_portfolio("agent-123")

# Verify using GCL engine
result = bridge.verify_with_gcl(
    runtime_commitment,
    context={"task_complete": True},
)
```

### Verification Integration

```python
from aegis.gcl import GCLAdapter

adapter = GCLAdapter()

# Create commitment through adapter
commitment = adapter.create_commitment(
    debtor="agent-123",
    creditor="user",
    action="process_data",
    condition="data_available",
)

# Verify with context
result = adapter.verify_commitment(
    commitment.id,
    context={"data_available": True, "processing_complete": False},
)

print(f"Verified: {result.verified}")
print(f"Status: {result.status}")
```

## Commitment Templates

Templates allow reusable commitment patterns:

```python
from aegis.commitments import CommitmentTemplate, BUILTIN_TEMPLATES

# Use built-in template
task_template = BUILTIN_TEMPLATES["task_completion"]

commitment = task_template.instantiate(
    debtor="agent-123",
    creditor="user",
    task_id="task-456",
    timeout_hours=2,
)

# Create custom template
custom_template = CommitmentTemplate(
    name="data_processing",
    action_template="process_{data_type}_data",
    condition_template="{data_type}_data_available",
    default_timeout=timedelta(hours=1),
)

commitment = custom_template.instantiate(
    debtor="agent-123",
    creditor="system",
    data_type="customer",
)
```

## Verification

### Expression Evaluator

The expression evaluator safely evaluates commitment conditions:

```python
from aegis.commitments import ExpressionEvaluator

evaluator = ExpressionEvaluator()

# Simple conditions
result = evaluator.evaluate(
    "task_complete == True",
    {"task_complete": True},
)
print(result)  # True

# Complex conditions
result = evaluator.evaluate(
    "progress >= 80 and errors == 0",
    {"progress": 85, "errors": 0},
)
print(result)  # True

# Logical operators
result = evaluator.evaluate(
    "status in ['done', 'complete'] or override",
    {"status": "done", "override": False},
)
print(result)  # True
```

### Runtime Verifier

```python
from aegis.commitments import RuntimeVerifier

verifier = RuntimeVerifier()

# Verify single commitment
result = verifier.verify(
    commitment,
    context={"condition_met": True},
)

print(f"Status: {result.status}")
print(f"Message: {result.message}")

# Batch verification
results = verifier.verify_batch(
    commitments,
    context={"global_condition": True},
)

for commitment_id, result in results.items():
    print(f"{commitment_id}: {result.status}")
```

### Custom Verification Hooks

```python
from aegis.commitments import VerificationHook

# Define custom hook
def my_verification_hook(commitment, context, result):
    # Custom verification logic
    if commitment.action.startswith("sensitive_"):
        # Additional checks for sensitive actions
        if not context.get("admin_approved"):
            result.status = VerificationStatus.FAILED
            result.message = "Admin approval required"
    return result

# Register hook
verifier.add_hook(my_verification_hook)
```

## Violation Detection

### Detecting Violations

```python
from aegis.recovery import ViolationDetector, ViolationType

detector = ViolationDetector()

# Check commitment for violations
violations = detector.check_commitment(commitment, context={})

for violation in violations:
    print(f"Type: {violation.type}")
    print(f"Severity: {violation.severity}")
    print(f"Description: {violation.description}")
```

### Violation Types

| Type | Description | Severity |
|------|-------------|----------|
| `DEADLINE_EXCEEDED` | Commitment deadline passed | 0.8 |
| `CONDITION_VIOLATED` | Required condition no longer holds | 0.7 |
| `POLICY_VIOLATED` | Action violates policy | 0.9 |
| `UNAUTHORIZED_ACTION` | Unauthorized action attempted | 0.95 |
| `RESOURCE_EXCEEDED` | Resource limit exceeded | 0.6 |
| `BEHAVIORAL_ANOMALY` | Unusual behavior detected | 0.5 |
| `COMMITMENT_ABANDONED` | No activity on commitment | 0.5 |

### Custom Condition Evaluators

```python
# Register custom condition evaluator
def my_condition_evaluator(condition: str, context: dict) -> bool:
    # Custom evaluation logic
    if condition.startswith("custom:"):
        custom_condition = condition[7:]
        return evaluate_custom(custom_condition, context)
    return True

detector.register_condition_evaluator("custom", my_condition_evaluator)
```

## Recovery Strategies

### Automatic Recovery

```python
from aegis.recovery import RecoveryOrchestrator

orchestrator = RecoveryOrchestrator()

# Register commitment for monitoring
orchestrator.register_commitment(commitment)

# Attempt recovery from violation
result = await orchestrator.recover(violation, commitment)

print(f"Status: {result.status}")
print(f"Strategy: {result.strategy_used}")
```

### Available Strategies

| Strategy | Use Case | Priority |
|----------|----------|----------|
| `retry` | Transient failures | Low severity |
| `renegotiate` | Deadline issues | Medium severity |
| `compensate` | Condition violations | Medium severity |
| `escalate` | Policy violations | High severity |

### Custom Recovery Strategy

```python
from aegis.recovery import RecoveryStrategy, RecoveryPlan

class MyRecoveryStrategy(RecoveryStrategy):
    @property
    def name(self) -> str:
        return "my_strategy"
    
    @property
    def supported_violations(self) -> list[ViolationType]:
        return [ViolationType.CONDITION_VIOLATED]
    
    def can_handle(self, violation, commitment) -> bool:
        return violation.severity < 0.8
    
    def create_plan(self, violation, commitment, context=None) -> RecoveryPlan:
        return RecoveryPlan(
            violation_id=violation.id,
            action=RecoveryAction.COMPENSATE,
            parameters={"custom_param": "value"},
        )
    
    async def execute(self, plan, commitment) -> dict:
        # Custom recovery logic
        return {"success": True, "action": "custom_recovery"}

# Register strategy
orchestrator.selector.register(MyRecoveryStrategy())
```

## Multi-Agent Commitments

### Commitment Protocol

```python
from aegis.multiagent import CommitmentProtocol

protocol = CommitmentProtocol()

# Agent A proposes commitment to Agent B
commitment = await protocol.propose(
    debtor_id="agent-b",
    creditor_id="agent-a",
    action="process_data",
    deadline=datetime.now(timezone.utc) + timedelta(hours=1),
)

# Agent B accepts
await protocol.accept(commitment.id, "agent-b")

# Agent B completes
await protocol.complete(commitment.id, "agent-b", result={"data": "processed"})
```

### Contract Net Protocol

```python
from aegis.multiagent import ContractNetProtocol, Task

cnp = ContractNetProtocol()

# Manager announces task
task = Task(
    requester_id="manager",
    description="Analyze customer data",
    required_capability="data_analysis",
    max_price=100.0,
)

# Collect bids and award
winner, commitment = await cnp.announce_and_award(task, bid_timeout=10.0)

if winner:
    print(f"Task awarded to: {winner.bidder_id}")
    print(f"Price: {winner.price}")
```

## Best Practices

### 1. Always Set Deadlines

```python
# Good: Explicit deadline
commitment = RuntimeCommitment(
    debtor="agent",
    creditor="user",
    action="task",
    deadline=datetime.now(timezone.utc) + timedelta(hours=2),
)

# Avoid: No deadline (may run indefinitely)
commitment = RuntimeCommitment(
    debtor="agent",
    creditor="user",
    action="task",
)
```

### 2. Use Specific Conditions

```python
# Good: Specific, verifiable condition
commitment = RuntimeCommitment(
    action="generate_report",
    condition="data_loaded and data_validated and user_authorized",
)

# Avoid: Vague condition
commitment = RuntimeCommitment(
    action="generate_report",
    condition="ready",
)
```

### 3. Monitor Active Commitments

```python
# Start monitoring
await orchestrator.start_monitoring(interval=60.0)

# Check for overdue commitments
overdue = protocol.get_overdue_commitments()
for c in overdue:
    print(f"Overdue: {c.id} - {c.action}")
```

### 4. Handle Violations Gracefully

```python
async def on_violation(violation):
    # Log violation
    logger.warning(f"Violation detected: {violation.type}")
    
    # Attempt recovery
    result = await orchestrator.recover(violation, commitment)
    
    if result.status == RecoveryStatus.FAILED:
        # Escalate to human
        await notify_operator(violation)

detector.add_handler(on_violation)
```

## Configuration

### Commitment Manager Settings

```python
from aegis.commitments import CommitmentPolicy

policy = CommitmentPolicy(
    max_active_commitments=10,
    default_timeout=timedelta(hours=24),
    require_deadline=True,
    auto_cancel_on_violation=False,
)

manager = CommitmentManager(policy=policy)
```

### GCL Bridge Settings

```python
bridge = GCLBridge(
    auto_sync=True,  # Sync with GCL library automatically
    verification_timeout=30.0,  # Verification timeout in seconds
)
```

## Troubleshooting

### Common Issues

**Commitment stuck in CREATED state:**
- Ensure condition is being evaluated
- Check that `activate_commitment()` is called when condition is met

**Verification always fails:**
- Check condition expression syntax
- Verify context contains required variables
- Enable debug logging for expression evaluator

**GCL library not found:**
- Install with: `pip install "autonomous-agent[gcl]"`
- Or use standalone mode without GCL integration

### Debug Logging

```python
import logging

# Enable debug logging for commitments
logging.getLogger("aegis.commitments").setLevel(logging.DEBUG)
logging.getLogger("aegis.gcl").setLevel(logging.DEBUG)
```
