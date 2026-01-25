# Aegis

<div align="center">

**Verifiable Agent Infrastructure**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-303%20passed-green.svg)](tests/)

*Production infrastructure for long-running autonomous agents with verifiable execution*

[Quickstart](#quick-start) • [Features](#key-features) • [Documentation](#documentation) • [Examples](#examples)

</div>

---

## Overview

**Aegis** provides enterprise-grade infrastructure for autonomous AI agents that need to:

- 🔄 **Survive Restarts** — Durable state machine with checkpoint/replay
- ✅ **Maintain Accountability** — Verifiable commitments via GCL integration
- 🛡️ **Operate Safely** — Constitutional AI policies and guardrails
- 📊 **Provide Transparency** — Complete audit trails with causal tracing
- 🤖 **Coordinate** — Multi-agent protocols (Contract Net, Auctions)

## Why Aegis?

In Greek mythology, the *aegis* was Zeus's shield—a symbol of protection and authority. Aegis brings these qualities to autonomous agents:

| Challenge | Aegis Solution |
|-----------|----------------|
| Agents lose state on restart | Durable state machine with automatic checkpointing |
| No way to verify commitments | GCL-based formal commitment verification |
| Insufficient safety guardrails | Constitutional AI principles + policy engine |
| Incomplete audit trails | Structured logging with distributed tracing |
| Agents work in isolation | Multi-agent coordination protocols |

## Quick Start

```bash
# Clone and install
git clone https://github.com/your-org/aegis.git
cd aegis
pip install -e ".[dev]"

# Start the API server
python -m uvicorn aegis.api.server:app --host 127.0.0.1 --port 8000

# Run the demo
python examples/basic_agent.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                            AEGIS                                 │
│                  Verifiable Agent Infrastructure                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Web UI    │  │   REST API  │  │     CLI     │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         └────────────────┴────────────────┘                     │
│                          │                                       │
│  ┌───────────────────────┴───────────────────────┐              │
│  │              Agent Runtime Core                │              │
│  │  • Durable State Machine                       │              │
│  │  • Checkpoint Manager                          │              │
│  │  • Replay Engine                               │              │
│  └───────────────────────┬───────────────────────┘              │
│                          │                                       │
│  ┌───────────────────────┴───────────────────────┐              │
│  │              Tool Gateway Layer                │              │
│  │  • Policy Engine    • Auth Delegator          │              │
│  │  • Rate Limiter     • Result Cache            │              │
│  └───────────────────────┬───────────────────────┘              │
│                          │                                       │
│  ┌───────────────────────┴───────────────────────┐              │
│  │           Commitment & Verification            │              │
│  │  • GCL Bridge       • Violation Detector      │              │
│  │  • Recovery Orchestrator                       │              │
│  └───────────────────────┬───────────────────────┘              │
│                          │                                       │
│  ┌───────────────────────┴───────────────────────┐              │
│  │              Audit & Tracing                   │              │
│  │  • Structured Logging  • Distributed Tracing  │              │
│  │  • Causal Graph Viewer                         │              │
│  └───────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### 🔄 Durable State Machine

Agents maintain state across restarts with automatic checkpointing:

```python
from aegis.state_machine import StateMachine

machine = StateMachine()
await machine.initialize(agent_id="agent-1")

# Process events with full durability
state = await machine.process_event(event)

# Checkpoint at any point
checkpoint_id = await machine.checkpoint()

# Restore from any checkpoint
await machine.restore(checkpoint_id)

# Branch for alternative execution paths
new_session = await machine.branch(checkpoint_id)
```

### 🛡️ Tool Gateway with Policy Enforcement

Secure tool execution with fine-grained policies:

```python
from aegis.tools import ToolGateway, PolicyRule, PolicyAction

gateway = ToolGateway()

# Define security policies
gateway.policy_engine.add_rule(PolicyRule(
    name="no_file_delete",
    action=PolicyAction.DENY,
    tool_pattern="file_*",
    argument_conditions={"operation": "delete"},
))

# Execute with policy enforcement
result = await gateway.invoke(
    tool_name="file_write",
    arguments={"path": "output.txt", "content": "Hello"},
    agent_id="agent-1",
)
```

### ✅ Verifiable Commitments (GCL)

Formal commitment tracking with verification:

```python
from aegis.commitments import CommitmentManager
from datetime import datetime, timezone, timedelta

manager = CommitmentManager()

# Create commitment
commitment = manager.create_commitment(
    debtor="agent-1",
    creditor="user",
    action="deliver_report",
    condition="data_collected",
    deadline=datetime.now(timezone.utc) + timedelta(hours=2),
)

# Verify against conditions
result = manager.verify_commitment(
    commitment.id,
    context={"data_collected": True, "report_ready": True},
)
```

### 🤖 Multi-Agent Coordination

Built-in protocols for agent collaboration:

```python
from aegis.multiagent import ContractNetProtocol, Task

cnp = ContractNetProtocol()

# Announce task for bidding
task = Task(
    requester_id="manager",
    description="Analyze dataset",
    required_capability="data_analysis",
)

# Collect bids and award
winner, commitment = await cnp.announce_and_award(task)
```

### 🛡️ Constitutional AI Policies

Principle-based evaluation inspired by Anthropic's approach:

```python
from aegis.constitutional import PolicyEvaluator, CONSTITUTIONAL_PRINCIPLES

evaluator = PolicyEvaluator(principles=CONSTITUTIONAL_PRINCIPLES)

# Evaluate action against principles
result = evaluator.evaluate_action(
    action="send_email",
    context={"recipient": "user@example.com", "content": "..."},
)

if result.decision == PolicyDecision.DENY:
    print(f"Blocked: {result.violations}")
```

### 🔧 Automatic Violation Recovery

Detect and recover from commitment violations:

```python
from aegis.recovery import RecoveryOrchestrator

orchestrator = RecoveryOrchestrator()

# Register commitment for monitoring
orchestrator.register_commitment(commitment)

# Automatic recovery on violation
result = await orchestrator.recover(violation, commitment)
print(f"Recovery: {result.strategy_used} -> {result.status}")
```

## Project Structure

```
aegis/
├── src/aegis/
│   ├── core/           # State, checkpoint, replay
│   ├── state_machine/  # Durable state machine
│   ├── tools/          # Tool gateway, policies, auth
│   ├── commitments/    # GCL integration
│   ├── gcl/            # GCL library bridge
│   ├── multiagent/     # Multi-agent coordination
│   ├── constitutional/ # Constitutional AI policies
│   ├── recovery/       # Violation detection & recovery
│   ├── audit/          # Logging and tracing
│   ├── llm/            # LLM client abstraction
│   ├── api/            # REST API (FastAPI)
│   └── cli/            # Command-line interface
├── tests/              # 303 passing tests
├── examples/           # Demo applications
├── docs/               # Documentation
└── web/                # React web UI
```

## Documentation

| Document | Description |
|----------|-------------|
| [Quickstart](docs/quickstart.md) | Get running in 5 minutes |
| [Architecture](docs/architecture.md) | System design and components |
| [API Reference](docs/api-reference.md) | REST API documentation |
| [GCL Integration](docs/gcl-integration.md) | Commitment verification guide |
| [Portfolio Report](docs/portfolio-report.md) | Comprehensive project overview |

## Examples

```bash
# Basic agent demo
python examples/basic_agent.py

# Tool usage demo
python examples/tool_usage.py

# Checkpoint and replay demo
python examples/checkpoint_demo.py

# Multi-agent coordination demo
python examples/multiagent_demo.py

# Violation recovery demo
python examples/violation_recovery_demo.py
```

## Testing

```bash
# Run all tests (303 tests)
pytest

# Run with coverage
pytest --cov=aegis --cov-report=html

# Run specific module tests
pytest tests/unit/test_state_machine.py -v
```

## API Server

```bash
# Start development server
python -m uvicorn aegis.api.server:app --reload

# Production with multiple workers
python -m uvicorn aegis.api.server:app --workers 4

# Access API docs
open http://localhost:8000/docs
```

## Configuration

Environment variables:

```bash
# LLM Configuration
export ANTHROPIC_API_KEY=your-api-key
export DEFAULT_MODEL=claude-sonnet-4-20250514

# Storage
export CHECKPOINT_DIR=./checkpoints
export AUDIT_LOG_DIR=./logs

# API
export API_HOST=0.0.0.0
export API_PORT=8000
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Data Validation | Pydantic |
| API Framework | FastAPI |
| CLI Framework | Click |
| Web UI | React + TypeScript |
| Testing | pytest + hypothesis |

## Design Principles

1. **Durability First** — All state survives restarts
2. **Verifiable Execution** — Every action is auditable
3. **Policy Enforcement** — Security at every layer
4. **Graceful Recovery** — Handle failures automatically
5. **Extensibility** — Easy to add tools, policies, strategies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by Anthropic's work on Constitutional AI
- Built on the Grounded Commitment Learning (GCL) framework
- Designed for production agent deployments

---

<div align="center">

**Aegis** — *Verifiable Agent Infrastructure*

Built with ❤️ for the future of autonomous agents

</div>
