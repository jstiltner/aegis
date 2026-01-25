# Architecture Overview

This document provides a comprehensive overview of the Autonomous Agent Infrastructure architecture, designed for production-grade long-running agents with verifiable execution.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Autonomous Agent Infrastructure                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Web UI    │  │   REST API  │  │     CLI     │  │  WebSocket  │         │
│  │  (React)    │  │  (FastAPI)  │  │   (Click)   │  │  (Real-time)│         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                │                 │
│         └────────────────┴────────────────┴────────────────┘                 │
│                                   │                                          │
│  ┌────────────────────────────────┴────────────────────────────────────┐    │
│  │                         Agent Runtime Core                           │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │    │
│  │  │ State Machine│  │  Checkpoint  │  │    Replay    │               │    │
│  │  │   (Durable)  │  │   Manager    │  │    Engine    │               │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                   │                                          │
│  ┌────────────────────────────────┴────────────────────────────────────┐    │
│  │                         Tool Gateway Layer                           │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │    │
│  │  │Policy Engine │  │ Auth Delegator│  │ Rate Limiter │               │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │    │
│  │  │ Tool Registry│  │  MCP Adapter │  │ Result Cache │               │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                   │                                          │
│  ┌────────────────────────────────┴────────────────────────────────────┐    │
│  │                      Commitment & Verification                       │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │    │
│  │  │  GCL Bridge  │  │  Commitment  │  │  Violation   │               │    │
│  │  │              │  │   Manager    │  │  Detector    │               │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                   │                                          │
│  ┌────────────────────────────────┴────────────────────────────────────┐    │
│  │                      Multi-Agent Coordination                        │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │    │
│  │  │Agent Registry│  │ Message Bus  │  │  Protocols   │               │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                   │                                          │
│  ┌────────────────────────────────┴────────────────────────────────────┐    │
│  │                         Audit & Tracing                              │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │    │
│  │  │ Audit Logger │  │Trace Context │  │ Trace Viewer │               │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Agent Runtime Core

The heart of the system, providing durable state management and execution control.

#### State Machine
- **Durable State**: All agent state is persisted and can survive restarts
- **Event-Driven**: State transitions are triggered by events (user input, LLM response, tool calls)
- **Validation**: Each transition is validated before application
- **Listeners**: External systems can subscribe to state changes

#### Checkpoint Manager
- **Automatic Checkpointing**: Based on configurable policies (time, transitions, events)
- **Manual Checkpoints**: Create checkpoints at any point
- **Branching**: Create alternative execution paths from any checkpoint
- **History**: Full checkpoint history with parent-child relationships

#### Replay Engine
- **Deterministic Replay**: Replay execution from any checkpoint
- **Event Filtering**: Replay specific event types
- **Debugging**: Step through execution for debugging

### 2. Tool Gateway

Secure, policy-enforced tool execution layer.

#### Policy Engine
- **Rule-Based**: Define allow/deny/require-approval rules
- **Context-Aware**: Rules can depend on agent, tool, arguments, time
- **Composable**: Combine multiple policies with priority ordering
- **Audit Trail**: All policy decisions are logged

#### Auth Delegator
- **Credential Management**: Secure storage of API keys and tokens
- **Per-Tool Auth**: Configure authentication per tool
- **Token Refresh**: Automatic token refresh for OAuth flows
- **Scoped Access**: Limit tool access based on agent permissions

#### Rate Limiter
- **Per-Agent Limits**: Separate rate limits per agent
- **Per-Tool Limits**: Different limits for different tools
- **Sliding Window**: Accurate rate limiting with sliding windows
- **Burst Handling**: Allow controlled bursts

### 3. Commitment System

Verifiable commitments based on Grounded Commitment Learning (GCL).

#### GCL Bridge
- **Bidirectional Conversion**: Convert between runtime and GCL commitments
- **Portfolio Management**: Track commitment portfolios per agent
- **Verification Integration**: Use GCL verification engine

#### Commitment Manager
- **Lifecycle Management**: Create, activate, fulfill, cancel commitments
- **Status Tracking**: Track commitment status through lifecycle
- **Verification**: Verify commitments against conditions
- **Templates**: Reusable commitment templates

#### Violation Detector
- **Deadline Detection**: Detect deadline violations
- **Condition Monitoring**: Monitor condition violations
- **Policy Violations**: Detect policy breaches
- **Anomaly Detection**: Detect behavioral anomalies

### 4. Multi-Agent Coordination

Infrastructure for multi-agent systems.

#### Agent Registry
- **Agent Discovery**: Find agents by capability
- **Status Tracking**: Track agent availability
- **Capability Matching**: Match tasks to capable agents

#### Message Bus
- **Priority Queues**: Messages processed by priority
- **Broadcast Support**: Send to all agents
- **Request-Response**: Correlate requests with responses

#### Coordination Protocols
- **Contract Net**: Task allocation through bidding
- **Auctions**: Resource allocation through auctions
- **Delegation**: Hierarchical task delegation

### 5. Constitutional AI

Policy evaluation based on constitutional principles.

#### Principles
- **Helpfulness**: Ensure responses address requests
- **Harmlessness**: Prevent harmful content
- **Honesty**: Ensure truthful responses
- **Transparency**: Explain reasoning
- **Privacy**: Protect user data
- **Fairness**: Avoid bias

#### Evaluator
- **Pre-Action Evaluation**: Check before executing
- **Post-Action Evaluation**: Verify after execution
- **Severity Levels**: Different handling for different severities

#### Self-Critique
- **Iterative Improvement**: Revise responses based on critique
- **Principle-Based**: Critique against constitutional principles

### 6. Audit & Tracing

Comprehensive logging and tracing for accountability.

#### Audit Logger
- **Structured Events**: Rich event metadata
- **Severity Levels**: Filter by importance
- **Buffered Writing**: Efficient batch writes

#### Trace Context
- **Distributed Tracing**: Track across components
- **Span Hierarchy**: Parent-child span relationships
- **Timing**: Accurate duration tracking

#### Trace Viewer
- **Timeline View**: Visualize execution timeline
- **Causal Graph**: Understand event causality
- **Search**: Find specific events

## Data Flow

### Request Processing

```
User Request
     │
     ▼
┌─────────────┐
│   API/CLI   │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│   Session   │────▶│   Agent     │
│   Manager   │     │   Runtime   │
└─────────────┘     └──────┬──────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    State    │     │     LLM     │     │    Tool     │
│   Machine   │     │   Client    │     │   Gateway   │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Audit     │
                    │   Logger    │
                    └─────────────┘
```

### Commitment Lifecycle

```
┌─────────────┐
│   Created   │
└──────┬──────┘
       │ activate()
       ▼
┌─────────────┐
│   Active    │◀──────────────┐
└──────┬──────┘               │
       │                      │
       ├─── verify() ─────────┤
       │                      │
       ├─── fulfill() ────────┼──▶ ┌─────────────┐
       │                      │    │  Fulfilled  │
       │                      │    └─────────────┘
       │                      │
       ├─── violate() ────────┼──▶ ┌─────────────┐
       │                      │    │  Violated   │
       │                      │    └─────────────┘
       │                      │
       └─── cancel() ─────────┴──▶ ┌─────────────┐
                                   │  Cancelled  │
                                   └─────────────┘
```

## Deployment Architecture

### Single Node

```
┌─────────────────────────────────────┐
│           Single Server             │
│  ┌─────────────────────────────┐   │
│  │      Agent Runtime          │   │
│  │  ┌─────────┐ ┌─────────┐   │   │
│  │  │ Agent 1 │ │ Agent 2 │   │   │
│  │  └─────────┘ └─────────┘   │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │      Storage (SQLite)       │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

### Distributed

```
┌─────────────────┐     ┌─────────────────┐
│   API Gateway   │     │   Load Balancer │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
    ▼                ▼                ▼
┌────────┐      ┌────────┐      ┌────────┐
│Worker 1│      │Worker 2│      │Worker 3│
└────┬───┘      └────┬───┘      └────┬───┘
     │               │               │
     └───────────────┼───────────────┘
                     │
              ┌──────┴──────┐
              │             │
              ▼             ▼
        ┌──────────┐  ┌──────────┐
        │PostgreSQL│  │  Redis   │
        │(Storage) │  │ (Cache)  │
        └──────────┘  └──────────┘
```

## Security Model

### Authentication
- API key authentication for external access
- JWT tokens for session management
- OAuth2 for tool authentication

### Authorization
- Role-based access control (RBAC)
- Policy-based tool access
- Agent-level permissions

### Audit
- All actions logged with full context
- Immutable audit trail
- Compliance-ready reporting

## Performance Considerations

### Scalability
- Horizontal scaling of worker nodes
- Stateless API servers
- Distributed checkpoint storage

### Latency
- In-memory caching for hot data
- Connection pooling for databases
- Async I/O throughout

### Reliability
- Automatic checkpoint recovery
- Graceful degradation
- Circuit breakers for external services

## Extension Points

### Custom Tools
```python
from aegis.tools import Tool, ToolDefinition

@registry.register
def my_custom_tool(arg1: str, arg2: int) -> str:
    """My custom tool description."""
    return f"Result: {arg1}, {arg2}"
```

### Custom Policies
```python
from aegis.tools import PolicyRule, PolicyAction

rule = PolicyRule(
    name="my_rule",
    action=PolicyAction.DENY,
    tool_pattern="dangerous_*",
    condition="context.get('user_role') != 'admin'",
)
```

### Custom Recovery Strategies
```python
from aegis.recovery import RecoveryStrategy

class MyStrategy(RecoveryStrategy):
    @property
    def name(self) -> str:
        return "my_strategy"
    
    async def execute(self, plan, commitment):
        # Custom recovery logic
        pass
```

## Related Documentation

- [API Reference](api-reference.md)
- [Configuration Guide](configuration.md)
- [Deployment Guide](deployment.md)
- [Security Guide](security.md)
