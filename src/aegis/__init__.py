"""
Agent Runtime: Production infrastructure for long-running autonomous agents.

This package provides:
- Durable state machine with checkpoint/replay
- Tool gateway with policy enforcement
- Audit logging with causal tracing
- GCL commitment verification integration
"""

__version__ = "0.1.0"

from aegis.core.agent import Agent
from aegis.core.state import AgentState, StateTransition
from aegis.core.events import Event, EventType

__all__ = [
    "Agent",
    "AgentState",
    "StateTransition",
    "Event",
    "EventType",
    "__version__",
]
