"""
State machine module for durable agent execution.

This module provides:
- Durable state machine with event sourcing
- State transitions with validation
- Persistence integration
"""

from aegis.state_machine.machine import StateMachine, StateMachineConfig
from aegis.state_machine.transitions import TransitionHandler, TransitionResult

__all__ = [
    "StateMachine",
    "StateMachineConfig",
    "TransitionHandler",
    "TransitionResult",
]
