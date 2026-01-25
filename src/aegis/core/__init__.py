"""
Core module: Fundamental data structures and runtime components.

This module contains:
- AgentState and StateTransition models
- Event definitions
- Checkpoint management
- Replay engine
- Base agent class
"""

from aegis.core.state import AgentState, StateTransition, Message
from aegis.core.events import Event, EventType
from aegis.core.checkpoint import CheckpointManager, Checkpoint
from aegis.core.replay import ReplayEngine

__all__ = [
    "AgentState",
    "StateTransition",
    "Message",
    "Event",
    "EventType",
    "CheckpointManager",
    "Checkpoint",
    "ReplayEngine",
]
