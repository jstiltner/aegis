"""
Event definitions for the agent runtime.

Events are the fundamental unit of state change in the system.
All state transitions are driven by events, enabling:
- Event sourcing for durability
- Deterministic replay
- Causal tracing
- Audit logging
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


class EventType(str, Enum):
    """
    Classification of events in the agent runtime.
    
    Events are organized into categories:
    - Agent lifecycle: start, stop, checkpoint, restore
    - LLM interactions: request, response, error
    - Tool operations: request, response, error, policy
    - GCL commitments: issued, verified, settled
    - User interactions: input, feedback, approval
    """
    
    # Agent lifecycle
    AGENT_START = "agent_start"
    AGENT_STOP = "agent_stop"
    AGENT_ERROR = "agent_error"
    CHECKPOINT_CREATE = "checkpoint_create"
    CHECKPOINT_RESTORE = "checkpoint_restore"
    
    # LLM interactions
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    LLM_STREAM_START = "llm_stream_start"
    LLM_STREAM_CHUNK = "llm_stream_chunk"
    LLM_STREAM_END = "llm_stream_end"
    LLM_ERROR = "llm_error"
    
    # Tool operations
    TOOL_REQUEST = "tool_request"
    TOOL_RESPONSE = "tool_response"
    TOOL_ERROR = "tool_error"
    POLICY_CHECK = "policy_check"
    POLICY_VIOLATION = "policy_violation"
    
    # GCL commitments
    COMMITMENT_ISSUED = "commitment_issued"
    COMMITMENT_VERIFIED = "commitment_verified"
    COMMITMENT_SETTLED = "commitment_settled"
    COMMITMENT_EXPIRED = "commitment_expired"
    
    # User interactions
    USER_INPUT = "user_input"
    USER_FEEDBACK = "user_feedback"
    HUMAN_APPROVAL_REQUEST = "human_approval_request"
    HUMAN_APPROVAL_RESPONSE = "human_approval_response"
    
    # State machine
    STATE_TRANSITION = "state_transition"
    
    # System
    SYSTEM_INFO = "system_info"
    SYSTEM_WARNING = "system_warning"
    SYSTEM_ERROR = "system_error"


class TokenCount(BaseModel):
    """Token usage for an LLM interaction."""
    
    model_config = ConfigDict(frozen=True)
    
    input_tokens: int = Field(ge=0, description="Number of input tokens")
    output_tokens: int = Field(ge=0, description="Number of output tokens")
    cache_read_tokens: int = Field(default=0, ge=0, description="Tokens read from cache")
    cache_write_tokens: int = Field(default=0, ge=0, description="Tokens written to cache")
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens


class Event(BaseModel):
    """
    Base event class for all runtime events.
    
    Events are immutable records of things that happened.
    They form the basis of the event-sourced state machine.
    
    Attributes:
        event_id: Unique identifier for this event
        event_type: Classification of the event
        timestamp: When the event occurred (UTC)
        agent_id: ID of the agent this event belongs to
        session_id: ID of the current session
        parent_event_id: ID of the event that caused this one (for causal tracing)
        caused_by: List of event IDs that contributed to this event
        data: Event-specific payload
        metadata: Additional context (not part of core event data)
    """
    
    model_config = ConfigDict(frozen=True)
    
    event_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this event",
    )
    event_type: EventType = Field(
        ...,
        description="Classification of the event",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the event occurred (UTC)",
    )
    agent_id: str = Field(
        ...,
        description="ID of the agent this event belongs to",
    )
    session_id: str = Field(
        ...,
        description="ID of the current session",
    )
    parent_event_id: str | None = Field(
        default=None,
        description="ID of the event that caused this one",
    )
    caused_by: list[str] = Field(
        default_factory=list,
        description="List of event IDs that contributed to this event",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific payload",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context",
    )
    
    def with_cause(self, parent: Event | str) -> Event:
        """Create a new event with this event as the cause."""
        parent_id = parent.event_id if isinstance(parent, Event) else parent
        return self.model_copy(update={"parent_event_id": parent_id})


# Specialized event types for type safety and convenience


class AgentStartEvent(Event):
    """Event emitted when an agent starts."""
    
    event_type: Literal[EventType.AGENT_START] = EventType.AGENT_START
    data: dict[str, Any] = Field(
        default_factory=lambda: {},
        description="Agent configuration and initial state",
    )


class AgentStopEvent(Event):
    """Event emitted when an agent stops."""
    
    event_type: Literal[EventType.AGENT_STOP] = EventType.AGENT_STOP
    data: dict[str, Any] = Field(
        default_factory=lambda: {},
        description="Final state and stop reason",
    )


class LLMRequestEvent(Event):
    """Event emitted when making an LLM request."""
    
    event_type: Literal[EventType.LLM_REQUEST] = EventType.LLM_REQUEST
    data: dict[str, Any] = Field(
        ...,
        description="Request details: model, messages, parameters",
    )


class LLMResponseEvent(Event):
    """Event emitted when receiving an LLM response."""
    
    event_type: Literal[EventType.LLM_RESPONSE] = EventType.LLM_RESPONSE
    data: dict[str, Any] = Field(
        ...,
        description="Response content and token usage",
    )


class ToolRequestEvent(Event):
    """Event emitted when requesting a tool invocation."""
    
    event_type: Literal[EventType.TOOL_REQUEST] = EventType.TOOL_REQUEST
    data: dict[str, Any] = Field(
        ...,
        description="Tool name, arguments, and commitment info",
    )


class ToolResponseEvent(Event):
    """Event emitted when receiving a tool response."""
    
    event_type: Literal[EventType.TOOL_RESPONSE] = EventType.TOOL_RESPONSE
    data: dict[str, Any] = Field(
        ...,
        description="Tool result and verification outcome",
    )


class CheckpointCreateEvent(Event):
    """Event emitted when creating a checkpoint."""
    
    event_type: Literal[EventType.CHECKPOINT_CREATE] = EventType.CHECKPOINT_CREATE
    data: dict[str, Any] = Field(
        ...,
        description="Checkpoint ID and state summary",
    )


class CheckpointRestoreEvent(Event):
    """Event emitted when restoring from a checkpoint."""
    
    event_type: Literal[EventType.CHECKPOINT_RESTORE] = EventType.CHECKPOINT_RESTORE
    data: dict[str, Any] = Field(
        ...,
        description="Checkpoint ID being restored",
    )


class CommitmentIssuedEvent(Event):
    """Event emitted when a GCL commitment is issued."""
    
    event_type: Literal[EventType.COMMITMENT_ISSUED] = EventType.COMMITMENT_ISSUED
    data: dict[str, Any] = Field(
        ...,
        description="Commitment details: id, trigger, stake, confidence",
    )


class CommitmentVerifiedEvent(Event):
    """Event emitted when a GCL commitment is verified."""
    
    event_type: Literal[EventType.COMMITMENT_VERIFIED] = EventType.COMMITMENT_VERIFIED
    data: dict[str, Any] = Field(
        ...,
        description="Verification result: success/failure, triggered failure mode",
    )


class UserInputEvent(Event):
    """Event emitted when receiving user input."""
    
    event_type: Literal[EventType.USER_INPUT] = EventType.USER_INPUT
    data: dict[str, Any] = Field(
        ...,
        description="User message content",
    )


class PolicyViolationEvent(Event):
    """Event emitted when a policy violation is detected."""
    
    event_type: Literal[EventType.POLICY_VIOLATION] = EventType.POLICY_VIOLATION
    data: dict[str, Any] = Field(
        ...,
        description="Policy name, rule violated, requested action",
    )


# Event factory for creating events with proper typing


class EventFactory:
    """Factory for creating events with consistent agent/session context."""
    
    def __init__(self, agent_id: str, session_id: str):
        self.agent_id = agent_id
        self.session_id = session_id
        self._last_event_id: str | None = None
    
    def _base_kwargs(self) -> dict[str, Any]:
        """Get base kwargs for event creation."""
        return {
            "agent_id": self.agent_id,
            "session_id": self.session_id,
        }
    
    def agent_start(self, config: dict[str, Any] | None = None) -> AgentStartEvent:
        """Create an agent start event."""
        event = AgentStartEvent(
            **self._base_kwargs(),
            data={"config": config or {}},
        )
        self._last_event_id = event.event_id
        return event
    
    def agent_stop(self, reason: str, final_state: dict[str, Any] | None = None) -> AgentStopEvent:
        """Create an agent stop event."""
        event = AgentStopEvent(
            **self._base_kwargs(),
            data={"reason": reason, "final_state": final_state or {}},
            parent_event_id=self._last_event_id,
        )
        self._last_event_id = event.event_id
        return event
    
    def llm_request(
        self,
        model: str,
        messages: list[dict[str, Any]],
        parameters: dict[str, Any] | None = None,
        parent_event_id: str | None = None,
    ) -> LLMRequestEvent:
        """Create an LLM request event."""
        event = LLMRequestEvent(
            **self._base_kwargs(),
            data={
                "model": model,
                "messages": messages,
                "parameters": parameters or {},
            },
            parent_event_id=parent_event_id or self._last_event_id,
        )
        self._last_event_id = event.event_id
        return event
    
    def llm_response(
        self,
        content: str,
        token_count: TokenCount | None = None,
        model: str | None = None,
        request_event_id: str | None = None,
        duration_ms: float | None = None,
    ) -> LLMResponseEvent:
        """Create an LLM response event."""
        event = LLMResponseEvent(
            **self._base_kwargs(),
            data={
                "content": content,
                "token_count": token_count.model_dump() if token_count else None,
                "model": model,
                "duration_ms": duration_ms,
            },
            parent_event_id=request_event_id or self._last_event_id,
        )
        self._last_event_id = event.event_id
        return event
    
    def tool_request(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        commitment_id: str | None = None,
        parent_event_id: str | None = None,
    ) -> ToolRequestEvent:
        """Create a tool request event."""
        event = ToolRequestEvent(
            **self._base_kwargs(),
            data={
                "tool_name": tool_name,
                "arguments": arguments,
                "commitment_id": commitment_id,
            },
            parent_event_id=parent_event_id or self._last_event_id,
        )
        self._last_event_id = event.event_id
        return event
    
    def tool_response(
        self,
        tool_name: str,
        result: Any,
        success: bool = True,
        error: str | None = None,
        request_event_id: str | None = None,
        duration_ms: float | None = None,
    ) -> ToolResponseEvent:
        """Create a tool response event."""
        event = ToolResponseEvent(
            **self._base_kwargs(),
            data={
                "tool_name": tool_name,
                "result": result,
                "success": success,
                "error": error,
                "duration_ms": duration_ms,
            },
            parent_event_id=request_event_id or self._last_event_id,
        )
        self._last_event_id = event.event_id
        return event
    
    def checkpoint_create(
        self,
        checkpoint_id: str,
        state_summary: dict[str, Any] | None = None,
    ) -> CheckpointCreateEvent:
        """Create a checkpoint creation event."""
        event = CheckpointCreateEvent(
            **self._base_kwargs(),
            data={
                "checkpoint_id": checkpoint_id,
                "state_summary": state_summary or {},
            },
            parent_event_id=self._last_event_id,
        )
        self._last_event_id = event.event_id
        return event
    
    def checkpoint_restore(
        self,
        checkpoint_id: str,
        reason: str | None = None,
    ) -> CheckpointRestoreEvent:
        """Create a checkpoint restore event."""
        event = CheckpointRestoreEvent(
            **self._base_kwargs(),
            data={
                "checkpoint_id": checkpoint_id,
                "reason": reason,
            },
            parent_event_id=self._last_event_id,
        )
        self._last_event_id = event.event_id
        return event
    
    def user_input(self, content: str, role: str = "user") -> UserInputEvent:
        """Create a user input event."""
        event = UserInputEvent(
            **self._base_kwargs(),
            data={
                "content": content,
                "role": role,
            },
            parent_event_id=self._last_event_id,
        )
        self._last_event_id = event.event_id
        return event
    
    def commitment_issued(
        self,
        commitment_id: str,
        trigger: str,
        stake: float,
        confidence: float,
        failure_modes: list[str],
    ) -> CommitmentIssuedEvent:
        """Create a commitment issued event."""
        event = CommitmentIssuedEvent(
            **self._base_kwargs(),
            data={
                "commitment_id": commitment_id,
                "trigger": trigger,
                "stake": stake,
                "confidence": confidence,
                "failure_modes": failure_modes,
            },
            parent_event_id=self._last_event_id,
        )
        self._last_event_id = event.event_id
        return event
    
    def commitment_verified(
        self,
        commitment_id: str,
        success: bool,
        triggered_failure_mode: str | None = None,
        reward: float | None = None,
    ) -> CommitmentVerifiedEvent:
        """Create a commitment verified event."""
        event = CommitmentVerifiedEvent(
            **self._base_kwargs(),
            data={
                "commitment_id": commitment_id,
                "success": success,
                "triggered_failure_mode": triggered_failure_mode,
                "reward": reward,
            },
            parent_event_id=self._last_event_id,
        )
        self._last_event_id = event.event_id
        return event
    
    def policy_violation(
        self,
        policy_name: str,
        rule: str,
        tool_name: str,
        arguments: dict[str, Any],
        reason: str,
    ) -> PolicyViolationEvent:
        """Create a policy violation event."""
        event = PolicyViolationEvent(
            **self._base_kwargs(),
            data={
                "policy_name": policy_name,
                "rule": rule,
                "tool_name": tool_name,
                "arguments": arguments,
                "reason": reason,
            },
            parent_event_id=self._last_event_id,
        )
        self._last_event_id = event.event_id
        return event
    
    def generic(
        self,
        event_type: EventType,
        data: dict[str, Any],
        parent_event_id: str | None = None,
    ) -> Event:
        """Create a generic event of any type."""
        event = Event(
            **self._base_kwargs(),
            event_type=event_type,
            data=data,
            parent_event_id=parent_event_id or self._last_event_id,
        )
        self._last_event_id = event.event_id
        return event
