"""
Audit event definitions.

This module provides:
- Audit event types and severity levels
- Structured audit event model
- Event factory for common events
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


class AuditEventType(str, Enum):
    """Types of audit events."""
    
    # Agent lifecycle
    AGENT_STARTED = "agent.started"
    AGENT_STOPPED = "agent.stopped"
    AGENT_PAUSED = "agent.paused"
    AGENT_RESUMED = "agent.resumed"
    AGENT_ERROR = "agent.error"
    
    # State changes
    STATE_CHANGED = "state.changed"
    STATE_CHECKPOINT = "state.checkpoint"
    STATE_RESTORED = "state.restored"
    
    # Tool invocations
    TOOL_INVOKED = "tool.invoked"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    TOOL_TIMEOUT = "tool.timeout"
    
    # Policy events
    POLICY_CHECKED = "policy.checked"
    POLICY_DENIED = "policy.denied"
    POLICY_APPROVAL_REQUIRED = "policy.approval_required"
    POLICY_APPROVED = "policy.approved"
    POLICY_REJECTED = "policy.rejected"
    
    # LLM interactions
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    LLM_ERROR = "llm.error"
    
    # Commitment events (GCL)
    COMMITMENT_CREATED = "commitment.created"
    COMMITMENT_VERIFIED = "commitment.verified"
    COMMITMENT_FAILED = "commitment.failed"
    COMMITMENT_EXPIRED = "commitment.expired"
    
    # Security events
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILURE = "auth.failure"
    TOKEN_ISSUED = "token.issued"
    TOKEN_REVOKED = "token.revoked"
    
    # System events
    SYSTEM_INFO = "system.info"
    SYSTEM_WARNING = "system.warning"
    SYSTEM_ERROR = "system.error"


class AuditEventSeverity(str, Enum):
    """Severity levels for audit events."""
    
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditEvent(BaseModel):
    """
    A structured audit event.
    
    Audit events capture important actions and state changes
    in the agent runtime for accountability and debugging.
    """
    
    model_config = ConfigDict(frozen=True)
    
    # Identity
    event_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this event",
    )
    event_type: AuditEventType = Field(
        ...,
        description="Type of event",
    )
    severity: AuditEventSeverity = Field(
        default=AuditEventSeverity.INFO,
        description="Event severity level",
    )
    
    # Timing
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the event occurred",
    )
    
    # Context
    trace_id: str | None = Field(
        default=None,
        description="Trace ID for correlation",
    )
    span_id: str | None = Field(
        default=None,
        description="Span ID within the trace",
    )
    parent_span_id: str | None = Field(
        default=None,
        description="Parent span ID",
    )
    
    # Actor
    agent_id: str | None = Field(
        default=None,
        description="ID of the agent that generated this event",
    )
    session_id: str | None = Field(
        default=None,
        description="ID of the session",
    )
    
    # Content
    message: str = Field(
        ...,
        description="Human-readable event message",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured event data",
    )
    
    # Metadata
    tags: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Tags for filtering",
    )
    source: str = Field(
        default="agent_runtime",
        description="Source component",
    )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "message": self.message,
            "data": self.data,
            "tags": list(self.tags),
            "source": self.source,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditEvent:
        """Create from dictionary."""
        return cls(
            event_id=data.get("event_id", str(uuid4())),
            event_type=AuditEventType(data["event_type"]),
            severity=AuditEventSeverity(data.get("severity", "info")),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else data.get("timestamp", datetime.now(timezone.utc)),
            trace_id=data.get("trace_id"),
            span_id=data.get("span_id"),
            parent_span_id=data.get("parent_span_id"),
            agent_id=data.get("agent_id"),
            session_id=data.get("session_id"),
            message=data["message"],
            data=data.get("data", {}),
            tags=tuple(data.get("tags", [])),
            source=data.get("source", "agent_runtime"),
        )


class EventFactory:
    """
    Factory for creating common audit events.
    
    Provides convenience methods for creating well-structured events.
    """
    
    def __init__(
        self,
        agent_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        source: str = "agent_runtime",
    ) -> None:
        """
        Initialize the factory with default context.
        
        Args:
            agent_id: Default agent ID
            session_id: Default session ID
            trace_id: Default trace ID
            source: Default source component
        """
        self.agent_id = agent_id
        self.session_id = session_id
        self.trace_id = trace_id
        self.source = source
    
    def _create_event(
        self,
        event_type: AuditEventType,
        message: str,
        severity: AuditEventSeverity = AuditEventSeverity.INFO,
        data: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Create an event with default context."""
        return AuditEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            data=data or {},
            tags=tuple(tags or []),
            agent_id=kwargs.get("agent_id", self.agent_id),
            session_id=kwargs.get("session_id", self.session_id),
            trace_id=kwargs.get("trace_id", self.trace_id),
            span_id=kwargs.get("span_id"),
            parent_span_id=kwargs.get("parent_span_id"),
            source=kwargs.get("source", self.source),
        )
    
    # Agent lifecycle events
    
    def agent_started(
        self,
        agent_id: str,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Create an agent started event."""
        return self._create_event(
            event_type=AuditEventType.AGENT_STARTED,
            message=f"Agent {agent_id} started",
            data={"config": config or {}},
            tags=["lifecycle"],
            agent_id=agent_id,
            **kwargs,
        )
    
    def agent_stopped(
        self,
        agent_id: str,
        reason: str = "normal",
        **kwargs: Any,
    ) -> AuditEvent:
        """Create an agent stopped event."""
        return self._create_event(
            event_type=AuditEventType.AGENT_STOPPED,
            message=f"Agent {agent_id} stopped: {reason}",
            data={"reason": reason},
            tags=["lifecycle"],
            agent_id=agent_id,
            **kwargs,
        )
    
    def agent_error(
        self,
        agent_id: str,
        error: str,
        error_type: str = "Error",
        **kwargs: Any,
    ) -> AuditEvent:
        """Create an agent error event."""
        return self._create_event(
            event_type=AuditEventType.AGENT_ERROR,
            severity=AuditEventSeverity.ERROR,
            message=f"Agent {agent_id} error: {error}",
            data={"error": error, "error_type": error_type},
            tags=["lifecycle", "error"],
            agent_id=agent_id,
            **kwargs,
        )
    
    # State events
    
    def state_changed(
        self,
        old_state: str,
        new_state: str,
        trigger: str | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Create a state changed event."""
        return self._create_event(
            event_type=AuditEventType.STATE_CHANGED,
            message=f"State changed: {old_state} -> {new_state}",
            data={
                "old_state": old_state,
                "new_state": new_state,
                "trigger": trigger,
            },
            tags=["state"],
            **kwargs,
        )
    
    def state_checkpoint(
        self,
        checkpoint_id: str,
        state_name: str,
        **kwargs: Any,
    ) -> AuditEvent:
        """Create a state checkpoint event."""
        return self._create_event(
            event_type=AuditEventType.STATE_CHECKPOINT,
            message=f"Checkpoint created: {checkpoint_id}",
            data={
                "checkpoint_id": checkpoint_id,
                "state_name": state_name,
            },
            tags=["state", "checkpoint"],
            **kwargs,
        )
    
    def state_restored(
        self,
        checkpoint_id: str,
        state_name: str,
        **kwargs: Any,
    ) -> AuditEvent:
        """Create a state restored event."""
        return self._create_event(
            event_type=AuditEventType.STATE_RESTORED,
            message=f"State restored from checkpoint: {checkpoint_id}",
            data={
                "checkpoint_id": checkpoint_id,
                "state_name": state_name,
            },
            tags=["state", "checkpoint"],
            **kwargs,
        )
    
    # Tool events
    
    def tool_invoked(
        self,
        tool_name: str,
        invocation_id: str,
        arguments: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Create a tool invoked event."""
        return self._create_event(
            event_type=AuditEventType.TOOL_INVOKED,
            message=f"Tool invoked: {tool_name}",
            data={
                "tool_name": tool_name,
                "invocation_id": invocation_id,
                "arguments": arguments or {},
            },
            tags=["tool"],
            **kwargs,
        )
    
    def tool_completed(
        self,
        tool_name: str,
        invocation_id: str,
        duration_ms: float,
        output_summary: str | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Create a tool completed event."""
        return self._create_event(
            event_type=AuditEventType.TOOL_COMPLETED,
            message=f"Tool completed: {tool_name} ({duration_ms:.2f}ms)",
            data={
                "tool_name": tool_name,
                "invocation_id": invocation_id,
                "duration_ms": duration_ms,
                "output_summary": output_summary,
            },
            tags=["tool"],
            **kwargs,
        )
    
    def tool_failed(
        self,
        tool_name: str,
        invocation_id: str,
        error: str,
        error_type: str = "Error",
        **kwargs: Any,
    ) -> AuditEvent:
        """Create a tool failed event."""
        return self._create_event(
            event_type=AuditEventType.TOOL_FAILED,
            severity=AuditEventSeverity.ERROR,
            message=f"Tool failed: {tool_name} - {error}",
            data={
                "tool_name": tool_name,
                "invocation_id": invocation_id,
                "error": error,
                "error_type": error_type,
            },
            tags=["tool", "error"],
            **kwargs,
        )
    
    def tool_timeout(
        self,
        tool_name: str,
        invocation_id: str,
        timeout_seconds: float,
        **kwargs: Any,
    ) -> AuditEvent:
        """Create a tool timeout event."""
        return self._create_event(
            event_type=AuditEventType.TOOL_TIMEOUT,
            severity=AuditEventSeverity.WARNING,
            message=f"Tool timeout: {tool_name} after {timeout_seconds}s",
            data={
                "tool_name": tool_name,
                "invocation_id": invocation_id,
                "timeout_seconds": timeout_seconds,
            },
            tags=["tool", "timeout"],
            **kwargs,
        )
    
    # Policy events
    
    def policy_denied(
        self,
        tool_name: str,
        policy_name: str,
        reason: str,
        **kwargs: Any,
    ) -> AuditEvent:
        """Create a policy denied event."""
        return self._create_event(
            event_type=AuditEventType.POLICY_DENIED,
            severity=AuditEventSeverity.WARNING,
            message=f"Policy denied: {tool_name} by {policy_name}",
            data={
                "tool_name": tool_name,
                "policy_name": policy_name,
                "reason": reason,
            },
            tags=["policy", "denied"],
            **kwargs,
        )
    
    def policy_approval_required(
        self,
        tool_name: str,
        policy_name: str,
        approval_type: str,
        **kwargs: Any,
    ) -> AuditEvent:
        """Create a policy approval required event."""
        return self._create_event(
            event_type=AuditEventType.POLICY_APPROVAL_REQUIRED,
            message=f"Approval required for {tool_name}: {approval_type}",
            data={
                "tool_name": tool_name,
                "policy_name": policy_name,
                "approval_type": approval_type,
            },
            tags=["policy", "approval"],
            **kwargs,
        )
    
    # LLM events
    
    def llm_request(
        self,
        provider: str,
        model: str,
        token_count: int | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Create an LLM request event."""
        return self._create_event(
            event_type=AuditEventType.LLM_REQUEST,
            message=f"LLM request: {provider}/{model}",
            data={
                "provider": provider,
                "model": model,
                "token_count": token_count,
            },
            tags=["llm"],
            **kwargs,
        )
    
    def llm_response(
        self,
        provider: str,
        model: str,
        duration_ms: float,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Create an LLM response event."""
        return self._create_event(
            event_type=AuditEventType.LLM_RESPONSE,
            message=f"LLM response: {provider}/{model} ({duration_ms:.2f}ms)",
            data={
                "provider": provider,
                "model": model,
                "duration_ms": duration_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            tags=["llm"],
            **kwargs,
        )
    
    def llm_error(
        self,
        provider: str,
        model: str,
        error: str,
        **kwargs: Any,
    ) -> AuditEvent:
        """Create an LLM error event."""
        return self._create_event(
            event_type=AuditEventType.LLM_ERROR,
            severity=AuditEventSeverity.ERROR,
            message=f"LLM error: {provider}/{model} - {error}",
            data={
                "provider": provider,
                "model": model,
                "error": error,
            },
            tags=["llm", "error"],
            **kwargs,
        )
    
    # Commitment events
    
    def commitment_created(
        self,
        commitment_id: str,
        action: str,
        verification: str,
        **kwargs: Any,
    ) -> AuditEvent:
        """Create a commitment created event."""
        return self._create_event(
            event_type=AuditEventType.COMMITMENT_CREATED,
            message=f"Commitment created: {commitment_id}",
            data={
                "commitment_id": commitment_id,
                "action": action,
                "verification": verification,
            },
            tags=["commitment", "gcl"],
            **kwargs,
        )
    
    def commitment_verified(
        self,
        commitment_id: str,
        result: bool,
        **kwargs: Any,
    ) -> AuditEvent:
        """Create a commitment verified event."""
        return self._create_event(
            event_type=AuditEventType.COMMITMENT_VERIFIED,
            message=f"Commitment verified: {commitment_id} = {result}",
            data={
                "commitment_id": commitment_id,
                "result": result,
            },
            tags=["commitment", "gcl"],
            **kwargs,
        )
    
    def commitment_failed(
        self,
        commitment_id: str,
        reason: str,
        **kwargs: Any,
    ) -> AuditEvent:
        """Create a commitment failed event."""
        return self._create_event(
            event_type=AuditEventType.COMMITMENT_FAILED,
            severity=AuditEventSeverity.WARNING,
            message=f"Commitment failed: {commitment_id} - {reason}",
            data={
                "commitment_id": commitment_id,
                "reason": reason,
            },
            tags=["commitment", "gcl", "failure"],
            **kwargs,
        )
    
    # Security events
    
    def auth_success(
        self,
        credential_name: str,
        tool_name: str | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Create an auth success event."""
        return self._create_event(
            event_type=AuditEventType.AUTH_SUCCESS,
            message=f"Authentication successful: {credential_name}",
            data={
                "credential_name": credential_name,
                "tool_name": tool_name,
            },
            tags=["auth", "security"],
            **kwargs,
        )
    
    def auth_failure(
        self,
        credential_name: str,
        reason: str,
        **kwargs: Any,
    ) -> AuditEvent:
        """Create an auth failure event."""
        return self._create_event(
            event_type=AuditEventType.AUTH_FAILURE,
            severity=AuditEventSeverity.WARNING,
            message=f"Authentication failed: {credential_name} - {reason}",
            data={
                "credential_name": credential_name,
                "reason": reason,
            },
            tags=["auth", "security", "failure"],
            **kwargs,
        )
    
    # System events
    
    def system_info(
        self,
        message: str,
        data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Create a system info event."""
        return self._create_event(
            event_type=AuditEventType.SYSTEM_INFO,
            message=message,
            data=data or {},
            tags=["system"],
            **kwargs,
        )
    
    def system_warning(
        self,
        message: str,
        data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Create a system warning event."""
        return self._create_event(
            event_type=AuditEventType.SYSTEM_WARNING,
            severity=AuditEventSeverity.WARNING,
            message=message,
            data=data or {},
            tags=["system", "warning"],
            **kwargs,
        )
    
    def system_error(
        self,
        message: str,
        error: str | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Create a system error event."""
        return self._create_event(
            event_type=AuditEventType.SYSTEM_ERROR,
            severity=AuditEventSeverity.ERROR,
            message=message,
            data={"error": error} if error else {},
            tags=["system", "error"],
            **kwargs,
        )
