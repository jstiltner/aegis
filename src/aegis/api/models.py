"""
API request and response models.

This module provides Pydantic models for API requests and responses.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# Session Models
# =============================================================================

class SessionCreate(BaseModel):
    """Request to create a new session."""
    
    agent_id: str = Field(..., description="Agent identifier")
    system_prompt: str | None = Field(
        default=None,
        description="System prompt for the agent",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Session metadata",
    )


class SessionStatus(str, Enum):
    """Session status."""
    
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class SessionResponse(BaseModel):
    """Session information response."""
    
    model_config = ConfigDict(from_attributes=True)
    
    session_id: str = Field(..., description="Session identifier")
    agent_id: str = Field(..., description="Agent identifier")
    status: SessionStatus = Field(..., description="Session status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    message_count: int = Field(default=0, description="Number of messages")
    checkpoint_count: int = Field(default=0, description="Number of checkpoints")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Session metadata",
    )


class SessionList(BaseModel):
    """List of sessions."""
    
    sessions: list[SessionResponse] = Field(
        default_factory=list,
        description="List of sessions",
    )
    total: int = Field(default=0, description="Total number of sessions")
    offset: int = Field(default=0, description="Offset for pagination")
    limit: int = Field(default=20, description="Limit for pagination")


# =============================================================================
# Agent Models
# =============================================================================

class AgentCreate(BaseModel):
    """Request to create a new agent."""
    
    name: str = Field(..., description="Agent name")
    description: str = Field(default="", description="Agent description")
    model: str = Field(
        default="claude-sonnet-4-20250514",
        description="LLM model to use",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Default system prompt",
    )
    tools: list[str] = Field(
        default_factory=list,
        description="List of tool names to enable",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent metadata",
    )


class AgentResponse(BaseModel):
    """Agent information response."""
    
    model_config = ConfigDict(from_attributes=True)
    
    agent_id: str = Field(..., description="Agent identifier")
    name: str = Field(..., description="Agent name")
    description: str = Field(default="", description="Agent description")
    model: str = Field(..., description="LLM model")
    system_prompt: str | None = Field(default=None, description="System prompt")
    tools: list[str] = Field(default_factory=list, description="Enabled tools")
    created_at: datetime = Field(..., description="Creation timestamp")
    session_count: int = Field(default=0, description="Number of sessions")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent metadata",
    )


class AgentList(BaseModel):
    """List of agents."""
    
    agents: list[AgentResponse] = Field(
        default_factory=list,
        description="List of agents",
    )
    total: int = Field(default=0, description="Total number of agents")


# =============================================================================
# Message Models
# =============================================================================

class MessageRole(str, Enum):
    """Message role."""
    
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MessageRequest(BaseModel):
    """Request to send a message."""
    
    content: str = Field(..., description="Message content")
    role: MessageRole = Field(
        default=MessageRole.USER,
        description="Message role",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Message metadata",
    )


class ToolUseResponse(BaseModel):
    """Tool use in a response."""
    
    id: str = Field(..., description="Tool use ID")
    name: str = Field(..., description="Tool name")
    input: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool input",
    )


class MessageResponse(BaseModel):
    """Message response."""
    
    model_config = ConfigDict(from_attributes=True)
    
    message_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Message identifier",
    )
    session_id: str = Field(..., description="Session identifier")
    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    tool_uses: list[ToolUseResponse] = Field(
        default_factory=list,
        description="Tool uses in the message",
    )
    created_at: datetime = Field(..., description="Creation timestamp")
    duration_ms: float | None = Field(
        default=None,
        description="Processing duration in milliseconds",
    )
    token_count: int | None = Field(
        default=None,
        description="Token count",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Message metadata",
    )


class MessageList(BaseModel):
    """List of messages."""
    
    messages: list[MessageResponse] = Field(
        default_factory=list,
        description="List of messages",
    )
    total: int = Field(default=0, description="Total number of messages")


# =============================================================================
# Trace Models
# =============================================================================

class SpanResponse(BaseModel):
    """Span information response."""
    
    model_config = ConfigDict(from_attributes=True)
    
    span_id: str = Field(..., description="Span identifier")
    trace_id: str = Field(..., description="Trace identifier")
    parent_span_id: str | None = Field(
        default=None,
        description="Parent span identifier",
    )
    name: str = Field(..., description="Span name")
    start_time: datetime = Field(..., description="Start timestamp")
    end_time: datetime | None = Field(
        default=None,
        description="End timestamp",
    )
    duration_ms: float | None = Field(
        default=None,
        description="Duration in milliseconds",
    )
    status: str = Field(default="ok", description="Span status")
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Span attributes",
    )
    events: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Span events",
    )


class TraceResponse(BaseModel):
    """Trace information response."""
    
    model_config = ConfigDict(from_attributes=True)
    
    trace_id: str = Field(..., description="Trace identifier")
    session_id: str | None = Field(
        default=None,
        description="Associated session",
    )
    root_span_id: str | None = Field(
        default=None,
        description="Root span identifier",
    )
    start_time: datetime = Field(..., description="Start timestamp")
    end_time: datetime | None = Field(
        default=None,
        description="End timestamp",
    )
    duration_ms: float | None = Field(
        default=None,
        description="Total duration in milliseconds",
    )
    span_count: int = Field(default=0, description="Number of spans")
    spans: list[SpanResponse] = Field(
        default_factory=list,
        description="Spans in the trace",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Trace metadata",
    )


class TraceList(BaseModel):
    """List of traces."""
    
    traces: list[TraceResponse] = Field(
        default_factory=list,
        description="List of traces",
    )
    total: int = Field(default=0, description="Total number of traces")


# =============================================================================
# Tool Models
# =============================================================================

class ToolParameter(BaseModel):
    """Tool parameter definition."""
    
    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type")
    description: str = Field(default="", description="Parameter description")
    required: bool = Field(default=False, description="Whether required")
    default: Any = Field(default=None, description="Default value")


class ToolResponse(BaseModel):
    """Tool information response."""
    
    model_config = ConfigDict(from_attributes=True)
    
    name: str = Field(..., description="Tool name")
    description: str = Field(default="", description="Tool description")
    parameters: list[ToolParameter] = Field(
        default_factory=list,
        description="Tool parameters",
    )
    category: str = Field(default="general", description="Tool category")
    requires_auth: bool = Field(
        default=False,
        description="Whether authentication is required",
    )
    enabled: bool = Field(default=True, description="Whether enabled")


class ToolList(BaseModel):
    """List of tools."""
    
    tools: list[ToolResponse] = Field(
        default_factory=list,
        description="List of tools",
    )
    total: int = Field(default=0, description="Total number of tools")


class ToolInvocation(BaseModel):
    """Tool invocation request."""
    
    tool_name: str = Field(..., description="Tool name")
    input: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool input",
    )
    session_id: str | None = Field(
        default=None,
        description="Session context",
    )


class ToolResult(BaseModel):
    """Tool invocation result."""
    
    tool_name: str = Field(..., description="Tool name")
    output: Any = Field(..., description="Tool output")
    is_error: bool = Field(default=False, description="Whether an error occurred")
    error_message: str | None = Field(
        default=None,
        description="Error message if error occurred",
    )
    duration_ms: float = Field(..., description="Execution duration")


# =============================================================================
# Checkpoint Models
# =============================================================================

class CheckpointResponse(BaseModel):
    """Checkpoint information response."""
    
    model_config = ConfigDict(from_attributes=True)
    
    checkpoint_id: str = Field(..., description="Checkpoint identifier")
    session_id: str = Field(..., description="Session identifier")
    version: int = Field(..., description="State version")
    created_at: datetime = Field(..., description="Creation timestamp")
    parent_checkpoint_id: str | None = Field(
        default=None,
        description="Parent checkpoint",
    )
    state_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="State summary",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Checkpoint metadata",
    )


class CheckpointList(BaseModel):
    """List of checkpoints."""
    
    checkpoints: list[CheckpointResponse] = Field(
        default_factory=list,
        description="List of checkpoints",
    )
    total: int = Field(default=0, description="Total number of checkpoints")


# =============================================================================
# Commitment Models
# =============================================================================

class CommitmentStatus(str, Enum):
    """Commitment status."""
    
    PENDING = "pending"
    ACTIVE = "active"
    FULFILLED = "fulfilled"
    VIOLATED = "violated"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class CommitmentResponse(BaseModel):
    """Commitment information response."""
    
    model_config = ConfigDict(from_attributes=True)
    
    commitment_id: str = Field(..., description="Commitment identifier")
    session_id: str = Field(..., description="Session identifier")
    debtor: str = Field(..., description="Debtor (who made the commitment)")
    creditor: str = Field(..., description="Creditor (who receives)")
    antecedent: str = Field(..., description="Trigger condition")
    consequent: str = Field(..., description="Promised action")
    status: CommitmentStatus = Field(..., description="Current status")
    created_at: datetime = Field(..., description="Creation timestamp")
    deadline: datetime | None = Field(
        default=None,
        description="Deadline for fulfillment",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Commitment metadata",
    )


class CommitmentList(BaseModel):
    """List of commitments."""
    
    commitments: list[CommitmentResponse] = Field(
        default_factory=list,
        description="List of commitments",
    )
    total: int = Field(default=0, description="Total number of commitments")


# =============================================================================
# Error Models
# =============================================================================

class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Error details",
    )
    trace_id: str | None = Field(
        default=None,
        description="Trace ID for debugging",
    )


# =============================================================================
# Health Models
# =============================================================================

class HealthStatus(str, Enum):
    """Health status."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: HealthStatus = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Uptime in seconds")
    checks: dict[str, HealthStatus] = Field(
        default_factory=dict,
        description="Individual health checks",
    )
