"""
GCL commitment models for the agent runtime.

This module provides runtime-specific commitment models that bridge
the agent runtime with the GCL framework.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


class CommitmentStatus(str, Enum):
    """Status of a runtime commitment."""
    
    PENDING = "pending"
    ACTIVE = "active"
    FULFILLED = "fulfilled"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class CommitmentResult(BaseModel):
    """
    Result of a commitment verification.
    
    Attributes:
        commitment_id: ID of the commitment
        status: Verification status
        success: Whether the commitment was fulfilled
        failure_mode: Name of triggered failure mode (if any)
        details: Additional result details
        timestamp: When verification occurred
        duration_ms: Verification duration
    """
    
    model_config = ConfigDict(frozen=True)
    
    commitment_id: str = Field(..., description="ID of the commitment")
    status: CommitmentStatus = Field(..., description="Verification status")
    success: bool = Field(..., description="Whether the commitment was fulfilled")
    failure_mode: str | None = Field(
        default=None,
        description="Name of triggered failure mode",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional result details",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When verification occurred",
    )
    duration_ms: float | None = Field(
        default=None,
        description="Verification duration in milliseconds",
    )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "commitment_id": self.commitment_id,
            "status": self.status.value,
            "success": self.success,
            "failure_mode": self.failure_mode,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
        }


class RuntimeCommitment(BaseModel):
    """
    A commitment in the agent runtime.
    
    This is a simplified commitment model for the runtime that can be
    converted to/from GCL's GroundedCommitment.
    
    Attributes:
        id: Unique identifier
        agent_id: ID of the agent making the commitment
        action_type: Type of action being committed to
        description: Human-readable description
        trigger_condition: When this commitment activates
        success_condition: What constitutes success
        failure_conditions: Ways this commitment can fail
        stake: Amount staked on this commitment
        confidence: Agent's confidence (0.0 to 1.0)
        timeout_seconds: Maximum time for fulfillment
        created_at: When the commitment was created
        expires_at: When the commitment expires
        status: Current status
        metadata: Additional data
    """
    
    model_config = ConfigDict(frozen=False)
    
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier",
    )
    agent_id: str = Field(..., description="ID of the agent making the commitment")
    action_type: str = Field(..., description="Type of action being committed to")
    description: str = Field(..., description="Human-readable description")
    
    # Conditions (simplified from GCL predicates)
    trigger_condition: str | None = Field(
        default=None,
        description="Expression for when this commitment activates",
    )
    success_condition: str = Field(
        ...,
        description="Expression defining successful completion",
    )
    failure_conditions: dict[str, str] = Field(
        default_factory=dict,
        description="Map of failure mode name to condition expression",
    )
    
    # Stakes and confidence
    stake: float = Field(
        default=1.0,
        ge=0.0,
        description="Amount staked on this commitment",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in fulfilling this commitment",
    )
    
    # Timing
    timeout_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Maximum time for fulfillment",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the commitment was created",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="When the commitment expires",
    )
    
    # Status
    status: CommitmentStatus = Field(
        default=CommitmentStatus.PENDING,
        description="Current status",
    )
    
    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional data",
    )
    
    # Tracking
    tool_call_id: str | None = Field(
        default=None,
        description="Associated tool call ID",
    )
    invocation_id: str | None = Field(
        default=None,
        description="Associated invocation ID",
    )
    
    def is_expired(self) -> bool:
        """Check if this commitment has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def is_active(self) -> bool:
        """Check if this commitment is active."""
        return self.status == CommitmentStatus.ACTIVE and not self.is_expired()
    
    def is_terminal(self) -> bool:
        """Check if this commitment is in a terminal state."""
        return self.status in (
            CommitmentStatus.FULFILLED,
            CommitmentStatus.FAILED,
            CommitmentStatus.EXPIRED,
            CommitmentStatus.CANCELLED,
        )
    
    def activate(self) -> RuntimeCommitment:
        """Activate this commitment."""
        return RuntimeCommitment(
            **{**self.model_dump(), "status": CommitmentStatus.ACTIVE}
        )
    
    def fulfill(self) -> RuntimeCommitment:
        """Mark this commitment as fulfilled."""
        return RuntimeCommitment(
            **{**self.model_dump(), "status": CommitmentStatus.FULFILLED}
        )
    
    def fail(self, failure_mode: str | None = None) -> RuntimeCommitment:
        """Mark this commitment as failed."""
        metadata = {**self.metadata}
        if failure_mode:
            metadata["failure_mode"] = failure_mode
        return RuntimeCommitment(
            **{**self.model_dump(), "status": CommitmentStatus.FAILED, "metadata": metadata}
        )
    
    def cancel(self) -> RuntimeCommitment:
        """Cancel this commitment."""
        return RuntimeCommitment(
            **{**self.model_dump(), "status": CommitmentStatus.CANCELLED}
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "description": self.description,
            "trigger_condition": self.trigger_condition,
            "success_condition": self.success_condition,
            "failure_conditions": self.failure_conditions,
            "stake": self.stake,
            "confidence": self.confidence,
            "timeout_seconds": self.timeout_seconds,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "status": self.status.value,
            "metadata": self.metadata,
            "tool_call_id": self.tool_call_id,
            "invocation_id": self.invocation_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeCommitment:
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid4())),
            agent_id=data["agent_id"],
            action_type=data["action_type"],
            description=data["description"],
            trigger_condition=data.get("trigger_condition"),
            success_condition=data["success_condition"],
            failure_conditions=data.get("failure_conditions", {}),
            stake=data.get("stake", 1.0),
            confidence=data.get("confidence", 0.8),
            timeout_seconds=data.get("timeout_seconds"),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.now(timezone.utc)),
            expires_at=datetime.fromisoformat(data["expires_at"]) if isinstance(data.get("expires_at"), str) else data.get("expires_at"),
            status=CommitmentStatus(data.get("status", "pending")),
            metadata=data.get("metadata", {}),
            tool_call_id=data.get("tool_call_id"),
            invocation_id=data.get("invocation_id"),
        )


class ToolCommitment(BaseModel):
    """
    A commitment specifically for tool execution.
    
    This is a convenience class for creating commitments around tool calls.
    
    Attributes:
        tool_name: Name of the tool
        expected_outcome: Description of expected outcome
        max_duration_seconds: Maximum execution time
        must_succeed: Whether the tool must succeed
        output_validation: Optional output validation expression
    """
    
    model_config = ConfigDict(frozen=True)
    
    tool_name: str = Field(..., description="Name of the tool")
    expected_outcome: str = Field(..., description="Description of expected outcome")
    max_duration_seconds: float = Field(
        default=30.0,
        ge=0,
        description="Maximum execution time",
    )
    must_succeed: bool = Field(
        default=True,
        description="Whether the tool must succeed",
    )
    output_validation: str | None = Field(
        default=None,
        description="Optional output validation expression",
    )
    
    def to_runtime_commitment(
        self,
        agent_id: str,
        tool_call_id: str | None = None,
        invocation_id: str | None = None,
        stake: float = 1.0,
        confidence: float = 0.8,
    ) -> RuntimeCommitment:
        """
        Convert to a RuntimeCommitment.
        
        Args:
            agent_id: ID of the agent
            tool_call_id: Associated tool call ID
            invocation_id: Associated invocation ID
            stake: Amount to stake
            confidence: Confidence level
            
        Returns:
            RuntimeCommitment for this tool execution
        """
        # Build success condition
        if self.output_validation:
            success_condition = f"tool_success and ({self.output_validation})"
        else:
            success_condition = "tool_success" if self.must_succeed else "true"
        
        # Build failure conditions
        failure_conditions = {
            "timeout": f"duration_seconds > {self.max_duration_seconds}",
        }
        if self.must_succeed:
            failure_conditions["tool_error"] = "tool_error"
        
        return RuntimeCommitment(
            agent_id=agent_id,
            action_type=f"tool:{self.tool_name}",
            description=f"Execute {self.tool_name}: {self.expected_outcome}",
            trigger_condition=f"tool_name == '{self.tool_name}'",
            success_condition=success_condition,
            failure_conditions=failure_conditions,
            stake=stake,
            confidence=confidence,
            timeout_seconds=self.max_duration_seconds,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=self.max_duration_seconds * 2),
            tool_call_id=tool_call_id,
            invocation_id=invocation_id,
        )


class CommitmentTemplate(BaseModel):
    """
    A reusable commitment template.
    
    Templates allow defining common commitment patterns that can be
    instantiated with specific parameters.
    
    Attributes:
        name: Template name
        description: Template description
        action_type: Type of action
        success_condition_template: Template for success condition
        failure_condition_templates: Templates for failure conditions
        default_stake: Default stake amount
        default_confidence: Default confidence level
        default_timeout_seconds: Default timeout
        parameters: Required parameters for instantiation
    """
    
    model_config = ConfigDict(frozen=True)
    
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    action_type: str = Field(..., description="Type of action")
    
    success_condition_template: str = Field(
        ...,
        description="Template for success condition (use {param} for substitution)",
    )
    failure_condition_templates: dict[str, str] = Field(
        default_factory=dict,
        description="Templates for failure conditions",
    )
    
    default_stake: float = Field(default=1.0, ge=0.0)
    default_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    default_timeout_seconds: float | None = Field(default=None, ge=0)
    
    parameters: list[str] = Field(
        default_factory=list,
        description="Required parameters for instantiation",
    )
    
    def instantiate(
        self,
        agent_id: str,
        params: dict[str, Any],
        stake: float | None = None,
        confidence: float | None = None,
        timeout_seconds: float | None = None,
    ) -> RuntimeCommitment:
        """
        Instantiate this template with parameters.
        
        Args:
            agent_id: ID of the agent
            params: Parameter values for substitution
            stake: Override default stake
            confidence: Override default confidence
            timeout_seconds: Override default timeout
            
        Returns:
            RuntimeCommitment instance
            
        Raises:
            ValueError: If required parameters are missing
        """
        # Check required parameters
        missing = [p for p in self.parameters if p not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
        
        # Substitute parameters in conditions
        success_condition = self.success_condition_template.format(**params)
        failure_conditions = {
            name: template.format(**params)
            for name, template in self.failure_condition_templates.items()
        }
        
        # Build description
        description = self.description.format(**params) if "{" in self.description else self.description
        
        return RuntimeCommitment(
            agent_id=agent_id,
            action_type=self.action_type,
            description=description,
            success_condition=success_condition,
            failure_conditions=failure_conditions,
            stake=stake if stake is not None else self.default_stake,
            confidence=confidence if confidence is not None else self.default_confidence,
            timeout_seconds=timeout_seconds if timeout_seconds is not None else self.default_timeout_seconds,
        )


# Common commitment templates
TOOL_EXECUTION_TEMPLATE = CommitmentTemplate(
    name="tool_execution",
    description="Execute tool {tool_name}",
    action_type="tool_execution",
    success_condition_template="tool_success and tool_name == '{tool_name}'",
    failure_condition_templates={
        "timeout": "duration_seconds > {timeout}",
        "error": "tool_error",
    },
    parameters=["tool_name", "timeout"],
)

RESPONSE_QUALITY_TEMPLATE = CommitmentTemplate(
    name="response_quality",
    description="Provide high-quality response",
    action_type="response",
    success_condition_template="response_quality >= {min_quality}",
    failure_condition_templates={
        "low_quality": "response_quality < {min_quality}",
        "timeout": "response_time_seconds > {max_time}",
    },
    parameters=["min_quality", "max_time"],
)

TASK_COMPLETION_TEMPLATE = CommitmentTemplate(
    name="task_completion",
    description="Complete task: {task_description}",
    action_type="task",
    success_condition_template="task_completed and task_id == '{task_id}'",
    failure_condition_templates={
        "incomplete": "not task_completed and deadline_passed",
        "error": "task_error",
    },
    parameters=["task_id", "task_description"],
)
