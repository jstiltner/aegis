"""
Agent state models for the durable state machine.

This module defines the core state structures that enable:
- Checkpoint/restore functionality
- Deterministic replay
- State versioning and branching
- Event-sourced state management
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict, computed_field

from aegis.core.events import Event, EventType


class MessageRole(str, Enum):
    """Role of a message in the conversation."""
    
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """
    A message in the conversation history.
    
    Messages are immutable records of the conversation between
    the user, assistant, and tools.
    """
    
    model_config = ConfigDict(frozen=True)
    
    message_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this message",
    )
    role: MessageRole = Field(
        ...,
        description="Role of the message sender",
    )
    content: str = Field(
        ...,
        description="Message content",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the message was created",
    )
    
    # Tool-specific fields
    tool_call_id: str | None = Field(
        default=None,
        description="ID of the tool call this message responds to",
    )
    tool_name: str | None = Field(
        default=None,
        description="Name of the tool (for tool messages)",
    )
    
    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional message metadata",
    )
    
    @classmethod
    def user(cls, content: str, **kwargs: Any) -> Message:
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content, **kwargs)
    
    @classmethod
    def assistant(cls, content: str, **kwargs: Any) -> Message:
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content, **kwargs)
    
    @classmethod
    def system(cls, content: str, **kwargs: Any) -> Message:
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content, **kwargs)
    
    @classmethod
    def tool(
        cls,
        content: str,
        tool_call_id: str,
        tool_name: str,
        **kwargs: Any,
    ) -> Message:
        """Create a tool response message."""
        return cls(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            **kwargs,
        )


class ToolCall(BaseModel):
    """
    A pending tool call that needs to be executed.
    
    Tool calls are created when the LLM requests a tool invocation
    and are resolved when the tool returns a result.
    """
    
    model_config = ConfigDict(frozen=True)
    
    tool_call_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this tool call",
    )
    tool_name: str = Field(
        ...,
        description="Name of the tool to invoke",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the tool",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the tool call was created",
    )
    
    # GCL integration
    commitment_id: str | None = Field(
        default=None,
        description="ID of the commitment associated with this tool call",
    )


class AgentStatus(str, Enum):
    """Current status of the agent."""
    
    INITIALIZING = "initializing"
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING_TOOL = "executing_tool"
    AWAITING_INPUT = "awaiting_input"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"


class AgentState(BaseModel):
    """
    Complete state of an agent at a point in time.
    
    AgentState is the fundamental unit of persistence. It captures
    everything needed to resume an agent from a checkpoint.
    
    Key design principles:
    - Immutable: State is never modified, only replaced
    - Serializable: Can be persisted to storage
    - Versioned: Each state has a version number
    - Traceable: Links to parent checkpoint for history
    
    Attributes:
        agent_id: Unique identifier for the agent
        session_id: Current session identifier
        version: Monotonically increasing version number
        checkpoint_id: ID of the checkpoint this state represents
        
        status: Current agent status
        conversation_history: List of messages in the conversation
        working_memory: Arbitrary key-value storage for agent use
        active_commitments: List of active GCL commitment IDs
        
        current_step: Current step in the execution
        pending_tool_calls: Tool calls waiting to be executed
        
        created_at: When this state was created
        updated_at: When this state was last updated
        parent_checkpoint_id: ID of the checkpoint this branched from
    """
    
    model_config = ConfigDict(frozen=True)
    
    # Identity
    agent_id: str = Field(
        ...,
        description="Unique identifier for the agent",
    )
    session_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Current session identifier",
    )
    version: int = Field(
        default=0,
        ge=0,
        description="Monotonically increasing version number",
    )
    checkpoint_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="ID of the checkpoint this state represents",
    )
    
    # Status
    status: AgentStatus = Field(
        default=AgentStatus.INITIALIZING,
        description="Current agent status",
    )
    
    # Conversation
    conversation_history: tuple[Message, ...] = Field(
        default_factory=tuple,
        description="List of messages in the conversation",
    )
    
    # Memory
    working_memory: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value storage for agent use",
    )
    
    # GCL integration
    active_commitments: tuple[str, ...] = Field(
        default_factory=tuple,
        description="List of active GCL commitment IDs",
    )
    
    # Execution state
    current_step: int = Field(
        default=0,
        ge=0,
        description="Current step in the execution",
    )
    pending_tool_calls: tuple[ToolCall, ...] = Field(
        default_factory=tuple,
        description="Tool calls waiting to be executed",
    )
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this state was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this state was last updated",
    )
    
    # Lineage
    parent_checkpoint_id: str | None = Field(
        default=None,
        description="ID of the checkpoint this branched from",
    )
    
    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional state metadata",
    )
    
    @computed_field
    @property
    def message_count(self) -> int:
        """Number of messages in the conversation."""
        return len(self.conversation_history)
    
    @computed_field
    @property
    def has_pending_tools(self) -> bool:
        """Whether there are pending tool calls."""
        return len(self.pending_tool_calls) > 0
    
    def with_message(self, message: Message) -> AgentState:
        """Create a new state with an additional message."""
        return self.model_copy(
            update={
                "conversation_history": (*self.conversation_history, message),
                "version": self.version + 1,
                "updated_at": datetime.now(timezone.utc),
            }
        )
    
    def with_messages(self, messages: list[Message]) -> AgentState:
        """Create a new state with additional messages."""
        return self.model_copy(
            update={
                "conversation_history": (*self.conversation_history, *messages),
                "version": self.version + 1,
                "updated_at": datetime.now(timezone.utc),
            }
        )
    
    def with_status(self, status: AgentStatus) -> AgentState:
        """Create a new state with updated status."""
        return self.model_copy(
            update={
                "status": status,
                "version": self.version + 1,
                "updated_at": datetime.now(timezone.utc),
            }
        )
    
    def with_tool_calls(self, tool_calls: list[ToolCall]) -> AgentState:
        """Create a new state with pending tool calls."""
        return self.model_copy(
            update={
                "pending_tool_calls": tuple(tool_calls),
                "status": AgentStatus.EXECUTING_TOOL if tool_calls else self.status,
                "version": self.version + 1,
                "updated_at": datetime.now(timezone.utc),
            }
        )
    
    def with_tool_call_resolved(self, tool_call_id: str) -> AgentState:
        """Create a new state with a tool call removed."""
        remaining = tuple(
            tc for tc in self.pending_tool_calls
            if tc.tool_call_id != tool_call_id
        )
        return self.model_copy(
            update={
                "pending_tool_calls": remaining,
                "status": AgentStatus.THINKING if not remaining else self.status,
                "version": self.version + 1,
                "updated_at": datetime.now(timezone.utc),
            }
        )
    
    def with_memory(self, key: str, value: Any) -> AgentState:
        """Create a new state with updated working memory."""
        new_memory = {**self.working_memory, key: value}
        return self.model_copy(
            update={
                "working_memory": new_memory,
                "version": self.version + 1,
                "updated_at": datetime.now(timezone.utc),
            }
        )
    
    def with_commitment(self, commitment_id: str) -> AgentState:
        """Create a new state with an active commitment."""
        return self.model_copy(
            update={
                "active_commitments": (*self.active_commitments, commitment_id),
                "version": self.version + 1,
                "updated_at": datetime.now(timezone.utc),
            }
        )
    
    def with_commitment_resolved(self, commitment_id: str) -> AgentState:
        """Create a new state with a commitment removed."""
        remaining = tuple(
            c for c in self.active_commitments
            if c != commitment_id
        )
        return self.model_copy(
            update={
                "active_commitments": remaining,
                "version": self.version + 1,
                "updated_at": datetime.now(timezone.utc),
            }
        )
    
    def with_step_increment(self) -> AgentState:
        """Create a new state with incremented step counter."""
        return self.model_copy(
            update={
                "current_step": self.current_step + 1,
                "version": self.version + 1,
                "updated_at": datetime.now(timezone.utc),
            }
        )
    
    def as_checkpoint(self, checkpoint_id: str | None = None) -> AgentState:
        """Create a checkpoint from this state."""
        return self.model_copy(
            update={
                "checkpoint_id": checkpoint_id or str(uuid4()),
                "parent_checkpoint_id": self.checkpoint_id,
            }
        )
    
    def get_last_message(self, role: MessageRole | None = None) -> Message | None:
        """Get the last message, optionally filtered by role."""
        for message in reversed(self.conversation_history):
            if role is None or message.role == role:
                return message
        return None
    
    def get_messages_for_llm(self) -> list[dict[str, Any]]:
        """Get messages formatted for LLM API calls."""
        messages = []
        for msg in self.conversation_history:
            if msg.role == MessageRole.TOOL:
                messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                })
            else:
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })
        return messages


class StateTransition(BaseModel):
    """
    Record of a state transition.
    
    StateTransitions form the event log that enables:
    - Deterministic replay
    - Audit trail
    - Causal tracing
    
    Each transition records:
    - The event that caused the transition
    - The before and after checkpoint IDs
    - Timing information
    - Any commitments affected
    """
    
    model_config = ConfigDict(frozen=True)
    
    transition_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this transition",
    )
    
    # State references
    from_checkpoint_id: str = Field(
        ...,
        description="Checkpoint ID before the transition",
    )
    to_checkpoint_id: str = Field(
        ...,
        description="Checkpoint ID after the transition",
    )
    from_version: int = Field(
        ...,
        ge=0,
        description="State version before the transition",
    )
    to_version: int = Field(
        ...,
        ge=0,
        description="State version after the transition",
    )
    
    # Event that caused this transition
    event: Event = Field(
        ...,
        description="The event that caused this transition",
    )
    
    # Timing
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the transition occurred",
    )
    duration_ms: float | None = Field(
        default=None,
        ge=0,
        description="How long the transition took in milliseconds",
    )
    
    # GCL integration
    commitments_issued: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Commitment IDs issued during this transition",
    )
    commitments_verified: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Commitment IDs verified during this transition",
    )
    
    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional transition metadata",
    )
    
    @property
    def event_type(self) -> EventType:
        """Get the type of event that caused this transition."""
        return self.event.event_type


class StateDiff(BaseModel):
    """
    Difference between two agent states.
    
    Used for debugging and understanding state changes.
    """
    
    model_config = ConfigDict(frozen=True)
    
    from_version: int = Field(..., description="Version of the source state")
    to_version: int = Field(..., description="Version of the target state")
    
    status_changed: bool = Field(default=False)
    status_from: AgentStatus | None = None
    status_to: AgentStatus | None = None
    
    messages_added: int = Field(default=0)
    messages_removed: int = Field(default=0)
    
    memory_keys_added: tuple[str, ...] = Field(default_factory=tuple)
    memory_keys_removed: tuple[str, ...] = Field(default_factory=tuple)
    memory_keys_changed: tuple[str, ...] = Field(default_factory=tuple)
    
    commitments_added: tuple[str, ...] = Field(default_factory=tuple)
    commitments_removed: tuple[str, ...] = Field(default_factory=tuple)
    
    tool_calls_added: int = Field(default=0)
    tool_calls_removed: int = Field(default=0)
    
    @classmethod
    def compute(cls, from_state: AgentState, to_state: AgentState) -> StateDiff:
        """Compute the difference between two states."""
        # Status
        status_changed = from_state.status != to_state.status
        
        # Messages
        from_msg_ids = {m.message_id for m in from_state.conversation_history}
        to_msg_ids = {m.message_id for m in to_state.conversation_history}
        messages_added = len(to_msg_ids - from_msg_ids)
        messages_removed = len(from_msg_ids - to_msg_ids)
        
        # Memory
        from_keys = set(from_state.working_memory.keys())
        to_keys = set(to_state.working_memory.keys())
        memory_keys_added = tuple(to_keys - from_keys)
        memory_keys_removed = tuple(from_keys - to_keys)
        memory_keys_changed = tuple(
            k for k in from_keys & to_keys
            if from_state.working_memory[k] != to_state.working_memory.get(k)
        )
        
        # Commitments
        from_commits = set(from_state.active_commitments)
        to_commits = set(to_state.active_commitments)
        commitments_added = tuple(to_commits - from_commits)
        commitments_removed = tuple(from_commits - to_commits)
        
        # Tool calls
        from_tc_ids = {tc.tool_call_id for tc in from_state.pending_tool_calls}
        to_tc_ids = {tc.tool_call_id for tc in to_state.pending_tool_calls}
        tool_calls_added = len(to_tc_ids - from_tc_ids)
        tool_calls_removed = len(from_tc_ids - to_tc_ids)
        
        return cls(
            from_version=from_state.version,
            to_version=to_state.version,
            status_changed=status_changed,
            status_from=from_state.status if status_changed else None,
            status_to=to_state.status if status_changed else None,
            messages_added=messages_added,
            messages_removed=messages_removed,
            memory_keys_added=memory_keys_added,
            memory_keys_removed=memory_keys_removed,
            memory_keys_changed=memory_keys_changed,
            commitments_added=commitments_added,
            commitments_removed=commitments_removed,
            tool_calls_added=tool_calls_added,
            tool_calls_removed=tool_calls_removed,
        )
    
    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return (
            self.status_changed
            or self.messages_added > 0
            or self.messages_removed > 0
            or len(self.memory_keys_added) > 0
            or len(self.memory_keys_removed) > 0
            or len(self.memory_keys_changed) > 0
            or len(self.commitments_added) > 0
            or len(self.commitments_removed) > 0
            or self.tool_calls_added > 0
            or self.tool_calls_removed > 0
        )
