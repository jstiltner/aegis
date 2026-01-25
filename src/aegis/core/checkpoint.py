"""
Checkpoint manager for durable agent state.

This module provides checkpoint functionality that enables:
- Saving agent state at any point
- Restoring from any checkpoint
- Branching execution from checkpoints
- State versioning and history
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from aegis.core.state import AgentState, StateTransition
from aegis.core.events import Event, EventType


class Checkpoint(BaseModel):
    """
    A checkpoint represents a saved agent state.
    
    Checkpoints are the unit of persistence and recovery.
    They contain:
    - The complete agent state
    - Metadata about when/why the checkpoint was created
    - Links to parent checkpoints for history
    """
    
    model_config = ConfigDict(frozen=True)
    
    checkpoint_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this checkpoint",
    )
    agent_id: str = Field(
        ...,
        description="ID of the agent this checkpoint belongs to",
    )
    session_id: str = Field(
        ...,
        description="Session ID when checkpoint was created",
    )
    
    # State
    state: AgentState = Field(
        ...,
        description="The complete agent state at this checkpoint",
    )
    
    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this checkpoint was created",
    )
    reason: str = Field(
        default="manual",
        description="Why this checkpoint was created",
    )
    
    # Lineage
    parent_checkpoint_id: str | None = Field(
        default=None,
        description="ID of the parent checkpoint",
    )
    
    # Events since parent
    events_since_parent: tuple[Event, ...] = Field(
        default_factory=tuple,
        description="Events that occurred since the parent checkpoint",
    )
    
    # Tags for organization
    tags: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Tags for organizing checkpoints",
    )
    
    @property
    def version(self) -> int:
        """Get the state version at this checkpoint."""
        return self.state.version
    
    @property
    def event_count(self) -> int:
        """Number of events since parent checkpoint."""
        return len(self.events_since_parent)


class CheckpointStorage(Protocol):
    """Protocol for checkpoint storage backends."""
    
    async def save(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint to storage."""
        ...
    
    async def load(self, checkpoint_id: str) -> Checkpoint | None:
        """Load a checkpoint from storage."""
        ...
    
    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint from storage."""
        ...
    
    async def list_checkpoints(
        self,
        agent_id: str,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[Checkpoint]:
        """List checkpoints for an agent."""
        ...
    
    async def get_latest(
        self,
        agent_id: str,
        session_id: str | None = None,
    ) -> Checkpoint | None:
        """Get the most recent checkpoint for an agent."""
        ...


class InMemoryCheckpointStorage:
    """
    In-memory checkpoint storage for testing and development.
    
    Not suitable for production use as checkpoints are lost on restart.
    """
    
    def __init__(self) -> None:
        self._checkpoints: dict[str, Checkpoint] = {}
    
    async def save(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint to memory."""
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
    
    async def load(self, checkpoint_id: str) -> Checkpoint | None:
        """Load a checkpoint from memory."""
        return self._checkpoints.get(checkpoint_id)
    
    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint from memory."""
        if checkpoint_id in self._checkpoints:
            del self._checkpoints[checkpoint_id]
            return True
        return False
    
    async def list_checkpoints(
        self,
        agent_id: str,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[Checkpoint]:
        """List checkpoints for an agent."""
        checkpoints = [
            cp for cp in self._checkpoints.values()
            if cp.agent_id == agent_id
            and (session_id is None or cp.session_id == session_id)
        ]
        # Sort by creation time, newest first
        checkpoints.sort(key=lambda cp: cp.created_at, reverse=True)
        return checkpoints[:limit]
    
    async def get_latest(
        self,
        agent_id: str,
        session_id: str | None = None,
    ) -> Checkpoint | None:
        """Get the most recent checkpoint for an agent."""
        checkpoints = await self.list_checkpoints(agent_id, session_id, limit=1)
        return checkpoints[0] if checkpoints else None
    
    def clear(self) -> None:
        """Clear all checkpoints (for testing)."""
        self._checkpoints.clear()


class FileCheckpointStorage:
    """
    File-based checkpoint storage.
    
    Stores checkpoints as JSON files in a directory structure:
    base_path/
      {agent_id}/
        {session_id}/
          {checkpoint_id}.json
    """
    
    def __init__(self, base_path: Path | str) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _checkpoint_path(self, checkpoint: Checkpoint) -> Path:
        """Get the file path for a checkpoint."""
        return (
            self.base_path
            / checkpoint.agent_id
            / checkpoint.session_id
            / f"{checkpoint.checkpoint_id}.json"
        )
    
    def _find_checkpoint_file(self, checkpoint_id: str) -> Path | None:
        """Find a checkpoint file by ID (searches all directories)."""
        for agent_dir in self.base_path.iterdir():
            if not agent_dir.is_dir():
                continue
            for session_dir in agent_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                checkpoint_file = session_dir / f"{checkpoint_id}.json"
                if checkpoint_file.exists():
                    return checkpoint_file
        return None
    
    async def save(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint to a file."""
        path = self._checkpoint_path(checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Serialize to JSON
        data = checkpoint.model_dump_json(indent=2)
        path.write_text(data)
    
    async def load(self, checkpoint_id: str) -> Checkpoint | None:
        """Load a checkpoint from a file."""
        path = self._find_checkpoint_file(checkpoint_id)
        if path is None or not path.exists():
            return None
        
        data = path.read_text()
        return Checkpoint.model_validate_json(data)
    
    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint file."""
        path = self._find_checkpoint_file(checkpoint_id)
        if path is None or not path.exists():
            return False
        
        path.unlink()
        return True
    
    async def list_checkpoints(
        self,
        agent_id: str,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[Checkpoint]:
        """List checkpoints for an agent."""
        agent_dir = self.base_path / agent_id
        if not agent_dir.exists():
            return []
        
        checkpoints: list[Checkpoint] = []
        
        session_dirs = (
            [agent_dir / session_id] if session_id
            else list(agent_dir.iterdir())
        )
        
        for session_dir in session_dirs:
            if not session_dir.is_dir():
                continue
            
            for checkpoint_file in session_dir.glob("*.json"):
                try:
                    data = checkpoint_file.read_text()
                    checkpoint = Checkpoint.model_validate_json(data)
                    checkpoints.append(checkpoint)
                except Exception:
                    # Skip invalid checkpoint files
                    continue
        
        # Sort by creation time, newest first
        checkpoints.sort(key=lambda cp: cp.created_at, reverse=True)
        return checkpoints[:limit]
    
    async def get_latest(
        self,
        agent_id: str,
        session_id: str | None = None,
    ) -> Checkpoint | None:
        """Get the most recent checkpoint for an agent."""
        checkpoints = await self.list_checkpoints(agent_id, session_id, limit=1)
        return checkpoints[0] if checkpoints else None


class CheckpointManager:
    """
    Manager for creating, storing, and restoring checkpoints.
    
    The CheckpointManager provides a high-level interface for:
    - Creating checkpoints from agent state
    - Restoring agent state from checkpoints
    - Managing checkpoint history
    - Branching execution from checkpoints
    
    Example:
        >>> manager = CheckpointManager(InMemoryCheckpointStorage())
        >>> 
        >>> # Create a checkpoint
        >>> checkpoint = await manager.create_checkpoint(
        ...     state=agent_state,
        ...     reason="before_tool_call",
        ... )
        >>> 
        >>> # Later, restore from checkpoint
        >>> restored_state = await manager.restore(checkpoint.checkpoint_id)
    """
    
    def __init__(
        self,
        storage: CheckpointStorage,
        auto_checkpoint_interval: int | None = None,
    ) -> None:
        """
        Initialize the checkpoint manager.
        
        Args:
            storage: Backend storage for checkpoints
            auto_checkpoint_interval: If set, automatically create checkpoints
                every N state transitions
        """
        self.storage = storage
        self.auto_checkpoint_interval = auto_checkpoint_interval
        
        # Track events since last checkpoint
        self._pending_events: list[Event] = []
        self._last_checkpoint_id: str | None = None
        self._transitions_since_checkpoint: int = 0
    
    async def create_checkpoint(
        self,
        state: AgentState,
        reason: str = "manual",
        tags: list[str] | None = None,
    ) -> Checkpoint:
        """
        Create a checkpoint from the current agent state.
        
        Args:
            state: The agent state to checkpoint
            reason: Why this checkpoint is being created
            tags: Optional tags for organizing checkpoints
            
        Returns:
            The created checkpoint
        """
        checkpoint = Checkpoint(
            agent_id=state.agent_id,
            session_id=state.session_id,
            state=state,
            reason=reason,
            parent_checkpoint_id=self._last_checkpoint_id,
            events_since_parent=tuple(self._pending_events),
            tags=tuple(tags or []),
        )
        
        await self.storage.save(checkpoint)
        
        # Reset tracking
        self._pending_events = []
        self._last_checkpoint_id = checkpoint.checkpoint_id
        self._transitions_since_checkpoint = 0
        
        return checkpoint
    
    async def restore(self, checkpoint_id: str) -> AgentState | None:
        """
        Restore agent state from a checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to restore
            
        Returns:
            The restored agent state, or None if checkpoint not found
        """
        checkpoint = await self.storage.load(checkpoint_id)
        if checkpoint is None:
            return None
        
        # Update tracking
        self._last_checkpoint_id = checkpoint_id
        self._pending_events = []
        self._transitions_since_checkpoint = 0
        
        return checkpoint.state
    
    async def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """Get a checkpoint by ID."""
        return await self.storage.load(checkpoint_id)
    
    async def get_latest_checkpoint(
        self,
        agent_id: str,
        session_id: str | None = None,
    ) -> Checkpoint | None:
        """Get the most recent checkpoint for an agent."""
        return await self.storage.get_latest(agent_id, session_id)
    
    async def list_checkpoints(
        self,
        agent_id: str,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[Checkpoint]:
        """List checkpoints for an agent."""
        return await self.storage.list_checkpoints(agent_id, session_id, limit)
    
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        return await self.storage.delete(checkpoint_id)
    
    async def branch_from(
        self,
        checkpoint_id: str,
        new_session_id: str | None = None,
    ) -> AgentState | None:
        """
        Create a new execution branch from a checkpoint.
        
        This restores the state from the checkpoint but assigns
        a new session ID, creating a separate execution branch.
        
        Args:
            checkpoint_id: ID of the checkpoint to branch from
            new_session_id: Optional new session ID (generated if not provided)
            
        Returns:
            The branched agent state, or None if checkpoint not found
        """
        checkpoint = await self.storage.load(checkpoint_id)
        if checkpoint is None:
            return None
        
        # Create new session ID if not provided
        session_id = new_session_id or str(uuid4())
        
        # Create branched state
        branched_state = checkpoint.state.model_copy(
            update={
                "session_id": session_id,
                "parent_checkpoint_id": checkpoint_id,
                "version": 0,  # Reset version for new branch
            }
        )
        
        # Reset tracking for new branch
        self._last_checkpoint_id = checkpoint_id
        self._pending_events = []
        self._transitions_since_checkpoint = 0
        
        return branched_state
    
    def record_event(self, event: Event) -> None:
        """
        Record an event for inclusion in the next checkpoint.
        
        Args:
            event: The event to record
        """
        self._pending_events.append(event)
    
    def record_transition(self, transition: StateTransition) -> None:
        """
        Record a state transition.
        
        This tracks transitions for auto-checkpointing.
        
        Args:
            transition: The state transition to record
        """
        self._pending_events.append(transition.event)
        self._transitions_since_checkpoint += 1
    
    def should_auto_checkpoint(self) -> bool:
        """Check if an automatic checkpoint should be created."""
        if self.auto_checkpoint_interval is None:
            return False
        return self._transitions_since_checkpoint >= self.auto_checkpoint_interval
    
    async def get_checkpoint_history(
        self,
        checkpoint_id: str,
        max_depth: int = 10,
    ) -> list[Checkpoint]:
        """
        Get the history of checkpoints leading to a given checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to get history for
            max_depth: Maximum number of ancestors to retrieve
            
        Returns:
            List of checkpoints from oldest to newest
        """
        history: list[Checkpoint] = []
        current_id: str | None = checkpoint_id
        
        while current_id is not None and len(history) < max_depth:
            checkpoint = await self.storage.load(current_id)
            if checkpoint is None:
                break
            
            history.append(checkpoint)
            current_id = checkpoint.parent_checkpoint_id
        
        # Reverse to get oldest first
        history.reverse()
        return history
    
    async def get_events_between(
        self,
        from_checkpoint_id: str,
        to_checkpoint_id: str,
    ) -> list[Event]:
        """
        Get all events between two checkpoints.
        
        Args:
            from_checkpoint_id: Starting checkpoint ID
            to_checkpoint_id: Ending checkpoint ID
            
        Returns:
            List of events in chronological order
        """
        # Get the path from from_checkpoint to to_checkpoint
        to_checkpoint = await self.storage.load(to_checkpoint_id)
        if to_checkpoint is None:
            return []
        
        events: list[Event] = []
        current_id: str | None = to_checkpoint_id
        
        while current_id is not None and current_id != from_checkpoint_id:
            checkpoint = await self.storage.load(current_id)
            if checkpoint is None:
                break
            
            # Prepend events (we're walking backwards)
            events = list(checkpoint.events_since_parent) + events
            current_id = checkpoint.parent_checkpoint_id
        
        return events


class CheckpointPolicy(BaseModel):
    """
    Policy for automatic checkpoint creation.
    
    Defines when checkpoints should be automatically created.
    """
    
    model_config = ConfigDict(frozen=True)
    
    # Interval-based checkpointing
    transition_interval: int | None = Field(
        default=None,
        ge=1,
        description="Create checkpoint every N transitions",
    )
    time_interval_seconds: float | None = Field(
        default=None,
        ge=1.0,
        description="Create checkpoint every N seconds",
    )
    
    # Event-based checkpointing
    checkpoint_on_tool_call: bool = Field(
        default=True,
        description="Create checkpoint before each tool call",
    )
    checkpoint_on_user_input: bool = Field(
        default=True,
        description="Create checkpoint after each user input",
    )
    checkpoint_on_llm_response: bool = Field(
        default=False,
        description="Create checkpoint after each LLM response",
    )
    checkpoint_on_commitment: bool = Field(
        default=True,
        description="Create checkpoint when commitments are issued/verified",
    )
    
    # Retention
    max_checkpoints_per_session: int = Field(
        default=100,
        ge=1,
        description="Maximum checkpoints to retain per session",
    )
    
    def should_checkpoint(
        self,
        event: Event,
        transitions_since_last: int,
        seconds_since_last: float,
    ) -> bool:
        """
        Determine if a checkpoint should be created.
        
        Args:
            event: The event that just occurred
            transitions_since_last: Number of transitions since last checkpoint
            seconds_since_last: Seconds since last checkpoint
            
        Returns:
            True if a checkpoint should be created
        """
        # Check interval-based triggers
        if (
            self.transition_interval is not None
            and transitions_since_last >= self.transition_interval
        ):
            return True
        
        if (
            self.time_interval_seconds is not None
            and seconds_since_last >= self.time_interval_seconds
        ):
            return True
        
        # Check event-based triggers
        if self.checkpoint_on_tool_call and event.event_type == EventType.TOOL_REQUEST:
            return True
        
        if self.checkpoint_on_user_input and event.event_type == EventType.USER_INPUT:
            return True
        
        if self.checkpoint_on_llm_response and event.event_type == EventType.LLM_RESPONSE:
            return True
        
        if self.checkpoint_on_commitment and event.event_type in (
            EventType.COMMITMENT_ISSUED,
            EventType.COMMITMENT_VERIFIED,
        ):
            return True
        
        return False
