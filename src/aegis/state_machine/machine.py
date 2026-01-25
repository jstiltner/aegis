"""
Durable state machine for agent execution.

This module provides the core state machine that:
- Manages agent state transitions
- Integrates with checkpointing for durability
- Supports event sourcing for replay
- Emits events for audit logging
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from aegis.core.state import AgentState, AgentStatus, StateTransition
from aegis.core.events import Event, EventType, EventFactory
from aegis.core.checkpoint import (
    CheckpointManager,
    CheckpointPolicy,
    Checkpoint,
    InMemoryCheckpointStorage,
)
from aegis.state_machine.transitions import (
    TransitionHandler,
    TransitionResult,
    TransitionStatus,
)


class StateMachineConfig(BaseModel):
    """Configuration for the state machine."""
    
    model_config = ConfigDict(frozen=True)
    
    # Checkpointing
    auto_checkpoint: bool = Field(
        default=True,
        description="Whether to automatically create checkpoints",
    )
    checkpoint_policy: CheckpointPolicy = Field(
        default_factory=CheckpointPolicy,
        description="Policy for automatic checkpointing",
    )
    
    # Validation
    validate_transitions: bool = Field(
        default=True,
        description="Whether to validate state transitions",
    )
    
    # Limits
    max_transitions_per_session: int = Field(
        default=10000,
        ge=1,
        description="Maximum transitions allowed per session",
    )
    max_events_in_memory: int = Field(
        default=1000,
        ge=1,
        description="Maximum events to keep in memory",
    )
    
    # Timeouts
    transition_timeout_seconds: float = Field(
        default=30.0,
        ge=0.1,
        description="Timeout for individual transitions",
    )


# Type for state change listeners
StateChangeListener = Callable[[AgentState, AgentState, Event], Awaitable[None]]


class StateMachine:
    """
    Durable state machine for agent execution.
    
    The StateMachine is the core runtime component that:
    - Manages the current agent state
    - Processes events to produce state transitions
    - Integrates with checkpointing for durability
    - Supports replay from any checkpoint
    - Emits events for audit logging
    
    Key features:
    - Event-sourced: All state changes are driven by events
    - Durable: State can be checkpointed and restored
    - Validated: Transitions are validated before application
    - Observable: Listeners can observe state changes
    
    Example:
        >>> # Create a state machine
        >>> machine = StateMachine(
        ...     agent_id="agent-1",
        ...     checkpoint_manager=checkpoint_manager,
        ... )
        >>> 
        >>> # Initialize
        >>> await machine.initialize()
        >>> 
        >>> # Process events
        >>> result = await machine.process_event(user_input_event)
        >>> if result.is_success:
        ...     print(f"New state: {machine.current_state}")
        >>> 
        >>> # Checkpoint
        >>> checkpoint = await machine.checkpoint("before_tool_call")
        >>> 
        >>> # Later, restore
        >>> await machine.restore(checkpoint.checkpoint_id)
    """
    
    def __init__(
        self,
        agent_id: str,
        checkpoint_manager: CheckpointManager | None = None,
        config: StateMachineConfig | None = None,
        initial_state: AgentState | None = None,
    ) -> None:
        """
        Initialize the state machine.
        
        Args:
            agent_id: Unique identifier for the agent
            checkpoint_manager: Manager for checkpoints (uses in-memory if not provided)
            config: Configuration options
            initial_state: Optional initial state (created if not provided)
        """
        self.agent_id = agent_id
        self.config = config or StateMachineConfig()
        
        # Checkpoint manager
        if checkpoint_manager is None:
            storage = InMemoryCheckpointStorage()
            checkpoint_manager = CheckpointManager(storage)
        self.checkpoint_manager = checkpoint_manager
        
        # Transition handler
        self._transition_handler = TransitionHandler()
        
        # State
        self._session_id = str(uuid4())
        self._current_state = initial_state or AgentState(
            agent_id=agent_id,
            session_id=self._session_id,
        )
        
        # Event tracking
        self._event_factory = EventFactory(agent_id, self._session_id)
        self._events: list[Event] = []
        self._transitions: list[StateTransition] = []
        self._transition_count = 0
        
        # Checkpointing
        self._last_checkpoint_time = datetime.now(timezone.utc)
        self._transitions_since_checkpoint = 0
        
        # Listeners
        self._state_change_listeners: list[StateChangeListener] = []
        
        # Control
        self._lock = asyncio.Lock()
        self._initialized = False
    
    @property
    def current_state(self) -> AgentState:
        """Get the current agent state."""
        return self._current_state
    
    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        return self._session_id
    
    @property
    def is_initialized(self) -> bool:
        """Check if the state machine is initialized."""
        return self._initialized
    
    @property
    def transition_count(self) -> int:
        """Get the total number of transitions in this session."""
        return self._transition_count
    
    @property
    def events(self) -> list[Event]:
        """Get all events in this session."""
        return self._events.copy()
    
    @property
    def transitions(self) -> list[StateTransition]:
        """Get all transitions in this session."""
        return self._transitions.copy()
    
    def add_state_change_listener(self, listener: StateChangeListener) -> None:
        """Add a listener for state changes."""
        self._state_change_listeners.append(listener)
    
    def remove_state_change_listener(self, listener: StateChangeListener) -> None:
        """Remove a state change listener."""
        self._state_change_listeners.remove(listener)
    
    async def initialize(self) -> AgentState:
        """
        Initialize the state machine.
        
        This emits an AGENT_START event and transitions to IDLE status.
        
        Returns:
            The initial agent state
        """
        async with self._lock:
            if self._initialized:
                return self._current_state
            
            # Emit start event
            start_event = self._event_factory.agent_start(
                config={"agent_id": self.agent_id}
            )
            
            # Apply the event
            result = await self._apply_event_internal(start_event)
            if not result.is_success:
                raise RuntimeError(f"Failed to initialize: {result.error}")
            
            self._initialized = True
            
            # Create initial checkpoint
            if self.config.auto_checkpoint:
                await self._create_checkpoint_internal("initialization")
            
            return self._current_state
    
    async def process_event(self, event: Event) -> TransitionResult:
        """
        Process an event and update state.
        
        This is the main entry point for state changes. It:
        1. Validates the transition
        2. Applies the event to produce new state
        3. Records the transition
        4. Notifies listeners
        5. Creates checkpoints if needed
        
        Args:
            event: The event to process
            
        Returns:
            TransitionResult with new state or error information
        """
        async with self._lock:
            if not self._initialized:
                raise RuntimeError("State machine not initialized")
            
            # Check transition limit
            if self._transition_count >= self.config.max_transitions_per_session:
                return TransitionResult(
                    status=TransitionStatus.REJECTED,
                    validation_errors=[
                        f"Maximum transitions ({self.config.max_transitions_per_session}) reached"
                    ],
                )
            
            # Apply the event
            result = await self._apply_event_internal(event)
            
            # Handle auto-checkpointing
            if result.is_success and self.config.auto_checkpoint:
                await self._maybe_auto_checkpoint(event)
            
            return result
    
    async def _apply_event_internal(self, event: Event) -> TransitionResult:
        """Internal method to apply an event."""
        old_state = self._current_state
        
        # Apply through transition handler
        result = await self._transition_handler.apply(
            self._current_state,
            event,
            skip_validation=not self.config.validate_transitions,
        )
        
        if result.is_success and result.new_state is not None:
            # Update state
            self._current_state = result.new_state
            self._transition_count += 1
            self._transitions_since_checkpoint += 1
            
            # Record event and transition
            self._events.append(event)
            if result.transition:
                self._transitions.append(result.transition)
            
            # Trim events if needed
            if len(self._events) > self.config.max_events_in_memory:
                self._events = self._events[-self.config.max_events_in_memory:]
            
            # Record in checkpoint manager
            self.checkpoint_manager.record_event(event)
            
            # Notify listeners
            await self._notify_listeners(old_state, self._current_state, event)
        
        return result
    
    async def _notify_listeners(
        self,
        old_state: AgentState,
        new_state: AgentState,
        event: Event,
    ) -> None:
        """Notify all state change listeners."""
        for listener in self._state_change_listeners:
            try:
                await listener(old_state, new_state, event)
            except Exception:
                # Don't let listener errors break the state machine
                pass
    
    async def _maybe_auto_checkpoint(self, event: Event) -> None:
        """Create a checkpoint if the policy says we should."""
        seconds_since_last = (
            datetime.now(timezone.utc) - self._last_checkpoint_time
        ).total_seconds()
        
        if self.config.checkpoint_policy.should_checkpoint(
            event,
            self._transitions_since_checkpoint,
            seconds_since_last,
        ):
            await self._create_checkpoint_internal(f"auto_{event.event_type.value}")
    
    async def checkpoint(self, reason: str = "manual") -> Checkpoint:
        """
        Create a checkpoint of the current state.
        
        Args:
            reason: Why this checkpoint is being created
            
        Returns:
            The created checkpoint
        """
        async with self._lock:
            return await self._create_checkpoint_internal(reason)
    
    async def _create_checkpoint_internal(self, reason: str) -> Checkpoint:
        """Internal method to create a checkpoint."""
        checkpoint = await self.checkpoint_manager.create_checkpoint(
            state=self._current_state,
            reason=reason,
        )
        
        # Emit checkpoint event
        checkpoint_event = self._event_factory.checkpoint_create(
            checkpoint_id=checkpoint.checkpoint_id,
            state_summary={
                "version": self._current_state.version,
                "status": self._current_state.status.value,
                "message_count": self._current_state.message_count,
            },
        )
        self._events.append(checkpoint_event)
        
        # Update tracking
        self._last_checkpoint_time = datetime.now(timezone.utc)
        self._transitions_since_checkpoint = 0
        
        return checkpoint
    
    async def restore(self, checkpoint_id: str) -> AgentState | None:
        """
        Restore state from a checkpoint.
        
        This resets the state machine to the checkpoint state.
        A new session is started from the restored state.
        
        Args:
            checkpoint_id: ID of the checkpoint to restore
            
        Returns:
            The restored state, or None if checkpoint not found
        """
        async with self._lock:
            state = await self.checkpoint_manager.restore(checkpoint_id)
            if state is None:
                return None
            
            # Start new session from restored state
            self._session_id = str(uuid4())
            self._current_state = state.model_copy(
                update={"session_id": self._session_id}
            )
            
            # Reset tracking
            self._event_factory = EventFactory(self.agent_id, self._session_id)
            self._events = []
            self._transitions = []
            self._transition_count = 0
            self._transitions_since_checkpoint = 0
            self._last_checkpoint_time = datetime.now(timezone.utc)
            
            # Emit restore event
            restore_event = self._event_factory.checkpoint_restore(
                checkpoint_id=checkpoint_id,
                reason="manual_restore",
            )
            self._events.append(restore_event)
            
            return self._current_state
    
    async def branch(self, checkpoint_id: str) -> AgentState | None:
        """
        Create a new execution branch from a checkpoint.
        
        This is similar to restore but explicitly creates a branch
        for what-if analysis or parallel execution.
        
        Args:
            checkpoint_id: ID of the checkpoint to branch from
            
        Returns:
            The branched state, or None if checkpoint not found
        """
        async with self._lock:
            state = await self.checkpoint_manager.branch_from(checkpoint_id)
            if state is None:
                return None
            
            # Update to use the branched state
            self._session_id = state.session_id
            self._current_state = state
            
            # Reset tracking
            self._event_factory = EventFactory(self.agent_id, self._session_id)
            self._events = []
            self._transitions = []
            self._transition_count = 0
            self._transitions_since_checkpoint = 0
            self._last_checkpoint_time = datetime.now(timezone.utc)
            
            return self._current_state
    
    async def stop(self, reason: str = "manual") -> TransitionResult:
        """
        Stop the state machine.
        
        This emits an AGENT_STOP event and transitions to STOPPED status.
        
        Args:
            reason: Why the agent is stopping
            
        Returns:
            TransitionResult from the stop transition
        """
        stop_event = self._event_factory.agent_stop(
            reason=reason,
            final_state={
                "version": self._current_state.version,
                "message_count": self._current_state.message_count,
            },
        )
        
        result = await self.process_event(stop_event)
        
        # Create final checkpoint
        if result.is_success and self.config.auto_checkpoint:
            async with self._lock:
                await self._create_checkpoint_internal("final")
        
        return result
    
    def get_event_factory(self) -> EventFactory:
        """Get the event factory for creating events."""
        return self._event_factory
    
    async def get_checkpoint_history(
        self,
        max_depth: int = 10,
    ) -> list[Checkpoint]:
        """
        Get the checkpoint history for the current state.
        
        Args:
            max_depth: Maximum number of checkpoints to retrieve
            
        Returns:
            List of checkpoints from oldest to newest
        """
        if not self._current_state.checkpoint_id:
            return []
        
        return await self.checkpoint_manager.get_checkpoint_history(
            self._current_state.checkpoint_id,
            max_depth,
        )
    
    async def list_checkpoints(self, limit: int = 100) -> list[Checkpoint]:
        """
        List all checkpoints for this agent.
        
        Args:
            limit: Maximum number of checkpoints to return
            
        Returns:
            List of checkpoints, newest first
        """
        return await self.checkpoint_manager.list_checkpoints(
            self.agent_id,
            limit=limit,
        )


class StateMachineFactory:
    """
    Factory for creating state machines with consistent configuration.
    
    Useful for creating multiple agents with the same settings.
    """
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager | None = None,
        config: StateMachineConfig | None = None,
    ) -> None:
        self.checkpoint_manager = checkpoint_manager
        self.config = config
    
    def create(
        self,
        agent_id: str,
        initial_state: AgentState | None = None,
    ) -> StateMachine:
        """
        Create a new state machine.
        
        Args:
            agent_id: Unique identifier for the agent
            initial_state: Optional initial state
            
        Returns:
            A new StateMachine instance
        """
        return StateMachine(
            agent_id=agent_id,
            checkpoint_manager=self.checkpoint_manager,
            config=self.config,
            initial_state=initial_state,
        )
    
    async def create_and_initialize(
        self,
        agent_id: str,
        initial_state: AgentState | None = None,
    ) -> StateMachine:
        """
        Create and initialize a new state machine.
        
        Args:
            agent_id: Unique identifier for the agent
            initial_state: Optional initial state
            
        Returns:
            An initialized StateMachine instance
        """
        machine = self.create(agent_id, initial_state)
        await machine.initialize()
        return machine
