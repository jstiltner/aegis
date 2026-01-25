"""
Replay engine for deterministic execution replay.

This module provides replay functionality that enables:
- Replaying execution from any checkpoint
- Deterministic reproduction of agent behavior
- Debugging and analysis of past executions
- What-if analysis with modified events
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from aegis.core.state import AgentState, StateTransition, StateDiff
from aegis.core.events import Event, EventType
from aegis.core.checkpoint import Checkpoint, CheckpointManager


class ReplayMode(str, Enum):
    """Mode of replay execution."""
    
    # Execute events as fast as possible
    FAST = "fast"
    
    # Execute events with original timing
    REALTIME = "realtime"
    
    # Step through events one at a time
    STEP = "step"
    
    # Execute until a breakpoint is hit
    BREAKPOINT = "breakpoint"


class ReplayStatus(str, Enum):
    """Status of the replay engine."""
    
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STEPPING = "stepping"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ReplayBreakpoint:
    """
    A breakpoint for replay execution.
    
    Breakpoints can be set on:
    - Specific event IDs
    - Event types
    - State conditions
    - Step numbers
    """
    
    breakpoint_id: str = field(default_factory=lambda: str(uuid4()))
    
    # Trigger conditions (any match triggers the breakpoint)
    event_id: str | None = None
    event_type: EventType | None = None
    step_number: int | None = None
    
    # Custom condition function
    condition: Callable[[AgentState, Event], bool] | None = None
    
    # Metadata
    name: str = ""
    enabled: bool = True
    
    def matches(self, state: AgentState, event: Event, step: int) -> bool:
        """Check if this breakpoint matches the current state/event."""
        if not self.enabled:
            return False
        
        if self.event_id is not None and event.event_id == self.event_id:
            return True
        
        if self.event_type is not None and event.event_type == self.event_type:
            return True
        
        if self.step_number is not None and step == self.step_number:
            return True
        
        if self.condition is not None and self.condition(state, event):
            return True
        
        return False


@dataclass
class ReplayStep:
    """
    A single step in the replay.
    
    Contains the event, resulting state, and timing information.
    """
    
    step_number: int
    event: Event
    state_before: AgentState
    state_after: AgentState
    diff: StateDiff
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0.0
    
    # Whether this step hit a breakpoint
    breakpoint_hit: ReplayBreakpoint | None = None


class ReplayResult(BaseModel):
    """
    Result of a replay execution.
    
    Contains the final state and all steps taken.
    """
    
    model_config = ConfigDict(frozen=True)
    
    replay_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this replay",
    )
    
    # Source
    from_checkpoint_id: str = Field(
        ...,
        description="Checkpoint ID replay started from",
    )
    
    # Status
    status: ReplayStatus = Field(
        ...,
        description="Final status of the replay",
    )
    
    # Results
    initial_state: AgentState = Field(
        ...,
        description="State at the start of replay",
    )
    final_state: AgentState = Field(
        ...,
        description="State at the end of replay",
    )
    
    # Statistics
    total_steps: int = Field(
        default=0,
        description="Total number of steps executed",
    )
    total_events: int = Field(
        default=0,
        description="Total number of events in the replay",
    )
    events_replayed: int = Field(
        default=0,
        description="Number of events actually replayed",
    )
    
    # Timing
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the replay started",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="When the replay completed",
    )
    
    # Error info
    error: str | None = Field(
        default=None,
        description="Error message if replay failed",
    )
    error_event_id: str | None = Field(
        default=None,
        description="Event ID where error occurred",
    )
    
    @property
    def duration_ms(self) -> float | None:
        """Total duration of the replay in milliseconds."""
        if self.completed_at is None:
            return None
        delta = self.completed_at - self.started_at
        return delta.total_seconds() * 1000


# Type for event handlers
EventHandler = Callable[[AgentState, Event], Awaitable[AgentState]]


class ReplayEngine:
    """
    Engine for replaying agent execution.
    
    The ReplayEngine takes a sequence of events and replays them
    against an initial state, producing the same sequence of states
    that occurred during the original execution.
    
    Key features:
    - Deterministic replay from checkpoints
    - Multiple replay modes (fast, realtime, step)
    - Breakpoints for debugging
    - Event modification for what-if analysis
    
    Example:
        >>> engine = ReplayEngine(checkpoint_manager)
        >>> 
        >>> # Replay from a checkpoint
        >>> result = await engine.replay_from_checkpoint(
        ...     checkpoint_id="abc-123",
        ...     mode=ReplayMode.FAST,
        ... )
        >>> 
        >>> # Step through replay
        >>> engine.set_mode(ReplayMode.STEP)
        >>> step = await engine.step()
    """
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        event_handlers: dict[EventType, EventHandler] | None = None,
    ) -> None:
        """
        Initialize the replay engine.
        
        Args:
            checkpoint_manager: Manager for loading checkpoints
            event_handlers: Custom handlers for processing events
        """
        self.checkpoint_manager = checkpoint_manager
        self.event_handlers = event_handlers or {}
        
        # Replay state
        self._status = ReplayStatus.IDLE
        self._mode = ReplayMode.FAST
        self._current_state: AgentState | None = None
        self._events: list[Event] = []
        self._current_step = 0
        self._steps: list[ReplayStep] = []
        
        # Breakpoints
        self._breakpoints: dict[str, ReplayBreakpoint] = {}
        
        # Control
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused initially
        self._step_event = asyncio.Event()
    
    @property
    def status(self) -> ReplayStatus:
        """Get the current replay status."""
        return self._status
    
    @property
    def mode(self) -> ReplayMode:
        """Get the current replay mode."""
        return self._mode
    
    @property
    def current_state(self) -> AgentState | None:
        """Get the current state during replay."""
        return self._current_state
    
    @property
    def current_step(self) -> int:
        """Get the current step number."""
        return self._current_step
    
    @property
    def steps(self) -> list[ReplayStep]:
        """Get all replay steps executed so far."""
        return self._steps.copy()
    
    def set_mode(self, mode: ReplayMode) -> None:
        """Set the replay mode."""
        self._mode = mode
        
        # If switching to step mode while running, pause
        if mode == ReplayMode.STEP and self._status == ReplayStatus.RUNNING:
            self._status = ReplayStatus.STEPPING
    
    def add_breakpoint(self, breakpoint: ReplayBreakpoint) -> str:
        """
        Add a breakpoint.
        
        Args:
            breakpoint: The breakpoint to add
            
        Returns:
            The breakpoint ID
        """
        self._breakpoints[breakpoint.breakpoint_id] = breakpoint
        return breakpoint.breakpoint_id
    
    def remove_breakpoint(self, breakpoint_id: str) -> bool:
        """
        Remove a breakpoint.
        
        Args:
            breakpoint_id: ID of the breakpoint to remove
            
        Returns:
            True if the breakpoint was found and removed
        """
        if breakpoint_id in self._breakpoints:
            del self._breakpoints[breakpoint_id]
            return True
        return False
    
    def clear_breakpoints(self) -> None:
        """Remove all breakpoints."""
        self._breakpoints.clear()
    
    def pause(self) -> None:
        """Pause the replay."""
        self._pause_event.clear()
        self._status = ReplayStatus.PAUSED
    
    def resume(self) -> None:
        """Resume the replay."""
        self._pause_event.set()
        if self._status == ReplayStatus.PAUSED:
            self._status = ReplayStatus.RUNNING
    
    async def step(self) -> ReplayStep | None:
        """
        Execute a single step in step mode.
        
        Returns:
            The executed step, or None if no more events
        """
        if self._current_state is None or self._current_step >= len(self._events):
            return None
        
        self._status = ReplayStatus.STEPPING
        step = await self._execute_step()
        self._status = ReplayStatus.PAUSED
        
        return step
    
    async def replay_from_checkpoint(
        self,
        checkpoint_id: str,
        to_checkpoint_id: str | None = None,
        mode: ReplayMode = ReplayMode.FAST,
    ) -> ReplayResult:
        """
        Replay execution from a checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to start from
            to_checkpoint_id: Optional ID of checkpoint to replay to
            mode: Replay mode to use
            
        Returns:
            ReplayResult with final state and statistics
        """
        # Load the starting checkpoint
        checkpoint = await self.checkpoint_manager.get_checkpoint(checkpoint_id)
        if checkpoint is None:
            return ReplayResult(
                from_checkpoint_id=checkpoint_id,
                status=ReplayStatus.ERROR,
                initial_state=AgentState(agent_id="unknown"),
                final_state=AgentState(agent_id="unknown"),
                error=f"Checkpoint {checkpoint_id} not found",
            )
        
        # Get events to replay
        if to_checkpoint_id:
            events = await self.checkpoint_manager.get_events_between(
                checkpoint_id, to_checkpoint_id
            )
        else:
            # Get events from checkpoint's recorded events
            events = list(checkpoint.events_since_parent)
        
        return await self.replay_events(
            initial_state=checkpoint.state,
            events=events,
            mode=mode,
            from_checkpoint_id=checkpoint_id,
        )
    
    async def replay_events(
        self,
        initial_state: AgentState,
        events: list[Event],
        mode: ReplayMode = ReplayMode.FAST,
        from_checkpoint_id: str | None = None,
    ) -> ReplayResult:
        """
        Replay a sequence of events.
        
        Args:
            initial_state: State to start from
            events: Events to replay
            mode: Replay mode to use
            from_checkpoint_id: Optional source checkpoint ID
            
        Returns:
            ReplayResult with final state and statistics
        """
        # Initialize replay state
        self._current_state = initial_state
        self._events = events
        self._current_step = 0
        self._steps = []
        self._mode = mode
        self._status = ReplayStatus.RUNNING
        
        started_at = datetime.now(timezone.utc)
        error: str | None = None
        error_event_id: str | None = None
        
        try:
            while self._current_step < len(events):
                # Check for pause
                await self._pause_event.wait()
                
                # Check for step mode
                if self._mode == ReplayMode.STEP:
                    self._status = ReplayStatus.PAUSED
                    await self._step_event.wait()
                    self._step_event.clear()
                
                # Execute the step
                step = await self._execute_step()
                
                # Check for breakpoint hit
                if step.breakpoint_hit is not None:
                    self._status = ReplayStatus.PAUSED
                    break
                
                # Handle realtime mode delay
                if self._mode == ReplayMode.REALTIME and self._current_step < len(events):
                    next_event = events[self._current_step]
                    current_event = step.event
                    
                    # Calculate delay based on original timestamps
                    delay = (
                        next_event.timestamp - current_event.timestamp
                    ).total_seconds()
                    
                    if delay > 0:
                        await asyncio.sleep(delay)
            
            if self._status == ReplayStatus.RUNNING:
                self._status = ReplayStatus.COMPLETED
                
        except Exception as e:
            self._status = ReplayStatus.ERROR
            error = str(e)
            if self._current_step < len(events):
                error_event_id = events[self._current_step].event_id
        
        return ReplayResult(
            from_checkpoint_id=from_checkpoint_id or "",
            status=self._status,
            initial_state=initial_state,
            final_state=self._current_state or initial_state,
            total_steps=len(self._steps),
            total_events=len(events),
            events_replayed=self._current_step,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            error=error,
            error_event_id=error_event_id,
        )
    
    async def _execute_step(self) -> ReplayStep:
        """Execute a single replay step."""
        import time
        
        event = self._events[self._current_step]
        state_before = self._current_state
        assert state_before is not None
        
        start_time = time.perf_counter()
        
        # Apply the event to get new state
        state_after = await self._apply_event(state_before, event)
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Compute diff
        diff = StateDiff.compute(state_before, state_after)
        
        # Check breakpoints
        breakpoint_hit: ReplayBreakpoint | None = None
        for bp in self._breakpoints.values():
            if bp.matches(state_after, event, self._current_step):
                breakpoint_hit = bp
                break
        
        # Create step record
        step = ReplayStep(
            step_number=self._current_step,
            event=event,
            state_before=state_before,
            state_after=state_after,
            diff=diff,
            duration_ms=duration_ms,
            breakpoint_hit=breakpoint_hit,
        )
        
        # Update state
        self._current_state = state_after
        self._current_step += 1
        self._steps.append(step)
        
        return step
    
    async def _apply_event(self, state: AgentState, event: Event) -> AgentState:
        """
        Apply an event to a state to produce a new state.
        
        This is the core of deterministic replay. Each event type
        has a defined effect on the state.
        
        Args:
            state: Current state
            event: Event to apply
            
        Returns:
            New state after applying the event
        """
        # Check for custom handler
        if event.event_type in self.event_handlers:
            handler = self.event_handlers[event.event_type]
            return await handler(state, event)
        
        # Default handlers for each event type
        return self._default_apply_event(state, event)
    
    def _default_apply_event(self, state: AgentState, event: Event) -> AgentState:
        """
        Default event application logic.
        
        This provides basic state transitions for common events.
        """
        from aegis.core.state import Message, MessageRole, ToolCall, AgentStatus
        
        match event.event_type:
            case EventType.USER_INPUT:
                # Add user message to conversation
                content = event.data.get("content", "")
                message = Message.user(content)
                return state.with_message(message).with_status(AgentStatus.THINKING)
            
            case EventType.LLM_RESPONSE:
                # Add assistant message to conversation
                content = event.data.get("content", "")
                message = Message.assistant(content)
                return state.with_message(message)
            
            case EventType.TOOL_REQUEST:
                # Add pending tool call
                tool_name = event.data.get("tool_name", "")
                arguments = event.data.get("arguments", {})
                commitment_id = event.data.get("commitment_id")
                
                tool_call = ToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    commitment_id=commitment_id,
                )
                return state.with_tool_calls([*state.pending_tool_calls, tool_call])
            
            case EventType.TOOL_RESPONSE:
                # Add tool response message and resolve tool call
                tool_name = event.data.get("tool_name", "")
                result = event.data.get("result", "")
                
                # Find the tool call to resolve
                tool_call = next(
                    (tc for tc in state.pending_tool_calls if tc.tool_name == tool_name),
                    None,
                )
                
                if tool_call:
                    message = Message.tool(
                        content=str(result),
                        tool_call_id=tool_call.tool_call_id,
                        tool_name=tool_name,
                    )
                    return (
                        state
                        .with_message(message)
                        .with_tool_call_resolved(tool_call.tool_call_id)
                    )
                return state
            
            case EventType.COMMITMENT_ISSUED:
                # Add commitment to active list
                commitment_id = event.data.get("commitment_id", "")
                return state.with_commitment(commitment_id)
            
            case EventType.COMMITMENT_VERIFIED | EventType.COMMITMENT_SETTLED:
                # Remove commitment from active list
                commitment_id = event.data.get("commitment_id", "")
                return state.with_commitment_resolved(commitment_id)
            
            case EventType.AGENT_START:
                return state.with_status(AgentStatus.IDLE)
            
            case EventType.AGENT_STOP:
                return state.with_status(AgentStatus.STOPPED)
            
            case EventType.AGENT_ERROR:
                return state.with_status(AgentStatus.ERROR)
            
            case EventType.CHECKPOINT_CREATE | EventType.CHECKPOINT_RESTORE:
                # Checkpoints don't change state
                return state
            
            case _:
                # Unknown event type - no state change
                return state
    
    def get_state_at_step(self, step_number: int) -> AgentState | None:
        """
        Get the state at a specific step.
        
        Args:
            step_number: The step number to get state for
            
        Returns:
            The state after that step, or None if step not found
        """
        if step_number < 0 or step_number >= len(self._steps):
            return None
        return self._steps[step_number].state_after
    
    def get_diff_at_step(self, step_number: int) -> StateDiff | None:
        """
        Get the state diff at a specific step.
        
        Args:
            step_number: The step number to get diff for
            
        Returns:
            The diff for that step, or None if step not found
        """
        if step_number < 0 or step_number >= len(self._steps):
            return None
        return self._steps[step_number].diff
    
    def find_steps_by_event_type(self, event_type: EventType) -> list[ReplayStep]:
        """
        Find all steps with a specific event type.
        
        Args:
            event_type: The event type to search for
            
        Returns:
            List of matching steps
        """
        return [
            step for step in self._steps
            if step.event.event_type == event_type
        ]
    
    def find_steps_with_changes(self) -> list[ReplayStep]:
        """
        Find all steps that resulted in state changes.
        
        Returns:
            List of steps where the state actually changed
        """
        return [step for step in self._steps if step.diff.has_changes]


class ReplayComparator:
    """
    Compare two replay executions.
    
    Useful for:
    - Verifying determinism
    - Analyzing divergence points
    - What-if analysis
    """
    
    def __init__(
        self,
        replay_a: ReplayResult,
        steps_a: list[ReplayStep],
        replay_b: ReplayResult,
        steps_b: list[ReplayStep],
    ) -> None:
        self.replay_a = replay_a
        self.steps_a = steps_a
        self.replay_b = replay_b
        self.steps_b = steps_b
    
    def find_divergence_point(self) -> int | None:
        """
        Find the first step where the replays diverge.
        
        Returns:
            Step number of first divergence, or None if identical
        """
        min_steps = min(len(self.steps_a), len(self.steps_b))
        
        for i in range(min_steps):
            state_a = self.steps_a[i].state_after
            state_b = self.steps_b[i].state_after
            
            # Compare key state attributes
            if (
                state_a.status != state_b.status
                or state_a.message_count != state_b.message_count
                or state_a.working_memory != state_b.working_memory
                or set(state_a.active_commitments) != set(state_b.active_commitments)
            ):
                return i
        
        # Check if one replay is longer
        if len(self.steps_a) != len(self.steps_b):
            return min_steps
        
        return None
    
    def are_identical(self) -> bool:
        """Check if the two replays produced identical results."""
        return self.find_divergence_point() is None
    
    def get_divergence_summary(self) -> dict[str, Any]:
        """
        Get a summary of differences between the replays.
        
        Returns:
            Dictionary with divergence information
        """
        divergence_point = self.find_divergence_point()
        
        if divergence_point is None:
            return {
                "identical": True,
                "divergence_point": None,
                "steps_a": len(self.steps_a),
                "steps_b": len(self.steps_b),
            }
        
        # Get states at divergence
        state_a = (
            self.steps_a[divergence_point].state_after
            if divergence_point < len(self.steps_a)
            else self.replay_a.final_state
        )
        state_b = (
            self.steps_b[divergence_point].state_after
            if divergence_point < len(self.steps_b)
            else self.replay_b.final_state
        )
        
        return {
            "identical": False,
            "divergence_point": divergence_point,
            "steps_a": len(self.steps_a),
            "steps_b": len(self.steps_b),
            "status_a": state_a.status.value,
            "status_b": state_b.status.value,
            "messages_a": state_a.message_count,
            "messages_b": state_b.message_count,
            "diff": StateDiff.compute(state_a, state_b).model_dump(),
        }
