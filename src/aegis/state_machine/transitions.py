"""
State transition handlers and validation.

This module provides:
- Transition handlers for different event types
- Transition validation
- Transition result tracking
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import uuid4

from aegis.core.state import AgentState, AgentStatus, StateTransition
from aegis.core.events import Event, EventType


class TransitionStatus(str, Enum):
    """Status of a transition attempt."""
    
    SUCCESS = "success"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass
class TransitionResult:
    """
    Result of a state transition attempt.
    
    Contains the new state (if successful) and metadata about the transition.
    """
    
    status: TransitionStatus
    new_state: AgentState | None = None
    transition: StateTransition | None = None
    
    # Error information
    error: str | None = None
    error_type: str | None = None
    
    # Validation failures
    validation_errors: list[str] = field(default_factory=list)
    
    # Timing
    duration_ms: float = 0.0
    
    @property
    def is_success(self) -> bool:
        """Check if the transition was successful."""
        return self.status == TransitionStatus.SUCCESS
    
    @property
    def is_rejected(self) -> bool:
        """Check if the transition was rejected."""
        return self.status == TransitionStatus.REJECTED
    
    @property
    def is_error(self) -> bool:
        """Check if the transition resulted in an error."""
        return self.status == TransitionStatus.ERROR


class TransitionValidator(ABC):
    """
    Abstract base class for transition validators.
    
    Validators check whether a transition is allowed before it's applied.
    """
    
    @abstractmethod
    def validate(
        self,
        current_state: AgentState,
        event: Event,
    ) -> list[str]:
        """
        Validate a transition.
        
        Args:
            current_state: The current agent state
            event: The event triggering the transition
            
        Returns:
            List of validation error messages (empty if valid)
        """
        ...


class StatusTransitionValidator(TransitionValidator):
    """
    Validates that status transitions are allowed.
    
    Defines which status transitions are valid based on the current status.
    """
    
    # Define allowed transitions: current_status -> set of allowed next statuses
    ALLOWED_TRANSITIONS: dict[AgentStatus, set[AgentStatus]] = {
        AgentStatus.INITIALIZING: {
            AgentStatus.IDLE,
            AgentStatus.ERROR,
            AgentStatus.STOPPED,
        },
        AgentStatus.IDLE: {
            AgentStatus.THINKING,
            AgentStatus.AWAITING_INPUT,
            AgentStatus.COMPLETED,
            AgentStatus.ERROR,
            AgentStatus.STOPPED,
        },
        AgentStatus.THINKING: {
            AgentStatus.IDLE,
            AgentStatus.EXECUTING_TOOL,
            AgentStatus.AWAITING_INPUT,
            AgentStatus.AWAITING_APPROVAL,
            AgentStatus.COMPLETED,
            AgentStatus.ERROR,
            AgentStatus.STOPPED,
        },
        AgentStatus.EXECUTING_TOOL: {
            AgentStatus.THINKING,
            AgentStatus.IDLE,
            AgentStatus.ERROR,
            AgentStatus.STOPPED,
        },
        AgentStatus.AWAITING_INPUT: {
            AgentStatus.THINKING,
            AgentStatus.IDLE,
            AgentStatus.ERROR,
            AgentStatus.STOPPED,
        },
        AgentStatus.AWAITING_APPROVAL: {
            AgentStatus.THINKING,
            AgentStatus.EXECUTING_TOOL,
            AgentStatus.IDLE,
            AgentStatus.ERROR,
            AgentStatus.STOPPED,
        },
        AgentStatus.COMPLETED: {
            AgentStatus.IDLE,  # Can restart
            AgentStatus.STOPPED,
        },
        AgentStatus.ERROR: {
            AgentStatus.IDLE,  # Can recover
            AgentStatus.STOPPED,
        },
        AgentStatus.STOPPED: set(),  # Terminal state
    }
    
    def validate(
        self,
        current_state: AgentState,
        event: Event,
    ) -> list[str]:
        """Validate that the implied status transition is allowed."""
        errors: list[str] = []
        
        # Determine the implied next status based on event type
        next_status = self._get_implied_status(event)
        if next_status is None:
            return errors  # Event doesn't imply a status change
        
        current_status = current_state.status
        allowed = self.ALLOWED_TRANSITIONS.get(current_status, set())
        
        if next_status not in allowed:
            errors.append(
                f"Invalid status transition: {current_status.value} -> {next_status.value}"
            )
        
        return errors
    
    def _get_implied_status(self, event: Event) -> AgentStatus | None:
        """Get the status implied by an event type."""
        match event.event_type:
            case EventType.AGENT_START:
                return AgentStatus.IDLE
            case EventType.AGENT_STOP:
                return AgentStatus.STOPPED
            case EventType.AGENT_ERROR:
                return AgentStatus.ERROR
            case EventType.USER_INPUT:
                return AgentStatus.THINKING
            case EventType.LLM_RESPONSE:
                return AgentStatus.IDLE  # LLM response completes thinking, return to IDLE
            case EventType.TOOL_REQUEST:
                return AgentStatus.EXECUTING_TOOL
            case EventType.TOOL_RESPONSE:
                return AgentStatus.THINKING
            case EventType.HUMAN_APPROVAL_REQUEST:
                return AgentStatus.AWAITING_APPROVAL
            case _:
                return None


class CommitmentValidator(TransitionValidator):
    """
    Validates commitment-related transitions.
    
    Ensures that:
    - Commitments are not issued when at max capacity
    - Commitments being verified exist
    """
    
    def __init__(self, max_active_commitments: int = 100) -> None:
        self.max_active_commitments = max_active_commitments
    
    def validate(
        self,
        current_state: AgentState,
        event: Event,
    ) -> list[str]:
        """Validate commitment-related transitions."""
        errors: list[str] = []
        
        if event.event_type == EventType.COMMITMENT_ISSUED:
            # Check capacity
            if len(current_state.active_commitments) >= self.max_active_commitments:
                errors.append(
                    f"Cannot issue commitment: at max capacity "
                    f"({self.max_active_commitments})"
                )
        
        elif event.event_type in (
            EventType.COMMITMENT_VERIFIED,
            EventType.COMMITMENT_SETTLED,
        ):
            # Check that commitment exists
            commitment_id = event.data.get("commitment_id")
            if commitment_id and commitment_id not in current_state.active_commitments:
                errors.append(
                    f"Cannot verify/settle commitment: {commitment_id} not found"
                )
        
        return errors


class ToolCallValidator(TransitionValidator):
    """
    Validates tool call transitions.
    
    Ensures that:
    - Tool responses match pending tool calls
    - Not too many concurrent tool calls
    """
    
    def __init__(self, max_pending_tool_calls: int = 10) -> None:
        self.max_pending_tool_calls = max_pending_tool_calls
    
    def validate(
        self,
        current_state: AgentState,
        event: Event,
    ) -> list[str]:
        """Validate tool call transitions."""
        errors: list[str] = []
        
        if event.event_type == EventType.TOOL_REQUEST:
            # Check capacity
            if len(current_state.pending_tool_calls) >= self.max_pending_tool_calls:
                errors.append(
                    f"Cannot request tool: at max pending calls "
                    f"({self.max_pending_tool_calls})"
                )
        
        elif event.event_type == EventType.TOOL_RESPONSE:
            # Check that there's a matching pending call
            tool_name = event.data.get("tool_name")
            matching_call = any(
                tc.tool_name == tool_name
                for tc in current_state.pending_tool_calls
            )
            if not matching_call:
                errors.append(
                    f"Tool response for '{tool_name}' has no matching pending call"
                )
        
        return errors


# Type alias for transition handler functions
TransitionHandlerFunc = Callable[[AgentState, Event], Awaitable[AgentState]]


class TransitionHandler:
    """
    Handles state transitions with validation and event application.
    
    The TransitionHandler:
    1. Validates transitions using registered validators
    2. Applies events to produce new states
    3. Records transitions for audit/replay
    
    Example:
        >>> handler = TransitionHandler()
        >>> handler.add_validator(StatusTransitionValidator())
        >>> 
        >>> result = await handler.apply(current_state, event)
        >>> if result.is_success:
        ...     new_state = result.new_state
    """
    
    def __init__(self) -> None:
        self._validators: list[TransitionValidator] = []
        self._event_handlers: dict[EventType, TransitionHandlerFunc] = {}
        
        # Register default validators
        self.add_validator(StatusTransitionValidator())
        self.add_validator(CommitmentValidator())
        self.add_validator(ToolCallValidator())
    
    def add_validator(self, validator: TransitionValidator) -> None:
        """Add a transition validator."""
        self._validators.append(validator)
    
    def remove_validator(self, validator: TransitionValidator) -> None:
        """Remove a transition validator."""
        self._validators.remove(validator)
    
    def register_handler(
        self,
        event_type: EventType,
        handler: TransitionHandlerFunc,
    ) -> None:
        """
        Register a custom handler for an event type.
        
        Custom handlers override the default event application logic.
        """
        self._event_handlers[event_type] = handler
    
    def validate(
        self,
        current_state: AgentState,
        event: Event,
    ) -> list[str]:
        """
        Validate a transition.
        
        Args:
            current_state: The current agent state
            event: The event triggering the transition
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []
        
        for validator in self._validators:
            errors.extend(validator.validate(current_state, event))
        
        return errors
    
    async def apply(
        self,
        current_state: AgentState,
        event: Event,
        skip_validation: bool = False,
    ) -> TransitionResult:
        """
        Apply an event to produce a new state.
        
        Args:
            current_state: The current agent state
            event: The event to apply
            skip_validation: If True, skip validation (for replay)
            
        Returns:
            TransitionResult with new state or error information
        """
        import time
        start_time = time.perf_counter()
        
        # Validate unless skipped
        if not skip_validation:
            validation_errors = self.validate(current_state, event)
            if validation_errors:
                return TransitionResult(
                    status=TransitionStatus.REJECTED,
                    validation_errors=validation_errors,
                    duration_ms=(time.perf_counter() - start_time) * 1000,
                )
        
        try:
            # Apply the event
            if event.event_type in self._event_handlers:
                # Use custom handler
                new_state = await self._event_handlers[event.event_type](
                    current_state, event
                )
            else:
                # Use default handler
                new_state = self._default_apply(current_state, event)
            
            # Create transition record
            transition = StateTransition(
                from_checkpoint_id=current_state.checkpoint_id,
                to_checkpoint_id=new_state.checkpoint_id,
                from_version=current_state.version,
                to_version=new_state.version,
                event=event,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )
            
            return TransitionResult(
                status=TransitionStatus.SUCCESS,
                new_state=new_state,
                transition=transition,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )
            
        except Exception as e:
            return TransitionResult(
                status=TransitionStatus.ERROR,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )
    
    def _default_apply(self, state: AgentState, event: Event) -> AgentState:
        """
        Default event application logic.
        
        This mirrors the replay engine's default logic to ensure consistency.
        """
        from aegis.core.state import Message, ToolCall
        
        match event.event_type:
            case EventType.USER_INPUT:
                content = event.data.get("content", "")
                message = Message.user(content)
                return state.with_message(message).with_status(AgentStatus.THINKING)
            
            case EventType.LLM_RESPONSE:
                content = event.data.get("content", "")
                message = Message.assistant(content)
                return state.with_message(message).with_status(AgentStatus.IDLE)
            
            case EventType.TOOL_REQUEST:
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
                tool_name = event.data.get("tool_name", "")
                result = event.data.get("result", "")
                
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
                commitment_id = event.data.get("commitment_id", "")
                return state.with_commitment(commitment_id)
            
            case EventType.COMMITMENT_VERIFIED | EventType.COMMITMENT_SETTLED:
                commitment_id = event.data.get("commitment_id", "")
                return state.with_commitment_resolved(commitment_id)
            
            case EventType.AGENT_START:
                return state.with_status(AgentStatus.IDLE)
            
            case EventType.AGENT_STOP:
                return state.with_status(AgentStatus.STOPPED)
            
            case EventType.AGENT_ERROR:
                return state.with_status(AgentStatus.ERROR)
            
            case _:
                # Unknown event type - increment step but no other changes
                return state.with_step_increment()
