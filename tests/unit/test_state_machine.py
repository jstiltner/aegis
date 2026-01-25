"""Tests for the state machine."""

import pytest
from datetime import datetime, timezone

from aegis.core.state import AgentState, AgentStatus, Message
from aegis.core.events import Event, EventType, EventFactory
from aegis.core.checkpoint import CheckpointManager, InMemoryCheckpointStorage
from aegis.state_machine.machine import StateMachine, StateMachineConfig
from aegis.state_machine.transitions import (
    TransitionHandler,
    TransitionResult,
    TransitionStatus,
    StatusTransitionValidator,
    CommitmentValidator,
    ToolCallValidator,
)


class TestTransitionHandler:
    """Tests for the TransitionHandler class."""
    
    @pytest.fixture
    def handler(self) -> TransitionHandler:
        return TransitionHandler()
    
    @pytest.mark.asyncio
    async def test_apply_user_input(
        self,
        handler: TransitionHandler,
        initial_state: AgentState,
        event_factory: EventFactory,
    ):
        """Test applying a user input event."""
        # First transition to IDLE
        state = initial_state.with_status(AgentStatus.IDLE)
        
        event = event_factory.user_input("Hello!")
        result = await handler.apply(state, event)
        
        assert result.is_success
        assert result.new_state is not None
        assert result.new_state.message_count == 1
        assert result.new_state.status == AgentStatus.THINKING
    
    @pytest.mark.asyncio
    async def test_apply_llm_response(
        self,
        handler: TransitionHandler,
        initial_state: AgentState,
        event_factory: EventFactory,
    ):
        """Test applying an LLM response event."""
        state = initial_state.with_status(AgentStatus.THINKING)
        
        event = event_factory.llm_response("Hi there!")
        result = await handler.apply(state, event)
        
        assert result.is_success
        assert result.new_state is not None
        assert result.new_state.message_count == 1
        assert result.new_state.get_last_message().content == "Hi there!"
    
    @pytest.mark.asyncio
    async def test_apply_tool_request(
        self,
        handler: TransitionHandler,
        initial_state: AgentState,
        event_factory: EventFactory,
    ):
        """Test applying a tool request event."""
        state = initial_state.with_status(AgentStatus.THINKING)
        
        event = event_factory.tool_request("search", {"query": "test"})
        result = await handler.apply(state, event)
        
        assert result.is_success
        assert result.new_state is not None
        assert result.new_state.has_pending_tools
        assert result.new_state.status == AgentStatus.EXECUTING_TOOL
    
    @pytest.mark.asyncio
    async def test_apply_tool_response(
        self,
        handler: TransitionHandler,
        initial_state: AgentState,
        event_factory: EventFactory,
    ):
        """Test applying a tool response event."""
        # Set up state with pending tool call
        state = initial_state.with_status(AgentStatus.THINKING)
        tool_event = event_factory.tool_request("search", {"query": "test"})
        result = await handler.apply(state, tool_event)
        state = result.new_state
        
        # Now apply response
        response_event = event_factory.tool_response("search", "Results: ...")
        result = await handler.apply(state, response_event)
        
        assert result.is_success
        assert result.new_state is not None
        assert not result.new_state.has_pending_tools
        assert result.new_state.status == AgentStatus.THINKING
    
    @pytest.mark.asyncio
    async def test_apply_commitment_issued(
        self,
        handler: TransitionHandler,
        initial_state: AgentState,
        event_factory: EventFactory,
    ):
        """Test applying a commitment issued event."""
        event = event_factory.commitment_issued(
            commitment_id="commit-123",
            trigger="task_start",
            stake=1.0,
            confidence=0.9,
            failure_modes=["timeout", "error"],
        )
        result = await handler.apply(initial_state, event)
        
        assert result.is_success
        assert result.new_state is not None
        assert "commit-123" in result.new_state.active_commitments
    
    @pytest.mark.asyncio
    async def test_apply_commitment_verified(
        self,
        handler: TransitionHandler,
        initial_state: AgentState,
        event_factory: EventFactory,
    ):
        """Test applying a commitment verified event."""
        # Add commitment first
        state = initial_state.with_commitment("commit-123")
        
        event = event_factory.commitment_verified(
            commitment_id="commit-123",
            success=True,
            reward=0.1,
        )
        result = await handler.apply(state, event)
        
        assert result.is_success
        assert result.new_state is not None
        assert "commit-123" not in result.new_state.active_commitments
    
    @pytest.mark.asyncio
    async def test_validation_rejection(
        self,
        handler: TransitionHandler,
        initial_state: AgentState,
        event_factory: EventFactory,
    ):
        """Test that invalid transitions are rejected."""
        # Try to process user input from INITIALIZING (should fail)
        event = event_factory.user_input("Hello!")
        result = await handler.apply(initial_state, event)
        
        assert result.is_rejected
        assert len(result.validation_errors) > 0
    
    @pytest.mark.asyncio
    async def test_skip_validation(
        self,
        handler: TransitionHandler,
        initial_state: AgentState,
        event_factory: EventFactory,
    ):
        """Test skipping validation for replay."""
        event = event_factory.user_input("Hello!")
        result = await handler.apply(initial_state, event, skip_validation=True)
        
        # Should succeed even though transition is normally invalid
        assert result.is_success


class TestStatusTransitionValidator:
    """Tests for status transition validation."""
    
    @pytest.fixture
    def validator(self) -> StatusTransitionValidator:
        return StatusTransitionValidator()
    
    def test_valid_idle_to_thinking(
        self,
        validator: StatusTransitionValidator,
        initial_state: AgentState,
        event_factory: EventFactory,
    ):
        """Test valid transition from IDLE to THINKING."""
        state = initial_state.with_status(AgentStatus.IDLE)
        event = event_factory.user_input("Hello!")
        
        errors = validator.validate(state, event)
        assert len(errors) == 0
    
    def test_invalid_stopped_transition(
        self,
        validator: StatusTransitionValidator,
        initial_state: AgentState,
        event_factory: EventFactory,
    ):
        """Test that STOPPED is terminal."""
        state = initial_state.with_status(AgentStatus.STOPPED)
        event = event_factory.user_input("Hello!")
        
        errors = validator.validate(state, event)
        assert len(errors) > 0
        assert "Invalid status transition" in errors[0]


class TestCommitmentValidator:
    """Tests for commitment validation."""
    
    @pytest.fixture
    def validator(self) -> CommitmentValidator:
        return CommitmentValidator(max_active_commitments=3)
    
    def test_commitment_capacity(
        self,
        validator: CommitmentValidator,
        initial_state: AgentState,
        event_factory: EventFactory,
    ):
        """Test commitment capacity limit."""
        # Add max commitments
        state = initial_state
        for i in range(3):
            state = state.with_commitment(f"commit-{i}")
        
        # Try to add one more
        event = event_factory.commitment_issued(
            commitment_id="commit-new",
            trigger="test",
            stake=1.0,
            confidence=0.9,
            failure_modes=[],
        )
        
        errors = validator.validate(state, event)
        assert len(errors) > 0
        assert "max capacity" in errors[0]
    
    def test_verify_nonexistent_commitment(
        self,
        validator: CommitmentValidator,
        initial_state: AgentState,
        event_factory: EventFactory,
    ):
        """Test verifying a nonexistent commitment."""
        event = event_factory.commitment_verified(
            commitment_id="nonexistent",
            success=True,
        )
        
        errors = validator.validate(initial_state, event)
        assert len(errors) > 0
        assert "not found" in errors[0]


class TestToolCallValidator:
    """Tests for tool call validation."""
    
    @pytest.fixture
    def validator(self) -> ToolCallValidator:
        return ToolCallValidator(max_pending_tool_calls=2)
    
    def test_tool_call_capacity(
        self,
        validator: ToolCallValidator,
        initial_state: AgentState,
        event_factory: EventFactory,
    ):
        """Test tool call capacity limit."""
        from aegis.core.state import ToolCall
        
        # Add max tool calls
        tool_calls = [
            ToolCall(tool_name=f"tool-{i}", arguments={})
            for i in range(2)
        ]
        state = initial_state.with_tool_calls(tool_calls)
        
        # Try to add one more
        event = event_factory.tool_request("new-tool", {})
        
        errors = validator.validate(state, event)
        assert len(errors) > 0
        assert "max pending" in errors[0]


class TestStateMachine:
    """Tests for the StateMachine class."""
    
    @pytest.mark.asyncio
    async def test_initialize(
        self,
        agent_id: str,
        checkpoint_manager: CheckpointManager,
    ):
        """Test state machine initialization."""
        machine = StateMachine(
            agent_id=agent_id,
            checkpoint_manager=checkpoint_manager,
        )
        
        assert not machine.is_initialized
        
        state = await machine.initialize()
        
        assert machine.is_initialized
        assert state.status == AgentStatus.IDLE
    
    @pytest.mark.asyncio
    async def test_process_event(self, state_machine: StateMachine):
        """Test processing an event."""
        # Note: initialize() already counts as 1 transition (AGENT_START event)
        initial_transition_count = state_machine.transition_count
        
        event_factory = state_machine.get_event_factory()
        event = event_factory.user_input("Hello!")
        
        result = await state_machine.process_event(event)
        
        assert result.is_success
        assert state_machine.current_state.message_count == 1
        # One additional transition after the initialization
        assert state_machine.transition_count == initial_transition_count + 1
    
    @pytest.mark.asyncio
    async def test_checkpoint_and_restore(self, state_machine: StateMachine):
        """Test checkpointing and restoring."""
        event_factory = state_machine.get_event_factory()
        
        # Process some events (user_input -> THINKING, llm_response -> IDLE)
        await state_machine.process_event(event_factory.user_input("Hello!"))
        await state_machine.process_event(event_factory.llm_response("Hi!"))
        
        # Checkpoint (state is now IDLE with 2 messages)
        checkpoint = await state_machine.checkpoint("test")
        
        # Process more events (valid: IDLE -> user_input -> THINKING -> llm_response -> IDLE)
        await state_machine.process_event(event_factory.user_input("More"))
        await state_machine.process_event(event_factory.llm_response("More response"))
        
        assert state_machine.current_state.message_count == 4
        
        # Restore - this resets the session and transition count
        restored = await state_machine.restore(checkpoint.checkpoint_id)
        
        assert restored is not None
        # After restore, message_count should be preserved from checkpoint
        assert state_machine.current_state.message_count == 2
    
    @pytest.mark.asyncio
    async def test_branch(self, state_machine: StateMachine):
        """Test branching from a checkpoint."""
        event_factory = state_machine.get_event_factory()
        
        # Process some events
        await state_machine.process_event(event_factory.user_input("Hello!"))
        
        # Checkpoint
        checkpoint = await state_machine.checkpoint("branch_point")
        original_session = state_machine.session_id
        
        # Branch
        branched = await state_machine.branch(checkpoint.checkpoint_id)
        
        assert branched is not None
        assert state_machine.session_id != original_session
        assert branched.parent_checkpoint_id == checkpoint.checkpoint_id
    
    @pytest.mark.asyncio
    async def test_stop(self, state_machine: StateMachine):
        """Test stopping the state machine."""
        result = await state_machine.stop("test_stop")
        
        assert result.is_success
        assert state_machine.current_state.status == AgentStatus.STOPPED
    
    @pytest.mark.asyncio
    async def test_state_change_listener(self, state_machine: StateMachine):
        """Test state change listeners."""
        changes = []
        
        async def listener(old_state, new_state, event):
            changes.append((old_state.version, new_state.version, event.event_type))
        
        state_machine.add_state_change_listener(listener)
        
        event_factory = state_machine.get_event_factory()
        await state_machine.process_event(event_factory.user_input("Hello!"))
        
        assert len(changes) == 1
        assert changes[0][2] == EventType.USER_INPUT
    
    @pytest.mark.asyncio
    async def test_transition_limit(
        self,
        agent_id: str,
        checkpoint_manager: CheckpointManager,
    ):
        """Test transition limit enforcement."""
        # Note: initialize() counts as 1 transition
        # Each user_input + llm_response pair = 2 transitions
        # So limit of 5 allows: 1 init + 2 pairs (4 transitions) = 5 total
        config = StateMachineConfig(
            max_transitions_per_session=5,
            auto_checkpoint=False,
        )
        machine = StateMachine(
            agent_id=agent_id,
            checkpoint_manager=checkpoint_manager,
            config=config,
        )
        await machine.initialize()
        
        event_factory = machine.get_event_factory()
        
        # Process valid transitions (user_input -> llm_response pairs)
        # After init (1), we can do 4 more transitions
        for i in range(2):
            result = await machine.process_event(event_factory.user_input(f"msg-{i}"))
            assert result.is_success, f"user_input {i} failed: {result.validation_errors}"
            result = await machine.process_event(event_factory.llm_response(f"response-{i}"))
            assert result.is_success, f"llm_response {i} failed: {result.validation_errors}"
        
        # Now at 5 transitions (1 init + 4 events), next should be rejected
        result = await machine.process_event(event_factory.user_input("too many"))
        assert result.is_rejected
        assert "Maximum transitions" in result.validation_errors[0]
    
    @pytest.mark.asyncio
    async def test_events_tracking(self, state_machine: StateMachine):
        """Test that events are tracked."""
        # Note: initialize() adds an AGENT_START event
        initial_event_count = len(state_machine.events)
        
        event_factory = state_machine.get_event_factory()
        
        await state_machine.process_event(event_factory.user_input("Hello!"))
        await state_machine.process_event(event_factory.llm_response("Hi!"))
        
        events = state_machine.events
        # Should have 2 more events than initial
        assert len(events) == initial_event_count + 2
        # The last two events should be our user input and LLM response
        assert events[-2].event_type == EventType.USER_INPUT
        assert events[-1].event_type == EventType.LLM_RESPONSE
    
    @pytest.mark.asyncio
    async def test_transitions_tracking(self, state_machine: StateMachine):
        """Test that transitions are tracked."""
        # Note: initialize() adds a transition
        initial_transition_count = len(state_machine.transitions)
        
        event_factory = state_machine.get_event_factory()
        
        await state_machine.process_event(event_factory.user_input("Hello!"))
        
        transitions = state_machine.transitions
        # Should have 1 more transition than initial
        assert len(transitions) == initial_transition_count + 1
        # The last transition should be our user input
        assert transitions[-1].event.event_type == EventType.USER_INPUT
