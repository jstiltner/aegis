"""Tests for checkpoint management."""

import pytest
from datetime import datetime, timezone
from pathlib import Path
import tempfile

from aegis.core.state import AgentState, Message
from aegis.core.events import Event, EventType, EventFactory
from aegis.core.checkpoint import (
    Checkpoint,
    CheckpointManager,
    CheckpointPolicy,
    InMemoryCheckpointStorage,
    FileCheckpointStorage,
)


class TestCheckpoint:
    """Tests for the Checkpoint class."""
    
    def test_create_checkpoint(self, initial_state: AgentState):
        """Test creating a checkpoint."""
        checkpoint = Checkpoint(
            agent_id=initial_state.agent_id,
            session_id=initial_state.session_id,
            state=initial_state,
            reason="test",
        )
        
        assert checkpoint.checkpoint_id is not None
        assert checkpoint.agent_id == initial_state.agent_id
        assert checkpoint.state == initial_state
        assert checkpoint.reason == "test"
        assert checkpoint.version == initial_state.version
    
    def test_checkpoint_with_events(
        self,
        initial_state: AgentState,
        event_factory: EventFactory,
    ):
        """Test creating a checkpoint with events."""
        events = [
            event_factory.user_input("Hello"),
            event_factory.llm_response("Hi there!"),
        ]
        
        checkpoint = Checkpoint(
            agent_id=initial_state.agent_id,
            session_id=initial_state.session_id,
            state=initial_state,
            reason="test",
            events_since_parent=tuple(events),
        )
        
        assert checkpoint.event_count == 2
    
    def test_checkpoint_with_parent(self, initial_state: AgentState):
        """Test creating a checkpoint with parent."""
        parent_id = "parent-checkpoint-123"
        
        checkpoint = Checkpoint(
            agent_id=initial_state.agent_id,
            session_id=initial_state.session_id,
            state=initial_state,
            reason="test",
            parent_checkpoint_id=parent_id,
        )
        
        assert checkpoint.parent_checkpoint_id == parent_id


class TestInMemoryCheckpointStorage:
    """Tests for in-memory checkpoint storage."""
    
    @pytest.fixture
    def storage(self) -> InMemoryCheckpointStorage:
        return InMemoryCheckpointStorage()
    
    @pytest.mark.asyncio
    async def test_save_and_load(
        self,
        storage: InMemoryCheckpointStorage,
        initial_state: AgentState,
    ):
        """Test saving and loading a checkpoint."""
        checkpoint = Checkpoint(
            agent_id=initial_state.agent_id,
            session_id=initial_state.session_id,
            state=initial_state,
            reason="test",
        )
        
        await storage.save(checkpoint)
        loaded = await storage.load(checkpoint.checkpoint_id)
        
        assert loaded is not None
        assert loaded.checkpoint_id == checkpoint.checkpoint_id
        assert loaded.state.agent_id == initial_state.agent_id
    
    @pytest.mark.asyncio
    async def test_load_nonexistent(self, storage: InMemoryCheckpointStorage):
        """Test loading a nonexistent checkpoint."""
        loaded = await storage.load("nonexistent-id")
        assert loaded is None
    
    @pytest.mark.asyncio
    async def test_delete(
        self,
        storage: InMemoryCheckpointStorage,
        initial_state: AgentState,
    ):
        """Test deleting a checkpoint."""
        checkpoint = Checkpoint(
            agent_id=initial_state.agent_id,
            session_id=initial_state.session_id,
            state=initial_state,
            reason="test",
        )
        
        await storage.save(checkpoint)
        
        # Delete should succeed
        result = await storage.delete(checkpoint.checkpoint_id)
        assert result is True
        
        # Should no longer exist
        loaded = await storage.load(checkpoint.checkpoint_id)
        assert loaded is None
        
        # Delete again should fail
        result = await storage.delete(checkpoint.checkpoint_id)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_list_checkpoints(
        self,
        storage: InMemoryCheckpointStorage,
        agent_id: str,
        session_id: str,
    ):
        """Test listing checkpoints."""
        # Create multiple checkpoints
        for i in range(5):
            state = AgentState(agent_id=agent_id, session_id=session_id, version=i)
            checkpoint = Checkpoint(
                agent_id=agent_id,
                session_id=session_id,
                state=state,
                reason=f"test-{i}",
            )
            await storage.save(checkpoint)
        
        # List all
        checkpoints = await storage.list_checkpoints(agent_id)
        assert len(checkpoints) == 5
        
        # List with limit
        checkpoints = await storage.list_checkpoints(agent_id, limit=3)
        assert len(checkpoints) == 3
        
        # List by session
        checkpoints = await storage.list_checkpoints(agent_id, session_id=session_id)
        assert len(checkpoints) == 5
    
    @pytest.mark.asyncio
    async def test_get_latest(
        self,
        storage: InMemoryCheckpointStorage,
        agent_id: str,
        session_id: str,
    ):
        """Test getting the latest checkpoint."""
        # Create checkpoints with different times
        import asyncio
        
        for i in range(3):
            state = AgentState(agent_id=agent_id, session_id=session_id, version=i)
            checkpoint = Checkpoint(
                agent_id=agent_id,
                session_id=session_id,
                state=state,
                reason=f"test-{i}",
            )
            await storage.save(checkpoint)
            await asyncio.sleep(0.01)  # Small delay to ensure different timestamps
        
        latest = await storage.get_latest(agent_id)
        assert latest is not None
        assert latest.reason == "test-2"  # Last one created


class TestFileCheckpointStorage:
    """Tests for file-based checkpoint storage."""
    
    @pytest.fixture
    def storage(self) -> FileCheckpointStorage:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield FileCheckpointStorage(Path(tmpdir))
    
    @pytest.mark.asyncio
    async def test_save_and_load(
        self,
        storage: FileCheckpointStorage,
        initial_state: AgentState,
    ):
        """Test saving and loading a checkpoint to file."""
        checkpoint = Checkpoint(
            agent_id=initial_state.agent_id,
            session_id=initial_state.session_id,
            state=initial_state,
            reason="test",
        )
        
        await storage.save(checkpoint)
        loaded = await storage.load(checkpoint.checkpoint_id)
        
        assert loaded is not None
        assert loaded.checkpoint_id == checkpoint.checkpoint_id
    
    @pytest.mark.asyncio
    async def test_list_checkpoints(
        self,
        storage: FileCheckpointStorage,
        agent_id: str,
        session_id: str,
    ):
        """Test listing checkpoints from files."""
        # Create multiple checkpoints
        for i in range(3):
            state = AgentState(agent_id=agent_id, session_id=session_id, version=i)
            checkpoint = Checkpoint(
                agent_id=agent_id,
                session_id=session_id,
                state=state,
                reason=f"test-{i}",
            )
            await storage.save(checkpoint)
        
        checkpoints = await storage.list_checkpoints(agent_id)
        assert len(checkpoints) == 3


class TestCheckpointManager:
    """Tests for the CheckpointManager class."""
    
    @pytest.mark.asyncio
    async def test_create_checkpoint(
        self,
        checkpoint_manager: CheckpointManager,
        initial_state: AgentState,
    ):
        """Test creating a checkpoint through the manager."""
        checkpoint = await checkpoint_manager.create_checkpoint(
            state=initial_state,
            reason="test",
            tags=["test", "unit"],
        )
        
        assert checkpoint.checkpoint_id is not None
        assert checkpoint.reason == "test"
        assert "test" in checkpoint.tags
    
    @pytest.mark.asyncio
    async def test_restore(
        self,
        checkpoint_manager: CheckpointManager,
        initial_state: AgentState,
    ):
        """Test restoring from a checkpoint."""
        # Create checkpoint
        checkpoint = await checkpoint_manager.create_checkpoint(
            state=initial_state,
            reason="test",
        )
        
        # Restore
        restored = await checkpoint_manager.restore(checkpoint.checkpoint_id)
        
        assert restored is not None
        assert restored.agent_id == initial_state.agent_id
        assert restored.version == initial_state.version
    
    @pytest.mark.asyncio
    async def test_restore_nonexistent(
        self,
        checkpoint_manager: CheckpointManager,
    ):
        """Test restoring from a nonexistent checkpoint."""
        restored = await checkpoint_manager.restore("nonexistent-id")
        assert restored is None
    
    @pytest.mark.asyncio
    async def test_branch_from(
        self,
        checkpoint_manager: CheckpointManager,
        initial_state: AgentState,
    ):
        """Test branching from a checkpoint."""
        # Create checkpoint
        checkpoint = await checkpoint_manager.create_checkpoint(
            state=initial_state,
            reason="test",
        )
        
        # Branch
        branched = await checkpoint_manager.branch_from(checkpoint.checkpoint_id)
        
        assert branched is not None
        assert branched.session_id != initial_state.session_id
        assert branched.parent_checkpoint_id == checkpoint.checkpoint_id
        assert branched.version == 0  # Reset for new branch
    
    @pytest.mark.asyncio
    async def test_checkpoint_history(
        self,
        checkpoint_manager: CheckpointManager,
        agent_id: str,
        session_id: str,
    ):
        """Test getting checkpoint history."""
        # Create chain of checkpoints
        state = AgentState(agent_id=agent_id, session_id=session_id)
        
        cp1 = await checkpoint_manager.create_checkpoint(state, "first")
        
        state = state.with_message(Message.user("Hello"))
        cp2 = await checkpoint_manager.create_checkpoint(state, "second")
        
        state = state.with_message(Message.assistant("Hi"))
        cp3 = await checkpoint_manager.create_checkpoint(state, "third")
        
        # Get history
        history = await checkpoint_manager.get_checkpoint_history(
            cp3.checkpoint_id,
            max_depth=10,
        )
        
        assert len(history) == 3
        assert history[0].reason == "first"
        assert history[2].reason == "third"
    
    @pytest.mark.asyncio
    async def test_record_event(
        self,
        checkpoint_manager: CheckpointManager,
        event_factory: EventFactory,
        initial_state: AgentState,
    ):
        """Test recording events for checkpoints."""
        # Record some events
        checkpoint_manager.record_event(event_factory.user_input("Hello"))
        checkpoint_manager.record_event(event_factory.llm_response("Hi"))
        
        # Create checkpoint
        checkpoint = await checkpoint_manager.create_checkpoint(
            state=initial_state,
            reason="test",
        )
        
        # Events should be included
        assert checkpoint.event_count == 2
    
    @pytest.mark.asyncio
    async def test_auto_checkpoint_tracking(
        self,
        checkpoint_storage: InMemoryCheckpointStorage,
    ):
        """Test auto-checkpoint interval tracking."""
        manager = CheckpointManager(
            checkpoint_storage,
            auto_checkpoint_interval=5,
        )
        
        # Initially should not need checkpoint
        assert not manager.should_auto_checkpoint()
        
        # Record transitions
        from aegis.core.state import StateTransition
        from aegis.core.events import Event, EventType
        
        for i in range(5):
            event = Event(
                event_type=EventType.USER_INPUT,
                agent_id="test",
                session_id="test",
                data={"content": f"msg-{i}"},
            )
            transition = StateTransition(
                from_checkpoint_id="cp1",
                to_checkpoint_id="cp2",
                from_version=i,
                to_version=i + 1,
                event=event,
            )
            manager.record_transition(transition)
        
        # Now should need checkpoint
        assert manager.should_auto_checkpoint()


class TestCheckpointPolicy:
    """Tests for the CheckpointPolicy class."""
    
    def test_transition_interval(self):
        """Test transition interval policy."""
        # Disable event-based checkpointing to test interval-based only
        policy = CheckpointPolicy(
            transition_interval=10,
            checkpoint_on_user_input=False,
            checkpoint_on_tool_call=False,
            checkpoint_on_commitment=False,
        )
        
        event = Event(
            event_type=EventType.USER_INPUT,
            agent_id="test",
            session_id="test",
            data={},
        )
        
        # Below threshold
        assert not policy.should_checkpoint(event, 5, 0)
        
        # At threshold
        assert policy.should_checkpoint(event, 10, 0)
    
    def test_time_interval(self):
        """Test time interval policy."""
        # Disable event-based checkpointing to test interval-based only
        policy = CheckpointPolicy(
            time_interval_seconds=60.0,
            checkpoint_on_user_input=False,
            checkpoint_on_tool_call=False,
            checkpoint_on_commitment=False,
        )
        
        event = Event(
            event_type=EventType.USER_INPUT,
            agent_id="test",
            session_id="test",
            data={},
        )
        
        # Below threshold
        assert not policy.should_checkpoint(event, 0, 30.0)
        
        # At threshold
        assert policy.should_checkpoint(event, 0, 60.0)
    
    def test_event_based_tool_call(self):
        """Test event-based policy for tool calls."""
        policy = CheckpointPolicy(checkpoint_on_tool_call=True)
        
        tool_event = Event(
            event_type=EventType.TOOL_REQUEST,
            agent_id="test",
            session_id="test",
            data={},
        )
        
        assert policy.should_checkpoint(tool_event, 0, 0)
    
    def test_event_based_user_input(self):
        """Test event-based policy for user input."""
        policy = CheckpointPolicy(checkpoint_on_user_input=True)
        
        user_event = Event(
            event_type=EventType.USER_INPUT,
            agent_id="test",
            session_id="test",
            data={},
        )
        
        assert policy.should_checkpoint(user_event, 0, 0)
    
    def test_event_based_commitment(self):
        """Test event-based policy for commitments."""
        policy = CheckpointPolicy(checkpoint_on_commitment=True)
        
        commit_event = Event(
            event_type=EventType.COMMITMENT_ISSUED,
            agent_id="test",
            session_id="test",
            data={},
        )
        
        assert policy.should_checkpoint(commit_event, 0, 0)
    
    def test_combined_policies(self):
        """Test combining multiple policy triggers."""
        policy = CheckpointPolicy(
            transition_interval=100,
            checkpoint_on_tool_call=True,
            checkpoint_on_user_input=False,
        )
        
        # Tool call should trigger
        tool_event = Event(
            event_type=EventType.TOOL_REQUEST,
            agent_id="test",
            session_id="test",
            data={},
        )
        assert policy.should_checkpoint(tool_event, 0, 0)
        
        # User input should not trigger (disabled)
        user_event = Event(
            event_type=EventType.USER_INPUT,
            agent_id="test",
            session_id="test",
            data={},
        )
        assert not policy.should_checkpoint(user_event, 0, 0)
        
        # But transition interval should still work
        assert policy.should_checkpoint(user_event, 100, 0)
