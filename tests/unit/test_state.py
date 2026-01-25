"""Tests for agent state models."""

import pytest
from datetime import datetime, timezone

from aegis.core.state import (
    AgentState,
    AgentStatus,
    Message,
    MessageRole,
    ToolCall,
    StateTransition,
    StateDiff,
)
from aegis.core.events import Event, EventType


class TestMessage:
    """Tests for the Message class."""
    
    def test_create_user_message(self):
        """Test creating a user message."""
        msg = Message.user("Hello!")
        
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello!"
        assert msg.message_id is not None
        assert msg.timestamp is not None
    
    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        msg = Message.assistant("Hi there!")
        
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Hi there!"
    
    def test_create_system_message(self):
        """Test creating a system message."""
        msg = Message.system("You are helpful.")
        
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == "You are helpful."
    
    def test_create_tool_message(self):
        """Test creating a tool message."""
        msg = Message.tool(
            content="Result: 42",
            tool_call_id="call-123",
            tool_name="calculator",
        )
        
        assert msg.role == MessageRole.TOOL
        assert msg.content == "Result: 42"
        assert msg.tool_call_id == "call-123"
        assert msg.tool_name == "calculator"
    
    def test_message_is_immutable(self):
        """Test that messages are immutable."""
        msg = Message.user("Hello!")
        
        with pytest.raises(Exception):  # Pydantic frozen model
            msg.content = "Changed"


class TestToolCall:
    """Tests for the ToolCall class."""
    
    def test_create_tool_call(self):
        """Test creating a tool call."""
        tc = ToolCall(
            tool_name="search",
            arguments={"query": "test"},
        )
        
        assert tc.tool_name == "search"
        assert tc.arguments == {"query": "test"}
        assert tc.tool_call_id is not None
        assert tc.commitment_id is None
    
    def test_tool_call_with_commitment(self):
        """Test creating a tool call with commitment."""
        tc = ToolCall(
            tool_name="search",
            arguments={"query": "test"},
            commitment_id="commit-123",
        )
        
        assert tc.commitment_id == "commit-123"


class TestAgentState:
    """Tests for the AgentState class."""
    
    def test_create_initial_state(self, agent_id: str, session_id: str):
        """Test creating an initial state."""
        state = AgentState(
            agent_id=agent_id,
            session_id=session_id,
        )
        
        assert state.agent_id == agent_id
        assert state.session_id == session_id
        assert state.version == 0
        assert state.status == AgentStatus.INITIALIZING
        assert state.message_count == 0
        assert not state.has_pending_tools
    
    def test_with_message(self, initial_state: AgentState):
        """Test adding a message to state."""
        msg = Message.user("Hello!")
        new_state = initial_state.with_message(msg)
        
        # Original state unchanged
        assert initial_state.message_count == 0
        
        # New state has message
        assert new_state.message_count == 1
        assert new_state.version == initial_state.version + 1
        assert new_state.conversation_history[-1] == msg
    
    def test_with_messages(self, initial_state: AgentState):
        """Test adding multiple messages to state."""
        messages = [
            Message.user("Hello!"),
            Message.assistant("Hi!"),
        ]
        new_state = initial_state.with_messages(messages)
        
        assert new_state.message_count == 2
        assert new_state.version == initial_state.version + 1
    
    def test_with_status(self, initial_state: AgentState):
        """Test changing status."""
        new_state = initial_state.with_status(AgentStatus.THINKING)
        
        assert initial_state.status == AgentStatus.INITIALIZING
        assert new_state.status == AgentStatus.THINKING
        assert new_state.version == initial_state.version + 1
    
    def test_with_tool_calls(self, initial_state: AgentState):
        """Test adding tool calls."""
        tc = ToolCall(tool_name="search", arguments={})
        new_state = initial_state.with_tool_calls([tc])
        
        assert not initial_state.has_pending_tools
        assert new_state.has_pending_tools
        assert len(new_state.pending_tool_calls) == 1
        assert new_state.status == AgentStatus.EXECUTING_TOOL
    
    def test_with_tool_call_resolved(self, initial_state: AgentState):
        """Test resolving a tool call."""
        tc = ToolCall(tool_name="search", arguments={})
        state_with_tc = initial_state.with_tool_calls([tc])
        
        resolved_state = state_with_tc.with_tool_call_resolved(tc.tool_call_id)
        
        assert not resolved_state.has_pending_tools
        assert resolved_state.status == AgentStatus.THINKING
    
    def test_with_memory(self, initial_state: AgentState):
        """Test setting working memory."""
        new_state = initial_state.with_memory("key", "value")
        
        assert "key" not in initial_state.working_memory
        assert new_state.working_memory["key"] == "value"
    
    def test_with_commitment(self, initial_state: AgentState):
        """Test adding a commitment."""
        new_state = initial_state.with_commitment("commit-123")
        
        assert len(initial_state.active_commitments) == 0
        assert "commit-123" in new_state.active_commitments
    
    def test_with_commitment_resolved(self, initial_state: AgentState):
        """Test resolving a commitment."""
        state_with_commit = initial_state.with_commitment("commit-123")
        resolved_state = state_with_commit.with_commitment_resolved("commit-123")
        
        assert "commit-123" not in resolved_state.active_commitments
    
    def test_get_last_message(self, state_with_messages: AgentState):
        """Test getting the last message."""
        last = state_with_messages.get_last_message()
        assert last is not None
        assert last.role == MessageRole.ASSISTANT
        
        last_user = state_with_messages.get_last_message(MessageRole.USER)
        assert last_user is not None
        assert last_user.content == "Hello!"
    
    def test_get_messages_for_llm(self, state_with_messages: AgentState):
        """Test formatting messages for LLM."""
        messages = state_with_messages.get_messages_for_llm()
        
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
    
    def test_as_checkpoint(self, initial_state: AgentState):
        """Test creating a checkpoint from state."""
        checkpoint_state = initial_state.as_checkpoint("new-checkpoint-id")
        
        assert checkpoint_state.checkpoint_id == "new-checkpoint-id"
        assert checkpoint_state.parent_checkpoint_id == initial_state.checkpoint_id


class TestStateDiff:
    """Tests for the StateDiff class."""
    
    def test_compute_no_changes(self, initial_state: AgentState):
        """Test computing diff with no changes."""
        diff = StateDiff.compute(initial_state, initial_state)
        
        assert not diff.has_changes
        assert not diff.status_changed
        assert diff.messages_added == 0
    
    def test_compute_status_change(self, initial_state: AgentState):
        """Test computing diff with status change."""
        new_state = initial_state.with_status(AgentStatus.THINKING)
        diff = StateDiff.compute(initial_state, new_state)
        
        assert diff.has_changes
        assert diff.status_changed
        assert diff.status_from == AgentStatus.INITIALIZING
        assert diff.status_to == AgentStatus.THINKING
    
    def test_compute_message_added(self, initial_state: AgentState):
        """Test computing diff with message added."""
        new_state = initial_state.with_message(Message.user("Hello!"))
        diff = StateDiff.compute(initial_state, new_state)
        
        assert diff.has_changes
        assert diff.messages_added == 1
        assert diff.messages_removed == 0
    
    def test_compute_memory_changes(self, initial_state: AgentState):
        """Test computing diff with memory changes."""
        state1 = initial_state.with_memory("key1", "value1")
        state2 = state1.with_memory("key2", "value2")
        
        diff = StateDiff.compute(state1, state2)
        
        assert diff.has_changes
        assert "key2" in diff.memory_keys_added
    
    def test_compute_commitment_changes(self, initial_state: AgentState):
        """Test computing diff with commitment changes."""
        state1 = initial_state.with_commitment("commit-1")
        state2 = state1.with_commitment("commit-2")
        
        diff = StateDiff.compute(state1, state2)
        
        assert diff.has_changes
        assert "commit-2" in diff.commitments_added
