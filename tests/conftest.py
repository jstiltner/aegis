"""Pytest configuration and fixtures for agent runtime tests."""

import pytest
import asyncio
from typing import AsyncGenerator

from aegis.core.state import AgentState, Message, MessageRole
from aegis.core.events import Event, EventType, EventFactory
from aegis.core.checkpoint import (
    CheckpointManager,
    InMemoryCheckpointStorage,
    Checkpoint,
)
from aegis.state_machine.machine import StateMachine, StateMachineConfig
from aegis.core.agent import Agent, AgentConfig


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def agent_id() -> str:
    """Provide a test agent ID."""
    return "test-agent-001"


@pytest.fixture
def session_id() -> str:
    """Provide a test session ID."""
    return "test-session-001"


@pytest.fixture
def event_factory(agent_id: str, session_id: str) -> EventFactory:
    """Create an event factory for tests."""
    return EventFactory(agent_id, session_id)


@pytest.fixture
def initial_state(agent_id: str, session_id: str) -> AgentState:
    """Create an initial agent state for tests."""
    return AgentState(
        agent_id=agent_id,
        session_id=session_id,
    )


@pytest.fixture
def state_with_messages(initial_state: AgentState) -> AgentState:
    """Create a state with some messages."""
    state = initial_state
    state = state.with_message(Message.system("You are a helpful assistant."))
    state = state.with_message(Message.user("Hello!"))
    state = state.with_message(Message.assistant("Hi there! How can I help?"))
    return state


@pytest.fixture
def checkpoint_storage() -> InMemoryCheckpointStorage:
    """Create an in-memory checkpoint storage."""
    return InMemoryCheckpointStorage()


@pytest.fixture
def checkpoint_manager(checkpoint_storage: InMemoryCheckpointStorage) -> CheckpointManager:
    """Create a checkpoint manager with in-memory storage."""
    return CheckpointManager(checkpoint_storage)


@pytest.fixture
def state_machine_config() -> StateMachineConfig:
    """Create a state machine configuration for tests."""
    return StateMachineConfig(
        auto_checkpoint=False,  # Disable auto-checkpoint for predictable tests
        validate_transitions=True,
    )


@pytest.fixture
async def state_machine(
    agent_id: str,
    checkpoint_manager: CheckpointManager,
    state_machine_config: StateMachineConfig,
) -> AsyncGenerator[StateMachine, None]:
    """Create and initialize a state machine for tests."""
    machine = StateMachine(
        agent_id=agent_id,
        checkpoint_manager=checkpoint_manager,
        config=state_machine_config,
    )
    await machine.initialize()
    yield machine


@pytest.fixture
def agent_config() -> AgentConfig:
    """Create an agent configuration for tests."""
    return AgentConfig(
        name="Test Agent",
        description="An agent for testing",
        system_prompt="You are a test assistant.",
    )


@pytest.fixture
async def agent(
    agent_id: str,
    agent_config: AgentConfig,
    checkpoint_manager: CheckpointManager,
) -> AsyncGenerator[Agent, None]:
    """Create and start an agent for tests."""
    agent = Agent(
        agent_id=agent_id,
        config=agent_config,
        checkpoint_manager=checkpoint_manager,
    )
    await agent.start()
    yield agent
    if agent.is_running:
        await agent.stop("test_cleanup")


# Helper functions for tests


def create_user_input_event(
    event_factory: EventFactory,
    content: str,
) -> Event:
    """Create a user input event."""
    return event_factory.user_input(content)


def create_llm_response_event(
    event_factory: EventFactory,
    content: str,
) -> Event:
    """Create an LLM response event."""
    return event_factory.llm_response(content)


def create_tool_request_event(
    event_factory: EventFactory,
    tool_name: str,
    arguments: dict,
) -> Event:
    """Create a tool request event."""
    return event_factory.tool_request(tool_name, arguments)


def create_tool_response_event(
    event_factory: EventFactory,
    tool_name: str,
    result: str,
) -> Event:
    """Create a tool response event."""
    return event_factory.tool_response(tool_name, result)
