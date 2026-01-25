"""
Base agent class for the autonomous agent runtime.

This module provides the Agent class that:
- Wraps the state machine for high-level operations
- Provides a clean API for agent interactions
- Integrates with tools, LLMs, and GCL
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from aegis.core.state import AgentState, AgentStatus, Message
from aegis.core.events import Event, EventType, EventFactory
from aegis.core.checkpoint import (
    CheckpointManager,
    Checkpoint,
    InMemoryCheckpointStorage,
)
from aegis.state_machine.machine import StateMachine, StateMachineConfig
from aegis.state_machine.transitions import TransitionResult


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    
    model_config = ConfigDict(frozen=True)
    
    # Identity
    name: str = Field(
        default="Agent",
        description="Human-readable name for the agent",
    )
    description: str = Field(
        default="",
        description="Description of the agent's purpose",
    )
    
    # System prompt
    system_prompt: str = Field(
        default="You are a helpful AI assistant.",
        description="System prompt for the agent",
    )
    
    # Behavior
    max_iterations: int = Field(
        default=100,
        ge=1,
        description="Maximum iterations per task",
    )
    max_tool_calls_per_iteration: int = Field(
        default=10,
        ge=1,
        description="Maximum tool calls per iteration",
    )
    
    # State machine config
    state_machine_config: StateMachineConfig = Field(
        default_factory=StateMachineConfig,
        description="Configuration for the state machine",
    )


class Agent:
    """
    Base agent class for autonomous execution.
    
    The Agent class provides a high-level interface for:
    - Processing user messages
    - Managing conversation state
    - Executing tools
    - Checkpointing and recovery
    
    This is designed to be subclassed for specific agent implementations
    that integrate with LLMs and tools.
    
    Example:
        >>> agent = Agent(
        ...     agent_id="my-agent",
        ...     config=AgentConfig(name="Research Agent"),
        ... )
        >>> 
        >>> # Initialize
        >>> await agent.start()
        >>> 
        >>> # Process a message
        >>> response = await agent.process_message("What is quantum computing?")
        >>> print(response)
        >>> 
        >>> # Checkpoint
        >>> checkpoint = await agent.checkpoint()
        >>> 
        >>> # Stop
        >>> await agent.stop()
    """
    
    def __init__(
        self,
        agent_id: str | None = None,
        config: AgentConfig | None = None,
        checkpoint_manager: CheckpointManager | None = None,
    ) -> None:
        """
        Initialize the agent.
        
        Args:
            agent_id: Unique identifier (generated if not provided)
            config: Agent configuration
            checkpoint_manager: Manager for checkpoints
        """
        self.agent_id = agent_id or str(uuid4())
        self.config = config or AgentConfig()
        
        # Create checkpoint manager if not provided
        if checkpoint_manager is None:
            storage = InMemoryCheckpointStorage()
            checkpoint_manager = CheckpointManager(storage)
        self.checkpoint_manager = checkpoint_manager
        
        # Create state machine
        self._state_machine = StateMachine(
            agent_id=self.agent_id,
            checkpoint_manager=checkpoint_manager,
            config=self.config.state_machine_config,
        )
        
        # Iteration tracking
        self._current_iteration = 0
        self._tool_calls_this_iteration = 0
        
        # Hooks for subclasses
        self._on_message_hooks: list[Callable[[Message], Awaitable[None]]] = []
        self._on_tool_call_hooks: list[Callable[[str, dict], Awaitable[None]]] = []
    
    @property
    def state(self) -> AgentState:
        """Get the current agent state."""
        return self._state_machine.current_state
    
    @property
    def status(self) -> AgentStatus:
        """Get the current agent status."""
        return self.state.status
    
    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        return self._state_machine.session_id
    
    @property
    def is_running(self) -> bool:
        """Check if the agent is running."""
        return self.status not in (
            AgentStatus.STOPPED,
            AgentStatus.COMPLETED,
            AgentStatus.ERROR,
        )
    
    @property
    def conversation_history(self) -> list[Message]:
        """Get the conversation history."""
        return list(self.state.conversation_history)
    
    @property
    def event_factory(self) -> EventFactory:
        """Get the event factory for creating events."""
        return self._state_machine.get_event_factory()
    
    async def start(self) -> AgentState:
        """
        Start the agent.
        
        This initializes the state machine and sets up the system prompt.
        
        Returns:
            The initial agent state
        """
        # Initialize state machine
        state = await self._state_machine.initialize()
        
        # Add system message if configured
        if self.config.system_prompt:
            system_message = Message.system(self.config.system_prompt)
            event = Event(
                event_type=EventType.STATE_TRANSITION,
                agent_id=self.agent_id,
                session_id=self.session_id,
                data={"action": "add_system_message"},
            )
            # Directly update state with system message
            new_state = state.with_message(system_message)
            self._state_machine._current_state = new_state
        
        return self.state
    
    async def stop(self, reason: str = "manual") -> TransitionResult:
        """
        Stop the agent.
        
        Args:
            reason: Why the agent is stopping
            
        Returns:
            TransitionResult from the stop transition
        """
        return await self._state_machine.stop(reason)
    
    async def process_message(self, content: str) -> str:
        """
        Process a user message and generate a response.
        
        This is the main entry point for user interactions.
        Subclasses should override _generate_response to implement
        actual LLM integration.
        
        Args:
            content: The user's message
            
        Returns:
            The agent's response
        """
        # Create and process user input event
        user_event = self.event_factory.user_input(content)
        result = await self._state_machine.process_event(user_event)
        
        if not result.is_success:
            raise RuntimeError(f"Failed to process user input: {result.error}")
        
        # Call message hooks
        user_message = Message.user(content)
        for hook in self._on_message_hooks:
            await hook(user_message)
        
        # Generate response (to be implemented by subclasses)
        response = await self._generate_response()
        
        # Create and process LLM response event
        response_event = self.event_factory.llm_response(
            content=response,
            model="base",  # Subclasses should override with actual model
        )
        result = await self._state_machine.process_event(response_event)
        
        if not result.is_success:
            raise RuntimeError(f"Failed to process response: {result.error}")
        
        return response
    
    async def _generate_response(self) -> str:
        """
        Generate a response to the current conversation.
        
        This is a placeholder that should be overridden by subclasses
        to implement actual LLM integration.
        
        Returns:
            The generated response
        """
        # Default implementation just echoes
        last_message = self.state.get_last_message()
        if last_message:
            return f"I received your message: {last_message.content}"
        return "Hello! How can I help you?"
    
    async def checkpoint(self, reason: str = "manual") -> Checkpoint:
        """
        Create a checkpoint of the current state.
        
        Args:
            reason: Why this checkpoint is being created
            
        Returns:
            The created checkpoint
        """
        return await self._state_machine.checkpoint(reason)
    
    async def restore(self, checkpoint_id: str) -> AgentState | None:
        """
        Restore state from a checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to restore
            
        Returns:
            The restored state, or None if not found
        """
        return await self._state_machine.restore(checkpoint_id)
    
    async def branch(self, checkpoint_id: str) -> AgentState | None:
        """
        Create a new execution branch from a checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to branch from
            
        Returns:
            The branched state, or None if not found
        """
        return await self._state_machine.branch(checkpoint_id)
    
    async def list_checkpoints(self, limit: int = 100) -> list[Checkpoint]:
        """
        List all checkpoints for this agent.
        
        Args:
            limit: Maximum number of checkpoints to return
            
        Returns:
            List of checkpoints, newest first
        """
        return await self._state_machine.list_checkpoints(limit)
    
    def add_message_hook(
        self,
        hook: Callable[[Message], Awaitable[None]],
    ) -> None:
        """Add a hook that's called when messages are processed."""
        self._on_message_hooks.append(hook)
    
    def add_tool_call_hook(
        self,
        hook: Callable[[str, dict], Awaitable[None]],
    ) -> None:
        """Add a hook that's called when tools are invoked."""
        self._on_tool_call_hooks.append(hook)
    
    async def set_memory(self, key: str, value: Any) -> None:
        """
        Set a value in the agent's working memory.
        
        Args:
            key: Memory key
            value: Value to store
        """
        new_state = self.state.with_memory(key, value)
        self._state_machine._current_state = new_state
    
    def get_memory(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the agent's working memory.
        
        Args:
            key: Memory key
            default: Default value if key not found
            
        Returns:
            The stored value or default
        """
        return self.state.working_memory.get(key, default)
    
    def get_state_summary(self) -> dict[str, Any]:
        """
        Get a summary of the current agent state.
        
        Returns:
            Dictionary with state summary
        """
        return {
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "version": self.state.version,
            "message_count": self.state.message_count,
            "active_commitments": len(self.state.active_commitments),
            "pending_tool_calls": len(self.state.pending_tool_calls),
            "current_step": self.state.current_step,
        }


class AgentRunner:
    """
    Runner for executing agents with lifecycle management.
    
    The AgentRunner provides:
    - Agent lifecycle management (start, run, stop)
    - Error handling and recovery
    - Graceful shutdown
    
    Example:
        >>> runner = AgentRunner(agent)
        >>> 
        >>> # Run until completion or error
        >>> await runner.run()
        >>> 
        >>> # Or run with a task
        >>> await runner.run_task("Research quantum computing")
    """
    
    def __init__(
        self,
        agent: Agent,
        max_errors: int = 3,
        error_recovery_delay: float = 1.0,
    ) -> None:
        """
        Initialize the runner.
        
        Args:
            agent: The agent to run
            max_errors: Maximum consecutive errors before stopping
            error_recovery_delay: Delay between error recovery attempts
        """
        self.agent = agent
        self.max_errors = max_errors
        self.error_recovery_delay = error_recovery_delay
        
        self._running = False
        self._error_count = 0
        self._stop_requested = False
    
    @property
    def is_running(self) -> bool:
        """Check if the runner is currently running."""
        return self._running
    
    async def start(self) -> None:
        """Start the agent."""
        if not self.agent.is_running:
            await self.agent.start()
    
    async def stop(self, reason: str = "runner_stop") -> None:
        """
        Stop the agent.
        
        Args:
            reason: Why the agent is stopping
        """
        self._stop_requested = True
        if self.agent.is_running:
            await self.agent.stop(reason)
        self._running = False
    
    async def run_task(self, task: str) -> str:
        """
        Run the agent with a specific task.
        
        Args:
            task: The task to execute
            
        Returns:
            The final response
        """
        await self.start()
        
        try:
            response = await self.agent.process_message(task)
            return response
        except Exception as e:
            await self._handle_error(e)
            raise
    
    async def run_interactive(
        self,
        input_handler: Callable[[], Awaitable[str | None]],
        output_handler: Callable[[str], Awaitable[None]],
    ) -> None:
        """
        Run the agent in interactive mode.
        
        Args:
            input_handler: Async function that returns user input (None to stop)
            output_handler: Async function that handles agent output
        """
        await self.start()
        self._running = True
        
        try:
            while self._running and not self._stop_requested:
                # Get user input
                user_input = await input_handler()
                if user_input is None:
                    break
                
                try:
                    # Process and output response
                    response = await self.agent.process_message(user_input)
                    await output_handler(response)
                    self._error_count = 0  # Reset on success
                    
                except Exception as e:
                    await self._handle_error(e)
                    if self._error_count >= self.max_errors:
                        break
                    
        finally:
            await self.stop("interactive_end")
    
    async def _handle_error(self, error: Exception) -> None:
        """Handle an error during execution."""
        self._error_count += 1
        
        # Create error event
        error_event = self.agent.event_factory.generic(
            event_type=EventType.AGENT_ERROR,
            data={
                "error": str(error),
                "error_type": type(error).__name__,
                "error_count": self._error_count,
            },
        )
        
        try:
            await self.agent._state_machine.process_event(error_event)
        except Exception:
            pass  # Don't let error handling errors propagate
        
        # Delay before potential retry
        if self._error_count < self.max_errors:
            await asyncio.sleep(self.error_recovery_delay)
