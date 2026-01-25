"""
Unified LLM client.

This module provides a unified client for interacting with LLMs,
supporting multiple providers, tool execution, and conversation management.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from .models import (
    LLMMessage,
    LLMResponse,
    LLMConfig,
    LLMUsage,
    Conversation,
    ContentBlock,
    TextContent,
    ToolUseContent,
    ToolResultContent,
    MessageRole,
    StopReason,
)
from .streaming import StreamingResponse, StreamEvent, StreamEventType
from .providers import LLMProvider, AnthropicProvider, MockProvider


class LLMClientConfig(BaseModel):
    """
    Configuration for the LLM client.
    
    Attributes:
        default_model: Default model to use
        default_max_tokens: Default max tokens
        default_temperature: Default temperature
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries in seconds
        timeout: Request timeout in seconds
    """
    
    model_config = ConfigDict(frozen=True)
    
    default_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Default model to use",
    )
    default_max_tokens: int = Field(
        default=4096,
        description="Default max tokens",
    )
    default_temperature: float = Field(
        default=1.0,
        description="Default temperature",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts",
    )
    retry_delay: float = Field(
        default=1.0,
        description="Delay between retries in seconds",
    )
    timeout: float = Field(
        default=300.0,
        description="Request timeout in seconds",
    )


class ToolExecutor:
    """
    Executes tools and returns results.
    
    Bridges between LLM tool use requests and actual tool implementations.
    """
    
    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Awaitable[Any]]] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
    
    def register(
        self,
        name: str,
        handler: Callable[..., Awaitable[Any]],
        schema: dict[str, Any] | None = None,
    ) -> None:
        """
        Register a tool.
        
        Args:
            name: Tool name
            handler: Async function to handle tool calls
            schema: JSON schema for the tool
        """
        self._tools[name] = handler
        if schema:
            self._schemas[name] = schema
    
    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        self._tools.pop(name, None)
        self._schemas.pop(name, None)
    
    def get_schemas(self) -> list[dict[str, Any]]:
        """Get all tool schemas for LLM."""
        return list(self._schemas.values())
    
    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    async def execute(
        self,
        tool_use: ToolUseContent,
    ) -> ToolResultContent:
        """
        Execute a tool use request.
        
        Args:
            tool_use: Tool use content from LLM
            
        Returns:
            ToolResultContent with the result
        """
        if tool_use.name not in self._tools:
            return ToolResultContent(
                tool_use_id=tool_use.id,
                content=f"Error: Unknown tool '{tool_use.name}'",
                is_error=True,
            )
        
        try:
            handler = self._tools[tool_use.name]
            result = await handler(**tool_use.input)
            
            # Convert result to string if needed
            if isinstance(result, str):
                content = result
            elif isinstance(result, dict):
                import json
                content = json.dumps(result, indent=2)
            else:
                content = str(result)
            
            return ToolResultContent(
                tool_use_id=tool_use.id,
                content=content,
            )
        except Exception as e:
            return ToolResultContent(
                tool_use_id=tool_use.id,
                content=f"Error executing tool: {str(e)}",
                is_error=True,
            )
    
    async def execute_all(
        self,
        tool_uses: list[ToolUseContent],
        parallel: bool = True,
    ) -> list[ToolResultContent]:
        """
        Execute multiple tool uses.
        
        Args:
            tool_uses: List of tool use requests
            parallel: Whether to execute in parallel
            
        Returns:
            List of tool results
        """
        if parallel:
            tasks = [self.execute(tu) for tu in tool_uses]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for tu in tool_uses:
                result = await self.execute(tu)
                results.append(result)
            return results


class LLMClient:
    """
    Unified LLM client.
    
    Provides a high-level interface for LLM interactions with support for:
    - Multiple providers
    - Tool execution
    - Conversation management
    - Streaming
    - Retries
    
    Example:
        >>> client = LLMClient()
        >>> response = await client.complete("Hello, world!")
        >>> print(response.get_text())
    """
    
    def __init__(
        self,
        provider: LLMProvider | None = None,
        config: LLMClientConfig | None = None,
        tool_executor: ToolExecutor | None = None,
    ) -> None:
        """
        Initialize the LLM client.
        
        Args:
            provider: LLM provider to use
            config: Client configuration
            tool_executor: Tool executor for handling tool calls
        """
        self._provider = provider or AnthropicProvider()
        self._config = config or LLMClientConfig()
        self._tool_executor = tool_executor or ToolExecutor()
        
        # Statistics
        self._total_requests = 0
        self._total_tokens = LLMUsage()
        self._total_errors = 0
    
    @property
    def provider(self) -> LLMProvider:
        """Get the current provider."""
        return self._provider
    
    @property
    def config(self) -> LLMClientConfig:
        """Get the client configuration."""
        return self._config
    
    @property
    def tool_executor(self) -> ToolExecutor:
        """Get the tool executor."""
        return self._tool_executor
    
    @property
    def total_requests(self) -> int:
        """Get total number of requests made."""
        return self._total_requests
    
    @property
    def total_tokens(self) -> LLMUsage:
        """Get total token usage."""
        return self._total_tokens
    
    def register_tool(
        self,
        name: str,
        handler: Callable[..., Awaitable[Any]],
        description: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """
        Register a tool for LLM use.
        
        Args:
            name: Tool name
            handler: Async function to handle tool calls
            description: Tool description
            parameters: JSON schema for parameters
        """
        schema = {
            "name": name,
            "description": description,
            "input_schema": parameters or {"type": "object", "properties": {}},
        }
        self._tool_executor.register(name, handler, schema)
    
    def _build_config(
        self,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMConfig:
        """Build LLM config with defaults."""
        tools = self._tool_executor.get_schemas() if self._tool_executor._tools else None
        
        return LLMConfig(
            model=model or self._config.default_model,
            max_tokens=max_tokens or self._config.default_max_tokens,
            temperature=temperature if temperature is not None else self._config.default_temperature,
            system=system,
            tools=tools,
            **kwargs,
        )
    
    async def _complete_with_retry(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        """Complete with retry logic."""
        last_error: Exception | None = None
        
        for attempt in range(self._config.max_retries):
            try:
                response = await self._provider.complete(messages, config)
                self._total_requests += 1
                self._total_tokens = LLMUsage(
                    input_tokens=self._total_tokens.input_tokens + response.usage.input_tokens,
                    output_tokens=self._total_tokens.output_tokens + response.usage.output_tokens,
                )
                return response
            except Exception as e:
                last_error = e
                self._total_errors += 1
                
                if attempt < self._config.max_retries - 1:
                    await asyncio.sleep(self._config.retry_delay * (attempt + 1))
        
        raise last_error or RuntimeError("Unknown error during completion")
    
    async def complete(
        self,
        prompt: str | list[LLMMessage],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion.
        
        Args:
            prompt: User prompt or list of messages
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system: System prompt
            **kwargs: Additional config options
            
        Returns:
            LLMResponse with generated content
        """
        # Convert prompt to messages
        if isinstance(prompt, str):
            messages = [LLMMessage.user(prompt)]
        else:
            messages = prompt
        
        config = self._build_config(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            **kwargs,
        )
        
        return await self._complete_with_retry(messages, config)
    
    async def stream(
        self,
        prompt: str | list[LLMMessage],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> StreamingResponse:
        """
        Generate a streaming completion.
        
        Args:
            prompt: User prompt or list of messages
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system: System prompt
            **kwargs: Additional config options
            
        Returns:
            StreamingResponse for async iteration
        """
        # Convert prompt to messages
        if isinstance(prompt, str):
            messages = [LLMMessage.user(prompt)]
        else:
            messages = prompt
        
        config = self._build_config(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            **kwargs,
        )
        
        self._total_requests += 1
        return await self._provider.stream(messages, config)
    
    async def chat(
        self,
        conversation: Conversation,
        user_message: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Continue a conversation.
        
        Args:
            conversation: Conversation to continue
            user_message: User's message
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional config options
            
        Returns:
            LLMResponse with generated content
        """
        # Add user message
        conversation.add_message(LLMMessage.user(user_message))
        
        config = self._build_config(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=conversation.system_prompt,
            **kwargs,
        )
        
        response = await self._complete_with_retry(conversation.messages, config)
        
        # Add assistant response
        conversation.add_message(LLMMessage.assistant(response.content))
        
        return response
    
    async def run_agent_loop(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        max_iterations: int = 10,
        on_tool_use: Callable[[ToolUseContent], Awaitable[None]] | None = None,
        on_response: Callable[[LLMResponse], Awaitable[None]] | None = None,
        **kwargs: Any,
    ) -> Conversation:
        """
        Run an agentic loop with tool execution.
        
        Continues until the model stops using tools or max iterations reached.
        
        Args:
            prompt: Initial user prompt
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system: System prompt
            max_iterations: Maximum loop iterations
            on_tool_use: Callback for tool use events
            on_response: Callback for response events
            **kwargs: Additional config options
            
        Returns:
            Conversation with full history
        """
        conversation = Conversation(system_prompt=system)
        conversation.add_message(LLMMessage.user(prompt))
        
        for iteration in range(max_iterations):
            config = self._build_config(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                **kwargs,
            )
            
            response = await self._complete_with_retry(conversation.messages, config)
            
            if on_response:
                await on_response(response)
            
            # Add assistant response
            conversation.add_message(LLMMessage.assistant(response.content))
            
            # Check for tool use
            tool_uses = response.get_tool_uses()
            
            if not tool_uses or response.stop_reason != StopReason.TOOL_USE:
                # No more tool use, we're done
                break
            
            # Execute tools
            tool_results: list[ContentBlock] = []
            for tool_use in tool_uses:
                if on_tool_use:
                    await on_tool_use(tool_use)
                
                result = await self._tool_executor.execute(tool_use)
                tool_results.append(result)
            
            # Add tool results as user message
            conversation.add_message(LLMMessage(
                role=MessageRole.USER,
                content=tool_results,
            ))
        
        return conversation
    
    async def run_agent_loop_streaming(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        max_iterations: int = 10,
        on_text: Callable[[str], Awaitable[None]] | None = None,
        on_tool_use: Callable[[ToolUseContent], Awaitable[None]] | None = None,
        **kwargs: Any,
    ) -> Conversation:
        """
        Run an agentic loop with streaming.
        
        Args:
            prompt: Initial user prompt
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system: System prompt
            max_iterations: Maximum loop iterations
            on_text: Callback for text deltas
            on_tool_use: Callback for tool use events
            **kwargs: Additional config options
            
        Returns:
            Conversation with full history
        """
        conversation = Conversation(system_prompt=system)
        conversation.add_message(LLMMessage.user(prompt))
        
        for iteration in range(max_iterations):
            config = self._build_config(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                **kwargs,
            )
            
            stream = await self._provider.stream(conversation.messages, config)
            
            if on_text:
                stream.on_text(on_text)
            
            # Collect the response
            response = await stream.collect()
            
            self._total_tokens = LLMUsage(
                input_tokens=self._total_tokens.input_tokens + response.usage.input_tokens,
                output_tokens=self._total_tokens.output_tokens + response.usage.output_tokens,
            )
            
            # Add assistant response
            conversation.add_message(LLMMessage.assistant(response.content))
            
            # Check for tool use
            tool_uses = response.get_tool_uses()
            
            if not tool_uses or response.stop_reason != StopReason.TOOL_USE:
                break
            
            # Execute tools
            tool_results: list[ContentBlock] = []
            for tool_use in tool_uses:
                if on_tool_use:
                    await on_tool_use(tool_use)
                
                result = await self._tool_executor.execute(tool_use)
                tool_results.append(result)
            
            # Add tool results
            conversation.add_message(LLMMessage(
                role=MessageRole.USER,
                content=tool_results,
            ))
        
        return conversation


def create_client(
    provider: str = "anthropic",
    api_key: str | None = None,
    **kwargs: Any,
) -> LLMClient:
    """
    Create an LLM client with the specified provider.
    
    Args:
        provider: Provider name ("anthropic", "litellm", "mock")
        api_key: API key for the provider
        **kwargs: Additional provider options
        
    Returns:
        Configured LLMClient
    """
    from .providers import LiteLLMProvider
    
    if provider == "anthropic":
        llm_provider = AnthropicProvider(api_key=api_key, **kwargs)
    elif provider == "litellm":
        llm_provider = LiteLLMProvider(api_key=api_key, **kwargs)
    elif provider == "mock":
        llm_provider = MockProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    return LLMClient(provider=llm_provider)
