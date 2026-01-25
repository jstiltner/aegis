"""
LLM provider implementations.

This module provides implementations for different LLM providers
including Anthropic Claude and LiteLLM for local/other models.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator
from uuid import uuid4

from .models import (
    LLMMessage,
    LLMResponse,
    LLMConfig,
    LLMUsage,
    StopReason,
    ContentBlock,
    TextContent,
    ToolUseContent,
    MessageRole,
)
from .streaming import (
    StreamingResponse,
    StreamEvent,
    StreamEventType,
)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Providers implement the actual API calls to LLM services.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the provider name."""
        ...
    
    @property
    @abstractmethod
    def supported_models(self) -> list[str]:
        """Get list of supported model identifiers."""
        ...
    
    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        """
        Generate a completion.
        
        Args:
            messages: Conversation messages
            config: Generation configuration
            
        Returns:
            LLMResponse with generated content
        """
        ...
    
    @abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> StreamingResponse:
        """
        Generate a streaming completion.
        
        Args:
            messages: Conversation messages
            config: Generation configuration
            
        Returns:
            StreamingResponse for async iteration
        """
        ...
    
    def supports_model(self, model: str) -> bool:
        """Check if this provider supports a model."""
        return model in self.supported_models


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude provider.
    
    Uses the Anthropic Python SDK for API calls.
    
    Example:
        >>> provider = AnthropicProvider(api_key="sk-...")
        >>> response = await provider.complete(messages, config)
    """
    
    MODELS = [
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]
    
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize the Anthropic provider.
        
        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            base_url: Optional custom base URL
            default_headers: Optional default headers
        """
        self._api_key = api_key
        self._base_url = base_url
        self._default_headers = default_headers or {}
        self._client = None
    
    @property
    def name(self) -> str:
        return "anthropic"
    
    @property
    def supported_models(self) -> list[str]:
        return self.MODELS
    
    def _get_client(self) -> Any:
        """Get or create the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                
                kwargs: dict[str, Any] = {}
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                if self._default_headers:
                    kwargs["default_headers"] = self._default_headers
                
                self._client = anthropic.AsyncAnthropic(**kwargs)
            except ImportError:
                raise ImportError(
                    "anthropic package is required for AnthropicProvider. "
                    "Install with: pip install anthropic"
                )
        return self._client
    
    def _convert_messages(
        self,
        messages: list[LLMMessage],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Convert messages to Anthropic format.
        
        Returns:
            Tuple of (system_prompt, messages)
        """
        system_prompt = None
        api_messages: list[dict[str, Any]] = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # Extract system prompt
                system_prompt = msg.get_text()
                continue
            
            api_msg = msg.to_dict()
            
            # Convert role
            if msg.role == MessageRole.TOOL:
                api_msg["role"] = "user"
            
            api_messages.append(api_msg)
        
        return system_prompt, api_messages
    
    def _convert_response(
        self,
        response: Any,
        duration_ms: float,
    ) -> LLMResponse:
        """Convert Anthropic response to LLMResponse."""
        content_blocks: list[ContentBlock] = []
        
        for block in response.content:
            if block.type == "text":
                content_blocks.append(TextContent(text=block.text))
            elif block.type == "tool_use":
                content_blocks.append(ToolUseContent(
                    id=block.id,
                    name=block.name,
                    input=block.input,
                ))
        
        # Map stop reason
        stop_reason_map = {
            "end_turn": StopReason.END_TURN,
            "max_tokens": StopReason.MAX_TOKENS,
            "stop_sequence": StopReason.STOP_SEQUENCE,
            "tool_use": StopReason.TOOL_USE,
        }
        stop_reason = stop_reason_map.get(response.stop_reason, StopReason.END_TURN)
        
        return LLMResponse(
            id=response.id,
            model=response.model,
            content=content_blocks,
            stop_reason=stop_reason,
            usage=LLMUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            duration_ms=duration_ms,
        )
    
    async def complete(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        """Generate a completion using Anthropic API."""
        client = self._get_client()
        
        system_prompt, api_messages = self._convert_messages(messages)
        
        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "messages": api_messages,
        }
        
        if system_prompt or config.system:
            kwargs["system"] = config.system or system_prompt
        
        if config.temperature != 1.0:
            kwargs["temperature"] = config.temperature
        
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        
        if config.top_k is not None:
            kwargs["top_k"] = config.top_k
        
        if config.stop_sequences:
            kwargs["stop_sequences"] = config.stop_sequences
        
        if config.tools:
            kwargs["tools"] = config.tools
        
        if config.tool_choice:
            kwargs["tool_choice"] = config.tool_choice
        
        # Make request
        start_time = time.perf_counter()
        response = await client.messages.create(**kwargs)
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        return self._convert_response(response, duration_ms)
    
    async def stream(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> StreamingResponse:
        """Generate a streaming completion using Anthropic API."""
        client = self._get_client()
        
        system_prompt, api_messages = self._convert_messages(messages)
        
        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "messages": api_messages,
        }
        
        if system_prompt or config.system:
            kwargs["system"] = config.system or system_prompt
        
        if config.temperature != 1.0:
            kwargs["temperature"] = config.temperature
        
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        
        if config.top_k is not None:
            kwargs["top_k"] = config.top_k
        
        if config.stop_sequences:
            kwargs["stop_sequences"] = config.stop_sequences
        
        if config.tools:
            kwargs["tools"] = config.tools
        
        if config.tool_choice:
            kwargs["tool_choice"] = config.tool_choice
        
        # Create stream
        async def event_generator() -> AsyncIterator[StreamEvent]:
            async with client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    yield self._convert_stream_event(event)
        
        return StreamingResponse(
            event_stream=event_generator(),
            model=config.model,
        )
    
    def _convert_stream_event(self, event: Any) -> StreamEvent:
        """Convert Anthropic stream event to StreamEvent."""
        event_type = event.type
        
        if event_type == "message_start":
            return StreamEvent(
                type=StreamEventType.MESSAGE_START,
                data={"message": event.message.model_dump() if hasattr(event.message, "model_dump") else {}},
            )
        
        elif event_type == "content_block_start":
            return StreamEvent(
                type=StreamEventType.CONTENT_BLOCK_START,
                index=event.index,
                data={"content_block": event.content_block.model_dump() if hasattr(event.content_block, "model_dump") else {}},
            )
        
        elif event_type == "content_block_delta":
            delta = event.delta
            if hasattr(delta, "text"):
                return StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    index=event.index,
                    data={"text": delta.text},
                )
            elif hasattr(delta, "partial_json"):
                return StreamEvent(
                    type=StreamEventType.TOOL_USE_DELTA,
                    index=event.index,
                    data={"partial_json": delta.partial_json},
                )
            return StreamEvent(
                type=StreamEventType.CONTENT_BLOCK_DELTA,
                index=event.index,
                data={},
            )
        
        elif event_type == "content_block_stop":
            return StreamEvent(
                type=StreamEventType.CONTENT_BLOCK_STOP,
                index=event.index,
            )
        
        elif event_type == "message_delta":
            return StreamEvent(
                type=StreamEventType.MESSAGE_DELTA,
                data={
                    "stop_reason": event.delta.stop_reason if hasattr(event.delta, "stop_reason") else None,
                    "usage": event.usage.model_dump() if hasattr(event, "usage") and event.usage else None,
                },
            )
        
        elif event_type == "message_stop":
            return StreamEvent(
                type=StreamEventType.MESSAGE_STOP,
            )
        
        elif event_type == "error":
            return StreamEvent(
                type=StreamEventType.ERROR,
                data={"message": str(event.error) if hasattr(event, "error") else "Unknown error"},
            )
        
        else:
            return StreamEvent(
                type=StreamEventType.PING,
            )


class LiteLLMProvider(LLMProvider):
    """
    LiteLLM provider for multiple LLM backends.
    
    Supports OpenAI, local models, and many other providers through LiteLLM.
    
    Example:
        >>> provider = LiteLLMProvider()
        >>> response = await provider.complete(messages, config)
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        custom_llm_provider: str | None = None,
    ) -> None:
        """
        Initialize the LiteLLM provider.
        
        Args:
            api_key: API key for the provider
            api_base: Custom API base URL
            custom_llm_provider: Custom provider name
        """
        self._api_key = api_key
        self._api_base = api_base
        self._custom_llm_provider = custom_llm_provider
    
    @property
    def name(self) -> str:
        return "litellm"
    
    @property
    def supported_models(self) -> list[str]:
        # LiteLLM supports many models
        return [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "ollama/llama2",
            "ollama/mistral",
            "ollama/codellama",
            "together_ai/llama-2-70b",
            "huggingface/meta-llama/Llama-2-7b-chat-hf",
        ]
    
    def supports_model(self, model: str) -> bool:
        # LiteLLM supports many models, be permissive
        return True
    
    def _convert_messages(
        self,
        messages: list[LLMMessage],
    ) -> list[dict[str, Any]]:
        """Convert messages to OpenAI format."""
        api_messages: list[dict[str, Any]] = []
        
        for msg in messages:
            api_msg: dict[str, Any] = {
                "role": msg.role.value,
            }
            
            if isinstance(msg.content, str):
                api_msg["content"] = msg.content
            else:
                # Convert content blocks
                content_parts = []
                for block in msg.content:
                    if isinstance(block, TextContent):
                        content_parts.append({
                            "type": "text",
                            "text": block.text,
                        })
                    elif isinstance(block, ToolUseContent):
                        # Tool use in assistant message
                        api_msg["tool_calls"] = api_msg.get("tool_calls", [])
                        api_msg["tool_calls"].append({
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": str(block.input),
                            },
                        })
                
                if content_parts:
                    api_msg["content"] = content_parts if len(content_parts) > 1 else content_parts[0].get("text", "")
            
            if msg.name:
                api_msg["name"] = msg.name
            
            api_messages.append(api_msg)
        
        return api_messages
    
    def _convert_response(
        self,
        response: Any,
        duration_ms: float,
    ) -> LLMResponse:
        """Convert LiteLLM response to LLMResponse."""
        content_blocks: list[ContentBlock] = []
        
        choice = response.choices[0]
        message = choice.message
        
        # Handle text content
        if message.content:
            content_blocks.append(TextContent(text=message.content))
        
        # Handle tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                import json
                try:
                    input_data = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    input_data = {}
                
                content_blocks.append(ToolUseContent(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    input=input_data,
                ))
        
        # Map finish reason
        finish_reason = choice.finish_reason
        stop_reason_map = {
            "stop": StopReason.END_TURN,
            "length": StopReason.MAX_TOKENS,
            "tool_calls": StopReason.TOOL_USE,
            "function_call": StopReason.TOOL_USE,
        }
        stop_reason = stop_reason_map.get(finish_reason, StopReason.END_TURN)
        
        # Extract usage
        usage = LLMUsage()
        if hasattr(response, "usage") and response.usage:
            usage = LLMUsage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
            )
        
        return LLMResponse(
            id=response.id if hasattr(response, "id") else str(uuid4()),
            model=response.model if hasattr(response, "model") else "unknown",
            content=content_blocks,
            stop_reason=stop_reason,
            usage=usage,
            duration_ms=duration_ms,
        )
    
    async def complete(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        """Generate a completion using LiteLLM."""
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "litellm package is required for LiteLLMProvider. "
                "Install with: pip install litellm"
            )
        
        api_messages = self._convert_messages(messages)
        
        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": config.model,
            "messages": api_messages,
            "max_tokens": config.max_tokens,
        }
        
        if config.temperature != 1.0:
            kwargs["temperature"] = config.temperature
        
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        
        if config.stop_sequences:
            kwargs["stop"] = config.stop_sequences
        
        if config.tools:
            # Convert to OpenAI function format
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": tool,
                }
                for tool in config.tools
            ]
        
        if self._api_key:
            kwargs["api_key"] = self._api_key
        
        if self._api_base:
            kwargs["api_base"] = self._api_base
        
        if self._custom_llm_provider:
            kwargs["custom_llm_provider"] = self._custom_llm_provider
        
        # Make request
        start_time = time.perf_counter()
        response = await litellm.acompletion(**kwargs)
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        return self._convert_response(response, duration_ms)
    
    async def stream(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> StreamingResponse:
        """Generate a streaming completion using LiteLLM."""
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "litellm package is required for LiteLLMProvider. "
                "Install with: pip install litellm"
            )
        
        api_messages = self._convert_messages(messages)
        
        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": config.model,
            "messages": api_messages,
            "max_tokens": config.max_tokens,
            "stream": True,
        }
        
        if config.temperature != 1.0:
            kwargs["temperature"] = config.temperature
        
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        
        if config.stop_sequences:
            kwargs["stop"] = config.stop_sequences
        
        if self._api_key:
            kwargs["api_key"] = self._api_key
        
        if self._api_base:
            kwargs["api_base"] = self._api_base
        
        async def event_generator() -> AsyncIterator[StreamEvent]:
            response = await litellm.acompletion(**kwargs)
            
            yield StreamEvent(type=StreamEventType.MESSAGE_START)
            yield StreamEvent(
                type=StreamEventType.CONTENT_BLOCK_START,
                data={"content_block": {"type": "text"}},
            )
            
            async for chunk in response:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        yield StreamEvent(
                            type=StreamEventType.TEXT_DELTA,
                            data={"text": delta.content},
                        )
            
            yield StreamEvent(type=StreamEventType.CONTENT_BLOCK_STOP)
            yield StreamEvent(type=StreamEventType.MESSAGE_STOP)
        
        return StreamingResponse(
            event_stream=event_generator(),
            model=config.model,
        )


class MockProvider(LLMProvider):
    """
    Mock provider for testing.
    
    Returns predefined responses for testing purposes.
    """
    
    def __init__(
        self,
        responses: list[LLMResponse] | None = None,
        default_response: str = "This is a mock response.",
    ) -> None:
        """
        Initialize the mock provider.
        
        Args:
            responses: List of responses to return in order
            default_response: Default response text
        """
        self._responses = list(responses) if responses else []
        self._default_response = default_response
        self._call_count = 0
    
    @property
    def name(self) -> str:
        return "mock"
    
    @property
    def supported_models(self) -> list[str]:
        return ["mock-model"]
    
    @property
    def call_count(self) -> int:
        """Get the number of calls made."""
        return self._call_count
    
    def add_response(self, response: LLMResponse) -> None:
        """Add a response to the queue."""
        self._responses.append(response)
    
    async def complete(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        """Return a mock response."""
        self._call_count += 1
        
        if self._responses:
            return self._responses.pop(0)
        
        return LLMResponse(
            model=config.model,
            content=[TextContent(text=self._default_response)],
            usage=LLMUsage(input_tokens=10, output_tokens=20),
        )
    
    async def stream(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> StreamingResponse:
        """Return a mock streaming response."""
        self._call_count += 1
        
        response_text = self._default_response
        if self._responses:
            response = self._responses.pop(0)
            response_text = response.get_text()
        
        async def event_generator() -> AsyncIterator[StreamEvent]:
            yield StreamEvent(type=StreamEventType.MESSAGE_START)
            yield StreamEvent(
                type=StreamEventType.CONTENT_BLOCK_START,
                data={"content_block": {"type": "text"}},
            )
            
            # Stream word by word
            words = response_text.split()
            for i, word in enumerate(words):
                text = word + (" " if i < len(words) - 1 else "")
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    data={"text": text},
                )
            
            yield StreamEvent(type=StreamEventType.CONTENT_BLOCK_STOP)
            yield StreamEvent(type=StreamEventType.MESSAGE_STOP)
        
        return StreamingResponse(
            event_stream=event_generator(),
            model=config.model,
        )
