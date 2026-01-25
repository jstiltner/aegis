"""
LLM integration for the agent runtime.

This package provides:
- Unified LLM interface supporting multiple providers
- Anthropic Claude integration
- Local model support via LiteLLM
- Streaming and non-streaming responses
- Token counting and cost tracking
"""

from __future__ import annotations

from .models import (
    LLMMessage,
    LLMResponse,
    LLMConfig,
    LLMUsage,
    Conversation,
    MessageRole,
    StopReason,
    ContentBlock,
    TextContent,
    ToolUseContent,
    ToolResultContent,
)
from .providers import (
    LLMProvider,
    AnthropicProvider,
    LiteLLMProvider,
    MockProvider,
)
from .client import (
    LLMClient,
    LLMClientConfig,
    ToolExecutor,
    create_client,
)
from .streaming import (
    StreamingResponse,
    StreamEvent,
    StreamEventType,
    StreamBuffer,
)

__all__ = [
    # Models
    "LLMMessage",
    "LLMResponse",
    "LLMConfig",
    "LLMUsage",
    "Conversation",
    "MessageRole",
    "StopReason",
    "ContentBlock",
    "TextContent",
    "ToolUseContent",
    "ToolResultContent",
    # Providers
    "LLMProvider",
    "AnthropicProvider",
    "LiteLLMProvider",
    "MockProvider",
    # Client
    "LLMClient",
    "LLMClientConfig",
    "ToolExecutor",
    "create_client",
    # Streaming
    "StreamingResponse",
    "StreamEvent",
    "StreamEventType",
    "StreamBuffer",
]
