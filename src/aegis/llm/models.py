"""
LLM data models.

This module provides data models for LLM interactions including
messages, responses, tool use, and configuration.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


class MessageRole(str, Enum):
    """Role of a message in a conversation."""
    
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentType(str, Enum):
    """Type of content in a message."""
    
    TEXT = "text"
    IMAGE = "image"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


class TextContent(BaseModel):
    """Text content in a message."""
    
    model_config = ConfigDict(frozen=True)
    
    type: Literal["text"] = "text"
    text: str = Field(..., description="The text content")


class ImageContent(BaseModel):
    """Image content in a message."""
    
    model_config = ConfigDict(frozen=True)
    
    type: Literal["image"] = "image"
    source: dict[str, Any] = Field(
        ...,
        description="Image source (base64 or URL)",
    )
    media_type: str = Field(
        default="image/png",
        description="MIME type of the image",
    )


class ToolUseContent(BaseModel):
    """Tool use request in a message."""
    
    model_config = ConfigDict(frozen=True)
    
    type: Literal["tool_use"] = "tool_use"
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique ID for this tool use",
    )
    name: str = Field(..., description="Name of the tool to use")
    input: dict[str, Any] = Field(
        default_factory=dict,
        description="Input arguments for the tool",
    )


class ToolResultContent(BaseModel):
    """Tool result in a message."""
    
    model_config = ConfigDict(frozen=True)
    
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str = Field(..., description="ID of the tool use this is a result for")
    content: str | list[dict[str, Any]] = Field(
        ...,
        description="Result content",
    )
    is_error: bool = Field(
        default=False,
        description="Whether this is an error result",
    )


# Union type for content
ContentBlock = TextContent | ImageContent | ToolUseContent | ToolResultContent


class LLMMessage(BaseModel):
    """
    A message in an LLM conversation.
    
    Supports text, images, and tool use/results.
    """
    
    model_config = ConfigDict(frozen=True)
    
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique message ID",
    )
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str | list[ContentBlock] = Field(
        ...,
        description="Message content (string or content blocks)",
    )
    name: str | None = Field(
        default=None,
        description="Optional name for the message sender",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    
    @classmethod
    def system(cls, content: str, **kwargs: Any) -> LLMMessage:
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content, **kwargs)
    
    @classmethod
    def user(cls, content: str | list[ContentBlock], **kwargs: Any) -> LLMMessage:
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content, **kwargs)
    
    @classmethod
    def assistant(cls, content: str | list[ContentBlock], **kwargs: Any) -> LLMMessage:
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content, **kwargs)
    
    @classmethod
    def tool_result(
        cls,
        tool_use_id: str,
        content: str,
        is_error: bool = False,
        **kwargs: Any,
    ) -> LLMMessage:
        """Create a tool result message."""
        return cls(
            role=MessageRole.USER,
            content=[ToolResultContent(
                tool_use_id=tool_use_id,
                content=content,
                is_error=is_error,
            )],
            **kwargs,
        )
    
    def get_text(self, separator: str = "") -> str:
        """Get the text content of the message."""
        if isinstance(self.content, str):
            return self.content
        
        texts = []
        for block in self.content:
            if isinstance(block, TextContent):
                texts.append(block.text)
        return separator.join(texts)
    
    def get_tool_uses(self) -> list[ToolUseContent]:
        """Get all tool use requests in the message."""
        if isinstance(self.content, str):
            return []
        
        return [
            block for block in self.content
            if isinstance(block, ToolUseContent)
        ]
    
    def has_tool_use(self) -> bool:
        """Check if the message contains tool use requests."""
        return len(self.get_tool_uses()) > 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API calls."""
        result: dict[str, Any] = {
            "role": self.role.value,
        }
        
        if isinstance(self.content, str):
            result["content"] = self.content
        else:
            result["content"] = [
                block.model_dump() for block in self.content
            ]
        
        if self.name:
            result["name"] = self.name
        
        return result


class LLMUsage(BaseModel):
    """Token usage information."""
    
    model_config = ConfigDict(frozen=True)
    
    input_tokens: int = Field(default=0, ge=0, description="Input tokens used")
    output_tokens: int = Field(default=0, ge=0, description="Output tokens generated")
    cache_read_tokens: int = Field(default=0, ge=0, description="Tokens read from cache")
    cache_write_tokens: int = Field(default=0, ge=0, description="Tokens written to cache")
    
    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.input_tokens + self.output_tokens
    
    def __add__(self, other: LLMUsage) -> LLMUsage:
        """Add two usage objects."""
        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


class StopReason(str, Enum):
    """Reason for stopping generation."""
    
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"
    ERROR = "error"


class LLMResponse(BaseModel):
    """
    Response from an LLM.
    
    Contains the generated content, usage information, and metadata.
    """
    
    model_config = ConfigDict(frozen=True)
    
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique response ID",
    )
    model: str = Field(..., description="Model that generated the response")
    content: list[ContentBlock] = Field(
        default_factory=list,
        description="Response content blocks",
    )
    stop_reason: StopReason = Field(
        default=StopReason.END_TURN,
        description="Reason for stopping",
    )
    usage: LLMUsage = Field(
        default_factory=LLMUsage,
        description="Token usage",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the response was created",
    )
    duration_ms: float | None = Field(
        default=None,
        description="Generation duration in milliseconds",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    
    def get_text(self, separator: str = "") -> str:
        """Get the text content of the response."""
        texts = []
        for block in self.content:
            if isinstance(block, TextContent):
                texts.append(block.text)
        return separator.join(texts)
    
    def get_tool_uses(self) -> list[ToolUseContent]:
        """Get all tool use requests in the response."""
        return [
            block for block in self.content
            if isinstance(block, ToolUseContent)
        ]
    
    def has_tool_use(self) -> bool:
        """Check if the response contains tool use requests."""
        return self.stop_reason == StopReason.TOOL_USE or len(self.get_tool_uses()) > 0


class ToolUseRequest(BaseModel):
    """
    A request to use a tool.
    
    Extracted from LLM responses for easier handling.
    """
    
    model_config = ConfigDict(frozen=True)
    
    id: str = Field(..., description="Tool use ID")
    name: str = Field(..., description="Tool name")
    input: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool input arguments",
    )
    
    @classmethod
    def from_content(cls, content: ToolUseContent) -> ToolUseRequest:
        """Create from a ToolUseContent block."""
        return cls(
            id=content.id,
            name=content.name,
            input=content.input,
        )


class ToolUseResult(BaseModel):
    """
    Result of a tool use.
    
    Used to construct tool result messages.
    """
    
    model_config = ConfigDict(frozen=True)
    
    tool_use_id: str = Field(..., description="ID of the tool use")
    output: str = Field(..., description="Tool output")
    is_error: bool = Field(default=False, description="Whether this is an error")
    
    def to_message(self) -> LLMMessage:
        """Convert to an LLM message."""
        return LLMMessage.tool_result(
            tool_use_id=self.tool_use_id,
            content=self.output,
            is_error=self.is_error,
        )


class LLMConfig(BaseModel):
    """
    Configuration for LLM requests.
    
    Attributes:
        model: Model identifier
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        stop_sequences: Sequences that stop generation
        system: System prompt
        tools: Available tools
    """
    
    model_config = ConfigDict(frozen=True)
    
    model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Model identifier",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        description="Maximum tokens to generate",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Top-k sampling parameter",
    )
    stop_sequences: list[str] = Field(
        default_factory=list,
        description="Sequences that stop generation",
    )
    system: str | None = Field(
        default=None,
        description="System prompt",
    )
    tools: list[dict[str, Any]] | None = Field(
        default=None,
        description="Available tools",
    )
    tool_choice: dict[str, Any] | str | None = Field(
        default=None,
        description="Tool choice configuration",
    )
    
    # Provider-specific
    timeout_seconds: float = Field(
        default=60.0,
        ge=0,
        description="Request timeout",
    )
    retry_count: int = Field(
        default=3,
        ge=0,
        description="Number of retries on failure",
    )
    
    def with_tools(self, tools: list[dict[str, Any]]) -> LLMConfig:
        """Create a new config with tools."""
        return LLMConfig(
            **{**self.model_dump(), "tools": tools}
        )
    
    def with_system(self, system: str) -> LLMConfig:
        """Create a new config with system prompt."""
        return LLMConfig(
            **{**self.model_dump(), "system": system}
        )


class ConversationTurn(BaseModel):
    """
    A turn in a conversation.
    
    Contains the user message, assistant response, and any tool interactions.
    """
    
    model_config = ConfigDict(frozen=True)
    
    user_message: LLMMessage = Field(..., description="User message")
    assistant_response: LLMResponse = Field(..., description="Assistant response")
    tool_uses: list[ToolUseRequest] = Field(
        default_factory=list,
        description="Tool uses in this turn",
    )
    tool_results: list[ToolUseResult] = Field(
        default_factory=list,
        description="Tool results in this turn",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this turn occurred",
    )


class Conversation(BaseModel):
    """
    A conversation with an LLM.
    
    Tracks messages, turns, and usage across the conversation.
    """
    
    model_config = ConfigDict(validate_assignment=True)
    
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Conversation ID",
    )
    system_prompt: str | None = Field(
        default=None,
        description="System prompt for the conversation",
    )
    messages: list[LLMMessage] = Field(
        default_factory=list,
        description="All messages in the conversation",
    )
    turns: list[ConversationTurn] = Field(
        default_factory=list,
        description="Conversation turns",
    )
    total_usage: LLMUsage = Field(
        default_factory=LLMUsage,
        description="Total token usage",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the conversation started",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    
    def add_message(self, message: LLMMessage) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
    
    def add_response(self, response: LLMResponse) -> None:
        """Add a response and update usage."""
        # Create assistant message from response
        message = LLMMessage.assistant(content=list(response.content))
        self.messages.append(message)
        self.total_usage = self.total_usage + response.usage
    
    def get_last_message(self) -> LLMMessage | None:
        """Get the last message in the conversation."""
        if not self.messages:
            return None
        return self.messages[-1]
    
    def clear(self) -> None:
        """Clear all messages from the conversation."""
        self.messages = []
        self.turns = []
    
    @property
    def estimated_tokens(self) -> int:
        """Estimate the number of tokens in the conversation."""
        total = 0
        for msg in self.messages:
            text = msg.get_text()
            # Rough estimate: ~4 characters per token
            total += len(text) // 4 + 1
        return total
    
    def get_messages_for_api(self) -> list[dict[str, Any]]:
        """Get messages formatted for API calls."""
        return [msg.to_dict() for msg in self.messages]
