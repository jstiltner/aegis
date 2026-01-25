"""
Streaming support for LLM responses.

This module provides streaming response handling for real-time
token-by-token output from LLMs.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Callable, Awaitable
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from .models import (
    LLMResponse,
    LLMUsage,
    StopReason,
    ContentBlock,
    TextContent,
    ToolUseContent,
)


class StreamEventType(str, Enum):
    """Types of streaming events."""
    
    # Message lifecycle
    MESSAGE_START = "message_start"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_STOP = "message_stop"
    
    # Content blocks
    CONTENT_BLOCK_START = "content_block_start"
    CONTENT_BLOCK_DELTA = "content_block_delta"
    CONTENT_BLOCK_STOP = "content_block_stop"
    
    # Text
    TEXT_DELTA = "text_delta"
    
    # Tool use
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_DELTA = "tool_use_delta"
    TOOL_USE_STOP = "tool_use_stop"
    
    # Errors
    ERROR = "error"
    
    # Ping/keepalive
    PING = "ping"


class StreamEvent(BaseModel):
    """
    A streaming event from an LLM.
    
    Events are emitted during streaming to provide real-time updates
    on the generation progress.
    """
    
    model_config = ConfigDict(frozen=True)
    
    type: StreamEventType = Field(..., description="Event type")
    index: int = Field(default=0, description="Content block index")
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Event data",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the event occurred",
    )
    
    # Convenience accessors
    
    @property
    def text(self) -> str | None:
        """Get text delta if this is a text event."""
        if self.type == StreamEventType.TEXT_DELTA:
            return self.data.get("text", "")
        return None
    
    @property
    def tool_name(self) -> str | None:
        """Get tool name if this is a tool use event."""
        if self.type == StreamEventType.TOOL_USE_START:
            return self.data.get("name")
        return None
    
    @property
    def tool_input_delta(self) -> str | None:
        """Get tool input delta if this is a tool use delta event."""
        if self.type == StreamEventType.TOOL_USE_DELTA:
            return self.data.get("partial_json", "")
        return None
    
    @property
    def error_message(self) -> str | None:
        """Get error message if this is an error event."""
        if self.type == StreamEventType.ERROR:
            return self.data.get("message")
        return None
    
    @property
    def stop_reason(self) -> StopReason | None:
        """Get stop reason if this is a message stop event."""
        if self.type == StreamEventType.MESSAGE_STOP:
            reason = self.data.get("stop_reason")
            if reason:
                return StopReason(reason)
        return None
    
    @property
    def usage(self) -> LLMUsage | None:
        """Get usage if this is a message delta/stop event."""
        if self.type in (StreamEventType.MESSAGE_DELTA, StreamEventType.MESSAGE_STOP):
            usage_data = self.data.get("usage")
            if usage_data:
                return LLMUsage(
                    input_tokens=usage_data.get("input_tokens", 0),
                    output_tokens=usage_data.get("output_tokens", 0),
                )
        return None


class StreamingResponse:
    """
    A streaming response from an LLM.
    
    Provides async iteration over streaming events and accumulates
    the final response.
    
    Example:
        >>> async with client.stream(messages, config) as stream:
        ...     async for event in stream:
        ...         if event.type == StreamEventType.TEXT_DELTA:
        ...             print(event.text, end="", flush=True)
        ...     response = stream.get_final_response()
    """
    
    def __init__(
        self,
        event_stream: AsyncIterator[StreamEvent],
        model: str,
        response_id: str | None = None,
    ) -> None:
        """
        Initialize the streaming response.
        
        Args:
            event_stream: Async iterator of stream events
            model: Model identifier
            response_id: Optional response ID
        """
        self._event_stream = event_stream
        self._model = model
        self._response_id = response_id or str(uuid4())
        
        # Accumulated state
        self._content_blocks: list[ContentBlock] = []
        self._current_text = ""
        self._current_tool_use: dict[str, Any] | None = None
        self._current_tool_input = ""
        self._stop_reason: StopReason = StopReason.END_TURN
        self._usage = LLMUsage()
        self._started_at = datetime.now(timezone.utc)
        self._finished = False
        self._error: str | None = None
        
        # Callbacks
        self._on_text: list[Callable[[str], Awaitable[None]]] = []
        self._on_tool_use: list[Callable[[ToolUseContent], Awaitable[None]]] = []
        self._on_complete: list[Callable[[LLMResponse], Awaitable[None]]] = []
    
    @property
    def response_id(self) -> str:
        """Get the response ID."""
        return self._response_id
    
    @property
    def model(self) -> str:
        """Get the model identifier."""
        return self._model
    
    @property
    def is_finished(self) -> bool:
        """Check if streaming is finished."""
        return self._finished
    
    @property
    def accumulated_text(self) -> str:
        """Get all accumulated text so far."""
        return self._current_text
    
    @property
    def error(self) -> str | None:
        """Get error message if an error occurred."""
        return self._error
    
    def on_text(self, callback: Callable[[str], Awaitable[None]]) -> None:
        """Register a callback for text deltas."""
        self._on_text.append(callback)
    
    def on_tool_use(self, callback: Callable[[ToolUseContent], Awaitable[None]]) -> None:
        """Register a callback for tool use completions."""
        self._on_tool_use.append(callback)
    
    def on_complete(self, callback: Callable[[LLMResponse], Awaitable[None]]) -> None:
        """Register a callback for completion."""
        self._on_complete.append(callback)
    
    async def __aiter__(self) -> AsyncIterator[StreamEvent]:
        """Iterate over stream events."""
        async for event in self._event_stream:
            # Process the event
            await self._process_event(event)
            
            # Yield to caller
            yield event
            
            # Check for completion
            if event.type == StreamEventType.MESSAGE_STOP:
                self._finished = True
                
                # Call completion callbacks
                response = self.get_final_response()
                for callback in self._on_complete:
                    try:
                        await callback(response)
                    except Exception:
                        pass
    
    async def _process_event(self, event: StreamEvent) -> None:
        """Process a stream event and update state."""
        if event.type == StreamEventType.TEXT_DELTA:
            text = event.text or ""
            self._current_text += text
            
            # Call text callbacks
            for callback in self._on_text:
                try:
                    await callback(text)
                except Exception:
                    pass
        
        elif event.type == StreamEventType.CONTENT_BLOCK_START:
            block_type = event.data.get("content_block", {}).get("type")
            if block_type == "text":
                self._current_text = ""
            elif block_type == "tool_use":
                self._current_tool_use = event.data.get("content_block", {})
                self._current_tool_input = ""
        
        elif event.type == StreamEventType.CONTENT_BLOCK_STOP:
            # Finalize the current content block
            if self._current_tool_use:
                # Parse accumulated JSON input
                import json
                try:
                    input_data = json.loads(self._current_tool_input) if self._current_tool_input else {}
                except json.JSONDecodeError:
                    input_data = {}
                
                tool_use = ToolUseContent(
                    id=self._current_tool_use.get("id", str(uuid4())),
                    name=self._current_tool_use.get("name", ""),
                    input=input_data,
                )
                self._content_blocks.append(tool_use)
                
                # Call tool use callbacks
                for callback in self._on_tool_use:
                    try:
                        await callback(tool_use)
                    except Exception:
                        pass
                
                self._current_tool_use = None
                self._current_tool_input = ""
            elif self._current_text:
                self._content_blocks.append(TextContent(text=self._current_text))
        
        elif event.type == StreamEventType.TOOL_USE_DELTA:
            self._current_tool_input += event.tool_input_delta or ""
        
        elif event.type == StreamEventType.MESSAGE_DELTA:
            if event.stop_reason:
                self._stop_reason = event.stop_reason
            if event.usage:
                self._usage = event.usage
        
        elif event.type == StreamEventType.MESSAGE_STOP:
            if event.stop_reason:
                self._stop_reason = event.stop_reason
            if event.usage:
                self._usage = event.usage
        
        elif event.type == StreamEventType.ERROR:
            self._error = event.error_message
    
    def get_final_response(self) -> LLMResponse:
        """
        Get the final accumulated response.
        
        Returns:
            LLMResponse with all accumulated content
        """
        # Add any remaining text
        if self._current_text and not any(
            isinstance(b, TextContent) and b.text == self._current_text
            for b in self._content_blocks
        ):
            self._content_blocks.append(TextContent(text=self._current_text))
        
        duration_ms = (datetime.now(timezone.utc) - self._started_at).total_seconds() * 1000
        
        return LLMResponse(
            id=self._response_id,
            model=self._model,
            content=self._content_blocks,
            stop_reason=self._stop_reason,
            usage=self._usage,
            created_at=self._started_at,
            duration_ms=duration_ms,
        )
    
    async def collect(self) -> LLMResponse:
        """
        Collect all events and return the final response.
        
        This consumes the entire stream.
        
        Returns:
            LLMResponse with all content
        """
        async for _ in self:
            pass
        return self.get_final_response()
    
    async def collect_text(self) -> str:
        """
        Collect all text from the stream.
        
        Returns:
            All accumulated text
        """
        await self.collect()
        return self._current_text


class StreamBuffer:
    """
    Buffer for accumulating stream events.
    
    Useful for replaying streams or buffering for batch processing.
    """
    
    def __init__(self) -> None:
        self._events: list[StreamEvent] = []
        self._text = ""
        self._tool_uses: list[ToolUseContent] = []
    
    @property
    def events(self) -> list[StreamEvent]:
        """Get all buffered events."""
        return self._events.copy()
    
    @property
    def text(self) -> str:
        """Get accumulated text."""
        return self._text
    
    @property
    def tool_uses(self) -> list[ToolUseContent]:
        """Get accumulated tool uses."""
        return self._tool_uses.copy()
    
    def add_event(self, event: StreamEvent) -> None:
        """Add an event to the buffer."""
        self._events.append(event)
        
        if event.type == StreamEventType.TEXT_DELTA and event.text:
            self._text += event.text
    
    def add_tool_use(self, tool_use: ToolUseContent) -> None:
        """Add a completed tool use."""
        self._tool_uses.append(tool_use)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self._events.clear()
        self._text = ""
        self._tool_uses.clear()
    
    async def replay(self) -> AsyncIterator[StreamEvent]:
        """Replay buffered events."""
        for event in self._events:
            yield event


async def merge_streams(
    *streams: AsyncIterator[StreamEvent],
) -> AsyncIterator[StreamEvent]:
    """
    Merge multiple stream event iterators.
    
    Events are yielded as they arrive from any stream.
    
    Args:
        *streams: Stream iterators to merge
        
    Yields:
        Events from all streams
    """
    pending: set[asyncio.Task[tuple[int, StreamEvent | None]]] = set()
    stream_iters = list(streams)
    
    async def get_next(index: int, stream: AsyncIterator[StreamEvent]) -> tuple[int, StreamEvent | None]:
        try:
            event = await stream.__anext__()
            return (index, event)
        except StopAsyncIteration:
            return (index, None)
    
    # Start initial tasks
    for i, stream in enumerate(stream_iters):
        task = asyncio.create_task(get_next(i, stream))
        pending.add(task)
    
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        
        for task in done:
            index, event = task.result()
            
            if event is not None:
                yield event
                
                # Schedule next event from this stream
                new_task = asyncio.create_task(get_next(index, stream_iters[index]))
                pending.add(new_task)
