"""
Audit logger for structured event logging.

This module provides:
- Centralized audit logging
- Multiple output handlers
- Async logging support
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable
from queue import Queue
from threading import Thread

from pydantic import BaseModel, Field, ConfigDict

from .events import AuditEvent, AuditEventType, AuditEventSeverity, EventFactory
from .context import current_trace, current_span


class AuditLogConfig(BaseModel):
    """Configuration for the audit logger."""
    
    model_config = ConfigDict(frozen=True)
    
    # Logging levels
    min_severity: AuditEventSeverity = Field(
        default=AuditEventSeverity.DEBUG,
        description="Minimum severity to log",
    )
    
    # Buffering
    buffer_size: int = Field(
        default=1000,
        ge=1,
        description="Maximum events to buffer",
    )
    flush_interval_seconds: float = Field(
        default=5.0,
        ge=0.1,
        description="Interval between flushes",
    )
    
    # Filtering
    include_event_types: tuple[AuditEventType, ...] | None = Field(
        default=None,
        description="Event types to include (None = all)",
    )
    exclude_event_types: tuple[AuditEventType, ...] = Field(
        default_factory=tuple,
        description="Event types to exclude",
    )
    include_tags: tuple[str, ...] | None = Field(
        default=None,
        description="Tags to include (None = all)",
    )
    exclude_tags: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Tags to exclude",
    )
    
    # Context
    include_trace_context: bool = Field(
        default=True,
        description="Whether to include trace context",
    )
    
    # Output
    log_to_console: bool = Field(
        default=True,
        description="Whether to log to console",
    )
    console_format: str = Field(
        default="[{timestamp}] [{severity}] {event_type}: {message}",
        description="Console log format",
    )


class AuditHandler(ABC):
    """Abstract base class for audit event handlers."""
    
    @abstractmethod
    async def handle(self, event: AuditEvent) -> None:
        """
        Handle an audit event.
        
        Args:
            event: The event to handle
        """
        ...
    
    @abstractmethod
    async def flush(self) -> None:
        """Flush any buffered events."""
        ...
    
    @abstractmethod
    async def close(self) -> None:
        """Close the handler and release resources."""
        ...


class ConsoleHandler(AuditHandler):
    """Handler that logs events to the console."""
    
    def __init__(
        self,
        format_string: str = "[{timestamp}] [{severity}] {event_type}: {message}",
        use_colors: bool = True,
    ) -> None:
        self.format_string = format_string
        self.use_colors = use_colors
        
        # Color codes
        self._colors = {
            AuditEventSeverity.DEBUG: "\033[90m",  # Gray
            AuditEventSeverity.INFO: "\033[0m",    # Default
            AuditEventSeverity.WARNING: "\033[93m", # Yellow
            AuditEventSeverity.ERROR: "\033[91m",   # Red
            AuditEventSeverity.CRITICAL: "\033[95m", # Magenta
        }
        self._reset = "\033[0m"
    
    async def handle(self, event: AuditEvent) -> None:
        """Log event to console."""
        formatted = self.format_string.format(
            timestamp=event.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            severity=event.severity.value.upper(),
            event_type=event.event_type.value,
            message=event.message,
            agent_id=event.agent_id or "-",
            session_id=event.session_id or "-",
            trace_id=event.trace_id or "-",
            span_id=event.span_id or "-",
        )
        
        if self.use_colors:
            color = self._colors.get(event.severity, "")
            formatted = f"{color}{formatted}{self._reset}"
        
        print(formatted)
    
    async def flush(self) -> None:
        """No buffering, nothing to flush."""
        pass
    
    async def close(self) -> None:
        """Nothing to close."""
        pass


class CallbackHandler(AuditHandler):
    """Handler that calls a callback function for each event."""
    
    def __init__(
        self,
        callback: Callable[[AuditEvent], Awaitable[None]],
    ) -> None:
        self.callback = callback
    
    async def handle(self, event: AuditEvent) -> None:
        """Call the callback with the event."""
        await self.callback(event)
    
    async def flush(self) -> None:
        """Nothing to flush."""
        pass
    
    async def close(self) -> None:
        """Nothing to close."""
        pass


class BufferedHandler(AuditHandler):
    """Handler that buffers events before passing to another handler."""
    
    def __init__(
        self,
        inner_handler: AuditHandler,
        buffer_size: int = 100,
        flush_interval_seconds: float = 5.0,
    ) -> None:
        self.inner_handler = inner_handler
        self.buffer_size = buffer_size
        self.flush_interval_seconds = flush_interval_seconds
        
        self._buffer: list[AuditEvent] = []
        self._flush_task: asyncio.Task[None] | None = None
        self._closed = False
    
    async def handle(self, event: AuditEvent) -> None:
        """Buffer the event."""
        if self._closed:
            return
        
        self._buffer.append(event)
        
        if len(self._buffer) >= self.buffer_size:
            await self.flush()
    
    async def flush(self) -> None:
        """Flush buffered events to inner handler."""
        if not self._buffer:
            return
        
        events = self._buffer
        self._buffer = []
        
        for event in events:
            await self.inner_handler.handle(event)
        
        await self.inner_handler.flush()
    
    async def close(self) -> None:
        """Flush and close."""
        self._closed = True
        await self.flush()
        await self.inner_handler.close()
    
    async def start_periodic_flush(self) -> None:
        """Start periodic flushing."""
        async def flush_loop():
            while not self._closed:
                await asyncio.sleep(self.flush_interval_seconds)
                await self.flush()
        
        self._flush_task = asyncio.create_task(flush_loop())
    
    async def stop_periodic_flush(self) -> None:
        """Stop periodic flushing."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None


class AuditLogger:
    """
    Centralized audit logger.
    
    The AuditLogger:
    - Collects audit events from the runtime
    - Filters events based on configuration
    - Dispatches events to handlers
    - Provides convenience methods for common events
    
    Example:
        >>> logger = AuditLogger()
        >>> logger.add_handler(ConsoleHandler())
        >>> 
        >>> await logger.log_event(
        ...     AuditEventType.TOOL_INVOKED,
        ...     "Tool search invoked",
        ...     data={"tool_name": "search"},
        ... )
    """
    
    def __init__(
        self,
        config: AuditLogConfig | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """
        Initialize the audit logger.
        
        Args:
            config: Logger configuration
            agent_id: Default agent ID
            session_id: Default session ID
        """
        self.config = config or AuditLogConfig()
        self.agent_id = agent_id
        self.session_id = session_id
        
        self._handlers: list[AuditHandler] = []
        self._event_factory = EventFactory(
            agent_id=agent_id,
            session_id=session_id,
        )
        
        # Add console handler if configured
        if self.config.log_to_console:
            self._handlers.append(ConsoleHandler(
                format_string=self.config.console_format,
            ))
    
    @property
    def event_factory(self) -> EventFactory:
        """Get the event factory."""
        return self._event_factory
    
    def add_handler(self, handler: AuditHandler) -> None:
        """Add an event handler."""
        self._handlers.append(handler)
    
    def remove_handler(self, handler: AuditHandler) -> None:
        """Remove an event handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)
    
    def _should_log(self, event: AuditEvent) -> bool:
        """Check if an event should be logged."""
        # Check severity
        severity_order = [
            AuditEventSeverity.DEBUG,
            AuditEventSeverity.INFO,
            AuditEventSeverity.WARNING,
            AuditEventSeverity.ERROR,
            AuditEventSeverity.CRITICAL,
        ]
        
        if severity_order.index(event.severity) < severity_order.index(self.config.min_severity):
            return False
        
        # Check event type inclusion
        if self.config.include_event_types is not None:
            if event.event_type not in self.config.include_event_types:
                return False
        
        # Check event type exclusion
        if event.event_type in self.config.exclude_event_types:
            return False
        
        # Check tag inclusion
        if self.config.include_tags is not None:
            if not any(tag in self.config.include_tags for tag in event.tags):
                return False
        
        # Check tag exclusion
        if any(tag in self.config.exclude_tags for tag in event.tags):
            return False
        
        return True
    
    def _enrich_event(self, event: AuditEvent) -> AuditEvent:
        """Enrich event with context."""
        updates: dict[str, Any] = {}
        
        # Add trace context if configured
        if self.config.include_trace_context:
            trace = current_trace()
            if trace:
                updates["trace_id"] = trace.trace_id
                span = trace.current_span
                if span:
                    updates["span_id"] = span.span_id
                    updates["parent_span_id"] = span.parent_span_id
        
        # Add default agent/session if not set
        if event.agent_id is None and self.agent_id:
            updates["agent_id"] = self.agent_id
        if event.session_id is None and self.session_id:
            updates["session_id"] = self.session_id
        
        if updates:
            # Create new event with updates
            return AuditEvent(
                event_id=event.event_id,
                event_type=event.event_type,
                severity=event.severity,
                timestamp=event.timestamp,
                trace_id=updates.get("trace_id", event.trace_id),
                span_id=updates.get("span_id", event.span_id),
                parent_span_id=updates.get("parent_span_id", event.parent_span_id),
                agent_id=updates.get("agent_id", event.agent_id),
                session_id=updates.get("session_id", event.session_id),
                message=event.message,
                data=event.data,
                tags=event.tags,
                source=event.source,
            )
        
        return event
    
    async def log(self, event: AuditEvent) -> None:
        """
        Log an audit event.
        
        Args:
            event: The event to log
        """
        if not self._should_log(event):
            return
        
        event = self._enrich_event(event)
        
        for handler in self._handlers:
            try:
                await handler.handle(event)
            except Exception:
                # Don't let handler errors break logging
                pass
    
    async def log_event(
        self,
        event_type: AuditEventType,
        message: str,
        severity: AuditEventSeverity = AuditEventSeverity.INFO,
        data: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """
        Create and log an event.
        
        Args:
            event_type: Type of event
            message: Event message
            severity: Event severity
            data: Event data
            tags: Event tags
            **kwargs: Additional event fields
            
        Returns:
            The created event
        """
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            data=data or {},
            tags=tuple(tags or []),
            agent_id=kwargs.get("agent_id", self.agent_id),
            session_id=kwargs.get("session_id", self.session_id),
            **{k: v for k, v in kwargs.items() if k not in ("agent_id", "session_id")},
        )
        
        await self.log(event)
        return event
    
    async def flush(self) -> None:
        """Flush all handlers."""
        for handler in self._handlers:
            try:
                await handler.flush()
            except Exception:
                pass
    
    async def close(self) -> None:
        """Close all handlers."""
        for handler in self._handlers:
            try:
                await handler.close()
            except Exception:
                pass
    
    # Convenience methods
    
    async def debug(self, message: str, **kwargs: Any) -> AuditEvent:
        """Log a debug event."""
        return await self.log_event(
            AuditEventType.SYSTEM_INFO,
            message,
            severity=AuditEventSeverity.DEBUG,
            **kwargs,
        )
    
    async def info(self, message: str, **kwargs: Any) -> AuditEvent:
        """Log an info event."""
        return await self.log_event(
            AuditEventType.SYSTEM_INFO,
            message,
            severity=AuditEventSeverity.INFO,
            **kwargs,
        )
    
    async def warning(self, message: str, **kwargs: Any) -> AuditEvent:
        """Log a warning event."""
        return await self.log_event(
            AuditEventType.SYSTEM_WARNING,
            message,
            severity=AuditEventSeverity.WARNING,
            **kwargs,
        )
    
    async def error(self, message: str, **kwargs: Any) -> AuditEvent:
        """Log an error event."""
        return await self.log_event(
            AuditEventType.SYSTEM_ERROR,
            message,
            severity=AuditEventSeverity.ERROR,
            **kwargs,
        )
    
    # Tool events
    
    async def tool_invoked(
        self,
        tool_name: str,
        invocation_id: str,
        arguments: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log a tool invocation."""
        event = self._event_factory.tool_invoked(
            tool_name=tool_name,
            invocation_id=invocation_id,
            arguments=arguments,
            **kwargs,
        )
        await self.log(event)
        return event
    
    async def tool_completed(
        self,
        tool_name: str,
        invocation_id: str,
        duration_ms: float,
        output_summary: str | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log a tool completion."""
        event = self._event_factory.tool_completed(
            tool_name=tool_name,
            invocation_id=invocation_id,
            duration_ms=duration_ms,
            output_summary=output_summary,
            **kwargs,
        )
        await self.log(event)
        return event
    
    async def tool_failed(
        self,
        tool_name: str,
        invocation_id: str,
        error: str,
        error_type: str = "Error",
        **kwargs: Any,
    ) -> AuditEvent:
        """Log a tool failure."""
        event = self._event_factory.tool_failed(
            tool_name=tool_name,
            invocation_id=invocation_id,
            error=error,
            error_type=error_type,
            **kwargs,
        )
        await self.log(event)
        return event
    
    # Policy events
    
    async def policy_denied(
        self,
        tool_name: str,
        policy_name: str,
        reason: str,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log a policy denial."""
        event = self._event_factory.policy_denied(
            tool_name=tool_name,
            policy_name=policy_name,
            reason=reason,
            **kwargs,
        )
        await self.log(event)
        return event
    
    # State events
    
    async def state_changed(
        self,
        old_state: str,
        new_state: str,
        trigger: str | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log a state change."""
        event = self._event_factory.state_changed(
            old_state=old_state,
            new_state=new_state,
            trigger=trigger,
            **kwargs,
        )
        await self.log(event)
        return event
    
    async def state_checkpoint(
        self,
        checkpoint_id: str,
        state_name: str,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log a checkpoint creation."""
        event = self._event_factory.state_checkpoint(
            checkpoint_id=checkpoint_id,
            state_name=state_name,
            **kwargs,
        )
        await self.log(event)
        return event
