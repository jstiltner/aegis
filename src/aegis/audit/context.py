"""
Trace context for distributed tracing.

This module provides:
- Trace and span context management
- Context propagation utilities
- Async context variables
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator
from uuid import uuid4
from contextlib import contextmanager


@dataclass
class SpanContext:
    """
    Context for a single span within a trace.
    
    A span represents a unit of work within a trace.
    """
    
    span_id: str = field(default_factory=lambda: str(uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    parent_span_id: str | None = None
    
    # Span metadata
    name: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    
    # Attributes
    attributes: dict[str, Any] = field(default_factory=dict)
    
    # Status
    status: str = "ok"  # ok, error, cancelled
    error_message: str | None = None
    
    @property
    def duration_ms(self) -> float | None:
        """Get span duration in milliseconds."""
        if self.ended_at is None:
            return None
        delta = self.ended_at - self.started_at
        return delta.total_seconds() * 1000
    
    @property
    def is_root(self) -> bool:
        """Check if this is a root span."""
        return self.parent_span_id is None
    
    def end(self, status: str = "ok", error_message: str | None = None) -> None:
        """End the span."""
        self.ended_at = datetime.now(timezone.utc)
        self.status = status
        self.error_message = error_message
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "status": self.status,
            "error_message": self.error_message,
        }


@dataclass
class TraceContext:
    """
    Context for a complete trace.
    
    A trace represents a complete operation that may span
    multiple components and services.
    """
    
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    
    # Trace metadata
    name: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    
    # Spans
    spans: list[SpanContext] = field(default_factory=list)
    _current_span: SpanContext | None = field(default=None, repr=False)
    
    # Context
    agent_id: str | None = None
    session_id: str | None = None
    
    # Attributes
    attributes: dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float | None:
        """Get trace duration in milliseconds."""
        if self.ended_at is None:
            return None
        delta = self.ended_at - self.started_at
        return delta.total_seconds() * 1000
    
    @property
    def current_span(self) -> SpanContext | None:
        """Get the current active span."""
        return self._current_span
    
    @property
    def root_span(self) -> SpanContext | None:
        """Get the root span."""
        for span in self.spans:
            if span.is_root:
                return span
        return None
    
    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> SpanContext:
        """
        Start a new span.
        
        Args:
            name: Span name
            attributes: Initial attributes
            
        Returns:
            The new span context
        """
        parent_span_id = self._current_span.span_id if self._current_span else None
        
        span = SpanContext(
            trace_id=self.trace_id,
            parent_span_id=parent_span_id,
            name=name,
            attributes=attributes or {},
        )
        
        self.spans.append(span)
        self._current_span = span
        
        return span
    
    def end_span(
        self,
        status: str = "ok",
        error_message: str | None = None,
    ) -> SpanContext | None:
        """
        End the current span.
        
        Args:
            status: Span status
            error_message: Error message if status is error
            
        Returns:
            The ended span, or None if no current span
        """
        if self._current_span is None:
            return None
        
        span = self._current_span
        span.end(status, error_message)
        
        # Find parent span
        if span.parent_span_id:
            for s in self.spans:
                if s.span_id == span.parent_span_id:
                    self._current_span = s
                    break
            else:
                self._current_span = None
        else:
            self._current_span = None
        
        return span
    
    def end(self) -> None:
        """End the trace."""
        self.ended_at = datetime.now(timezone.utc)
        
        # End any open spans
        while self._current_span:
            self.end_span()
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set a trace attribute."""
        self.attributes[key] = value
    
    def get_span(self, span_id: str) -> SpanContext | None:
        """Get a span by ID."""
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None
    
    def get_child_spans(self, span_id: str) -> list[SpanContext]:
        """Get child spans of a span."""
        return [s for s in self.spans if s.parent_span_id == span_id]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_ms": self.duration_ms,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "attributes": self.attributes,
            "spans": [s.to_dict() for s in self.spans],
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceContext:
        """Create from dictionary."""
        trace = cls(
            trace_id=data["trace_id"],
            name=data.get("name", ""),
            started_at=datetime.fromisoformat(data["started_at"]),
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            agent_id=data.get("agent_id"),
            session_id=data.get("session_id"),
            attributes=data.get("attributes", {}),
        )
        
        # Reconstruct spans
        for span_data in data.get("spans", []):
            span = SpanContext(
                span_id=span_data["span_id"],
                trace_id=span_data["trace_id"],
                parent_span_id=span_data.get("parent_span_id"),
                name=span_data.get("name", ""),
                started_at=datetime.fromisoformat(span_data["started_at"]),
                ended_at=datetime.fromisoformat(span_data["ended_at"]) if span_data.get("ended_at") else None,
                attributes=span_data.get("attributes", {}),
                status=span_data.get("status", "ok"),
                error_message=span_data.get("error_message"),
            )
            trace.spans.append(span)
        
        return trace


# Context variables for async context propagation
_trace_context_var: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "trace_context",
    default=None,
)


def current_trace() -> TraceContext | None:
    """Get the current trace context."""
    return _trace_context_var.get()


def current_span() -> SpanContext | None:
    """Get the current span context."""
    trace = current_trace()
    return trace.current_span if trace else None


def set_trace_context(trace: TraceContext | None) -> contextvars.Token[TraceContext | None]:
    """Set the current trace context."""
    return _trace_context_var.set(trace)


def reset_trace_context(token: contextvars.Token[TraceContext | None]) -> None:
    """Reset the trace context to a previous value."""
    _trace_context_var.reset(token)


@contextmanager
def trace_context(
    name: str = "",
    agent_id: str | None = None,
    session_id: str | None = None,
    trace_id: str | None = None,
) -> Iterator[TraceContext]:
    """
    Context manager for creating a trace context.
    
    Args:
        name: Trace name
        agent_id: Agent ID
        session_id: Session ID
        trace_id: Optional trace ID (generates new if not provided)
        
    Yields:
        The trace context
        
    Example:
        >>> with trace_context("my_operation") as trace:
        ...     with span_context("step_1"):
        ...         do_step_1()
        ...     with span_context("step_2"):
        ...         do_step_2()
    """
    trace = TraceContext(
        trace_id=trace_id or str(uuid4()),
        name=name,
        agent_id=agent_id,
        session_id=session_id,
    )
    
    token = set_trace_context(trace)
    try:
        yield trace
    finally:
        trace.end()
        reset_trace_context(token)


@contextmanager
def span_context(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Iterator[SpanContext]:
    """
    Context manager for creating a span within the current trace.
    
    Args:
        name: Span name
        attributes: Initial attributes
        
    Yields:
        The span context
        
    Raises:
        RuntimeError: If no trace context is active
        
    Example:
        >>> with trace_context("my_operation") as trace:
        ...     with span_context("step_1") as span:
        ...         span.set_attribute("key", "value")
        ...         do_step_1()
    """
    trace = current_trace()
    if trace is None:
        raise RuntimeError("No trace context active")
    
    span = trace.start_span(name, attributes)
    try:
        yield span
    except Exception as e:
        trace.end_span(status="error", error_message=str(e))
        raise
    else:
        trace.end_span()


class TraceContextCarrier:
    """
    Carrier for propagating trace context across boundaries.
    
    Used for passing trace context in HTTP headers, message queues, etc.
    """
    
    TRACE_ID_HEADER = "X-Trace-ID"
    SPAN_ID_HEADER = "X-Span-ID"
    PARENT_SPAN_ID_HEADER = "X-Parent-Span-ID"
    
    @classmethod
    def inject(cls, headers: dict[str, str]) -> dict[str, str]:
        """
        Inject trace context into headers.
        
        Args:
            headers: Headers dict to inject into
            
        Returns:
            Updated headers
        """
        trace = current_trace()
        if trace is None:
            return headers
        
        headers[cls.TRACE_ID_HEADER] = trace.trace_id
        
        span = trace.current_span
        if span:
            headers[cls.SPAN_ID_HEADER] = span.span_id
            if span.parent_span_id:
                headers[cls.PARENT_SPAN_ID_HEADER] = span.parent_span_id
        
        return headers
    
    @classmethod
    def extract(cls, headers: dict[str, str]) -> tuple[str | None, str | None, str | None]:
        """
        Extract trace context from headers.
        
        Args:
            headers: Headers dict to extract from
            
        Returns:
            Tuple of (trace_id, span_id, parent_span_id)
        """
        trace_id = headers.get(cls.TRACE_ID_HEADER)
        span_id = headers.get(cls.SPAN_ID_HEADER)
        parent_span_id = headers.get(cls.PARENT_SPAN_ID_HEADER)
        
        return trace_id, span_id, parent_span_id
    
    @classmethod
    def continue_trace(
        cls,
        headers: dict[str, str],
        name: str = "",
    ) -> TraceContext:
        """
        Continue a trace from headers.
        
        Args:
            headers: Headers containing trace context
            name: Name for the continued trace
            
        Returns:
            TraceContext (new or continued)
        """
        trace_id, span_id, parent_span_id = cls.extract(headers)
        
        trace = TraceContext(
            trace_id=trace_id or str(uuid4()),
            name=name,
        )
        
        # If we have a span ID from the parent, create a span with it as parent
        if span_id:
            trace.start_span(name, {"parent_trace_span_id": span_id})
        
        return trace
