"""
Audit logging and tracing for the agent runtime.

This package provides:
- Event logging with structured data
- Trace context propagation
- Audit log storage and retrieval
- Trace visualization utilities
"""

from __future__ import annotations

from .events import (
    AuditEvent,
    AuditEventType,
    AuditEventSeverity,
    EventFactory,
)
from .context import (
    TraceContext,
    SpanContext,
    trace_context,
    current_trace,
    current_span,
)
from .logger import (
    AuditLogger,
    AuditLogConfig,
    AuditHandler,
    ConsoleHandler,
    CallbackHandler,
    BufferedHandler,
)
from .storage import (
    AuditStorage,
    InMemoryAuditStorage,
    FileAuditStorage,
)
from .viewer import (
    TraceViewer,
    TraceTree,
    SpanNode,
    TimelineEntry,
    CausalGraph,
    CausalLink,
    format_trace_tree,
    format_timeline,
)

__all__ = [
    # Events
    "AuditEvent",
    "AuditEventType",
    "AuditEventSeverity",
    "EventFactory",
    # Context
    "TraceContext",
    "SpanContext",
    "trace_context",
    "current_trace",
    "current_span",
    # Logger
    "AuditLogger",
    "AuditLogConfig",
    "AuditHandler",
    "ConsoleHandler",
    "CallbackHandler",
    "BufferedHandler",
    # Storage
    "AuditStorage",
    "InMemoryAuditStorage",
    "FileAuditStorage",
    # Viewer
    "TraceViewer",
    "TraceTree",
    "SpanNode",
    "TimelineEntry",
    "CausalGraph",
    "CausalLink",
    "format_trace_tree",
    "format_timeline",
]
