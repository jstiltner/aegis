"""
Trace viewer for visualizing execution history.

This module provides:
- Trace visualization utilities
- Causal tracing analysis
- Timeline generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
from collections import defaultdict

from .events import AuditEvent, AuditEventType, AuditEventSeverity
from .context import TraceContext, SpanContext
from .storage import AuditStorage


@dataclass
class SpanNode:
    """A node in the span tree."""
    
    span: SpanContext
    events: list[AuditEvent] = field(default_factory=list)
    children: list[SpanNode] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> float | None:
        """Get span duration in milliseconds."""
        if self.span.ended_at is None:
            return None
        delta = self.span.ended_at - self.span.started_at
        return delta.total_seconds() * 1000
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "span_id": self.span.span_id,
            "name": self.span.name,
            "started_at": self.span.started_at.isoformat(),
            "ended_at": self.span.ended_at.isoformat() if self.span.ended_at else None,
            "duration_ms": self.duration_ms,
            "status": self.span.status,
            "attributes": dict(self.span.attributes),
            "events": [e.to_dict() for e in self.events],
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class TraceTree:
    """A tree representation of a trace."""
    
    trace: TraceContext
    root_spans: list[SpanNode] = field(default_factory=list)
    orphan_events: list[AuditEvent] = field(default_factory=list)
    
    @property
    def total_events(self) -> int:
        """Get total number of events."""
        count = len(self.orphan_events)
        
        def count_events(node: SpanNode) -> int:
            return len(node.events) + sum(count_events(c) for c in node.children)
        
        for root in self.root_spans:
            count += count_events(root)
        
        return count
    
    @property
    def total_spans(self) -> int:
        """Get total number of spans."""
        def count_spans(node: SpanNode) -> int:
            return 1 + sum(count_spans(c) for c in node.children)
        
        return sum(count_spans(root) for root in self.root_spans)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace.trace_id,
            "agent_id": self.trace.agent_id,
            "session_id": self.trace.session_id,
            "started_at": self.trace.started_at.isoformat(),
            "ended_at": self.trace.ended_at.isoformat() if self.trace.ended_at else None,
            "total_events": self.total_events,
            "total_spans": self.total_spans,
            "root_spans": [s.to_dict() for s in self.root_spans],
            "orphan_events": [e.to_dict() for e in self.orphan_events],
        }


@dataclass
class TimelineEntry:
    """An entry in the execution timeline."""
    
    timestamp: datetime
    event_type: str
    message: str
    span_name: str | None = None
    severity: AuditEventSeverity = AuditEventSeverity.INFO
    data: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "message": self.message,
            "span_name": self.span_name,
            "severity": self.severity.value,
            "data": self.data,
        }


@dataclass
class CausalLink:
    """A causal link between events."""
    
    cause_event_id: str
    effect_event_id: str
    link_type: str  # "triggers", "enables", "requires", etc.
    confidence: float = 1.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cause_event_id": self.cause_event_id,
            "effect_event_id": self.effect_event_id,
            "link_type": self.link_type,
            "confidence": self.confidence,
        }


@dataclass
class CausalGraph:
    """A graph of causal relationships between events."""
    
    events: list[AuditEvent]
    links: list[CausalLink] = field(default_factory=list)
    
    def get_causes(self, event_id: str) -> list[AuditEvent]:
        """Get events that caused this event."""
        cause_ids = {
            link.cause_event_id
            for link in self.links
            if link.effect_event_id == event_id
        }
        return [e for e in self.events if e.event_id in cause_ids]
    
    def get_effects(self, event_id: str) -> list[AuditEvent]:
        """Get events caused by this event."""
        effect_ids = {
            link.effect_event_id
            for link in self.links
            if link.cause_event_id == event_id
        }
        return [e for e in self.events if e.event_id in effect_ids]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "events": [e.to_dict() for e in self.events],
            "links": [l.to_dict() for l in self.links],
        }


class TraceViewer:
    """
    Viewer for trace visualization and analysis.
    
    The TraceViewer provides:
    - Tree visualization of traces
    - Timeline generation
    - Causal tracing analysis
    - Event filtering and search
    
    Example:
        >>> viewer = TraceViewer(storage)
        >>> tree = await viewer.build_trace_tree(trace_id)
        >>> timeline = await viewer.build_timeline(trace_id)
    """
    
    def __init__(self, storage: AuditStorage) -> None:
        """
        Initialize the trace viewer.
        
        Args:
            storage: Audit storage backend
        """
        self.storage = storage
    
    async def build_trace_tree(self, trace_id: str) -> TraceTree | None:
        """
        Build a tree representation of a trace.
        
        Args:
            trace_id: The trace ID
            
        Returns:
            TraceTree or None if trace not found
        """
        trace = await self.storage.get_trace(trace_id)
        if trace is None:
            return None
        
        # Get all events for this trace
        events = await self.storage.query_events(
            trace_id=trace_id,
            limit=10000,
        )
        
        # Build span nodes
        span_nodes: dict[str, SpanNode] = {}
        
        for span in trace.spans:
            span_nodes[span.span_id] = SpanNode(span=span)
        
        # Assign events to spans
        orphan_events: list[AuditEvent] = []
        
        for event in events:
            if event.span_id and event.span_id in span_nodes:
                span_nodes[event.span_id].events.append(event)
            else:
                orphan_events.append(event)
        
        # Sort events within spans
        for node in span_nodes.values():
            node.events.sort(key=lambda e: e.timestamp)
        
        # Build tree structure
        root_spans: list[SpanNode] = []
        
        for span_id, node in span_nodes.items():
            if node.span.parent_span_id and node.span.parent_span_id in span_nodes:
                parent = span_nodes[node.span.parent_span_id]
                parent.children.append(node)
            else:
                root_spans.append(node)
        
        # Sort children by start time
        def sort_children(node: SpanNode) -> None:
            node.children.sort(key=lambda n: n.span.started_at)
            for child in node.children:
                sort_children(child)
        
        for root in root_spans:
            sort_children(root)
        
        root_spans.sort(key=lambda n: n.span.started_at)
        
        return TraceTree(
            trace=trace,
            root_spans=root_spans,
            orphan_events=sorted(orphan_events, key=lambda e: e.timestamp),
        )
    
    async def build_timeline(
        self,
        trace_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        include_spans: bool = True,
    ) -> list[TimelineEntry]:
        """
        Build a timeline of events.
        
        Args:
            trace_id: Filter by trace ID
            agent_id: Filter by agent ID
            session_id: Filter by session ID
            start_time: Start of time range
            end_time: End of time range
            include_spans: Whether to include span start/end events
            
        Returns:
            List of timeline entries
        """
        entries: list[TimelineEntry] = []
        
        # Get events
        events = await self.storage.query_events(
            trace_id=trace_id,
            agent_id=agent_id,
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            limit=10000,
        )
        
        # Get span info if we have a trace
        span_names: dict[str, str] = {}
        
        if trace_id:
            trace = await self.storage.get_trace(trace_id)
            if trace:
                for span in trace.spans:
                    span_names[span.span_id] = span.name
                
                # Add span events
                if include_spans:
                    for span in trace.spans:
                        entries.append(TimelineEntry(
                            timestamp=span.started_at,
                            event_type="span_start",
                            message=f"Started span: {span.name}",
                            span_name=span.name,
                            severity=AuditEventSeverity.DEBUG,
                            data={"span_id": span.span_id},
                        ))
                        
                        if span.ended_at:
                            entries.append(TimelineEntry(
                                timestamp=span.ended_at,
                                event_type="span_end",
                                message=f"Ended span: {span.name} ({span.status})",
                                span_name=span.name,
                                severity=AuditEventSeverity.DEBUG,
                                data={
                                    "span_id": span.span_id,
                                    "status": span.status,
                                },
                            ))
        
        # Add event entries
        for event in events:
            span_name = span_names.get(event.span_id) if event.span_id else None
            
            entries.append(TimelineEntry(
                timestamp=event.timestamp,
                event_type=event.event_type.value,
                message=event.message,
                span_name=span_name,
                severity=event.severity,
                data=event.data,
            ))
        
        # Sort by timestamp
        entries.sort(key=lambda e: e.timestamp)
        
        return entries
    
    async def build_causal_graph(
        self,
        trace_id: str,
    ) -> CausalGraph | None:
        """
        Build a causal graph for a trace.
        
        This analyzes events to infer causal relationships based on:
        - Temporal ordering
        - Event types (e.g., tool invocation -> tool completion)
        - Span hierarchy
        
        Args:
            trace_id: The trace ID
            
        Returns:
            CausalGraph or None if trace not found
        """
        trace = await self.storage.get_trace(trace_id)
        if trace is None:
            return None
        
        events = await self.storage.query_events(
            trace_id=trace_id,
            limit=10000,
        )
        
        events.sort(key=lambda e: e.timestamp)
        
        links: list[CausalLink] = []
        
        # Track tool invocations
        tool_invocations: dict[str, AuditEvent] = {}
        
        # Track state changes
        last_state_event: AuditEvent | None = None
        
        for event in events:
            # Tool invocation -> completion/failure
            if event.event_type == AuditEventType.TOOL_INVOKED:
                invocation_id = event.data.get("invocation_id")
                if invocation_id:
                    tool_invocations[invocation_id] = event
            
            elif event.event_type in (AuditEventType.TOOL_COMPLETED, AuditEventType.TOOL_FAILED):
                invocation_id = event.data.get("invocation_id")
                if invocation_id and invocation_id in tool_invocations:
                    cause = tool_invocations[invocation_id]
                    links.append(CausalLink(
                        cause_event_id=cause.event_id,
                        effect_event_id=event.event_id,
                        link_type="triggers",
                        confidence=1.0,
                    ))
            
            # State changes
            elif event.event_type == AuditEventType.STATE_CHANGED:
                if last_state_event:
                    links.append(CausalLink(
                        cause_event_id=last_state_event.event_id,
                        effect_event_id=event.event_id,
                        link_type="enables",
                        confidence=0.9,
                    ))
                last_state_event = event
            
            # Policy decisions
            elif event.event_type == AuditEventType.POLICY_DENIED:
                # Find the tool invocation that was denied
                tool_name = event.data.get("tool_name")
                for inv_id, inv_event in tool_invocations.items():
                    if inv_event.data.get("tool_name") == tool_name:
                        links.append(CausalLink(
                            cause_event_id=inv_event.event_id,
                            effect_event_id=event.event_id,
                            link_type="triggers",
                            confidence=0.8,
                        ))
                        break
            
            # Checkpoint -> restore
            elif event.event_type == AuditEventType.STATE_RESTORED:
                checkpoint_id = event.data.get("checkpoint_id")
                for prev_event in events:
                    if (prev_event.event_type == AuditEventType.STATE_CHECKPOINT and
                        prev_event.data.get("checkpoint_id") == checkpoint_id):
                        links.append(CausalLink(
                            cause_event_id=prev_event.event_id,
                            effect_event_id=event.event_id,
                            link_type="enables",
                            confidence=1.0,
                        ))
                        break
        
        return CausalGraph(events=events, links=links)
    
    async def get_event_context(
        self,
        event_id: str,
        context_window: int = 5,
    ) -> dict[str, Any]:
        """
        Get context around an event.
        
        Args:
            event_id: The event ID
            context_window: Number of events before/after
            
        Returns:
            Context information
        """
        event = await self.storage.get_event(event_id)
        if event is None:
            return {"error": "Event not found"}
        
        # Get surrounding events
        events = await self.storage.query_events(
            trace_id=event.trace_id,
            limit=10000,
        )
        
        events.sort(key=lambda e: e.timestamp)
        
        # Find event index
        event_index = -1
        for i, e in enumerate(events):
            if e.event_id == event_id:
                event_index = i
                break
        
        if event_index == -1:
            return {"error": "Event not found in trace"}
        
        # Get context window
        start = max(0, event_index - context_window)
        end = min(len(events), event_index + context_window + 1)
        
        before = events[start:event_index]
        after = events[event_index + 1:end]
        
        # Get span info
        span_info = None
        if event.trace_id and event.span_id:
            trace = await self.storage.get_trace(event.trace_id)
            if trace:
                for span in trace.spans:
                    if span.span_id == event.span_id:
                        span_info = {
                            "name": span.name,
                            "started_at": span.started_at.isoformat(),
                            "ended_at": span.ended_at.isoformat() if span.ended_at else None,
                            "status": span.status,
                        }
                        break
        
        return {
            "event": event.to_dict(),
            "span": span_info,
            "before": [e.to_dict() for e in before],
            "after": [e.to_dict() for e in after],
        }
    
    async def search_events(
        self,
        query: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """
        Search events by message content.
        
        Args:
            query: Search query (case-insensitive substring match)
            agent_id: Filter by agent ID
            session_id: Filter by session ID
            limit: Maximum results
            
        Returns:
            List of matching events
        """
        # Get all events (with filters)
        events = await self.storage.query_events(
            agent_id=agent_id,
            session_id=session_id,
            limit=10000,
        )
        
        # Filter by query
        query_lower = query.lower()
        results: list[AuditEvent] = []
        
        for event in events:
            if query_lower in event.message.lower():
                results.append(event)
                if len(results) >= limit:
                    break
        
        return results
    
    async def get_statistics(
        self,
        agent_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Get statistics for events.
        
        Args:
            agent_id: Filter by agent ID
            session_id: Filter by session ID
            trace_id: Filter by trace ID
            
        Returns:
            Statistics dictionary
        """
        events = await self.storage.query_events(
            agent_id=agent_id,
            session_id=session_id,
            trace_id=trace_id,
            limit=10000,
        )
        
        # Count by type
        by_type: dict[str, int] = defaultdict(int)
        for event in events:
            by_type[event.event_type.value] += 1
        
        # Count by severity
        by_severity: dict[str, int] = defaultdict(int)
        for event in events:
            by_severity[event.severity.value] += 1
        
        # Time range
        if events:
            events_sorted = sorted(events, key=lambda e: e.timestamp)
            first_event = events_sorted[0]
            last_event = events_sorted[-1]
            duration = (last_event.timestamp - first_event.timestamp).total_seconds()
        else:
            first_event = None
            last_event = None
            duration = 0
        
        # Tool statistics
        tool_invocations = [e for e in events if e.event_type == AuditEventType.TOOL_INVOKED]
        tool_completions = [e for e in events if e.event_type == AuditEventType.TOOL_COMPLETED]
        tool_failures = [e for e in events if e.event_type == AuditEventType.TOOL_FAILED]
        
        tools_by_name: dict[str, int] = defaultdict(int)
        for event in tool_invocations:
            tool_name = event.data.get("tool_name", "unknown")
            tools_by_name[tool_name] += 1
        
        return {
            "total_events": len(events),
            "by_type": dict(by_type),
            "by_severity": dict(by_severity),
            "time_range": {
                "start": first_event.timestamp.isoformat() if first_event else None,
                "end": last_event.timestamp.isoformat() if last_event else None,
                "duration_seconds": duration,
            },
            "tools": {
                "total_invocations": len(tool_invocations),
                "total_completions": len(tool_completions),
                "total_failures": len(tool_failures),
                "by_name": dict(tools_by_name),
            },
        }


def format_trace_tree(tree: TraceTree, indent: int = 2) -> str:
    """
    Format a trace tree as a string.
    
    Args:
        tree: The trace tree
        indent: Indentation per level
        
    Returns:
        Formatted string
    """
    lines: list[str] = []
    
    lines.append(f"Trace: {tree.trace.trace_id}")
    lines.append(f"  Agent: {tree.trace.agent_id or '-'}")
    lines.append(f"  Session: {tree.trace.session_id or '-'}")
    lines.append(f"  Started: {tree.trace.started_at.isoformat()}")
    if tree.trace.ended_at:
        lines.append(f"  Ended: {tree.trace.ended_at.isoformat()}")
    lines.append(f"  Spans: {tree.total_spans}")
    lines.append(f"  Events: {tree.total_events}")
    lines.append("")
    
    def format_span(node: SpanNode, level: int) -> None:
        prefix = " " * (level * indent)
        duration = f" ({node.duration_ms:.1f}ms)" if node.duration_ms else ""
        status = f" [{node.span.status}]" if node.span.status != "ok" else ""
        
        lines.append(f"{prefix}├─ {node.span.name}{duration}{status}")
        
        for event in node.events:
            event_prefix = " " * ((level + 1) * indent)
            severity_marker = {
                AuditEventSeverity.DEBUG: "·",
                AuditEventSeverity.INFO: "○",
                AuditEventSeverity.WARNING: "△",
                AuditEventSeverity.ERROR: "✗",
                AuditEventSeverity.CRITICAL: "✗✗",
            }.get(event.severity, "○")
            
            lines.append(f"{event_prefix}{severity_marker} {event.message}")
        
        for child in node.children:
            format_span(child, level + 1)
    
    for root in tree.root_spans:
        format_span(root, 0)
    
    if tree.orphan_events:
        lines.append("")
        lines.append("Orphan Events:")
        for event in tree.orphan_events:
            lines.append(f"  - {event.message}")
    
    return "\n".join(lines)


def format_timeline(entries: list[TimelineEntry]) -> str:
    """
    Format a timeline as a string.
    
    Args:
        entries: Timeline entries
        
    Returns:
        Formatted string
    """
    lines: list[str] = []
    
    for entry in entries:
        time_str = entry.timestamp.strftime("%H:%M:%S.%f")[:-3]
        severity_marker = {
            AuditEventSeverity.DEBUG: " ",
            AuditEventSeverity.INFO: "·",
            AuditEventSeverity.WARNING: "!",
            AuditEventSeverity.ERROR: "✗",
            AuditEventSeverity.CRITICAL: "✗",
        }.get(entry.severity, " ")
        
        span_info = f" [{entry.span_name}]" if entry.span_name else ""
        
        lines.append(f"{time_str} {severity_marker} {entry.event_type}{span_info}: {entry.message}")
    
    return "\n".join(lines)
