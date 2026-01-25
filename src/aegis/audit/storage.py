"""
Audit log storage backends.

This module provides:
- Abstract storage interface
- In-memory storage for testing
- File-based storage for persistence
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterator
from collections import defaultdict

from .events import AuditEvent, AuditEventType, AuditEventSeverity
from .context import TraceContext


class AuditStorage(ABC):
    """
    Abstract base class for audit log storage.
    
    Storage backends persist audit events and traces for
    later retrieval and analysis.
    """
    
    @abstractmethod
    async def store_event(self, event: AuditEvent) -> None:
        """
        Store an audit event.
        
        Args:
            event: The event to store
        """
        ...
    
    @abstractmethod
    async def store_trace(self, trace: TraceContext) -> None:
        """
        Store a trace.
        
        Args:
            trace: The trace to store
        """
        ...
    
    @abstractmethod
    async def get_event(self, event_id: str) -> AuditEvent | None:
        """
        Get an event by ID.
        
        Args:
            event_id: Event ID
            
        Returns:
            The event, or None if not found
        """
        ...
    
    @abstractmethod
    async def get_trace(self, trace_id: str) -> TraceContext | None:
        """
        Get a trace by ID.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            The trace, or None if not found
        """
        ...
    
    @abstractmethod
    async def query_events(
        self,
        agent_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        event_types: list[AuditEventType] | None = None,
        min_severity: AuditEventSeverity | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEvent]:
        """
        Query events with filters.
        
        Args:
            agent_id: Filter by agent ID
            session_id: Filter by session ID
            trace_id: Filter by trace ID
            event_types: Filter by event types
            min_severity: Minimum severity level
            start_time: Start of time range
            end_time: End of time range
            tags: Filter by tags (any match)
            limit: Maximum results
            offset: Result offset
            
        Returns:
            List of matching events
        """
        ...
    
    @abstractmethod
    async def query_traces(
        self,
        agent_id: str | None = None,
        session_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TraceContext]:
        """
        Query traces with filters.
        
        Args:
            agent_id: Filter by agent ID
            session_id: Filter by session ID
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum results
            offset: Result offset
            
        Returns:
            List of matching traces
        """
        ...
    
    @abstractmethod
    async def delete_events(
        self,
        before: datetime | None = None,
        agent_id: str | None = None,
    ) -> int:
        """
        Delete events matching criteria.
        
        Args:
            before: Delete events before this time
            agent_id: Delete events for this agent
            
        Returns:
            Number of events deleted
        """
        ...
    
    @abstractmethod
    async def get_event_count(
        self,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """
        Get count of events.
        
        Args:
            agent_id: Filter by agent ID
            session_id: Filter by session ID
            
        Returns:
            Event count
        """
        ...


class InMemoryAuditStorage(AuditStorage):
    """
    In-memory audit storage for testing and development.
    
    Events are stored in memory and lost on restart.
    """
    
    def __init__(self, max_events: int = 10000, max_traces: int = 1000) -> None:
        """
        Initialize in-memory storage.
        
        Args:
            max_events: Maximum events to store
            max_traces: Maximum traces to store
        """
        self.max_events = max_events
        self.max_traces = max_traces
        
        self._events: dict[str, AuditEvent] = {}
        self._traces: dict[str, TraceContext] = {}
        
        # Indexes for efficient querying
        self._events_by_agent: dict[str, list[str]] = defaultdict(list)
        self._events_by_session: dict[str, list[str]] = defaultdict(list)
        self._events_by_trace: dict[str, list[str]] = defaultdict(list)
        self._events_by_type: dict[AuditEventType, list[str]] = defaultdict(list)
    
    async def store_event(self, event: AuditEvent) -> None:
        """Store an event in memory."""
        # Evict oldest if at capacity
        if len(self._events) >= self.max_events:
            oldest_id = min(
                self._events.keys(),
                key=lambda k: self._events[k].timestamp,
            )
            await self._remove_event(oldest_id)
        
        self._events[event.event_id] = event
        
        # Update indexes
        if event.agent_id:
            self._events_by_agent[event.agent_id].append(event.event_id)
        if event.session_id:
            self._events_by_session[event.session_id].append(event.event_id)
        if event.trace_id:
            self._events_by_trace[event.trace_id].append(event.event_id)
        self._events_by_type[event.event_type].append(event.event_id)
    
    async def _remove_event(self, event_id: str) -> None:
        """Remove an event from storage."""
        event = self._events.pop(event_id, None)
        if event is None:
            return
        
        # Update indexes
        if event.agent_id and event_id in self._events_by_agent[event.agent_id]:
            self._events_by_agent[event.agent_id].remove(event_id)
        if event.session_id and event_id in self._events_by_session[event.session_id]:
            self._events_by_session[event.session_id].remove(event_id)
        if event.trace_id and event_id in self._events_by_trace[event.trace_id]:
            self._events_by_trace[event.trace_id].remove(event_id)
        if event_id in self._events_by_type[event.event_type]:
            self._events_by_type[event.event_type].remove(event_id)
    
    async def store_trace(self, trace: TraceContext) -> None:
        """Store a trace in memory."""
        # Evict oldest if at capacity
        if len(self._traces) >= self.max_traces:
            oldest_id = min(
                self._traces.keys(),
                key=lambda k: self._traces[k].started_at,
            )
            del self._traces[oldest_id]
        
        self._traces[trace.trace_id] = trace
    
    async def get_event(self, event_id: str) -> AuditEvent | None:
        """Get an event by ID."""
        return self._events.get(event_id)
    
    async def get_trace(self, trace_id: str) -> TraceContext | None:
        """Get a trace by ID."""
        return self._traces.get(trace_id)
    
    async def query_events(
        self,
        agent_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        event_types: list[AuditEventType] | None = None,
        min_severity: AuditEventSeverity | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEvent]:
        """Query events with filters."""
        # Start with candidate event IDs
        if agent_id:
            candidate_ids = set(self._events_by_agent.get(agent_id, []))
        elif session_id:
            candidate_ids = set(self._events_by_session.get(session_id, []))
        elif trace_id:
            candidate_ids = set(self._events_by_trace.get(trace_id, []))
        elif event_types and len(event_types) == 1:
            candidate_ids = set(self._events_by_type.get(event_types[0], []))
        else:
            candidate_ids = set(self._events.keys())
        
        # Filter candidates
        severity_order = [
            AuditEventSeverity.DEBUG,
            AuditEventSeverity.INFO,
            AuditEventSeverity.WARNING,
            AuditEventSeverity.ERROR,
            AuditEventSeverity.CRITICAL,
        ]
        
        results: list[AuditEvent] = []
        
        for event_id in candidate_ids:
            event = self._events.get(event_id)
            if event is None:
                continue
            
            # Apply filters
            if agent_id and event.agent_id != agent_id:
                continue
            if session_id and event.session_id != session_id:
                continue
            if trace_id and event.trace_id != trace_id:
                continue
            if event_types and event.event_type not in event_types:
                continue
            if min_severity:
                if severity_order.index(event.severity) < severity_order.index(min_severity):
                    continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            if tags and not any(t in event.tags for t in tags):
                continue
            
            results.append(event)
        
        # Sort by timestamp descending
        results.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Apply pagination
        return results[offset:offset + limit]
    
    async def query_traces(
        self,
        agent_id: str | None = None,
        session_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TraceContext]:
        """Query traces with filters."""
        results: list[TraceContext] = []
        
        for trace in self._traces.values():
            # Apply filters
            if agent_id and trace.agent_id != agent_id:
                continue
            if session_id and trace.session_id != session_id:
                continue
            if start_time and trace.started_at < start_time:
                continue
            if end_time and trace.started_at > end_time:
                continue
            
            results.append(trace)
        
        # Sort by start time descending
        results.sort(key=lambda t: t.started_at, reverse=True)
        
        # Apply pagination
        return results[offset:offset + limit]
    
    async def delete_events(
        self,
        before: datetime | None = None,
        agent_id: str | None = None,
    ) -> int:
        """Delete events matching criteria."""
        to_delete: list[str] = []
        
        for event_id, event in self._events.items():
            if before and event.timestamp >= before:
                continue
            if agent_id and event.agent_id != agent_id:
                continue
            to_delete.append(event_id)
        
        for event_id in to_delete:
            await self._remove_event(event_id)
        
        return len(to_delete)
    
    async def get_event_count(
        self,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Get count of events."""
        if agent_id:
            return len(self._events_by_agent.get(agent_id, []))
        if session_id:
            return len(self._events_by_session.get(session_id, []))
        return len(self._events)
    
    def clear(self) -> None:
        """Clear all stored data."""
        self._events.clear()
        self._traces.clear()
        self._events_by_agent.clear()
        self._events_by_session.clear()
        self._events_by_trace.clear()
        self._events_by_type.clear()


class FileAuditStorage(AuditStorage):
    """
    File-based audit storage for persistence.
    
    Events are stored as JSON files organized by date.
    """
    
    def __init__(
        self,
        base_path: str | Path,
        events_subdir: str = "events",
        traces_subdir: str = "traces",
    ) -> None:
        """
        Initialize file storage.
        
        Args:
            base_path: Base directory for storage
            events_subdir: Subdirectory for events
            traces_subdir: Subdirectory for traces
        """
        self.base_path = Path(base_path)
        self.events_path = self.base_path / events_subdir
        self.traces_path = self.base_path / traces_subdir
        
        # Create directories
        self.events_path.mkdir(parents=True, exist_ok=True)
        self.traces_path.mkdir(parents=True, exist_ok=True)
    
    def _get_event_path(self, event: AuditEvent) -> Path:
        """Get the file path for an event."""
        date_str = event.timestamp.strftime("%Y-%m-%d")
        return self.events_path / date_str / f"{event.event_id}.json"
    
    def _get_trace_path(self, trace: TraceContext) -> Path:
        """Get the file path for a trace."""
        date_str = trace.started_at.strftime("%Y-%m-%d")
        return self.traces_path / date_str / f"{trace.trace_id}.json"
    
    async def store_event(self, event: AuditEvent) -> None:
        """Store an event to file."""
        path = self._get_event_path(event)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(event.to_dict(), f, indent=2)
    
    async def store_trace(self, trace: TraceContext) -> None:
        """Store a trace to file."""
        path = self._get_trace_path(trace)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(trace.to_dict(), f, indent=2)
    
    async def get_event(self, event_id: str) -> AuditEvent | None:
        """Get an event by ID."""
        # Search through date directories
        for date_dir in self.events_path.iterdir():
            if not date_dir.is_dir():
                continue
            
            event_file = date_dir / f"{event_id}.json"
            if event_file.exists():
                with open(event_file) as f:
                    data = json.load(f)
                return AuditEvent.from_dict(data)
        
        return None
    
    async def get_trace(self, trace_id: str) -> TraceContext | None:
        """Get a trace by ID."""
        # Search through date directories
        for date_dir in self.traces_path.iterdir():
            if not date_dir.is_dir():
                continue
            
            trace_file = date_dir / f"{trace_id}.json"
            if trace_file.exists():
                with open(trace_file) as f:
                    data = json.load(f)
                return TraceContext.from_dict(data)
        
        return None
    
    def _iter_event_files(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> Iterator[Path]:
        """Iterate over event files in time range."""
        for date_dir in sorted(self.events_path.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue
            
            try:
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue
            
            # Check date range
            if start_time and dir_date < start_time.replace(
                hour=0, minute=0, second=0, microsecond=0
            ):
                continue
            if end_time and dir_date > end_time:
                continue
            
            for event_file in date_dir.iterdir():
                if event_file.suffix == ".json":
                    yield event_file
    
    async def query_events(
        self,
        agent_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        event_types: list[AuditEventType] | None = None,
        min_severity: AuditEventSeverity | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEvent]:
        """Query events with filters."""
        severity_order = [
            AuditEventSeverity.DEBUG,
            AuditEventSeverity.INFO,
            AuditEventSeverity.WARNING,
            AuditEventSeverity.ERROR,
            AuditEventSeverity.CRITICAL,
        ]
        
        results: list[AuditEvent] = []
        skipped = 0
        
        for event_file in self._iter_event_files(start_time, end_time):
            try:
                with open(event_file) as f:
                    data = json.load(f)
                event = AuditEvent.from_dict(data)
            except Exception:
                continue
            
            # Apply filters
            if agent_id and event.agent_id != agent_id:
                continue
            if session_id and event.session_id != session_id:
                continue
            if trace_id and event.trace_id != trace_id:
                continue
            if event_types and event.event_type not in event_types:
                continue
            if min_severity:
                if severity_order.index(event.severity) < severity_order.index(min_severity):
                    continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            if tags and not any(t in event.tags for t in tags):
                continue
            
            # Handle pagination
            if skipped < offset:
                skipped += 1
                continue
            
            results.append(event)
            
            if len(results) >= limit:
                break
        
        # Sort by timestamp descending
        results.sort(key=lambda e: e.timestamp, reverse=True)
        
        return results
    
    async def query_traces(
        self,
        agent_id: str | None = None,
        session_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TraceContext]:
        """Query traces with filters."""
        results: list[TraceContext] = []
        skipped = 0
        
        for date_dir in sorted(self.traces_path.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue
            
            for trace_file in date_dir.iterdir():
                if trace_file.suffix != ".json":
                    continue
                
                try:
                    with open(trace_file) as f:
                        data = json.load(f)
                    trace = TraceContext.from_dict(data)
                except Exception:
                    continue
                
                # Apply filters
                if agent_id and trace.agent_id != agent_id:
                    continue
                if session_id and trace.session_id != session_id:
                    continue
                if start_time and trace.started_at < start_time:
                    continue
                if end_time and trace.started_at > end_time:
                    continue
                
                # Handle pagination
                if skipped < offset:
                    skipped += 1
                    continue
                
                results.append(trace)
                
                if len(results) >= limit:
                    break
            
            if len(results) >= limit:
                break
        
        return results
    
    async def delete_events(
        self,
        before: datetime | None = None,
        agent_id: str | None = None,
    ) -> int:
        """Delete events matching criteria."""
        deleted = 0
        
        for event_file in self._iter_event_files(end_time=before):
            try:
                with open(event_file) as f:
                    data = json.load(f)
                event = AuditEvent.from_dict(data)
            except Exception:
                continue
            
            # Check filters
            if before and event.timestamp >= before:
                continue
            if agent_id and event.agent_id != agent_id:
                continue
            
            # Delete file
            event_file.unlink()
            deleted += 1
        
        # Clean up empty directories
        for date_dir in self.events_path.iterdir():
            if date_dir.is_dir() and not any(date_dir.iterdir()):
                date_dir.rmdir()
        
        return deleted
    
    async def get_event_count(
        self,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Get count of events."""
        count = 0
        
        for event_file in self._iter_event_files():
            if agent_id or session_id:
                try:
                    with open(event_file) as f:
                        data = json.load(f)
                    
                    if agent_id and data.get("agent_id") != agent_id:
                        continue
                    if session_id and data.get("session_id") != session_id:
                        continue
                except Exception:
                    continue
            
            count += 1
        
        return count
