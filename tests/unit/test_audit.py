"""
Unit tests for the audit logging and tracing module.
"""

import pytest
from datetime import datetime, timezone, timedelta

from aegis.audit import (
    AuditEvent,
    AuditEventType,
    AuditEventSeverity,
    EventFactory,
    TraceContext,
    SpanContext,
    trace_context,
    current_trace,
    current_span,
    AuditLogger,
    AuditLogConfig,
    CallbackHandler,
    BufferedHandler,
    InMemoryAuditStorage,
    TraceViewer,
)
from aegis.audit.context import span_context


# =============================================================================
# AuditEvent Tests
# =============================================================================

class TestAuditEvent:
    """Tests for AuditEvent model."""
    
    def test_create_event(self):
        """Test creating an audit event."""
        event = AuditEvent(
            event_type=AuditEventType.TOOL_INVOKED,
            message="Tool search invoked",
            data={"tool_name": "search"},
        )
        
        assert event.event_type == AuditEventType.TOOL_INVOKED
        assert event.message == "Tool search invoked"
        assert event.data["tool_name"] == "search"
        assert event.severity == AuditEventSeverity.INFO
        assert event.event_id is not None
        assert event.timestamp is not None
    
    def test_event_with_all_fields(self):
        """Test creating an event with all fields."""
        now = datetime.now(timezone.utc)
        
        event = AuditEvent(
            event_id="evt-123",
            event_type=AuditEventType.TOOL_FAILED,
            severity=AuditEventSeverity.ERROR,
            timestamp=now,
            trace_id="trace-123",
            span_id="span-456",
            parent_span_id="span-000",
            agent_id="agent-1",
            session_id="session-1",
            message="Tool failed",
            data={"error": "timeout"},
            tags=("critical", "tool"),
            source="test",
        )
        
        assert event.event_id == "evt-123"
        assert event.trace_id == "trace-123"
        assert event.span_id == "span-456"
        assert event.parent_span_id == "span-000"
        assert event.agent_id == "agent-1"
        assert event.session_id == "session-1"
        assert event.tags == ("critical", "tool")
        assert event.source == "test"
    
    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event = AuditEvent(
            event_type=AuditEventType.STATE_CHANGED,
            message="State changed",
            data={"old": "idle", "new": "running"},
        )
        
        d = event.to_dict()
        
        assert d["event_type"] == "state.changed"
        assert d["message"] == "State changed"
        assert d["data"]["old"] == "idle"
        assert "timestamp" in d
    
    def test_event_from_dict(self):
        """Test creating event from dictionary."""
        d = {
            "event_id": "evt-123",
            "event_type": "tool.invoked",
            "severity": "info",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "message": "Test event",
            "data": {"key": "value"},
            "tags": ["tag1", "tag2"],
        }
        
        event = AuditEvent.from_dict(d)
        
        assert event.event_id == "evt-123"
        assert event.event_type == AuditEventType.TOOL_INVOKED
        assert event.severity == AuditEventSeverity.INFO
        assert event.message == "Test event"
        assert event.data["key"] == "value"


class TestEventFactory:
    """Tests for EventFactory."""
    
    def test_create_factory(self):
        """Test creating an event factory."""
        factory = EventFactory(
            agent_id="agent-1",
            session_id="session-1",
        )
        
        assert factory.agent_id == "agent-1"
        assert factory.session_id == "session-1"
    
    def test_tool_invoked_event(self):
        """Test creating tool invoked event."""
        factory = EventFactory(agent_id="agent-1")
        
        event = factory.tool_invoked(
            tool_name="search",
            invocation_id="inv-123",
            arguments={"query": "test"},
        )
        
        assert event.event_type == AuditEventType.TOOL_INVOKED
        assert event.agent_id == "agent-1"
        assert event.data["tool_name"] == "search"
        assert event.data["invocation_id"] == "inv-123"
        assert event.data["arguments"]["query"] == "test"
    
    def test_tool_completed_event(self):
        """Test creating tool completed event."""
        factory = EventFactory()
        
        event = factory.tool_completed(
            tool_name="search",
            invocation_id="inv-123",
            duration_ms=150.5,
            output_summary="Found 10 results",
        )
        
        assert event.event_type == AuditEventType.TOOL_COMPLETED
        assert event.data["duration_ms"] == 150.5
        assert event.data["output_summary"] == "Found 10 results"
    
    def test_tool_failed_event(self):
        """Test creating tool failed event."""
        factory = EventFactory()
        
        event = factory.tool_failed(
            tool_name="search",
            invocation_id="inv-123",
            error="Connection timeout",
            error_type="TimeoutError",
        )
        
        assert event.event_type == AuditEventType.TOOL_FAILED
        assert event.severity == AuditEventSeverity.ERROR
        assert event.data["error"] == "Connection timeout"
        assert event.data["error_type"] == "TimeoutError"
    
    def test_policy_denied_event(self):
        """Test creating policy denied event."""
        factory = EventFactory()
        
        event = factory.policy_denied(
            tool_name="delete_file",
            policy_name="no_delete",
            reason="File deletion not allowed",
        )
        
        assert event.event_type == AuditEventType.POLICY_DENIED
        assert event.severity == AuditEventSeverity.WARNING
        assert event.data["policy_name"] == "no_delete"
    
    def test_state_changed_event(self):
        """Test creating state changed event."""
        factory = EventFactory()
        
        event = factory.state_changed(
            old_state="idle",
            new_state="running",
            trigger="start_command",
        )
        
        assert event.event_type == AuditEventType.STATE_CHANGED
        assert event.data["old_state"] == "idle"
        assert event.data["new_state"] == "running"
        assert event.data["trigger"] == "start_command"


# =============================================================================
# TraceContext Tests
# =============================================================================

class TestTraceContext:
    """Tests for TraceContext."""
    
    def test_create_trace(self):
        """Test creating a trace context."""
        trace = TraceContext(
            agent_id="agent-1",
            session_id="session-1",
        )
        
        assert trace.trace_id is not None
        assert trace.agent_id == "agent-1"
        assert trace.session_id == "session-1"
        assert trace.started_at is not None
        assert trace.ended_at is None
        assert len(trace.spans) == 0
    
    def test_start_span(self):
        """Test starting a span."""
        trace = TraceContext()
        
        span = trace.start_span("test_operation")
        
        assert span.name == "test_operation"
        assert span.span_id is not None
        assert span.parent_span_id is None
        assert span.started_at is not None
        assert span.ended_at is None
        assert span in trace.spans
    
    def test_nested_spans(self):
        """Test nested spans."""
        trace = TraceContext()
        
        parent = trace.start_span("parent")
        child = trace.start_span("child")
        
        assert child.parent_span_id == parent.span_id
        
        grandchild = trace.start_span("grandchild")
        assert grandchild.parent_span_id == child.span_id
    
    def test_end_span(self):
        """Test ending a span."""
        trace = TraceContext()
        
        span = trace.start_span("test")
        trace.end_span(status="ok")
        
        assert span.ended_at is not None
        assert span.status == "ok"
    
    def test_end_trace(self):
        """Test ending a trace."""
        trace = TraceContext()
        trace.start_span("test")
        
        trace.end()
        
        assert trace.ended_at is not None
        # All spans should be ended
        for span in trace.spans:
            assert span.ended_at is not None
    
    def test_trace_to_dict(self):
        """Test converting trace to dictionary."""
        trace = TraceContext(agent_id="agent-1")
        trace.start_span("test")
        trace.end()
        
        d = trace.to_dict()
        
        assert d["trace_id"] == trace.trace_id
        assert d["agent_id"] == "agent-1"
        assert len(d["spans"]) == 1
    
    def test_trace_from_dict(self):
        """Test creating trace from dictionary."""
        d = {
            "trace_id": "trace-123",
            "agent_id": "agent-1",
            "session_id": "session-1",
            "started_at": "2024-01-01T00:00:00+00:00",
            "ended_at": "2024-01-01T00:01:00+00:00",
            "spans": [
                {
                    "span_id": "span-1",
                    "trace_id": "trace-123",
                    "name": "test",
                    "started_at": "2024-01-01T00:00:00+00:00",
                    "ended_at": "2024-01-01T00:00:30+00:00",
                    "status": "ok",
                }
            ],
            "attributes": {"key": "value"},
        }
        
        trace = TraceContext.from_dict(d)
        
        assert trace.trace_id == "trace-123"
        assert trace.agent_id == "agent-1"
        assert len(trace.spans) == 1
        assert trace.spans[0].name == "test"


class TestSpanContext:
    """Tests for SpanContext."""
    
    def test_create_span(self):
        """Test creating a span context."""
        span = SpanContext(name="test_operation")
        
        assert span.span_id is not None
        assert span.name == "test_operation"
        assert span.started_at is not None
        assert span.ended_at is None
        assert span.status == "ok"
    
    def test_span_with_parent(self):
        """Test creating a span with parent."""
        span = SpanContext(
            name="child",
            parent_span_id="parent-123",
        )
        
        assert span.parent_span_id == "parent-123"
    
    def test_span_attributes(self):
        """Test span attributes."""
        span = SpanContext(
            name="test",
            attributes={"key": "value"},
        )
        
        assert span.attributes["key"] == "value"
    
    def test_span_end(self):
        """Test ending a span."""
        span = SpanContext(name="test")
        span.end(status="error")
        
        assert span.ended_at is not None
        assert span.status == "error"
    
    def test_span_duration(self):
        """Test span duration calculation."""
        span = SpanContext(name="test")
        span.end()
        
        duration = span.duration_ms
        assert duration is not None
        assert duration >= 0


class TestTraceContextManager:
    """Tests for trace context manager."""
    
    def test_trace_context_manager(self):
        """Test using trace as context manager."""
        with trace_context(agent_id="agent-1") as trace:
            assert current_trace() == trace
            
            with span_context("operation") as span:
                assert current_span() == span
        
        # Context should be cleared after exiting
        assert current_trace() is None
    
    def test_nested_trace_context(self):
        """Test nested trace contexts."""
        with trace_context() as outer:
            assert current_trace() == outer
            
            # Inner trace should replace outer
            with trace_context() as inner:
                assert current_trace() == inner
            
            # Should restore outer
            assert current_trace() == outer


# =============================================================================
# AuditLogger Tests
# =============================================================================

class TestAuditLogger:
    """Tests for AuditLogger."""
    
    @pytest.fixture
    def logger(self):
        """Create a logger without console output."""
        config = AuditLogConfig(log_to_console=False)
        return AuditLogger(config=config, agent_id="test-agent")
    
    @pytest.mark.asyncio
    async def test_log_event(self, logger):
        """Test logging an event."""
        events = []
        
        async def capture(event):
            events.append(event)
        
        logger.add_handler(CallbackHandler(capture))
        
        await logger.log_event(
            AuditEventType.TOOL_INVOKED,
            "Tool invoked",
            data={"tool": "test"},
        )
        
        assert len(events) == 1
        assert events[0].event_type == AuditEventType.TOOL_INVOKED
        assert events[0].agent_id == "test-agent"
    
    @pytest.mark.asyncio
    async def test_severity_filtering(self, logger):
        """Test severity filtering."""
        events = []
        
        async def capture(event):
            events.append(event)
        
        logger.config = AuditLogConfig(
            log_to_console=False,
            min_severity=AuditEventSeverity.WARNING,
        )
        logger.add_handler(CallbackHandler(capture))
        
        await logger.debug("Debug message")
        await logger.info("Info message")
        await logger.warning("Warning message")
        await logger.error("Error message")
        
        # Only warning and error should be logged
        assert len(events) == 2
        assert events[0].severity == AuditEventSeverity.WARNING
        assert events[1].severity == AuditEventSeverity.ERROR
    
    @pytest.mark.asyncio
    async def test_event_type_filtering(self, logger):
        """Test event type filtering."""
        events = []
        
        async def capture(event):
            events.append(event)
        
        logger.config = AuditLogConfig(
            log_to_console=False,
            include_event_types=(AuditEventType.TOOL_INVOKED,),
        )
        logger.add_handler(CallbackHandler(capture))
        
        await logger.log_event(AuditEventType.TOOL_INVOKED, "Invoked")
        await logger.log_event(AuditEventType.TOOL_COMPLETED, "Completed")
        
        assert len(events) == 1
        assert events[0].event_type == AuditEventType.TOOL_INVOKED
    
    @pytest.mark.asyncio
    async def test_tool_events(self, logger):
        """Test tool event convenience methods."""
        events = []
        
        async def capture(event):
            events.append(event)
        
        logger.add_handler(CallbackHandler(capture))
        
        await logger.tool_invoked("search", "inv-1", {"query": "test"})
        await logger.tool_completed("search", "inv-1", 100.0)
        await logger.tool_failed("search", "inv-2", "Error")
        
        assert len(events) == 3
        assert events[0].event_type == AuditEventType.TOOL_INVOKED
        assert events[1].event_type == AuditEventType.TOOL_COMPLETED
        assert events[2].event_type == AuditEventType.TOOL_FAILED
    
    @pytest.mark.asyncio
    async def test_trace_context_enrichment(self, logger):
        """Test that events are enriched with trace context."""
        events = []
        
        async def capture(event):
            events.append(event)
        
        logger.add_handler(CallbackHandler(capture))
        
        with trace_context() as trace:
            with span_context("test"):
                await logger.info("Test message")
        
        assert len(events) == 1
        assert events[0].trace_id == trace.trace_id


class TestBufferedHandler:
    """Tests for BufferedHandler."""
    
    @pytest.mark.asyncio
    async def test_buffering(self):
        """Test event buffering."""
        inner_events = []
        
        async def capture(event):
            inner_events.append(event)
        
        inner = CallbackHandler(capture)
        handler = BufferedHandler(inner, buffer_size=3)
        
        # Add events
        for i in range(2):
            await handler.handle(AuditEvent(
                event_type=AuditEventType.SYSTEM_INFO,
                message=f"Event {i}",
            ))
        
        # Not flushed yet
        assert len(inner_events) == 0
        
        # Add one more to trigger flush
        await handler.handle(AuditEvent(
            event_type=AuditEventType.SYSTEM_INFO,
            message="Event 2",
        ))
        
        # Should be flushed
        assert len(inner_events) == 3
    
    @pytest.mark.asyncio
    async def test_manual_flush(self):
        """Test manual flush."""
        inner_events = []
        
        async def capture(event):
            inner_events.append(event)
        
        inner = CallbackHandler(capture)
        handler = BufferedHandler(inner, buffer_size=100)
        
        await handler.handle(AuditEvent(
            event_type=AuditEventType.SYSTEM_INFO,
            message="Test",
        ))
        
        assert len(inner_events) == 0
        
        await handler.flush()
        
        assert len(inner_events) == 1


# =============================================================================
# AuditStorage Tests
# =============================================================================

class TestInMemoryAuditStorage:
    """Tests for InMemoryAuditStorage."""
    
    @pytest.fixture
    def storage(self):
        """Create in-memory storage."""
        return InMemoryAuditStorage(max_events=100, max_traces=10)
    
    @pytest.mark.asyncio
    async def test_store_and_get_event(self, storage):
        """Test storing and retrieving an event."""
        event = AuditEvent(
            event_type=AuditEventType.TOOL_INVOKED,
            message="Test event",
        )
        
        await storage.store_event(event)
        
        retrieved = await storage.get_event(event.event_id)
        
        assert retrieved is not None
        assert retrieved.event_id == event.event_id
        assert retrieved.message == "Test event"
    
    @pytest.mark.asyncio
    async def test_store_and_get_trace(self, storage):
        """Test storing and retrieving a trace."""
        trace = TraceContext(agent_id="agent-1")
        trace.start_span("test")
        trace.end()
        
        await storage.store_trace(trace)
        
        retrieved = await storage.get_trace(trace.trace_id)
        
        assert retrieved is not None
        assert retrieved.trace_id == trace.trace_id
        assert retrieved.agent_id == "agent-1"
    
    @pytest.mark.asyncio
    async def test_query_events_by_agent(self, storage):
        """Test querying events by agent ID."""
        for i in range(5):
            await storage.store_event(AuditEvent(
                event_type=AuditEventType.SYSTEM_INFO,
                message=f"Event {i}",
                agent_id="agent-1" if i < 3 else "agent-2",
            ))
        
        results = await storage.query_events(agent_id="agent-1")
        
        assert len(results) == 3
        for event in results:
            assert event.agent_id == "agent-1"
    
    @pytest.mark.asyncio
    async def test_query_events_by_type(self, storage):
        """Test querying events by type."""
        await storage.store_event(AuditEvent(
            event_type=AuditEventType.TOOL_INVOKED,
            message="Invoked",
        ))
        await storage.store_event(AuditEvent(
            event_type=AuditEventType.TOOL_COMPLETED,
            message="Completed",
        ))
        
        results = await storage.query_events(
            event_types=[AuditEventType.TOOL_INVOKED]
        )
        
        assert len(results) == 1
        assert results[0].event_type == AuditEventType.TOOL_INVOKED
    
    @pytest.mark.asyncio
    async def test_query_events_by_severity(self, storage):
        """Test querying events by minimum severity."""
        await storage.store_event(AuditEvent(
            event_type=AuditEventType.SYSTEM_INFO,
            severity=AuditEventSeverity.DEBUG,
            message="Debug",
        ))
        await storage.store_event(AuditEvent(
            event_type=AuditEventType.SYSTEM_INFO,
            severity=AuditEventSeverity.ERROR,
            message="Error",
        ))
        
        results = await storage.query_events(
            min_severity=AuditEventSeverity.WARNING
        )
        
        assert len(results) == 1
        assert results[0].severity == AuditEventSeverity.ERROR
    
    @pytest.mark.asyncio
    async def test_query_events_pagination(self, storage):
        """Test event query pagination."""
        for i in range(10):
            await storage.store_event(AuditEvent(
                event_type=AuditEventType.SYSTEM_INFO,
                message=f"Event {i}",
            ))
        
        page1 = await storage.query_events(limit=3, offset=0)
        page2 = await storage.query_events(limit=3, offset=3)
        
        assert len(page1) == 3
        assert len(page2) == 3
        
        # Should be different events
        page1_ids = {e.event_id for e in page1}
        page2_ids = {e.event_id for e in page2}
        assert page1_ids.isdisjoint(page2_ids)
    
    @pytest.mark.asyncio
    async def test_delete_events(self, storage):
        """Test deleting events."""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=7)
        
        # Store old and new events
        old_event = AuditEvent(
            event_type=AuditEventType.SYSTEM_INFO,
            message="Old event",
            timestamp=old_time,
        )
        new_event = AuditEvent(
            event_type=AuditEventType.SYSTEM_INFO,
            message="New event",
            timestamp=now,
        )
        
        await storage.store_event(old_event)
        await storage.store_event(new_event)
        
        # Delete old events
        deleted = await storage.delete_events(before=now - timedelta(days=1))
        
        assert deleted == 1
        assert await storage.get_event(old_event.event_id) is None
        assert await storage.get_event(new_event.event_id) is not None
    
    @pytest.mark.asyncio
    async def test_event_count(self, storage):
        """Test getting event count."""
        for i in range(5):
            await storage.store_event(AuditEvent(
                event_type=AuditEventType.SYSTEM_INFO,
                message=f"Event {i}",
                agent_id="agent-1" if i < 3 else "agent-2",
            ))
        
        total = await storage.get_event_count()
        agent1_count = await storage.get_event_count(agent_id="agent-1")
        
        assert total == 5
        assert agent1_count == 3
    
    @pytest.mark.asyncio
    async def test_max_events_eviction(self, storage):
        """Test that old events are evicted when max is reached."""
        storage = InMemoryAuditStorage(max_events=5)
        
        for i in range(10):
            await storage.store_event(AuditEvent(
                event_type=AuditEventType.SYSTEM_INFO,
                message=f"Event {i}",
            ))
        
        count = await storage.get_event_count()
        assert count == 5


# =============================================================================
# TraceViewer Tests
# =============================================================================

class TestTraceViewer:
    """Tests for TraceViewer."""
    
    @pytest.fixture
    def storage(self):
        """Create in-memory storage."""
        return InMemoryAuditStorage()
    
    @pytest.fixture
    def viewer(self, storage):
        """Create trace viewer."""
        return TraceViewer(storage)
    
    @pytest.mark.asyncio
    async def test_build_trace_tree(self, storage, viewer):
        """Test building a trace tree."""
        # Create and store a trace
        trace = TraceContext(agent_id="agent-1")
        parent = trace.start_span("parent")
        child = trace.start_span("child")
        trace.end_span()  # End child
        trace.end_span()  # End parent
        trace.end()
        
        await storage.store_trace(trace)
        
        # Store some events
        await storage.store_event(AuditEvent(
            event_type=AuditEventType.TOOL_INVOKED,
            message="Tool invoked",
            trace_id=trace.trace_id,
            span_id=child.span_id,
        ))
        
        # Build tree
        tree = await viewer.build_trace_tree(trace.trace_id)
        
        assert tree is not None
        assert tree.trace.trace_id == trace.trace_id
        assert len(tree.root_spans) == 1
        assert tree.root_spans[0].span.name == "parent"
        assert len(tree.root_spans[0].children) == 1
        assert tree.root_spans[0].children[0].span.name == "child"
    
    @pytest.mark.asyncio
    async def test_build_timeline(self, storage, viewer):
        """Test building a timeline."""
        trace = TraceContext()
        trace.start_span("test")
        trace.end()
        
        await storage.store_trace(trace)
        
        await storage.store_event(AuditEvent(
            event_type=AuditEventType.TOOL_INVOKED,
            message="Event 1",
            trace_id=trace.trace_id,
        ))
        await storage.store_event(AuditEvent(
            event_type=AuditEventType.TOOL_COMPLETED,
            message="Event 2",
            trace_id=trace.trace_id,
        ))
        
        timeline = await viewer.build_timeline(trace_id=trace.trace_id)
        
        # Should have events + span start/end
        assert len(timeline) >= 2
    
    @pytest.mark.asyncio
    async def test_build_causal_graph(self, storage, viewer):
        """Test building a causal graph."""
        trace = TraceContext()
        trace.start_span("test")
        trace.end()
        
        await storage.store_trace(trace)
        
        # Store tool invocation and completion
        await storage.store_event(AuditEvent(
            event_type=AuditEventType.TOOL_INVOKED,
            message="Tool invoked",
            trace_id=trace.trace_id,
            data={"invocation_id": "inv-1", "tool_name": "search"},
        ))
        await storage.store_event(AuditEvent(
            event_type=AuditEventType.TOOL_COMPLETED,
            message="Tool completed",
            trace_id=trace.trace_id,
            data={"invocation_id": "inv-1", "tool_name": "search"},
        ))
        
        graph = await viewer.build_causal_graph(trace.trace_id)
        
        assert graph is not None
        assert len(graph.events) == 2
        assert len(graph.links) == 1
        assert graph.links[0].link_type == "triggers"
    
    @pytest.mark.asyncio
    async def test_get_event_context(self, storage, viewer):
        """Test getting event context."""
        trace = TraceContext()
        trace.start_span("test")
        trace.end()
        
        await storage.store_trace(trace)
        
        events = []
        for i in range(10):
            event = AuditEvent(
                event_type=AuditEventType.SYSTEM_INFO,
                message=f"Event {i}",
                trace_id=trace.trace_id,
            )
            events.append(event)
            await storage.store_event(event)
        
        # Get context for middle event
        context = await viewer.get_event_context(
            events[5].event_id,
            context_window=2,
        )
        
        assert "event" in context
        assert "before" in context
        assert "after" in context
        assert len(context["before"]) <= 2
        assert len(context["after"]) <= 2
    
    @pytest.mark.asyncio
    async def test_search_events(self, storage, viewer):
        """Test searching events."""
        await storage.store_event(AuditEvent(
            event_type=AuditEventType.TOOL_INVOKED,
            message="Search tool invoked",
        ))
        await storage.store_event(AuditEvent(
            event_type=AuditEventType.TOOL_COMPLETED,
            message="File read completed",
        ))
        
        results = await viewer.search_events("search")
        
        assert len(results) == 1
        assert "Search" in results[0].message
    
    @pytest.mark.asyncio
    async def test_get_statistics(self, storage, viewer):
        """Test getting statistics."""
        await storage.store_event(AuditEvent(
            event_type=AuditEventType.TOOL_INVOKED,
            message="Invoked",
            data={"tool_name": "search"},
        ))
        await storage.store_event(AuditEvent(
            event_type=AuditEventType.TOOL_COMPLETED,
            message="Completed",
        ))
        await storage.store_event(AuditEvent(
            event_type=AuditEventType.TOOL_FAILED,
            message="Failed",
        ))
        
        stats = await viewer.get_statistics()
        
        assert stats["total_events"] == 3
        assert stats["by_type"]["tool.invoked"] == 1
        assert stats["by_type"]["tool.completed"] == 1
        assert stats["by_type"]["tool.failed"] == 1
        assert stats["tools"]["total_invocations"] == 1
        assert stats["tools"]["total_completions"] == 1
        assert stats["tools"]["total_failures"] == 1
