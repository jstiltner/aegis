"""
Unit tests for the API module.

Tests cover:
- API models
- Route handlers
- Server configuration
"""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient

from aegis.api import (
    create_app,
    SessionCreate,
    SessionResponse,
    AgentCreate,
    AgentResponse,
    MessageRequest,
    MessageResponse,
    TraceResponse,
    ToolResponse,
)
from aegis.api.models import (
    SessionStatus,
    SessionList,
    AgentList,
    MessageRole,
    MessageList,
    ToolUseResponse,
    TraceList,
    SpanResponse,
    ToolList,
    ToolParameter,
    ToolInvocation,
    ToolResult,
    CheckpointResponse,
    CheckpointList,
    CommitmentResponse,
    CommitmentList,
    CommitmentStatus,
    ErrorResponse,
    HealthResponse,
    HealthStatus,
)


# =============================================================================
# API Model Tests
# =============================================================================

class TestSessionModels:
    """Tests for session models."""
    
    def test_session_create(self):
        """Test SessionCreate model."""
        request = SessionCreate(
            agent_id="agent-001",
            system_prompt="You are helpful.",
            metadata={"key": "value"},
        )
        
        assert request.agent_id == "agent-001"
        assert request.system_prompt == "You are helpful."
        assert request.metadata == {"key": "value"}
    
    def test_session_response(self):
        """Test SessionResponse model."""
        now = datetime.now(timezone.utc)
        response = SessionResponse(
            session_id="sess-001",
            agent_id="agent-001",
            status=SessionStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            message_count=10,
            checkpoint_count=2,
        )
        
        assert response.session_id == "sess-001"
        assert response.status == SessionStatus.ACTIVE
        assert response.message_count == 10
    
    def test_session_list(self):
        """Test SessionList model."""
        sessions = SessionList(
            sessions=[],
            total=0,
            offset=0,
            limit=20,
        )
        
        assert sessions.total == 0
        assert sessions.limit == 20


class TestAgentModels:
    """Tests for agent models."""
    
    def test_agent_create(self):
        """Test AgentCreate model."""
        request = AgentCreate(
            name="Test Agent",
            description="A test agent",
            model="claude-sonnet-4-20250514",
            tools=["read_file", "write_file"],
        )
        
        assert request.name == "Test Agent"
        assert len(request.tools) == 2
    
    def test_agent_response(self):
        """Test AgentResponse model."""
        now = datetime.now(timezone.utc)
        response = AgentResponse(
            agent_id="agent-001",
            name="Test Agent",
            model="claude-sonnet-4-20250514",
            created_at=now,
        )
        
        assert response.agent_id == "agent-001"
        assert response.name == "Test Agent"


class TestMessageModels:
    """Tests for message models."""
    
    def test_message_request(self):
        """Test MessageRequest model."""
        request = MessageRequest(
            content="Hello, world!",
            role=MessageRole.USER,
        )
        
        assert request.content == "Hello, world!"
        assert request.role == MessageRole.USER
    
    def test_message_response(self):
        """Test MessageResponse model."""
        now = datetime.now(timezone.utc)
        response = MessageResponse(
            session_id="sess-001",
            role=MessageRole.ASSISTANT,
            content="Hello!",
            created_at=now,
        )
        
        assert response.role == MessageRole.ASSISTANT
        assert response.content == "Hello!"
    
    def test_message_with_tool_uses(self):
        """Test MessageResponse with tool uses."""
        now = datetime.now(timezone.utc)
        response = MessageResponse(
            session_id="sess-001",
            role=MessageRole.ASSISTANT,
            content="Let me search for that.",
            tool_uses=[
                ToolUseResponse(
                    id="tool-1",
                    name="search",
                    input={"query": "test"},
                ),
            ],
            created_at=now,
        )
        
        assert len(response.tool_uses) == 1
        assert response.tool_uses[0].name == "search"


class TestTraceModels:
    """Tests for trace models."""
    
    def test_span_response(self):
        """Test SpanResponse model."""
        now = datetime.now(timezone.utc)
        span = SpanResponse(
            span_id="span-001",
            trace_id="trace-001",
            name="test.span",
            start_time=now,
            duration_ms=100.0,
        )
        
        assert span.span_id == "span-001"
        assert span.name == "test.span"
    
    def test_trace_response(self):
        """Test TraceResponse model."""
        now = datetime.now(timezone.utc)
        trace = TraceResponse(
            trace_id="trace-001",
            session_id="sess-001",
            start_time=now,
            span_count=5,
        )
        
        assert trace.trace_id == "trace-001"
        assert trace.span_count == 5


class TestToolModels:
    """Tests for tool models."""
    
    def test_tool_parameter(self):
        """Test ToolParameter model."""
        param = ToolParameter(
            name="path",
            type="string",
            description="File path",
            required=True,
        )
        
        assert param.name == "path"
        assert param.required is True
    
    def test_tool_response(self):
        """Test ToolResponse model."""
        tool = ToolResponse(
            name="read_file",
            description="Read a file",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    required=True,
                ),
            ],
            category="filesystem",
        )
        
        assert tool.name == "read_file"
        assert len(tool.parameters) == 1
    
    def test_tool_invocation(self):
        """Test ToolInvocation model."""
        invocation = ToolInvocation(
            tool_name="read_file",
            input={"path": "/tmp/test.txt"},
        )
        
        assert invocation.tool_name == "read_file"
    
    def test_tool_result(self):
        """Test ToolResult model."""
        result = ToolResult(
            tool_name="read_file",
            output="File contents",
            is_error=False,
            duration_ms=50.0,
        )
        
        assert result.output == "File contents"
        assert result.is_error is False


class TestCheckpointModels:
    """Tests for checkpoint models."""
    
    def test_checkpoint_response(self):
        """Test CheckpointResponse model."""
        now = datetime.now(timezone.utc)
        checkpoint = CheckpointResponse(
            checkpoint_id="cp-001",
            session_id="sess-001",
            version=5,
            created_at=now,
        )
        
        assert checkpoint.checkpoint_id == "cp-001"
        assert checkpoint.version == 5


class TestCommitmentModels:
    """Tests for commitment models."""
    
    def test_commitment_response(self):
        """Test CommitmentResponse model."""
        now = datetime.now(timezone.utc)
        commitment = CommitmentResponse(
            commitment_id="commit-001",
            session_id="sess-001",
            debtor="agent",
            creditor="user",
            antecedent="request_received",
            consequent="provide_response",
            status=CommitmentStatus.ACTIVE,
            created_at=now,
        )
        
        assert commitment.commitment_id == "commit-001"
        assert commitment.status == CommitmentStatus.ACTIVE


class TestErrorModels:
    """Tests for error models."""
    
    def test_error_response(self):
        """Test ErrorResponse model."""
        error = ErrorResponse(
            error="validation_error",
            message="Invalid input",
            details={"field": "name"},
        )
        
        assert error.error == "validation_error"
        assert error.message == "Invalid input"


class TestHealthModels:
    """Tests for health models."""
    
    def test_health_response(self):
        """Test HealthResponse model."""
        health = HealthResponse(
            status=HealthStatus.HEALTHY,
            version="0.1.0",
            uptime_seconds=3600.0,
            checks={
                "api": HealthStatus.HEALTHY,
                "database": HealthStatus.HEALTHY,
            },
        )
        
        assert health.status == HealthStatus.HEALTHY
        assert health.uptime_seconds == 3600.0


# =============================================================================
# API Server Tests
# =============================================================================

class TestAPIServer:
    """Tests for API server."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app(debug=True)
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
    
    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data
    
    def test_request_id_header(self, client):
        """Test request ID header is added."""
        response = client.get("/health")
        
        assert "X-Request-ID" in response.headers
    
    def test_response_time_header(self, client):
        """Test response time header is added."""
        response = client.get("/health")
        
        assert "X-Response-Time" in response.headers


# =============================================================================
# Session Route Tests
# =============================================================================

class TestSessionRoutes:
    """Tests for session routes."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app(debug=True)
        return TestClient(app)
    
    def test_create_session(self, client):
        """Test creating a session."""
        response = client.post(
            "/api/v1/sessions",
            json={
                "agent_id": "agent-001",
                "system_prompt": "You are helpful.",
            },
        )
        
        assert response.status_code == 201
        data = response.json()
        assert "session_id" in data
        assert data["agent_id"] == "agent-001"
        assert data["status"] == "active"
    
    def test_list_sessions(self, client):
        """Test listing sessions."""
        response = client.get("/api/v1/sessions")
        
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "total" in data
    
    def test_get_session_not_found(self, client):
        """Test getting non-existent session."""
        response = client.get("/api/v1/sessions/nonexistent")
        
        assert response.status_code == 404
    
    def test_send_message(self, client):
        """Test sending a message."""
        # First create a session
        create_response = client.post(
            "/api/v1/sessions",
            json={"agent_id": "agent-001"},
        )
        session_id = create_response.json()["session_id"]
        
        # Send a message
        response = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"content": "Hello!"},
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["role"] == "assistant"
        assert "content" in data
    
    def test_get_messages(self, client):
        """Test getting session messages."""
        response = client.get("/api/v1/sessions/sess-001/messages")
        
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
    
    def test_get_checkpoints(self, client):
        """Test getting session checkpoints."""
        response = client.get("/api/v1/sessions/sess-001/checkpoints")
        
        assert response.status_code == 200
        data = response.json()
        assert "checkpoints" in data
    
    def test_get_commitments(self, client):
        """Test getting session commitments."""
        response = client.get("/api/v1/sessions/sess-001/commitments")
        
        assert response.status_code == 200
        data = response.json()
        assert "commitments" in data


# =============================================================================
# Agent Route Tests
# =============================================================================

class TestAgentRoutes:
    """Tests for agent routes."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app(debug=True)
        return TestClient(app)
    
    def test_create_agent(self, client):
        """Test creating an agent."""
        response = client.post(
            "/api/v1/agents",
            json={
                "name": "Test Agent",
                "model": "claude-sonnet-4-20250514",
            },
        )
        
        assert response.status_code == 201
        data = response.json()
        assert "agent_id" in data
        assert data["name"] == "Test Agent"
    
    def test_list_agents(self, client):
        """Test listing agents."""
        response = client.get("/api/v1/agents")
        
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
    
    def test_get_agent_not_found(self, client):
        """Test getting non-existent agent."""
        response = client.get("/api/v1/agents/nonexistent")
        
        assert response.status_code == 404


# =============================================================================
# Trace Route Tests
# =============================================================================

class TestTraceRoutes:
    """Tests for trace routes."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app(debug=True)
        return TestClient(app)
    
    def test_list_traces(self, client):
        """Test listing traces."""
        response = client.get("/api/v1/traces")
        
        assert response.status_code == 200
        data = response.json()
        assert "traces" in data
    
    def test_get_trace_not_found(self, client):
        """Test getting non-existent trace."""
        response = client.get("/api/v1/traces/nonexistent")
        
        assert response.status_code == 404


# =============================================================================
# Tool Route Tests
# =============================================================================

class TestToolRoutes:
    """Tests for tool routes."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app(debug=True)
        return TestClient(app)
    
    def test_list_tools(self, client):
        """Test listing tools."""
        response = client.get("/api/v1/tools")
        
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert len(data["tools"]) > 0
    
    def test_list_tools_by_category(self, client):
        """Test listing tools by category."""
        response = client.get("/api/v1/tools?category=filesystem")
        
        assert response.status_code == 200
        data = response.json()
        for tool in data["tools"]:
            assert tool["category"] == "filesystem"
    
    def test_invoke_tool(self, client):
        """Test invoking a tool."""
        response = client.post(
            "/api/v1/tools/read_file/invoke",
            json={
                "tool_name": "read_file",
                "input": {"path": "/tmp/test.txt"},
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["tool_name"] == "read_file"
        assert "output" in data
