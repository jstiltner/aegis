"""
API server for the agent runtime.

This package provides:
- FastAPI-based REST API
- WebSocket support for real-time updates
- Session management endpoints
- Trace and audit log viewing
"""

from __future__ import annotations

from .server import (
    create_app,
    AgentAPI,
)
from .routes import (
    sessions_router,
    agents_router,
    traces_router,
    tools_router,
)
from .models import (
    SessionCreate,
    SessionResponse,
    AgentCreate,
    AgentResponse,
    MessageRequest,
    MessageResponse,
    TraceResponse,
    ToolResponse,
)

__all__ = [
    # Server
    "create_app",
    "AgentAPI",
    # Routes
    "sessions_router",
    "agents_router",
    "traces_router",
    "tools_router",
    # Models
    "SessionCreate",
    "SessionResponse",
    "AgentCreate",
    "AgentResponse",
    "MessageRequest",
    "MessageResponse",
    "TraceResponse",
    "ToolResponse",
]
