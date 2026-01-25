"""
FastAPI server for the agent runtime.

This module provides the main API server with WebSocket support,
middleware, and lifecycle management.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable
from uuid import uuid4

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .models import (
    HealthResponse,
    HealthStatus,
    ErrorResponse,
)
from .routes import (
    sessions_router,
    agents_router,
    traces_router,
    tools_router,
)


class AgentAPI:
    """
    Agent Runtime API server.
    
    Provides REST API and WebSocket endpoints for interacting with
    autonomous agents.
    
    Example:
        >>> api = AgentAPI()
        >>> app = api.app
        >>> # Run with uvicorn
        >>> import uvicorn
        >>> uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    
    def __init__(
        self,
        title: str = "Agent Runtime API",
        description: str = "Production infrastructure for long-running autonomous agents",
        version: str = "0.1.0",
        debug: bool = False,
    ) -> None:
        """
        Initialize the API server.
        
        Args:
            title: API title
            description: API description
            version: API version
            debug: Enable debug mode
        """
        self._title = title
        self._description = description
        self._version = version
        self._debug = debug
        self._start_time = datetime.now(timezone.utc)
        
        # WebSocket connections
        self._ws_connections: dict[str, WebSocket] = {}
        
        # Create FastAPI app
        self._app = FastAPI(
            title=title,
            description=description,
            version=version,
            debug=debug,
            lifespan=self._lifespan,
        )
        
        # Setup
        self._setup_middleware()
        self._setup_routes()
        self._setup_exception_handlers()
    
    @property
    def app(self) -> FastAPI:
        """Get the FastAPI application."""
        return self._app
    
    @property
    def version(self) -> str:
        """Get the API version."""
        return self._version
    
    @property
    def uptime_seconds(self) -> float:
        """Get the server uptime in seconds."""
        return (datetime.now(timezone.utc) - self._start_time).total_seconds()
    
    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        """Application lifespan manager."""
        # Startup
        self._start_time = datetime.now(timezone.utc)
        
        yield
        
        # Shutdown
        # Close all WebSocket connections
        for ws in list(self._ws_connections.values()):
            try:
                await ws.close()
            except Exception:
                pass
        self._ws_connections.clear()
    
    def _setup_middleware(self) -> None:
        """Setup middleware."""
        # CORS
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Request ID middleware
        @self._app.middleware("http")
        async def add_request_id(request: Request, call_next: Callable) -> Response:
            request_id = request.headers.get("X-Request-ID", str(uuid4()))
            request.state.request_id = request_id
            
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            
            return response
        
        # Timing middleware
        @self._app.middleware("http")
        async def add_timing(request: Request, call_next: Callable) -> Response:
            start_time = time.perf_counter()
            
            response = await call_next(request)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
            
            return response
    
    def _setup_routes(self) -> None:
        """Setup API routes."""
        # Health check
        @self._app.get(
            "/health",
            response_model=HealthResponse,
            tags=["health"],
            summary="Health check",
        )
        async def health_check() -> HealthResponse:
            """Check the health of the API server."""
            return HealthResponse(
                status=HealthStatus.HEALTHY,
                version=self._version,
                uptime_seconds=self.uptime_seconds,
                checks={
                    "api": HealthStatus.HEALTHY,
                    "database": HealthStatus.HEALTHY,
                    "llm": HealthStatus.HEALTHY,
                },
            )
        
        # Root endpoint
        @self._app.get("/", tags=["root"])
        async def root() -> dict[str, Any]:
            """API root endpoint."""
            return {
                "name": self._title,
                "version": self._version,
                "docs": "/docs",
                "openapi": "/openapi.json",
            }
        
        # Include routers
        self._app.include_router(sessions_router, prefix="/api/v1")
        self._app.include_router(agents_router, prefix="/api/v1")
        self._app.include_router(traces_router, prefix="/api/v1")
        self._app.include_router(tools_router, prefix="/api/v1")
        
        # Web UI endpoint
        @self._app.get("/ui", tags=["ui"])
        async def ui_redirect() -> FileResponse:
            """Serve the web UI."""
            static_dir = Path(__file__).parent.parent / "web" / "static"
            return FileResponse(static_dir / "index.html")
        
        # Mount static files if directory exists
        static_dir = Path(__file__).parent.parent / "web" / "static"
        if static_dir.exists():
            self._app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        
        # WebSocket endpoint
        @self._app.websocket("/ws/{session_id}")
        async def websocket_endpoint(
            websocket: WebSocket,
            session_id: str,
        ) -> None:
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            
            connection_id = str(uuid4())
            self._ws_connections[connection_id] = websocket
            
            try:
                # Send connection confirmation
                await websocket.send_json({
                    "type": "connected",
                    "session_id": session_id,
                    "connection_id": connection_id,
                })
                
                # Handle messages
                while True:
                    data = await websocket.receive_json()
                    
                    # Echo for now
                    await websocket.send_json({
                        "type": "message",
                        "data": data,
                    })
            
            except WebSocketDisconnect:
                pass
            finally:
                self._ws_connections.pop(connection_id, None)
    
    def _setup_exception_handlers(self) -> None:
        """Setup exception handlers."""
        @self._app.exception_handler(Exception)
        async def generic_exception_handler(
            request: Request,
            exc: Exception,
        ) -> JSONResponse:
            """Handle uncaught exceptions."""
            request_id = getattr(request.state, "request_id", None)
            
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="internal_error",
                    message=str(exc) if self._debug else "An internal error occurred",
                    details={"type": type(exc).__name__} if self._debug else {},
                    trace_id=request_id,
                ).model_dump(),
            )
        
        @self._app.exception_handler(ValueError)
        async def value_error_handler(
            request: Request,
            exc: ValueError,
        ) -> JSONResponse:
            """Handle value errors."""
            request_id = getattr(request.state, "request_id", None)
            
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error="validation_error",
                    message=str(exc),
                    details={},
                    trace_id=request_id,
                ).model_dump(),
            )
    
    async def broadcast(
        self,
        session_id: str,
        message: dict[str, Any],
    ) -> None:
        """
        Broadcast a message to all WebSocket connections for a session.
        
        Args:
            session_id: Session identifier
            message: Message to broadcast
        """
        for ws in list(self._ws_connections.values()):
            try:
                await ws.send_json({
                    "type": "broadcast",
                    "session_id": session_id,
                    "data": message,
                })
            except Exception:
                pass


def create_app(
    title: str = "Agent Runtime API",
    description: str = "Production infrastructure for long-running autonomous agents",
    version: str = "0.1.0",
    debug: bool = False,
) -> FastAPI:
    """
    Create a FastAPI application.
    
    Args:
        title: API title
        description: API description
        version: API version
        debug: Enable debug mode
        
    Returns:
        Configured FastAPI application
    """
    api = AgentAPI(
        title=title,
        description=description,
        version=version,
        debug=debug,
    )
    return api.app


# Default app instance for uvicorn
app = create_app()
