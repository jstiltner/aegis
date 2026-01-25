"""
MCP (Model Context Protocol) adapter for tool integration.

This module provides:
- Connection to MCP servers
- Tool discovery from MCP servers
- Tool invocation through MCP protocol
- Resource access through MCP
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from aegis.tools.models import (
    Tool,
    ToolDefinition,
    ToolParameter,
    ParameterType,
    ToolResult,
    ToolError,
    ToolHandler,
)
from aegis.tools.registry import ToolRegistry


class MCPServerType(str, Enum):
    """Type of MCP server connection."""
    
    STDIO = "stdio"
    HTTP = "http"
    WEBSOCKET = "websocket"


class MCPServerConfig(BaseModel):
    """Configuration for connecting to an MCP server."""
    
    model_config = ConfigDict(frozen=True)
    
    name: str = Field(
        ...,
        description="Name for this MCP server",
    )
    server_type: MCPServerType = Field(
        default=MCPServerType.STDIO,
        description="Type of server connection",
    )
    
    # STDIO config
    command: str | None = Field(
        default=None,
        description="Command to start the server (for STDIO)",
    )
    args: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Arguments for the command",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables for the server",
    )
    
    # HTTP/WebSocket config
    url: str | None = Field(
        default=None,
        description="URL for HTTP/WebSocket connection",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Headers for HTTP/WebSocket connection",
    )
    
    # Common config
    timeout_seconds: float = Field(
        default=30.0,
        ge=0.1,
        description="Connection timeout",
    )
    auto_reconnect: bool = Field(
        default=True,
        description="Whether to auto-reconnect on disconnect",
    )


class MCPResource(BaseModel):
    """A resource available from an MCP server."""
    
    model_config = ConfigDict(frozen=True)
    
    uri: str = Field(
        ...,
        description="Resource URI",
    )
    name: str = Field(
        ...,
        description="Resource name",
    )
    description: str = Field(
        default="",
        description="Resource description",
    )
    mime_type: str = Field(
        default="text/plain",
        description="MIME type of the resource",
    )


class MCPPrompt(BaseModel):
    """A prompt template from an MCP server."""
    
    model_config = ConfigDict(frozen=True)
    
    name: str = Field(
        ...,
        description="Prompt name",
    )
    description: str = Field(
        default="",
        description="Prompt description",
    )
    arguments: tuple[dict[str, Any], ...] = Field(
        default_factory=tuple,
        description="Prompt arguments",
    )


class MCPServerStatus(str, Enum):
    """Status of an MCP server connection."""
    
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class MCPServerState:
    """State of an MCP server connection."""
    
    config: MCPServerConfig
    status: MCPServerStatus = MCPServerStatus.DISCONNECTED
    
    # Discovered capabilities
    tools: list[ToolDefinition] = field(default_factory=list)
    resources: list[MCPResource] = field(default_factory=list)
    prompts: list[MCPPrompt] = field(default_factory=list)
    
    # Connection info
    connected_at: datetime | None = None
    last_error: str | None = None
    
    # Server info
    server_name: str | None = None
    server_version: str | None = None
    protocol_version: str | None = None


class MCPTransport(ABC):
    """Abstract base class for MCP transport implementations."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the server."""
        ...
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close the connection."""
        ...
    
    @abstractmethod
    async def send(self, message: dict[str, Any]) -> None:
        """Send a message to the server."""
        ...
    
    @abstractmethod
    async def receive(self) -> dict[str, Any]:
        """Receive a message from the server."""
        ...
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected."""
        ...


class StdioTransport(MCPTransport):
    """
    STDIO transport for MCP servers.
    
    Communicates with the server via stdin/stdout.
    """
    
    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._connected = False
    
    async def connect(self) -> None:
        """Start the server process."""
        if self.config.command is None:
            raise ValueError("Command is required for STDIO transport")
        
        # Prepare environment
        import os
        env = {**os.environ, **self.config.env}
        
        # Start process
        self._process = await asyncio.create_subprocess_exec(
            self.config.command,
            *self.config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        
        self._connected = True
    
    async def disconnect(self) -> None:
        """Terminate the server process."""
        if self._process is not None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
            self._process = None
        
        self._connected = False
    
    async def send(self, message: dict[str, Any]) -> None:
        """Send a JSON-RPC message to the server."""
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Not connected")
        
        import json
        data = json.dumps(message) + "\n"
        self._process.stdin.write(data.encode())
        await self._process.stdin.drain()
    
    async def receive(self) -> dict[str, Any]:
        """Receive a JSON-RPC message from the server."""
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("Not connected")
        
        import json
        line = await self._process.stdout.readline()
        if not line:
            raise RuntimeError("Server closed connection")
        
        return json.loads(line.decode())
    
    @property
    def is_connected(self) -> bool:
        return self._connected and self._process is not None


class MCPClient:
    """
    Client for communicating with an MCP server.
    
    Handles the JSON-RPC protocol and message routing.
    """
    
    def __init__(self, transport: MCPTransport) -> None:
        self.transport = transport
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._receive_task: asyncio.Task[None] | None = None
    
    async def connect(self) -> None:
        """Connect to the server."""
        await self.transport.connect()
        
        # Start receive loop
        self._receive_task = asyncio.create_task(self._receive_loop())
    
    async def disconnect(self) -> None:
        """Disconnect from the server."""
        if self._receive_task is not None:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
        
        await self.transport.disconnect()
    
    async def _receive_loop(self) -> None:
        """Background task to receive messages."""
        while self.transport.is_connected:
            try:
                message = await self.transport.receive()
                await self._handle_message(message)
            except asyncio.CancelledError:
                break
            except Exception:
                break
    
    async def _handle_message(self, message: dict[str, Any]) -> None:
        """Handle an incoming message."""
        # Check if it's a response to a request
        if "id" in message and message["id"] in self._pending_requests:
            future = self._pending_requests.pop(message["id"])
            if "error" in message:
                future.set_exception(Exception(message["error"].get("message", "Unknown error")))
            else:
                future.set_result(message.get("result", {}))
    
    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """
        Send a request and wait for response.
        
        Args:
            method: JSON-RPC method name
            params: Method parameters
            timeout: Request timeout
            
        Returns:
            Response result
        """
        self._request_id += 1
        request_id = self._request_id
        
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        
        # Create future for response
        future: asyncio.Future[dict[str, Any]] = asyncio.Future()
        self._pending_requests[request_id] = future
        
        try:
            await self.transport.send(message)
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise
    
    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """
        Send a notification (no response expected).
        
        Args:
            method: JSON-RPC method name
            params: Method parameters
        """
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }
        
        await self.transport.send(message)


class MCPServerConnection:
    """
    Connection to a single MCP server.
    
    Manages the lifecycle and provides access to server capabilities.
    """
    
    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.state = MCPServerState(config=config)
        
        # Create transport based on type
        if config.server_type == MCPServerType.STDIO:
            transport = StdioTransport(config)
        else:
            raise NotImplementedError(f"Transport type {config.server_type} not implemented")
        
        self._client = MCPClient(transport)
    
    @property
    def status(self) -> MCPServerStatus:
        return self.state.status
    
    @property
    def is_connected(self) -> bool:
        return self.state.status == MCPServerStatus.CONNECTED
    
    async def connect(self) -> None:
        """Connect to the server and discover capabilities."""
        self.state.status = MCPServerStatus.CONNECTING
        
        try:
            await self._client.connect()
            
            # Initialize connection
            init_result = await self._client.request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {},
                },
                "clientInfo": {
                    "name": "agent-runtime",
                    "version": "0.1.0",
                },
            })
            
            # Store server info
            self.state.server_name = init_result.get("serverInfo", {}).get("name")
            self.state.server_version = init_result.get("serverInfo", {}).get("version")
            self.state.protocol_version = init_result.get("protocolVersion")
            
            # Send initialized notification
            await self._client.notify("notifications/initialized")
            
            # Discover capabilities
            await self._discover_tools()
            await self._discover_resources()
            await self._discover_prompts()
            
            self.state.status = MCPServerStatus.CONNECTED
            self.state.connected_at = datetime.now(timezone.utc)
            
        except Exception as e:
            self.state.status = MCPServerStatus.ERROR
            self.state.last_error = str(e)
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from the server."""
        await self._client.disconnect()
        self.state.status = MCPServerStatus.DISCONNECTED
    
    async def _discover_tools(self) -> None:
        """Discover available tools from the server."""
        try:
            result = await self._client.request("tools/list")
            tools = result.get("tools", [])
            
            self.state.tools = [
                self._parse_tool_definition(t) for t in tools
            ]
        except Exception:
            self.state.tools = []
    
    async def _discover_resources(self) -> None:
        """Discover available resources from the server."""
        try:
            result = await self._client.request("resources/list")
            resources = result.get("resources", [])
            
            self.state.resources = [
                MCPResource(
                    uri=r.get("uri", ""),
                    name=r.get("name", ""),
                    description=r.get("description", ""),
                    mime_type=r.get("mimeType", "text/plain"),
                )
                for r in resources
            ]
        except Exception:
            self.state.resources = []
    
    async def _discover_prompts(self) -> None:
        """Discover available prompts from the server."""
        try:
            result = await self._client.request("prompts/list")
            prompts = result.get("prompts", [])
            
            self.state.prompts = [
                MCPPrompt(
                    name=p.get("name", ""),
                    description=p.get("description", ""),
                    arguments=tuple(p.get("arguments", [])),
                )
                for p in prompts
            ]
        except Exception:
            self.state.prompts = []
    
    def _parse_tool_definition(self, tool_data: dict[str, Any]) -> ToolDefinition:
        """Parse a tool definition from MCP format."""
        name = tool_data.get("name", "")
        description = tool_data.get("description", "")
        
        # Parse input schema
        input_schema = tool_data.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        required = set(input_schema.get("required", []))
        
        parameters: list[ToolParameter] = []
        for param_name, param_schema in properties.items():
            param_type = self._map_json_type(param_schema.get("type", "string"))
            
            parameters.append(ToolParameter(
                name=param_name,
                type=param_type,
                description=param_schema.get("description", ""),
                required=param_name in required,
                enum=param_schema.get("enum"),
            ))
        
        return ToolDefinition(
            name=f"{self.config.name}_{name}",  # Prefix with server name
            description=description,
            parameters=tuple(parameters),
            category="mcp",
            tags=(self.config.name,),
        )
    
    def _map_json_type(self, json_type: str) -> ParameterType:
        """Map JSON Schema type to ParameterType."""
        mapping = {
            "string": ParameterType.STRING,
            "integer": ParameterType.INTEGER,
            "number": ParameterType.NUMBER,
            "boolean": ParameterType.BOOLEAN,
            "array": ParameterType.ARRAY,
            "object": ParameterType.OBJECT,
        }
        return mapping.get(json_type, ParameterType.STRING)
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """
        Call a tool on the server.
        
        Args:
            tool_name: Name of the tool (without server prefix)
            arguments: Tool arguments
            
        Returns:
            Tool result
        """
        result = await self._client.request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        
        # Extract content from result
        content = result.get("content", [])
        if content:
            # Return first text content
            for item in content:
                if item.get("type") == "text":
                    return item.get("text", "")
            # Return first item if no text
            return content[0]
        
        return result
    
    async def read_resource(self, uri: str) -> str:
        """
        Read a resource from the server.
        
        Args:
            uri: Resource URI
            
        Returns:
            Resource content
        """
        result = await self._client.request("resources/read", {
            "uri": uri,
        })
        
        contents = result.get("contents", [])
        if contents:
            return contents[0].get("text", "")
        
        return ""
    
    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get a prompt from the server.
        
        Args:
            name: Prompt name
            arguments: Prompt arguments
            
        Returns:
            List of messages
        """
        result = await self._client.request("prompts/get", {
            "name": name,
            "arguments": arguments or {},
        })
        
        return result.get("messages", [])


class MCPAdapter:
    """
    Adapter for integrating MCP servers with the tool gateway.
    
    The MCPAdapter:
    - Manages connections to multiple MCP servers
    - Registers MCP tools with the tool registry
    - Routes tool invocations to the appropriate server
    
    Example:
        >>> adapter = MCPAdapter(registry)
        >>> 
        >>> # Add an MCP server
        >>> await adapter.add_server(MCPServerConfig(
        ...     name="filesystem",
        ...     command="npx",
        ...     args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        ... ))
        >>> 
        >>> # Tools are now available in the registry
        >>> tool = registry.get("filesystem_read_file")
    """
    
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self._servers: dict[str, MCPServerConnection] = {}
    
    @property
    def servers(self) -> list[MCPServerConnection]:
        """Get all server connections."""
        return list(self._servers.values())
    
    def get_server(self, name: str) -> MCPServerConnection | None:
        """Get a server connection by name."""
        return self._servers.get(name)
    
    async def add_server(self, config: MCPServerConfig) -> MCPServerConnection:
        """
        Add and connect to an MCP server.
        
        Args:
            config: Server configuration
            
        Returns:
            The server connection
        """
        if config.name in self._servers:
            raise ValueError(f"Server '{config.name}' already exists")
        
        connection = MCPServerConnection(config)
        await connection.connect()
        
        self._servers[config.name] = connection
        
        # Register tools from this server
        self._register_server_tools(connection)
        
        return connection
    
    async def remove_server(self, name: str) -> bool:
        """
        Remove and disconnect from an MCP server.
        
        Args:
            name: Server name
            
        Returns:
            True if server was found and removed
        """
        connection = self._servers.pop(name, None)
        if connection is None:
            return False
        
        # Unregister tools
        self._unregister_server_tools(connection)
        
        # Disconnect
        await connection.disconnect()
        
        return True
    
    def _register_server_tools(self, connection: MCPServerConnection) -> None:
        """Register tools from a server with the registry."""
        for tool_def in connection.state.tools:
            # Create handler that routes to the server
            handler = self._create_tool_handler(connection, tool_def)
            
            tool = Tool(definition=tool_def)
            tool.set_handler(handler)
            
            try:
                self.registry.register_tool(tool)
            except ValueError:
                # Tool already exists - skip
                pass
    
    def _unregister_server_tools(self, connection: MCPServerConnection) -> None:
        """Unregister tools from a server."""
        for tool_def in connection.state.tools:
            self.registry.unregister_tool(tool_def.name)
    
    def _create_tool_handler(
        self,
        connection: MCPServerConnection,
        tool_def: ToolDefinition,
    ) -> ToolHandler:
        """Create a handler that routes to the MCP server."""
        # Extract original tool name (without server prefix)
        prefix = f"{connection.config.name}_"
        original_name = tool_def.name
        if original_name.startswith(prefix):
            original_name = original_name[len(prefix):]
        
        async def handler(arguments: dict[str, Any]) -> Any:
            # Remove auth token if present (MCP handles its own auth)
            args = {k: v for k, v in arguments.items() if not k.startswith("_")}
            return await connection.call_tool(original_name, args)
        
        return handler
    
    async def refresh_server(self, name: str) -> bool:
        """
        Refresh a server's capabilities.
        
        Args:
            name: Server name
            
        Returns:
            True if server was found and refreshed
        """
        connection = self._servers.get(name)
        if connection is None:
            return False
        
        # Unregister old tools
        self._unregister_server_tools(connection)
        
        # Reconnect to refresh capabilities
        await connection.disconnect()
        await connection.connect()
        
        # Register new tools
        self._register_server_tools(connection)
        
        return True
    
    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for connection in self._servers.values():
            await connection.disconnect()
        self._servers.clear()
    
    def get_all_resources(self) -> list[tuple[str, MCPResource]]:
        """Get all resources from all servers."""
        resources: list[tuple[str, MCPResource]] = []
        for name, connection in self._servers.items():
            for resource in connection.state.resources:
                resources.append((name, resource))
        return resources
    
    def get_all_prompts(self) -> list[tuple[str, MCPPrompt]]:
        """Get all prompts from all servers."""
        prompts: list[tuple[str, MCPPrompt]] = []
        for name, connection in self._servers.items():
            for prompt in connection.state.prompts:
                prompts.append((name, prompt))
        return prompts
    
    async def read_resource(self, server_name: str, uri: str) -> str:
        """
        Read a resource from a server.
        
        Args:
            server_name: Name of the server
            uri: Resource URI
            
        Returns:
            Resource content
        """
        connection = self._servers.get(server_name)
        if connection is None:
            raise ValueError(f"Server '{server_name}' not found")
        
        return await connection.read_resource(uri)
    
    async def get_prompt(
        self,
        server_name: str,
        prompt_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get a prompt from a server.
        
        Args:
            server_name: Name of the server
            prompt_name: Prompt name
            arguments: Prompt arguments
            
        Returns:
            List of messages
        """
        connection = self._servers.get(server_name)
        if connection is None:
            raise ValueError(f"Server '{server_name}' not found")
        
        return await connection.get_prompt(prompt_name, arguments)
