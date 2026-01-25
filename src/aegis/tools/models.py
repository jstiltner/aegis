"""
Tool models for the agent runtime.

This module defines the core data structures for tools:
- Tool definitions with parameters
- Tool results and errors
- Tool invocation requests
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


class ParameterType(str, Enum):
    """Types of tool parameters."""
    
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ToolParameter(BaseModel):
    """
    Definition of a tool parameter.
    
    Follows JSON Schema conventions for compatibility with
    OpenAI function calling and MCP protocol.
    """
    
    model_config = ConfigDict(frozen=True)
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Parameter name",
    )
    type: ParameterType = Field(
        ...,
        description="Parameter type",
    )
    description: str = Field(
        default="",
        max_length=500,
        description="Parameter description",
    )
    required: bool = Field(
        default=False,
        description="Whether the parameter is required",
    )
    default: Any = Field(
        default=None,
        description="Default value if not provided",
    )
    
    # Type-specific constraints
    enum: list[Any] | None = Field(
        default=None,
        description="Allowed values for string/integer parameters",
    )
    minimum: float | None = Field(
        default=None,
        description="Minimum value for number/integer parameters",
    )
    maximum: float | None = Field(
        default=None,
        description="Maximum value for number/integer parameters",
    )
    min_length: int | None = Field(
        default=None,
        ge=0,
        description="Minimum length for string/array parameters",
    )
    max_length: int | None = Field(
        default=None,
        ge=0,
        description="Maximum length for string/array parameters",
    )
    pattern: str | None = Field(
        default=None,
        description="Regex pattern for string parameters",
    )
    
    # Array-specific
    items: dict[str, Any] | None = Field(
        default=None,
        description="Schema for array items",
    )
    
    # Object-specific
    properties: dict[str, Any] | None = Field(
        default=None,
        description="Schema for object properties",
    )
    
    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: dict[str, Any] = {
            "type": self.type.value,
            "description": self.description,
        }
        
        if self.enum is not None:
            schema["enum"] = self.enum
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if self.min_length is not None:
            schema["minLength"] = self.min_length
        if self.max_length is not None:
            schema["maxLength"] = self.max_length
        if self.pattern is not None:
            schema["pattern"] = self.pattern
        if self.items is not None:
            schema["items"] = self.items
        if self.properties is not None:
            schema["properties"] = self.properties
        if self.default is not None:
            schema["default"] = self.default
            
        return schema
    
    def validate_value(self, value: Any) -> list[str]:
        """
        Validate a value against this parameter's constraints.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []
        
        # Type checking
        type_valid = self._check_type(value)
        if not type_valid:
            errors.append(
                f"Parameter '{self.name}' expected type {self.type.value}, "
                f"got {type(value).__name__}"
            )
            return errors  # Skip other checks if type is wrong
        
        # Enum check
        if self.enum is not None and value not in self.enum:
            errors.append(
                f"Parameter '{self.name}' value must be one of {self.enum}"
            )
        
        # Numeric constraints
        if self.type in (ParameterType.INTEGER, ParameterType.NUMBER):
            if self.minimum is not None and value < self.minimum:
                errors.append(
                    f"Parameter '{self.name}' must be >= {self.minimum}"
                )
            if self.maximum is not None and value > self.maximum:
                errors.append(
                    f"Parameter '{self.name}' must be <= {self.maximum}"
                )
        
        # String constraints
        if self.type == ParameterType.STRING:
            if self.min_length is not None and len(value) < self.min_length:
                errors.append(
                    f"Parameter '{self.name}' must have length >= {self.min_length}"
                )
            if self.max_length is not None and len(value) > self.max_length:
                errors.append(
                    f"Parameter '{self.name}' must have length <= {self.max_length}"
                )
            if self.pattern is not None:
                import re
                if not re.match(self.pattern, value):
                    errors.append(
                        f"Parameter '{self.name}' must match pattern {self.pattern}"
                    )
        
        # Array constraints
        if self.type == ParameterType.ARRAY:
            if self.min_length is not None and len(value) < self.min_length:
                errors.append(
                    f"Parameter '{self.name}' must have length >= {self.min_length}"
                )
            if self.max_length is not None and len(value) > self.max_length:
                errors.append(
                    f"Parameter '{self.name}' must have length <= {self.max_length}"
                )
        
        return errors
    
    def _check_type(self, value: Any) -> bool:
        """Check if value matches the expected type."""
        match self.type:
            case ParameterType.STRING:
                return isinstance(value, str)
            case ParameterType.INTEGER:
                return isinstance(value, int) and not isinstance(value, bool)
            case ParameterType.NUMBER:
                return isinstance(value, (int, float)) and not isinstance(value, bool)
            case ParameterType.BOOLEAN:
                return isinstance(value, bool)
            case ParameterType.ARRAY:
                return isinstance(value, list)
            case ParameterType.OBJECT:
                return isinstance(value, dict)
            case _:
                return True


class ToolDefinition(BaseModel):
    """
    Definition of a tool that can be invoked.
    
    This is the metadata about a tool, not the implementation.
    Compatible with OpenAI function calling and MCP protocol.
    """
    
    model_config = ConfigDict(frozen=True)
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$",
        description="Tool name (must be a valid identifier)",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Description of what the tool does",
    )
    parameters: tuple[ToolParameter, ...] = Field(
        default_factory=tuple,
        description="Tool parameters",
    )
    
    # Metadata
    category: str = Field(
        default="general",
        description="Tool category for organization",
    )
    tags: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Tags for filtering tools",
    )
    
    # Execution hints
    timeout_seconds: float = Field(
        default=30.0,
        ge=0.1,
        description="Default timeout for tool execution",
    )
    requires_confirmation: bool = Field(
        default=False,
        description="Whether tool requires human confirmation",
    )
    is_dangerous: bool = Field(
        default=False,
        description="Whether tool can cause irreversible changes",
    )
    
    @property
    def required_parameters(self) -> list[ToolParameter]:
        """Get required parameters."""
        return [p for p in self.parameters if p.required]
    
    @property
    def optional_parameters(self) -> list[ToolParameter]:
        """Get optional parameters."""
        return [p for p in self.parameters if not p.required]
    
    def get_parameter(self, name: str) -> ToolParameter | None:
        """Get a parameter by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None
    
    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        """
        Validate arguments against parameter definitions.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []
        
        # Check required parameters
        for param in self.required_parameters:
            if param.name not in arguments:
                errors.append(f"Missing required parameter: {param.name}")
        
        # Validate provided arguments
        for name, value in arguments.items():
            param = self.get_parameter(name)
            if param is None:
                errors.append(f"Unknown parameter: {name}")
            else:
                errors.extend(param.validate_value(value))
        
        return errors
    
    def to_openai_function(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
    
    def to_anthropic_tool(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class ToolResultStatus(str, Enum):
    """Status of a tool execution."""
    
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    POLICY_DENIED = "policy_denied"


class ToolError(BaseModel):
    """
    Error information from a tool execution.
    """
    
    model_config = ConfigDict(frozen=True)
    
    error_type: str = Field(
        ...,
        description="Type of error",
    )
    message: str = Field(
        ...,
        description="Error message",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional error details",
    )
    recoverable: bool = Field(
        default=True,
        description="Whether the error is recoverable",
    )
    
    @classmethod
    def from_exception(cls, exc: Exception) -> ToolError:
        """Create a ToolError from an exception."""
        return cls(
            error_type=type(exc).__name__,
            message=str(exc),
            recoverable=not isinstance(exc, (SystemExit, KeyboardInterrupt)),
        )


class ToolResult(BaseModel):
    """
    Result of a tool execution.
    
    Contains the output, status, and metadata about the execution.
    """
    
    model_config = ConfigDict(frozen=True)
    
    result_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this result",
    )
    tool_name: str = Field(
        ...,
        description="Name of the tool that was executed",
    )
    
    # Status
    status: ToolResultStatus = Field(
        ...,
        description="Execution status",
    )
    
    # Output
    output: Any = Field(
        default=None,
        description="Tool output (if successful)",
    )
    output_type: str = Field(
        default="text",
        description="Type of output (text, json, binary, etc.)",
    )
    
    # Error
    error: ToolError | None = Field(
        default=None,
        description="Error information (if failed)",
    )
    
    # Timing
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When execution started",
    )
    completed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When execution completed",
    )
    
    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    
    @property
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ToolResultStatus.SUCCESS
    
    @property
    def is_error(self) -> bool:
        """Check if execution resulted in an error."""
        return self.status == ToolResultStatus.ERROR
    
    @property
    def duration_ms(self) -> float:
        """Get execution duration in milliseconds."""
        delta = self.completed_at - self.started_at
        return delta.total_seconds() * 1000
    
    def to_message_content(self) -> str:
        """Convert result to a string for message content."""
        if self.is_success:
            if isinstance(self.output, str):
                return self.output
            elif self.output is None:
                return "Tool executed successfully (no output)"
            else:
                import json
                return json.dumps(self.output, indent=2, default=str)
        else:
            error_msg = self.error.message if self.error else "Unknown error"
            return f"Error: {error_msg}"
    
    @classmethod
    def success(
        cls,
        tool_name: str,
        output: Any,
        output_type: str = "text",
        **kwargs: Any,
    ) -> ToolResult:
        """Create a successful result."""
        return cls(
            tool_name=tool_name,
            status=ToolResultStatus.SUCCESS,
            output=output,
            output_type=output_type,
            **kwargs,
        )
    
    @classmethod
    def error(
        cls,
        tool_name: str,
        error: ToolError | Exception | str,
        **kwargs: Any,
    ) -> ToolResult:
        """Create an error result."""
        if isinstance(error, str):
            tool_error = ToolError(error_type="Error", message=error)
        elif isinstance(error, Exception):
            tool_error = ToolError.from_exception(error)
        else:
            tool_error = error
        
        return cls(
            tool_name=tool_name,
            status=ToolResultStatus.ERROR,
            error=tool_error,
            **kwargs,
        )
    
    @classmethod
    def timeout(cls, tool_name: str, timeout_seconds: float) -> ToolResult:
        """Create a timeout result."""
        return cls(
            tool_name=tool_name,
            status=ToolResultStatus.TIMEOUT,
            error=ToolError(
                error_type="TimeoutError",
                message=f"Tool execution timed out after {timeout_seconds}s",
                recoverable=True,
            ),
        )
    
    @classmethod
    def policy_denied(
        cls,
        tool_name: str,
        reason: str,
        policy_name: str | None = None,
    ) -> ToolResult:
        """Create a policy denied result."""
        return cls(
            tool_name=tool_name,
            status=ToolResultStatus.POLICY_DENIED,
            error=ToolError(
                error_type="PolicyViolation",
                message=reason,
                details={"policy_name": policy_name} if policy_name else {},
                recoverable=False,
            ),
        )


class ToolInvocation(BaseModel):
    """
    A request to invoke a tool.
    
    Contains all information needed to execute a tool.
    """
    
    model_config = ConfigDict(frozen=True)
    
    invocation_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this invocation",
    )
    tool_name: str = Field(
        ...,
        description="Name of the tool to invoke",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the tool",
    )
    
    # Context
    agent_id: str = Field(
        ...,
        description="ID of the agent making the invocation",
    )
    session_id: str = Field(
        ...,
        description="ID of the current session",
    )
    
    # GCL integration
    commitment_id: str | None = Field(
        default=None,
        description="ID of the commitment associated with this invocation",
    )
    
    # Timing
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the invocation was created",
    )
    timeout_seconds: float | None = Field(
        default=None,
        description="Custom timeout for this invocation",
    )
    
    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


# Type alias for tool handler functions
ToolHandler = Callable[[dict[str, Any]], Awaitable[Any]]


class Tool(BaseModel):
    """
    A complete tool with definition and handler.
    
    This combines the tool metadata with its implementation.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    definition: ToolDefinition = Field(
        ...,
        description="Tool definition with parameters",
    )
    
    # The handler is stored separately since it's not serializable
    _handler: ToolHandler | None = None
    
    @property
    def name(self) -> str:
        """Get the tool name."""
        return self.definition.name
    
    @property
    def description(self) -> str:
        """Get the tool description."""
        return self.definition.description
    
    def set_handler(self, handler: ToolHandler) -> None:
        """Set the tool handler."""
        self._handler = handler
    
    async def invoke(self, arguments: dict[str, Any]) -> Any:
        """
        Invoke the tool with the given arguments.
        
        Args:
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool output
            
        Raises:
            ValueError: If no handler is set
            Exception: Any exception from the handler
        """
        if self._handler is None:
            raise ValueError(f"No handler set for tool '{self.name}'")
        
        return await self._handler(arguments)
    
    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        """Validate arguments against the tool definition."""
        return self.definition.validate_arguments(arguments)
    
    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        handler: ToolHandler,
        parameters: list[ToolParameter] | None = None,
        **kwargs: Any,
    ) -> Tool:
        """
        Create a tool with definition and handler.
        
        Args:
            name: Tool name
            description: Tool description
            handler: Async function to handle invocations
            parameters: Tool parameters
            **kwargs: Additional definition fields
            
        Returns:
            Configured Tool instance
        """
        definition = ToolDefinition(
            name=name,
            description=description,
            parameters=tuple(parameters or []),
            **kwargs,
        )
        
        tool = cls(definition=definition)
        tool.set_handler(handler)
        return tool
