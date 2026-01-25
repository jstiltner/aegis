"""
Tool registry for managing available tools.

This module provides:
- Tool registration and discovery
- Tool lookup by name or category
- Tool validation
"""

from __future__ import annotations

from typing import Any, Callable, Awaitable
from functools import wraps

from aegis.tools.models import (
    Tool,
    ToolDefinition,
    ToolParameter,
    ParameterType,
    ToolHandler,
)


class ToolRegistry:
    """
    Registry for managing available tools.
    
    The ToolRegistry provides:
    - Tool registration
    - Tool lookup by name
    - Tool filtering by category/tags
    - Decorator for easy tool creation
    
    Example:
        >>> registry = ToolRegistry()
        >>> 
        >>> # Register a tool
        >>> @registry.register(
        ...     name="search",
        ...     description="Search the web",
        ...     parameters=[
        ...         ToolParameter(name="query", type=ParameterType.STRING, required=True),
        ...     ],
        ... )
        ... async def search(args: dict) -> str:
        ...     return f"Results for: {args['query']}"
        >>> 
        >>> # Get the tool
        >>> tool = registry.get("search")
        >>> result = await tool.invoke({"query": "test"})
    """
    
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
    
    @property
    def tools(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    @property
    def tool_names(self) -> list[str]:
        """Get names of all registered tools."""
        return list(self._tools.keys())
    
    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool.
        
        Args:
            tool: The tool to register
            
        Raises:
            ValueError: If a tool with the same name already exists
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
    
    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            name: Name of the tool to unregister
            
        Returns:
            True if the tool was found and removed
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
    def get(self, name: str) -> Tool | None:
        """
        Get a tool by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            The tool, or None if not found
        """
        return self._tools.get(name)
    
    def get_definition(self, name: str) -> ToolDefinition | None:
        """
        Get a tool definition by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            The tool definition, or None if not found
        """
        tool = self.get(name)
        return tool.definition if tool else None
    
    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def get_by_category(self, category: str) -> list[Tool]:
        """
        Get tools by category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of tools in the category
        """
        return [
            tool for tool in self._tools.values()
            if tool.definition.category == category
        ]
    
    def get_by_tag(self, tag: str) -> list[Tool]:
        """
        Get tools by tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of tools with the tag
        """
        return [
            tool for tool in self._tools.values()
            if tag in tool.definition.tags
        ]
    
    def get_categories(self) -> list[str]:
        """Get all unique categories."""
        return list(set(
            tool.definition.category
            for tool in self._tools.values()
        ))
    
    def get_tags(self) -> list[str]:
        """Get all unique tags."""
        tags: set[str] = set()
        for tool in self._tools.values():
            tags.update(tool.definition.tags)
        return list(tags)
    
    def get_definitions(self) -> list[ToolDefinition]:
        """Get all tool definitions."""
        return [tool.definition for tool in self._tools.values()]
    
    def get_openai_functions(self) -> list[dict[str, Any]]:
        """Get all tools in OpenAI function calling format."""
        return [
            tool.definition.to_openai_function()
            for tool in self._tools.values()
        ]
    
    def get_anthropic_tools(self) -> list[dict[str, Any]]:
        """Get all tools in Anthropic tool format."""
        return [
            tool.definition.to_anthropic_tool()
            for tool in self._tools.values()
        ]
    
    def register(
        self,
        name: str,
        description: str,
        parameters: list[ToolParameter] | None = None,
        category: str = "general",
        tags: list[str] | None = None,
        timeout_seconds: float = 30.0,
        requires_confirmation: bool = False,
        is_dangerous: bool = False,
    ) -> Callable[[ToolHandler], ToolHandler]:
        """
        Decorator for registering a tool.
        
        Args:
            name: Tool name
            description: Tool description
            parameters: Tool parameters
            category: Tool category
            tags: Tool tags
            timeout_seconds: Default timeout
            requires_confirmation: Whether confirmation is required
            is_dangerous: Whether the tool is dangerous
            
        Returns:
            Decorator function
            
        Example:
            >>> @registry.register(
            ...     name="echo",
            ...     description="Echo the input",
            ...     parameters=[
            ...         ToolParameter(name="message", type=ParameterType.STRING, required=True),
            ...     ],
            ... )
            ... async def echo(args: dict) -> str:
            ...     return args["message"]
        """
        def decorator(handler: ToolHandler) -> ToolHandler:
            definition = ToolDefinition(
                name=name,
                description=description,
                parameters=tuple(parameters or []),
                category=category,
                tags=tuple(tags or []),
                timeout_seconds=timeout_seconds,
                requires_confirmation=requires_confirmation,
                is_dangerous=is_dangerous,
            )
            
            tool = Tool(definition=definition)
            tool.set_handler(handler)
            
            self.register_tool(tool)
            
            return handler
        
        return decorator
    
    def clear(self) -> None:
        """Remove all registered tools."""
        self._tools.clear()
    
    def merge(self, other: ToolRegistry, overwrite: bool = False) -> None:
        """
        Merge another registry into this one.
        
        Args:
            other: Registry to merge from
            overwrite: Whether to overwrite existing tools
        """
        for tool in other.tools:
            if tool.name in self._tools and not overwrite:
                continue
            self._tools[tool.name] = tool


# Global registry instance
_global_registry = ToolRegistry()


def get_global_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _global_registry


def register_tool(
    name: str,
    description: str,
    parameters: list[ToolParameter] | None = None,
    **kwargs: Any,
) -> Callable[[ToolHandler], ToolHandler]:
    """
    Decorator for registering a tool in the global registry.
    
    This is a convenience function that uses the global registry.
    
    Example:
        >>> @register_tool(
        ...     name="greet",
        ...     description="Greet someone",
        ...     parameters=[
        ...         ToolParameter(name="name", type=ParameterType.STRING, required=True),
        ...     ],
        ... )
        ... async def greet(args: dict) -> str:
        ...     return f"Hello, {args['name']}!"
    """
    return _global_registry.register(
        name=name,
        description=description,
        parameters=parameters,
        **kwargs,
    )


def tool(
    name: str | None = None,
    description: str | None = None,
    **kwargs: Any,
) -> Callable[[ToolHandler], ToolHandler]:
    """
    Simple decorator for creating tools.
    
    Uses the function name and docstring if name/description not provided.
    
    Example:
        >>> @tool()
        ... async def my_tool(args: dict) -> str:
        ...     '''This is my tool.'''
        ...     return "result"
    """
    def decorator(handler: ToolHandler) -> ToolHandler:
        tool_name = name or handler.__name__
        tool_description = description or handler.__doc__ or f"Tool: {tool_name}"
        
        return _global_registry.register(
            name=tool_name,
            description=tool_description,
            **kwargs,
        )(handler)
    
    return decorator
