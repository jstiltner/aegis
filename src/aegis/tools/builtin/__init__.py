"""
Built-in tools for the agent runtime.

This package provides a collection of pre-built tools for common operations:
- Filesystem operations (read, write, list, search)
- Web operations (HTTP requests, scraping)
- Code execution (Python, shell)
- System utilities (time, environment, encoding)
"""

from __future__ import annotations

from aegis.tools.registry import ToolRegistry

from .filesystem import register_filesystem_tools
from .web import register_web_tools
from .code import register_code_tools
from .system import register_system_tools


def register_all_builtin_tools(registry: ToolRegistry) -> None:
    """Register all built-in tools with the registry."""
    register_filesystem_tools(registry)
    register_web_tools(registry)
    register_code_tools(registry)
    register_system_tools(registry)


__all__ = [
    "register_all_builtin_tools",
    "register_filesystem_tools",
    "register_web_tools",
    "register_code_tools",
    "register_system_tools",
]
