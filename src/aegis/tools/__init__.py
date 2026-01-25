"""
Tools module for the agent runtime.

This module provides:
- Tool definitions and registry
- Tool gateway with policy enforcement
- MCP (Model Context Protocol) adapter
- Authentication delegation
- Built-in tools
"""

from aegis.tools.models import (
    Tool,
    ToolParameter,
    ToolResult,
    ToolError,
    ToolDefinition,
)
from aegis.tools.gateway import ToolGateway, ToolGatewayConfig
from aegis.tools.policy import (
    Policy,
    PolicyRule,
    PolicyEngine,
    PolicyDecision,
    PolicyViolation,
)
from aegis.tools.registry import ToolRegistry

__all__ = [
    # Models
    "Tool",
    "ToolParameter",
    "ToolResult",
    "ToolError",
    "ToolDefinition",
    # Gateway
    "ToolGateway",
    "ToolGatewayConfig",
    # Policy
    "Policy",
    "PolicyRule",
    "PolicyEngine",
    "PolicyDecision",
    "PolicyViolation",
    # Registry
    "ToolRegistry",
]
