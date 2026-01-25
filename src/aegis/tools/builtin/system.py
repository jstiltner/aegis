"""
System tools for the agent runtime.

Provides tools for:
- Environment information
- Time and date operations
- System utilities
"""

from __future__ import annotations

import os
import platform
import time
from datetime import datetime, timezone
from typing import Any

from aegis.tools.models import ToolParameter, ParameterType, Tool
from aegis.tools.registry import ToolRegistry


async def get_current_time(args: dict[str, Any]) -> dict[str, Any]:
    """Get the current time."""
    tz_name = args.get("timezone")
    format_str = args.get("format", "%Y-%m-%d %H:%M:%S")
    
    now = datetime.now(timezone.utc)
    
    if tz_name:
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(tz_name)
            now = now.astimezone(tz)
        except ImportError:
            # zoneinfo not available, use UTC
            pass
        except Exception:
            # Invalid timezone, use UTC
            pass
    
    return {
        "iso": now.isoformat(),
        "formatted": now.strftime(format_str),
        "timestamp": now.timestamp(),
        "timezone": str(now.tzinfo) if now.tzinfo else "UTC",
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "hour": now.hour,
        "minute": now.minute,
        "second": now.second,
        "weekday": now.strftime("%A"),
    }


async def sleep(args: dict[str, Any]) -> str:
    """Sleep for a specified duration."""
    import asyncio
    
    seconds = args["seconds"]
    max_sleep = args.get("max_sleep", 60.0)
    
    if seconds > max_sleep:
        raise ValueError(f"Sleep duration {seconds}s exceeds maximum {max_sleep}s")
    
    if seconds < 0:
        raise ValueError("Sleep duration cannot be negative")
    
    await asyncio.sleep(seconds)
    return f"Slept for {seconds} seconds"


async def get_environment_variable(args: dict[str, Any]) -> dict[str, Any]:
    """Get an environment variable."""
    name = args["name"]
    default = args.get("default")
    
    value = os.environ.get(name, default)
    
    return {
        "name": name,
        "value": value,
        "exists": name in os.environ,
    }


async def list_environment_variables(args: dict[str, Any]) -> dict[str, str]:
    """List environment variables."""
    prefix = args.get("prefix")
    include_sensitive = args.get("include_sensitive", False)
    
    # Sensitive variable patterns
    sensitive_patterns = [
        "password", "secret", "key", "token", "credential",
        "auth", "api_key", "apikey", "private",
    ]
    
    result: dict[str, str] = {}
    
    for name, value in os.environ.items():
        # Filter by prefix
        if prefix and not name.startswith(prefix):
            continue
        
        # Check if sensitive
        name_lower = name.lower()
        is_sensitive = any(p in name_lower for p in sensitive_patterns)
        
        if is_sensitive and not include_sensitive:
            result[name] = "***REDACTED***"
        else:
            result[name] = value
    
    return result


async def get_system_info(args: dict[str, Any]) -> dict[str, Any]:
    """Get system information."""
    import sys
    
    info: dict[str, Any] = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "path": sys.path[:5],  # First 5 paths
        },
    }
    
    # Add memory info if psutil is available
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        info["memory"] = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "percent_used": memory.percent,
        }
        
        disk = psutil.disk_usage("/")
        info["disk"] = {
            "total_gb": round(disk.total / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "percent_used": disk.percent,
        }
        
        info["cpu"] = {
            "count": psutil.cpu_count(),
            "percent": psutil.cpu_percent(interval=0.1),
        }
    except ImportError:
        pass
    
    return info


async def generate_uuid(args: dict[str, Any]) -> str:
    """Generate a UUID."""
    import uuid
    
    version = args.get("version", 4)
    
    if version == 1:
        return str(uuid.uuid1())
    elif version == 4:
        return str(uuid.uuid4())
    else:
        raise ValueError(f"Unsupported UUID version: {version}")


async def hash_text(args: dict[str, Any]) -> dict[str, str]:
    """Hash text using various algorithms."""
    import hashlib
    
    text = args["text"]
    algorithms = args.get("algorithms", ["sha256"])
    
    text_bytes = text.encode("utf-8")
    
    result: dict[str, str] = {}
    
    for algo in algorithms:
        if algo in hashlib.algorithms_available:
            h = hashlib.new(algo)
            h.update(text_bytes)
            result[algo] = h.hexdigest()
        else:
            result[algo] = f"Algorithm not available: {algo}"
    
    return result


async def encode_base64(args: dict[str, Any]) -> str:
    """Encode text to base64."""
    import base64
    
    text = args["text"]
    url_safe = args.get("url_safe", False)
    
    text_bytes = text.encode("utf-8")
    
    if url_safe:
        return base64.urlsafe_b64encode(text_bytes).decode("ascii")
    else:
        return base64.b64encode(text_bytes).decode("ascii")


async def decode_base64(args: dict[str, Any]) -> str:
    """Decode base64 to text."""
    import base64
    
    encoded = args["encoded"]
    url_safe = args.get("url_safe", False)
    
    try:
        if url_safe:
            decoded_bytes = base64.urlsafe_b64decode(encoded)
        else:
            decoded_bytes = base64.b64decode(encoded)
        
        return decoded_bytes.decode("utf-8")
    except Exception as e:
        raise ValueError(f"Failed to decode base64: {e}")


async def json_encode(args: dict[str, Any]) -> str:
    """Encode data to JSON."""
    import json
    
    data = args["data"]
    pretty = args.get("pretty", False)
    
    if pretty:
        return json.dumps(data, indent=2, sort_keys=True)
    else:
        return json.dumps(data)


async def json_decode(args: dict[str, Any]) -> Any:
    """Decode JSON to data."""
    import json
    
    text = args["text"]
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")


def register_system_tools(registry: ToolRegistry) -> None:
    """Register system tools with the registry."""
    
    # Get current time
    registry.register_tool(Tool.create(
        name="get_current_time",
        description="Get the current time",
        handler=get_current_time,
        parameters=[
            ToolParameter(
                name="timezone",
                type=ParameterType.STRING,
                description="Timezone name (e.g., 'America/New_York')",
                required=False,
            ),
            ToolParameter(
                name="format",
                type=ParameterType.STRING,
                description="strftime format string",
                required=False,
                default="%Y-%m-%d %H:%M:%S",
            ),
        ],
        category="system",
        tags=["time", "date"],
    ))
    
    # Sleep
    registry.register_tool(Tool.create(
        name="sleep",
        description="Sleep for a specified duration",
        handler=sleep,
        parameters=[
            ToolParameter(
                name="seconds",
                type=ParameterType.NUMBER,
                description="Number of seconds to sleep",
                required=True,
            ),
            ToolParameter(
                name="max_sleep",
                type=ParameterType.NUMBER,
                description="Maximum allowed sleep duration",
                required=False,
                default=60.0,
            ),
        ],
        category="system",
        tags=["time", "wait"],
    ))
    
    # Get environment variable
    registry.register_tool(Tool.create(
        name="get_environment_variable",
        description="Get an environment variable",
        handler=get_environment_variable,
        parameters=[
            ToolParameter(
                name="name",
                type=ParameterType.STRING,
                description="Name of the environment variable",
                required=True,
            ),
            ToolParameter(
                name="default",
                type=ParameterType.STRING,
                description="Default value if not found",
                required=False,
            ),
        ],
        category="system",
        tags=["environment", "config"],
    ))
    
    # List environment variables
    registry.register_tool(Tool.create(
        name="list_environment_variables",
        description="List environment variables",
        handler=list_environment_variables,
        parameters=[
            ToolParameter(
                name="prefix",
                type=ParameterType.STRING,
                description="Filter by prefix",
                required=False,
            ),
            ToolParameter(
                name="include_sensitive",
                type=ParameterType.BOOLEAN,
                description="Whether to include sensitive values",
                required=False,
                default=False,
            ),
        ],
        category="system",
        tags=["environment", "config"],
    ))
    
    # Get system info
    registry.register_tool(Tool.create(
        name="get_system_info",
        description="Get system information",
        handler=get_system_info,
        parameters=[],
        category="system",
        tags=["info", "platform"],
    ))
    
    # Generate UUID
    registry.register_tool(Tool.create(
        name="generate_uuid",
        description="Generate a UUID",
        handler=generate_uuid,
        parameters=[
            ToolParameter(
                name="version",
                type=ParameterType.INTEGER,
                description="UUID version (1 or 4)",
                required=False,
                default=4,
                enum=[1, 4],
            ),
        ],
        category="system",
        tags=["uuid", "id"],
    ))
    
    # Hash text
    registry.register_tool(Tool.create(
        name="hash_text",
        description="Hash text using various algorithms",
        handler=hash_text,
        parameters=[
            ToolParameter(
                name="text",
                type=ParameterType.STRING,
                description="Text to hash",
                required=True,
            ),
            ToolParameter(
                name="algorithms",
                type=ParameterType.ARRAY,
                description="Hash algorithms to use",
                required=False,
                default=["sha256"],
            ),
        ],
        category="system",
        tags=["hash", "crypto"],
    ))
    
    # Encode base64
    registry.register_tool(Tool.create(
        name="encode_base64",
        description="Encode text to base64",
        handler=encode_base64,
        parameters=[
            ToolParameter(
                name="text",
                type=ParameterType.STRING,
                description="Text to encode",
                required=True,
            ),
            ToolParameter(
                name="url_safe",
                type=ParameterType.BOOLEAN,
                description="Use URL-safe encoding",
                required=False,
                default=False,
            ),
        ],
        category="system",
        tags=["encode", "base64"],
    ))
    
    # Decode base64
    registry.register_tool(Tool.create(
        name="decode_base64",
        description="Decode base64 to text",
        handler=decode_base64,
        parameters=[
            ToolParameter(
                name="encoded",
                type=ParameterType.STRING,
                description="Base64 encoded string",
                required=True,
            ),
            ToolParameter(
                name="url_safe",
                type=ParameterType.BOOLEAN,
                description="Use URL-safe decoding",
                required=False,
                default=False,
            ),
        ],
        category="system",
        tags=["decode", "base64"],
    ))
    
    # JSON encode
    registry.register_tool(Tool.create(
        name="json_encode",
        description="Encode data to JSON",
        handler=json_encode,
        parameters=[
            ToolParameter(
                name="data",
                type=ParameterType.OBJECT,
                description="Data to encode",
                required=True,
            ),
            ToolParameter(
                name="pretty",
                type=ParameterType.BOOLEAN,
                description="Pretty print output",
                required=False,
                default=False,
            ),
        ],
        category="system",
        tags=["json", "encode"],
    ))
    
    # JSON decode
    registry.register_tool(Tool.create(
        name="json_decode",
        description="Decode JSON to data",
        handler=json_decode,
        parameters=[
            ToolParameter(
                name="text",
                type=ParameterType.STRING,
                description="JSON string to decode",
                required=True,
            ),
        ],
        category="system",
        tags=["json", "decode"],
    ))
