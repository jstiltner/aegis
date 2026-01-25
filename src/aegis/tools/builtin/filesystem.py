"""
Filesystem tools for the agent runtime.

Provides tools for:
- Reading files
- Writing files
- Listing directories
- File operations (copy, move, delete)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from aegis.tools.models import ToolParameter, ParameterType, Tool, ToolDefinition
from aegis.tools.registry import ToolRegistry


async def read_file(args: dict[str, Any]) -> str:
    """Read the contents of a file."""
    path = Path(args["path"])
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    
    # Check file size
    max_size = args.get("max_size", 1024 * 1024)  # 1MB default
    if path.stat().st_size > max_size:
        raise ValueError(f"File too large: {path.stat().st_size} > {max_size}")
    
    encoding = args.get("encoding", "utf-8")
    return path.read_text(encoding=encoding)


async def write_file(args: dict[str, Any]) -> str:
    """Write content to a file."""
    path = Path(args["path"])
    content = args["content"]
    
    # Create parent directories if needed
    if args.get("create_dirs", True):
        path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists and overwrite is disabled
    if path.exists() and not args.get("overwrite", True):
        raise FileExistsError(f"File already exists: {path}")
    
    encoding = args.get("encoding", "utf-8")
    path.write_text(content, encoding=encoding)
    
    return f"Successfully wrote {len(content)} characters to {path}"


async def append_file(args: dict[str, Any]) -> str:
    """Append content to a file."""
    path = Path(args["path"])
    content = args["content"]
    
    # Create parent directories if needed
    if args.get("create_dirs", True):
        path.parent.mkdir(parents=True, exist_ok=True)
    
    encoding = args.get("encoding", "utf-8")
    
    with open(path, "a", encoding=encoding) as f:
        f.write(content)
    
    return f"Successfully appended {len(content)} characters to {path}"


async def list_directory(args: dict[str, Any]) -> list[dict[str, Any]]:
    """List contents of a directory."""
    path = Path(args["path"])
    
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")
    
    recursive = args.get("recursive", False)
    include_hidden = args.get("include_hidden", False)
    pattern = args.get("pattern", "*")
    
    entries: list[dict[str, Any]] = []
    
    if recursive:
        items = path.rglob(pattern)
    else:
        items = path.glob(pattern)
    
    for item in items:
        # Skip hidden files if not requested
        if not include_hidden and item.name.startswith("."):
            continue
        
        try:
            stat = item.stat()
            entries.append({
                "name": item.name,
                "path": str(item),
                "type": "directory" if item.is_dir() else "file",
                "size": stat.st_size if item.is_file() else None,
                "modified": stat.st_mtime,
            })
        except (PermissionError, OSError):
            # Skip files we can't access
            continue
    
    return entries


async def delete_file(args: dict[str, Any]) -> str:
    """Delete a file or directory."""
    path = Path(args["path"])
    
    if not path.exists():
        if args.get("ignore_missing", False):
            return f"Path does not exist (ignored): {path}"
        raise FileNotFoundError(f"Path not found: {path}")
    
    if path.is_file():
        path.unlink()
        return f"Deleted file: {path}"
    elif path.is_dir():
        if args.get("recursive", False):
            import shutil
            shutil.rmtree(path)
            return f"Deleted directory recursively: {path}"
        else:
            path.rmdir()
            return f"Deleted empty directory: {path}"
    else:
        raise ValueError(f"Unknown path type: {path}")


async def copy_file(args: dict[str, Any]) -> str:
    """Copy a file or directory."""
    import shutil
    
    source = Path(args["source"])
    destination = Path(args["destination"])
    
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    
    if destination.exists() and not args.get("overwrite", False):
        raise FileExistsError(f"Destination already exists: {destination}")
    
    # Create parent directories if needed
    if args.get("create_dirs", True):
        destination.parent.mkdir(parents=True, exist_ok=True)
    
    if source.is_file():
        shutil.copy2(source, destination)
        return f"Copied file: {source} -> {destination}"
    elif source.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        return f"Copied directory: {source} -> {destination}"
    else:
        raise ValueError(f"Unknown source type: {source}")


async def move_file(args: dict[str, Any]) -> str:
    """Move a file or directory."""
    import shutil
    
    source = Path(args["source"])
    destination = Path(args["destination"])
    
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    
    if destination.exists() and not args.get("overwrite", False):
        raise FileExistsError(f"Destination already exists: {destination}")
    
    # Create parent directories if needed
    if args.get("create_dirs", True):
        destination.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.move(str(source), str(destination))
    return f"Moved: {source} -> {destination}"


async def file_info(args: dict[str, Any]) -> dict[str, Any]:
    """Get information about a file or directory."""
    path = Path(args["path"])
    
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    
    stat = path.stat()
    
    info = {
        "name": path.name,
        "path": str(path.absolute()),
        "type": "directory" if path.is_dir() else "file",
        "size": stat.st_size,
        "created": stat.st_ctime,
        "modified": stat.st_mtime,
        "accessed": stat.st_atime,
        "permissions": oct(stat.st_mode)[-3:],
    }
    
    if path.is_file():
        # Try to detect file type
        suffix = path.suffix.lower()
        info["extension"] = suffix
        
        # Check if text file
        try:
            with open(path, "r", encoding="utf-8") as f:
                f.read(1024)
            info["is_text"] = True
        except (UnicodeDecodeError, PermissionError):
            info["is_text"] = False
    
    return info


async def create_directory(args: dict[str, Any]) -> str:
    """Create a directory."""
    path = Path(args["path"])
    
    if path.exists():
        if args.get("exist_ok", True):
            return f"Directory already exists: {path}"
        raise FileExistsError(f"Path already exists: {path}")
    
    parents = args.get("parents", True)
    path.mkdir(parents=parents, exist_ok=True)
    
    return f"Created directory: {path}"


async def search_files(args: dict[str, Any]) -> list[dict[str, Any]]:
    """Search for files matching criteria."""
    path = Path(args["path"])
    
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    
    pattern = args.get("pattern", "*")
    content_pattern = args.get("content_pattern")
    max_results = args.get("max_results", 100)
    
    results: list[dict[str, Any]] = []
    
    for item in path.rglob(pattern):
        if len(results) >= max_results:
            break
        
        if not item.is_file():
            continue
        
        # Check content pattern if specified
        if content_pattern:
            try:
                content = item.read_text(encoding="utf-8")
                import re
                if not re.search(content_pattern, content):
                    continue
            except (UnicodeDecodeError, PermissionError):
                continue
        
        try:
            stat = item.stat()
            results.append({
                "name": item.name,
                "path": str(item),
                "size": stat.st_size,
                "modified": stat.st_mtime,
            })
        except (PermissionError, OSError):
            continue
    
    return results


def register_filesystem_tools(registry: ToolRegistry) -> None:
    """Register filesystem tools with the registry."""
    
    # Read file
    registry.register_tool(Tool.create(
        name="read_file",
        description="Read the contents of a file",
        handler=read_file,
        parameters=[
            ToolParameter(
                name="path",
                type=ParameterType.STRING,
                description="Path to the file to read",
                required=True,
            ),
            ToolParameter(
                name="encoding",
                type=ParameterType.STRING,
                description="File encoding (default: utf-8)",
                required=False,
                default="utf-8",
            ),
            ToolParameter(
                name="max_size",
                type=ParameterType.INTEGER,
                description="Maximum file size in bytes (default: 1MB)",
                required=False,
                default=1024 * 1024,
            ),
        ],
        category="filesystem",
        tags=["read", "file"],
    ))
    
    # Write file
    registry.register_tool(Tool.create(
        name="write_file",
        description="Write content to a file",
        handler=write_file,
        parameters=[
            ToolParameter(
                name="path",
                type=ParameterType.STRING,
                description="Path to the file to write",
                required=True,
            ),
            ToolParameter(
                name="content",
                type=ParameterType.STRING,
                description="Content to write",
                required=True,
            ),
            ToolParameter(
                name="encoding",
                type=ParameterType.STRING,
                description="File encoding (default: utf-8)",
                required=False,
                default="utf-8",
            ),
            ToolParameter(
                name="overwrite",
                type=ParameterType.BOOLEAN,
                description="Whether to overwrite existing file",
                required=False,
                default=True,
            ),
            ToolParameter(
                name="create_dirs",
                type=ParameterType.BOOLEAN,
                description="Whether to create parent directories",
                required=False,
                default=True,
            ),
        ],
        category="filesystem",
        tags=["write", "file"],
        is_dangerous=True,
    ))
    
    # Append file
    registry.register_tool(Tool.create(
        name="append_file",
        description="Append content to a file",
        handler=append_file,
        parameters=[
            ToolParameter(
                name="path",
                type=ParameterType.STRING,
                description="Path to the file",
                required=True,
            ),
            ToolParameter(
                name="content",
                type=ParameterType.STRING,
                description="Content to append",
                required=True,
            ),
            ToolParameter(
                name="encoding",
                type=ParameterType.STRING,
                description="File encoding (default: utf-8)",
                required=False,
                default="utf-8",
            ),
        ],
        category="filesystem",
        tags=["write", "file", "append"],
        is_dangerous=True,
    ))
    
    # List directory
    registry.register_tool(Tool.create(
        name="list_directory",
        description="List contents of a directory",
        handler=list_directory,
        parameters=[
            ToolParameter(
                name="path",
                type=ParameterType.STRING,
                description="Path to the directory",
                required=True,
            ),
            ToolParameter(
                name="recursive",
                type=ParameterType.BOOLEAN,
                description="Whether to list recursively",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="include_hidden",
                type=ParameterType.BOOLEAN,
                description="Whether to include hidden files",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="pattern",
                type=ParameterType.STRING,
                description="Glob pattern to filter files",
                required=False,
                default="*",
            ),
        ],
        category="filesystem",
        tags=["read", "directory", "list"],
    ))
    
    # Delete file
    registry.register_tool(Tool.create(
        name="delete_file",
        description="Delete a file or directory",
        handler=delete_file,
        parameters=[
            ToolParameter(
                name="path",
                type=ParameterType.STRING,
                description="Path to delete",
                required=True,
            ),
            ToolParameter(
                name="recursive",
                type=ParameterType.BOOLEAN,
                description="Whether to delete directories recursively",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="ignore_missing",
                type=ParameterType.BOOLEAN,
                description="Whether to ignore missing paths",
                required=False,
                default=False,
            ),
        ],
        category="filesystem",
        tags=["delete", "file", "directory"],
        is_dangerous=True,
        requires_confirmation=True,
    ))
    
    # Copy file
    registry.register_tool(Tool.create(
        name="copy_file",
        description="Copy a file or directory",
        handler=copy_file,
        parameters=[
            ToolParameter(
                name="source",
                type=ParameterType.STRING,
                description="Source path",
                required=True,
            ),
            ToolParameter(
                name="destination",
                type=ParameterType.STRING,
                description="Destination path",
                required=True,
            ),
            ToolParameter(
                name="overwrite",
                type=ParameterType.BOOLEAN,
                description="Whether to overwrite existing destination",
                required=False,
                default=False,
            ),
        ],
        category="filesystem",
        tags=["copy", "file", "directory"],
        is_dangerous=True,
    ))
    
    # Move file
    registry.register_tool(Tool.create(
        name="move_file",
        description="Move a file or directory",
        handler=move_file,
        parameters=[
            ToolParameter(
                name="source",
                type=ParameterType.STRING,
                description="Source path",
                required=True,
            ),
            ToolParameter(
                name="destination",
                type=ParameterType.STRING,
                description="Destination path",
                required=True,
            ),
            ToolParameter(
                name="overwrite",
                type=ParameterType.BOOLEAN,
                description="Whether to overwrite existing destination",
                required=False,
                default=False,
            ),
        ],
        category="filesystem",
        tags=["move", "file", "directory"],
        is_dangerous=True,
    ))
    
    # File info
    registry.register_tool(Tool.create(
        name="file_info",
        description="Get information about a file or directory",
        handler=file_info,
        parameters=[
            ToolParameter(
                name="path",
                type=ParameterType.STRING,
                description="Path to get info for",
                required=True,
            ),
        ],
        category="filesystem",
        tags=["read", "file", "info"],
    ))
    
    # Create directory
    registry.register_tool(Tool.create(
        name="create_directory",
        description="Create a directory",
        handler=create_directory,
        parameters=[
            ToolParameter(
                name="path",
                type=ParameterType.STRING,
                description="Path of directory to create",
                required=True,
            ),
            ToolParameter(
                name="parents",
                type=ParameterType.BOOLEAN,
                description="Whether to create parent directories",
                required=False,
                default=True,
            ),
            ToolParameter(
                name="exist_ok",
                type=ParameterType.BOOLEAN,
                description="Whether to ignore if directory exists",
                required=False,
                default=True,
            ),
        ],
        category="filesystem",
        tags=["create", "directory"],
    ))
    
    # Search files
    registry.register_tool(Tool.create(
        name="search_files",
        description="Search for files matching criteria",
        handler=search_files,
        parameters=[
            ToolParameter(
                name="path",
                type=ParameterType.STRING,
                description="Directory to search in",
                required=True,
            ),
            ToolParameter(
                name="pattern",
                type=ParameterType.STRING,
                description="Glob pattern for file names",
                required=False,
                default="*",
            ),
            ToolParameter(
                name="content_pattern",
                type=ParameterType.STRING,
                description="Regex pattern to search in file contents",
                required=False,
            ),
            ToolParameter(
                name="max_results",
                type=ParameterType.INTEGER,
                description="Maximum number of results",
                required=False,
                default=100,
            ),
        ],
        category="filesystem",
        tags=["search", "file"],
    ))
