"""
Code execution tools for the agent runtime.

Provides tools for:
- Python code execution
- Shell command execution
- Code analysis
"""

from __future__ import annotations

import ast
import asyncio
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any

from aegis.tools.models import ToolParameter, ParameterType, Tool
from aegis.tools.registry import ToolRegistry


async def execute_python(args: dict[str, Any]) -> dict[str, Any]:
    """Execute Python code in a sandboxed environment."""
    code = args["code"]
    timeout = args.get("timeout", 30.0)
    capture_output = args.get("capture_output", True)
    allowed_imports = args.get("allowed_imports", [
        "math", "json", "datetime", "re", "collections",
        "itertools", "functools", "operator", "string",
        "random", "statistics", "decimal", "fractions",
    ])
    
    # Parse and validate code
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "success": False,
            "error": f"Syntax error: {e}",
            "error_type": "SyntaxError",
        }
    
    # Check for disallowed imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                if module not in allowed_imports:
                    return {
                        "success": False,
                        "error": f"Import not allowed: {module}",
                        "error_type": "ImportError",
                    }
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module.split(".")[0]
                if module not in allowed_imports:
                    return {
                        "success": False,
                        "error": f"Import not allowed: {module}",
                        "error_type": "ImportError",
                    }
    
    # Create restricted globals
    restricted_globals = {
        "__builtins__": {
            # Safe builtins
            "abs": abs, "all": all, "any": any, "ascii": ascii,
            "bin": bin, "bool": bool, "bytearray": bytearray,
            "bytes": bytes, "callable": callable, "chr": chr,
            "complex": complex, "dict": dict, "divmod": divmod,
            "enumerate": enumerate, "filter": filter, "float": float,
            "format": format, "frozenset": frozenset, "getattr": getattr,
            "hasattr": hasattr, "hash": hash, "hex": hex, "id": id,
            "int": int, "isinstance": isinstance, "issubclass": issubclass,
            "iter": iter, "len": len, "list": list, "map": map,
            "max": max, "min": min, "next": next, "object": object,
            "oct": oct, "ord": ord, "pow": pow, "print": print,
            "range": range, "repr": repr, "reversed": reversed,
            "round": round, "set": set, "slice": slice, "sorted": sorted,
            "str": str, "sum": sum, "tuple": tuple, "type": type,
            "zip": zip,
            # Exceptions
            "Exception": Exception, "ValueError": ValueError,
            "TypeError": TypeError, "KeyError": KeyError,
            "IndexError": IndexError, "AttributeError": AttributeError,
            "RuntimeError": RuntimeError, "StopIteration": StopIteration,
        },
        "__name__": "__main__",
    }
    
    # Import allowed modules
    for module_name in allowed_imports:
        try:
            restricted_globals[module_name] = __import__(module_name)
        except ImportError:
            pass
    
    # Capture output
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    result: dict[str, Any] = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "result": None,
    }
    
    async def run_code():
        nonlocal result
        try:
            if capture_output:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(compile(tree, "<code>", "exec"), restricted_globals)
            else:
                exec(compile(tree, "<code>", "exec"), restricted_globals)
            
            result["success"] = True
            
            # Try to get the last expression result
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                # Re-execute just the last expression to get its value
                last_expr = ast.Expression(body=tree.body[-1].value)
                result["result"] = eval(
                    compile(last_expr, "<code>", "eval"),
                    restricted_globals
                )
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            result["error_type"] = type(e).__name__
            result["traceback"] = traceback.format_exc()
    
    try:
        await asyncio.wait_for(run_code(), timeout=timeout)
    except asyncio.TimeoutError:
        result["success"] = False
        result["error"] = f"Execution timed out after {timeout} seconds"
        result["error_type"] = "TimeoutError"
    
    result["stdout"] = stdout_capture.getvalue()
    result["stderr"] = stderr_capture.getvalue()
    
    return result


async def execute_shell(args: dict[str, Any]) -> dict[str, Any]:
    """Execute a shell command."""
    command = args["command"]
    timeout = args.get("timeout", 30.0)
    cwd = args.get("cwd")
    env = args.get("env")
    shell = args.get("shell", True)
    
    try:
        if shell:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
        else:
            # Split command for non-shell execution
            import shlex
            cmd_parts = shlex.split(command)
            process = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return {
                "success": False,
                "exit_code": -1,
                "error": f"Command timed out after {timeout} seconds",
                "error_type": "TimeoutError",
            }
        
        return {
            "success": process.returncode == 0,
            "exit_code": process.returncode,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
        }
        
    except Exception as e:
        return {
            "success": False,
            "exit_code": -1,
            "error": str(e),
            "error_type": type(e).__name__,
        }


async def analyze_python_code(args: dict[str, Any]) -> dict[str, Any]:
    """Analyze Python code for structure and potential issues."""
    code = args["code"]
    
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "valid": False,
            "error": f"Syntax error at line {e.lineno}: {e.msg}",
        }
    
    # Collect information about the code
    functions: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    imports: list[str] = []
    global_vars: list[str] = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append({
                "name": node.name,
                "args": [arg.arg for arg in node.args.args],
                "decorators": [
                    ast.unparse(d) if hasattr(ast, 'unparse') else str(d)
                    for d in node.decorator_list
                ],
                "lineno": node.lineno,
                "is_async": False,
            })
        elif isinstance(node, ast.AsyncFunctionDef):
            functions.append({
                "name": node.name,
                "args": [arg.arg for arg in node.args.args],
                "decorators": [
                    ast.unparse(d) if hasattr(ast, 'unparse') else str(d)
                    for d in node.decorator_list
                ],
                "lineno": node.lineno,
                "is_async": True,
            })
        elif isinstance(node, ast.ClassDef):
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(item.name)
            classes.append({
                "name": node.name,
                "bases": [
                    ast.unparse(b) if hasattr(ast, 'unparse') else str(b)
                    for b in node.bases
                ],
                "methods": methods,
                "lineno": node.lineno,
            })
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    
    # Get top-level assignments
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    global_vars.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                global_vars.append(node.target.id)
    
    return {
        "valid": True,
        "functions": functions,
        "classes": classes,
        "imports": list(set(imports)),
        "global_variables": global_vars,
        "line_count": len(code.splitlines()),
    }


async def format_python_code(args: dict[str, Any]) -> dict[str, Any]:
    """Format Python code using standard formatting."""
    code = args["code"]
    line_length = args.get("line_length", 88)
    
    # First validate the code
    try:
        ast.parse(code)
    except SyntaxError as e:
        return {
            "success": False,
            "error": f"Syntax error: {e}",
        }
    
    # Try to use black if available
    try:
        import black
        
        mode = black.Mode(
            line_length=line_length,
            string_normalization=True,
        )
        formatted = black.format_str(code, mode=mode)
        return {
            "success": True,
            "formatted_code": formatted,
            "formatter": "black",
        }
    except ImportError:
        pass
    except Exception as e:
        return {
            "success": False,
            "error": f"Formatting error: {e}",
        }
    
    # Fallback: just return the original code
    return {
        "success": True,
        "formatted_code": code,
        "formatter": "none",
        "note": "No formatter available, code returned unchanged",
    }


def register_code_tools(registry: ToolRegistry) -> None:
    """Register code execution tools with the registry."""
    
    # Execute Python
    registry.register_tool(Tool.create(
        name="execute_python",
        description="Execute Python code in a sandboxed environment",
        handler=execute_python,
        parameters=[
            ToolParameter(
                name="code",
                type=ParameterType.STRING,
                description="Python code to execute",
                required=True,
            ),
            ToolParameter(
                name="timeout",
                type=ParameterType.NUMBER,
                description="Execution timeout in seconds",
                required=False,
                default=30.0,
            ),
            ToolParameter(
                name="capture_output",
                type=ParameterType.BOOLEAN,
                description="Whether to capture stdout/stderr",
                required=False,
                default=True,
            ),
            ToolParameter(
                name="allowed_imports",
                type=ParameterType.ARRAY,
                description="List of allowed module imports",
                required=False,
            ),
        ],
        category="code",
        tags=["python", "execute", "sandbox"],
        is_dangerous=True,
        requires_confirmation=True,
    ))
    
    # Execute shell
    registry.register_tool(Tool.create(
        name="execute_shell",
        description="Execute a shell command",
        handler=execute_shell,
        parameters=[
            ToolParameter(
                name="command",
                type=ParameterType.STRING,
                description="Shell command to execute",
                required=True,
            ),
            ToolParameter(
                name="timeout",
                type=ParameterType.NUMBER,
                description="Execution timeout in seconds",
                required=False,
                default=30.0,
            ),
            ToolParameter(
                name="cwd",
                type=ParameterType.STRING,
                description="Working directory for the command",
                required=False,
            ),
            ToolParameter(
                name="env",
                type=ParameterType.OBJECT,
                description="Environment variables",
                required=False,
            ),
            ToolParameter(
                name="shell",
                type=ParameterType.BOOLEAN,
                description="Whether to use shell execution",
                required=False,
                default=True,
            ),
        ],
        category="code",
        tags=["shell", "execute", "command"],
        is_dangerous=True,
        requires_confirmation=True,
    ))
    
    # Analyze Python code
    registry.register_tool(Tool.create(
        name="analyze_python_code",
        description="Analyze Python code for structure and potential issues",
        handler=analyze_python_code,
        parameters=[
            ToolParameter(
                name="code",
                type=ParameterType.STRING,
                description="Python code to analyze",
                required=True,
            ),
        ],
        category="code",
        tags=["python", "analyze", "lint"],
    ))
    
    # Format Python code
    registry.register_tool(Tool.create(
        name="format_python_code",
        description="Format Python code using standard formatting",
        handler=format_python_code,
        parameters=[
            ToolParameter(
                name="code",
                type=ParameterType.STRING,
                description="Python code to format",
                required=True,
            ),
            ToolParameter(
                name="line_length",
                type=ParameterType.INTEGER,
                description="Maximum line length",
                required=False,
                default=88,
            ),
        ],
        category="code",
        tags=["python", "format", "style"],
    ))
