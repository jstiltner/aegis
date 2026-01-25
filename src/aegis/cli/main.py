"""
Main CLI entry point.

This module provides the main CLI commands using Typer.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.syntax import Syntax
from rich.markdown import Markdown

# Create Typer app
app = typer.Typer(
    name="agent-runtime",
    help="Production infrastructure for long-running autonomous agents",
    add_completion=True,
)

# Rich console for output
console = Console()


# =============================================================================
# Server Commands
# =============================================================================

@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of workers"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
) -> None:
    """
    Start the API server.
    
    Example:
        agent-runtime serve --port 8000 --reload
    """
    try:
        import uvicorn
    except ImportError:
        console.print("[red]Error: uvicorn is required. Install with: pip install uvicorn[/red]")
        raise typer.Exit(1)
    
    console.print(Panel.fit(
        f"[bold green]Starting Agent Runtime API Server[/bold green]\n"
        f"Host: {host}\n"
        f"Port: {port}\n"
        f"Workers: {workers}\n"
        f"Debug: {debug}",
        title="🚀 Server",
    ))
    
    uvicorn.run(
        "aegis.api.server:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level="debug" if debug else "info",
    )


# =============================================================================
# Agent Commands
# =============================================================================

agents_app = typer.Typer(help="Manage agents")
app.add_typer(agents_app, name="agents")


@agents_app.command("list")
def list_agents(
    format: str = typer.Option("table", "--format", "-f", help="Output format (table, json)"),
) -> None:
    """
    List all configured agents.
    """
    # Placeholder data
    agents = [
        {
            "agent_id": "agent-001",
            "name": "Code Assistant",
            "model": "claude-sonnet-4-20250514",
            "sessions": 5,
        },
        {
            "agent_id": "agent-002",
            "name": "Research Agent",
            "model": "claude-3-opus-20240229",
            "sessions": 2,
        },
    ]
    
    if format == "json":
        console.print_json(json.dumps(agents, indent=2))
    else:
        table = Table(title="Agents")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Model")
        table.add_column("Sessions", justify="right")
        
        for agent in agents:
            table.add_row(
                agent["agent_id"],
                agent["name"],
                agent["model"],
                str(agent["sessions"]),
            )
        
        console.print(table)


@agents_app.command("create")
def create_agent(
    name: str = typer.Argument(..., help="Agent name"),
    model: str = typer.Option("claude-sonnet-4-20250514", "--model", "-m", help="LLM model"),
    system_prompt: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
    tools: Optional[str] = typer.Option(None, "--tools", "-t", help="Comma-separated tool names"),
) -> None:
    """
    Create a new agent.
    
    Example:
        agent-runtime agents create "My Agent" --model claude-sonnet-4-20250514
    """
    tool_list = tools.split(",") if tools else []
    
    console.print(Panel.fit(
        f"[bold green]Agent Created[/bold green]\n"
        f"Name: {name}\n"
        f"Model: {model}\n"
        f"Tools: {', '.join(tool_list) or 'None'}",
        title="✅ Success",
    ))


@agents_app.command("delete")
def delete_agent(
    agent_id: str = typer.Argument(..., help="Agent ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """
    Delete an agent.
    """
    if not force:
        confirm = typer.confirm(f"Are you sure you want to delete agent {agent_id}?")
        if not confirm:
            raise typer.Abort()
    
    console.print(f"[green]Agent {agent_id} deleted[/green]")


# =============================================================================
# Session Commands
# =============================================================================

sessions_app = typer.Typer(help="Manage sessions")
app.add_typer(sessions_app, name="sessions")


@sessions_app.command("list")
def list_sessions(
    agent_id: Optional[str] = typer.Option(None, "--agent", "-a", help="Filter by agent ID"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
    format: str = typer.Option("table", "--format", "-f", help="Output format"),
) -> None:
    """
    List sessions.
    """
    # Placeholder data
    sessions = [
        {
            "session_id": "sess-001",
            "agent_id": "agent-001",
            "status": "active",
            "messages": 15,
            "created": "2024-01-15 10:30:00",
        },
        {
            "session_id": "sess-002",
            "agent_id": "agent-001",
            "status": "completed",
            "messages": 42,
            "created": "2024-01-14 14:20:00",
        },
    ]
    
    if format == "json":
        console.print_json(json.dumps(sessions, indent=2))
    else:
        table = Table(title="Sessions")
        table.add_column("ID", style="cyan")
        table.add_column("Agent", style="green")
        table.add_column("Status")
        table.add_column("Messages", justify="right")
        table.add_column("Created")
        
        for session in sessions:
            status_style = "green" if session["status"] == "active" else "dim"
            table.add_row(
                session["session_id"],
                session["agent_id"],
                f"[{status_style}]{session['status']}[/{status_style}]",
                str(session["messages"]),
                session["created"],
            )
        
        console.print(table)


@sessions_app.command("show")
def show_session(
    session_id: str = typer.Argument(..., help="Session ID"),
) -> None:
    """
    Show session details.
    """
    console.print(Panel.fit(
        f"[bold]Session: {session_id}[/bold]\n"
        f"Agent: agent-001\n"
        f"Status: active\n"
        f"Messages: 15\n"
        f"Checkpoints: 3\n"
        f"Created: 2024-01-15 10:30:00",
        title="📋 Session Details",
    ))


@sessions_app.command("replay")
def replay_session(
    session_id: str = typer.Argument(..., help="Session ID"),
    checkpoint_id: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="Start from checkpoint"),
    speed: float = typer.Option(1.0, "--speed", "-s", help="Replay speed multiplier"),
) -> None:
    """
    Replay a session from the beginning or a checkpoint.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Replaying session {session_id}...", total=None)
        
        # Simulate replay
        import time
        time.sleep(2)
        
        progress.update(task, description="Replay complete!")
    
    console.print("[green]Session replay completed[/green]")


# =============================================================================
# Trace Commands
# =============================================================================

traces_app = typer.Typer(help="View traces and audit logs")
app.add_typer(traces_app, name="traces")


@traces_app.command("list")
def list_traces(
    session_id: Optional[str] = typer.Option(None, "--session", "-s", help="Filter by session"),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of traces to show"),
) -> None:
    """
    List traces.
    """
    table = Table(title="Traces")
    table.add_column("Trace ID", style="cyan")
    table.add_column("Session")
    table.add_column("Spans", justify="right")
    table.add_column("Duration")
    table.add_column("Time")
    
    # Placeholder data
    traces = [
        ("trace-001", "sess-001", 5, "1.2s", "10:30:15"),
        ("trace-002", "sess-001", 3, "0.8s", "10:30:45"),
        ("trace-003", "sess-002", 8, "2.5s", "14:20:30"),
    ]
    
    for trace in traces:
        table.add_row(*[str(x) for x in trace])
    
    console.print(table)


@traces_app.command("show")
def show_trace(
    trace_id: str = typer.Argument(..., help="Trace ID"),
    format: str = typer.Option("tree", "--format", "-f", help="Output format (tree, json)"),
) -> None:
    """
    Show trace details with span tree.
    """
    if format == "json":
        trace_data = {
            "trace_id": trace_id,
            "session_id": "sess-001",
            "spans": [
                {"span_id": "span-1", "name": "agent.run", "duration_ms": 1200},
                {"span_id": "span-2", "name": "llm.complete", "duration_ms": 800, "parent": "span-1"},
                {"span_id": "span-3", "name": "tool.execute", "duration_ms": 300, "parent": "span-1"},
            ],
        }
        console.print_json(json.dumps(trace_data, indent=2))
    else:
        console.print(f"\n[bold cyan]Trace: {trace_id}[/bold cyan]\n")
        console.print("├── [green]agent.run[/green] (1200ms)")
        console.print("│   ├── [yellow]llm.complete[/yellow] (800ms)")
        console.print("│   └── [blue]tool.execute[/blue] (300ms)")
        console.print("")


# =============================================================================
# Run Command (Interactive)
# =============================================================================

@app.command()
def run(
    agent_id: Optional[str] = typer.Option(None, "--agent", "-a", help="Agent ID to use"),
    model: str = typer.Option("claude-sonnet-4-20250514", "--model", "-m", help="LLM model"),
    system_prompt: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """
    Run an interactive agent session.
    
    Example:
        agent-runtime run --model claude-sonnet-4-20250514
    """
    console.print(Panel.fit(
        "[bold green]Interactive Agent Session[/bold green]\n"
        f"Model: {model}\n"
        "Type 'exit' or 'quit' to end the session.\n"
        "Type '/help' for commands.",
        title="🤖 Agent Runtime",
    ))
    
    session_id = f"sess-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    console.print(f"[dim]Session: {session_id}[/dim]\n")
    
    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")
            
            if user_input.lower() in ("exit", "quit"):
                console.print("\n[yellow]Ending session...[/yellow]")
                break
            
            if user_input.startswith("/"):
                # Handle commands
                cmd = user_input[1:].lower()
                if cmd == "help":
                    console.print(Markdown("""
## Commands
- `/help` - Show this help
- `/status` - Show session status
- `/checkpoint` - Create a checkpoint
- `/history` - Show message history
- `/clear` - Clear the screen
- `exit` or `quit` - End session
                    """))
                elif cmd == "status":
                    console.print(f"[dim]Session: {session_id}, Messages: 0[/dim]")
                elif cmd == "clear":
                    console.clear()
                else:
                    console.print(f"[red]Unknown command: {cmd}[/red]")
                continue
            
            # Simulate agent response
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Thinking...", total=None)
                import time
                time.sleep(0.5)
            
            console.print(f"[bold green]Agent:[/bold green] This is a placeholder response to: {user_input}\n")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
        except EOFError:
            break
    
    console.print("[green]Session ended.[/green]")


# =============================================================================
# Config Commands
# =============================================================================

config_app = typer.Typer(help="Manage configuration")
app.add_typer(config_app, name="config")


@config_app.command("show")
def show_config() -> None:
    """
    Show current configuration.
    """
    config = {
        "api_url": "http://localhost:8000",
        "default_model": "claude-sonnet-4-20250514",
        "log_level": "info",
        "checkpoint_interval": 10,
    }
    
    console.print(Panel(
        Syntax(json.dumps(config, indent=2), "json"),
        title="Configuration",
    ))


@config_app.command("set")
def set_config(
    key: str = typer.Argument(..., help="Configuration key"),
    value: str = typer.Argument(..., help="Configuration value"),
) -> None:
    """
    Set a configuration value.
    """
    console.print(f"[green]Set {key} = {value}[/green]")


# =============================================================================
# Version Command
# =============================================================================

@app.command()
def version() -> None:
    """
    Show version information.
    """
    console.print(Panel.fit(
        "[bold]Agent Runtime[/bold]\n"
        "Version: 0.1.0\n"
        "Python: " + sys.version.split()[0],
        title="📦 Version",
    ))


# =============================================================================
# CLI Entry Point
# =============================================================================

def cli() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    cli()
