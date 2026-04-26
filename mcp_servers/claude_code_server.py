"""MCP server that wraps the Claude Code CLI (claude -p).

Working directory for all claude invocations: /Users/nemulin/develop
"""

import asyncio
import os

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

CLAUDE_CODE_CWD = os.environ.get("CLAUDE_CODE_CWD", "/Users/nemulin/develop")
TIMEOUT_SECONDS = 30

app = Server("claude-code")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="run_claude_code",
            description=(
                f"Execute a development task using the Claude Code CLI. "
                f"Working directory: '{CLAUDE_CODE_CWD}'. "
                "Supports file creation/editing, code analysis, refactoring, and more."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task or prompt to pass to Claude Code",
                    },
                },
                "required": ["task"],
            },
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name != "run_claude_code":
        raise ValueError(f"Unknown tool: {name}")

    task = arguments["task"]

    cmd = ["claude", "-p", task, "--output-format", "text", "--dangerously-skip-permissions"]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=CLAUDE_CODE_CWD,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return [
                types.TextContent(
                    type="text",
                    text=f"Error: Claude Code timed out after {TIMEOUT_SECONDS} seconds.",
                )
            ]

        if proc.returncode == 0:
            return [types.TextContent(type="text", text=stdout.decode().strip())]

        error = (stderr.decode() or stdout.decode()).strip()
        return [
            types.TextContent(
                type="text",
                text=f"Claude Code failed (exit {proc.returncode}):\n{error}",
            )
        ]

    except FileNotFoundError:
        return [
            types.TextContent(
                type="text",
                text=(
                    "Error: 'claude' command not found. "
                    "Make sure Claude Code CLI is installed and available on PATH."
                ),
            )
        ]
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error: {e}")]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
