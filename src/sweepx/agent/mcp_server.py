"""MCP server — exposes the sweep-agent tool registry over the Model Context Protocol.

The built-in agent (``sweep-agent chat`` / ``ui``) brings its own local LLM and
ReAct loop. This module serves the SAME tools to any MCP client instead (Claude
Code, Claude Desktop, Cursor, ...), which supplies the model and the loop:

    pip install "sweepx[mcp]"
    claude mcp add sweepx -- sweep-agent mcp --workdir /path/to/runs

Tool results go to whatever model the client uses — for a hosted model that means
leaving the machine. Use public/synthetic models, or ``--no-images`` to keep the
saved figures to file paths only.

Only stdio transport. The protocol owns fds 0 and 1, so :func:`serve` points
fd 1 at stderr and fd 0 at /dev/null before any tool runs: prints from the runner
or the C core, and subprocesses the tools start, cannot touch the JSON-RPC stream.
The protocol loop runs in a background thread and tools run one at a time in the
main thread, where ``run_task``'s SIGALRM timeout works.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import sys
import threading
from concurrent.futures import Future
from io import TextIOWrapper
from pathlib import Path
from typing import Any

from sweepx.agent.tools import Registry, registry as default_registry

# What `sweep-agent mcp` offers by default. A strong client model covers every
# task type through describe_task_schema + build_spec; the specialised builders
# and one-call wrappers (`--tools all`) exist for small local models and cost
# ~2.5x the prompt (each spec builder alone is ~7 KB of schema).
CORE_TOOLS = (
    "list_equations", "list_benchmark_models", "get_benchmark_model", "make_synthetic_model",
    "inspect_file", "check_parameters", "run_forward_sweep",
    "describe_task_schema", "build_spec", "run_task",
    "submit_task", "job_status", "cancel_job", "read_status", "list_artifacts",
    "plot_model", "plot_shot_gather", "plot_convergence",
)

INSTRUCTIONS = """\
Tools of sweepx, a seismic wave-equation modelling / FWI / RTM stack built on the
`sweep` solver. Inputs and outputs are file paths (velocity models as .npy, runs
under ./sweep_runs/<task_id>/). Call inspect_file before assuming a file's shape.
Models from get_benchmark_model / make_synthetic_model carry their grid spacing:
leave dh out and the recorded one is used. A quick forward shot: run_forward_sweep,
and read the `physics` block it returns. Anything else (forward, wavefield, fwi,
rtm, lsrtm): describe_task_schema -> build_spec -> run_task -> plot_*.
list_equations lists the wave equations. Long or multi-GPU runs: submit_task
(Slurm) -> job_status, not run_task, which computes in this process.
Pass the task_dir a tool RETURNED to later tools; never invent one.
"""


def resolve_tool_names(tools: str | None, registry: Registry = default_registry) -> list[str]:
    """``"core"`` (default), ``"all"``, or a comma-separated list of tool names."""
    if tools in (None, "", "core"):
        return [n for n in CORE_TOOLS if registry.get(n) is not None]
    if tools == "all":
        return registry.names()
    names = [n.strip() for n in tools.split(",") if n.strip()]
    unknown = [n for n in names if registry.get(n) is None]
    if unknown:
        raise ValueError(f"unknown tool(s) {unknown}; `sweep-agent tools` lists them.")
    return names

# Figures handed back as image content (the client's model can look at them).
# Others (svg, mp4) and oversized files are returned as paths only.
_IMAGE_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}
_MAX_IMAGE_BYTES = 4 * 1024 * 1024


def _image_blocks(result_json: str) -> list[Any]:
    from mcp import types

    from sweepx.agent.webui import _figures_in_result

    blocks = []
    for path in _figures_in_result(result_json):
        mime = _IMAGE_MIME.get(Path(path).suffix.lower())
        if mime is None or os.path.getsize(path) > _MAX_IMAGE_BYTES:
            continue
        data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        blocks.append(types.ImageContent(type="image", data=data, mimeType=mime))
    return blocks


def build_server(registry: Registry = default_registry, *, images: bool = True, run_tool=None,
                 tools: str | None = "core"):
    """Return an ``mcp`` low-level ``Server`` serving ``registry``'s tools.

    The tool list is the one the local agent offers (``Registry.openai_specs``),
    limited to :func:`resolve_tool_names` (``tools``: ``"core"``, ``"all"`` or a
    comma-separated list).
    ``run_tool(tool, arguments)`` is an async callable returning the
    ``ToolResult``; the default runs the tool in a worker thread (FWI calls take
    minutes and must not block the protocol loop). :func:`serve` passes one that
    runs it on the main thread.
    """
    import anyio
    from mcp import types
    from mcp.server.lowlevel import Server

    from sweepx.agent import __version__

    if run_tool is None:
        async def run_tool(tool, arguments):
            return await anyio.to_thread.run_sync(tool.invoke, arguments)

    server = Server("sweepx", version=__version__, instructions=INSTRUCTIONS)
    specs = [s["function"] for s in registry.openai_specs(resolve_tool_names(tools, registry))]
    offered = {s["name"] for s in specs}

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [
            types.Tool(name=s["name"], description=s["description"], inputSchema=s["parameters"])
            for s in specs
        ]

    # Arguments are validated by the tool's own Pydantic model in Tool.invoke,
    # whose error message goes back to the model; skip the SDK's jsonschema pass.
    @server.call_tool(validate_input=False)
    async def _call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        tool = registry.get(name) if name in offered else None
        if tool is None:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps({"error": f"unknown tool: {name}"}))],
                isError=True,
            )
        result = await run_tool(tool, arguments or {})
        text = result.to_json()
        content: list[Any] = [types.TextContent(type="text", text=text)]
        if result.ok and images:
            content += _image_blocks(text)
        return types.CallToolResult(content=content, isError=not result.ok)

    return server


def serve(*, workdir: str | None = None, images: bool = True, tools: str | None = "core") -> int:
    """Run the MCP server on stdio until the client disconnects."""
    import anyio
    from mcp.server.stdio import stdio_server

    if workdir is not None:
        Path(workdir).mkdir(parents=True, exist_ok=True)
        os.chdir(workdir)  # tools default to ./sweep_runs

    # Keep the real stdin/stdout for the protocol. Everything else written to
    # fd 1 (Python prints, C/CUDA printf) goes to stderr; fd 0 reads /dev/null.
    sys.stdout.flush()
    proto_in_fd, proto_out_fd = os.dup(0), os.dup(1)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    proto_in = TextIOWrapper(os.fdopen(proto_in_fd, "rb"), encoding="utf-8")
    proto_out = TextIOWrapper(os.fdopen(proto_out_fd, "wb", buffering=0), encoding="utf-8", write_through=True)

    calls: queue.Queue = queue.Queue()

    async def run_on_main(tool, arguments):
        fut: Future = Future()
        calls.put((tool, arguments, fut))
        return await anyio.to_thread.run_sync(fut.result)

    server = build_server(images=images, run_tool=run_on_main, tools=tools)

    async def _run() -> None:
        async with stdio_server(stdin=anyio.wrap_file(proto_in), stdout=anyio.wrap_file(proto_out)) as (read, write):
            await server.run(read, write, server.create_initialization_options())

    failed: list[BaseException] = []

    def _protocol() -> None:
        try:
            anyio.run(_run)
        except BaseException as exc:  # reported after the main loop stops
            failed.append(exc)
        finally:
            calls.put(None)

    threading.Thread(target=_protocol, name="mcp-protocol", daemon=True).start()
    while (item := calls.get()) is not None:
        tool, arguments, fut = item
        if fut.set_running_or_notify_cancel():
            try:
                fut.set_result(tool.invoke(arguments))
            except Exception as exc:
                fut.set_exception(exc)
    if failed:
        raise failed[0]
    return 0
