"""MCP adapter: the registry served over the Model Context Protocol.

In-memory client for the tool surface; one stdio subprocess for the transport:
a tool that writes to stdout (Python print and raw fd 1) and reads stdin, the
main-thread tool loop, and run_task's timeout through it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import anyio
import numpy as np
import pytest

pytest.importorskip("mcp")

from mcp import StdioServerParameters, types  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402
from mcp.client.session import ClientSession  # noqa: E402
from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from sweepx.agent.mcp_server import CORE_TOOLS, build_server, resolve_tool_names  # noqa: E402
from sweepx.agent.tools import registry  # noqa: E402


def _session_run(fn, **server_kw):
    async def _main():
        server = build_server(**server_kw)
        async with create_connected_server_and_client_session(server) as client:
            return await fn(client)

    return anyio.run(_main)


def _payload(result: types.CallToolResult) -> dict:
    return json.loads(result.content[0].text)


def test_lists_the_registry_tools_with_their_schemas():
    async def fn(client):
        return (await client.list_tools()).tools

    tools = _session_run(fn, tools="all")
    specs = {s["function"]["name"]: s["function"] for s in registry.openai_specs()}
    assert {t.name for t in tools} == set(specs)
    for t in tools:
        assert t.inputSchema == specs[t.name]["parameters"]
        assert t.description == specs[t.name]["description"]


def test_default_offers_the_core_tools_only():
    async def fn(client):
        listed = (await client.list_tools()).tools
        hidden = await client.call_tool("build_forward_spec", {})
        return listed, hidden

    listed, hidden = _session_run(fn)
    assert {t.name for t in listed} == set(CORE_TOOLS)
    assert all(registry.get(n) is not None for n in CORE_TOOLS)
    assert hidden.isError and "unknown tool" in _payload(hidden)["error"]
    full = len(json.dumps(registry.openai_specs()))
    core = len(json.dumps(registry.openai_specs(list(CORE_TOOLS))))
    assert core < full / 2  # the point of the core set: a much smaller prompt


def test_tool_selection_by_name():
    assert resolve_tool_names("inspect_file, list_equations") == ["inspect_file", "list_equations"]
    assert resolve_tool_names("all") == registry.names()
    with pytest.raises(ValueError, match="no_such_tool"):
        resolve_tool_names("inspect_file,no_such_tool")


def test_call_tool_round_trip(tmp_path):
    out = tmp_path / "vp.npy"

    async def fn(client):
        made = await client.call_tool("make_synthetic_model", {"kind": "layered", "nz": 20, "nx": 30, "out_path": str(out)})
        info = await client.call_tool("inspect_file", {"path": str(out)})
        return made, info

    made, info = _session_run(fn)
    assert not made.isError and not info.isError
    assert out.is_file()
    assert _payload(info)["shape"] == [20, 30]


def test_bad_arguments_and_unknown_tool_are_errors():
    async def fn(client):
        bad = await client.call_tool("make_synthetic_model", {"nz": 2})  # ge=8
        unknown = await client.call_tool("no_such_tool", {})
        return bad, unknown

    bad, unknown = _session_run(fn)
    assert bad.isError and "validation" in _payload(bad)["error"]
    assert unknown.isError and "unknown tool" in _payload(unknown)["error"]


@pytest.mark.parametrize("images", [True, False])
def test_saved_figure_is_attached_as_image(tmp_path, images):
    vp, png = tmp_path / "vp.npy", tmp_path / "vp.png"

    async def fn(client):
        await client.call_tool("make_synthetic_model", {"nz": 20, "nx": 30, "out_path": str(vp)})
        return await client.call_tool("plot_model", {"model_path": str(vp), "out_path": str(png)})

    res = _session_run(fn, images=images, tools="all")
    assert not res.isError and png.is_file()
    kinds = [c.type for c in res.content]
    assert kinds == (["text", "image"] if images else ["text"])
    if images:
        assert res.content[1].mimeType == "image/png"


_NOISY_SERVER = textwrap.dedent("""
    import os
    import threading
    from pydantic import BaseModel
    from sweepx.agent.tools import register
    from sweepx.agent.mcp_server import serve

    class NoArgs(BaseModel):
        pass

    @register(name="list_equations_noisy", description="writes to stdout", params_model=NoArgs)
    def _noisy(_):
        print("python print to stdout")
        os.write(1, b"raw write to fd 1\\n")
        return {"stdin": os.read(0, 100).decode(), "main_thread": threading.current_thread() is threading.main_thread()}

    serve(workdir=WORKDIR, tools="all")
""")


def test_stdio_transport(tmp_path):
    vp_path = tmp_path / "vp.npy"
    np.save(vp_path, np.full((24, 32), 2000.0, np.float32))
    slow = registry.get("build_forward_spec").invoke({
        "vp_path": str(vp_path), "dh": 10.0, "dt": 0.001, "nt": 200000, "fm": 10.0,
        "output_dir": str(tmp_path / "runs"), "task_id": "slow", "device": "cpu",
    }).value["yaml_path"]
    params = StdioServerParameters(
        command=sys.executable,
        args=["-c", _NOISY_SERVER.replace("WORKDIR", repr(str(tmp_path / "runs")))],
        env=dict(os.environ),  # the SDK passes only a few variables by default (no PYTHONPATH)
    )
    errlog = open(tmp_path / "stderr.log", "w")

    async def _main():
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as client:
                init = await client.initialize()
                names = {t.name for t in (await client.list_tools()).tools}
                res = await client.call_tool("list_equations_noisy", {})
                timed = await client.call_tool("run_task", {"yaml_path": slow, "timeout_s": 2})
                return init, names, res, timed

    init, names, res, timed = anyio.run(_main)
    errlog.close()
    assert init.serverInfo.name == "sweepx"
    assert "list_equations_noisy" in names
    assert not res.isError and _payload(res) == {"stdin": "", "main_thread": True}
    assert "TimeoutError" in _payload(timed)["error"]
    log = (tmp_path / "stderr.log").read_text()
    assert "python print to stdout" in log and "raw write to fd 1" in log
    assert (tmp_path / "runs").is_dir()


def test_cli_mcp_subcommand_is_wired():
    out = subprocess.run(
        [sys.executable, "-m", "sweepx.agent.cli", "mcp", "--help"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "--workdir" in out and "--no-images" in out
