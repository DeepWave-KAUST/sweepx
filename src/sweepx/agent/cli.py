"""`sweep-agent` CLI — `chat`, `serve-llm`, `tools` subcommands."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Sequence

from sweep_agent.agent import Agent
from sweep_agent.llm.vllm_backend import (
    DEFAULT_URL,
    MODEL_ALIASES,
    VLLMBackend,
    backend_hint,
    default_model_for,
    detect_endpoint,
    ensure_ollama_model,
    resolve_model,
)
from sweep_agent.tools import registry


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sweep-agent", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    # chat ----------------------------------------------------------------
    pc = sub.add_parser("chat", help="Interactive natural-language session with sweep-agent.")
    pc.add_argument("--url", default=None, help=f"vLLM URL (default: $SWEEP_AGENT_LLM_URL or {DEFAULT_URL}).")
    pc.add_argument("--model", default=None, help="Model name or alias (default: $SWEEP_AGENT_LLM_MODEL).")
    pc.add_argument("--temperature", type=float, default=0.2)
    pc.add_argument("--max-steps", type=int, default=12, help="Max tool-call rounds per user turn.")
    pc.add_argument("-q", "--quiet", action="store_true", help="Suppress tool-call traces.")
    pc.set_defaults(func=_cmd_chat)

    # serve-llm -----------------------------------------------------------
    ps = sub.add_parser("serve-llm", help="Launch a local vLLM OpenAI-compatible server.")
    ps.add_argument(
        "--model",
        default="qwen2.5-14b-instruct",
        help=f"Model name or alias. Known aliases: {', '.join(MODEL_ALIASES)}.",
    )
    ps.add_argument("--port", type=int, default=8000)
    ps.add_argument("--host", default="0.0.0.0")
    ps.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ps.add_argument("--max-model-len", type=int, default=8192)
    ps.add_argument("--tensor-parallel-size", type=int, default=1)
    ps.add_argument("--dtype", default="auto", help="float16 / bfloat16 / auto")
    ps.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra flags passed to `vllm serve`.")
    ps.set_defaults(func=_cmd_serve)

    # ui ------------------------------------------------------------------
    pu = sub.add_parser("ui", help="Launch the Gradio chat window (drag-drop files, inline figures).")
    pu.add_argument("--host", default="0.0.0.0", help="Bind address (default 0.0.0.0).")
    pu.add_argument("--port", type=int, default=7860, help="Port (default 7860).")
    pu.add_argument("--url", default=None, help=f"vLLM URL the agent talks to (default: $SWEEP_AGENT_LLM_URL or {DEFAULT_URL}).")
    pu.add_argument("--model", default=None, help="Model name/alias (default: $SWEEP_AGENT_LLM_MODEL).")
    pu.add_argument("--max-steps", type=int, default=24, help="Max tool-call rounds per user turn.")
    pu.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    pu.set_defaults(func=_cmd_ui)

    # tools ---------------------------------------------------------------
    pt = sub.add_parser("tools", help="List the tools the LLM can call.")
    pt.add_argument("--json", action="store_true", help="Emit OpenAI tool specs as JSON.")
    pt.set_defaults(func=_cmd_tools)

    return p


def _cmd_ui(args: argparse.Namespace) -> int:
    # Resolve the endpoint/model once (explicit flag, then env, then auto-detect)
    # and pin them in the environment so the web UI uses the same ones, and
    # auto-pull the model if it's a missing Ollama model.
    url = args.url or os.environ.get("SWEEP_AGENT_LLM_URL") or detect_endpoint()
    hint = backend_hint(url)  # don't launch a UI that can't reach any LLM
    if hint is not None:
        print("[sweep-agent] " + hint, file=sys.stderr)
        return 1
    model = args.model or os.environ.get("SWEEP_AGENT_LLM_MODEL") or default_model_for(url)
    os.environ["SWEEP_AGENT_LLM_URL"] = url
    os.environ["SWEEP_AGENT_LLM_MODEL"] = model
    os.environ["SWEEP_AGENT_MAX_STEPS"] = str(args.max_steps)
    ensure_ollama_model(url, resolve_model(model))
    # Single launch path: webui.launch() owns theme/css/js + allowed_paths.
    from sweep_agent.webui import launch

    launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    from sweep_agent.tools.selection import select_tool_names
    url = args.url or os.environ.get("SWEEP_AGENT_LLM_URL") or detect_endpoint()
    hint = backend_hint(url)  # nothing reachable? tell the user how to get an LLM
    if hint is not None:
        print("[sweep-agent] " + hint, file=sys.stderr)
        return 1
    backend = VLLMBackend(url=url, model=args.model)
    ensure_ollama_model(backend.url, backend.model_id)  # one-time auto-pull on Ollama
    agent = Agent(llm=backend, max_steps=args.max_steps, tool_selector=select_tool_names)
    print(f"[sweep-agent] model={backend.model_id}  url={backend.url}")
    print("[sweep-agent] type your request, Ctrl-D to exit.\n")
    try:
        while True:
            try:
                user = input(">>> ").strip()
            except EOFError:
                print()
                return 0
            if not user:
                continue
            if user in {":reset", "/reset"}:
                agent.reset()
                print("[sweep-agent] history cleared.")
                continue
            for step in agent.iter_chat(user):
                if step.kind == "tool" and not args.quiet:
                    print(f"  · {step.tool_name}({step.tool_args}) → {step.tool_result_json[:200]}")
                elif step.kind == "final":
                    print(step.message.content or "")
    finally:
        backend.close()


def _cmd_serve(args: argparse.Namespace) -> int:
    if shutil.which("vllm") is None:
        print("error: `vllm` is not on PATH. install with `pip install vllm`.", file=sys.stderr)
        return 1
    model_id = resolve_model(args.model)
    cmd = [
        "vllm", "serve", model_id,
        "--host", args.host,
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--dtype", args.dtype,
        # tool-calling requires explicit opt-in on vLLM
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes" if "qwen" in model_id.lower() else "llama3_json",
    ]
    if args.extra:
        cmd.extend(args.extra)
    print(f"[sweep-agent] launching: {' '.join(cmd)}", file=sys.stderr)
    return subprocess.call(cmd, env=os.environ)


def _cmd_tools(args: argparse.Namespace) -> int:
    if args.json:
        import json
        print(json.dumps(registry.openai_specs(), indent=2, ensure_ascii=False))
        return 0
    for tool in registry.tools.values():
        desc = tool.description.split("\n")[0]
        print(f"  {tool.name:30s}  {desc}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
