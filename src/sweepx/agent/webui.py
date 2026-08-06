"""Gradio chat window for sweep-agent.

A browser chat where you **drag-and-drop** velocity-model / data files and get the
synthetic shot gathers, wavefield GIFs, FWI results, etc. rendered **inline** —
the natural-language counterpart of ``sweep-agent chat`` for people who don't
live in a terminal.

How it maps onto the existing agent:
- Uploaded files land in a temp dir; their paths are appended to the user
  message so the agent's tools (which take file paths) can read them.
- ``Agent.iter_chat`` is streamed: each tool call is shown as a trace line, and
  any figure a tool saved (``image_path`` / ``gif_path`` / FWI result images) is
  parsed out of the tool result and pushed to a gallery beside the chat.

Run:
    pip install -e ".[ui]"          # installs gradio
    export SWEEP_AGENT_LLM_URL=http://localhost:8001/v1
    sweep-agent ui                  # or: python -m sweep_agent.webui
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

from sweep_agent.agent import Agent
from sweep_agent.llm.vllm_backend import VLLMBackend
from sweep_agent.prompts import build_system_prompt
from sweep_agent.tools import registry as default_registry

# Tool-result keys that may hold a saved figure, in priority order.
_FIGURE_KEYS = ("image_path", "gif_path", "model_image", "convergence_image")
_FIGURE_EXTS = (".png", ".gif", ".jpg", ".jpeg", ".svg", ".mp4", ".webp")


def _figures_in_result(result_json: str | None) -> list[str]:
    """Extract existing figure file paths from one tool's JSON result."""
    if not result_json:
        return []
    try:
        data = json.loads(result_json)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, dict):
        return []
    found: list[str] = []
    # Named keys first (preserves a meaningful order), then any value that looks
    # like an image/animation file on disk.
    for key in _FIGURE_KEYS:
        v = data.get(key)
        if isinstance(v, str) and v.lower().endswith(_FIGURE_EXTS) and Path(v).is_file():
            if v not in found:
                found.append(v)
    for v in data.values():
        if isinstance(v, str) and v.lower().endswith(_FIGURE_EXTS) and Path(v).is_file():
            if v not in found:
                found.append(v)
    return found


def _compose_prompt(text: str, files: list[str]) -> str:
    """Fold dropped-file paths into the natural-language message so the agent's
    path-taking tools can use them."""
    text = (text or "").strip()
    if not files:
        return text
    listed = "\n".join(f"- {f}" for f in files)
    prefix = text or "请处理我拖入的文件。"
    return f"{prefix}\n\n[已上传文件 / uploaded files]\n{listed}"


def _trace_line(tool_name: str | None, tool_args: dict | None) -> str:
    args = json.dumps(tool_args or {}, ensure_ascii=False)
    if len(args) > 220:
        args = args[:220] + "…"
    return f"🔧 `{tool_name}` {args}"


# Markdown image embed pointing at a LOCAL file: ![alt](/tmp/x.png) or (file.png).
# Gradio's chatbot only serves files via its /file= proxy, so a raw local path
# renders as a BROKEN image icon. The figures are already shown inline (from the
# tool result), so we strip these embeds from the final reply — no info lost.
_MD_LOCAL_IMG = re.compile(r"!\[[^\]]*\]\(\s*(?!https?://|data:)[^)]*?\.(?:png|gif|jpe?g|svg|webp|mp4)\s*\)",
                           re.IGNORECASE)


def _strip_local_image_embeds(text: str) -> str:
    """Remove markdown image embeds that point at local files (shown inline
    already) so the final reply doesn't carry broken image icons. Leaves a bare
    reference to a leftover bullet/label clean."""
    if not text:
        return text
    out = _MD_LOCAL_IMG.sub("", text)
    # tidy up now-empty "- **Label**:" bullets and doubled blank lines left behind
    out = re.sub(r"(?m)^[ \t]*[-*][ \t]*(\*\*[^*]+\*\*:?)?[ \t]*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


# Claude-style action labels: what the agent is DOING, not the function name.
# {placeholders} are filled from the tool args (paths shortened to basenames).
_ACTION_TMPL: dict[str, str] = {
    "inspect_file": "Inspecting {base}",
    "list_equations": "Looking up wave equations",
    "describe_task_schema": "Reading the {task_type} task schema",
    "build_forward_spec": "Building the forward-modelling spec",
    "build_fwi_spec": "Building the FWI spec",
    "build_wavefield_spec": "Building the wavefield spec",
    "build_spec": "Building the {task_type} spec",
    "run_task": "Running the task",
    "read_status": "Checking task status",
    "list_artifacts": "Listing task artifacts",
    "check_parameters": "Checking stability (dh={dh}, dt={dt}, fm={fm})",
    "make_synthetic_model": "Making a {kind} synthetic model",
    "run_forward_and_plot": "Forward modelling + shot gather",
    "run_fwi": "Running the FWI inversion",
    "run_multiscale_fwi": "Running multiscale FWI",
    "animate_wavefield": "Simulating + animating the wavefield",
    "compare_equation_wavefields": "Comparing wavefronts across equations",
    "plot_shot_gather": "Plotting the shot gather",
    "plot_wavefield": "Plotting a wavefield snapshot",
    "make_wavefield_gif": "Animating the wavefield",
    "compare_wavefields": "Comparing wavefields",
    "plot_model": "Plotting the model",
    "plot_velocity_model": "Plotting the velocity model",
    "plot_velocity_slice": "Plotting velocity-depth profiles",
    "plot_convergence": "Plotting the convergence curve",
    "plot_observed_data": "Plotting the observed data",
    "plot_segy": "Plotting the SEG-Y data",
    "compare_shot_gathers": "Comparing shot gathers",
    "plot_wavelet": "Plotting the source wavelet",
    "animate_fwi_evolution": "Animating the FWI evolution",
}


class _SafeArgs(dict):
    def __missing__(self, key: str) -> str:  # unfilled placeholder → ellipsis
        return "…"


def _describe_call(tool_name: str | None, tool_args: dict | None) -> str:
    """Human-readable one-liner for a tool call (falls back to the tool name)."""
    args = tool_args or {}
    fmt = _SafeArgs()
    for k, v in args.items():
        fmt[k] = os.path.basename(v.rstrip("/")) if isinstance(v, str) and "/" in v else v
    fmt["base"] = os.path.basename(str(args.get("path") or args.get("model_path")
                                       or args.get("vp_path") or "file"))
    tmpl = _ACTION_TMPL.get(tool_name or "")
    if tmpl:
        return tmpl.format_map(fmt)
    return (tool_name or "tool").replace("_", " ").capitalize()


def run_agent_turn(agent, prompt: str, history: list):
    """Stream one agent turn, updating ``history`` and yielding it after each
    step. Figures the tools save are appended INLINE in the chat (as image
    messages); tool calls go into a COLLAPSED accordion so the chat stays clean.
    Pure logic — no gradio — so it is unit-testable with any object exposing
    ``iter_chat`` (see tests/test_p18_webui.py)."""
    def _tool_block(step) -> dict:
        args = json.dumps(step.tool_args or {}, ensure_ascii=False)
        if len(args) > 500:
            args = args[:500] + "…"
        return {
            "role": "assistant",
            # Human-readable action in the title; the real function name + args
            # stay inside the collapsed body for debugging.
            "content": f"`{step.tool_name}` · args: {args}",
            "metadata": {"title": f"🔧 {_describe_call(step.tool_name, step.tool_args)}"},
        }

    live_block: dict | None = None  # the block of the currently-running tool
    t0 = 0.0
    try:
        for step in agent.iter_chat(prompt):
            if step.kind == "note":
                # The model's own narration of what it's about to do.
                history.append({"role": "assistant", "content": step.message.content or ""})
                yield history
            elif step.kind == "tool_start":
                # Show the action live: spinner + open accordion while it runs.
                live_block = _tool_block(step)
                live_block["metadata"]["status"] = "pending"
                t0 = time.time()
                history.append(live_block)
                yield history
            elif step.kind == "tool":
                if live_block is None:  # e.g. a driver without tool_start events
                    live_block = _tool_block(step)
                    t0 = time.time()
                    history.append(live_block)
                md = live_block["metadata"]
                # Done → collapse; annotate duration + outcome next to the title.
                md["status"] = "done"
                md["duration"] = round(time.time() - t0, 1)
                figs = _figures_in_result(step.tool_result_json)
                try:
                    parsed = json.loads(step.tool_result_json or "")
                    err_msg = parsed.get("error") if isinstance(parsed, dict) else None
                except (ValueError, TypeError):
                    err_msg = None
                if err_msg:
                    md["log"] = "⚠ error"
                    # Surface WHY it failed inside the (expandable) block, not just the args.
                    live_block["content"] += f"\n\n**⚠ error:** {err_msg}"
                elif figs:
                    md["log"] = os.path.basename(figs[0])
                live_block = None
                # Any figure a tool saved is shown INLINE in the conversation.
                # Pass an absolute realpath so gradio's allowed_paths check matches
                # (esp. on macOS where /tmp resolves to /private/tmp).
                for fig in figs:
                    history.append({"role": "assistant", "content": {"path": os.path.realpath(fig)}})
                yield history
            elif step.kind == "final":
                # Figures already render inline; drop the model's local-path
                # image embeds so the reply has no broken-image icons.
                history.append({"role": "assistant",
                                "content": _strip_local_image_embeds(step.message.content or "")})
                yield history
    except Exception as exc:  # surface backend/connection errors in the chat
        name = type(exc).__name__
        if "Connect" in name or "Connection refused" in str(exc) or "ConnectError" in name:
            msg = ("⚠️ Can't reach the LLM backend — the vLLM server appears to be **down**. "
                   "Restart it (e.g. `bash examples/start_vllm_qwen7b.sh`), then resend.")
        else:
            msg = f"⚠️ error: {name}: {exc}"
        history.append({"role": "assistant", "content": msg})
        yield history


# --- message queue (Claude-style) -------------------------------------------
# A submit that lands while a turn is still streaming is captured instantly,
# echoed as "queued", and processed automatically when the current turn ends.
# State is module-level on purpose: this is a single-user local tool, and
# gradio's per-submit component snapshots would otherwise race (a message sent
# mid-turn carries a STALE chatbot/agent snapshot from the moment of submit,
# so threading them through gr.State can drop the running turn's output).
_pending: deque = deque()  # (text, files) submits waiting their turn
_chat: dict[str, Any] = {"history": [], "agent": None}  # canonical chat state


def queue_message(message) -> None:
    """Capture one submit into the pending queue (instant, never blocks)."""
    if isinstance(message, dict):
        text, files = message.get("text", ""), list(message.get("files", []) or [])
    else:
        text, files = str(message or ""), []
    if (text or "").strip() or files:
        _pending.append((text, files))


def _queued_lines() -> list[dict]:
    """Grey placeholder rows shown below the live turn for waiting messages."""
    out = []
    for text, files in list(_pending):
        label = (text or "(no text)").strip()
        if files:
            label += f" 📎{len(files)}"
        out.append({"role": "user", "content": f"⏳ *queued:* {label}"})
    return out


def drain_queue(make_agent):
    """Process every queued message in order, yielding display snapshots.

    Yields ``(display, phase, t_start)`` with phase "begin" (a turn starts),
    "step" (mid-turn update) or "done" (queue empty). The caller serialises
    invocations (gradio queues events per listener); an invocation that finds
    the queue already drained by its predecessor simply yields nothing."""
    if not _pending:
        return
    if _chat["agent"] is None:
        _chat["agent"] = make_agent()
    agent, history = _chat["agent"], _chat["history"]
    start = time.time()
    while _pending:
        text, files = _pending.popleft()
        echo = text or "(no text)"
        if files:
            echo += "\n\n📎 " + ", ".join(os.path.basename(f) for f in files)
        history.append({"role": "user", "content": echo})
        start = time.time()
        yield history + _queued_lines(), "begin", start
        prompt = _compose_prompt(text, files)
        for hist in run_agent_turn(agent, prompt, history):
            yield list(hist) + _queued_lines(), "step", start
    yield list(history), "done", start


# --- visual identity ---------------------------------------------------------
# Dark, glassy, geophysics-flavoured. Everything is self-contained (no external
# fonts/CDNs — this box may be offline): system font stack + inline SVG.

_FORCE_DARK_JS = """
() => {
  const u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'dark') {
    u.searchParams.set('__theme', 'dark');
    window.location.replace(u);
  }
}
"""

_CSS = """
:root {
  --sw-bg: #0a1018; --sw-panel: rgba(148,163,184,.055); --sw-line: rgba(148,163,184,.14);
  --sw-cyan: #22d3ee; --sw-cyan2: #0ea5e9; --sw-amber: #fbbf24;
  --sw-text: #e2e8f0; --sw-dim: #8fa3bb;
}
body, .gradio-container {
  background:
    radial-gradient(1100px 560px at 12% -8%, rgba(34,211,238,.09), transparent 60%),
    radial-gradient(900px 480px at 96% -4%, rgba(14,165,233,.07), transparent 55%),
    var(--sw-bg) !important;
}
.gradio-container { max-width: 1060px !important; margin: 0 auto !important; padding-top: 6px !important; }
footer { display: none !important; }

/* tighten the vertical rhythm so the top bars don't eat the page */
.gradio-container .gap { gap: 8px !important; }
.gradio-container > .main, .gradio-container .contain { padding-top: 4px !important; }
.sw-meter { margin: 0 !important; }
/* kill the empty label row above the context meter / hero HTML blocks */
.gradio-container .html-container { padding: 0 !important; }

/* dark scrollbars — the pale default thumb reads as a white bar on this bg */
* { scrollbar-color: rgba(148,163,184,.32) transparent; scrollbar-width: thin; }
*::-webkit-scrollbar { width: 8px; height: 8px; }
*::-webkit-scrollbar-track { background: transparent; }
*::-webkit-scrollbar-thumb { background: rgba(148,163,184,.28); border-radius: 999px; }
*::-webkit-scrollbar-thumb:hover { background: rgba(148,163,184,.5); }
*::-webkit-scrollbar-corner { background: transparent; }

/* hero */
.sw-hero { position: relative; padding: 9px 18px; border-radius: 13px;
  background: linear-gradient(135deg, rgba(34,211,238,.10), transparent 55%), var(--sw-panel);
  border: 1px solid var(--sw-line); overflow: hidden; }
.sw-hero h1 { margin: 0; font-size: 1.1rem; font-weight: 750; letter-spacing: .015em;
  color: var(--sw-text); display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
  padding-right: 220px; }
.sw-hero h1 .accent { color: var(--sw-cyan); }
.sw-hero .sw-tag { font-size: .76rem; font-weight: 400; color: var(--sw-dim); letter-spacing: .01em; }
.sw-badge { position: absolute; top: 50%; transform: translateY(-50%); right: 16px;
  display: flex; align-items: center; gap: 7px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .71rem;
  color: var(--sw-dim); padding: 5px 11px; border: 1px solid var(--sw-line); border-radius: 999px;
  background: rgba(10,16,24,.55); backdrop-filter: blur(6px); }
.sw-badge .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--sw-cyan);
  box-shadow: 0 0 8px rgba(34,211,238,.9); }

/* context meter — thin single strip */
.sw-meter { display: flex; align-items: center; gap: 12px; padding: 5px 14px;
  background: var(--sw-panel); border: 1px solid var(--sw-line); border-radius: 11px; }
.sw-meter-top { display: flex; justify-content: space-between; flex: none; gap: 10px; font-size: .72rem;
  letter-spacing: .09em; color: var(--sw-dim); white-space: nowrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.sw-meter-top span:first-child { display: none; }  /* drop the 'CONTEXT' label; the bar is self-evident */
.sw-meter-bar { flex: 1; height: 6px; border-radius: 999px; background: rgba(148,163,184,.14); overflow: hidden; order: 1; }
.sw-meter-top { order: 2; }  /* numbers sit to the RIGHT of the bar */
.sw-meter-fill { height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, var(--sw-cyan), var(--sw-cyan2));
  transition: width .45s ease; box-shadow: 0 0 10px rgba(34,211,238,.35); }
.sw-meter-fill.warn { background: linear-gradient(90deg, #fbbf24, #f59e0b); box-shadow: 0 0 10px rgba(251,191,36,.3); }
.sw-meter-fill.hot { background: linear-gradient(90deg, #f87171, #ef4444); box-shadow: 0 0 10px rgba(248,113,113,.35); }

/* status line */
.sw-status { display: flex; align-items: center; gap: 9px; font-size: .86rem; color: var(--sw-dim); padding: 1px 4px; }
.sw-status .dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
.sw-ready .dot { background: #34d399; box-shadow: 0 0 7px rgba(52,211,153,.7); }
.sw-done  .dot { background: #34d399; box-shadow: 0 0 7px rgba(52,211,153,.7); }
.sw-busy  .dot { background: var(--sw-amber); animation: sw-pulse 1.1s ease-in-out infinite; }
@keyframes sw-pulse { 0%,100% { opacity: .35; transform: scale(.8);} 50% { opacity: 1; transform: scale(1.2);} }

/* chat card */
#sw-chat { background: var(--sw-panel) !important; border: 1px solid var(--sw-line) !important;
  border-radius: 18px !important; box-shadow: 0 22px 70px rgba(0,0,0,.45); }
#sw-chat .message { border-radius: 14px !important; }
#sw-chat img { border-radius: 12px; }

/* input */
#sw-input { border-radius: 16px !important; border: 1px solid var(--sw-line) !important;
  background: var(--sw-panel) !important; transition: border-color .2s, box-shadow .2s; }
#sw-input:focus-within { border-color: rgba(34,211,238,.55) !important;
  box-shadow: 0 0 0 3px rgba(34,211,238,.13) !important; }

/* example chips + clear */
.sw-chips button { border-radius: 999px !important; font-size: .82rem !important;
  background: rgba(148,163,184,.07) !important; border: 1px solid var(--sw-line) !important;
  color: var(--sw-dim) !important; transition: all .18s ease; }
.sw-chips button:hover { color: var(--sw-cyan) !important; border-color: rgba(34,211,238,.5) !important;
  transform: translateY(-1px); }
#sw-clear { background: transparent !important; border: 1px dashed var(--sw-line) !important;
  color: var(--sw-dim) !important; border-radius: 12px !important; }
#sw-clear:hover { color: #f87171 !important; border-color: rgba(248,113,113,.45) !important; }
"""


def _hero_html(model: str) -> str:
    short = model.split("/")[-1]
    return f"""
<div class="sw-hero">
  <div class="sw-badge"><span class="dot"></span>{short}</div>
  <h1>sweep<span class="accent">-agent</span>
    <span class="sw-tag">natural-language seismic modelling &amp; inversion — figures render inline</span>
  </h1>
</div>"""


def _status_html(kind: str, text: str) -> str:
    """Status pill: kind is 'ready' | 'busy' | 'done' (drives the dot colour/pulse)."""
    return f'<div class="sw-status sw-{kind}"><span class="dot"></span><span>{text}</span></div>'


def _make_token_counter(model_id: str):
    """Return a ``text -> n_tokens`` estimator. Uses the model's real tokenizer
    when transformers + the cached files are available; otherwise a ~chars/4
    heuristic. Either way it's just a gauge for the context meter."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)

        def _count(text: str) -> int:
            return len(tok.encode(text or "", add_special_tokens=False))

        return _count
    except Exception:
        return lambda text: (len(text or "") + 3) // 4


def _server_max_len(url: str, default: int = 32768) -> int:
    """Best-effort fetch of the served model's max context length."""
    try:
        import httpx

        base = url.rstrip("/")
        data = httpx.get(f"{base}/models", timeout=4.0).json()
        for m in data.get("data", []):
            if isinstance(m.get("max_model_len"), int):
                return m["max_model_len"]
    except Exception:
        pass
    return default


def _context_line(agent, fallback_tools_tokens: int, max_len: int, count) -> str:
    """A one-line context-budget meter for the UI. Counts the tool specs the
    agent ACTUALLY sent last request (the per-turn subset), not all ~30 tools —
    the full registry would overstate usage ~2.5x."""
    if agent is not None and getattr(agent, "history", None):
        hist_tokens = sum(count(m.content or "") for m in agent.history)
    else:
        hist_tokens = count(build_system_prompt())
    specs = getattr(agent, "last_tool_specs", None) if agent is not None else None
    tools_tokens = count(json.dumps(specs)) if specs else fallback_tools_tokens
    used = hist_tokens + tools_tokens
    left = max(0, max_len - used)
    pct = min(100, round(100 * used / max_len)) if max_len else 0
    fill_cls = "hot" if pct >= 85 else ("warn" if pct >= 65 else "")
    return (
        '<div class="sw-meter">'
        f'<div class="sw-meter-top"><span>CONTEXT</span>'
        f'<span>{used:,} / {max_len:,} tk &middot; {left:,} left &middot; {pct}%</span></div>'
        f'<div class="sw-meter-bar"><div class="sw-meter-fill {fill_cls}" style="width:{pct}%"></div></div>'
        '</div>'
    )


def build_app(max_steps: int = 24):
    """Build (but don't launch) the Gradio Blocks app."""
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            "gradio is required for the web UI. Install it with:\n"
            "    pip install -e \".[ui]\"   (from the sweep-agent repo)\n"
            f"(import error: {exc})"
        )

    from sweep_agent.llm.vllm_backend import default_model_for, detect_endpoint

    url = os.environ.get("SWEEP_AGENT_LLM_URL") or detect_endpoint()
    model = os.environ.get("SWEEP_AGENT_LLM_MODEL") or default_model_for(url)
    count = _make_token_counter(model)
    # Before the first turn we don't know the subset yet — estimate with the
    # keyword-router's no-match default (CORE + fallback), not all ~30 tools.
    from sweep_agent.tools.selection import select_tool_names
    tools_tokens = count(json.dumps(default_registry.openai_specs(select_tool_names(""))))
    max_len = _server_max_len(url)

    def _new_agent() -> Agent:
        from sweep_agent.tools.selection import select_tool_names
        return Agent(llm=VLLMBackend(), max_steps=max_steps, tool_selector=select_tool_names)

    def _submit(message):
        """Instant path for every submit: queue the message, clear the box.
        Runs even while a turn is streaming (own listener, never blocked)."""
        queue_message(message)
        return gr.MultimodalTextbox(value=None)

    def _drain():
        """Stream queued turns. Gradio serialises this listener, so a submit
        that lands mid-turn waits automatically and is processed next."""
        drained = False
        for display, phase, start in drain_queue(_new_agent):
            drained = True
            ctx = _context_line(_chat["agent"], tools_tokens, max_len, count)
            if phase == "begin":
                yield display, ctx, start, gr.Timer(active=True), _status_html("busy", "Thinking… 0.0s")
            elif phase == "done":
                yield (display, ctx, None, gr.Timer(active=False),
                       _status_html("done", f"Done in {time.time() - start:.1f}s"))
            else:
                yield display, ctx, gr.skip(), gr.skip(), gr.skip()
        if not drained:  # backlog already consumed by the previous invocation
            yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

    with gr.Blocks(title="sweep-agent", fill_height=True) as app:
        gr.HTML(_hero_html(model))
        start_ts = gr.State(None)        # turn start time while "thinking"
        timer = gr.Timer(0.5, active=False)  # ticks the elapsed clock while busy

        ctx_md = gr.HTML(_context_line(None, tools_tokens, max_len, count))
        chatbot = gr.Chatbot(
            height=360, label="Conversation", render_markdown=True, elem_id="sw-chat",
            # Safety net: render well-formed math if the model still emits any
            # (the system prompt asks it to avoid LaTeX altogether).
            latex_delimiters=[
                {"left": "$$", "right": "$$", "display": True},
                {"left": "\\(", "right": "\\)", "display": False},
                {"left": "\\[", "right": "\\]", "display": True},
            ],
        )
        # Thinking indicator sits right above the input box — most visible there.
        status_md = gr.HTML(_status_html("ready", "Ready"))
        box = gr.MultimodalTextbox(
            placeholder="Drag in .npy files, or just type — e.g. 'run forward modelling on /tmp/vp.npy and plot the shot gather'",
            file_count="multiple",
            show_label=False,
            elem_id="sw-input",
        )
        with gr.Row():
            clear = gr.Button("🗑 Clear conversation", variant="secondary", scale=1, elem_id="sw-clear")
        # Quick-start example chips — click to fill the input box.
        with gr.Row(elem_classes=["sw-chips"]):
            ex_model = gr.Button("📈 Make a 2-layer model", size="sm")
            ex_fwd = gr.Button("🌊 Forward + shot gather", size="sm")
            ex_wave = gr.Button("🎞 Elastic wavefield movie", size="sm")
            ex_fwi = gr.Button("🔬 Hello-FWI", size="sm")
        _EXAMPLES = [
            (ex_model, "Generate a two-layer synthetic velocity model and show it."),
            (ex_fwd, "Run forward modelling on /tmp/synthetic_vp.npy (dh=10 m, record 0.6 s, fm 10 Hz) and plot the shot gather."),
            (ex_wave, "Make a homogeneous elastic model (vp/vs/rho), put the source at the centre, and animate how the P and S waves propagate."),
            (ex_fwi, "Run a synthetic FWI: make a true two-layer model and a smooth starting model, invert, and show the inverted-vs-true model and the convergence curve."),
        ]
        for _btn, _txt in _EXAMPLES:
            # Plain value patch — NOT gr.MultimodalTextbox(...): a full component
            # update re-mounts the input in gradio 6, which spuriously fires its
            # submit event (the chip "sent itself"). gr.update only sets value.
            _btn.click(lambda t=_txt: gr.update(value={"text": t, "files": []}), outputs=[box])

        # --- "thinking" indicator (pulsing dot + live elapsed clock) ----------
        def _tick(start):
            if not start:
                return _status_html("ready", "Ready")
            line = f"Thinking… {time.time() - start:.1f}s"
            if _pending:  # instant feedback for messages queued mid-turn
                line += f" &middot; ⏳ {len(_pending)} queued"
            return _status_html("busy", line)

        timer.tick(_tick, inputs=[start_ts], outputs=[status_md])

        # Two-stage submit (Claude-style queueing): `_submit` captures the
        # message and clears the box INSTANTLY, even mid-turn; `_drain` streams
        # the turns and is serialised by gradio's per-listener queue, so a
        # message sent while busy waits for the running turn automatically.
        box.submit(_submit, inputs=[box], outputs=[box]).then(
            _drain,
            outputs=[chatbot, ctx_md, start_ts, timer, status_md],
        )

        def _reset():
            _pending.clear()
            _chat["history"] = []
            _chat["agent"] = None
            return ([], _context_line(None, tools_tokens, max_len, count), None,
                    gr.Timer(active=False), _status_html("ready", "Ready"))

        clear.click(_reset, outputs=[chatbot, ctx_md, start_ts, timer, status_md])

    return app


def launch(server_name: str = "0.0.0.0", server_port: int = 7860, share: bool = False, **kwargs: Any) -> None:
    """Build and launch the web UI."""
    import gradio as gr

    app = build_app(max_steps=int(os.environ.get("SWEEP_AGENT_MAX_STEPS", "24")))
    # Tools save figures under /tmp/sweep_runs, $TMPDIR, task dirs, cwd … — gradio
    # only serves files from allowed roots, so whitelist them or the inline images
    # raise InvalidPathError. Use REALPATHS: on macOS /tmp and $TMPDIR are symlinks
    # (/private/tmp, /private/var/folders/…) and gradio resolves each served file
    # to its realpath, so "/tmp" alone would never match and every figure breaks.
    import tempfile
    _roots = {os.path.realpath(p) for p in ("/tmp", tempfile.gettempdir(), os.getcwd())}
    kwargs.setdefault("allowed_paths", sorted(_roots))
    # Gradio 6 takes theme/css/js on launch(), not on Blocks().
    kwargs.setdefault("theme", gr.themes.Base(
        primary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
        font=["-apple-system", "BlinkMacSystemFont", "Segoe UI", "Inter", "Roboto",
              "Helvetica Neue", "sans-serif"],
        font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
    ))
    kwargs.setdefault("css", _CSS)
    kwargs.setdefault("js", _FORCE_DARK_JS)
    app.queue().launch(server_name=server_name, server_port=server_port, share=share, **kwargs)


if __name__ == "__main__":  # python -m sweep_agent.webui
    launch()
