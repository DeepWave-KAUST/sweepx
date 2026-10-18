"""P22: per-query tool subsetting (select_tool_names) + Agent integration."""

from __future__ import annotations

from sweep_agent.agent import Agent
from sweep_agent.llm.base import BaseLLM, ChatMessage
from sweep_agent.tools import registry
from sweep_agent.tools.selection import CORE, select_tool_names


def test_core_always_present():
    names = select_tool_names("anything at all")
    assert set(CORE).issubset(set(names))


def test_routing_picks_relevant_groups():
    fwd = select_tool_names("do forward modelling and plot the shot gather")
    assert "run_forward_and_plot" in fwd
    wave = select_tool_names("animate how the wave propagates")
    assert "animate_wavefield" in wave
    fwi = select_tool_names("run a full waveform inversion")
    assert "run_fwi" in fwi
    aniso = select_tool_names("compare VTI and TTI wavefronts")
    assert "compare_equation_wavefields" in aniso
    model = select_tool_names("生成一个速度模型并看纵向切片")
    assert "make_synthetic_model" in model and "plot_velocity_slice" in model
    params = select_tool_names("这些参数稳不稳，会不会频散")
    assert "check_parameters" in params


def test_subset_is_smaller_than_full():
    names = select_tool_names("plot the shot gather")
    assert len(names) < len(registry.names())  # genuinely a subset


def test_unmatched_falls_back_broad():
    names = select_tool_names("hello there")
    assert "run_forward_and_plot" in names  # fallback includes the common tasks


def test_filter_to_available():
    names = select_tool_names("forward modelling", available=["inspect_file", "run_task"])
    assert set(names).issubset({"inspect_file", "run_task"})


class _CaptureLLM(BaseLLM):
    """Records how many tools it was handed, then ends the turn."""
    def __init__(self):
        self.n_tools = None

    def chat(self, messages, tools=None, **kw):
        self.n_tools = len(tools or [])
        return ChatMessage(role="assistant", content="ok")

    def close(self):
        pass


def test_agent_uses_subset():
    llm = _CaptureLLM()
    agent = Agent(llm=llm, tool_selector=select_tool_names, max_steps=3)
    list(agent.iter_chat("plot the shot gather for /tmp/vp.npy"))
    assert llm.n_tools is not None
    assert llm.n_tools < len(registry.names())   # fewer than all 30
    assert llm.n_tools >= len(CORE)               # but at least the core


def test_agent_without_selector_sees_all():
    llm = _CaptureLLM()
    agent = Agent(llm=llm, max_steps=3)  # no selector
    list(agent.iter_chat("anything"))
    assert llm.n_tools == len(registry.names())


def test_followup_keeps_used_tools():
    """A second turn lacking keywords still keeps tools used in the first turn."""
    class _OneToolLLM(BaseLLM):
        def __init__(self): self.turn = 0
        def chat(self, messages, tools=None, **kw):
            self._names = [t["function"]["name"] for t in (tools or [])]
            self.turn += 1
            if self.turn == 1:
                from sweep_agent.llm.base import ToolCall
                return ChatMessage(role="assistant", content=None,
                                   tool_calls=[ToolCall(id="c1", name="run_fwi", arguments={})])
            return ChatMessage(role="assistant", content="done")
        def close(self): pass

    llm = _OneToolLLM()
    agent = Agent(llm=llm, tool_selector=select_tool_names, max_steps=4)
    list(agent.iter_chat("run a full waveform inversion"))  # turn 1 uses run_fwi
    list(agent.iter_chat("ok thanks"))                       # turn 2: no keywords
    assert "run_fwi" in llm._names  # carried over from the first turn
