"""P27: what the LLM demos ran into — strict tool arguments, relative paths,
directory out_paths, the shot-gather axis, the elastic source, claimed figures,
the LLM server's errors, SEG-Y geometry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from sweepx.agent.agent import Agent
from sweepx.agent.llm.base import BaseLLM, ChatMessage, ToolCall
from sweepx.agent.tools import registry


# --------------------------------------------------------------- tool arguments

def test_unknown_argument_is_an_error_naming_the_valid_ones():
    res = registry.get("plot_wavelet").invoke({"fm": 10.0, "dt": 1e-3, "colour": "red"})
    assert not res.ok
    assert "colour" in res.error and "fm" in res.error


# --------------------------------------------------------------- paths

def test_relative_paths_become_absolute_in_the_yaml(tmp_path, monkeypatch):
    pytest.importorskip("sweepx.tasks")
    monkeypatch.chdir(tmp_path)
    np.save("vp.npy", np.full((40, 48), 2000.0, dtype=np.float32))
    b = registry.get("build_forward_spec").invoke({
        "vp_path": "vp.npy", "dh": 10.0, "dt": 1e-3, "nt": 60, "fm": 12.0,
        "source_step": 48, "receiver_step": 4, "abcn": 10, "spatial_order": 4,
    })
    assert b.ok and "error" not in b.value, b.value
    raw = yaml.safe_load(Path(b.value["yaml_path"]).read_text())
    here = tmp_path.resolve()
    assert raw["models"][0]["path"] == str(here / "vp.npy")
    assert raw["output_dir"] == str(here / "sweep_runs")
    r = registry.get("run_task").invoke({"yaml_path": b.value["yaml_path"], "timeout_s": 120})
    assert r.value.get("state") == "success", r.value
    assert Path(r.value["task_dir"]).resolve().parent == here / "sweep_runs"   # not sweep_runs/sweep_runs


def test_output_file_takes_a_directory():
    from sweepx.agent.tools.build_forward import _output_file

    assert _output_file("/tmp/run", "w.gif") == Path("/tmp/run/w.gif")
    assert _output_file("/tmp/run/a.png", "w.gif") == Path("/tmp/run/a.png")


@pytest.mark.timeout(180)
def test_run_forward_and_plot_one_shot_into_a_directory(tmp_path):
    pytest.importorskip("sweepx.tasks")
    vp = tmp_path / "vp.npy"
    np.save(vp, np.full((40, 60), 2000.0, dtype=np.float32))
    r = registry.get("run_forward_and_plot").invoke({
        "vp_path": str(vp), "out_path": str(tmp_path / "out"), "dh": 10.0,
        "record_length_s": 0.2, "fm": 12.0, "receiver_step": 2,
        "output_dir": str(tmp_path / "runs"), "spatial_order": 4, "abcn": 10,
    })
    assert r.ok and "error" not in r.value, r.value
    assert r.value["image_path"] == str(tmp_path / "out" / "shot_gather.png")
    assert Path(r.value["image_path"]).is_file()
    out = Path(r.value["task_dir"]) / "output"
    assert np.load(out / "record.npy").shape[0] == 1                       # one shot
    assert np.load(out / "sources.npy").tolist() == [[30, 1]]              # in the middle


def test_shot_gather_axis_is_the_receiver_positions(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from sweepx.agent.tools.visualize import _receiver_x
    from sweepx.tasks.viz.seismic import plot_shot

    out = tmp_path / "output"
    out.mkdir()
    rec = np.stack([np.arange(0, 96, 2), np.full(48, 4)], axis=1)[None]   # every 2nd cell
    np.save(out / "receivers.npy", rec)
    x = _receiver_x(tmp_path, 0, 12.5, 48)
    assert x[0] == 0.0 and x[-1] == 94 * 12.5                              # not 48 * 12.5
    ax = plot_shot(np.zeros((10, 48)), dt=1e-3, x=x)
    assert ax.images[0].get_extent()[:2] == [0.0, 1175.0]
    assert _receiver_x(tmp_path, 0, None, 48) is None                     # no dh: trace axis
    np.save(out / "receivers.npy", np.zeros((1, 48, 3)))                   # 3-D: trace axis
    assert _receiver_x(tmp_path, 0, 12.5, 48) is None


# --------------------------------------------------------------- elastic source

def test_elastic_animation_source_radiates_p_and_s():
    pytest.importorskip("sweep")
    from sweepx.agent.tools.build_wavefield import _p_and_s_source

    assert _p_and_s_source("Elastic") == ["vz"]
    assert _p_and_s_source("Acoustic") is None


def test_wavefield_builder_defaults_an_elastic_source_to_p_and_s(tmp_path):
    pytest.importorskip("sweepx.tasks")
    for name, val in (("vp", 2500.0), ("vs", 1500.0), ("rho", 2200.0)):
        np.save(tmp_path / f"{name}.npy", np.full((40, 40), val, dtype=np.float32))
    common = {"vp_path": str(tmp_path / "vp.npy"), "equation": "Elastic", "dh": 10.0, "dt": 1e-3,
              "nt": 50, "snapshot_times": [40], "extra_models": {"vs": str(tmp_path / "vs.npy"),
              "rho": str(tmp_path / "rho.npy")}, "output_dir": str(tmp_path / "runs")}
    b = registry.get("build_wavefield_spec").invoke(common)
    assert b.ok and b.value["spec"]["physics"]["source_type"] == ["vz"], b.value
    b = registry.get("build_wavefield_spec").invoke({**common, "source_type": ["sxx", "szz"]})
    assert b.value["spec"]["physics"]["source_type"] == ["sxx", "szz"]


# --------------------------------------------------------------- claimed figures

class _ScriptLLM(BaseLLM):
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []
        self._model_id = "fake"

    def chat(self, messages, tools=None, **kw):
        self.seen.append([m.content for m in messages])
        return self.replies.pop(0)


def test_reply_naming_a_missing_figure_is_sent_back_once(tmp_path):
    llm = _ScriptLLM([
        ChatMessage(role="assistant", content=f"The gather is at {tmp_path}/g.png."),
        ChatMessage(role="assistant", content="I could not make a figure."),
    ])
    steps = list(Agent(llm=llm).iter_chat("run the forward"))
    assert steps[-1].message.content == "I could not make a figure."
    assert any(f"{tmp_path}/g.png" in (c or "") and "do not exist" in (c or "")
               for c in llm.seen[1])


def test_a_figure_still_missing_after_the_nudge_is_flagged(tmp_path):
    llm = _ScriptLLM([
        ChatMessage(role="assistant", content="Saved to wavefield.gif"),
        ChatMessage(role="assistant", content="Here: ![](wavefield.gif)"),
    ])
    steps = list(Agent(llm=llm).iter_chat("run it"))
    assert steps[-1].message.content.endswith("[sweep-agent] No tool wrote wavefield.gif in this turn.")


def test_asked_for_a_figure_but_no_tool_made_one():
    llm = _ScriptLLM([
        ChatMessage(role="assistant", content="Done."),
        ChatMessage(role="assistant", content="Done, really."),
    ])
    list(Agent(llm=llm).iter_chat("plot the shot gather"))
    assert len(llm.seen) == 2 and "no tool made one" in llm.seen[1][-1]


def test_figure_a_tool_made_passes(tmp_path):
    pytest.importorskip("matplotlib")
    png = tmp_path / "w.png"

    class _Plot(BaseLLM):
        n = 0
        _model_id = "fake"

        def chat(self, messages, tools=None, **kw):
            self.n += 1
            if self.n == 1:
                return ChatMessage(role="assistant", content="Plotting.", tool_calls=[
                    ToolCall(id="c1", name="plot_wavelet", arguments={"fm": 10.0, "out_path": str(png)})])
            return ChatMessage(role="assistant", content="Here it is: w.png")

    llm = _Plot()
    list(Agent(llm=llm).iter_chat("plot the wavelet"))
    assert png.is_file() and llm.n == 2                     # no nudge


# --------------------------------------------------------------- LLM server

class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, json.dumps(body)

    def json(self):
        return self._body


def test_server_error_carries_the_servers_reason():
    from sweepx.agent.llm.vllm_backend import _server_error

    err = _server_error("http://localhost:8000/v1", _Resp(400, {"error": {"message": (
        "This model's maximum context length is 8192 tokens. However, your request has "
        "12275 input tokens.")}}))
    assert "12275" in str(err) and "--max-model-len 32768" in str(err)


def test_served_model_is_asked(monkeypatch):
    import httpx

    from sweepx.agent.llm import vllm_backend as vb

    monkeypatch.setattr(httpx, "get", lambda url, timeout=0: _Resp(200, {"data": [{"id": "Qwen/Qwen2.5-14B-Instruct"}]}))
    assert vb.served_model("http://localhost:8000/v1") == "Qwen/Qwen2.5-14B-Instruct"
    assert vb.served_model("http://localhost:11434/v1") is None


def test_serve_llm_defaults_to_a_32k_context():
    from sweepx.agent.cli import _build_argparser

    assert _build_argparser().parse_args(["serve-llm"]).max_model_len == 32768


# --------------------------------------------------------------- SEG-Y geometry

def test_inspect_segy_reports_the_geometry(tmp_path):
    segyio = pytest.importorskip("segyio")
    path = tmp_path / "obs.segy"
    spec = segyio.spec()
    spec.format, spec.samples, spec.tracecount = 5, list(range(50)), 6
    with segyio.create(str(path), spec) as f:
        f.bin[segyio.BinField.Interval] = 2000
        for i in range(6):
            shot, rec = divmod(i, 3)
            f.header[i] = {segyio.TraceField.SourceX: 1000 * shot,
                           segyio.TraceField.GroupX: 1000 * shot + 250 * (rec + 1),
                           segyio.TraceField.SourceGroupScalar: -10}
            f.trace[i] = np.zeros(50, dtype=np.float32)
    r = registry.get("inspect_file").invoke({"path": str(path)})
    assert r.ok, r.error
    v = r.value
    assert v["n_shots"] == 2 and v["receivers_per_shot"] == [3, 3]
    assert v["source_x_range"] == [0.0, 100.0] and v["offset_range"] == [25.0, 75.0]
    assert "--geometry" not in v["hint"]


# --------------------------------------------------------------- FWI defaults

def test_fwi_builder_defaults_every_shot_and_a_vp_band(tmp_path):
    pytest.importorskip("sweepx.tasks")
    init = np.full((40, 96), 1500.0, dtype=np.float32)
    init[20:] = 2000.0
    np.save(tmp_path / "init.npy", init)
    np.save(tmp_path / "true.npy", init + 100.0)
    b = registry.get("build_fwi_spec").invoke({
        "init_model_path": str(tmp_path / "init.npy"), "synthetic_true_vp_path": str(tmp_path / "true.npy"),
        "dh": 10.0, "dt": 1e-3, "record_length_s": 0.3, "source_step": 20,
        "output_dir": str(tmp_path / "runs"),
    })
    assert b.ok and "error" not in b.value, b.value
    spec = b.value["spec"]
    assert spec["batchsize"] == 5                                   # x = 0, 20, 40, 60, 80
    assert spec["model_bounds"]["vp"] == {"enabled": True, "min": 1125.0, "max": 3500.0}


def test_auto_device_picks_what_the_machine_has():
    torch = pytest.importorskip("torch")
    from sweepx.agent.tools.build_forward import _auto_device

    want = "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    assert _auto_device("auto") == want
    assert _auto_device("cpu") == "cpu"                # an explicit choice is kept


def test_auto_impl_takes_the_cuda_kernels_only_where_they_run():
    pytest.importorskip("sweep")
    from sweepx.agent.tools.build_forward import _auto_impl

    assert _auto_impl("auto", "cpu", "Acoustic") == "eager"
    assert _auto_impl("auto", "cuda", "Acoustic", snapshots=True) == "eager"   # snapshots are eager-only
    assert _auto_impl("auto", "cuda", "ElasticCurvilinear") == "eager"         # no fused kernel
    assert _auto_impl("eager", "cuda", "Acoustic") == "eager"                  # explicit choice kept
