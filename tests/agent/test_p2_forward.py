"""P2 smoke test: drive build_forward_spec → run_task → read_status with FakeLLM.

Two layers:
  - ``test_build_forward_spec_offline``: pure schema check, no torch / sweep run.
  - ``test_forward_endtoend_cpu``: tiny eager-backend run on CPU; ~1-5 seconds.

Requires ``sweep_tasks`` to be importable (it is in the ifwitorch env). The
end-to-end test will be skipped automatically if ``sweep`` itself is not
available.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pytest

from sweep_agent.agent import Agent
from sweep_agent.llm.base import BaseLLM, ChatMessage, ToolCall
from sweep_agent.tools import registry


class FakeLLM(BaseLLM):
    def __init__(self, scripted: list[ChatMessage]) -> None:
        self._queue = list(scripted)
        self.calls: list[list[ChatMessage]] = []
        self._model_id = "fake"

    def chat(
        self,
        messages: list[ChatMessage],
        tools: Iterable[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        self.calls.append(list(messages))
        if not self._queue:
            raise AssertionError("FakeLLM ran out of scripted replies")
        return self._queue.pop(0)


# ---------------------------------------------------------------------------
# Offline: just validate that build_forward_spec produces a parseable ForwardSpec.
# ---------------------------------------------------------------------------

def test_build_forward_spec_offline(tmp_path):
    pytest.importorskip("sweep_tasks")
    vp_path = tmp_path / "vp_tiny.npy"
    np.save(vp_path, np.full((32, 40), 1800.0, dtype=np.float32))
    out_root = tmp_path / "runs"

    tool = registry.get("build_forward_spec")
    assert tool is not None
    result = tool.invoke({
        "vp_path": str(vp_path),
        "dh": 12.5,
        "dt": 0.001,
        "nt": 100,
        "fm": 15.0,
        "source_step": 20,
        "source_depth": 2,
        "receiver_step": 5,
        "receiver_depth": 8,
        "backend_impl": "eager",
        "output_dir": str(out_root),
        "task_id": "p2_offline",
        "spatial_order": 4,
        "abcn": 10,
        "device": "cpu",
    })
    assert result.ok, result.error
    payload = result.value
    assert payload["spec"]["task_type"] == "forward"
    assert payload["spec"]["physics"]["equation"] == "Acoustic"
    assert payload["spec"]["geometry"]["sources"]["step"] == 20
    yaml_path = Path(payload["yaml_path"])
    assert yaml_path.is_file()
    assert "Acoustic" in yaml_path.read_text()


# ---------------------------------------------------------------------------
# End-to-end: actually run the tiny forward via TaskRunner.
# ---------------------------------------------------------------------------

@pytest.mark.timeout(120)
def test_forward_endtoend_cpu(tmp_path):
    sweep = pytest.importorskip("sweep")
    pytest.importorskip("sweep_tasks")

    vp_path = tmp_path / "vp_tiny.npy"
    np.save(vp_path, np.full((32, 40), 1800.0, dtype=np.float32))
    out_root = tmp_path / "runs"

    scripted = [
        ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id="c1", name="inspect_file", arguments={"path": str(vp_path)})],
        ),
        ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id="c2", name="build_forward_spec", arguments={
                "vp_path": str(vp_path),
                "dh": 12.5,
                "dt": 0.001,
                "nt": 80,
                "fm": 15.0,
                "source_step": 20,
                "source_depth": 2,
                "receiver_step": 5,
                "receiver_depth": 8,
                "backend_impl": "eager",
                "output_dir": str(out_root),
                "task_id": "p2_e2e",
                "spatial_order": 4,
                "abcn": 10,
                "device": "cpu",
                # build_forward_spec defaults use_compile=False, so no extra
                # needed here. Passing it explicitly keeps the intent obvious.
                "use_compile": False,
            })],
        ),
        # The agent loop won't know the yaml_path until it inspects the
        # previous tool_result; for the FakeLLM script we hard-code the path
        # that build_forward_spec deterministically returns.
        ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id="c3", name="run_task", arguments={
                "yaml_path": str((out_root / "p2_e2e.yaml").resolve()),
                "timeout_s": 60,
            })],
        ),
        # Same trick for the read_status step — TaskRunner namespaces the dir
        # with a timestamped task_id; we read status.json via the directory.
        # Plug in a placeholder; the agent rewrites the arg from the prior obs.
        ChatMessage(
            role="assistant",
            content="forward done — synthetic shot in runs/<task_id>/output/.",
        ),
    ]
    fake = FakeLLM(scripted)
    agent = Agent(llm=fake)

    final = agent.chat("跑一下 vp_tiny.npy 的 forward")
    assert "forward done" in (final.content or "")

    # Verify the run actually produced a task_dir with a success status.
    run_tool_msg = next(
        m for m in agent.history
        if m.role == "tool" and m.name == "run_task"
    )
    run_payload = json.loads(run_tool_msg.content)
    assert run_payload.get("state") == "success", run_payload
    task_dir = Path(run_payload["task_dir"])
    assert task_dir.is_dir()
    status_path = task_dir / "status.json"
    assert status_path.is_file()
    status = json.loads(status_path.read_text())
    assert status["state"] == "success"
    assert status["artifacts"], "forward run should produce at least one artifact"
