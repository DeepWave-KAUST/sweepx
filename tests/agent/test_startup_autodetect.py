"""Zero-config startup: endpoint auto-detection, 7B default, Ollama auto-pull.

These are the guarantees that let `sweep-agent chat` work with no env vars and
no manual `ollama pull`. All network / subprocess calls are mocked, so the suite
needs neither a running server nor a model download.
"""

from __future__ import annotations

import shutil
import subprocess

import httpx
import pytest

from sweep_agent.llm import vllm_backend as vb


def test_is_ollama_and_default_model_for_backend():
    assert vb.is_ollama("http://localhost:11434/v1")
    assert not vb.is_ollama("http://localhost:8000/v1")
    assert vb.default_model_for("http://localhost:11434/v1") == "qwen2.5:7b"
    assert vb.default_model_for("http://localhost:8000/v1") == "Qwen/Qwen2.5-7B-Instruct"


def test_detect_endpoint_prefers_ollama_then_vllm_then_fallback(monkeypatch):
    # only vLLM :8000 answers -> pick it
    monkeypatch.setattr(vb, "_endpoint_alive", lambda u, timeout=0.6: u == vb.VLLM_URLS[0])
    assert vb.detect_endpoint() == vb.VLLM_URLS[0]
    # Ollama answers -> preferred over vLLM
    monkeypatch.setattr(vb, "_endpoint_alive", lambda u, timeout=0.6: True)
    assert vb.detect_endpoint() == vb.OLLAMA_URL
    # nothing answers -> Ollama fallback (so an auto-pull can bring it up)
    monkeypatch.setattr(vb, "_endpoint_alive", lambda u, timeout=0.6: False)
    assert vb.detect_endpoint() == vb.OLLAMA_URL


def test_backend_autodetects_url_and_model_when_unset(monkeypatch):
    monkeypatch.delenv("SWEEP_AGENT_LLM_URL", raising=False)
    monkeypatch.delenv("SWEEP_AGENT_LLM_MODEL", raising=False)
    monkeypatch.setattr(vb, "detect_endpoint", lambda: vb.OLLAMA_URL)
    b = vb.VLLMBackend()
    try:
        assert b.url == "http://localhost:11434/v1"
        assert b.model_id == "qwen2.5:7b"
    finally:
        b.close()


def test_explicit_arg_and_env_beat_autodetect(monkeypatch):
    def _boom() -> str:
        raise AssertionError("detect_endpoint should not be called when a URL is given")

    monkeypatch.setattr(vb, "detect_endpoint", _boom)
    monkeypatch.delenv("SWEEP_AGENT_LLM_URL", raising=False)
    monkeypatch.delenv("SWEEP_AGENT_LLM_MODEL", raising=False)
    b = vb.VLLMBackend(url="http://host:9999/v1", model="qwen2.5-14b-instruct")
    try:
        assert b.url == "http://host:9999/v1"
        assert b.model_id == "Qwen/Qwen2.5-14B-Instruct"  # alias resolved
    finally:
        b.close()

    monkeypatch.setenv("SWEEP_AGENT_LLM_URL", "http://envhost:1/v1")
    monkeypatch.setenv("SWEEP_AGENT_LLM_MODEL", "qwen2.5-7b-instruct")
    b2 = vb.VLLMBackend()
    try:
        assert b2.url == "http://envhost:1/v1"
        assert b2.model_id == "Qwen/Qwen2.5-7B-Instruct"
    finally:
        b2.close()


class _Tags:
    def __init__(self, names):
        self._names = names

    def json(self):
        return {"models": [{"name": n} for n in self._names]}


def test_ensure_ollama_model_pulls_when_missing(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Tags(["mistral:latest"]))
    monkeypatch.setattr(shutil, "which", lambda x: "/usr/bin/ollama")
    seen = {}
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: seen.update(cmd=cmd))
    vb.ensure_ollama_model("http://localhost:11434/v1", "qwen2.5:7b")
    assert seen["cmd"] == ["ollama", "pull", "qwen2.5:7b"]


def test_ensure_ollama_model_skips_when_present(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Tags(["qwen2.5:7b"]))
    monkeypatch.setattr(shutil, "which", lambda x: "/usr/bin/ollama")
    n = {"calls": 0}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: n.update(calls=n["calls"] + 1))
    vb.ensure_ollama_model("http://localhost:11434/v1", "qwen2.5:7b")
    assert n["calls"] == 0


def test_ensure_ollama_model_is_noop_for_vllm(monkeypatch):
    n = {"calls": 0}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: n.update(calls=n["calls"] + 1))
    vb.ensure_ollama_model("http://localhost:8000/v1", "Qwen/Qwen2.5-7B-Instruct")
    assert n["calls"] == 0


def test_backend_hint_none_when_endpoint_alive(monkeypatch):
    monkeypatch.setattr(vb, "_endpoint_alive", lambda u, timeout=0.6: True)
    assert vb.backend_hint("http://localhost:11434/v1") is None


def test_backend_hint_ollama_installed_but_not_running(monkeypatch):
    monkeypatch.setattr(vb, "_endpoint_alive", lambda u, timeout=0.6: False)
    monkeypatch.setattr(shutil, "which", lambda x: "/usr/local/bin/ollama")
    msg = vb.backend_hint("http://localhost:11434/v1")
    assert msg is not None and "ollama serve" in msg


def test_backend_hint_ollama_not_installed_points_at_installer(monkeypatch):
    monkeypatch.setattr(vb, "_endpoint_alive", lambda u, timeout=0.6: False)
    monkeypatch.setattr(shutil, "which", lambda x: None)
    msg = vb.backend_hint("http://localhost:11434/v1")
    assert msg is not None and "install.sh" in msg and "Homebrew" in msg


def test_backend_hint_vllm_down_names_the_url(monkeypatch):
    monkeypatch.setattr(vb, "_endpoint_alive", lambda u, timeout=0.6: False)
    msg = vb.backend_hint("http://localhost:8000/v1")
    assert msg is not None and "8000" in msg
