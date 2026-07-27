"""Per-query tool subsetting.

Exposing all ~30 tools to a small model on every request bloats the prompt and
makes tool selection error-prone. ``select_tool_names`` keyword-routes the user's
message to a small relevant subset (core backbone + matched task groups), so the
model sees ~8-16 tools instead of 30. Conservative by design: unmatched queries
fall back to a broad default, and the Agent additionally keeps any tools already
used earlier in the conversation.
"""

from __future__ import annotations

# Always available — the inspect / run / discover backbone.
CORE = [
    "inspect_file", "run_task", "read_status", "list_artifacts",
    "list_equations", "describe_task_schema", "build_spec", "run_forward_sweep",
]

# (trigger keywords, tool names) — a query adds every group whose keywords hit.
GROUPS: list[tuple[tuple[str, ...], list[str]]] = [
    (("forward", "正演", "shot gather", "shot-gather", "炮记录", "合成记录", "record", "gather", "synthetic record"),
     ["build_forward_spec", "run_forward_and_plot", "plot_shot_gather", "run_forward_sweep"]),
    (("wavefield", "波场", "snapshot", "快照", "animate", "animation", "动画", "movie", "gif", "propagat", "传播", "p wave", "s wave", "p波", "s波", "p-wave", "s-wave"),
     ["build_wavefield_spec", "animate_wavefield", "make_wavefield_gif", "plot_wavefield"]),
    (("anisotrop", "各向异性", "vti", "tti", "wavefront", "波前", "compare equation", "对比方程", "elastic", "弹性"),
     ["compare_equation_wavefields", "compare_wavefields", "animate_wavefield", "build_wavefield_spec"]),
    (("fwi", "反演", "inversion", "invert", "full waveform", "全波形", "multiscale", "多尺度", "converg", "收敛", "loss"),
     ["build_fwi_spec", "run_fwi", "run_multiscale_fwi", "plot_model", "plot_convergence", "animate_fwi_evolution"]),
    (("model", "模型", "generate", "make a", "create", "生成", "造", "velocity", "速度", "slice", "切片", "profile", "剖面"),
     ["make_synthetic_model", "plot_velocity_model", "plot_velocity_slice"]),
    (("segy", "sgy", "observed", "观测", "real data", "真实数据", "field data", "misfit", "残差", "compare data", "对比观测"),
     ["plot_segy", "plot_observed_data", "compare_shot_gathers"]),
    (("parameter", "参数", "cfl", "stable", "稳定", "dispersion", "频散", "wavelet", "子波", "spectrum", "频谱", "frequency content"),
     ["check_parameters", "plot_wavelet"]),
    (("rtm", "lsrtm", "migration", "偏移", "成像"),
     ["build_spec", "describe_task_schema"]),
]

# When nothing matches, a broad-but-bounded default (the common tasks).
_FALLBACK = [
    "build_forward_spec", "run_forward_and_plot", "plot_shot_gather",
    "build_wavefield_spec", "animate_wavefield", "make_synthetic_model",
    "plot_velocity_model", "run_fwi", "check_parameters", "run_forward_sweep",
]


def select_tool_names(message: str, available: list[str] | None = None) -> list[str]:
    """Return the subset of tool names relevant to ``message`` (core + matched
    groups, or a broad fallback). Filtered to ``available`` if given."""
    text = (message or "").lower()
    picked: list[str] = list(CORE)
    matched = False
    for keywords, tools in GROUPS:
        if any(k in text for k in keywords):
            matched = True
            picked.extend(tools)
    if not matched:
        picked.extend(_FALLBACK)
    # de-dup, preserve order
    seen: set[str] = set()
    out = [t for t in picked if not (t in seen or seen.add(t))]
    if available is not None:
        avail = set(available)
        out = [t for t in out if t in avail]
    return out
