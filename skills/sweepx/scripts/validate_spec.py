#!/usr/bin/env python
"""Validate a sweepx task YAML without running it.

Uses sweepx's own loader (``sweepx.tasks.load_task_from_dict``, the call
``sweep-tasks run`` makes, including ``--override`` handling), then prints the
resolved spec -- every default filled in -- or the validation errors.

    python validate_spec.py task.yaml
    python validate_spec.py task.yaml --override epochs=20 --override loss.kind=l1
    python validate_spec.py task.yaml --preflight     # also load inputs and lint
    python validate_spec.py task.yaml --quiet         # only OK / errors

``--preflight`` goes further than the schema: it loads the models and geometry
the way the runner does (``sweepx.tasks.introspect``) and reports the grid,
shot/receiver counts, the Courant number with the runner's thresholds, points
per wavelength, an ``obs.npy_path`` layout check, and settings that load fine
but are ignored at run time. It never builds a solver.

Exit status: 0 valid (warnings allowed), 1 invalid or a preflight error, 2 usage.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import yaml


def _fmt_loc(loc) -> str:
    # Drop pydantic's union/validator wrapper names, keep the field path.
    parts = [str(p) for p in loc if "[" not in str(p)]
    return ".".join(parts) or "<root>"


def _print_errors(err) -> None:
    from pydantic import ValidationError

    if isinstance(err, ValidationError):
        errs = err.errors()
        print(f"INVALID: {len(errs)} error(s)")
        for e in errs:
            msg = e.get("msg", "")
            if msg.startswith("Value error, "):
                msg = msg[len("Value error, "):]
            print(f"  - {_fmt_loc(e.get('loc', ()))}: {msg}")
    else:
        print(f"INVALID: {type(err).__name__}: {err}")


def load(path: Path, overrides: list[str]):
    from sweepx.tasks import load_task_from_dict
    from sweepx.tasks.runtime.config import apply_overrides

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    if overrides:
        apply_overrides(raw, overrides)
    return load_task_from_dict(raw, base_dir=path.resolve().parent)


# ---------------------------------------------------------------- preflight --

def preflight(spec) -> tuple[list[str], list[str], list[str]]:
    """(info, warnings, errors) for a validated spec."""
    import numpy as np

    from sweepx.tasks import introspect as ins
    from sweepx.tasks.schemas import (ForwardSpec, FWISpec, LossSpec, LSRTMSpec,
                                      RTMSpec, WavefieldSpec)

    info: list[str] = []
    warn: list[str] = []
    errs: list[str] = []
    t = spec.task_type
    if spec.device not in ("auto", "cpu", "cuda", "mps") and not str(spec.device).startswith(("cuda:", "mps:")):
        warn.append(f"device={spec.device!r}: use 'auto', 'cpu' or 'cuda:N' (a bare index is not a torch device).")
    if not isinstance(spec, (ForwardSpec, WavefieldSpec, FWISpec, RTMSpec, LSRTMSpec)):
        return info, warn, errs

    freqsel = (isinstance(spec, FWISpec) and spec.source_encoding is not None
               and spec.source_encoding.enabled
               and spec.source_encoding.mode == "frequency_selection")

    # grid + models
    try:
        shape = ins.grid_shape(spec)
        info.append(f"grid shape {tuple(shape)} (model axes, z first), dh={spec.grid.dh} m")
    except Exception as e:  # noqa: BLE001
        errs.append(f"grid shape / model load failed: {type(e).__name__}: {e}")
        return info, warn, errs

    refs = []
    if isinstance(spec, (ForwardSpec, WavefieldSpec)):
        refs = list(spec.models)
    elif isinstance(spec, FWISpec):
        refs = [spec.init_model] if spec.init_model is not None else list(spec.init_models)
        obs = spec.obs
        if obs is not None and obs.synthetic_from is not None:
            refs.append(obs.synthetic_from)
        if obs is not None and obs.synthetic_from_models:
            refs.extend(obs.synthetic_from_models)
    elif isinstance(spec, RTMSpec):
        refs = [spec.velocity_model]
    elif isinstance(spec, LSRTMSpec):
        refs = [spec.background_model, spec.true_model]

    from sweepx.tasks._helpers.model import _dataset_payload, _model_array
    vmin = vmax = None
    for ref in refs:
        try:
            arr = _model_array(ref, mmap=True)
        except Exception as e:  # noqa: BLE001
            errs.append(f"model {ref.name!r} failed to load: {type(e).__name__}: {e}")
            continue
        a_min, a_max = float(np.min(arr)), float(np.max(arr))
        info.append(f"model {ref.name!r}: shape {tuple(arr.shape)}, range [{a_min:g}, {a_max:g}]")
        if not np.all(np.isfinite(arr)):
            errs.append(f"model {ref.name!r} has NaN/Inf")
        if tuple(arr.shape) != tuple(shape) and spec.grid.shape is None:
            errs.append(f"model {ref.name!r} shape {tuple(arr.shape)} != grid {tuple(shape)}")
        if ref.name in ("vp", "vp0", "vh", "vv"):
            vmin = a_min if vmin is None else min(vmin, a_min)
            vmax = a_max if vmax is None else max(vmax, a_max)
        if ref.dataset is not None:
            try:
                ddh = _dataset_payload(ref).get("dh")
                if ddh is not None:
                    ddh = float(ddh[0] if isinstance(ddh, (tuple, list)) else ddh)
                    if not math.isclose(ddh, float(spec.grid.dh), rel_tol=1e-6):
                        warn.append(f"model {ref.name!r} comes from dataset {ref.dataset!r} "
                                    f"at dh={ddh:g} m but grid.dh={spec.grid.dh:g}: grid.dh is "
                                    "NOT taken from the dataset; set it to match.")
            except Exception:  # noqa: BLE001
                pass

    # time axis, CFL, dispersion
    if not freqsel:
        dt, nt = ins.time_axis(spec)
        info.append(f"time: dt={dt:g} s, nt={nt}, record {dt * nt:g} s")
        if vmax is not None:
            c = ins.cfl_number(vmax, spec.grid.dh, dt)
            msg = f"Courant vmax*dt/dh = {c:.3f} (runner warns > {ins.CFL_WARN}, refuses > {ins.CFL_RAISE})"
            # Measured: 2-D Acoustic, spatial_order 8 (impl c and eager) is stable at 0.50 and
            # blows up at 0.56, far below the runner's gate. 3-D scaled by sqrt(2/3) (estimate).
            safe = 0.5 * math.sqrt(2.0 / len(shape))
            if c > ins.CFL_RAISE:
                errs.append(msg)
            elif c > safe:
                warn.append(msg + f"; above ~{safe:.2f}, where an 8th-order run is likely to blow up "
                            "(2-D Acoustic measured: stable at 0.50, NaN at 0.56)")
            else:
                info.append(msg)
        fm = getattr(spec.wavelet, "fm", None) if spec.wavelet is not None else None
        if fm and vmin:
            ppw = vmin / (2.5 * fm * spec.grid.dh)
            msg = (f"points per wavelength at 2.5*fm={2.5 * fm:g} Hz and vmin={vmin:g}: "
                   f"{ppw:.1f} (aim for >= 5)")
            (warn if ppw < 5 else info).append(msg)

    # stages: the grid each stage runs on (a stage inherits the previous one's)
    stages = getattr(spec, "stages", None)
    if stages and not freqsel and isinstance(spec, FWISpec):
        from sweepx.tasks._helpers.stages import _stage_grid_chain
        dt0, nt0 = ins.time_axis(spec)
        chain = _stage_grid_chain(stages, float(spec.grid.dh), dt0, nt0)
        for i, (st, g) in enumerate(zip(stages, chain)):
            bp = f", bandpass {st.bandpass.lo_hz:g}-{st.bandpass.hi_hz:g} Hz" if st.bandpass else ", broadband"
            info.append(f"stage {i}: {st.epochs} epochs, dh={g['dh']:g}, dt={g['dt']:g}, nt={g['nt']}{bp}")
            if vmax is not None and ins.cfl_number(vmax, g["dh"], g["dt"]) > 0.5 * math.sqrt(2.0 / len(shape)):
                warn.append(f"stage {i}: Courant {ins.cfl_number(vmax, g['dh'], g['dt']):.3f} on its grid")
        if not math.isclose(chain[-1]["dh"], float(spec.grid.dh)):
            warn.append(f"the last stage runs at dh={chain[-1]['dh']:g}, not grid.dh={spec.grid.dh:g} "
                        "(stages inherit the previous stage's dh_m): inverted_*.npy will be on that grid. "
                        "Set dh_m (and dt_s) on the stage that should return to the native grid.")
    if freqsel:
        # frequency selection reads each stage's dt_s / dh_m, else time.dt / grid.dh (no inheritance)
        def _dt(st):
            return float(st.dt_s) if st is not None and st.dt_s else float(spec.time.dt)
        st0 = stages[0] if stages else None
        fs_blocks = [("source_encoding.frequency", spec.source_encoding.frequency, _dt(st0))]
        fs_blocks += [(f"stages[{i}].frequency", st.frequency, _dt(st))
                      for i, st in enumerate(stages or ()) if st.frequency is not None]
        for where, f, dt in fs_blocks:
            bins = f.k_hi - f.k_lo + 1
            info.append(f"{where}: comb {f.k_lo / (f.probe_samples * dt):g}-"
                        f"{f.k_hi / (f.probe_samples * dt):g} Hz at dt={dt:g}, {bins} bins, "
                        f"record {f.steady_samples + f.slack_samples + f.probe_samples} steps")
            if f.k_hi < f.k_lo:
                errs.append(f"{where}: k_hi < k_lo")
            if f.random_batch is not None and f.random_batch > bins:
                errs.append(f"{where}: random_batch={f.random_batch} > {bins} comb bins")
            if f.synthesize_from_true and math.ceil(f.synth_n_nodes / f.n_pools) > bins:
                errs.append(f"{where}: {math.ceil(f.synth_n_nodes / f.n_pools)} nodes per pool > {bins} bins")

    # geometry
    if not freqsel and not (isinstance(spec, FWISpec) and spec.obs is not None
                            and spec.obs.plan is not None):
        try:
            src, rec = ins.build_geometry_arrays(spec)
            info.append(f"geometry: {src.shape[0]} shots, {rec.shape[1]} receivers/shot")
            # points are (x, z) / (x, y, z); model axes are (z, x) / (z, y, x)
            lim = tuple(reversed(shape))
            for nm, pts in (("source", src.reshape(-1, src.shape[-1])),
                            ("receiver", rec.reshape(-1, rec.shape[-1]))):
                bad = np.any((pts < 0) | (pts >= np.array(lim)), axis=1)
                if bad.any():
                    errs.append(f"{int(bad.sum())} {nm} point(s) outside the grid "
                                f"(points are (x, z) / (x, y, z) indices < {lim})")
            if isinstance(spec, FWISpec) and spec.obs is not None and spec.obs.npy_path is not None:
                obs = np.load(spec.obs.npy_path, mmap_mode="r")
                want = (src.shape[0], ins.time_axis(spec)[1], rec.shape[1])
                if obs.ndim != 4 or tuple(obs.shape[:3]) != want:
                    errs.append(f"obs.npy_path shape {obs.shape}: expected (nshots, nt, nrec, 1) = "
                                f"{(*want, 1)} (the layout of a forward run's record.npy). "
                                "The runner checks axis 0 only and would broadcast a wrong layout.")
        except Exception as e:  # noqa: BLE001
            errs.append(f"geometry failed: {type(e).__name__}: {e}")

    # settings that load but are not used
    loss = getattr(spec, "loss", None)
    if isinstance(spec, RTMSpec):
        if "kind" in loss.model_fields_set:
            warn.append("rtm: loss.kind is ignored; the image uses imaging.loss_kind "
                        f"(= {spec.imaging.loss_kind!r}).")
    elif isinstance(loss, LossSpec) and not freqsel:
        per_kind = {"huber_delta": "huber", "cc_max_lag_samples": "cc_traveltime",
                    "cc_beta": "cc_traveltime", "trace_cosine_demean": "trace_cosine",
                    "trace_cosine_eps": "trace_cosine"}
        for f, k in per_kind.items():
            if loss.kind != k and getattr(loss, f) != LossSpec.model_fields[f].default:
                warn.append(f"loss.{f} is only read with loss.kind={k!r} (kind is {loss.kind!r}).")
        if loss.kind == "trace_cosine" and loss.data_mask_path and not loss.data_mask_window_mode:
            warn.append("loss.kind=trace_cosine with data_mask_path but data_mask_window_mode=false: "
                        "the mask only re-weights whole traces; set data_mask_window_mode: true to window.")
    if isinstance(spec, FWISpec) and spec.reparam is not None:
        info.append(f"reparam: the network trains at reparam.lr={spec.reparam.lr:g}; "
                    "optimizer.lr is not used.")
    obs = getattr(spec, "obs", None)
    recorded = (obs is not None and obs.synthetic_from is None and obs.synthetic_from_models is None) or (
        freqsel and not spec.source_encoding.frequency.synthesize_from_true)
    if isinstance(spec, (FWISpec, RTMSpec)) and recorded and getattr(spec, "water", None) \
            and not spec.physics.free_surface:
        warn.append("recorded data with a water: block but physics.free_surface=false: marine "
                    "recordings contain the sea-surface ghosts and multiples; set free_surface: true.")
    for gname in ("geometry",):
        g = getattr(spec, gname, None)
        bm = getattr(g, "byte_map", None)
        obs_segy = getattr(getattr(spec, "obs", None), "segy", None)
        for where, m in ((f"{gname}.byte_map", bm), ("obs.segy.byte_map", getattr(obs_segy, "byte_map", None))):
            if not m:
                continue
            from sweepx.io.segy_index import SEGY_REV1_BYTES
            unknown = sorted(set(m) - set(SEGY_REV1_BYTES))
            if unknown:
                warn.append(f"{where}: keys {unknown} are not read (known: {sorted(SEGY_REV1_BYTES)}).")
            odd = {k: v for k, v in m.items() if k in SEGY_REV1_BYTES and v % 2 == 1}
            if odd:
                warn.append(f"{where}: {odd} look 1-based; byte_map takes 0-based offsets "
                            "(sx is 72, not 73).")
    return info, warn, errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_file", type=Path)
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted-key override, as `sweep-tasks run --override`")
    ap.add_argument("--preflight", action="store_true",
                    help="also load models/geometry and lint (no solver is built)")
    ap.add_argument("-q", "--quiet", action="store_true", help="do not print the resolved spec")
    args = ap.parse_args(argv)
    if not args.task_file.is_file():
        print(f"error: {args.task_file} not found")
        return 2
    try:
        spec = load(args.task_file, args.override)
    except Exception as err:  # noqa: BLE001
        _print_errors(err)
        return 1

    out_dir = Path(spec.output_dir)
    run = spec.task_id or f"{spec.task_type}-<timestamp>"
    print(f"OK: task_type={spec.task_type}; run dir will be {out_dir / run}")
    rc = 0
    if args.preflight:
        info, warn, errs = preflight(spec)
        for m in info:
            print(f"  info: {m}")
        for m in warn:
            print(f"  WARNING: {m}")
        for m in errs:
            print(f"  ERROR: {m}")
        rc = 1 if errs else 0
    if not args.quiet:
        print("--- resolved spec ---")
        print(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False), end="")
    return rc


if __name__ == "__main__":
    sys.exit(main())
