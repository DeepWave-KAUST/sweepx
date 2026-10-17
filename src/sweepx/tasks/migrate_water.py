"""``sweep-tasks migrate-water``: rewrite retired water keys as one ``water:`` block.

The water layer used to be configured five unrelated ways --
``freeze_top_n_rows``, ``freeze_water_layer``, ``reparam.mask_water_layer`` /
``water_vp_m_s`` / ``seabed_depth_path`` / ``water_reset_each_stage``, and the
frequency-selection path's silent pin of every starting-model cell at exactly
1500 m/s. This rewrites a task YAML into the single top-level ``water:`` block
(:class:`sweep_tasks.schemas.WaterSpec`) where the translation is unambiguous,
and refuses -- saying why -- where it is not. It never guesses the water from
velocities: where the old run's water was a velocity rule, the user provides
the file.

The YAML is edited line by line, so comments elsewhere survive; the result is
validated against the schema before anything is written, and printed as a
diff.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

_TOP_KEYS = ("freeze_top_n_rows", "freeze_water_layer")
_REPARAM_KEYS = ("mask_water_layer", "water_vp_m_s", "seabed_depth_path",
                 "water_reset_each_stage")
_DEFAULTS = {"freeze_top_n_rows": 0, "mask_water_layer": False,
             "water_vp_m_s": 1500.0, "seabed_depth_path": None,
             "water_reset_each_stage": False}
_OLD_PIN_VP = 1500.0


@dataclass
class Migration:
    """The outcome for one YAML: the new text (``None`` when refused)."""

    old_text: str
    new_text: str | None = None
    water: dict | None = None
    notes: list[str] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)
    files: dict = field(default_factory=dict)     # Path -> np.ndarray to save

    @property
    def changed(self) -> bool:
        return self.new_text is not None and self.new_text != self.old_text

    def diff(self, old_name: str = "old", new_name: str = "new") -> str:
        if self.new_text is None:
            return ""
        return "".join(difflib.unified_diff(
            self.old_text.splitlines(keepends=True),
            self.new_text.splitlines(keepends=True),
            fromfile=old_name, tofile=new_name))


def _is_default(key, value) -> bool:
    if key == "freeze_water_layer":
        return value is None or (isinstance(value, dict)
                                 and not value.get("enabled", False))
    d = _DEFAULTS[key]
    if value is None:
        return True
    if isinstance(d, bool):
        return isinstance(value, (bool, int)) and not value
    try:
        return float(value) == float(d)
    except (TypeError, ValueError):
        return False


def _abs(p, base: Path) -> Path:
    import os

    p = Path(os.path.expandvars(str(p)))
    return p if p.is_absolute() else (base / p).resolve()


# ---------------------------------------------------------------- model --

def _init_ref(doc: dict) -> dict | None:
    if doc.get("task_type") == "lsrtm":
        return doc.get("background_model")
    if doc.get("init_model") is not None:
        return doc["init_model"]
    refs = doc.get("init_models") or []
    return next((r for r in refs if r.get("name") == "vp"), refs[0] if refs else None)


def _native_shape(doc: dict, base: Path) -> tuple | None:
    """The starting model's grid shape, without loading big arrays."""
    if (doc.get("grid") or {}).get("shape"):
        return tuple(int(v) for v in doc["grid"]["shape"])
    ref = _init_ref(doc)
    if ref is None:
        return None
    if ref.get("shape") and (ref.get("constant") is not None
                             or ref.get("linear_gradient") is not None):
        return tuple(int(v) for v in ref["shape"])
    if ref.get("path"):
        p = _abs(ref["path"], base)
        if p.is_file():
            return tuple(np.load(p, mmap_mode="r").shape)
    return None


def _init_model(doc: dict, base: Path):
    """The starting model as an array, or ``None`` if it cannot be read here
    (a ``dataset:`` source is not resolved -- that can mean a download)."""
    from sweep_tasks._helpers.model import _model_array
    from sweep_tasks.schemas import ModelRef

    ref = _init_ref(doc)
    if ref is None or ref.get("dataset") is not None:
        return None
    rr = dict(ref)
    if rr.get("path"):
        rr["path"] = str(_abs(rr["path"], base))
        if not Path(rr["path"]).is_file():
            return None
    try:
        mref = ModelRef(**rr)
    except Exception:  # noqa: BLE001
        return None
    return _model_array(mref, mmap=not rr.get("smooth_sigma_cells"))


def _one_value(values) -> float | None:
    """The single value of ``values``, written the shortest way that reads
    back to the same float32 (1485.6, not 1485.5999755859375), else None."""
    v = np.unique(np.asarray(values))
    return float(str(np.float32(v[0]))) if v.size == 1 else None


def _is_freqsel(doc: dict) -> bool:
    se = doc.get("source_encoding") or {}
    return (doc.get("task_type") == "fwi" and bool(se.get("enabled"))
            and se.get("mode") == "frequency_selection")


# --------------------------------------------------------- line editing --

def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_body(line: str) -> bool:
    s = line.strip()
    return bool(s) and not s.startswith("#")


def _block_end(lines: list[str], i: int) -> int:
    """Index after the block that starts at key line ``i`` (its children are
    the following lines indented deeper than it)."""
    ind = _indent(lines[i])
    j = i + 1
    last = i
    while j < len(lines):
        if _is_body(lines[j]):
            if _indent(lines[j]) <= ind:
                break
            last = j
        j += 1
    return last + 1


def _find_key(lines, key, lo, hi, ind) -> int | None:
    for i in range(lo, hi):
        ln = lines[i]
        if _is_body(ln) and _indent(ln) == ind and ln.strip().startswith(key + ":"):
            return i
    return None


def _drop_keys(text: str, doc: dict):
    """Remove the retired keys' lines. Returns ``(lines, insert_at)`` or
    ``None`` when a key cannot be found line by line (flow style)."""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    drop: list[tuple[int, int]] = []
    insert_at = None
    for key in _TOP_KEYS:
        if key not in doc:
            continue
        i = _find_key(lines, key, 0, len(lines), 0)
        if i is None:
            return None
        drop.append((i, _block_end(lines, i)))
        insert_at = i if insert_at is None else min(insert_at, i)
    rp = doc.get("reparam")
    if isinstance(rp, dict) and any(k in rp for k in _REPARAM_KEYS):
        r = _find_key(lines, "reparam", 0, len(lines), 0)
        if r is None:
            return None
        r_end = _block_end(lines, r)
        child = next((_indent(lines[j]) for j in range(r + 1, r_end)
                      if _is_body(lines[j])), None)
        if child is None:
            return None
        for key in _REPARAM_KEYS:
            if key not in rp:
                continue
            i = _find_key(lines, key, r + 1, r_end, child)
            if i is None:
                return None
            drop.append((i, _block_end(lines, i)))
        if insert_at is None:
            insert_at = r_end
    keep = []
    shift = 0
    for i, ln in enumerate(lines):
        if any(a <= i < b for a, b in drop):
            if insert_at is not None and i < insert_at:
                shift += 1
            continue
        keep.append(ln)
    return keep, (len(keep) if insert_at is None else insert_at - shift)


def _yaml_scalar(v) -> str:
    s = yaml.safe_dump(v, default_flow_style=True, width=1 << 30).strip()
    return s[:-3].strip() if s.endswith("...") else s


def _water_lines(water: dict) -> list[str]:
    out = ["water:\n"]
    for k in ("seabed", "mask", "velocity", "buffer_cells", "mode"):
        if k in water:
            out.append(f"  {k}: {_yaml_scalar(water[k])}\n")
    return out


# ------------------------------------------------------------- planning --

def plan_migration(text: str, yaml_path: Path, out_path: Path | None = None
                   ) -> Migration:
    """Work out the ``water:`` block for one task YAML; write nothing."""
    yaml_path = Path(yaml_path).resolve()
    base = yaml_path.parent
    out_dir = (Path(out_path).resolve().parent if out_path is not None else base)
    mig = Migration(old_text=text)
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        mig.refusals.append("not a task mapping")
        return mig
    task = doc.get("task_type")
    rp = doc.get("reparam") if isinstance(doc.get("reparam"), dict) else {}
    old = {k: doc[k] for k in _TOP_KEYS if k in doc}
    old.update({k: rp[k] for k in _REPARAM_KEYS if k in rp})
    live = {k: v for k, v in old.items() if not _is_default(k, v)}
    freqsel = _is_freqsel(doc)

    if not old:
        if freqsel and "water" not in doc:
            _check_implicit_pin(doc, base, base, mig, None)
        if not mig.refusals:
            mig.new_text = text
            mig.notes.append("no retired water keys")
        return mig
    if "water" in doc and live:
        mig.refusals.append(
            "the YAML already has a `water:` block AND retired water keys at "
            f"non-default values ({sorted(live)}); resolve that by hand")
        return mig

    blocks = []       # water blocks the retired keys translate to
    stem = (Path(out_path).stem if out_path is not None else yaml_path.stem)
    n = live.get("freeze_top_n_rows")
    if n is not None:
        blocks.append(_from_top_rows(doc, int(n), base, out_dir, stem, mig))
    fwl = live.get("freeze_water_layer")
    if fwl is not None:
        blocks.append(_from_freeze_water_layer(doc, fwl, base, out_dir, mig))
    reparam_block = None
    if any(k in live for k in _REPARAM_KEYS):
        reparam_block = _from_reparam(doc, rp, live, base, out_dir, mig, freqsel)
    if mig.refusals:
        return mig
    blocks = [b for b in blocks if b is not None]
    if reparam_block is not None:
        blocks.append(reparam_block)
    water = None
    if blocks:
        srcs = {(("seabed", b["seabed"]) if "seabed" in b else ("mask", b["mask"]))
                for b in blocks}
        if len(srcs) > 1:
            mig.refusals.append(
                "the retired keys name different water layers "
                f"({sorted(s for _, s in srcs)}); keep one by hand")
            return mig
        water = {}
        for b in blocks:
            water.update(b)
        if reparam_block is not None:
            # the network pinned at water_vp_m_s, whatever the start held
            water["velocity"] = reparam_block["velocity"]
    if freqsel and doc.get("water") is None:
        _check_implicit_pin(doc, base, out_dir, mig, water)
        if mig.refusals:
            return mig

    edited = _drop_keys(text, doc)
    if edited is None:
        mig.refusals.append(
            "a retired key is written in flow style ({...}) and cannot be "
            "edited line by line; move it to block style or edit by hand")
        return mig
    lines, at = edited
    if water is not None:
        block = _water_lines(water)
        prev = lines[at - 1].strip() if at > 0 else ""
        if prev and not prev.startswith("#"):   # keep a comment attached
            block = ["\n"] + block
        if at < len(lines) and lines[at].strip():
            block = block + ["\n"]
        lines[at:at] = block
    new_text = "".join(lines)
    # The rewrite must be a valid task before it is offered.
    try:
        from sweep_tasks.yaml_io import load_task_from_dict

        load_task_from_dict(yaml.safe_load(new_text), base_dir=out_dir)
    except Exception as err:  # noqa: BLE001
        mig.refusals.append(f"the rewritten YAML does not validate: {err}")
        mig.files.clear()
        return mig
    mig.new_text = new_text
    mig.water = water
    return mig


def _from_top_rows(doc, n, base, out_dir, stem, mig):
    task = doc.get("task_type")
    if (doc.get("model_plan") or {}).get("z_window_m"):
        mig.refusals.append(
            "freeze_top_n_rows counted rows of the model_plan z-crop, not of "
            "the starting model; write the mask by hand")
        return None
    shape = _native_shape(doc, base)
    if shape is None:
        mig.refusals.append(
            "freeze_top_n_rows: cannot tell the native grid shape (set "
            "grid.shape, or point init_model at a .npy) to write the mask")
        return None
    mask = np.zeros(shape, dtype=bool)
    mask[: min(n, shape[0])] = True
    water = {}
    if task != "lsrtm":
        vp = _init_model(doc, base)
        if vp is None:
            mig.refusals.append(
                "freeze_top_n_rows held the starting model's values in those "
                "rows; the starting model cannot be read here to find their "
                "velocity (a dataset: source?) -- write the block by hand")
            return None
        v = _one_value(np.asarray(vp[: min(n, shape[0])]))
        if v is None:
            top = np.asarray(vp[: min(n, shape[0])])
            mig.refusals.append(
                f"freeze_top_n_rows held the starting model's top {n} rows, "
                f"which are not one velocity ({float(top.min()):g}..."
                f"{float(top.max()):g} m/s); water: pin holds one velocity "
                "there -- decide which by hand")
            return None
        water["velocity"] = v
    mpath = out_dir / f"{stem}_water_mask.npy"
    mig.files[mpath] = mask
    mig.notes.append(f"writes {mpath.name}: rows 0..{n - 1} of the native grid "
                     f"{shape} are water")
    if doc.get("stages") and any(isinstance(s, dict) and s.get("dh_m")
                                 for s in doc["stages"]):
        mig.notes.append(
            "a stage with its own dh_m now takes, cell by cell, the native row "
            "nearest to it; freeze_top_n_rows took the stage rows at depths "
            "<= (n - 1) * grid.dh, which can differ by one stage row")
    if task == "lsrtm":
        mig.notes.append("LSRTM holds the water's reflectivity at zero")
    return {"mask": mpath.name, **water}


def _water_velocity(doc, base, mask_fn, what, mig):
    vp = _init_model(doc, base)
    if vp is None:
        mig.refusals.append(
            f"{what} held the starting model's water values; the starting "
            "model cannot be read here to find their velocity -- write the "
            "block by hand")
        return None
    try:
        m = mask_fn(tuple(vp.shape))
    except Exception as err:  # noqa: BLE001
        mig.refusals.append(f"{what}: {err}")
        return None
    vals = np.asarray(vp)[m]
    if vals.size == 0:
        mig.refusals.append(f"{what}: the seabed leaves no water cell")
        return None
    v = _one_value(vals)
    if v is None:
        mig.refusals.append(
            f"{what} held the starting model's water at its own values "
            f"({float(vals.min()):g}...{float(vals.max()):g} m/s); water: pin "
            "holds one velocity there -- decide which by hand")
    return v


def _seabed_ok(p, base, what, mig):
    pa = _abs(p, base)
    if not pa.is_file():
        mig.refusals.append(f"{what}: {pa} does not exist")
        return None
    obj = np.load(pa, allow_pickle=False)
    if isinstance(obj, np.ndarray):
        mig.refusals.append(
            f"{what}: {pa} is a plain array; `water.seabed` is an .npz with "
            "key 'seabed_depth'. Convert it once with "
            "`sweep_tasks.bathymetry.save_seabed_depth_npz(<new>.npz, "
            "np.load(<old>.npy))` (depths in metres) and rerun")
        return None
    files = list(obj.files)
    obj.close()
    if "seabed_depth" not in files:
        mig.refusals.append(f"{what}: {pa} has no 'seabed_depth' key ({files})")
        return None
    return pa


def _seabed_mask_fn(doc, seabed: Path, buffer_cells: int):
    from sweep_tasks._helpers.water import WaterRegion
    from sweep_tasks.schemas import WaterSpec

    dh = float((doc.get("grid") or {}).get("dh", 1.0))
    dxyz = (doc.get("geometry") or {}).get("dh_xyz_m")

    def fn(shape):
        d = tuple(float(v) for v in dxyz) if dxyz and len(shape) == 3 else dh
        reg = WaterRegion(WaterSpec(seabed=seabed, buffer_cells=buffer_cells),
                          shape, d)
        return reg.mask_on_grid(shape, d)
    return fn


def _from_freeze_water_layer(doc, fwl, base, out_dir, mig):
    if not isinstance(fwl, dict):
        mig.refusals.append("freeze_water_layer is not a mapping")
        return None
    sb = fwl.get("seabed_depth_path")
    if sb is None:
        mig.refusals.append("freeze_water_layer.enabled without seabed_depth_path")
        return None
    pa = _seabed_ok(sb, base, "freeze_water_layer.seabed_depth_path", mig)
    if pa is None:
        return None
    buf = int(fwl.get("buffer_cells", 0) or 0)
    v = _water_velocity(doc, base, _seabed_mask_fn(doc, pa, buf),
                        "freeze_water_layer", mig)
    if v is None:
        return None
    w = {"seabed": _rel(pa, out_dir), "velocity": v}
    if buf:
        w["buffer_cells"] = buf
    plan = doc.get("obs", {}) or {}
    streaming = (plan.get("plan") or {}).get("sampling") is not None
    if not _is_freqsel(doc) and not streaming:
        mig.notes.append(
            "the conventional (static-obs) path IGNORED freeze_water_layer; "
            "with `water:` this run now holds the water -- a behaviour change")
    if doc.get("reparam") is not None and (_is_freqsel(doc) or streaming):
        mig.notes.append(
            "with a reparam network freeze_water_layer only zeroed the "
            "network's input gradient in the water, which never held it; "
            "`water:` pins the network's render there -- a behaviour change")
    if streaming:
        mig.notes.append(
            "plan-streaming rounded the seabed down (floor(depth / dz)); "
            "`water:` marks a cell water when its centre is above the seabed, "
            "one more cell where depth / dz has a fractional part > 0.5")
    if buf and doc.get("stages") and any(isinstance(s, dict) and s.get("dh_m")
                                         for s in doc["stages"]):
        mig.notes.append(
            f"buffer_cells={buf} is now {buf} NATIVE cells as a depth on every "
            "stage grid (it was that many cells of each stage grid)")
    return w


def _from_reparam(doc, rp, live, base, out_dir, mig, freqsel):
    mask_on = not _is_default("mask_water_layer", rp.get("mask_water_layer"))
    reset = not _is_default("water_reset_each_stage",
                            rp.get("water_reset_each_stage"))
    sb = rp.get("seabed_depth_path")
    v = rp.get("water_vp_m_s", 1500.0)
    v = float(v) if v is not None else 1500.0
    vrz = (doc.get("physics") or {}).get("equation") in ("AcousticVRZ",
                                                          "AcousticVRZ3D")
    if mask_on and sb is not None:
        pa = _seabed_ok(sb, base, "reparam.seabed_depth_path", mig)
        if pa is None:
            return None
        w = {"seabed": _rel(pa, out_dir), "velocity": v}
        if reset:
            w["mode"] = "reset_each_stage"
            if freqsel or ((doc.get("obs") or {}).get("plan") or {}).get("sampling"):
                mig.notes.append(
                    "water_reset_each_stage was IGNORED on this path (only the "
                    "conventional path implemented it); mode: reset_each_stage "
                    "now applies -- a behaviour change")
        if not freqsel and not ((doc.get("obs") or {}).get("plan") or {}).get("sampling"):
            mig.notes.append(
                "the conventional path ignored reparam.seabed_depth_path and "
                f"pinned the cells at exactly {v:g} m/s; it reads the seabed now")
        return w
    if mask_on or reset:
        mig.refusals.append(
            ("reparam.mask_water_layer" if mask_on else
             "reparam.water_reset_each_stage")
            + f" without seabed_depth_path found the water as the starting "
            f"model's cells at exactly {v:g} m/s. The water is no longer read "
            "off velocities: provide it -- a seabed (.npz, key 'seabed_depth') "
            "or a boolean mask of the native grid -- and write "
            f"`water: {{seabed|mask: <file>, velocity: {v:g}"
            + (", mode: reset_each_stage" if reset else "") + "}`")
        return None
    if sb is not None:
        mig.notes.append("reparam.seabed_depth_path did nothing without "
                         "mask_water_layer; dropped")
    if "water_vp_m_s" in live:
        if freqsel or vrz:
            mig.refusals.append(
                f"reparam.water_vp_m_s={v:g} set "
                + ("the velocity this frequency-selection run pinned (every "
                   "starting-model cell at exactly that value)" if freqsel else
                   "the VRZ Gardner water threshold (vp <= water_vp_m_s + 5)")
                + "; that now comes from `water.velocity`, which needs a "
                "`water:` block with a seabed or mask you provide")
            return None
        mig.notes.append(f"reparam.water_vp_m_s={v:g} did nothing on this "
                         "path without mask_water_layer; dropped")
    return None


def _rel(p: Path, out_dir: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(out_dir))
    except ValueError:
        return str(Path(p).resolve())


def _check_implicit_pin(doc, base, out_dir, mig, water):
    """The frequency-selection path pinned every starting-model cell at
    exactly 1500 m/s. Refuse when the migrated config would stop pinning
    cells the old run held."""
    vp = _init_model(doc, base)
    if vp is None:
        mig.notes.append(
            "frequency selection: the starting model could not be read here, "
            "so whether the old run pinned cells at exactly 1500 m/s is "
            "unchecked; the runner refuses such a model without `water:`")
        return
    pinned = np.asarray(vp) == _OLD_PIN_VP
    n_old = int(pinned.sum())
    if not n_old:
        return
    if water is None:
        mig.refusals.append(
            f"frequency selection: the old run pinned {n_old:,} starting-model "
            "cells at exactly 1500 m/s without being asked. Provide that "
            "water explicitly -- `water: {seabed: <.npz>}` or `water: {mask: "
            "<boolean .npy of the native grid>}` (the cells it pinned, to "
            "reproduce it) -- or write `water: false` to invert them freely")
        return
    src = water.get("seabed") or water.get("mask")
    try:
        if "mask" in water and (out_dir / water["mask"]) in mig.files:
            new = mig.files[out_dir / water["mask"]]
        else:
            from sweep_tasks._helpers.water import WaterRegion
            from sweep_tasks.schemas import WaterSpec

            spec = WaterSpec(**{k: (str(_abs(x, out_dir))
                                    if k in ("seabed", "mask") and isinstance(x, str)
                                    else x)
                                for k, x in water.items()})
            dh = float((doc.get("grid") or {}).get("dh", 1.0))
            new = WaterRegion(spec, vp.shape, dh).mask_on_grid(vp.shape, dh)
    except Exception as err:  # noqa: BLE001
        mig.refusals.append(f"frequency selection: cannot compare {src} with "
                            f"the cells the old run pinned: {err}")
        return
    extra = int((pinned & ~new).sum())
    if extra:
        mig.refusals.append(
            f"frequency selection: the old run also pinned {extra:,} cells at "
            f"exactly 1500 m/s that {src} does not mark as water; `water:` "
            "cannot hold both -- reproduce it with a mask of those cells, or "
            "accept the change by editing by hand")


# ------------------------------------------------------------------ CLI --

def main_migrate(path: str, out: str | None, *, stream=None) -> int:
    import sys

    stream = stream or sys.stdout
    src = Path(path)
    text = src.read_text()
    out_path = Path(out) if out is not None else None
    mig = plan_migration(text, src, out_path)
    for n in mig.notes:
        print(f"note: {n}", file=stream)
    if mig.refusals:
        for r in mig.refusals:
            print(f"refused: {r}", file=stream)
        print(f"{src}: NOT migrated; nothing written", file=stream)
        return 1
    if not mig.changed:
        print(f"{src}: nothing to migrate", file=stream)
        return 0
    print(mig.diff(str(src), str(out_path or src)), file=stream, end="")
    if out_path is None:
        print(f"(dry run: pass -o NEW.yaml to write"
              + (", and the mask file(s) " + ", ".join(p.name for p in mig.files)
                 if mig.files else "") + ")", file=stream)
        return 0
    for p, arr in mig.files.items():
        np.save(p, arr)
        print(f"wrote {p}", file=stream)
    out_path.write_text(mig.new_text)
    print(f"wrote {out_path}", file=stream)
    return 0


__all__ = ["Migration", "plan_migration", "main_migrate"]
