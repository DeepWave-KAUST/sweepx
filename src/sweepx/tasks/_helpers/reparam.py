"""Reparameterisation network (VelocityINR) build + hash schedule + tiled render. Verbatim from runner.py."""
import torch


def _build_reparam_net(spec, base_vp, bounds, water_mask_override=None):
    """Construct a :class:`sweep_nn.VelocityINR` from a ReparamSpec.

    ``base_vp`` is the initial vp tensor at the first stage's grid. The net
    keeps it as a buffer (its forward returns ``base + delta``). ``bounds``
    is the optional :class:`ModelBounds` for vp — if provided, the network
    clamps its render output to those limits.
    """
    import torch  # local import — runner.py keeps torch imports per-function
    from sweep_nn import VelocityINR

    bounds_tuple = None
    if bounds is not None and bounds.min is not None and bounds.max is not None:
        bounds_tuple = (float(bounds.min), float(bounds.max))
    # Water-layer pin: build a boolean mask whose True voxels are
    # rendered as a fixed water velocity instead of the SIREN output.
    # When ``water_mask_override`` is supplied the caller has already
    # built the right mask (typically from a 2-D seabed_depth.npz,
    # cropped to the post-model_plan window) — use it verbatim. Else
    # fall back to ``init_vp == water_vp_m_s`` exact equality.
    water_mask = None
    if water_mask_override is not None:
        water_mask = water_mask_override.to(
            device=base_vp.device, dtype=torch.bool,
        )
        if tuple(water_mask.shape) != tuple(base_vp.shape):
            raise ValueError(
                f"water_mask_override shape {tuple(water_mask.shape)} != "
                f"base_vp shape {tuple(base_vp.shape)}"
            )
    elif bool(getattr(spec, "mask_water_layer", False)):
        water_vp_val = float(getattr(spec, "water_vp_m_s", 1500.0))
        water_mask = (base_vp.detach() == water_vp_val)
        if not bool(water_mask.any()):
            import warnings
            warnings.warn(
                f"reparam.mask_water_layer=True but no voxels in init_vp "
                f"equal water_vp_m_s={water_vp_val}; mask will be empty.",
                stacklevel=2,
            )
    net = VelocityINR(
        base_vp.detach(),
        vp_mean=float(spec.vp_mean),
        vp_std=float(spec.vp_std),
        hidden_features=int(spec.hidden_features),
        hidden_layers=int(spec.hidden_layers),
        first_omega0=float(spec.first_omega0),
        hidden_omega0=float(spec.hidden_omega0),
        use_bias=bool(spec.use_bias),
        use_hash_encoding=bool(spec.hash.enabled),
        hash_levels=int(spec.hash.levels),
        hash_features_per_level=int(spec.hash.features_per_level),
        hash_log2_size=int(spec.hash.log2_size),
        hash_base_resolution=(list(spec.hash.base_resolution)
                              if isinstance(spec.hash.base_resolution, list)
                              else int(spec.hash.base_resolution)),
        hash_finest_resolution=(list(spec.hash.finest_resolution)
                                if isinstance(spec.hash.finest_resolution, list)
                                else int(spec.hash.finest_resolution)),
        hash_c2f=bool(getattr(spec.hash, "c2f", None) is not None
                      and spec.hash.c2f.enabled),
        hash_c2f_base_levels=int(getattr(getattr(spec.hash, "c2f", None),
                                         "base_levels", 2) or 2),
        hash_c2f_ramp=str(getattr(getattr(spec.hash, "c2f", None),
                                  "ramp", "cosine") or "cosine"),
        hash_growing=bool(getattr(getattr(spec.hash, "c2f", None), "growing", False)),
        hash_backend=str(getattr(spec.hash, "backend", "pytorch") or "pytorch"),
        use_fourier_encoding=bool(getattr(getattr(spec, "fourier", None), "enabled", False)),
        fourier_levels=int(getattr(getattr(spec, "fourier", None), "levels", 6) or 6),
        fourier_include_input=bool(getattr(getattr(spec, "fourier", None), "include_input", True)),
        direct_velocity=bool(spec.direct_velocity),
        coord_min=float(spec.coord_min),
        coord_max=float(spec.coord_max),
        bounds=bounds_tuple,
        water_mask=water_mask,
        water_vp=float(getattr(spec, "water_vp_m_s", 1500.0)),
        lateral_downsample=getattr(spec, "lateral_downsample", 1),
        compile_render=bool(getattr(spec, "compile_render", False)),
    ).to(base_vp.device)
    # Warm-start from a saved reparam net (a previous run's ``reparam_net.pt``,
    # dumped via SWEEP_SAVE_REPARAM_NET=1) — continue the SAME network across
    # separate processes/bands (e.g. run 2-4Hz, then resume 2-8Hz). A
    # GrowingHashGrid auto-grows its levels to match the checkpoint on load, so
    # the second run keeps the first run's latents and grows further. The hash
    # config (levels/base/finest/log2/features) must match across runs.
    _init_from = getattr(spec, "init_from", None)
    if _init_from:
        sd = torch.load(str(_init_from), map_location=base_vp.device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        net.load_state_dict(sd)  # strict: mismatched hash config fails loudly
        print(f"[reparam] init_from: warm-started network <- {_init_from} "
              f"(encoder levels now {getattr(net.encoder, 'n_active', 'n/a')})",
              flush=True)
    return net


def _has_hash_schedule(net) -> bool:
    """True if ``net``'s encoder supports a coarse-to-fine level schedule
    (either a CoarseToFineHashGrid mask or a GrowingHashGrid lazy allocator)."""
    enc = getattr(net, "encoder", None)
    return hasattr(enc, "set_progress") or hasattr(enc, "grow_to_progress")


def _advance_hash_schedule(net, progress, c2f_cfg, optimizer):
    """Advance the hash coarse-to-fine schedule by one epoch (``progress`` in [0,1]).

    Two encoder mechanisms, picked by duck-typing:
      * ``GrowingHashGrid``      -> allocate fine levels ON DEMAND and register the
        new latent Parameters with ``optimizer`` (``add_param_group``) so Adam
        trains them (saves latent memory: fine levels aren't allocated until due).
      * ``CoarseToFineHashGrid`` -> soft per-level mask (``set_progress``).

    Both read ``base_levels -> final_levels`` over ``[warmup, ramp_end]`` from
    ``c2f_cfg``. Returns ``(active_levels: float, total_levels: int)`` for logging,
    or ``None`` if the encoder has no schedulable hash grid.
    """
    enc = getattr(net, "encoder", None)
    if enc is None:
        return None
    warmup = float(c2f_cfg.warmup)
    ramp_end = float(c2f_cfg.ramp_end)
    final = (None if getattr(c2f_cfg, "final_levels", None) is None
             else int(c2f_cfg.final_levels))
    if hasattr(enc, "grow_to_progress"):          # GrowingHashGrid (lazy alloc)
        base = (None if getattr(c2f_cfg, "base_levels", None) is None
                else int(c2f_cfg.base_levels))
        new = enc.grow_to_progress(progress, base_levels=base, final_levels=final,
                                   warmup=warmup, ramp_end=ramp_end)
        if new and optimizer is not None:
            optimizer.add_param_group({"params": new})
        return float(enc.n_active), int(enc.L)
    if hasattr(enc, "set_progress"):              # CoarseToFineHashGrid (soft mask)
        enc.set_progress(progress, warmup=warmup, ramp_end=ramp_end, final_levels=final)
        return float(enc.n_active_levels), int(enc.L)
    return None


def _render_full_to_cpu_tiled(net, cz: int = 1, cy: int = 294):
    """Render a VelocityINR's full base grid to a CPU tensor, z+y tiled.

    The DD snapshot render (rank 0 only) reconstructs the GLOBAL model, but
    a full-lateral z-slab render materializes O(ny*nx * 16 levels * 8 corners)
    hash vertex positions -> many GB on a large grid, even at
    chunk_rows=1. On a 32 GB V100 tile the solver working set already fills
    most of the card, so even the
    chunk_rows=1 full-lateral render OOM'd rank 0. Tiling BOTH z and y bounds
    each ``render_window`` to cz*cy*nx points; at (cz=1, cy=294) peak render
    overhead is about a GB (probed on RTX 6000 Ada), which fits the
    headroom. Each tile is copied to CPU immediately, so GPU only ever
    holds one tile's intermediates. 3-D only; callers fall back to
    ``render(chunk_rows=1)`` for 2-D.
    """
    import torch
    full = tuple(int(s) for s in net.base_velocity.shape)
    if len(full) != 3:
        return net.render(chunk_rows=1).detach().to("cpu")
    nz, ny, nx = full
    out = torch.empty(full, dtype=torch.float32, device="cpu")
    for z0 in range(0, nz, cz):
        z1 = min(nz, z0 + cz)
        for y0 in range(0, ny, cy):
            y1 = min(ny, y0 + cy)
            with torch.no_grad():
                win = net.render_window(z0, z1, y0, y1, 0, nx)
            out[z0:z1, y0:y1] = win.detach().to("cpu")
            del win
    return out
