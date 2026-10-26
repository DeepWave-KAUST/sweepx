"""Optimizer / scheduler construction + per-stage lr scaling. Verbatim from runner.py."""

import functools


def _build_optimizer(opt_spec, inv_tensors_by_name, required_names, model_bounds=None):
    """Construct a torch optimizer; supports per-model lr via dict.

    ``model_bounds`` (``spec.model_bounds``) gives L-BFGS the box of a single
    inverted model, so its line search never evaluates a point outside it."""

    import torch

    params_in_order = [inv_tensors_by_name[name] for name in required_names]

    def _build_param_groups(lr_value):
        if isinstance(lr_value, dict):
            groups = []
            for name in required_names:
                if name not in lr_value:
                    raise ValueError(
                        f"optimizer.lr dict missing entry for inverted model '{name}'. "
                        f"Provided: {list(lr_value)}."
                    )
                groups.append({"params": [inv_tensors_by_name[name]], "lr": float(lr_value[name])})
            return groups
        return [{"params": params_in_order, "lr": float(lr_value)}]

    kind = opt_spec.kind
    if kind == "adam":
        return torch.optim.Adam(
            _build_param_groups(opt_spec.lr),
            eps=opt_spec.eps,
            betas=tuple(opt_spec.betas),
        )
    if kind == "sgd":
        return torch.optim.SGD(
            _build_param_groups(opt_spec.lr),
            momentum=opt_spec.momentum,
            nesterov=opt_spec.nesterov,
            weight_decay=opt_spec.weight_decay,
        )
    if kind == "lbfgs":
        bounds = None
        if len(required_names) == 1:
            from sweepx.tasks._helpers.bounds import _effective_bound
            b = _effective_bound(model_bounds, required_names[0])
            if b is not None and (b.min is not None or b.max is not None):
                bounds = (b.min, b.max)
        return _task_lbfgs_class()(
            params_in_order,
            lr=float(opt_spec.lr),
            history_size=opt_spec.history_size,
            bounds=bounds,
            initial_step=_lbfgs_initial_step(opt_spec.initial_step, params_in_order),
        )
    raise ValueError(f"Unknown optimizer kind '{kind}'.")


@functools.lru_cache(maxsize=1)
def _task_lbfgs_class():
    from sweepx.opt.optim import LBFGS

    class _TaskLBFGS(LBFGS):
        """``sweepx.opt`` L-BFGS whose step multiplier sits in
        ``param_groups[0]["lr"]``, where the runner's schedulers, per-stage lr
        scales and resume read and write every optimizer's lr."""

        def __init__(self, params, *, lr, **kwargs):
            super().__init__(params, alpha_init=lr, alpha_max=10.0 * lr, **kwargs)
            self.defaults["lr"] = lr
            for group in self.param_groups:
                group["lr"] = lr

        def step(self, closure=None):
            lr = float(self.param_groups[0]["lr"])
            self.alpha_init, self.alpha_max = lr, 10.0 * lr
            return super().step(closure)

    return _TaskLBFGS


def _lbfgs_initial_step(initial_step, params):
    """``initial_step`` of an L-BFGS spec in the model's units; ``"auto"`` is 1% of
    the mean |model| (25 m/s on a 2500 m/s velocity model)."""
    if initial_step != "auto":
        return float(initial_step)
    import torch

    with torch.no_grad():
        mean_abs = float(torch.cat([p.detach().abs().reshape(-1) for p in params]).mean())
    if not mean_abs > 0.0:
        raise ValueError(
            "optimizer.initial_step='auto' takes its scale from the model, which is "
            "all zeros here (a reflectivity's first iterate): set initial_step to "
            "the size of the first L-BFGS step in the model's units.")
    return 0.01 * mean_abs


def _build_reparam_optimizer(opt_spec, net_params, lr: float):
    """Build a torch optimizer over a reparam network's parameters.

    Mirrors :func:`_build_optimizer` for the network-as-vp case. The
    ``lr`` comes from ``spec.reparam.lr`` (≈ 1e-4); the optimizer kind
    and other hyperparameters come from ``spec.optimizer``.
    """
    import torch

    params = list(net_params)
    if not params:
        raise ValueError("reparam network has no trainable parameters")
    kind = opt_spec.kind
    if kind == "adam":
        return torch.optim.Adam(
            params, lr=float(lr), eps=opt_spec.eps, betas=tuple(opt_spec.betas),
        )
    if kind == "sgd":
        return torch.optim.SGD(
            params, lr=float(lr), momentum=opt_spec.momentum,
            nesterov=opt_spec.nesterov, weight_decay=opt_spec.weight_decay,
        )
    if kind == "lbfgs":
        if "initial_step" in opt_spec.model_fields_set:
            raise ValueError(
                "optimizer.initial_step sizes the first step in the model's units and "
                "has no meaning for a reparam network's weights; remove it (the "
                "network's L-BFGS steps start from reparam.lr).")
        return torch.optim.LBFGS(
            params, lr=float(lr), max_iter=opt_spec.max_iter,
            history_size=opt_spec.history_size,
            line_search_fn=opt_spec.line_search_fn,
        )
    raise ValueError(f"Unknown optimizer kind '{kind}' for reparam.")


def _build_scheduler(sched_spec, optimizer, total_epochs):
    """Dispatch the LR scheduler via `sweepx.tasks.runtime.scheduler.build`.

    The runner's `build()` accepts any object with the right `kind` +
    field attributes, so our Pydantic `Scheduler*` discriminated union
    plugs in directly without conversion.
    """
    from sweepx.tasks.runtime.scheduler import build as build_scheduler
    return build_scheduler(sched_spec, optimizer, total_epochs)


def _bind_stage_scheduler(sched_spec, optimizer, total_epochs, epoch, *,
                          resume_lrs=None):
    """Put the run's LR schedule on the optimizer a stage steps.

    ``scheduler`` is ONE schedule over the whole run, indexed by the global
    epoch ``t`` (``0 .. total_epochs - 1`` across all stages). Each stage has
    its own base lr -- ``optimizer.lr`` (``reparam.lr``) times ``lr_scale``
    (times ``inr_lr_scale`` under reparam) -- and at epoch ``t`` it runs at
    ``runtime.scheduler.lr_at(spec, base, t)``. A rebuilt optimizer (a
    ``dh_m`` change, ``optimizer_reset``) drops Adam's moments, not the
    schedule's position, and ``lr_scale`` rescales the schedule instead of
    restarting it. A cosine with the default ``t_max`` (all stages' epochs)
    therefore anneals once, over the whole run.

    Call it at every stage entry, after the stage's base lr is in
    ``optimizer.param_groups`` (``_apply_stage_lr_scale``); ``epoch`` is the
    global epoch the stage starts at. Within the stage the torch scheduler
    steps in its usual chained form, so a run without ``stages`` steps exactly
    as a plain torch scheduler would.

    ``resume_lrs``: on a resume that lands inside a stage, the lrs of the
    checkpointed optimizer. A group's checkpointed lr is kept when it is this
    schedule's lr at ``epoch`` (equal up to the rounding of the chained steps
    that produced it), so the resumed run is bit-identical to the
    uninterrupted one; otherwise -- the schedule or the group's base lr
    changed since the checkpoint -- the schedule's lr is used.

    Every param group is bound, a ``hash.c2f.growing`` network's grown levels
    included: they run at the network's lr (``_apply_stage_lr_scale`` gives
    them its base, ``_helpers.reparam._add_grown_levels`` joins a level
    grown during the stage to this schedule).

    Returns the torch scheduler, or ``None`` for ``kind: constant``.
    """
    if sched_spec.kind == "constant":
        return None
    import math
    from sweepx.tasks.runtime.scheduler import lr_at
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]     # the stage's base lr
    scheduler = _build_scheduler(sched_spec, optimizer, total_epochs)
    if epoch > 0:
        lrs = [lr_at(sched_spec, base, epoch, total_epochs)
               for base in scheduler.base_lrs]
        if resume_lrs is not None and len(resume_lrs) == len(lrs):
            lrs = [r if math.isclose(r, s, rel_tol=1e-9) else s
                   for r, s in zip(resume_lrs, lrs)]
        for group, lr in zip(optimizer.param_groups, lrs):
            group["lr"] = lr
        # the state torch's own step() would leave at this epoch
        scheduler.last_epoch = int(epoch)
        scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
    return scheduler


def _remember_initial_lrs(optimizer) -> list[float]:
    return [float(g["lr"]) for g in optimizer.param_groups]


def _apply_stage_lr_scale(optimizer, initial_lrs: list[float], scale: float) -> None:
    """Set every param group to its base lr x ``scale``: the stage's base lr.

    ``initial_lrs`` are the bases of the groups the optimizer was built with.
    A group added after that, a level a ``hash.c2f.growing`` network grew,
    belongs to the network of the first group and takes its base.
    """
    for i, group in enumerate(optimizer.param_groups):
        base = initial_lrs[i] if i < len(initial_lrs) else initial_lrs[0]
        group["lr"] = base * float(scale)
