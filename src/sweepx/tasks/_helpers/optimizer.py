"""Optimizer / scheduler construction + per-stage lr scaling. Verbatim from runner.py."""

def _build_optimizer(opt_spec, inv_tensors_by_name, required_names):
    """Construct a torch optimizer; supports per-model lr via dict."""

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
        return torch.optim.LBFGS(
            params_in_order,
            lr=float(opt_spec.lr),
            max_iter=opt_spec.max_iter,
            history_size=opt_spec.history_size,
            line_search_fn=opt_spec.line_search_fn,
        )
    raise ValueError(f"Unknown optimizer kind '{kind}'.")


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
        return torch.optim.LBFGS(
            params, lr=float(lr), max_iter=opt_spec.max_iter,
            history_size=opt_spec.history_size,
            line_search_fn=opt_spec.line_search_fn,
        )
    raise ValueError(f"Unknown optimizer kind '{kind}' for reparam.")


def _build_scheduler(sched_spec, optimizer, total_epochs):
    """Dispatch the LR scheduler via `sweep_tasks.runtime.scheduler.build`.

    The runner's `build()` accepts any object with the right `kind` +
    field attributes, so our Pydantic `Scheduler*` discriminated union
    plugs in directly without conversion.
    """
    from sweep_tasks.runtime.scheduler import build as build_scheduler
    return build_scheduler(sched_spec, optimizer, total_epochs)


def _remember_initial_lrs(optimizer) -> list[float]:
    return [float(g["lr"]) for g in optimizer.param_groups]


def _apply_stage_lr_scale(optimizer, initial_lrs: list[float], scale: float) -> None:
    for group, base in zip(optimizer.param_groups, initial_lrs):
        group["lr"] = base * float(scale)
