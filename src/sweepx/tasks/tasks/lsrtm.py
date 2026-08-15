"""LSRTM task runner (mixin). Verbatim from runner.py."""
from pathlib import Path
import numpy as np
from sweep_tasks.schemas import LSRTMSpec, PhysicsSpec
from sweep_tasks._helpers.bounds import _apply_bounds
from sweep_tasks._helpers.run_checkpoint import (
    _load_checkpoint,
    _save_checkpoint,
    _zero_top_rows,
)
from sweep_tasks._helpers.geometry import _build_geometry_2d
from sweep_tasks._helpers.loss import (
    _loss_sum,
    _mask_chunk,
)
from sweep_tasks._helpers.model import (
    _get_equation_class,
    _load_model_tensor,
)
from sweep_tasks._helpers.optimizer import (
    _apply_stage_lr_scale,
    _build_optimizer,
    _build_scheduler,
    _remember_initial_lrs,
)
from sweep_tasks._helpers.plotting import (
    _lsrtm_background_equation,
    _plot_loss_curve,
    _save_illumination,
)
from sweep_tasks._helpers.solver_build import (
    _build_solver,
    _resolve_modeling_inputs,
    _validate_single_model,
)
from sweep_tasks._helpers.stages import _normalise_stage_list
from sweep_tasks._helpers.stop import _GracefulStopper
from sweep_tasks._helpers.util import _apply_seed
from sweep_tasks._helpers.wavelet_build import _build_wavelet


class LSRTMRunnerMixin:
    def _run_lsrtm(self, spec: LSRTMSpec, task_dir: Path):
        import torch

        from sweep_tasks.runtime import distributed as _dist

        dist_info = getattr(self, "_dist", None)
        if dist_info is None:
            dist_info = _dist.init_distributed_if_needed()
            self._dist = dist_info

        if dist_info.is_distributed and spec.optimizer.kind == "lbfgs":
            raise ValueError(
                "LBFGS optimiser is not supported under torchrun in Phase 1. "
                "Use adam or sgd."
            )

        _apply_seed(spec.seed)
        dev = _dist.resolve_dist_device(spec.device, dist_info.local_rank)

        bg_equation_name = _lsrtm_background_equation(spec.physics.equation)
        bg_cls = _get_equation_class(bg_equation_name)
        _validate_single_model(bg_cls, spec.background_model)

        if spec.grid.shape is not None:
            shape = tuple(int(v) for v in spec.grid.shape)
        elif spec.background_model.constant is not None:
            shape = tuple(int(v) for v in spec.background_model.shape)
        else:
            shape = tuple(np.load(spec.background_model.path, mmap_mode="r").shape)

        # Background Acoustic solver uses the plain ["h1"] receiver type.
        bg_physics = PhysicsSpec(
            equation=bg_equation_name,
            spatial_order=spec.physics.spatial_order,
            abcn=spec.physics.abcn,
            free_surface=spec.physics.free_surface,
            pml_type=spec.physics.pml_type,
            source_type=spec.physics.source_type,
            receiver_type=["h1"],
        )
        acoustic_solver = _build_solver(
            bg_physics, spec.backend, shape, spec.grid.dh, spec.time.dt, spec.time.nt, dev
        )
        lsrtm_solver = _build_solver(
            spec.physics, spec.backend, shape, spec.grid.dh, spec.time.dt, spec.time.nt, dev
        )

        wavelet = _build_wavelet(spec.wavelet, spec.time)
        sources, receivers = _build_geometry_2d(spec.geometry, shape)
        nshots = int(sources.shape[0])
        if dist_info.is_root:
            print(f"[lsrtm] shape={shape} nshots={nshots} epochs={spec.epochs} "
                  f"world_size={dist_info.world_size}")

        true_tensor = _load_model_tensor(spec.true_model).to(dev)
        bg_tensor = _load_model_tensor(spec.background_model).to(dev)

        mod_wavelet, mod_sources, mod_receivers, used_override = _resolve_modeling_inputs(
            spec, wavelet, sources, receivers, shape
        )
        if used_override and dist_info.is_root:
            print(f"[lsrtm] modeling_override applied (wavelet={spec.modeling_override.wavelet is not None}, "
                  f"geometry={spec.modeling_override.geometry is not None})")

        # scattered observed data = forward(true) - forward(background) — same on every rank.
        with torch.no_grad():
            obs_true = acoustic_solver(mod_wavelet, mod_sources, mod_receivers,
                                       models=[true_tensor]).detach().clone()
            obs_bg = acoustic_solver(mod_wavelet, mod_sources, mod_receivers,
                                     models=[bg_tensor]).detach().clone()
        obs = (obs_true - obs_bg).detach().cpu()
        del obs_true, obs_bg, true_tensor
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        # Reflectivity is the only inverted parameter; vp stays as background.
        vp = bg_tensor
        ref = torch.zeros_like(vp, requires_grad=True)
        inv_by_name = {"reflectivity": ref}
        required_names = ["reflectivity"]

        total_epochs = (sum(s.epochs for s in spec.stages)
                        if spec.stages else int(spec.epochs))
        optimizer = _build_optimizer(spec.optimizer, inv_by_name, required_names)
        scheduler = _build_scheduler(spec.scheduler, optimizer, total_epochs)
        initial_lrs = _remember_initial_lrs(optimizer)

        # Resume: same two modes as FWI — see ``_run_fwi`` for semantics.
        losses: list[float] = []
        start_epoch = 0
        resume_src: str | None = None
        if spec.resume_from:
            resume_src = spec.resume_from
            ckpt_dir = Path(spec.output_dir).expanduser() / spec.resume_from
        elif spec.resume and (task_dir / "checkpoint.pt").exists():
            resume_src = task_dir.name
            ckpt_dir = task_dir
        else:
            ckpt_dir = None
        if ckpt_dir is not None:
            if dist_info.is_root:
                ckpt = _load_checkpoint(ckpt_dir)
            else:
                ckpt = None
            ckpt = _dist.broadcast_object(ckpt, dist_info, src=0)
            ref.data.copy_(ckpt["models"]["reflectivity"].to(dev))
            optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler is not None and ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            losses = list(ckpt.get("losses", []))
            start_epoch = int(ckpt["epoch"]) + 1
            if dist_info.is_root:
                torch.set_rng_state(ckpt["torch_rng"])
                np.random.set_state(ckpt["numpy_rng"])
                print(f"[lsrtm] resumed from '{resume_src}' at epoch {start_epoch}")

        # Bounds keyed by "reflectivity" to reuse the FWI helper.
        bounds_by_name: dict = {}
        if spec.reflectivity_bounds is not None:
            bounds_by_name["reflectivity"] = spec.reflectivity_bounds

        stages = _normalise_stage_list(spec)
        out_dir = task_dir / "output"
        snapshots_dir = out_dir / "epochs"
        if dist_info.is_root:
            snapshots_dir.mkdir(exist_ok=True)
        _dist.barrier(dist_info)
        epoch_global = start_epoch

        # Graceful ctrl-c — see ``_run_fwi`` for the full rationale.
        stopper = _GracefulStopper()
        stopper.install(label="lsrtm")
        interrupted = False
        interrupted_at_epoch: int | None = None

        for stage_idx, stage in enumerate(stages):
            stage_offset = sum(s.epochs for s in stages[:stage_idx])
            already_done_in_stage = max(0, epoch_global - stage_offset)
            remaining = stage.epochs - already_done_in_stage
            if remaining <= 0:
                continue
            stage_wavelet = (_build_wavelet(stage.wavelet, spec.time)
                             if stage.wavelet is not None else wavelet)
            _apply_stage_lr_scale(optimizer, initial_lrs, stage.lr_scale)
            if dist_info.is_root:
                print(f"[lsrtm] stage {stage_idx} ({remaining}/{stage.epochs} epochs) "
                      f"lr_scale={stage.lr_scale} wavelet_overridden={stage.wavelet is not None}")
            for _ in range(remaining):
                loss_value = self._lsrtm_train_step(
                    spec, lsrtm_solver, stage_wavelet, sources, receivers,
                    vp, ref, obs, optimizer, nshots, dev,
                    dist_info=dist_info,
                )
                losses.append(loss_value)
                if scheduler is not None:
                    scheduler.step()
                _apply_bounds(inv_by_name, bounds_by_name)
                if dist_info.is_root:
                    print(f"[lsrtm] stage {stage_idx} epoch {epoch_global:04d} loss={loss_value:.6e}")

                snapshot_now = (epoch_global % spec.show_every == 0
                                or epoch_global == total_epochs - 1)
                if snapshot_now and dist_info.is_root:
                    np.save(snapshots_dir / f"reflectivity_epoch_{epoch_global:04d}.npy",
                            ref.detach().cpu().numpy())
                    if spec.save_illumination:
                        _save_illumination(lsrtm_solver, snapshots_dir, epoch_global)

                if dist_info.is_root:
                    _save_checkpoint(task_dir, {
                        "epoch": epoch_global,
                        "models": {"reflectivity": ref.detach().cpu().clone()},
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "losses": losses,
                        "torch_rng": torch.get_rng_state(),
                        "numpy_rng": np.random.get_state(),
                    })
                if stopper.should_stop(dist_info):
                    interrupted = True
                    interrupted_at_epoch = epoch_global
                    if dist_info.is_root:
                        print(f"[lsrtm] stopping after epoch {epoch_global} "
                              f"(checkpoint.pt saved). Re-run the same YAML "
                              f"with `resume: true` to continue.")
                    epoch_global += 1
                    break
                epoch_global += 1
            if interrupted:
                break

        stopper.uninstall()

        # Final outputs (rank 0 only).
        artifacts: list[Path] = []
        if dist_info.is_root:
            final_ref_path = out_dir / "reflectivity.npy"
            np.save(final_ref_path, ref.detach().cpu().numpy())
            loss_path = out_dir / "loss.npy"
            np.save(loss_path, np.array(losses, dtype=np.float64))
            artifacts.extend([final_ref_path, loss_path])
            try:
                artifacts.append(_plot_loss_curve(losses, out_dir / "loss.png", title="LSRTM Loss"))
            except Exception as plot_err:  # noqa: BLE001
                print(f"[lsrtm] loss plot skipped: {plot_err}")
        _dist.barrier(dist_info)

        summary = {
            "epochs": total_epochs,
            "epochs_completed": len(losses),
            "final_loss": losses[-1] if losses else None,
            "loss_decreased": (losses[-1] < losses[0]) if len(losses) >= 2 else None,
            "num_stages": len(stages),
            "resumed_from": resume_src,
            "interrupted": interrupted,
            "interrupted_at_epoch": interrupted_at_epoch,
            "world_size": dist_info.world_size,
        }
        return artifacts, summary

    def _lsrtm_train_step(self, spec, lsrtm_solver, wavelet, sources, receivers,
                          vp, ref, obs, optimizer, nshots, dev,
                          *, dist_info=None) -> float:
        import torch

        from sweep_tasks.runtime import distributed as _dist

        if dist_info is None:
            dist_info = getattr(self, "_dist", None) or _dist.init_distributed_if_needed()

        inv_in_order = [ref]
        global_batchsize = min(spec.batchsize, nshots)

        if dist_info.is_root:
            shot_idx_global = np.random.choice(nshots, size=global_batchsize, replace=False)
        else:
            shot_idx_global = None
        shot_idx_global = _dist.broadcast_shot_indices(
            shot_idx_global, global_batchsize, dist_info, src=0
        )
        local_shots = _dist.split_for_rank(shot_idx_global, dist_info)
        chunk_size = spec.train_shot_batchsize or max(len(local_shots), 1)
        chunks = ([local_shots[i:i + chunk_size]
                   for i in range(0, len(local_shots), chunk_size)]
                  if len(local_shots) > 0 else [])

        sample = obs[:1]
        per_shot_numel = int(sample.numel())
        global_norm = float(per_shot_numel * global_batchsize)
        data_mask = self._get_data_mask(spec, obs, dev)
        if data_mask is not None:
            global_norm = float(global_norm * max(float(data_mask.float().mean()), 1.0e-6))

        if spec.optimizer.kind == "lbfgs":
            def _closure():
                optimizer.zero_grad()
                acc_loss = 0.0
                for chunk in chunks:
                    syn = lsrtm_solver(wavelet, sources[chunk], receivers[chunk],
                                       models=[vp, ref])
                    obs_chunk = obs[chunk].to(dev)
                    loss_t = _loss_sum(syn, obs_chunk, spec.loss, _mask_chunk(data_mask, chunk, dev))
                    (loss_t / global_norm).backward()
                    acc_loss += float(loss_t.detach().cpu())
                _zero_top_rows(inv_in_order, spec.freeze_top_n_rows)
                return torch.tensor(acc_loss / global_norm if global_norm else 0.0)
            result = optimizer.step(_closure)
            return float(result) if result is not None else 0.0

        optimizer.zero_grad()
        acc_loss_local = 0.0
        for chunk in chunks:
            syn = lsrtm_solver(wavelet, sources[chunk], receivers[chunk],
                               models=[vp, ref])
            obs_chunk = obs[chunk].to(dev)
            loss_t = _loss_sum(syn, obs_chunk, spec.loss, _mask_chunk(data_mask, chunk, dev))
            (loss_t / global_norm).backward()
            acc_loss_local += float(loss_t.detach().cpu())
        _zero_top_rows(inv_in_order, spec.freeze_top_n_rows)
        _dist.all_reduce_grad_sum(inv_in_order, dist_info)
        optimizer.step()
        acc_loss_global = _dist.all_reduce_scalar_sum(acc_loss_local, dist_info)
        return acc_loss_global / global_norm if global_norm else 0.0
