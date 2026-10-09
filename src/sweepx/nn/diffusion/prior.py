"""Use a trained DDPM as a plug-and-play (PnP/RED) prior for FWI.

A ``DiffusionVelocityPrior`` wraps a trained ``UNet2D`` + ``GaussianDiffusion``
checkpoint and turns it into a *denoiser* ``D(vp)`` acting on a physical velocity
field (m/s).  From that denoiser we form the RED / proximal prior term used to
regularise FWI:

    L_prior(vp) = 0.5 * mean( (vp - D(vp).detach())^2 )   # grad = vp - D(vp)

so adding ``lambda * L_prior`` to the data misfit pulls the velocity toward the
manifold learned by the diffusion model, while keeping the whole objective
differentiable w.r.t. either the pixels or an INR's parameters.

The denoiser handles the size / distribution mismatch between the (small,
64x64, CurveVel-A) generative model and an arbitrary FWI grid via two modes:

* ``mode="patch"`` — slide a 64x64 window (native training size, in-distribution),
  denoise every patch, overlap-average.  Preserves fine detail.
* ``mode="resize"`` — bilinearly resize the whole field to 64x64, denoise, resize
  back.  A coarse, whole-model structural prior.

Denoising itself is DiffPIR-style purification: treat the current estimate as the
state at timestep ``t_start = strength * T``, then run deterministic DDIM reverse
sampling down to t=0.  Larger ``strength`` = stronger projection onto the prior.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .unet import UNet2D
from .ddpm import GaussianDiffusion


class DiffusionVelocityPrior:
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        mode: str = "patch",
        strength: float = 0.3,
        ddim_steps: int = 10,
        patch: int = 64,
        stride: int = 32,
        vmin: float | None = None,
        vmax: float | None = None,
        use_ema: bool = True,
    ):
        self.device = device
        self.mode = mode
        self.strength = float(strength)
        self.ddim_steps = int(ddim_steps)
        self.patch = int(patch)
        self.stride = int(stride)

        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ck["config"]
        stats = ck.get("stats", {})
        self.vmin = float(vmin if vmin is not None else stats.get("vmin", 1500.0))
        self.vmax = float(vmax if vmax is not None else stats.get("vmax", 4500.0))
        self.train_size = int(cfg.get("target_size", 64))
        self.ckpt_step = int(ck.get("step", -1))

        self.model = UNet2D(
            in_channels=1, base_channels=cfg["base_channels"],
            channel_mults=tuple(cfg["channel_mults"]), num_res_blocks=cfg["num_res_blocks"],
            attn_resolutions=tuple(cfg["attn_res"]), dropout=cfg["dropout"],
            image_size=self.train_size,
        ).to(device)
        sd = ck["ema"] if (use_ema and "ema" in ck) else ck["model"]
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.diff = GaussianDiffusion(
            cfg["timesteps"], cfg["schedule"], cfg["param"]
        ).to(device)

    # ---- normalisation (physical m/s <-> [-1,1]) ----
    def to_norm(self, vp: torch.Tensor) -> torch.Tensor:
        vp = vp.clamp(self.vmin, self.vmax)
        return (vp - self.vmin) / (self.vmax - self.vmin) * 2.0 - 1.0

    def to_phys(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 1.0) * 0.5 * (self.vmax - self.vmin) + self.vmin

    # ---- core denoiser on a batch of (B,1,S,S) in [-1,1] ----
    @torch.no_grad()
    def _purify(self, x_norm: torch.Tensor) -> torch.Tensor:
        T = self.diff.num_timesteps
        t_start = max(1, min(T - 1, int(self.strength * T)))
        # treat the current estimate as the state at t_start (deterministic, no re-noise)
        x = x_norm
        times = torch.linspace(t_start, 0, self.ddim_steps + 1, dtype=torch.long).tolist()
        for cur, nxt in zip(times[:-1], times[1:]):
            t = torch.full((x.shape[0],), cur, device=x.device, dtype=torch.long)
            eps = self.model(x, t)
            x0 = self.diff._pred_x0_from(x, t, eps).clamp(-1, 1)
            if nxt <= 0:
                x = x0
                break
            eps = self.diff._pred_eps_from_x0(x, t, x0)
            a_next = self.diff.alphas_cumprod[nxt]
            x = torch.sqrt(a_next) * x0 + torch.sqrt((1 - a_next).clamp(min=0)) * eps
        return x0

    # ---- denoise a full field, returned in normalised [-1,1] space ----
    @torch.no_grad()
    def denoise_norm(self, vp: torch.Tensor) -> torch.Tensor:
        x = self.to_norm(vp)[None, None]  # (1,1,nz,nx)
        if self.mode == "resize":
            xs = F.interpolate(x, size=(self.train_size, self.train_size),
                               mode="bilinear", align_corners=False)
            ys = self._purify(xs)
            y = F.interpolate(ys, size=x.shape[-2:], mode="bilinear", align_corners=False)
        elif self.mode == "patch":
            y = self._denoise_patches(x)
        else:
            raise ValueError(self.mode)
        return y[0, 0]

    # ---- denoise a full physical velocity field (nz,nx) ----
    @torch.no_grad()
    def denoise(self, vp: torch.Tensor) -> torch.Tensor:
        return self.to_phys(self.denoise_norm(vp))

    def _denoise_patches(self, x: torch.Tensor) -> torch.Tensor:
        S, st = self.train_size, self.stride
        _, _, H, W = x.shape
        # reflect-pad so patches tile and cover the borders
        padH = (S - H % st) % st if H > S else max(0, S - H)
        padW = (S - W % st) % st if W > S else max(0, S - W)
        xp = F.pad(x, (0, padW, 0, padH), mode="reflect")
        Hp, Wp = xp.shape[-2:]
        zs = list(range(0, max(1, Hp - S + 1), st));  zs = zs or [0]
        ws = list(range(0, max(1, Wp - S + 1), st));  ws = ws or [0]
        if zs[-1] != Hp - S:
            zs.append(Hp - S)
        if ws[-1] != Wp - S:
            ws.append(Wp - S)
        acc = torch.zeros_like(xp)
        cnt = torch.zeros_like(xp)
        patches, coords = [], []
        for z0 in zs:
            for w0 in ws:
                patches.append(xp[:, :, z0:z0 + S, w0:w0 + S])
                coords.append((z0, w0))
        batch = torch.cat(patches, 0)  # (P,1,S,S)
        den = self._purify(batch)
        # Hann window so overlapping patches blend smoothly instead of leaving
        # hard seams (the main source of "blocky" reassembly artifacts).
        w1 = torch.hann_window(S, periodic=False, device=x.device).clamp(min=1e-3)
        win = (w1[:, None] * w1[None, :])[None, None]
        for i, (z0, w0) in enumerate(coords):
            acc[:, :, z0:z0 + S, w0:w0 + S] += den[i:i + 1] * win
            cnt[:, :, z0:z0 + S, w0:w0 + S] += win
        y = acc / cnt.clamp(min=1e-6)
        return y[:, :, :H, :W]

    # ---- RED / proximal prior term (differentiable in vp) ----
    def red_loss(self, vp: torch.Tensor) -> torch.Tensor:
        """RED term computed in NORMALISED [-1,1] space so its magnitude is O(1)
        and comparable to a normalised data misfit.

        L = 0.5 * mean((x_norm(vp) - D(vp).detach())^2), with x_norm = to_norm(vp)
        differentiable in vp (linear in-range), so grad_vp carries the correct
        2/(vmax-vmin) chain-rule factor automatically.
        """
        x = self.to_norm(vp)                       # differentiable in vp (in-range)
        d = self.denoise_norm(vp.detach())         # denoised, normalised, detached
        return 0.5 * (x - d).pow(2).mean()

    def residual(self, vp: torch.Tensor) -> torch.Tensor:
        """vp - D(vp) in physical units (the raw prior descent direction)."""
        return vp - self.denoise(vp)

    # ---- Score Distillation Sampling (SDS) gradient ----
    @torch.no_grad()
    def _sds_field(self, x: torch.Tensor, t_lo: float, t_hi: float) -> torch.Tensor:
        """SDS gradient field g = w(t)*(eps_phi(x_t,t) - eps) for x in [-1,1],
        shape (1,1,H,W).  w(t)=1.  Tiled to the 64x64 training size."""
        T = self.diff.num_timesteps
        S, st = self.train_size, self.stride
        _, _, H, W = x.shape
        if self.mode == "resize":
            xs = F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False)
            t = torch.randint(int(t_lo * T), int(t_hi * T), (xs.shape[0],), device=x.device)
            noise = torch.randn_like(xs)
            x_t = (self.sqrt_acp(t) * xs + self.sqrt_omacp(t) * noise)
            g = self.model(x_t, t) - noise
            return F.interpolate(g, size=(H, W), mode="bilinear", align_corners=False)
        # patch mode
        padH = (S - H % st) % st if H > S else max(0, S - H)
        padW = (S - W % st) % st if W > S else max(0, S - W)
        xp = F.pad(x, (0, padW, 0, padH), mode="reflect")
        Hp, Wp = xp.shape[-2:]
        zs = list(range(0, max(1, Hp - S + 1), st)) or [0]
        ws = list(range(0, max(1, Wp - S + 1), st)) or [0]
        if zs[-1] != Hp - S:
            zs.append(Hp - S)
        if ws[-1] != Wp - S:
            ws.append(Wp - S)
        patches, coords = [], []
        for z0 in zs:
            for w0 in ws:
                patches.append(xp[:, :, z0:z0 + S, w0:w0 + S]); coords.append((z0, w0))
        batch = torch.cat(patches, 0)
        t = torch.randint(int(t_lo * T), int(t_hi * T), (batch.shape[0],), device=x.device)
        noise = torch.randn_like(batch)
        x_t = self.sqrt_acp(t) * batch + self.sqrt_omacp(t) * noise
        gb = self.model(x_t, t) - noise
        acc = torch.zeros_like(xp); cnt = torch.zeros_like(xp)
        for i, (z0, w0) in enumerate(coords):
            acc[:, :, z0:z0 + S, w0:w0 + S] += gb[i:i + 1]
            cnt[:, :, z0:z0 + S, w0:w0 + S] += 1.0
        return (acc / cnt.clamp(min=1.0))[:, :, :H, :W]

    def sqrt_acp(self, t):
        return self.diff.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)

    def sqrt_omacp(self, t):
        return self.diff.sqrt_one_minus_acp[t].reshape(-1, 1, 1, 1)

    def sds_loss(self, vp: torch.Tensor, t_lo: float = 0.02, t_hi: float = 0.5) -> torch.Tensor:
        """Surrogate whose grad w.r.t. vp is the SDS prior gradient (score-matching
        distillation of the diffusion model into whatever produces vp — pixels or
        an INR).  Computed in normalised space; magnitude O(1)."""
        x = self.to_norm(vp)
        g = self._sds_field(x[None, None], t_lo, t_hi)[0, 0]
        return (x * g.detach()).mean()
