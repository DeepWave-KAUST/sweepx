"""Render the landing animation frames from sim.npz.

Timeline (30 fps): forward -> residual -> adjoint (+ gradient building) ->
gradient hold -> fade back to the start, so the loop is seamless.  The frames
carry no text: the step labels are HTML over the video, and the step
boundaries (PHASES below) are repeated in overrides/home.html.
Frames go to frames/NNNN.png; encode them with

    ffmpeg -framerate 30 -i frames/%04d.png -c:v libvpx-vp9 -pix_fmt yuv420p \
        -crf 40 -b:v 0 -row-mt 1 -an hero_loop.webm
    ffmpeg -framerate 30 -i frames/%04d.png -c:v libx264 -pix_fmt yuv420p \
        -crf 26 -preset slow -movflags +faststart -an hero_loop.mp4

and use frame 0290 as the poster.
"""
import pathlib
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.colors import to_rgb
from scipy.ndimage import gaussian_filter

HERE = pathlib.Path(__file__).parent
S = np.load(HERE / "sim.npz")
OUT = HERE / "frames"
OUT.mkdir(exist_ok=True)

FPS = 30
W, H, DPI = 1840, 530, 100

# palette (site tokens)
PAPER = "#F0ECE4"
CARD = "#E7E3DA"
INK = "#1D1D1F"
TEAL, ORANGE, PURPLE = "#1AA690", "#ED8B2E", "#A855F7"
ADJ_NEG = "#7C83E8"
LAYER_COLORS = ["#CFE8E0", "#E2E8DA", "#D3E7E3", "#E4E6C9", "#EEDCC0"]

nz, nx = S["vp0"].shape
dh = float(S["dh"])
dt = float(S["dt"])
nt = int(S["nt"])
k_every = int(S["snap_every"])
u = S["u"].astype(np.float32)
lam = S["lam"].astype(np.float32)
gcum = S["grad_cum"]
ns = u.shape[0]
tops = S["tops"]
src = S["src"]
rec_ix = S["rec_ix"]
d_obs = S["d_obs"]
res = S["res"]

# model drawing box (pixels, origin top-left)
MX0, MY0 = 40, 100
CELL = 2.4
MW, MH = nx * CELL, nz * CELL          # 1152 x 384
# gather card box
GX0, GY0, GW, GH = 1262, 40, 520, 450

# ----- amplitude scales -------------------------------------------------
def frame_scale(f, skip=6):
    return np.array([np.percentile(np.abs(a[skip:]), 99.6) + 1e-30 for a in f])

u_scale = gaussian_filter(frame_scale(u), 3)
lam_scale = gaussian_filter(frame_scale(lam), 3)
u_scale = np.maximum(u_scale, 0.04 * u_scale.max())
lam_scale = np.maximum(lam_scale, 0.04 * lam_scale.max())

taper = np.ones((nz, 1), np.float32)
taper[:10, 0] = np.sin(0.5 * np.pi * np.arange(10) / 10) ** 2   # display taper near the surface
g_final = gcum[0] * taper
g_scale = np.percentile(np.abs(g_final), 99.7)


def rgba_signed(a, pos, neg, gain=2.5, alpha_max=0.9, gamma=0.75):
    v = np.tanh(gain * a)
    out = np.zeros(a.shape + (4,), np.float32)
    p, n = np.array(to_rgb(pos)), np.array(to_rgb(neg))
    out[..., :3] = np.where(v[..., None] >= 0, p, n)
    out[..., 3] = alpha_max * np.abs(v) ** gamma
    return out


# ----- timeline ---------------------------------------------------------
T_FWD, T_RES, T_ADJ, T_HOLD, T_FADE = 3.6, 1.1, 3.8, 1.6, 0.6
PHASES = [("forward", T_FWD), ("residual", T_RES), ("adjoint", T_ADJ),
          ("gradient", T_HOLD), ("fade", T_FADE)]
N_FRAMES = int(round(sum(d for _, d in PHASES) * FPS))


def state(i):
    t = i / FPS
    for name, d in PHASES:
        if t < d:
            return name, t / d
        t -= d
    return "fade", 1.0


def ease(x):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0, 1))


# ----- gather drawing ---------------------------------------------------
TR = np.arange(0, len(rec_ix), 3)
tgain = (np.arange(nt) * dt / (nt * dt)) ** 1.3
obs_g = d_obs[:, TR] * tgain[:, None]
obs_n = np.tanh(obs_g / np.percentile(np.abs(obs_g), 88.0))
res_n = np.tanh(res[:, TR] / np.percentile(np.abs(res), 99.0))
tt = np.arange(nt) * dt
pad_x = 26
gx = GX0 + pad_x + np.linspace(0, GW - 2 * pad_x, len(TR))
gy_top, gy_bot = GY0 + 26, GY0 + GH - 26
gy = gy_top + (gy_bot - gy_top) * tt / tt[-1]
dx_tr = gx[1] - gx[0]


def draw_wiggles(ax, data, upto, fill, alpha=1.0, consumed_from=None):
    amp = 0.9 * dx_tr
    for j in range(len(TR)):
        tr = data[:, j] * amp
        sl = slice(0, upto)
        x = gx[j] + tr[sl]
        y = gy[sl]
        a = alpha
        if consumed_from is not None:
            # samples later than the adjoint cursor have been injected
            k = consumed_from
            ax.plot(gx[j] + tr[k:upto], gy[k:upto], color=INK, lw=0.7, alpha=0.18 * a)
            x, y = x[:k], y[:k]
        ax.fill_betweenx(y, gx[j], x, where=x > gx[j], color=fill, alpha=0.85 * a, lw=0)
        ax.plot(x, y, color=INK, lw=0.7, alpha=0.75 * a)


# ----- static model layers ---------------------------------------------
xs = MX0 + np.arange(nx) * CELL


def draw_model(ax):
    zb = [tops[i] for i in range(len(tops))] + [np.full(nx, nz * dh)]
    for i in range(len(tops)):
        y0 = MY0 + zb[i] / dh * CELL
        y1 = MY0 + zb[i + 1] / dh * CELL
        ax.fill_between(xs, y0, y1, color=LAYER_COLORS[i], lw=0, zorder=1)
        if i > 0:
            ax.plot(xs, y0, color="white", lw=2.2, alpha=0.9, zorder=2)


# ----- frame ----------------------------------------------------------------


def render(i):
    phase, p = state(i)
    fig = plt.figure(figsize=((W + 0.5) / DPI, (H + 0.5) / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)

    draw_model(ax)
    ext = (MX0, MX0 + MW, MY0 + MH, MY0)

    # wavefields / gradient overlays
    g_alpha = 0.0
    if phase == "forward":
        k = min(int(p * (ns - 1)), ns - 1)
        ax.imshow(rgba_signed(u[k] / u_scale[k], TEAL, ORANGE, gain=4.0, gamma=1.0), extent=ext,
                  interpolation="bilinear", zorder=3)
        cursor = (k * k_every)
    elif phase == "residual":
        cursor = nt - 1
        fadeu = 1 - ease(p * 2)
        if fadeu > 0:
            im = rgba_signed(u[-1] / u_scale[-1], TEAL, ORANGE)
            im[..., 3] *= fadeu
            ax.imshow(im, extent=ext, interpolation="bilinear", zorder=3)
    elif phase == "adjoint":
        k = min(int((1 - p) * (ns - 1)), ns - 1)
        cursor = k * k_every
        g = gcum[k] * taper
        im = rgba_signed(g / g_scale, ORANGE, TEAL, gain=1.6, alpha_max=0.95, gamma=0.9)
        ax.imshow(im, extent=ext, interpolation="bilinear", zorder=3)
        fw = rgba_signed(u[k] / u_scale[k], TEAL, ORANGE, alpha_max=0.28)
        ax.imshow(fw, extent=ext, interpolation="bilinear", zorder=4)
        ax.imshow(rgba_signed(lam[k] / lam_scale[k], PURPLE, ADJ_NEG, gain=1.4, alpha_max=0.6, gamma=1.0),
                  extent=ext, interpolation="bilinear", zorder=5)
    elif phase in ("gradient", "fade"):
        cursor = 0
        g_alpha = 1.0 if phase == "gradient" else 1 - ease(p)
        im = rgba_signed(g_final / g_scale, ORANGE, TEAL, gain=1.6, alpha_max=0.95, gamma=0.9)
        im[..., 3] *= g_alpha
        ax.imshow(im, extent=ext, interpolation="bilinear", zorder=3)

    # surface line, receivers, source
    ax.plot([MX0, MX0 + MW], [MY0, MY0], color=INK, lw=1.0, alpha=0.55, zorder=6)
    rx = MX0 + rec_ix * CELL
    rec_col = PURPLE if phase == "adjoint" else INK
    rec_a = 0.9 if phase == "adjoint" else 0.45
    ax.scatter(rx, np.full_like(rx, MY0, dtype=float), s=5, color=rec_col, alpha=rec_a,
               lw=0, zorder=7)
    sx, sy = MX0 + src[0] * CELL, MY0
    if phase == "forward" and p < 0.35:
        r = 10 + 60 * (p / 0.35)
        ax.add_patch(Circle((sx, sy), r, fill=False, ec=ORANGE, lw=2, alpha=1 - p / 0.35, zorder=8))
    ax.add_patch(Circle((sx, sy), 9, color=ORANGE, zorder=9))
    ax.add_patch(Circle((sx, sy), 15, color=ORANGE, alpha=0.25, zorder=8))

    # gather card
    ax.add_patch(FancyBboxPatch((GX0, GY0), GW, GH, boxstyle="round,pad=0,rounding_size=16",
                                fc=CARD, ec="#D9D4CA", lw=1.2, zorder=1))
    if phase == "forward":
        draw_wiggles(ax, obs_n, cursor + 1, ORANGE)
    elif phase == "residual":
        a = ease(p * 1.6)
        draw_wiggles(ax, obs_n, nt, ORANGE, alpha=1 - a)
        draw_wiggles(ax, res_n, nt, PURPLE, alpha=a)
    elif phase == "adjoint":
        draw_wiggles(ax, res_n, nt, PURPLE, consumed_from=cursor)
    else:
        draw_wiggles(ax, res_n, nt, PURPLE, alpha=0.18 if phase == "gradient" else 0.18 * g_alpha)
    if phase in ("forward", "adjoint"):
        yc = gy[min(cursor, nt - 1)]
        col = ORANGE if phase == "forward" else PURPLE
        ax.plot([GX0 + 12, GX0 + GW - 12], [yc, yc], color=col, lw=1.6, alpha=0.9, zorder=5)

    fig.savefig(OUT / f"{i:04d}.png", dpi=DPI, facecolor=PAPER)
    plt.close(fig)
    return i


if __name__ == "__main__":
    frames = range(N_FRAMES) if len(sys.argv) < 2 else [int(a) for a in sys.argv[1:]]
    with Pool(24) as pool:
        for _ in pool.imap_unordered(render, frames):
            pass
    print("frames:", N_FRAMES)
