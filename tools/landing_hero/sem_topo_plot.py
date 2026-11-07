"""Draw the SEM spotlight image from sem_topo.npz: vz under the two-hill
surface, the element edges of the body-fitted mesh, in the site palette.

    python sem_topo_plot.py [time_s] [out.png] [narrow]

`narrow` draws the tighter crop used on phones.
"""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

HERE = pathlib.Path(__file__).parent
d = np.load(HERE / "sem_topo.npz")
t_show = float(sys.argv[1]) if len(sys.argv) > 1 else 0.30
out = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "sem_topo.png")

PAPER, AIR, INK = "#F0ECE4", "#F8F6F1", "#1D1D1F"
STAR = "#ED8B2E"
cmap = LinearSegmentedColormap.from_list(
    "site", ["#0B5E52", "#1AA690", "#A9D9CF", PAPER, "#F5C99A", "#ED8B2E", "#A84E0E"],
    N=257)                              # odd, so that zero is exactly the paper colour

X_MAX, DH = float(d["x_max"]), float(d["dh"])
narrow = len(sys.argv) > 3 and sys.argv[3] == "narrow"
# 1.8 km centred on the valley (1.2 km on phones)
X0, X1, ZSHOW = (630.0, 1830.0, 720.0) if narrow else (330.0, 2130.0, 900.0)
xg, ztop = d["xg"], d["ztop"]
k = int(np.argmin(np.abs(d["snap_ts"] - t_show)))
w = d["w"][k].astype(float)
zq = np.arange(w.shape[0])[:, None] * DH
w[zq < ztop[None, :]] = np.nan
w /= np.nanpercentile(np.abs(w), 99.5)

# element edges inside the shown box
xe, ze = d["x_e"], d["z_e"]
segs = []
for e in range(xe.shape[0]):
    for a, b in ((xe[e, 0, :], ze[e, 0, :]), (xe[e, -1, :], ze[e, -1, :]),
                 (xe[e, :, 0], ze[e, :, 0]), (xe[e, :, -1], ze[e, :, -1])):
        if a.max() < X0 or a.min() > X1 or b.min() > ZSHOW:
            continue
        segs.append(np.stack([a, b], -1))

W_IN = 16.0
fig = plt.figure(figsize=(W_IN, W_IN * ZSHOW / (X1 - X0)), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor(PAPER)
fig.patch.set_facecolor(PAPER)
ax.imshow(w, extent=[0, X_MAX, zq[-1, 0], 0], aspect="auto", cmap=cmap,
          vmin=-1, vmax=1, interpolation="bilinear", zorder=1)
ax.add_collection(LineCollection(segs, colors=INK, linewidths=0.6, alpha=0.22, zorder=2))
ax.fill_between(xg, 0, ztop, color=AIR, zorder=3)
ax.plot(xg, ztop, color=INK, lw=2.0, zorder=4)
ax.plot(d["src_x"], d["src_z"], marker="*", color=STAR, ms=34, mec=INK, mew=1.2, zorder=5)
ax.set_xlim(X0, X1)
ax.set_ylim(ZSHOW, 0)
ax.axis("off")
fig.savefig(out, facecolor=PAPER)
print(f"t = {d['snap_ts'][k]:.2f} s ->", out)
