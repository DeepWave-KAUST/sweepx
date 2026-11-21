"""Draw the 3-D anisotropy spotlight images from sem_aniso3d.npz: vz on the
x-z plane through the force in the isotropic, VTI and tilted media, side by
side (and the tilted one alone, for phones), in the site palette.

    python sem_aniso3d_plot.py [wide.png] [narrow.png]
"""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = pathlib.Path(__file__).parent
d = np.load(HERE / "sem_aniso3d.npz")
wide = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "sem_aniso3d.png")
narrow = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "sem_aniso3d_tilted.png")

PAPER = "#F0ECE4"
cmap = LinearSegmentedColormap.from_list(
    "site", ["#0B5E52", "#1AA690", "#A9D9CF", PAPER, "#F5C99A", "#ED8B2E", "#A84E0E"],
    N=257)                              # odd, so that zero is exactly the paper colour
CROP = (0.11, 0.89)                     # the part of the cube the fronts have reached
GAMMA = 0.6                             # sign(v) |v|^0.6: the weaker P fronts still show

peak = np.percentile(np.abs(d["iso_xz"]), 99.5)


def panel(v):
    n = v.shape[0]
    i0, i1 = int(CROP[0] * n), int(CROP[1] * n)
    v = v[i0:i1, i0:i1] / peak
    return np.sign(v) * np.minimum(np.abs(v), 1.0) ** GAMMA


panels = [panel(d[k]) for k in ("iso_xz", "vti_xz", "tti_xz")]


def draw(images, out, side=6.0, gap=0.24):
    """Square panels of `side` inches, `gap` inches apart, at 100 dpi."""
    n = len(images)
    w_in = n * side + (n - 1) * gap
    fig = plt.figure(figsize=(w_in, side), dpi=100)
    fig.patch.set_facecolor(PAPER)
    for j, im in enumerate(images):
        ax = fig.add_axes([j * (side + gap) / w_in, 0, side / w_in, 1])
        ax.imshow(im, cmap=cmap, vmin=-1, vmax=1, interpolation="bicubic")
        ax.axis("off")
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)
    print("->", out)


draw(panels, wide)
draw(panels[2:], narrow)
