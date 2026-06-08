# Created on Tue Jul 1 09:28:17 2025
# Author: Rajesh Shrestha
# Journal-ready Applied Optics style figure

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path

# ========================== DATA ==========================
orders = np.arange(-2, 5)

angles = np.array([
    0.229183152,
    0.152788768,
    0.229183152,
    0.152788768,
    0.152788768,
    0.152788768,
    0.152788768
])

# ========================== OUTPUT ==========================
outdir = Path(".")
basename = "Angular_Resolution_vs_Diffraction_Order"

SHOW_POINT_LABELS = True   # Set False for cleaner journal version

# ========================== FONT / EXPORT CONFIG ==========================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",

    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,

    "axes.linewidth": 0.75,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# ========================== FIGURE SETUP ==========================
# Wider than strict one-column, but much safer for labels and annotations.
fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=600)

# Manually control margins so labels never crop.
fig.subplots_adjust(
    left=0.16,
    right=0.97,
    bottom=0.18,
    top=0.92
)

# ========================== PLOT ==========================
ax.plot(
    orders,
    angles,
    "-o",
    color="black",
    linewidth=1.1,
    markersize=4.6,
    markerfacecolor="white",
    markeredgecolor="black",
    markeredgewidth=1.0,
    zorder=3
)

# ========================== AXIS LABELS ==========================
ax.set_xlabel(r"Diffraction order", labelpad=5)
ax.set_ylabel(r"Angular resolution, $\theta_{\text{Nyq}}$ (deg)", labelpad=7)

# Do NOT use a title for journal manuscript figures.
# Put the title/explanation in the caption instead.
# If you absolutely need a title for internal use, uncomment:
# ax.set_title("Angular Resolution vs. Diffraction Order", fontsize=10, pad=8)

# ========================== LIMITS AND TICKS ==========================
ax.set_xlim(-2.35, 4.35)
ax.set_xticks(orders)

ax.set_ylim(0.135, 0.245)
ax.set_yticks([0.15, 0.18, 0.21, 0.24])
ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

# ========================== GRID ==========================
ax.grid(
    axis="y",
    linestyle=(0, (3, 3)),
    linewidth=0.45,
    color="0.82",
    zorder=0
)

# ========================== SPINES AND TICKS ==========================
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(
    axis="both",
    which="major",
    direction="out",
    top=False,
    right=False,
    pad=3
)

# ========================== ANNOTATIONS ==========================
if SHOW_POINT_LABELS:
    for x, y in zip(orders, angles):
        ax.text(
            x,
            y + 0.0045,
            f"{y:.3f}°",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color="0.25",
            clip_on=False
        )

# ========================== EXPORT ==========================
# Do NOT use extremely small pad_inches here.
fig.savefig(
    outdir / f"{basename}.pdf",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.08
)

fig.savefig(
    outdir / f"{basename}.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.08
)

fig.savefig(
    outdir / f"{basename}.svg",
    bbox_inches="tight",
    pad_inches=0.08
)

plt.show()