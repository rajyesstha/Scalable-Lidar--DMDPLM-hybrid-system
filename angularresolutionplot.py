# Created on Tue Jul 1 09:28:17 2025
# Author: Rajesh Shrestha
# Journal-ready Applied Optics style figure

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",

    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# ========================== FIGURE SETUP ==========================
fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=300, layout="constrained")

# Color palette aligned exactly with the regression figure style.
order_colors = {
    -2: "#8080FF",
    -1: "#00FFFF",
    0:  "#FF00FF",
    1:  "#C8C800",
    2:  "#FF0000",
    3:  "#00FF00",
    4:  "#0000FF",
}
marker_edge_color = "#222222"
grid_color = "#b0b0b0"

# ========================== PLOT ==========================
for i, order in enumerate(orders):
    color = order_colors.get(int(order), "#4C72B0")

    if i > 0:
        ax.plot(
            [orders[i - 1], order],
            [angles[i - 1], angles[i]],
            color="#888888",
            linewidth=1.2,
            zorder=2,
        )

    ax.scatter(
        [order],
        [angles[i]],
        s=55,
        color=color,
        edgecolor=marker_edge_color,
        linewidth=0.8,
        zorder=3,
    )

# ========================== AXIS LABELS ==========================
ax.set_xlabel(r"Diffraction order", fontsize=13, labelpad=8)
ax.set_ylabel(r"Angular resolution, $\theta_{\text{Nyq}}$ (deg)", fontsize=13, labelpad=8)

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
ax.set_axisbelow(True)
ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.6,
    color=grid_color,
    alpha=0.45,
    zorder=0
)

# ========================== SPINES AND TICKS ==========================
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for spine in [ax.spines["left"], ax.spines["bottom"]]:
    spine.set_linewidth(0.8)

ax.tick_params(
    axis="both",
    which="major",
    direction="out",
    top=False,
    right=False,
    length=3.0,
    pad=3,
    colors="black"
)

# ========================== LEGEND ==========================
legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        color="white",
        markerfacecolor=color,
        markeredgecolor=marker_edge_color,
        markeredgewidth=0.8,
        markersize=5.5,
        label=f"Order {order}",
    )
    for order, color in order_colors.items()
]

leg = ax.legend(
    handles=legend_handles,
    title="Diffraction Order",
    loc="upper right",
    frameon=True,
    framealpha=0.95,
    edgecolor="#b0b0b0",
    fontsize=9,
    title_fontsize=10,
)
leg.get_frame().set_linewidth(0.6)

# ========================== ANNOTATIONS ==========================
if SHOW_POINT_LABELS:
    for x, y in zip(orders, angles):
        ax.annotate(
            f"{y:.3f}°",
            xy=(x, y),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
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