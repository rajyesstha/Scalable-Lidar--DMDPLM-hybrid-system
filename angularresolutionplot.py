# Created on Tue Jul 1 09:28:17 2025
# Author: Rajesh Shrestha
# Converted to Python

import numpy as np
import matplotlib.pyplot as plt

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

# ========================== FIGURE SETUP ==========================
fig, ax = plt.subplots(
    figsize=(12, 8),
    dpi=300,
    facecolor="white"
)

# ========================== PLOT ==========================
ax.plot(
    orders,
    angles,
    "-o",
    color=(0, 0.25, 0.48),          # Professional navy blue
    linewidth=3,
    markersize=12,
    markeredgecolor=(0.5, 0, 0),    # Dark red edge
    markerfacecolor=(0.8, 0.2, 0.2) # Red face
)

# ========================== AXIS FORMATTING ==========================
ax.set_facecolor("white")

ax.tick_params(
    axis="both",
    which="both",
    direction="in",
    labelsize=28,
    width=1.5
)

for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_visible(True)

# Font setup
plt.rcParams["font.family"] = "Times New Roman"

# ========================== LABELS AND TITLE ==========================
ax.set_title(
    r"Angular Resolution ($R_x$) vs. Diffraction Orders",
    fontsize=36,
    pad=20
)

ax.set_xlabel(
    r"Diffraction Order ($m$)",
    fontsize=32,
    labelpad=12
)

ax.set_ylabel(
    "Angular Resolution (deg)",
    fontsize=32,
    labelpad=12
)

# ========================== LIMITS AND TICKS ==========================
ax.set_xlim(-2.5, 4.5)
ax.set_xticks(np.arange(-2, 5, 1))

ax.set_ylim(0.12, 0.25)
ax.set_yticks(np.arange(0.12, 0.251, 0.02))

# ========================== GRID ==========================
ax.grid(
    True,
    linestyle=":",
    alpha=0.4
)

# ========================== ANNOTATIONS ==========================
for x, y in zip(orders, angles):
    ax.text(
        x,
        y - 0.02,
        f"{y:.3f}",
        ha="center",
        va="bottom",
        fontsize=22,
        fontweight="bold",
        color=(0.2, 0.2, 0.2),
        fontname="Times New Roman"
    )

# ========================== LAYOUT ==========================
plt.tight_layout()

# ========================== EXPORT ==========================
# Save high-resolution PNG
plt.savefig(
    "Angular_Resolution_Plot.png",
    dpi=600,
    facecolor="white",
    bbox_inches="tight"
)

# Save vector PDF for journal submission
plt.savefig(
    "Angular_Resolution_Plot.pdf",
    facecolor="white",
    bbox_inches="tight"
)

plt.show()