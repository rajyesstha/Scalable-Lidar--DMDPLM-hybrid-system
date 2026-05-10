# =============================================================================
# newphasevisualization.py
# Professional 3x3 CGH/Phase Montage for Journal Publication
# Converted from MATLAB to Python
# Updated with matched horizontal and vertical axis spacing
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


# ========================== PARAMETERS ==========================
final_size = 512
cell_rows = final_size // 2
cell_cols = final_size // 2

# Grating parameters
Lambda_H = 2.1
Lambda_D = 1.5

# 3x3 steering configuration
tile_num = np.array([
    [1, 2, 3],
    [5, 9, 4],
    [6, 7, 8]
])

theta_map = np.array([
    [135, 90, 45],
    [180, np.nan, 0],
    [225, 270, 315]
], dtype=float)

Lambda_map = np.array([
    [Lambda_D, Lambda_H, Lambda_D],
    [Lambda_H, np.nan, Lambda_H],
    [Lambda_D, Lambda_H, Lambda_D]
], dtype=float)


# ========================== PLM PHASE CALIBRATION ==========================
thresholds_pct = np.array([
    1.37420442, 2.140099448, 3.339651934, 4.665104972,
    5.801707182, 7.928027624, 11.68755801, 20.25229282,
    27.6139779, 30.60411602, 35.11205525, 41.69735359,
    48.24418232, 55.61985635, 65.7058895, 100
])

lambda_nm = 905
max_displacement_nm = 296.25

displacements_nm = thresholds_pct / 100 * max_displacement_nm
phase_pi_905 = (4 * displacements_nm) / lambda_nm
phase_max_pi = np.max(phase_pi_905)


# ========================== 2x2 BINARY BIT TILES ==========================
bit_tiles = [
    np.array([[0, 1], [0, 1]]),
    np.array([[0, 1], [0, 0]]),
    np.array([[0, 0], [0, 1]]),
    np.array([[0, 1], [1, 1]]),

    np.array([[0, 0], [0, 0]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 0], [1, 1]]),
    np.array([[0, 0], [1, 0]]),

    np.array([[1, 1], [0, 1]]),
    np.array([[1, 1], [0, 0]]),
    np.array([[1, 0], [0, 1]]),
    np.array([[1, 0], [0, 0]]),

    np.array([[1, 1], [1, 1]]),
    np.array([[1, 1], [1, 0]]),
    np.array([[1, 0], [1, 1]]),
    np.array([[1, 0], [1, 0]])
]


# ========================== VISUALIZATION MODE ==========================
show_phase_profile = True
# True  = phase profile
# False = binary CGH


# ========================== LOCAL FUNCTIONS ==========================
def make_index_map(cell_rows, cell_cols, gratingperiod, theta, thresholds_pct):
    flip_vert = False
    flip_horiz = False

    if 45 <= theta < 180:
        theta_eff = (theta + 180) % 360
        flip_vert = True

    elif 180 <= theta < 225:
        theta_eff = (theta + 180) % 360
        flip_horiz = True

    else:
        theta_eff = theta

    t_rad = np.deg2rad(theta_eff)

    i, j = np.meshgrid(
        np.arange(cell_rows),
        np.arange(cell_cols),
        indexing="ij"
    )

    v = j * np.cos(t_rad) - i * np.sin(t_rad) + 1e-9
    n = np.mod((100 / abs(gratingperiod)) * v, 100)

    idx_map = np.searchsorted(thresholds_pct, n, side="left") + 1
    idx_map = np.clip(idx_map, 1, len(thresholds_pct))

    if flip_vert:
        idx_map = np.flipud(idx_map)

    elif flip_horiz:
        idx_map = np.fliplr(idx_map)

    return idx_map.astype(int)


def make_binary_CGH_from_index(idx_map, bit_tiles):
    cell_rows, cell_cols = idx_map.shape
    cgh = np.zeros((2 * cell_rows, 2 * cell_cols))

    for i in range(cell_rows):
        for j in range(cell_cols):
            tile = bit_tiles[idx_map[i, j] - 1]

            r0 = 2 * i
            c0 = 2 * j

            cgh[r0:r0 + 2, c0:c0 + 2] = tile

    return cgh


def make_phase_map(idx_map, phase_pi_905):
    phase_small = phase_pi_905[idx_map - 1]
    phase_map = np.kron(phase_small, np.ones((2, 2)))
    return phase_map


# ========================== FIGURE SETUP ==========================
fig = plt.figure(
    figsize=(7.5, 7.5),
    dpi=150,
    facecolor="white"
)

# Main 3x3 montage region must be square.
# width  = right - left  = 0.86
# height = top - bottom = 0.86
left = 0.02
right = 0.88
bottom = 0.07
top = 0.93

axis_gap = 0.03

gs = fig.add_gridspec(
    3, 3,
    left=left,
    right=right,
    bottom=bottom,
    top=top,
    wspace=axis_gap,
    hspace=axis_gap
)

axes = np.empty((3, 3), dtype=object)

for rr in range(3):
    for cc in range(3):
        axes[rr, cc] = fig.add_subplot(gs[rr, cc])

# ========================== PLOT 3x3 MONTAGE ==========================
last_im = None

for r in range(3):
    for c in range(3):
        ax = axes[r, c]

        if np.isnan(theta_map[r, c]):
            if show_phase_profile:
                img = np.zeros((final_size, final_size))
            else:
                img = np.ones((final_size, final_size))

            top_label = f"Tile {tile_num[r, c]}: Flat"

        else:
            theta = theta_map[r, c]
            Lambda = Lambda_map[r, c]

            idx_map = make_index_map(
                cell_rows,
                cell_cols,
                Lambda,
                theta,
                thresholds_pct
            )

            if show_phase_profile:
                img = make_phase_map(idx_map, phase_pi_905)
            else:
                img = make_binary_CGH_from_index(idx_map, bit_tiles)

            top_label = f"Tile {tile_num[r, c]}: {int(theta)}°, Λ = {Lambda:.1f}"

        if show_phase_profile:
            # gray_r makes 0 phase white and max phase black
            last_im = ax.imshow(
                img,
                cmap="gray_r",
                vmin=0,
                vmax=phase_max_pi,
                interpolation="nearest"
            )
        else:
            last_im = ax.imshow(
                img,
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest"
            )

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

        # Border for the flat center tile
        if np.isnan(theta_map[r, c]):
            rect = Rectangle(
                (0, 0),
                final_size - 1,
                final_size - 1,
                linewidth=1.5,
                edgecolor=(0.25, 0.25, 0.25),
                facecolor="none"
            )
            ax.add_patch(rect)

        # Tile label
        ax.text(
            12,
            28,
            top_label,
            fontsize=9.5,
            fontweight="bold",
            color="black",
            family="Arial",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                pad=2
            )
        )


# ========================== COLORBAR ==========================
if show_phase_profile:
    cbar_ax = fig.add_axes([0.905, bottom + 0.10, 0.025, (top - bottom) - 0.20])

    cb = fig.colorbar(
        last_im,
        cax=cbar_ax
    )

    ticks = np.linspace(0, phase_max_pi, 6)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{x:.2f}$\\pi$" for x in ticks])

    cb.ax.tick_params(labelsize=11)
    cb.set_label(
        "Phase delay ($\\pi$ radians)",
        fontsize=12,
        family="Arial"
    )

    export_name = "Fig9a_phase_profile_3x3_512px.png"

else:
    export_name = "Fig9a_binary_CGH_3x3_512px.png"

# ========================== EXPORT FIGURE ==========================
pdf_name = export_name.replace(".png", ".pdf")

# Do not use bbox_inches="tight" here.
# It can crop the figure unevenly and make spacing look inconsistent.
fig.savefig(
    export_name,
    dpi=600,
    facecolor="white"
)

fig.savefig(
    pdf_name,
    dpi=600,
    facecolor="white"
)

print(f"Figure saved successfully: {export_name}")
print(f"PDF saved successfully: {pdf_name}")

plt.show()


# ========================== SAVE INDIVIDUAL CGHs ==========================
outdir = "CGH_512x512_patterns"
os.makedirs(outdir, exist_ok=True)

for r in range(3):
    for c in range(3):

        if np.isnan(theta_map[r, c]):
            cgh = np.ones((final_size, final_size))

        else:
            idx_map = make_index_map(
                cell_rows,
                cell_cols,
                Lambda_map[r, c],
                theta_map[r, c],
                thresholds_pct
            )

            cgh = make_binary_CGH_from_index(idx_map, bit_tiles)

        cgh_uint8 = np.uint8(cgh * 255)

        filename = os.path.join(
            outdir,
            f"CGH_tile_{tile_num[r, c]}_512x512.bmp"
        )

        Image.fromarray(cgh_uint8).save(filename)

print("Individual 512×512 BMP files saved for PLM upload.")