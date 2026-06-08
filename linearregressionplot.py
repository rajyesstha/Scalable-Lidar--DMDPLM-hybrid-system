import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- Data ---
ground_truth = np.array([
    117, 127, 137, 147, 110, 125, 137, 158, 168,
    102, 125, 133, 152, 162, 103, 124, 134, 147, 170,
    92, 108, 111, 145, 163, 105, 120, 132, 155, 167,
    115, 125, 140, 155, 160
])

computed = np.array([
    120.03, 130.13, 141.89, 151.77, 111.68, 126.45, 138.79, 159.47, 169.75,
    103.37, 126.87, 134.22, 154.47, 164.48, 103.16, 124.01, 134.58, 147.09, 170.34,
    93.49, 109.92, 112.18, 146.42, 164.75, 106.38, 122.99, 133.46, 158.83, 170.45,
    116.79, 127.45, 141.79, 157.79, 161.74
])

errors = np.abs(ground_truth - computed)

orders = [
    '-2Y', '-2O', '-2U', '-2Z',
    '-1Y', '-1O', '-1U', '-1Z', '-1D',
    '0Y',  '0O',  '0U',  '0Z',  '0D',
    '1Y',  '1O',  '1U',  '1Z',  '1D',
    '2Y',  '2O',  '2U',  '2Z',  '2D',
    '3Y',  '3O',  '3U',  '3Z',  '3D',
    '4Y',  '4O',  '4U',  '4Z',  '4D'
]

unique_orders = sorted(list(set(o[:-1] for o in orders)), key=lambda x: int(x))

rms_by_order = {}
for order in unique_orders:
    indices = [i for i, o in enumerate(orders) if o.startswith(order)]
    gt = ground_truth[indices]
    meas = computed[indices]
    residuals = meas - gt
    rms = np.sqrt(np.mean(residuals ** 2))
    rms_by_order[order] = rms

# ========================== GLOBAL FONT CONFIGURATION ==========================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# ========================== FIGURE SETUP ==========================
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300, layout='constrained')

# YOUR ORIGINAL COLOR DICTIONARY RESTORED EXACTLY
order_colors = {
    "-2": "#8080FF",
    "-1": "#00FFFF",
    "0":  "#FF00FF",
    "1":  "#C8C800",
    "2":  "#FF0000",
    "3":  "#00FF00",
    "4":  "#0000FF"
}

# --- Ideal 1:1 Reference Line ---
min_val = min(np.min(ground_truth), np.min(computed)) - 5
max_val = max(np.max(ground_truth), np.max(computed)) + 5

ax.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle='--',
    linewidth=1.0,
    color='#777777',
    label="Ideal 1:1 Line",
    zorder=1
)

# --- Plot Fits and Data ---
for order in unique_orders:
    color = order_colors[order]
    indices = [i for i, o in enumerate(orders) if o.startswith(order)]

    x = ground_truth[indices]
    y = computed[indices]
    yerr = errors[indices]

    # Plot linear regression line
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = model.predict(x_fit.reshape(-1, 1))

    ax.plot(
        x_fit,
        y_fit,
        linestyle='-',
        linewidth=1.2,
        color=color,
        zorder=2
    )

    # UNIFIED SHAPE STYLING:
    # Uses a single circle ('o') but colors the face with your original hex color.
    # A sleek dark-charcoal outline ('#222222') is added to make the markers pop.
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt='o',
        capsize=3,
        color=color,
        ecolor=color,
        elinewidth=1.0,
        markersize=6.0,
        markerfacecolor=color,       # Solid colored fill keeps things cohesive
        markeredgecolor='#222222',   # Crisp dark border replaces the thick colored edge
        markeredgewidth=0.8,
        label=f"Order {order}",
        zorder=3
    )

# ========================== AXIS FORMATTING ==========================
ax.set_xlabel("Ground Truth Distance (cm)", fontsize=13, labelpad=8)
ax.set_ylabel("Measured Distance (cm)", fontsize=13, labelpad=12)

ax.tick_params(axis="both", which="major", labelsize=10.5, width=0.8)
for spine in ax.spines.values():
    spine.set_linewidth(0.8)

ax.set_xlim(85, 180)
ax.set_ylim(85, 180)

ax.grid(True, linestyle="--", alpha=0.4, color='#b0b0b0', linewidth=0.6)

# ========================== LEGEND & RMS TEXT BOX ==========================
leg = ax.legend(
    title="Diffraction Order",
    fontsize=9,
    title_fontsize=10.5,
    loc="upper left",
    frameon=True,
    framealpha=0.9,
    edgecolor='#b0b0b0'
)
leg.get_frame().set_linewidth(0.6)

# Borderless structured box block anchored beautifully on the lower right workspace margin
rms_text = "$\\mathbf{RMS Deviations:}$\n" + "\n".join([
    f"Order {order}: {rms_by_order[order]:.2f} cm"
    for order in unique_orders
])

ax.text(
    0.97,
    0.03,
    rms_text,
    transform=ax.transAxes,
    fontsize=9,
    fontname="Arial",
    color='#333333',
    verticalalignment="bottom",
    horizontalalignment="right",
    linespacing=1.3,
    bbox=dict(
        facecolor="white",
        edgecolor="#b0b0b0",
        linewidth=0.6,
        alpha=0.9,
        boxstyle='round,pad=0.5'
    )
)

# ========================== EXPORT ==========================
fig.set_constrained_layout_pads(w_pad=4/72, h_pad=4/72, hspace=0, wspace=0)

plt.savefig("Distance linear regression.png", dpi=600)
plt.savefig("Distance linear regression.pdf", dpi=600)

plt.show()