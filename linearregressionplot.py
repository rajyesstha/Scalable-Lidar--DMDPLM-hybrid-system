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

# --- Group diffraction orders ---
unique_orders = sorted(
    list(set(o[:-1] for o in orders)),
    key=lambda x: int(x)
)

# --- RMS deviation for each diffraction order ---
print("RMS deviation for each diffraction order:")

rms_by_order = {}

for order in unique_orders:
    indices = [i for i, o in enumerate(orders) if o.startswith(order)]

    gt = ground_truth[indices]
    meas = computed[indices]

    residuals = meas - gt
    rms = np.sqrt(np.mean(residuals ** 2))

    rms_by_order[order] = rms

    print(f"Order {order}: RMS deviation = {rms:.3f} cm")


# --- Plotting ---
order_colors = {
    "-2": "#8080FF",   # light purple/blue
    "-1": "#00FFFF",   # cyan
    "0":  "#FF00FF",   # magenta
    "1":  "#C8C800",   # yellow/olive
    "2":  "#FF0000",   # red
    "3":  "#00FF00",   # green
    "4":  "#0000FF"    # blue
}

plt.figure(figsize=(11, 6), dpi=300)

for order in unique_orders:
    color = order_colors[order]

    indices = [i for i, o in enumerate(orders) if o.startswith(order)]

    x = ground_truth[indices]
    y = computed[indices]
    yerr = errors[indices]

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        fmt='o',
        capsize=4,
        color=color,
        ecolor=color,
        markerfacecolor=color,
        markeredgecolor='black',
        markeredgewidth=0.6,
        label=f"Order {order}"
    )

    model = LinearRegression().fit(x.reshape(-1, 1), y)

    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = model.predict(x_fit.reshape(-1, 1))

    plt.plot(
        x_fit,
        y_fit,
        linestyle='-',
        linewidth=2,
        color=color
    )

# --- Optional 1:1 reference line ---
#min_val = min(np.min(ground_truth), np.min(computed))
#max_val = max(np.max(ground_truth), np.max(computed))
#
#plt.plot(
#    [min_val, max_val],
#    [min_val, max_val],
#    'k--',
#    linewidth=1.5,
#    label="Ideal 1:1 Line"
#)

# ========================== RMS TEXT BOX ==========================
rms_text = "\n".join([
    f"Order {order}: RMS = {rms_by_order[order]:.2f} cm"
    for order in unique_orders
])

plt.text(
    0.98,
    0.02,
    rms_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="bottom",
    horizontalalignment="right",
    bbox=dict(
        facecolor="white",
        edgecolor="black",
        alpha=0.85
    )
)

# --- Formatting ---
#lt.title(
#   "Linear Regression Analysis for Distance at 7 Diffraction Orders",
#   fontsize=25
#


plt.xlabel(
    "Ground Truth Distance (cm)",
    fontsize=20
)

plt.ylabel(
    "Measured Distance (cm)",
    fontsize=20
)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.grid(True, alpha=0.3)

plt.legend(
    title="Diffraction Order",
    fontsize=10,
    title_fontsize=12,
    loc="upper left",
    frameon=True
)

plt.tight_layout()
plt.show()