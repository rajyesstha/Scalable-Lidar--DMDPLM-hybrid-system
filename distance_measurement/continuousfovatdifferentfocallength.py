import numpy as np
import matplotlib.pyplot as plt

# ========================== DATA & CALCULATIONS ==========================
orders = np.arange(-2, 5)
theta_i = 45.0             # Incident angle in degrees
lambda_um = 0.905          # Wavelength in micrometers
pitch = 13.68 / np.sqrt(2) # Pitch in micrometers
sensor_size = 3.2          # Sensor size in mm
focal_length = 25          # Single focal length in mm
text_offset = 0.25         # Shift text slightly to the right of the dashed bounds

# Grating equation calculations
theta_d = np.degrees(np.arcsin(np.sin(np.radians(theta_i)) - orders * lambda_um / pitch))

# Calculate FOV at 0th order (angular)
fov_0_angle = 2 * np.degrees(np.arctan(sensor_size / (2 * focal_length)))

# Calculate chief ray angles
chief_1 = theta_i - fov_0_angle / 2
chief_2 = theta_i + fov_0_angle / 2

# Calculate sub-FOVs (in degrees)
sub_fov = np.degrees(np.arcsin(np.sin(np.radians(chief_2)) - orders * lambda_um / pitch)) - \
          np.degrees(np.arcsin(np.sin(np.radians(chief_1)) - orders * lambda_um / pitch))

# Calculate error bars (half-FOV)
error_bars = sub_fov / 2

# ========================== GLOBAL FONT CONFIGURATION ==========================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# ========================== FIGURE SETUP ==========================
fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300, layout='constrained')

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

# --- 1. DISCRETE HORIZONTAL BOUNDS, SHADING, & DATA ---
for x, y, err in zip(orders, theta_d, error_bars):
    y_top = y + err
    y_bottom = y - err
    color = order_colors[str(x)]  # Fetch the specific color for this order
    
    # Soft background shading matched to each individual order's color scheme
    ax.axvspan(x - 0.2, x + 0.2, ymin=(y_bottom - 10)/(75 - 10), ymax=(y_top - 10)/(75 - 10), 
               color=color, alpha=0.06, zorder=1)
    
    # Clean horizontal dashed lines highlighting the sub-FOV step limits
    # Clean horizontal dashed lines highlighting the sub-FOV step limits across the entire graph width
    ax.axhline(y_top, color='#555555', linestyle='--', linewidth=0.6, alpha=0.5, zorder=2)
    ax.axhline(y_bottom, color='#555555', linestyle='--', linewidth=0.6, alpha=0.5, zorder=2)
    # Calculate the full chief-ray angular extent for this channel
    total_fov_span = err * 2
    
    # Main data markers mapped to your specific color dictionary
    ax.errorbar(
        x, 
        y, 
        yerr=err, 
        fmt='o', 
        capsize=0,                 
        color=color,
        ecolor=color, 
        elinewidth=1.5,
        markersize=6.0,
        markerfacecolor='white',   # Sleek open-circle visual format
        markeredgecolor=color,
        markeredgewidth=1.5,
        label=f'Order {x:2d}: Full sub-FOV = {total_fov_span:.2f}°',
        zorder=3
    )

# --- 2. LABELS AND ANNOTATIONS ---
for x, y, err in zip(orders, theta_d, error_bars):
    annotation_text = f"{y:.2f}° $\\pm$ {err:.2f}°"
    ax.text(
        x + text_offset, 
        y, 
        annotation_text, 
        ha='left', 
        va='center', 
        fontsize=9.5, 
        color='#c1272d',       # Deep crimson accent color for clarity
        weight='bold',
        zorder=4
    )

# ========================== AXIS FORMATTING ==========================
ax.set_xlabel('Diffraction Orders', fontsize=12, labelpad=4)
ax.set_ylabel('RxDMD sub-FOV steering angles (deg)', fontsize=12, labelpad=4)

ax.tick_params(axis='both', which='major', labelsize=10.5, width=0.8)
for spine in ax.spines.values():
    spine.set_linewidth(0.8)

ax.set_xlim(-2.5, 5.5) 
ax.set_xticks(orders)
ax.set_ylim(10, 75)

ax.grid(True, linestyle='--', alpha=0.25, color='#b0b0b0', linewidth=0.5)

# ========================== LEGEND STRUCTURE ==========================
leg = ax.legend(
    title=rf'$\mathbf{{\theta_i = {theta_i}^\circ\ (f = {focal_length}\text{{ mm}})}}$',
    loc='upper right', 
    fontsize=9.0, 
    title_fontsize=10.0,
    frameon=True, 
    framealpha=0.95, 
    edgecolor='#b0b0b0'
)
leg.get_frame().set_linewidth(0.6)

# ========================== EXPORT ==========================
fig.set_constrained_layout_pads(w_pad=4/72, h_pad=4/72, hspace=0, wspace=0)

plt.savefig('RxDMD_Discrete_Steering_Analysis.png', dpi=600)
plt.savefig('RxDMD_Discrete_Steering_Analysis.pdf', dpi=600)

plt.show()