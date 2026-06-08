import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.ndimage import median_filter, label

# === CONFIGURABLE PARAMETERS ===
file_paths = [
    "distance_measurement/4/Y.txt",
    "distance_measurement/4/O.txt",
    "distance_measurement/4/U.txt",
    "distance_measurement/4/Z.txt",
    "distance_measurement/4/D.txt"
]

labels = ['Y', 'O', 'U', 'Z', 'D']

# Professional, muted, colorblind-friendly publication palette
colors = [
    '#3170a7',  # Soft Blue
    '#3a8138',  # Forest Green
    '#d68d37',  # Muted Orange
    '#743775',  # Deep Purple
    '#b23131'   # Deep Red
]

mppc_pixels       = 32
frame_size        = mppc_pixels ** 2
start_frame       = 10
end_frame         = 300
median_filter_size = 2
valid_range       = (80, 200)   # KEEPING YOUR ORIGINAL DATA PIPELINE RANGE UNTOUCHED
lens_offset       = 9
ns                = 1e-9
c_cm_per_s        = 2.99e10
round_trip        = 0.5
slope, offset     = -1.9, 861
c_adjust          = 0.15
iqr_bounds        = (45, 55)

# === HELPER FUNCTION (UNCHANGED) ===
def extract_iqr_data(file_path):
    try:
        raw = np.loadtxt(file_path, dtype=str)
    except FileNotFoundError:
        return np.array([])
    
    img = np.array([int(d[6:10], 16) for d in raw])
    tof_ns = img * slope + offset
    tof_array = (tof_ns * ns * c_cm_per_s * round_trip * c_adjust) - lens_offset

    stacked = np.zeros((mppc_pixels, mppc_pixels))
    valid_counts = np.zeros_like(stacked)

    for f in range(start_frame, end_frame + 1):
        s, e = (f - 1) * frame_size, f * frame_size
        if e > len(tof_array):
            break
        frame = tof_array[s:e].reshape(mppc_pixels, mppc_pixels)
        frame = median_filter(frame, size=median_filter_size)
        mask = (frame >= valid_range[0]) & (frame <= valid_range[1])
        frame_masked = np.where(mask, frame, np.nan)
        stacked += np.nan_to_num(frame_masked, nan=0.0)
        valid_counts += mask.astype(int)

    with np.errstate(invalid='ignore'):
        avg_frame = stacked / valid_counts
        avg_frame[valid_counts == 0] = np.nan

    mask = ~np.isnan(avg_frame) & (avg_frame < valid_range[1])
    labeled, num = label(mask)
    if num == 0:
        return np.array([])
    max_label = np.argmax(np.bincount(labeled.ravel()[labeled.ravel() > 0]))
    object_mask = labeled == max_label
    target_frame = np.where(object_mask, avg_frame, np.nan)

    data = target_frame[~np.isnan(target_frame)]
    if len(data) == 0:
        return np.array([])
    q_low, q_high = np.percentile(data, iqr_bounds)
    return data[(data >= q_low) & (data <= q_high)]

# === PROCESS FILES ===
iqr_data_list = [extract_iqr_data(fp) for fp in file_paths]

# === PLOT CONFIGURATION FOR APPLIED OPTICS ===
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
plt.rcParams['mathtext.fontset'] = 'dejavusans'

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300, layout =  'constrained') # Slightly taller to give text headroom

# Match your new data range viewport
plot_view_range = (110, 170)

# Generate fine spacing across the absolute pipeline limit to keep calculations exact
x = np.linspace(valid_range[0], valid_range[1], 2000)
bins = np.linspace(valid_range[0], valid_range[1], 600)  
bin_width = bins[1] - bins[0]

max_y_value = 0

for data, label_str, color in zip(iqr_data_list, labels, colors):
    if len(data) == 0:
        continue

    mu, sigma = np.mean(data), np.std(data)

    # Plot raw histogram (put bars below the curve using zorder)
    counts, _, _ = ax.hist(
        data, bins=bins, density=False,
        alpha=0.35, color=color, edgecolor=color, linewidth=0.5, zorder=1
    )

    # Compute Gaussian PDF and scale it to the histogram peak so the curve
    # visually sits inside the bars without changing mu/sigma
    pdf = norm.pdf(x, mu, sigma)
    pdf_peak = np.max(pdf) if np.max(pdf) > 0 else 1.0
    hist_peak = np.max(counts) if len(counts) > 0 else 1.0
    scale_factor = hist_peak / pdf_peak
    scaled_pdf = pdf * scale_factor

    ax.plot(x, scaled_pdf, lw=1.8, color=color, zorder=3)

    peak_y = max(np.max(scaled_pdf), hist_peak)
    max_y_value = max(max_y_value, peak_y)

    # --- CLEAN, CENTERED INLINE ANNOTATIONS ---
    # Properly formats the labels directly above each peak center
    # Use explicit math-mode segments so matplotlib's mathtext parses correctly
    annotation_text = "$\\mathbf{" + label_str + "}$\n" + f"$\\mu$ = {mu:.2f} cm\n" + f"$\\sigma$ = {sigma:.2f} cm"
    
    # Place text exactly at (mu), and slightly higher than the top of the curve peak (peak_y)
    # Using center alignment ensures it won't drift or collide sideways
    ax.text(
        mu, 
        peak_y + 0.3, 
        annotation_text, 
        color=color, 
        fontsize=9.5, 
        va='bottom', 
        ha='center',
        linespacing=1.2
    )

# === PLOT WINDOW CROPPING & HEADROOM ADJUSTMENT ===
ax.set_xlabel("Distance(cm)", fontsize=13, labelpad=6)
ax.set_ylabel("Counts", fontsize=13, labelpad=-4)
ax.tick_params(axis='both', which='major', labelsize=10.5)

ax.set_xlim(plot_view_range)  

# Crucial fix: Multiply by 1.35 to create a 35% empty margin at the top 
# so the floating text never gets cut off by the border
ax.set_ylim(0, max_y_value * 1.35)

# Clean, lightweight grid layout suitable for publication templates
ax.grid(True, linestyle='--', alpha=0.4, color='#b0b0b0', linewidth=0.6)

plt.tight_layout()
plt.savefig('distance_distributions_journal.pdf', bbox_inches='tight', dpi=600)
plt.show()