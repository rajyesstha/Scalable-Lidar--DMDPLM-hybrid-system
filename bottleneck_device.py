import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 7 × 9 DMD–PLM LiDAR TIMING & PERFORMANCE GRAPH
# ============================================================

# -----------------------------
# Timing parameters
# -----------------------------
DMD_PERIOD_US = 30.72          # DLP7000 pattern period
PLM_PERIOD_US = 694.44         # DLP6750 @ 1440 holograms/s
LC_RISE_MS = 31.3              # LCC1221-B published rise time

N_PERIPHERAL = 56
N_CENTRAL = 7
N_TOTAL = N_PERIPHERAL + N_CENTRAL

# Timeline calculations (Proposed 1-Switch Strategy)
t_peripheral_end = N_PERIPHERAL * (PLM_PERIOD_US / 1000.0)      # ~38.89 ms
t_lc_switch_end = t_peripheral_end + LC_RISE_MS                 # ~70.19 ms
t_total_end = t_lc_switch_end + N_CENTRAL * (PLM_PERIOD_US / 1000.0) # ~75.05 ms

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

fig, (ax1, ax2) = plt.subplots(
    2, 1, 
    figsize=(12, 8), 
    gridspec_kw={'height_ratios': [2, 1]}
)

# ============================================================
# TOP GRAPH: Continuous Waveform Logic Analyzer Traces
# ============================================================
t = np.linspace(0, t_total_end, 2000)

# LC State transition curve (exponential response during switch)
lc_state = np.zeros_like(t)
mask_peripheral = t < t_peripheral_end
mask_switch = (t >= t_peripheral_end) & (t < t_lc_switch_end)
mask_central = t >= t_lc_switch_end

lc_state[mask_peripheral] = 0.0
lc_state[mask_switch] = 1.0 - np.exp(-(t[mask_switch] - t_peripheral_end) / (LC_RISE_MS / 3.0))
lc_state[mask_central] = 1.0

# PLM Trigger Pulse Train
plm_pulses = np.zeros_like(t)
tile_times = list(np.arange(0, t_peripheral_end, PLM_PERIOD_US / 1000.0)) + \
             list(np.arange(t_lc_switch_end, t_total_end, PLM_PERIOD_US / 1000.0))

for tt in tile_times:
    idx = np.argmin(np.abs(t - tt))
    plm_pulses[idx:idx+5] = 1.0

# Plot signal waveforms with vertical offset for digital channel view
ax1.plot(t, lc_state + 3.0, 'b-', label='LC Retarder State (0: Periph, 1: Central)', linewidth=2)
ax1.plot(t, plm_pulses + 1.5, 'g-', label='PLM Trigger Pulses (694 µs period)', linewidth=1.2)

# Highlight operating phase regions
ax1.axvspan(0, t_peripheral_end, color='green', alpha=0.1, label='Peripheral Acquisition (56 tiles)')
ax1.axvspan(t_peripheral_end, t_lc_switch_end, color='red', alpha=0.15, label=f'LC Transition Delay ({LC_RISE_MS} ms)')
ax1.axvspan(t_lc_switch_end, t_total_end, color='blue', alpha=0.1, label='Central Acquisition (7 tiles)')

ax1.set_yticks([1.5, 2.0, 3.0, 4.0])
ax1.set_yticklabels(['PLM Off', 'PLM Trigger', 'LC State 1', 'LC State 2'])
ax1.set_xlabel('Time (ms)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Digital Traces & States', fontsize=11, fontweight='bold')
ax1.set_title('Continuous Waveform & Timing Signals (Proposed Linear Strategy)', fontsize=13, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper right', frameon=True)

# ============================================================
# BOTTOM GRAPH: Frame Rate & Scan Time vs LC Switches
# ============================================================
switches = np.arange(1, 8)
lc_times = switches * LC_RISE_MS
plm_time_total = N_TOTAL * (PLM_PERIOD_US / 1000.0)
total_scan_times = plm_time_total + lc_times
frame_rates = 1000.0 / total_scan_times

color1 = 'tab:blue'
ax2.set_xlabel('Number of LC Switching Events per Full Frame', fontsize=11, fontweight='bold')
ax2.set_ylabel('Total Scan Time (ms)', color=color1, fontsize=11, fontweight='bold')
line1 = ax2.plot(switches, total_scan_times, color=color1, marker='o', linewidth=2.5, label='Total Scan Time (ms)')
ax2.tick_params(axis='y', labelcolor=color1)

ax2_twin = ax2.twinx()
color2 = 'tab:red'
ax2_twin.set_ylabel('Ideal Scan Rate (Hz)', color=color2, fontsize=11, fontweight='bold')
line2 = ax2_twin.plot(switches, frame_rates, color=color2, marker='s', linestyle='--', linewidth=2.5, label='Frame Rate (Hz)')
ax2_twin.tick_params(axis='y', labelcolor=color2)

# Annotate strategies
ax2.annotate('Proposed Strategy\n(1 LC Switch, 13.3 Hz)', xy=(1, total_scan_times[0]), xytext=(1.5, 120),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6), fontweight='bold')
ax2.annotate('Raster Strategy\n(7 LC Switches, 3.8 Hz)', xy=(7, total_scan_times[-1]), xytext=(5.2, 200),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6), fontweight='bold')

ax2.set_title('Impact of LC Switching Events on LiDAR Frame Rate', fontsize=13, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

# Save
plt.savefig('DMD_PLM_LiDAR_Timing_Graph.png', dpi=300, bbox_inches='tight')
plt.savefig('DMD_PLM_LiDAR_Timing_Graph.pdf', bbox_inches='tight')

plt.show()