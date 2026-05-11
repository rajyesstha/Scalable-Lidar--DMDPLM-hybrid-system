#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 09:28:17 2025

@author: rajeshshrestha
"""

import matplotlib.pyplot as plt
import numpy as np

# Data
orders = np.arange(-2, 5)  # Diffraction orders from -2 to 4
angles = [0.229183152, 0.152788768, 0.229183152, 0.152788768, 0.152788768, 0.152788768, 0.152788768]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(orders, angles, 'o-', color='blue', markerfacecolor='red', markersize=8, linewidth=2, label='Angular Resolution')

# Add title and labels
plt.title('Angular Resolution (Rx) vs Diffraction Orders', fontsize=25, pad=15)
plt.xlabel('Diffraction Orders', fontsize=20)
plt.ylabel('Angular Resolution (Deg)', fontsize=20)

# Set y-axis limits and ticks
plt.ylim(0.1, 0.3)
plt.yticks(np.arange(0, 1.1, 0.2))

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend(fontsize=12)

# Annotate each point with the angular resolution value
for i, (order, angle) in enumerate(zip(orders, angles)):
    plt.text(order, angle - 0.015, f'{angle:.3f}', ha='center', va='top', fontsize=9)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show plot
plt.show()