"""
Copyright (C) Andrew 2022
"""

import numpy as np
import matplotlib.pyplot as plt
from comparison_data import data, num_dataset
import random

# CONFIGURATION:
degrees = 65
bottom_adjustment = 0.2
fsize = 10
transparency_ratio = 0.6  # control transparency of circles
hfont = {'fontname': 'Arial'}


def build_color_():
    # Incase that: automatically generates the list of colors
    colors_lamd = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF), range(n)))

    # This is fixed 6 colors
    list_color_rgb = [(255 / 255., 0, 0, transparency_ratio),
                      (0, 255 / 255., 0, transparency_ratio),
                      (0, 0, 255 / 255., transparency_ratio),
                      (255 / 255., 255 / 255., 0, transparency_ratio),
                      (0, 255 / 255., 255 / 255., transparency_ratio),
                      (255 / 255., 0, 255 / 255., transparency_ratio),
                      (0, 128 / 255., 128 / 255., transparency_ratio)
                      ]
    colors_lamd = lambda n: list(map(lambda i: list_color_rgb[i], range(n)))
    return colors_lamd


colors_ = build_color_()

# 'data' structure like: [Name, Model-size, Time-step, MSE]
# build the label to show
labels = []
for i in data[:, 0]:
    if i not in labels:
        labels.append(i)

# Prepare list of color for each label
colors = colors_(len(labels))

fig, ax = plt.subplots()

# Start to plot the circles
for i, label in enumerate(labels):
    for record in data:
        if record[0] == label:
            area = np.pi * (20 * record[1])  # size of point
            # order of scatter: x-axis (here is time step), y-axis (here is MSE), size-of-point, color
            ax.scatter(float(record[2]), float(record[3]), s=area / 10000, c=colors[i], label=label)

# plot the central point of circles
ax.scatter(data[:, 2].astype(float), data[:, 3].astype(float), s=1, c=[[0, 0, 0]])

ax.grid()
plt.xlabel('Forecast horizon', **hfont)
plt.ylabel(f'Dataset {num_dataset + 1} (MSE)', **hfont)

# Legend
from matplotlib.lines import Line2D

legend_elements = []
for color, label in zip(colors, labels):
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=label,
                                  markerfacecolor=color, markersize=10))
plt.legend(handles=legend_elements)

plt.savefig('./figures/accuracy_ops_modelsize.png', format='png', dpi=220)

plt.show()
