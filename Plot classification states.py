# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 19:32:47 2022

@author: JK-WORK
"""

import matplotlib.pyplot as plt
import numpy as np


true_dist = [10e5, 10e4, 10e3, 10e2, 10e1]
PWG_class = [4e5, 7e4, 10e3, 10e2, 10e1]
BIG_class = [5e5, 8e4, 10e3, 10e2, 10e1]
G_class = [5e4, 3e4, 9e3, 10e2, 10e1]


x = np.arange(len(PWG_class))  # the label locations

fig, ax = plt.subplots()
rects0 = ax.bar(x - 0.3, true_dist, 0.2, label='Ideal classification')
rects1 = ax.bar(x - 0.1, BIG_class, 0.2, label='Bigaussian')
rects2 = ax.bar(x + 0.1, PWG_class, 0.2, label='PWG')
rects3 = ax.bar(x + 0.3, G_class, 0.2, label='Gaussian')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Classsification score')
ax.set_xlabel('Event duration')
ax.legend()

#ax.bar_label(rects1, padding=4)
#ax.bar_label(rects2, padding=4)
#ax.bar_label(rects3, padding=4)
plt.gca().set_yscale("log")
fig.tight_layout()

plt.show()