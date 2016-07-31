# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import csv

plt_file = "output.txt"

g, Q2, P2 = [], [], []

with open(plt_file, 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)

    for row in reader:
        g.append(float(row[0]))
        Q2.append(float(row[1]))
        P2.append(float(row[2]))


# Plot of Q2
plt.subplot(2, 1, 1)
plt.plot(g, np.sqrt(Q2), linewidth=3)
plt.xlabel(r"$r$")
plt.ylabel(r"$\Delta Q$")
plt.xlim(1e-4, 1e-1)
plt.ylim(0.05, 0.5)
plt.xscale("log")
plt.yscale("log")
plt.yticks([0.05, 0.1, 0.2, 0.5], [0.05, 0.1, 0.2, 0.5])

# Plot of P2
plt.subplot(2, 1, 2)
plt.plot(g, np.sqrt(P2), linewidth=3)
plt.xlabel(r"$r$")
plt.ylabel(r"$\Delta P$")
plt.xlim(1e-4, 1e-1)
plt.ylim(5, 100)
plt.xscale("log")
plt.yscale("log")
plt.yticks([5, 10, 20, 50, 100], [5, 10, 20, 50, 100])


plt.show()

