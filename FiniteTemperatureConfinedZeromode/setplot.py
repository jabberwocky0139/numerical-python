# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import csv

#plt_file1 = "output_T1e-1.txt"
#plt_file2 = "output_T5e-2.txt"
#plt_file3 = "output_T1e-2.txt"

plt_file_arr = ["output_T1e-1.txt", "output_T5e-2.txt", "output_T1e-2.txt", "output_T1e-3.txt"]

g  = []
Q2, P2, Ntot = [[[] for _ in range(5)] for _ in range(3)]

for index, plt_file in enumerate(plt_file_arr):
    
    with open(plt_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        
        for row in reader:
            if(index==0):
                g.append(float(row[0]))
            Q2[index].append(float(row[1]))
            P2[index].append(float(row[2]))
            Ntot[index].append(float(row[3]))


# Plot of Q2
plt.subplot(2, 2, 1)
plt.plot(g, np.sqrt(Q2[0]), linewidth=3)
plt.plot(g, np.sqrt(Q2[1]), linewidth=3)
plt.plot(g, np.sqrt(Q2[2]), linewidth=3)
plt.plot(g, np.sqrt(Q2[3]), linewidth=3)
plt.xlabel(r"$g$")
plt.ylabel(r"$\Delta Q$")
plt.xlim(1e-4, 1e-1)
plt.ylim(0.05, 0.5)
plt.xscale("log")
plt.yscale("log")
plt.yticks([0.05, 0.1, 0.2, 0.5], [0.05, 0.1, 0.2, 0.5])

# Plot of P2
plt.subplot(2, 2, 3)
plt.plot(g, np.sqrt(P2[0]), linewidth=3)
plt.plot(g, np.sqrt(P2[1]), linewidth=3)
plt.plot(g, np.sqrt(P2[2]), linewidth=3)
plt.plot(g, np.sqrt(P2[3]), linewidth=3)
plt.xlabel(r"$g$")
plt.ylabel(r"$\Delta P$")
plt.xlim(1e-4, 1e-1)
plt.ylim(5, 100)
plt.xscale("log")
plt.yscale("log")
plt.yticks([5, 10, 20, 50, 100], [5, 10, 20, 50, 100])

# Plot of depletion
plt.subplot(2, 2, 2)
plt.plot(g, 1 - 1e3/np.array(Ntot[0]), linewidth=3)
plt.plot(g, 1 - 1e3/np.array(Ntot[1]), linewidth=3)
plt.plot(g, 1 - 1e3/np.array(Ntot[2]), linewidth=3)
plt.plot(g, 1 - 1e3/np.array(Ntot[3]), linewidth=3)
plt.xlabel(r"$g$")
plt.ylabel(r"${\rm depletion}\ \ \ 1 - N_0/N$")
plt.xlim(1e-4, 1e-1)
plt.ylim(1e-6, 1e-1)
plt.xscale("log")
plt.yscale("log")
#plt.yticks([5, 10, 20, 50, 100], [5, 10, 20, 50, 100])


plt.show()

