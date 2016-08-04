# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import csv

#plt_file1 = "output_T1e-1.txt"
#plt_file2 = "output_T5e-2.txt"
#plt_file3 = "output_T1e-2.txt"

#plt_file_arr_g = ["output_T1e-1.txt", "output_T-2.txt", "output_T1e-3.txt", "output_T1e-4.txt"]
plt_file_arr_g = ["for T/output_T0.txt", "for T/output_T1e-3.txt", "for T/output_T1e-2.txt", "for T/output_T5e-2.txt", "for T/output_T1e-1.txt"]
#plt_file_arr_g = ["output_T1e-3.txt"]
#plt_file_arr_T = ["output_g1e-4.txt"]
plt_file_arr_T = ["for g/output_g1e-4.txt", "for g/output_g1e-1.txt", "for g/output_g1e-2.txt", "for g/output_g1e-3.txt"]


g, T = [], []
Q2, P2, Ntot, Cv, U, Nc = [[[] for _ in range(6)] for _ in range(6)]

#which = "g"
which = "T"

if(which=="T"):
    iterable = plt_file_arr_T
elif(which=="g"):
    iterable = plt_file_arr_g

# for each T
for index, plt_file in enumerate(iterable):

    with open(plt_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)

        for row in reader:
            if (index == 0):
                g.append(float(row[0]))
                T.append(float(row[1]))
            Q2[index].append(float(row[2]))
            P2[index].append(float(row[3]))
            Ntot[index].append(float(row[4]))
            U[index].append(float(row[5]))
            Cv[index].append(float(row[6]))
            Nc[index].append(float(row[8]))

# Plot of Q2
plt.subplot(2, 2, 1)

if(which == "T"):
    for i in range(len(plt_file_arr_T)):
        #plt.plot(T, np.sqrt(Q2[i]), ".", linewidth=3)
        plt.plot(T, np.sqrt(Q2[i]), linewidth=3)
    plt.xlim(1e-3, 5e-1)
    plt.ylim(0, 0.5)
    plt.xlabel(r"$T$")
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], [0, 0.1, 0.2, 0.3, 0.4, 0.5])

elif(which == "g"):
    for i in range(len(plt_file_arr_g)):
        #plt.plot(g, np.sqrt(Q2[i]), ".", linewidth=3)
        plt.plot(g, np.sqrt(Q2[i]), linewidth=3)
    plt.xlim(1e-4, 1e-1)
    plt.ylim(0.05, 0.5)
    plt.xlabel(r"$g$")
    plt.yscale("log")
    plt.yticks([0.05, 0.1, 0.2, 0.5], [0.05, 0.1, 0.2, 0.5])

else:
    print("invalid key!!")
    raise KeyError

plt.xscale("log")
plt.ylabel(r"$\Delta Q$")


# Plot of P2
plt.subplot(2, 2, 3)

if(which == "T"):
    for i in range(len(plt_file_arr_T)):
        #plt.plot(T, np.sqrt(P2[i]), ".", linewidth=3)
        plt.plot(T, np.sqrt(P2[i]), linewidth=3)
    plt.xlim(1e-3, 5e-1)
    plt.ylim(0, 120)
    plt.xlabel(r"$T$")
    plt.yticks([0, 20, 40, 60, 80, 100, 120], [0, 20, 40, 60, 80, 100, 120])

elif(which == "g"):
    for i in range(len(plt_file_arr_g)):
        #plt.plot(g, np.sqrt(P2[i]), ".", linewidth=3)
        plt.plot(g, np.sqrt(P2[i]), linewidth=3)
    plt.xlim(1e-4, 1e-1)
    plt.ylim(5, 100)
    plt.xlabel(r"$g$")
    plt.yscale("log")
    plt.yticks([5, 10, 20, 50, 100], [5, 10, 20, 50, 100])

else:
    print("invalid key!!")
    raise KeyError


plt.ylabel(r"$\Delta P$")
plt.xscale("log")

# Plot of depletion
plt.subplot(2, 2, 2)
if(which == "g"):
    for i in range(len(plt_file_arr_g)):
        if(i == 0):
            continue
        plt.plot(g, 1 - np.array(Nc[i])/np.array(Ntot[i]), linewidth=3)
    plt.xlabel(r"$g$")
    plt.ylabel(r"${\rm depletion}\ \ \ 1 - N_0/N$")
    plt.xlim(1e-4, 1e-1)
    plt.ylim(1e-6, 1e-1)
    plt.xscale("log")
    plt.yscale("log")

if(which == "T"):
    pass


# Plot of depletion
plt.subplot(2, 2, 4)
if(which == "T"):
    for i in range(len(plt_file_arr_T)):
        plt.plot(T, Cv[i], linewidth=3)

    plt.xlabel(r"$T$")
    plt.ylabel(r"${\rm Specific heat}$")
    plt.xlim(1e-3, 5e-1)
    plt.ylim(1e-6, 1e1)
    plt.yscale("log")
    plt.xscale("log")
    plt.yticks([1e-6, 1e-4, 1e-2, 1e0], [r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$", r"$10^{0}$"])

plt.show()
