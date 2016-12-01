# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import csv
#plt_file1 = "output_T1e-1.txt"
#plt_file2 = "output_T5e-2.txt"
#plt_file3 = "output_T1e-2.txt"

plt_file_arr_g = ["output_T0.txt", "output_T1e-3.txt", "output_T1e-2.txt", "output_T5e-2.txt", "output_T1e-1.txt"]
#plt_file_arr_g = ["for T/output_T0.txt", "for T/output_T1e-3.txt", "for T/output_T1e-2.txt", "for T/output_T5e-2.txt", "for T/output_T1e-1.txt"]
#plt_file_arr_g = ["output_T1e-3.txt"]
#plt_file_arr_T = ["output_g1e-4.txt", "output_g1e-3.txt"]
plt_file_arr_T = ["for g/output_g1e-1.txt", "for g/output_g1e-2.txt", "for g/output_g1e-3.txt", "for g/output_g1e-4.txt"]


g, T, Q2, P2, Ntot, Cv, U, Nc, J, MJ, LY = [[[] for _ in range(6)] for _ in range(11)]

which = "g"
#which = "T"

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
            """
            if (index == 0):
                g.append(float(row[0]))
                T.append(float(row[1]))
            """
            g[index].append(float(row[0]))
            T[index].append(float(row[1]))
            Q2[index].append(float(row[2]))
            P2[index].append(float(row[3]))
            Ntot[index].append(float(row[4]))
            U[index].append(float(row[5]))
            Cv[index].append(float(row[6]))
            Nc[index].append(float(row[8]))
            #J[index].append(float(row[9]))
            #MJ[index].append(float(row[10]))
            #LY[index].append(float(row[11]))


# Plot of Q2
plt.subplot(1, 3, 1)

if(which == "T"):
    for i, interaction in zip(range(len(plt_file_arr_T)), [0.1, 0.01, 0.001, 0.0001]):
        #plt.plot(T, np.sqrt(Q2[i]), ".", linewidth=3)
        plt.plot(T[i], np.sqrt(Q2[i]), linewidth=3, label="$a_s/a_o$ = {0}".format(interaction), color=plt.cm.jet(i/4))
    plt.xlim(1e-3, 5e-1)
    plt.ylim(0, 0.5)
    plt.xlabel(r"$T$", fontsize=18)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], [0, 0.1, 0.2, 0.3, 0.4, 0.5])

elif(which == "g"):
    for i, temp in zip(range(len(plt_file_arr_g)), [0, 0.001, 0.01, 0.05, 0.1]):
        #plt.plot(g, np.sqrt(Q2[i]), ".", linewidth=3)
        plt.plot(g[i], np.sqrt(Q2[i]), linewidth=3, label="$T/T_c$ = {0}".format(temp), color=plt.cm.jet(i/5))
    plt.xlim(1e-4, 1e-1)
    plt.ylim(0.05, 0.5)
    plt.xlabel(r"$g$", fontsize=18)
    plt.yscale("log")
    plt.yticks([0.05, 0.1, 0.2, 0.5], [0.05, 0.1, 0.2, 0.5])

else:
    print("invalid key!!")
    raise KeyError

plt.xscale("log")
plt.ylabel(r"$\Delta Q$", fontsize=18)
plt.legend(loc="upper left")

# Plot of P2
plt.subplot(1, 3, 2)

if(which == "T"):
    for i, interaction in zip(range(len(plt_file_arr_T)), [0.1, 0.01, 0.001, 0.0001]):
        #plt.plot(T, np.sqrt(P2[i]), ".", linewidth=3)
        plt.plot(T[i], np.sqrt(P2[i]), linewidth=3, label="$a_s/a_o$ = {0}".format(interaction), color=plt.cm.jet(i/4))
    plt.xlim(1e-3, 5e-1)
    plt.ylim(0, 120)
    plt.xlabel(r"$T$", fontsize=18)
    plt.yticks([0, 20, 40, 60, 80, 100, 120], [0, 20, 40, 60, 80, 100, 120])

elif(which == "g"):
    for i, temp in zip(range(len(plt_file_arr_g)), [0, 0.001, 0.01, 0.05, 0.1]):
        #plt.plot(g, np.sqrt(P2[i]), ".", linewidth=3)
        plt.plot(g[i], np.sqrt(P2[i]), linewidth=3, label="$T/T_c$ = {0}".format(temp), color=plt.cm.jet(i/5))
    plt.xlim(1e-4, 1e-1)
    plt.ylim(5, 100)
    plt.xlabel(r"$g$", fontsize=18)
    plt.yscale("log")
    plt.yticks([5, 10, 20, 50, 100], [5, 10, 20, 50, 100])

else:
    print("invalid key!!")
    raise KeyError


plt.ylabel(r"$\Delta P$", fontsize=18)
plt.xscale("log")
plt.legend(loc="upper left")

"""
# Plot of depletion
plt.subplot(2, 2, 2)
if(which == "g"):
    for i in range(len(plt_file_arr_g)):
        if(i == 0):
            continue
        plt.plot(g[i], 1 - np.array(Nc[i])/np.array(Ntot[i]), linewidth=3)
    plt.xlabel(r"$g$")
    plt.ylabel(r"${\rm depletion}\ \ \ 1 - N_0/N$")
    plt.xlim(1e-4, 1e-1)
    plt.ylim(1e-6, 1e-1)
    plt.xscale("log")
    plt.yscale("log")

if(which == "T"):
    pass

"""
# Plot of depletion
plt.subplot(1, 3, 3)
if(which == "T"):
    J1, J2 = [], []
    for i, interaction in zip(range(len(plt_file_arr_T)), [0.1, 0.01, 0.001, 0.0001]):
        #plt.plot(T, U[i], linewidth=3)
        plt.plot(np.array(T[i][:len(T[i])-1]), (np.array(U[i][1:]) - np.array(U[i][:len(U[i])-1]))/(np.array(T[i][1:]) - np.array(T[i][:len(T[i])-1]))/np.array(Ntot[i][:len(Ntot[i])-1]), linewidth=3, label="$a_s/a_o$ = {0}".format(interaction), color=plt.cm.jet(i/4))
        #
        # 2階微分. 読みづらい.
        #J1.append((np.array(J[i][1:]) - np.array(J[i][:len(J[i])-1]))/(np.array(T[i][1:]) - np.array(T[i][:len(T[i])-1])))
        #plt.plot(T[i][:len(T[i])-2] , -np.array(T[i][:len(T[i])-2])*((np.array(J1[i][1:]) - np.array(J1[i][:len(J1[i])-1]))/(np.array(T[i][2:]) - np.array(T[i][:len(T[i])-2]))/np.array(Ntot[i][:len(Ntot[i])-2])), "--", linewidth=3)
        #J2.append((np.array(MJ[i][1:]) - np.array(MJ[i][:len(MJ[i])-1]))/(np.array(T[i][1:]) - np.array(T[i][:len(T[i])-1])))
        #plt.plot(T[i][:len(T[i])-2] , -np.array(T[i][:len(T[i])-2])*((np.array(J2[i][1:]) - np.array(J2[i][:len(J2[i])-1]))/(np.array(T[i][2:]) - np.array(T[i][:len(T[i])-2]))/np.array(Ntot[i][:len(Ntot[i])-2])), linewidth=3)
        #np.array(Ntot[i][:len(Ntot[i])-2])

    plt.xlabel(r"$T$")
    plt.ylabel(r"${\rm Specific\ heat}$")
    plt.xlim(1e-3, 5e-1)
    plt.ylim(1e-6, 1e1)
    plt.yscale("log")
    plt.xscale("log")
    plt.yticks([1e-6, 1e-4, 1e-2, 1e0], [r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$", r"$10^{0}$"])
else:
    for i, temp in zip(range(len(plt_file_arr_g)), [0, 0.001, 0.01, 0.05, 0.1]):
        #plt.plot(g, np.sqrt(P2[i]), ".", linewidth=3)
        plt.plot(g[i], U[i], linewidth=3, label="$T/T_c$ = {0}".format(temp), color=plt.cm.jet(i/5))
    plt.xlim(1e-4, 1e-1)
    plt.ylim(1e-4, 1)
    plt.xlabel(r"$g$", fontsize=18)
    plt.xscale("log")
    plt.yscale("log")
    #plt.yticks([5, 10, 20, 50, 100], [5, 10, 20, 50, 100])

plt.legend(loc = 'left')
#plt.legend()
plt.show()
