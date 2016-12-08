import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import csv

full = []
qpq = []
q4 = []
Number = []

with open('full_energy_N.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for index, value in reader:
        Number.append(int(index))
        full.append(float(value))

with open('qpq_energy_N.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for index, value in reader:
        qpq.append(float(value))

with open('q4_energy_N.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for index, value in reader:
        q4.append(float(value))





plt.plot(Number, full, label=r'${\rm Full}$')
plt.plot(Number, q4, label=r'$Q^4$')
plt.plot(Number, qpq, label=r'$QPQ$')
plt.xlabel(r'$N$', fontsize=18)
plt.ylabel(r'$\rm 1st\ excited\ energy$', fontsize=18)
plt.xlim(50000, 500000)
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='best')
plt.show()




# N=1e6

# full_ground = full[0]
# q4_ground = variational[0]
# variational_ground = 14.58
# error_abs = abs(full_ground - q4_ground)

# width = np.array([-0.3, 0.3])

# plt.plot(width + 1, [full_ground]*2, '-', linewidth=3, label=r'${\rm Full}, QPQ$')
# plt.plot([1.3, 1.7], [full_ground, q4_ground], '--', color='black')
# plt.plot(width + 2, [q4_ground]*2, '-', linewidth=3, label=r'$Q^4$')
# plt.plot([2.3, 2.7], [q4_ground, variational_ground], '--', color='black')
# plt.plot(width + 3, [variational_ground]*2, '-', linewidth=3, label=r'${\rm Variational}$')
# plt.plot(width + 4, [error_abs]*2, '-', linewidth=3, label=r'${\rm Error}$')
# plt.xlim(0, 5)
# plt.ylim(0, 20)
# plt.xticks(range(1, 5), (r'${\rm Full}$', r"${\rm Q^4}$", r"${\rm Variational}$", r"${\rm Error}$"), fontsize=16)

# plt.text(1, full_ground, '{0:.2f}'.format(full_ground), ha = 'center', va = 'bottom')
# plt.text(2, q4_ground, '{0:.2f}'.format(q4_ground), ha = 'center', va = 'bottom')
# plt.text(3, variational_ground, '{0:.2f}'.format(variational_ground), ha = 'center', va = 'bottom')
# plt.text(4, error_abs, '{0:.2f}'.format(error_abs), ha = 'center', va = 'bottom')

#plt.legend(loc='best')
plt.show()
