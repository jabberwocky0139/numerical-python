# coding: utf-8

import numpy as np
from scipy import linalg
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sbn

# --*-- Constants' and variables' class --*--

class Variables():
    def __init__(self, N):
        # -*- Constants on real space -*-
        
        # Particle number
        self.N = N
        # Volume
        self.V = 1
        # Interaction constant
        self.g = 1e-3
        # Integrals for oder parameter and ajoint parameter
        self.A = self.g*self.N**2/self.V
        self.B = self.g*self.N/self.V/2
        self.C = self.g/self.V/4
        self.D = self.g/self.V/self.N/8
        self.E = self.g/self.V/self.N**2/16
        self.I = self.g/self.V
    
        # -*- Constant on Zeromode space -*-

        # Volume of Zeromode q-space
        self.L = 20*(self.N*self.V)**(-1/3)
        # Step numbers of Zeromode q-space
        self.Nq = 200
        # Step size of Zeromode q-space
        self.dq = self.L/self.Nq
        # Zeromode q-space coordinate
        self.q = np.arange(self.Nq)

# Make Zeromode Hamiltonian(Lower triangle)
def MakeZeromodeHamiltonian(dmu, v):
    
    alpha = 3*v.E/v.dq**4 + (v.I -4*v.D)/v.dq**2 + 2*v.C*(v.q-v.Nq/2.0)**2 -2*v.dq**2*v.B*(v.q-v.Nq/2.0)**2 + 0.5*v.dq**4*v.A*(v.q-v.Nq/2.0)**4
    beta = -2*v.E/v.dq**4 + 2.0j*v.D/v.dq**3 - 0.5*(v.I - 4*v.D)/v.dq**2 - 0.5j*(dmu + 4*v.C)/v.dq - (v.C - 1j*v.dq*v.B)*(v.q-v.Nq/2)*(v.q-v.Nq/2+1)
    gamma = [0.5*v.E/v.dq**4 - 1j*v.D/v.dq**3]*v.Nq
    
    return np.vstack((np.vstack((alpha, beta)), gamma))


def ZeromodeEquation(dmu, v, Q2=False):
    
    Hqp = MakeZeromodeHamiltonian(dmu, v)

    # 基底状態のみ
    w, val = linalg.eig_banded(Hqp, lower=True, select="i", select_range=(0, 0), overwrite_a_band=True)
    val = val.reshape(v.Nq)
    P = np.imag(np.dot(np.conj(val[:-1]),val[1:]))/v.dq

    # このif文なんとかしたい
    if(np.abs(val[0]) > 0.01):
        print("warning!!!")

    if(Q2):
        return np.sqrt(v.dq**2*(np.sum(np.abs((v.q-v.Nq/2)*val)**2)))
    else:
        return P

    
# Output expected value P for counter term dμ
def PMu(start, end, N):
    ans = []
    ran = [start, end]
    v = Variables(N)
    for n, dmu in enumerate(np.linspace(ran[0], ran[1], 1000)):
        ans.append(ZeromodeEquation(dmu, v))
        print(n, ", ", end="", flush=True)

    plt.plot(np.linspace(ran[0], ran[1], 1000), ans)
    plt.xlabel("dμ")
    plt.ylabel("P")
    plt.suptitle("Expected value P for counter term dμ")
    plt.xlim(ran[0], ran[1])
    plt.show()

    
# Output ground state of Zeromode Hamiltonian
def PrintZeromodeGroundFunction(dmu, N):
    v = Variables(N)
    Hqp = MakeZeromodeHamiltonian(dmu, v)
    val = linalg.eig_banded(Hqp, lower=True, select="i", select_range=(0, 0), overwrite_a_band=True)[1].reshape(v.Nq)
    plt.plot(v.q, np.abs(val))
    plt.show()


if __name__ == "__main__":
    
    #PrintZeromodeGroundFunction(0.0001, 1e7)
    #PMu(-0.5, 0.5, 1e4)
    

    arr_Q2 = []
    arr_N = np.logspace(2, 4, num=100)
    for N in arr_N:
        v = Variables(N)
        #dmu = optimize.bisect(lambda dmu : ZeromodeEquation(dmu, v), -1e-4, 2e-4)
        dmu = optimize.bisect(lambda dmu : ZeromodeEquation(dmu, v), -1e-1, 2e-1)
        print("N = {1}, dmu = {0}".format(dmu, N))
        arr_Q2.append(ZeromodeEquation(dmu, v, Q2=True))
    
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(arr_N[0], arr_N[-1])
    plt.ylim(0.01, 1)
    plt.plot(arr_N, arr_Q2)
    plt.show()


    

