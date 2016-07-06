# coding: utf-8

import numpy as np
from scipy import linalg
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sbn
from abc import abstractmethod, ABCMeta

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

# --*-- Basis class --*--
class Procedures(metaclass=ABCMeta):
    
    # Make Zeromode hamiltonian(Lower triangle)
    def __MakeZeromodeHamiltonian(self, v, dmu):

        alpha = 3*v.E/v.dq**4 + (v.I -4*v.D)/v.dq**2 + 2*v.C*(v.q-v.Nq/2.0)**2 -2*v.dq**2*v.B*(v.q-v.Nq/2.0)**2 + 0.5*v.dq**4*v.A*(v.q-v.Nq/2.0)**4
        beta = -2*v.E/v.dq**4 + 2.0j*v.D/v.dq**3 - 0.5*(v.I - 4*v.D)/v.dq**2 - 0.5j*(dmu + 4*v.C)/v.dq - (v.C - 1j*v.dq*v.B)*(v.q-v.Nq/2)*(v.q-v.Nq/2+1)
        gamma = [0.5*v.E/v.dq**4 - 1j*v.D/v.dq**3]*v.Nq
    
        return np.vstack((np.vstack((alpha, beta)), gamma))

    
    # Solve eigenvalue problem for Zeromode hamiltonian
    def ZeromodeEquation(self, v, dmu):
        Hqp = self.__MakeZeromodeHamiltonian(v, dmu)
        # Only ground state
        w, val = linalg.eig_banded(Hqp, lower=True, select="i", select_range=(0, 0), overwrite_a_band=True)
        val = val.reshape(v.Nq)
        
        return val

    
    # Output expected value of Zeromode operator P
    def OutputP(self, v, dmu):
        val = self.ZeromodeEquation(v, dmu)

        # Check boundary value of Zeromode q-space
        if(np.abs(val[0]) > 0.01):
            print("warning!!!")
        
        return np.imag(np.dot(np.conj(val[:-1]),val[1:]))/v.dq

    
    # Find proper counter term dmu
    def SelfConsistent(self, v):
        dmu = optimize.bisect(lambda pro_dmu : self.OutputP(v, pro_dmu), -1e-1, 2e-1)
        
        return dmu


    def SetPlot(self, plot_x, plot_y, xlim=False, ylim=False, logscale_x=False, logscale_y=False, xlabel=False, ylabel=False, title=False):
        plt.plot(plot_x, plot_y)
        if(title):
            plt.title(title)
        if(xlabel):
            plt.xlabel(xlabel)
        if(ylabel):
            plt.ylabel(ylabel)
        if(xlim):
            plt.xlim(xlim[0], xlim[1])
        if(ylim):
            plt.ylim(ylim[0], ylim[1])
        if(logscale_x):
            plt.xscale("log")
        if(logscale_y):
            plt.yscale("log")
            
        plt.show()

    # Use this method with inheritance
    @abstractmethod
    def Procedure(self, N):
        pass
    

class OutputP_Mu(Procedures):

    # Set range of dmu
    def __init__(self, start=-0.5, end=0.5):
        self.start = start
        self.end = end

    def Procedure(self, N):
        v = Variables(N)
        ans = []
        ran = [self.start, self.end]
        for n, dmu in enumerate(np.linspace(ran[0], ran[1], 1000)):
            ans.append(self.OutputP(v, dmu))
            print(n, ", ", end="", flush=True)

        self.SetPlot(plot_x=np.linspace(ran[0], ran[1], 1000), plot_y=ans, xlim=ran, xlabel="dμ", ylabel="P", title="Expected value P for counter term dμ")

        
class OutputZeromodeGroundFunction(Procedures):

    # Set free pro_dmu
    def __init__(self, pro_dmu=False):
        self.pro_dmu = pro_dmu
        
    def Procedure(self, N):
        v = Variables(N)
        if(self.pro_dmu):
            dmu = self.pro_dmu
        else:
            dmu = self.SelfConsistent(v)
        val = self.ZeromodeEquation(v, dmu)

        self.SetPlot(plot_x=v.q, plot_y=np.abs(val), xlabel="q", ylabel="Psi_0", title="Zeromode ground function for a dmu")

        
class OutputQ2_N(Procedures):

    # Set range of N which is power of 10 (like 10^N)
    def __init__(self, N_start=2, N_end=4):
        self.N_start = N_start
        self.N_end = N_end
    
    def Procedure(self, N=None):
        arr_Q2 = []
        arr_N = np.logspace(self.N_start, self.N_end, num=100)
        for N in arr_N:
            v = Variables(N)
            dmu = self.SelfConsistent(v)
            val = self.ZeromodeEquation(v, dmu)
            print("N = {1}, dmu = {0}".format(dmu, N))
            arr_Q2.append(np.sqrt(v.dq**2*(np.sum(np.abs((v.q-v.Nq/2)*val)**2))))

        self.SetPlot(plot_x=arr_N, plot_y=arr_Q2, xlim=(arr_N[0], arr_N[-1]), ylim=(0.01, 1), logscale_x=True, logscale_y=True, xlabel="N", ylabel="Q^2", title="Expected value Q^2 for N")


if __name__ == "__main__":
    
    Q2 = OutputQ2_N()
    Zero = OutputZeromodeGroundFunction(-10)
    PMu = OutputP_Mu()
    Zero.Procedure(N=1e4)

    

