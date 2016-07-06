# coding: utf-8

import numpy as np
from scipy.integrate import simps
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy.fftpack import fftfreq
import matplotlib.pyplot as plt
import seaborn as sbn

class GrossPitaevskii():
    
    # --*-- Set constants and variables for initial condition --*--
    def __init__(self, gN=50):

        # Volume, Maximum x-step
        self.L, self.xN = 16.0, 1024

        # Interaction constant, Chemical potential
        self.gN, self.mu = gN, 10

        # Maximum time, Time step 
        self.tMax, self.tN = 2, 256

        # x-step, Time step
        self.h, self.dt = self.L/self.xN, self.tMax/self.tN

        # Set x-space
        self.x = np.linspace(-self.L/2, self.L/2, self.xN)

        # Set time propagator of poptential term except non-linear term
        self.arr_Psi = self.__Psi_0(self.x)
        self.pre_expV = np.exp((self.mu - self.__V(self.x))*self.dt)
        self.expK = np.exp(-(2*np.pi*fftfreq(self.xN, d=1/self.xN)/self.L)**2*self.dt)
        self.expV = np.exp((-self.gN*self.__Psi_0(self.x)**2)*self.dt)*self.pre_expV

    
    # --*-- Set functions --*--

    # Set initial function
    def __Psi_0(self, x):
        return x*np.exp(-x**2/2)

    # Set potential
    def __V(self, x):
        return 25*np.sin(x/5)**2

    # Time evolution
    def __Symplectic(self):
        
        for i in range(self.tN):
            # Time evolution
            self.arr_Psi = ifft(fft(self.arr_Psi*self.expV)*self.expK)
            
            # Correction of chemical potential mu
            self.mu -= (simps(self.arr_Psi**2, self.x) - 1)/(2*self.dt)

            # Normalization of order parameter arr_Psi
            self.arr_Psi /= np.sqrt(simps(np.real(self.arr_Psi**2), self.x))

            # Reconfigure expV
            self.expV = np.exp((-self.gN*self.arr_Psi**2)*self.dt)*self.pre_expV
            

    def PrintProcedure(self):
        self.__Symplectic()

        # --*-- Matplotlib configuration --*--
        plt.plot(self.x, np.real(self.arr_Psi**2), label="gN = {0}".format(self.gN))
        plt.plot(self.x, self.__V(self.x)/50, label="Potential\n(provisional)")
        plt.ylim(0, 0.5)
        plt.xlabel("q-space")
        plt.ylabel("Order parameter")
        plt.title("Static soliton of Gross-Pitaevskii equation")
        plt.legend(loc = 'center right')
        plt.show()


#GrossPitaevskii.PrintProcedure()
hundle = GrossPitaevskii(gN=50)
hundle.PrintProcedure()

