# coding: utf-8

import numpy as np
from scipy.integrate import simps
from scipy.fftpack import fft2
from scipy.fftpack import ifft2
from scipy.fftpack import fftfreq
import matplotlib.pyplot as plt
import seaborn as sbn
import itertools as itr

class GrossPitaevskii():
    
    # --*-- Set constants and variables for initial condition --*--
    
    def __init__(self, gN=70):

        # Volume, Maximum x-step
        self.L, self.xN = 12.0, 256

        # Interaction constant, Chemical potential
        self.gN, self.mu = gN, 10

        # Maximum time, Time step 
        self.tMax, self.tN = 2, 256

        # x-step, Time step
        self.h, self.dt = self.L/self.xN, self.tMax/self.tN

        # Set xy-space
        self.x = np.arange(-self.xN/2, self.xN/2)
        self.y = self.x.reshape(self.xN, 1)
        self.x = self.h*np.sqrt(self.y**2+self.x**2)

        # Set kx, ky
        self.kx = 2*np.pi*fftfreq(self.xN, d=1/self.xN)/self.L
        self.ky = self.kx.reshape(self.xN, 1)
        
        
        # Set time propagator of poptential term except non-linear term
        self.arr_Psi = self.__Psi_0(self.x)
        self.pre_expV = np.exp((self.mu - self.__V(self.x))*self.dt)
        self.expK = np.exp(-(self.kx**2 + self.ky**2)*self.dt)
        self.expV = np.exp((-self.gN*self.__Psi_0(self.x)**2)*self.dt)*self.pre_expV
    
    # --*-- Set functions --*--
            
    # Set initial function
    def __Psi_0(self, x):
        #return x*np.exp(-self.x**2/2)
        return x*np.exp(-self.x**2)
    # Set potential
    def __V(self, x):
        #return 25*np.sin(self.x/5)**2
        return x**2
    
    # Time evolution
    def Symplectic(self):
        self.tN = 20
        for i in range(self.tN):
            print(i)
            # Time evolution
            self.arr_Psi = ifft2(fft2(self.arr_Psi*self.expV)*self.expK)
            
            # Correction of chemical potential mu
            # 矩形積分. 雑なので直したい
            norm = np.sum(self.arr_Psi**2)*self.h**2
            #self.mu -= (norm - 1)/(2*self.dt)
            #print(self.mu)

            # Normalization of order parameter arr_Psi
            self.arr_Psi /= np.sqrt(norm)

            # Reconfigure expV
            self.expV = np.exp((-self.gN*self.arr_Psi**2)*self.dt)*self.pre_expV
            
    

#class QuantumVortex(GrossPitaevskii):


        
    

#GrossPitaevskii.PrintProcedure()
hundle = GrossPitaevskii(gN=70)
print(hundle.arr_Psi)
x = np.arange(hundle.xN)
y = np.arange(hundle.xN)

X, Y = np.meshgrid(x, y)
#hundle.Symplectic()
plt.pcolor(X, Y, np.real(hundle.arr_Psi))
plt.axis("equal")
plt.xlim(0, 250)
plt.ylim(0, 250)
plt.show()

hundle.Symplectic()
plt.pcolor(X, Y, np.real(hundle.arr_Psi))
plt.axis("equal")
plt.xlim(0, 250)
plt.ylim(0, 250)
plt.show()
