# coding: utf-8

import sys
import numpy as np
from scipy.integrate import simps
from scipy.linalg import solve
from scipy.fftpack import fft2
from scipy.fftpack import ifft2
from scipy.fftpack import fftfreq
import matplotlib.pyplot as plt
import seaborn as sbn
from tqdm import tqdm


class GrossPitaevskiiFor1DPolar():
    
    # --*-- Set constants and variables for initial condition --*--
    def __init__(self, gN=5):

        # Volume, Maximum x-step
        self.L, self.xN = 12, 256

        # Interaction constant, Chemical potential
        self.gN, self.mu, self.kappa = gN, 10, 2

        # Maximum time, Time step 
        self.tMax, self.tN = 1, 512

        # x-step, Time step
        self.h, self.dt = self.L/self.xN, self.tMax/self.tN

        # Set x-space
        self.x = np.linspace(0, self.L, self.xN)

        # Set time propagator of poptential term except non-linear term
        self.arr_Psi = self.__Psi_0(self.x)                

    
    # --*-- Set functions --*--

    # Set initial function
    def __Psi_0(self, x): return x**(self.kappa)*np.exp(-x**2/2)

    
    # Set potential
    def V(self, x): return x**2


    # Make matrix for Crank-Nicolson
    def __MakeCrankMatrix(self):
        
        a = np.diag(-2 - self.kappa**2/np.arange(1, self.xN+1)**2 + self.h**2*(self.mu - self.V(self.x) - self.gN*self.arr_Psi**2 - 2/self.dt))                
        c = np.diag(1 + 0.5/np.arange(0, self.xN))
        c = np.vstack((c[1:], np.array([0]*self.xN)))
                
        d = np.diag(1 - 0.5/np.arange(1, self.xN+1))
        d = np.vstack((d[1:], np.array([0]*self.xN))).T
        
        return a + c + d

    
    # Make vector for Crank-Nicolson
    def __MakeCrankVector(self):
        
        a = (-2 - self.kappa**2/np.arange(1, self.xN+1)**2 + self.h**2*(self.mu - self.V(self.x) - self.gN*self.arr_Psi**2 + 2/self.dt))*self.arr_Psi        
        c = (1 + 0.5/np.arange(1, self.xN+1))*np.hstack((self.arr_Psi[1:], [0]))                
        d = (1 - 0.5/np.arange(1, self.xN+1))*np.hstack(([0], self.arr_Psi[:self.xN-1]))
        
        return -a - c - d

    # Algorithm of Gauss-Seidel
    def __GaussSeidel(self):
        
        # Set Matrix
        a = self.__MakeCrankMatrix()
        
        # Set Vector
        b = self.__MakeCrankVector()
        
        # Solve matrix equation
        self.arr_Psi = solve(a, b)
        
        # Correction of chemical potential mu
        norm = simps(self.x*self.arr_Psi**2, self.x)/(4*np.pi)
        self.mu -= (norm - 1)/(2*self.dt)

        # Normalize arr_Psi
        self.arr_Psi /= np.sqrt(norm)

        
    def GaussSeidelLoop(self):
        
        old_mu = 0
        print("\nGauss-Seidel Start! Please wait...")
        while(np.abs(self.mu - old_mu) > 1e-5):
            old_mu = self.mu
            self.__GaussSeidel()
            sys.stdout.write("\r mu = {0}".format(self.mu))
            sys.stdout.flush()
        print("\nFinished!\n")

        
    def PrintProcedureFor1D(self):

        # --*-- Matplotlib configuration --*--
        plt.plot(self.x, np.real(self.arr_Psi**2), label="gN = {0}".format(self.gN))
        plt.plot(self.x, self.V(self.x), label="Potential")
        plt.ylim(0, 3)
        plt.xlim(0, 6)
        plt.xlabel("r-space")
        plt.ylabel("Order parameter")
        plt.title("Static vortex of Gross-Pitaevskii equation")
        plt.legend(loc = 'center right')
        plt.show()

    def ProcedureFor1DPolar(self):
        
        self.GaussSeidelLoop()
        self.PrintProcedureFor1D()


class GrossPitaevskiiFor2D(GrossPitaevskiiFor1DPolar):

    def __init__(self, gN=5):
        super().__init__(gN)
        
        # -*- For 2D-Symplectic -*-        
                
        # Set xy
        self.I = np.array([1]*self.xN).reshape(self.xN, 1)
        self.x2d = np.linspace(-self.L/2, self.L/2, self.xN)
        self.x22d = self.x2d**2*self.I
        self.xy = np.sqrt(self.x22d + self.x22d.T)

        # Set kx, ky
        kx = 2*np.pi*fftfreq(self.xN, d=1/self.xN)/self.L
        self.kx2d = kx**2*self.I
        self.kxy2d = self.kx2d + self.kx2d.T

        # Set time propagator of poptential term except non-linear term
        self.arr_Psi2D = None
        self.pre_expV = None
        self.expK = None
        self.expV = None


    # Set instable potential
    def __VInst(self, xy):
        
        x = self.x2d*self.I
        phase = np.cos(self.kappa*np.angle(x + 1j*x.T))
        
        return xy**2*phase

    
    # Expand arr_Psi for 2nd Dimension with phase factor
    def __ExpandFor2D(self):

        x = self.x2d * self.I
        phase = np.exp(1j * self.kappa * np.angle(x + 1j*x.T))
        
        index = np.sqrt(self.x22d + self.x22d.T)/self.h
        index = index.astype(int)

        self.arr_Psi2D = self.arr_Psi[index]*phase
        
        
    # Time evolution
    def __Symplectic(self, interval=False):
        
        self.pre_expV = np.exp(-1j*self.V(self.xy)*self.dt)
        self.pre_expV_inst = np.exp(-1j*self.__VInst(self.xy)*self.dt)
        self.expK = np.exp(-1j*self.kxy2d*self.dt)
        self.expV = np.exp(-1j*(-self.mu + self.gN*np.abs(self.arr_Psi2D)**2)*self.dt)*self.pre_expV_inst
        
        print("Symplectic Start! Please wait...")
        
        for i in tqdm(range(self.tN)):
            # Time evolution
            self.arr_Psi2D = ifft2(fft2(self.arr_Psi2D*self.expV)*self.expK)
            
            # Correction of chemical potential mu
            norm = simps(simps(np.abs(self.arr_Psi2D)**2, self.x2d), self.x2d)
            self.mu -= (norm - 1)/(2*self.dt)

            # Normalization of order parameter arr_Psi
            self.arr_Psi2D /= np.sqrt(norm)

            # Reconfigure expV
            if(i > 20):
                self.expV = np.exp(-1j*(-self.mu + self.gN*np.abs(self.arr_Psi2D)**2)*self.dt)*self.pre_expV
            else:
                self.expV = np.exp(-1j*(-self.mu +self.gN*np.abs(self.arr_Psi2D)**2)*self.dt)*self.pre_expV_inst

            # Print arr_Psi in the middle of time evolution
            if(i is interval):
                self.__PrintProcedureFor2D(type="abs")
                self.__PrintProcedureFor2D(type="angle")
            
        print("Finished!")
            

    def __PrintProcedureFor2D(self, type="abs", target=False):
        
        x = np.arange(self.xN)
        y = np.arange(self.xN)
        X, Y = np.meshgrid(x, y)
        
        if(target is not False):
            plt.pcolor(X, Y, self.xy)
            plt.colorbar()

        # Print density profile of arr_Psi
        elif(type == "abs"):
            plt.pcolor(X, Y, np.abs(self.arr_Psi2D))
            plt.clim(0, self.arr_Psi.max()/4)
            plt.colorbar()
        # Print phase profile of arr_Psi
        elif(type == "angle"):
            plt.pcolor(X, Y, np.angle(self.arr_Psi2D))
        
        plt.axis("equal")
        plt.hot()
        plt.xlim(0, self.xN)
        plt.ylim(0, self.xN)
        plt.show()

    
    def ProcedureForVortex(self):
        self.GaussSeidelLoop()
        self.__ExpandFor2D()
        self.__Symplectic(interval=0)
        self.__PrintProcedureFor2D(type="abs")
        self.__PrintProcedureFor2D(type="angle")

        
if(__name__ == "__main__"):
    hundle = GrossPitaevskiiFor2D(gN=5)
    hundle.ProcedureForVortex()

